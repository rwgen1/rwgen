import os
import warnings
import datetime
import itertools

import numpy as np
import pandas as pd
import scipy.spatial
import scipy.stats
import scipy.optimize
import statsmodels.api as sm
import gstools


class Preprocessor:

    # Assumption of daily data currently

    def __init__(
            self,
            spatial_model,
            input_timeseries,
            point_metadata,
            climatology_grids,  # dict of file paths (or opened files)
            output_folder,
            xmin,
            xmax,
            ymin,
            ymax,
            spatial_method,
            max_buffer,  # km - but easting/northing in m so convert - could be set to zero
            min_points,  # minimum number of stations if using interpolation or pooling
            wet_threshold,
            use_neighbours,  # use neighbouring precipitation record to try to infill wet days
            neighbour_radius,  # km - but easting/northing in m so convert
            calculation_period,
            completeness_threshold,
            predictors,
            input_variables,  # list of variables to work with - primarily geared at sunshine duration vs incoming SW
            season_length,  # 'month' or 'half-month'
            offset,
    ):
        self.spatial_model = spatial_model
        self.input_timeseries = input_timeseries  # infer input variables
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.spatial_method = spatial_method  # 'uniform', 'pool' or 'interpolate'
        self.max_buffer = max_buffer  # km - but easting/northing in m so convert
        # self.min_years = min_years
        self.min_points = min_points
        self.use_neighbours = use_neighbours  # use neighbouring precipitation record
        self.neighbour_radius = neighbour_radius  # km - but easting/northing in m so convert
        self.wet_threshold = wet_threshold
        self.calculation_period = calculation_period
        self.completeness_threshold = completeness_threshold
        self.output_folder = output_folder
        self.predictors = predictors
        self.input_variables = input_variables  # should not include precipitation
        self.season_length = season_length

        self.n_years = {v: 0 for v in self.input_variables}
        self.n_points = {v: 0 for v in self.input_variables}
        self.data_series = {}

        self.metadata = None
        self.kd_tree = None
        self.climatology_grids = None
        if self.spatial_model:
            if self.spatial_method != 'uniform':
                if isinstance(point_metadata, str):
                    self.metadata = pd.read_csv(point_metadata)
                else:
                    self.metadata = point_metadata
                self.metadata.columns = [column.lower() for column in self.metadata.columns]

                if self.use_neighbours:
                    self.kd_tree = scipy.spatial.KDTree(self.metadata[['easting', 'northing']])

            if climatology_grids is not None:
                if isinstance(climatology_grids, str):
                    self.climatology_grids = read_grids(climatology_grids)
                else:
                    self.climatology_grids = climatology_grids

        period_length = (
                datetime.datetime(self.calculation_period[1], 12, 31)
                - datetime.datetime(self.calculation_period[0], 1, 1)
        )
        self.period_length = period_length.days + 1

        self.offset = offset  # in box-cox transformation after standardisation (unlikely to get anything much below -3)

        if self.season_length == 'month':
            self.seasons = list(range(1, 12+1))
        elif self.season_length == 'half-month':
            self.seasons = list(range(1, 24 + 1))

        self.simulation_variables = self.input_variables.copy()
        self.simulation_variables.append('prcp')

        self.data_series = None
        self.raw_statistics = None
        self.transformed_statistics = None

        self.transformations = {}
        self.regressions = {}
        self.parameters = {}
        self.residuals = {}
        self.r2 = {}
        self.standard_errors = {}

        self.statistics_variograms = {}
        self.residuals_variograms = {}
        self.r2_variograms = {}
        self.se_variograms = {}
        self.noise_models = {}

    def preprocess(self):
        # Assuming that a reasonable input series is passed in for a point model or for a spatially uniform model
        # - no explicit checks that sufficient data are available...
        if self.spatial_model and (self.spatial_method != 'uniform'):
            self.process_stations()
        else:
            self.process_station()

        # Subset point metadata on points that are being used
        if self.spatial_model:
            self.metadata = self.metadata.loc[self.metadata['point_id'].isin(self.data_series['point_id'].unique())]

        # Pooling - dfs in self.data_series then contain point_id and pool_id to help pool if required (via groupby)
        if self.spatial_method == 'interpolate':
            self.data_series['pool_id'] = self.data_series['point_id']
        else:
            self.data_series['pool_id'] = 1

        # Transformation and second standardisation (reshaping data to wide with lagged series as columns)
        self.transform_series()

    def fit(self):
        self.do_regression()

        # Fit variogram models for interpolation
        if self.spatial_model and (self.spatial_method != 'uniform'):
            self.estimate_statistic_variograms()
            self.estimate_r2_variograms()
            self.estimate_se_variograms()

    def process_stations(self):
        buffer = 0.0
        processed_ids = []
        while (min(self.n_points.values()) < self.min_points) and (buffer <= self.max_buffer):
            metadata = self.metadata.loc[
                (self.metadata['easting'] >= self.xmin - buffer * 1000.0)
                & (self.metadata['easting'] <= self.xmax + buffer * 1000.0)
                & (self.metadata['northing'] >= self.ymin - buffer * 1000.0)
                & (self.metadata['northing'] <= self.ymax - buffer * 1000.0)
            ]

            for point_id in metadata['point_id']:
                if point_id not in processed_ids:
                    # print(point_id)
                    self.process_station(point_id)
                    # sys.exit()
                processed_ids.append(point_id)

            if (buffer + 20.0) < self.max_buffer:
                buffer += 20.0
            elif buffer == self.max_buffer:
                break
            else:
                buffer = self.max_buffer

    def process_station(self, point_id=1):
        df, df1, completeness = self.prepare_series(point_id)  # time series, mean+std, completeness by variable
        if df is not None:
            if self.data_series is None:
                self.data_series = df.copy()
            else:
                self.data_series = pd.concat([self.data_series, df])
            if self.raw_statistics is None:
                self.raw_statistics = df1.copy()
            else:
                self.raw_statistics = pd.concat([self.raw_statistics, df1])
            for variable in self.simulation_variables:
                if variable in completeness.keys():
                    if completeness[variable] >= self.completeness_threshold:
                        self.n_years[variable] += ((completeness[variable] / 100.0) * self.period_length) / 365.25
                        self.n_points[variable] += 1

    def prepare_series(self, point_id=1):
        if self.spatial_model and (self.spatial_method != 'uniform'):
            file_name = self.metadata.loc[self.metadata['point_id'] == point_id, 'file_name'].values[0]
            easting = self.metadata.loc[self.metadata['point_id'] == point_id, 'easting'].values[0]
            northing = self.metadata.loc[self.metadata['point_id'] == point_id, 'northing'].values[0]
            input_path = os.path.join(self.input_timeseries, file_name)
        else:
            input_path = self.input_timeseries

        # Read data
        df = pd.read_csv(input_path, index_col=0, parse_dates=True, infer_datetime_format=True)
        df.columns = [column.lower() for column in df.columns]

        # Assign month or half-month identifiers
        if self.season_length == 'half-month':
            df['season'] = identify_half_months(df.index)
        elif self.season_length == 'month':
            df['season'] = df.index.month

        # Subset on calculation period
        if self.calculation_period is not None:
            df = df.loc[(df.index.year >= self.calculation_period[0]) & (df.index.year <= self.calculation_period[1])]

        # Check enough data here to avoid crashing below (full completeness checks carried out next)
        if df.shape[0] >= 365:

            # Add initial wet day indicator column
            df['wet_day'] = np.where(np.isfinite(df['prcp']) & (df['prcp'] >= self.wet_threshold), 1, 0)
            df['wet_day'] = np.where(~np.isfinite(df['prcp']), np.nan, df['wet_day'])

            # Try using nearest neighbours to infill wet/dry day indicator and temperature
            if self.spatial_model and self.use_neighbours:
                distances, indices = self.kd_tree.query([easting, northing], k=10)
                distances = distances[1:]
                indices = indices[1:]

                for distance, index in zip(distances, indices):
                    if distance < self.neighbour_radius * 1000.0:
                        neighbour_file = self.metadata['file_name'].values[index]
                        neighbour_path = os.path.join(self.input_timeseries, neighbour_file)

                        df1 = pd.read_csv(neighbour_path, index_col=0, parse_dates=True, infer_datetime_format=True)
                        df1.columns = [column.lower() for column in df1.columns]
                        df1.reset_index(inplace=True)
                        df1.rename(columns={
                            'prcp': 'prcp_neighbour',
                        }, inplace=True)
                        df = pd.merge(
                            df, df1[['datetime', 'prcp_neighbour']],
                            how='left', on='datetime'
                        )
                        df['wet_day_neighbour'] = np.where(df['prcp_neighbour'] >= self.wet_threshold, 1, 0)
                        df['wet_day'] = np.where(
                            ~np.isfinite(df['wet_day']) & np.isfinite(df['prcp_neighbour']),
                            df['wet_day_neighbour'],
                            df['wet_day']
                        )
                        for variable in ['prcp']:
                            df[variable] = np.where(
                                ~np.isfinite(df[variable]) & np.isfinite(df[variable + '_neighbour']),
                                df[variable + '_neighbour'],
                                df[variable]
                            )
                        df.drop(columns={
                            'prcp_neighbour', 'wet_day_neighbour'},
                            inplace=True
                        )

            # Check all variables present and complete in case the series has been updated
            df['temp_avg'] = (df['temp_min'] + df['temp_max']) / 2.0
            df['dtr'] = df['temp_max'] - df['temp_min']

            # Identify completeness by variable
            completeness = {}
            for variable in self.input_variables:
                if variable in df.columns:
                    if df.shape[0] > 0:
                        if (variable in ['temp_avg', 'dtr']) and ('prcp' in df.columns):
                            completeness[variable] = (
                                np.sum(np.isfinite(df['prcp']) & np.isfinite(df[variable])) / self.period_length * 100
                            )
                        elif ('prcp' in df.columns) and ('temp_avg' in df.columns):
                            completeness[variable] = (
                                np.sum(np.isfinite(df['prcp']) & np.isfinite(df['temp_avg']) & np.isfinite(df[variable]))
                                / self.period_length * 100
                            )
                        else:
                            completeness[variable] = 0.0
                        completeness[variable] = min(completeness[variable], 100.0)
                        if completeness[variable] < (self.completeness_threshold / 100.0):
                            df.drop(columns=[variable], inplace=True)
                    else:
                        completeness[variable] = 0.0
                else:
                    completeness[variable] = 0.0

            # Need at least one variable to have sufficient completeness to proceed
            if max(completeness.values()) >= self.completeness_threshold:

                # If not using neighbours or only a point model then datetime is not necessarily a column at this point
                if 'datetime' not in df.columns:
                    df.reset_index(inplace=True)

                # Reshape to long for processing as groups
                df = pd.melt(df, id_vars=['datetime', 'season', 'prcp', 'wet_day'])

                # Subset on variables that need to be taken forward
                df = df.loc[~df['variable'].isin(['temp_mean', 'temp_min', 'temp_max', 'rel_hum'])]

                # Transition states
                df['wet_day_lag1'] = df['wet_day'].shift(1)
                df['wet_day_lag2'] = df['wet_day'].shift(2)
                df['transition'] = 'NA'
                df['transition'] = np.where((df['wet_day_lag1'] == 1) & (df['wet_day'] == 1), 'WW', df['transition'])
                df['transition'] = np.where((df['wet_day_lag1'] == 0) & (df['wet_day'] == 1), 'DW', df['transition'])
                df['transition'] = np.where((df['wet_day_lag1'] == 1) & (df['wet_day'] == 0), 'WD', df['transition'])
                df['transition'] = np.where((df['wet_day_lag1'] == 0) & (df['wet_day'] == 0), 'DD', df['transition'])
                df['transition'] = np.where(
                    (df['wet_day_lag2'] == 0) & (df['wet_day_lag1'] == 0) & (df['wet_day'] == 0), 'DDD', df['transition']
                )
                df.drop(columns=['wet_day', 'wet_day_lag1', 'wet_day_lag2'], inplace=True)

                # Move precipitation in as a variable
                tmp1 = df.loc[
                    df['variable'] == df['variable'].unique()[0], ['datetime', 'season', 'transition', 'prcp']
                ].copy()
                tmp1.rename(columns={'prcp': 'value'}, inplace=True)
                tmp1['variable'] = 'prcp'
                df.drop(columns=['prcp'], inplace=True)
                df = pd.concat([df, tmp1])

                # Store statistics for station and standardise series
                df1 = df.loc[df['transition'] != 'NA']
                df1 = df1.groupby(['variable', 'season'])['value'].agg(['mean', 'std'])
                df1.reset_index(inplace=True)
                df = pd.merge(df, df1, on=['variable', 'season'])
                df['z_score'] = (df['value'] - df['mean']) / df['std']
                df.drop(columns=['mean', 'std'], inplace=True)

                # Add point ID in preparation for storing in one big df (and pooling)
                df['point_id'] = point_id
                df1['point_id'] = point_id

            else:
                df = None
                completeness = None
                df1 = None

        # If very little data return nones
        else:
            df = None
            completeness = None
            df1 = None

        return df, df1, completeness

    def transform_series(self):
        # Transformations to normal distribution
        # - not doing by transition state for now as sample size can get quite low - warnings in optimisation for lamda
        # - need to track (and output) lambda so that series can be back-transformed after simulation
        # - latent gaussian variable technique needed for sunshine duration
        # -- need to know probability of zero sunshine hours

        # TODO: Check that this referencing updates self.data_series
        # - handled now by returning series that contains transformed and (again) standardised series
        df = self.data_series

        # Factors by which to stratify transformation
        variables = df['variable'].unique()
        pool_ids = df['pool_id'].unique()

        # Main loop
        dfs = []
        for season, variable, pool_id in itertools.product(self.seasons, variables, pool_ids):  # !221212
            df1 = df.loc[
                (df['season'] == season) & (df['variable'] == variable)
                & (df['pool_id'] == pool_id) & (np.isfinite(df['z_score']))
            ].copy()

            if (df1.shape[0] > 0) and (variable in ['temp_avg', 'dtr', 'vap_press', 'wind_speed']):
                # Numerical issues in optimisation of lamda can be avoided by ensuring minimum is small-ish+positive

                # TODO: Use larger and constant offset in box-cox transformations?
                # - see CRU code - it looks like an offset of 30 is used
                # - in fact check the CRU code for how each variable is treated + perturbations

                bc_value, lamda = scipy.stats.boxcox(df1['z_score'] + self.offset)
                df1['bc_value'] = bc_value
                dfs.append(df1)
                self.transformations[(pool_id, variable, season, 'lamda')] = lamda

            elif (df1.shape[0] > 0) and (variable == 'sun_dur'):
                # TODO: Keep track of min/max used in scaling? Or assume min=0 and max=fully sunny day?
                # Alternatively could calculate day length here and use this - may be the most accurate

                p0 = df1.loc[df1['value'] < 0.01, 'value'].shape[0] / df1.shape[0]
                df1['scaled'] = (df1['value'] - df1['value'].min()) / (df1['value'].max() - df1['value'].min())

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    a, b, loc, scale = scipy.stats.beta.fit(df1.loc[df1['value'] >= 0.01, 'scaled'])
                self.transformations[(pool_id, variable, season, 'p0')] = p0
                self.transformations[(pool_id, variable, season, 'a')] = a
                self.transformations[(pool_id, variable, season, 'b')] = b
                self.transformations[(pool_id, variable, season, 'loc')] = loc
                self.transformations[(pool_id, variable, season, 'scale')] = scale

                # Recording min/max of observations for now, but see above
                self.transformations[(pool_id, variable, season, 'obs_min')] = df1['value'].min()
                self.transformations[(pool_id, variable, season, 'obs_max')] = df1['value'].max()

                # Probability associated with non-zero values
                df1['probability'] = scipy.stats.beta.cdf(df1['scaled'], a, b, loc, scale)
                df1['probability'] = (1 - p0) * df1['probability'] + p0
                df1.loc[df1['value'] < 0.01, 'probability'] = p0

                # Standard normal values - use sampling for <= p0
                rng = np.random.default_rng()  # TODO: Seed for reproducibility
                dummy_probability = rng.uniform(low=0, high=p0, size=df1.shape[0])
                # dummy_probability = p0  # ! TESTING ONLY !
                df1['probability'] = np.where(df1['value'] < 0.01, dummy_probability, df1['probability'])
                df1['bc_value'] = scipy.stats.norm.ppf(df1['probability'], 0, 1)

                df1.drop(columns=['scaled', 'probability'], inplace=True)
                dfs.append(df1)

            elif (df1.shape[0] > 0) and (variable == 'prcp'):
                # Not transforming precipitation currently
                # - prcp will be standardised later, but this does not affect the regression
                # TODO: Consider whether wet-day precipitation should undergo (just box-cox?) transformation
                # - box-cox does not seem to work very well based on lerwick test
                # - weibull might work reasonably... two parameters
                df1['bc_value'] = df1['value']
                dfs.append(df1)

            else:
                df1['bc_value'] = np.nan

        # Join all back into one dataframe
        df = pd.concat(dfs)
        df.sort_values(['pool_id', 'point_id', 'variable', 'datetime'], inplace=True)

        # Standardisation of transformed values

        # Calculate statistics for standardisation
        df1 = df.loc[df['transition'] != 'NA']
        df1 = df1.groupby(['pool_id', 'variable', 'season'])['bc_value'].agg(['mean', 'std'])  # , 'transition'
        df1.reset_index(inplace=True)

        # Check if not all transitions represented - use average if missing for now
        tmp1 = expand_grid(
            ['pool_id', 'variable', 'season'],
            df1['pool_id'].unique(), df1['variable'].unique(), df1['season'].unique()
        )
        tmp2 = df1.groupby(['pool_id', 'variable', 'season'])[['mean', 'std']].mean()
        tmp2.reset_index(inplace=True)
        tmp2.rename(columns={'mean': 'tmp_mean', 'std': 'tmp_std'}, inplace=True)
        df1 = pd.merge(df1, tmp1, how='right')
        df1 = pd.merge(df1, tmp2, how='left')
        df1['mean'] = np.where(~np.isfinite(df1['mean']), df1['tmp_mean'], df1['mean'])
        df1['std'] = np.where(~np.isfinite(df1['std']), df1['tmp_std'], df1['std'])
        df1.drop(columns=['tmp_mean', 'tmp_std'], inplace=True)
        df1.rename(columns={'mean': 'bc_mean', 'std': 'bc_std'}, inplace=True)

        # Standardise time series
        # - keep series contiguous (i.e. using NA) to ensure that lag-1 value is identified correctly
        df = pd.merge(df, df1, how='left')
        df['sd_value'] = (df['bc_value'] - df['bc_mean']) / df['bc_std']
        df['sd_lag1'] = df.groupby(['pool_id', 'point_id', 'variable', 'season'])['sd_value'].transform(shift_)

        # Wide dataframe containing standardised values and lag-1 standardised values for all variables
        index_columns = ['pool_id', 'point_id', 'datetime', 'season', 'transition']  #
        tmp1 = df.pivot(index=index_columns, columns='variable', values='sd_value')
        tmp1.reset_index(inplace=True)
        tmp2 = df.pivot(index=index_columns, columns='variable', values='sd_lag1')
        tmp2.reset_index(inplace=True)
        tmp2.columns = [col + '_lag1' if col not in index_columns else col for col in tmp2.columns]
        df2 = pd.merge(tmp1, tmp2)

        self.transformed_statistics = df1
        self.data_series = df2

    def do_regression(self):
        # df2 is wide df - not in self.data_series
        df2 = self.data_series

        # Set up factors to loop
        pool_ids = df2['pool_id'].unique().tolist()
        transitions = df2['transition'].unique().tolist()
        if 'NA' in transitions:
            transitions.remove('NA')
        variables = self.input_variables.copy()

        # Calculate regression parameters by transition state
        for pool_id, season, transition, variable in itertools.product(pool_ids, self.seasons, transitions, variables):

            # Subset on relevant finite values - successively for dependent and then each independent variable
            df2a = df2.loc[
                (df2['pool_id'] == pool_id) & (df2['season'] == season) & (df2['transition'] == transition)
                & (np.isfinite(df2[variable]))
            ]
            for predictor in self.predictors[(variable, transition)]:
                df2a = df2a.loc[np.isfinite(df2a[predictor])]

            # Populate array for regression
            n_times = df2a.shape[0]
            n_predictors = len(self.predictors[(variable, transition)])
            X = np.zeros((n_times, n_predictors))
            col_idx = 0
            for predictor in self.predictors[(variable, transition)]:
                X[:,col_idx] = df2a[predictor].values
                col_idx += 1

            # Set a minimum number of days for performing regression - as user option?
            if X.shape[0] >= 10:

                # Need regression parameters, r-squared and residuals for spatial correlation
                X = sm.add_constant(X)  # adds column of ones - required for intercept to be estimated
                model = sm.OLS(df2a[variable].values, X)
                results = model.fit()
                self.parameters[(pool_id, season, variable, transition)] = results.params
                df2b = df2a[['datetime', 'point_id', variable]].copy()
                df2b['residual'] = results.resid
                self.residuals[(pool_id, season, variable, transition)] = df2b

                # Calculate r2 by point (not pool)
                df2b['fitted'] = results.fittedvalues
                df2c = df2b.groupby('point_id')[[variable, 'fitted']].corr().unstack().iloc[:, 1]  # series
                df2c = df2c.to_frame('r')
                df2c['r2'] = df2c['r'] ** 2
                df2c.reset_index(inplace=True)
                for _, row in df2c.iterrows():
                    self.r2[(row['point_id'], season, variable, transition)] = row['r2']

                df2d = df2b.groupby('point_id')['residual'].std()
                df2d = df2d.to_frame('residual')
                df2d.reset_index(inplace=True)

                for _, row in df2d.iterrows():
                    self.standard_errors[(row['point_id'], season, variable, transition)] = row['residual']

            else:
                print(season, transition, variable)
                # TODO: Need to handle this case explicitly

    def estimate_statistic_variograms(self):
        # TODO: Add elevation to metadata so that it can be included
        df = pd.merge(
            self.raw_statistics, self.metadata[['point_id', 'easting', 'northing']], on='point_id', how='left',  # , 'elevation'
        )
        for season, variable in itertools.product(self.seasons, self.simulation_variables):  # df['variable'].unique()):
            df1 = df.loc[(df['season'] == season) & (df['variable'] == variable)]
            for statistic in ['mean', 'std']:
                if df1['point_id'].unique().shape[0] >= self.min_points:
                    interpolator = fit_variogram_model(
                        df1, include_elevation=False, value=statistic  # TODO: Use elevation
                    )
                    interpolation_type = 'kriging'
                else:
                    interpolator = make_idw_interpolator(df1, value=statistic)
                    interpolation_type = 'idw'

                self.statistics_variograms[(season, variable, statistic)] = (interpolation_type, interpolator)

    def estimate_r2_variograms(self):
        # TODO: Add elevation to metadata so that it can be included
        transitions = ['DDD', 'DD', 'DW', 'WD', 'WW']
        for season, variable, transition in itertools.product(self.seasons, self.input_variables, transitions):

            df1 = self.metadata[['point_id', 'easting', 'northing']].copy()
            df1['r2'] = np.nan
            for point_id in self.metadata['point_id']:
                if (point_id, season, variable, transition) in self.r2.keys():
                    df1.loc[df1['point_id'] == point_id, 'r2'] = self.r2[(point_id, season, variable, transition)]

            df1 = df1.loc[np.isfinite(df1['r2'])]

            if df1['point_id'].unique().shape[0] >= self.min_points:
                interpolator = fit_variogram_model(
                    df1, include_elevation=False, value='r2'  # TODO: Use elevation
                )
                interpolation_type = 'kriging'
            else:
                interpolator = make_idw_interpolator(df1, value='r2')
                interpolation_type = 'idw'

            self.r2_variograms[(season, variable, transition)] = (interpolation_type, interpolator)

    def estimate_se_variograms(self):
        # TODO: Add elevation to metadata so that it can be included

        transitions = ['DDD', 'DD', 'DW', 'WD', 'WW']
        for season, variable, transition in itertools.product(self.seasons, self.input_variables, transitions):

            df1 = self.metadata[['point_id', 'easting', 'northing']].copy()
            df1['r2'] = np.nan
            for point_id in self.metadata['point_id']:
                if (point_id, season, variable, transition) in self.standard_errors.keys():
                    df1.loc[df1['point_id'] == point_id, 'se'] = (
                        self.standard_errors[(point_id, season, variable, transition)]
                    )

            df1 = df1.loc[np.isfinite(df1['se'])]

            if df1['point_id'].unique().shape[0] >= self.min_points:
                interpolator = fit_variogram_model(
                    df1, include_elevation=False, value='se'  # TODO: Use elevation
                )
                interpolation_type = 'kriging'
            else:
                interpolator = make_idw_interpolator(df1, value='se')
                interpolation_type = 'idw'

            self.se_variograms[(season, variable, transition)] = (interpolation_type, interpolator)


def exponential_model(distance, variance, length_scale, nugget, flip=True):
    x = variance * (1.0 - np.exp(-distance / length_scale)) + nugget
    if flip:
        x = 1.0 - x
    return x


def fit_covariance_model(df1, value='value'):
    distances = []
    correlations = []
    covariances = []
    for point1, point2 in itertools.combinations(df1['point_id'].unique(), 2):
        distance = (
            (df1.loc[df1['point_id'] == point1, 'easting'].values[0]
             - df1.loc[df1['point_id'] == point2, 'easting'].values[0]) ** 2
            + (df1.loc[df1['point_id'] == point1, 'northing'].values[0]
               - df1.loc[df1['point_id'] == point2, 'northing'].values[0]) ** 2
        ) ** 0.5

        df1a = df1.loc[(df1['point_id'] == point1) | (df1['point_id'] == point2)]
        df1a = df1a.pivot(index=['datetime'], columns='point_id', values=value)
        df1a.reset_index(inplace=True)
        df1a = df1a.loc[np.isfinite(df1a[point1]) & np.isfinite(df1a[point2])]

        result = scipy.stats.pearsonr(df1a[point1], df1a[point2])
        correlation = result[0]  # newer scipy (1.9.0) would need result.statistic

        covariance = np.cov(df1a[[point1, point2]].values, rowvar=False)[0,1]

        distances.append(distance)
        correlations.append(correlation)
        covariances.append(covariance)

    distances = np.asarray(distances)
    correlations = np.asarray(correlations)

    # fixing sill to one and optimising nugget (0-1 bounds ok?)
    bounds = ([0.99, 0.0, 0.0], [1.0, 100000000.0, 1.0])

    parameters, _ = scipy.optimize.curve_fit(exponential_model, distances, correlations, bounds=bounds)
    variance, length_scale, nugget = parameters

    return variance, length_scale, nugget


def fit_noise_model(df1, value='value'):
    # TODO: Sort out duplicates systematically (even if only a small number)
    df1.drop_duplicates(subset=['point_id', 'datetime'], inplace=True)

    distances = []
    differences = []
    for point1, point2 in itertools.combinations(df1['point_id'].unique(), 2):
        distance = (
            (df1.loc[df1['point_id'] == point1, 'easting'].values[0]
             - df1.loc[df1['point_id'] == point2, 'easting'].values[0]) ** 2
            + (df1.loc[df1['point_id'] == point1, 'northing'].values[0]
               - df1.loc[df1['point_id'] == point2, 'northing'].values[0]) ** 2
        ) ** 0.5

        df1a = df1.loc[(df1['point_id'] == point1) | (df1['point_id'] == point2), ['point_id', 'datetime', value]]
        df1a = df1a.pivot(index=['datetime'], columns='point_id', values=value)
        df1a.reset_index(inplace=True)
        df1a = df1a.loc[np.isfinite(df1a[point1]) & np.isfinite(df1a[point2])]
        df1a['diff'] = df1a[point1] - df1a[point2]
        df1a['abs_diff'] = np.absolute(df1a['diff'])

        mean_diff = np.mean(df1a['abs_diff'])

        if np.isfinite(mean_diff):
            distances.append(distance)
            differences.append(mean_diff)

    distances = np.asarray(distances)
    differences = np.asarray(differences)

    if distances.shape[0] >= 5:
        bounds = ([0.0, 0.0], [1.0, 100000000.0])  # TODO: Increase bound on variance for noise model?

        parameters, _ = scipy.optimize.curve_fit(exponential_model, distances, differences, bounds=bounds)
        variance, length_scale = parameters  # nugget still not estimated at the moment

    else:
        variance = None
        length_scale = None

    return variance, length_scale  # case where nugget is and is not estimated


def fit_variogram_model(
        df1, include_elevation=True, distance_bins=7, easting='easting', northing='northing', value='value',
        elevation='elevation', return_interpolator=True, return_model=False,
):
    # Test for elevation-dependence using linear regression
    if include_elevation:
        regression_model = scipy.stats.linregress(df1[elevation], df1[value])
        if regression_model.pvalue <= 0.05:
            significant_regression = True
        else:
            significant_regression = False
    else:
        significant_regression = False

    # Remove elevation signal from data first to allow better variogram fit
    if include_elevation and significant_regression:
        values = df1[value] - (df1[elevation] * regression_model.slope + regression_model.intercept)
    else:
        values = df1[value]

    # Calculate bin edges
    max_distance = np.max(scipy.spatial.distance.pdist(np.asarray(df1[[easting, northing]])))
    interval = max_distance / distance_bins
    bin_edges = np.arange(0.0, max_distance + 0.1, interval)
    bin_edges[-1] = max_distance + 0.1  # ensure that all points covered

    # Estimate empirical variogram
    if include_elevation and significant_regression:
        bin_centres, gamma, counts = gstools.vario_estimate(
            (df1[easting].values, df1[northing].values), values, bin_edges, return_counts=True
        )
    else:
        bin_centres, gamma, counts = gstools.vario_estimate(
            (df1[easting].values, df1[northing].values), values, bin_edges, return_counts=True
        )
    bin_centres = bin_centres[counts > 0]
    gamma = gamma[counts > 0]

    # Identify best fit from exponential and spherical covariance models
    variogram_model = gstools.Exponential(dim=2)
    _, _, exponential_r2 = variogram_model.fit_variogram(bin_centres, gamma, nugget=False, return_r2=True)

    # Instantiate appropriate kriging object
    if return_interpolator:
        if include_elevation and significant_regression:
            interpolator = gstools.krige.ExtDrift(
                variogram_model, (df1[easting].values, df1[northing].values), values, df1[elevation].values
            )
        else:
            interpolator = gstools.krige.Ordinary(
                variogram_model, (df1[easting].values, df1[northing].values), values
            )

    if return_interpolator and return_model:
        return interpolator, variogram_model
    elif return_interpolator:
        return interpolator
    elif return_model:
        return variogram_model


def make_idw_interpolator(df1, easting='easting', northing='northing', value='value', point_id='point_id'):
    # - just a wrapper for getting values into interpolator function...
    # -- any values grabbed from this function scope by _interpolator will be available

    def _interpolator(coords):
        x, y = coords
        weights = {}

        # Find distances/weights for each input point relative to each station
        distances = {}
        ref_valid = []
        for _, row in df1.iterrows():
            ref_x = row[easting]
            ref_y = row[northing]
            val_ref = row[value]
            ref_id = row[point_id]
            if np.isfinite(val_ref):
                ref_valid.append(ref_id)
                distances[ref_id] = ((x - ref_x) ** 2 + (y - ref_y) ** 2) ** 0.5
                distances[ref_id][distances[ref_id] == 0.0] = 0.0000001
                weights[ref_id] = 1.0 / (distances[ref_id] ** 2.0)  # hardcoded IDW exponent

        # Normalise weights
        i = 0
        for ref_id in ref_valid:
            if i == 0:
                sum_weights = weights[ref_id].copy()
            else:
                sum_weights += weights[ref_id]
            i += 1
        for ref_id in ref_valid:
            weights[ref_id] /= sum_weights

        # Interpolate station values
        i = 0
        for _, row in df1.iterrows():
            ref_id = row[point_id]
            ref_val = row[value]
            if ref_id in ref_valid:
                if i == 0:
                    values = np.zeros(weights[ref_id].shape)
                values += (weights[ref_id] * ref_val)
                i += 1

        return values

    return _interpolator


def shift_(x, lag=1):
    y = np.zeros(x.shape, dtype=x.dtype)
    y.fill(np.nan)
    y[lag:] = x[:-lag]
    return y


def identify_half_months(date_series):
    half_months = np.zeros(date_series.shape[0], dtype=int)
    current_half_month = 1
    for month in range(1, 12+1):
        half_months[(date_series.month == month) & (date_series.day <= 15)] = current_half_month
        current_half_month += 1
        half_months[(date_series.month == month) & (date_series.day > 15)] = current_half_month
        current_half_month += 1
    return half_months


def expand_grid(column_names, *args):  # args are lists/arrays of unique values corresponding with each column
    mesh = np.meshgrid(*args)
    dc = {}
    for col, m in zip(column_names, mesh):
        dc[col] = m.flatten()
    df = pd.DataFrame(dc)
    return df


def read_grids(file_paths):
    raise NotImplementedError
