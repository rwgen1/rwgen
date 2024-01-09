import os
import sys
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

# - strong assumption that only daily data is coming in at the moment
# -- could at least make a check on time series inputs and perform aggregation if needed

# TODO: Use HadUK-Grid to supplement MIDAS-Open station time series
# - use 0900-0900 days and relative sunshine duration (0-1) for day with greatest overlap

# TODO: Spatial correlation of residuals
# TODO: Spatial interpolation of statistics and parameters

# TODO: Augmentation with externally derived mean fields, e.g. HadUK-Grid
# - need to calculate the adjustment increments/factors at each grid cell (available for interpolation)

# TODO: Option to use months with "tilting" rather than half-months
# TODO: Adjust order of input arguments - some things near bottom now perhaps can go higher up

# TODO: Elevation required in metadata (or used if available)
# TODO: Option to include elevation in setting up interpolations

# TODO: Option to work with shortwave radiation input
# TODO: Include sunshine duration lag-1 as optional in temperature regressions

# TODO: IN SIMULATION NEED SOME CHECKS ON PLAUSIBILITY OF VALUES (E.G. POSITIVE WIND SPEED, VAPOUR PRESSURE, ...)

# TODO: Option of 'ALL' transition state (at least for tertiary variables) to check whether wet/dry transitions is best

# TODO: Check for duplicated dates/values in data series

# - use climatology grids in preprocessing by figuring out adjustments to final simulated time series needed to match
# means - e.g. difference in interpolated tavg vs haduk-grid tavg as an array at each haduk-grid point (which can be
# interpolated simply to an off-centre location if needed)
# - or at least read in preprocessing so available in weather model to do something along these lines

# - no output from preprocessing yet... ultimately statistics and parameters but at output locations - so perhaps best
# written once the interpolation etc is done

# - currently no guarantee that sufficient stations with sufficient completeness will be found
# - for the moment use a large enough buffer (or try a higher buffer)
# - ultimately the search area could be expanded until some criteria met, but this means knowing how much data is
# available for each station+variable combination
# !! or add a function to check validity of preprocessing; if invalid then add to buffer and try again (i.e. call
# get metadata again etc !!

# - variograms for transformation parameters, statistics, regression parameters and residuals could be done here
# - actual interpolation to required points/grid cell can be done simulation

# - switch from half-months to "tilted months"

# - how about the following flow:
# -- standardise raw variables
# -- pool series
# -- transform using box-cox / mixed distribution for sunshine hours [transform by transition if pooled...?]
# -- standardise transformed series
# -- do regressions using pooled series [well, standardised-pooled-transformed-standardised series]
# -- calculate residuals by station
# -- estimate variograms for (1) raw statistics, (2) r**2, (3) residuals [not transformation or regression parameters]
# --- test for easting, northing and elevation as predictors first
# - the simulation proceeds by:
# -- interpolate raw statistics and input statistics [means] if given
# -- interpolate r**2
# -- timestep loop
# --- do regressions
# --- simulate the error/residual term as a standard normal variable
# --- rescale (or quantile-based transform) error/residual to fit with location-specific r**2 [and add to prediction]
# --- adjust final values (additively or multiplicatively) using input statistics [means] if given
# - means still need to interpolate or model mean and sd (+ r**2 and residuals spatial correlation), but it gets rid
# of need to interpolate transformation parameters and regression parameters
# -- still potentially need wide area to be able to interpolate statistics
# - option to assume spatially homogeneous mean, sd, r**2 and ignore residuals spatial correlation?
# -- option to assume constant coefficient of variation if supplying input mean?
# - option to adjust shorter record station statistics [mean, sd] for consistency with full period?

# - start off with spatially uniform error/residual? or better just to go straight to simulation?


class Preprocessor:

    def __init__(
            self,
            spatial_model,
            input_timeseries,  # infer input variables | file path or folder of files
            point_metadata,  # optional if single site | rename as point_metadata in line with rainfall model?
            climatology_grids,  # dict of file paths (or opened files)
            output_folder,
            xmin,
            xmax,
            ymin,
            ymax,
            spatial_method,  # use one station, pool multiple stations or interpolate multiple stations
            max_buffer,  # km - but easting/northing in m so convert - could be set to zero
            # min_years,  # minimum number of years data in pooled series (e.g. 30 * 365.25 - no missing)
            min_points,  # minimum number of stations if using interpolation or pooling
            wet_threshold,
            use_neighbours,  # use neighbouring precipitation record to try to infill wet days
            neighbour_radius,  # km - but easting/northing in m so convert
            calculation_period,
            completeness_threshold,
            predictors,
            input_variables,  # list of variables to work with - primarily geared out sunshine duration vs incoming SW
            season_length,  # 'month' or 'half-month' - make "tilting" optional for monthly?
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
        self.input_variables = input_variables  # ! needs to match column names after they have been made lower case !
        # - could be done via a dictionary? or just put a check here on items in list
        # - input variables should not include precipitation
        self.season_length = season_length

        # TODO: Add checks on input arguments, e.g. input_variables, season_lengths - ensure options are valid

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
        self.regressions = {}  # rename as parameters? including r**2 as a parameter because used in simulation
        self.parameters = {}
        self.residuals = {}
        self.r2 = {}
        self.standard_errors = {}
        # self.parameter_variograms = {}  # but would also need variograms for transformation parameters and statistics

        # self.elevation_gradients = {}  # no longer used if focusing on variograms...?

        self.statistics_variograms = {}
        self.residuals_variograms = {}
        self.r2_variograms = {}
        self.se_variograms = {}
        self.noise_models = {}

        # Run the preprocessing
        # self.run()

    # def get_metadata(self, point_metadata):
    #     # Read metadata file
    #     if isinstance(point_metadata, str):
    #         metadata = pd.read_csv(point_metadata)
    #     else:
    #         metadata = point_metadata
    #     metadata.columns = [column.lower() for column in metadata.columns]
    #
    #     # Subset on geographical domain
    #     if self.find_stations:
    #         metadata = metadata.loc[
    #             (self.metadata['easting'] >= self.xmin - self.max_buffer * 1000.0)
    #             & (self.metadata['easting'] <= self.xmax + self.max_buffer * 1000.0)
    #             & (self.metadata['northing'] >= self.ymin - self.max_buffer * 1000.0)
    #             & (self.metadata['northing'] <= self.ymax - self.max_buffer * 1000.0)
    #         ]
    #
    #     # Construct KD-tree to enable use of neighbouring precipitation record
    #     if self.use_neighbours:
    #         kd_tree = scipy.spatial.KDTree(metadata[['easting', 'northing']])
    #     else:
    #         kd_tree = None
    #
    #     return metadata, kd_tree

    def preprocess(self):
        # print('--', self.metadata.shape[0], '--')
        # i = 0
        # for point_id in self.metadata['point_id']:
        #     print(point_id, '(' + str(i) + ')')
        #     self.process_station(point_id)
        #     i += 1

        # Assuming that a reasonable input series is passed in for a point model or for a spatially uniform model
        # - no explicit checks that sufficient data are available...
        # print('  - Getting data')
        if self.spatial_model and (self.spatial_method != 'uniform'):
            self.process_stations()
        else:
            self.process_station()

        # Subset point metadata on points that are being used
        if self.spatial_model:
            self.metadata = self.metadata.loc[self.metadata['point_id'].isin(self.data_series['point_id'].unique())]

        # Pooling - dfs in self.data_series then contain point_id and pool_id to help pool if required (via groupby)
        # self.data_series = pd.concat(list(self.data_series.values()))
        if self.spatial_method == 'interpolate':
            self.data_series['pool_id'] = self.data_series['point_id']
        else:
            self.data_series['pool_id'] = 1

        # print(self.data_series)
        # print(self.data_series.columns)
        # sys.exit()

        # print(self.data_series['point_id'].unique())
        # print(self.n_years)
        # print(self.n_points)
        # sys.exit()

        # Transformation and second standardisation (reshaping data to wide with lagged series as columns)
        # print('  - Transformations')
        self.transform_series()

        # sys.exit()

    def fit(self):

        # Regression
        # print('  - Regressions')
        self.do_regression()

        # pool_id = 1
        # # i = 0
        # for season, variable, transition in itertools.product(
        #         self.seasons, self.input_variables, ['DDD', 'DD', 'DW', 'WD', 'WW']
        # ):
        #     print(season, variable, transition, self.regressions[(pool_id, season, variable, transition, 'r-squared')])
        #     # print(self.regressions[(pool_id, season, variable, transition, 'parameters')])
        #     # i += 1
        #     # if i == 5:
        #     #     break
        # # print('y')
        # sys.exit()
        # !! so now no failed regressions due to nan prcp - good !!
        # !! BUT only first two values of parameters are non-zero in regressions - looks suspicious? !!
        # - ok looking better now - fixed assignment to columns of X

        # !! where to do the tilting?? !! - perhaps just leave out for now... especially if pooling is an option

        # Fit variogram models for interpolation
        if self.spatial_model and (self.spatial_method != 'uniform'):
            # print('  - Fitting variograms')
            # self.estimate_noise_variograms()
            # sys.exit()
            # !! self.estimate_residual_variograms()  # for attempting to simulate spatial residuals fields directly
            self.estimate_statistic_variograms()
            self.estimate_r2_variograms()
            self.estimate_se_variograms()

    def process_stations(self):
        buffer = 0.0
        processed_ids = []
        # while (min(n_years.values()) < self.min_years) and (buffer <= self.max_buffer):
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
        # print(point_id)
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
        df = pd.read_csv(input_path, index_col=0, parse_dates=True, infer_datetime_format=True, dayfirst=True)
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
                        df1 = pd.read_csv(
                            neighbour_path, index_col=0, parse_dates=True, infer_datetime_format=True, dayfirst=True
                        )
                        df1.columns = [column.lower() for column in df1.columns]
                        df1.reset_index(inplace=True)
                        df1.rename(columns={
                            'prcp': 'prcp_neighbour',
                            # 'temp_min': 'temp_min_neighbour',
                            # 'temp_max': 'temp_max_neighbour',
                        }, inplace=True)
                        df = pd.merge(
                            df, df1[['datetime', 'prcp_neighbour']],  # , 'temp_min_neighbour', 'temp_max_neighbour']],
                            how='left', on='datetime'
                        )
                        df['wet_day_neighbour'] = np.where(df['prcp_neighbour'] >= self.wet_threshold, 1, 0)
                        df['wet_day'] = np.where(
                            ~np.isfinite(df['wet_day']) & np.isfinite(df['prcp_neighbour']),
                            df['wet_day_neighbour'],
                            df['wet_day']
                        )
                        for variable in ['prcp']:  # , 'temp_min', 'temp_max']:
                            df[variable] = np.where(
                                ~np.isfinite(df[variable]) & np.isfinite(df[variable + '_neighbour']),
                                df[variable + '_neighbour'],
                                df[variable]
                            )
                        df.drop(columns={
                            'prcp_neighbour', 'wet_day_neighbour'},  # 'temp_min_neighbour', 'temp_max_neighbour',
                            inplace=True
                        )

            # Check all variables present and complete in case the series has been updated
            df['temp_avg'] = (df['temp_min'] + df['temp_max']) / 2.0
            df['dtr'] = df['temp_max'] - df['temp_min']

            # print(df)
            # print(df.columns)

            # Identify completeness by variable
            completeness = {}
            for variable in self.input_variables:
                # print(variable)
                if variable in df.columns:
                    # if df.shape[0] >= (self.completeness_threshold / 100.0) * self.period_length:
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
                        # print(variable, completeness[variable])
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
                # df.reset_index(inplace=True)
                df = pd.melt(df, id_vars=['datetime', 'season', 'prcp', 'wet_day'])

                # Subset on variables that need to be taken forward
                df = df.loc[~df['variable'].isin(['temp_mean', 'temp_min', 'temp_max', 'rel_hum'])]
                # ! could be done based on self.input_variables now !

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

        # !! Check that this referencing updates self.data_series !!
        # - handled now by returning series that contains transformed and (again) standardised series
        df = self.data_series

        # Factors by which to stratify transformation
        transitions = ['DDD', 'DD', 'DW', 'WD', 'WW', 'NA']  # ! is NA needed to keep serially complete here?
        variables = df['variable'].unique()
        pool_ids = df['pool_id'].unique()

        # Main loop
        dfs = []
        # for season, transition, variable, pool_id in itertools.product(self.seasons, transitions, variables, pool_ids):  # !221212
        for season, variable, pool_id in itertools.product(self.seasons, variables, pool_ids):  # !221212
            df1 = df.loc[
                (df['season'] == season) & (df['variable'] == variable)  # (df['transition'] == transition) &  # !221212
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

                # print(season, lamda)

                # if (variable == 'temp_avg') and (season == 1):
                #     print(season, transition, variable, df1.shape[0], np.sum(~np.isfinite(df1['z_score'])), lamda)
                #     df1.to_csv('H:/Projects/rwgen/working/iss13/weather/bc_check1e.csv', index=False)
                #     sys.exit()

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

                # --
                # df1.to_csv('H:/Projects/rwgen/working/iss13/weather/beta1.csv', index=False)
                # print(p0, a, b, loc, scale)
                # sys.exit()
                # --

                df1.drop(columns=['scaled', 'probability'], inplace=True)
                dfs.append(df1)

            elif (df1.shape[0] > 0) and (variable == 'prcp'):
                # Not transforming precipitation currently
                # - prcp will be standardised later, but this does not affect the regression
                # TODO: Consider whether wet-day precipitation should undergo (just box-cox?) transformation
                # - box-cox does not seem to work very well based on lerwick test
                # - weibull might work reasonably... two parameters
                df1['bc_value'] = df1['value']
                # df1['bc_value'] = df1['z_score']  # TESTING
                dfs.append(df1)

            else:
                df1['bc_value'] = np.nan

        # Join all back into one dataframe
        df = pd.concat(dfs)
        df.sort_values(['pool_id', 'point_id', 'variable', 'datetime'], inplace=True)

        # print(df.loc[(df['season'] == 1) & (df['transition'] == 'DD') & (df['variable'] == 'prcp')])
        # sys.exit()

        # ---

        # Standardisation of transformed values
        # !221212 - testing without stratification by transition state

        # Calculate statistics for standardisation
        df1 = df.loc[df['transition'] != 'NA']
        df1 = df1.groupby(['pool_id', 'variable', 'season'])['bc_value'].agg(['mean', 'std'])  # , 'transition'
        df1.reset_index(inplace=True)

        # ! In DD or DDD, mean and sd of prcp are zero by definition => standardise to nan !
        # print(df1.loc[(df1['season'] == 1) & (df1['transition'] == 'DD') & (df1['variable'] == 'prcp')])
        # sys.exit()

        # Check if not all transitions represented - use average if missing for now
        # - interpolate? - e.g. DDD and DW to give you DD if missing? or find most "similar" available transition?
        tmp1 = expand_grid(
            ['pool_id', 'variable', 'season'],  # , 'transition'
            df1['pool_id'].unique(), df1['variable'].unique(), df1['season'].unique()  # , df1['transition'].unique()
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
        # self.transformed_statistics = df1  # set by returning

        # Standardise time series
        # - keep series contiguous (i.e. using NA) to ensure that lag-1 value is identified correctly
        df = pd.merge(df, df1, how='left')
        df['sd_value'] = (df['bc_value'] - df['bc_mean']) / df['bc_std']
        df['sd_lag1'] = df.groupby(['pool_id', 'point_id', 'variable', 'season'])['sd_value'].transform(shift_)

        # print(df.loc[(df['season'] == 1) & (df['variable'] == 'prcp')])  # (df['transition'] == 'DD') &
        # sys.exit()

        # Wide dataframe containing standardised values and lag-1 standardised values for all variables
        index_columns = ['pool_id', 'point_id', 'datetime', 'season', 'transition']  #
        tmp1 = df.pivot(index=index_columns, columns='variable', values='sd_value')
        tmp1.reset_index(inplace=True)
        tmp2 = df.pivot(index=index_columns, columns='variable', values='sd_lag1')
        tmp2.reset_index(inplace=True)
        tmp2.columns = [col + '_lag1' if col not in index_columns else col for col in tmp2.columns]
        df2 = pd.merge(tmp1, tmp2)

        # print(df2.columns)
        # sys.exit()
        # print(
        #     df2.loc[(df2['season'] == 1) & (df2['transition'] == 'DD'),
        #             ['prcp', 'prcp_lag1', 'vap_press', 'vap_press_lag1']]
        # )
        # sys.exit()

        # ---

        self.transformed_statistics = df1
        self.data_series = df2
        # return df1, df2

        # print(self.transformed_statistics)
        # print(self.data_series)
        # print(self.data_series.columns)
        # sys.exit()

    def do_regression(self):
        # df2 is wide df - not in self.data_series
        df2 = self.data_series  # refactor variable name ultimately

        # df2.to_csv('H:/Projects/rwgen/working/iss13/weather/df2_1.csv', index=False)
        # sys.exit()

        # Set up factors to loop
        pool_ids = df2['pool_id'].unique().tolist()
        transitions = df2['transition'].unique().tolist()
        if 'NA' in transitions:
            transitions.remove('NA')
        variables = self.input_variables.copy()
        # if 'prcp' in variables:
        #     variables.remove('prcp')

        # Calculate regression parameters by transition state
        # regression_results = {}  # switched to self.regressions
        for pool_id, season, transition, variable in itertools.product(pool_ids, self.seasons, transitions, variables):

            # Subset on relevant finite values - successively for dependent and then each independent variable
            df2a = df2.loc[
                (df2['pool_id'] == pool_id) & (df2['season'] == season) & (df2['transition'] == transition)
                & (np.isfinite(df2[variable]))
            ]
            for predictor in self.predictors[(variable, transition)]:
                df2a = df2a.loc[np.isfinite(df2a[predictor])]

            # print(df2a)
            # print(df2a.columns)
            # sys.exit()

            # Populate array for regression
            n_times = df2a.shape[0]
            n_predictors = len(self.predictors[(variable, transition)])
            X = np.zeros((n_times, n_predictors))
            col_idx = 0
            for predictor in self.predictors[(variable, transition)]:
                X[:,col_idx] = df2a[predictor].values
                col_idx += 1

            # if (season == 1) and (variable == 'temp_avg') and (transition == 'DW'):
            #     print(df2a['prcp'])
            #     print(df2a.columns)
            #     print(X)
            #     sys.exit()

            # Set a minimum number of days for performing regression - as user option?
            if X.shape[0] >= 10:

                # Need regression parameters, r-squared and residuals for spatial correlation
                X = sm.add_constant(X)  # adds column of ones - required for intercept to be estimated
                model = sm.OLS(df2a[variable].values, X)
                results = model.fit()
                # self.regressions[(pool_id, season, variable, transition, 'parameters')] = results.params
                self.parameters[(pool_id, season, variable, transition)] = results.params
                # self.regressions[(pool_id, season, variable, transition, 'r-squared')] = results.rsquared
                df2b = df2a[['datetime', 'point_id', variable]].copy()
                df2b['residual'] = results.resid
                # self.regressions[(pool_id, season, variable, transition, 'residuals')] = df2b
                self.residuals[(pool_id, season, variable, transition)] = df2b

                # Calculate r2 by point (not pool)
                df2b['fitted'] = results.fittedvalues
                # df2b['check'] = df2b[variable] - df2b['fitted'] - df2b['residual']
                # print(df2b)
                # print(df2b.columns)
                # print(dir(results))
                # print(np.min(df2b['check']), np.max(df2b['check']))
                # df2c = df2b.groupby(['point_id'])[[variable, 'fitted']].agg(corr_)
                df2c = df2b.groupby('point_id')[[variable, 'fitted']].corr().unstack().iloc[:, 1]  # series
                df2c = df2c.to_frame('r')
                df2c['r2'] = df2c['r'] ** 2
                df2c.reset_index(inplace=True)
                for _, row in df2c.iterrows():
                    self.r2[(row['point_id'], season, variable, transition)] = row['r2']

                # print(self.r2)
                # print(isinstance(df2c, pd.DataFrame), isinstance(df2c, pd.Series))

                # TESTING - se as well as r2
                df2d = df2b.groupby('point_id')['residual'].std()
                # print(df2d)
                df2d = df2d.to_frame('residual')
                df2d.reset_index(inplace=True)
                # print(df2d)
                # sys.exit()
                for _, row in df2d.iterrows():
                    self.standard_errors[(row['point_id'], season, variable, transition)] = row['residual']

                # sys.exit()

                # - next do r2 by point_id - fraction of variance explained by regression
                # -- know that variance of dependent variable is 1
                # -- so 1 - r2 = fraction of unexplained variance = unexplained variance

                # split up so have self.parameters, self.r2 and self.residuals?
                # --

            else:
                print(season, transition, variable)
                # TODO: Need to handle this case explicitly

        # TODO: Fill in missing parameters? Average of available transitions?
        # - problem may diminish with use of months rather than half-months (or higher completeness thresholds)
        # - easier to infill via dataframe operations than with dictionary - unless do within loop above?
        # - alternative to infilling is just to let spatial interpolation do its thing - just means that the point will
        # not be used in figuring out the spatial correlation of residuals

    def estimate_gradients(self):  # elevation gradients in raw statistics only currently
        df = pd.merge(self.raw_statistics, self.point_metadata[['point_id', 'elevation']], on='point_id')
        for season, variable in itertools.product(self.seasons, df['variable'].unique()):
            df1 = df.loc[(df['season'] == season) & (df['variable'] == variable)]
            X = np.zeros((df1.shape[0], 1))
            X[:,0] = df1['elevation'].values
            X = sm.add_constant(X)

            for statistic in ['mean', 'std']:
                y = df1[statistic].values  # - df1[statistic].mean()
                model = sm.OLS(y, X)
                results = model.fit()
                self.elevation_gradients[(season, variable, statistic, 'parameters')] = results.params
                self.elevation_gradients[(season, variable, statistic, 'r-squared')] = results.rsquared
                self.elevation_gradients[(season, variable, statistic, 'residuals')] = results.resid
                # TODO: Get p-value on gradient term
                # TODO: Need to store residuals coincident with point_id in df so can interpolate

    # 31/08 new approach to spatially varying residuals
    # - start from residuals - in self.residuals - similar to estimate_residual_variograms()
    # - loop season, variable and maybe transition
    # - standardise/scale residuals for each point so mean = 0 and sd = 1
    # - fit covariance model
    # -- try calculating pairwise covariance and fitting to this (vs separation distance) - update fit_covariance_model
    # - then in simulation use covariance model for field simulation
    # -- for many simulations, each point should have a normal distribution with mean = 0 and sd = 1
    # - scale/transform residuals for point using standard error for that point

    def estimate_residual_variograms(self):
        # old notes:
        # not stratifying by transition state currently - assuming similar spatial correlation structure
        # ! is a further standardisation of residuals necessary here? !
        # - perhaps yes, as the variogram will be based on differences?
        # - or will it just average out?
        # - better to explicitly fit a covariance/correlation using correlations (of residuals) vs separation distance?

        pool_ids = self.data_series['pool_id'].unique()
        transitions = ['DDD', 'DD', 'DW', 'WD', 'WW']
        for season, variable, transition in itertools.product(self.seasons, self.input_variables, transitions):

            # Loop pools and transitions to bring all residuals together in one df
            # - one variogram per season + variable + transition - no per season+variable?
            # TODO: See about switching to stratification by transition - check works reasonably data-wise
            dfs = []
            for pool_id in pool_ids:
                for transition in ['DDD', 'DD', 'DW', 'WD', 'WW']:

                    # Standardise residuals
                    # - mean of residuals should be close to zero
                    # - know standard deviation of residuals as stored in self.standard_errors, but should be the same
                    # if recalculated here...
                    tmp = self.residuals[(pool_id, season, variable, transition)].copy()
                    tmp['sd'] = tmp.groupby('point_id')['residual'].transform('std')
                    tmp['residual_sa'] = (tmp['residual'] - 0.0) / tmp['sd']
                    dfs.append(tmp)
            df = pd.concat(dfs)

            # Join easting and northing
            df = pd.merge(df, self.metadata[['point_id', 'easting', 'northing']], how='left')

            # print(df)
            # print(df.columns)
            # sys.exit()

            # print(season, variable)
            # df.to_csv('H:/Projects/rwgen/working/iss13/weather/cov_1.csv', index=False)
            # sys.exit()

            # for point_id in df['point_id'].unique():
            #     print(self.standard_errors[(point_id, season, variable, transition)])
            # sys.exit()

            # Set up interpolator for residuals
            if df['point_id'].unique().shape[0] >= self.min_points:
                variance, length_scale, nugget = fit_covariance_model(df, value='residual_sa')
                self.residuals_variograms[(season, variable)] = (variance, length_scale, nugget)
            else:
                self.residuals_variograms[(season, variable)] = None

            # print(season, variable, variance, length_scale)

    def estimate_noise_variograms(self):
        # transitions = ['DDD', 'DD', 'DW', 'WD', 'WW']
        for season, variable in itertools.product(self.seasons, self.input_variables):  # , transitions):
            # print(season, variable)
            df = self.data_series.loc[
                (self.data_series['season'] == season),  # & (self.data_series['transition'] == transition),
                ['point_id', 'datetime', variable]
            ]

            # Join easting and northing
            df = pd.merge(df, self.metadata[['point_id', 'easting', 'northing']], how='left')

            # print(season, variable)
            # df.to_csv('H:/Projects/rwgen/working/iss13/weather/cov_1.csv', index=False)
            # sys.exit()

            # Set up interpolator
            if df['point_id'].unique().shape[0] >= self.min_points:
                # print(season, variable, transition, df.loc[np.isfinite(df[variable])].shape[0])
                variance, length_scale = fit_noise_model(df, value=variable)

                # -- testing only --
                # variogram_model = fit_variogram_model(
                #     df, include_elevation=False, value=variable, return_model=True, return_interpolator=False
                # )
                # print(variogram_model)
                # print(variogram_model.var, variogram_model.len_scale)
                # sys.exit()
                # -- testing only --

                if variance is not None:
                    self.noise_models[(season, variable)] = (variance, length_scale)
                else:
                    self.noise_models[(season, variable)] = None  # ! infill/interpolate somehow? !
            else:
                self.noise_models[(season, variable)] = None

            # print(season, variable, variance, length_scale)

    def estimate_statistic_variograms(self):
        # TODO: Add elevation to metadata so that it can be included
        df = pd.merge(
            self.raw_statistics, self.metadata[['point_id', 'easting', 'northing']], on='point_id', how='left',  # , 'elevation'
        )
        for season, variable in itertools.product(self.seasons, self.simulation_variables):  # df['variable'].unique()):
            df1 = df.loc[(df['season'] == season) & (df['variable'] == variable)]
            # print(season, variable)
            # print(df1)
            for statistic in ['mean', 'std']:
                if df1['point_id'].unique().shape[0] >= self.min_points:
                    interpolator = fit_variogram_model(
                        df1, include_elevation=False, value=statistic  # TODO: Use elevation
                    )
                    interpolation_type = 'kriging'
                else:
                    interpolator = make_idw_interpolator(df1, value=statistic)
                    interpolation_type = 'idw'

                # --
                # if interpolation_type == 'kriging':
                #     value = interpolator(
                #         (np.array([385000+1000]), np.array(175000+1000)),
                #         mesh_type='unstructured',
                #         return_var=False
                #     )
                # else:
                #     value = interpolator(
                #         (np.array([385000 + 1000]), np.array(175000 + 1000))
                #     )
                #     print(interpolation_type, season, variable, statistic, value)
                # print(interpolation_type, season, variable, statistic, value)
                # --

                self.statistics_variograms[(season, variable, statistic)] = (interpolation_type, interpolator)
        # sys.exit()

    def estimate_r2_variograms(self):
        # TODO: Add elevation to metadata so that it can be included

        # self.r2[(row['point_id'], season, variable, transition)] = row['r2']

        # df = pd.merge(
        #     self.r2, self.metadata[['point_id', 'easting', 'northing']], on='point_id', how='left',  # , 'elevation'
        # )

        transitions = ['DDD', 'DD', 'DW', 'WD', 'WW']
        for season, variable, transition in itertools.product(self.seasons, self.input_variables, transitions):

            df1 = self.metadata[['point_id', 'easting', 'northing']].copy()  # 'elevation'
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

            # --
            # if interpolation_type == 'kriging':
            #     value = interpolator(
            #         (np.array([385000+1000]), np.array(175000+1000)),
            #         mesh_type='unstructured',
            #         return_var=False
            #     )
            # else:
            #     value = interpolator(
            #         (np.array([385000 + 1000]), np.array(175000 + 1000))
            #     )
            #     # print(interpolation_type, season, variable, statistic, value)
            # print(interpolation_type, season, variable, transition, value)
            # --

            self.r2_variograms[(season, variable, transition)] = (interpolation_type, interpolator)
        # sys.exit()

    # --
    # COPY OF ABOVE  # TODO: Rationalise
    def estimate_se_variograms(self):
        # TODO: Add elevation to metadata so that it can be included

        transitions = ['DDD', 'DD', 'DW', 'WD', 'WW']
        for season, variable, transition in itertools.product(self.seasons, self.input_variables, transitions):

            df1 = self.metadata[['point_id', 'easting', 'northing']].copy()  # 'elevation'
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
        # sys.exit()
    # --

    def calculate_diurnal_cycles(self):
        pass
        # need hourly data as input for this if doing based on data...
        # just simple profiles for now based on shortwave radiation / sunshine cycles (or uniform...)
        # perhaps do in simulation for now


def corr_(x, y):
    return np.corrcoef(x, y)[0,1]


def exponential_model(distance, variance, length_scale, nugget, flip=True):
    x = variance * (1.0 - np.exp(-distance / length_scale)) + nugget
    if flip:
        x = 1.0 - x
    return x


def fit_covariance_model(df1, value='value'):
    # self.standard_errors[(row['point_id'], season, variable, transition)]

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

        # df1a.to_csv('Z:/DP/Work/ER/rwgen/testing/weather/cov1.csv', index=False)

        # print(df1a)
        # print(df1a.columns)
        # sys.exit()

        result = scipy.stats.pearsonr(df1a[point1], df1a[point2])
        correlation = result[0]  # newer scipy (1.9.0) would need result.statistic
        # print(point1, point2, distance, correlation)

        covariance = np.cov(df1a[[point1, point2]].values, rowvar=False)[0,1]
        # print(covariance)
        # print(covariance[0,1])
        # sys.exit()

        # cov = np.cov(df1a[point1], df1a[point2])
        # print(correlation)
        # print(cov)
        # sys.exit()

        distances.append(distance)
        correlations.append(correlation)
        covariances.append(covariance)

    # fp = 'Z:/DP/Work/ER/rwgen/testing/weather/cov1d.csv'
    # with open(fp, 'w') as fh:
    #     fh.write('distance,covariance,correlation\n')
    #     for d, cov, cor in zip(distances, covariances, correlations):
    #         ol = ','.join(str(item) for item in [d, cov, cor])
    #         fh.write(ol + '\n')
    # sys.exit()

    # After standardisation covariances and correlations are similar - use correlations for now
    # - hardcode a correlation at distance zero and a nugget of for now
    # distances.append(0.0)
    # correlations.append(1.0)
    distances = np.asarray(distances)
    correlations = np.asarray(correlations)

    # fixing sill to one and optimising nugget (0-1 bounds ok?)
    bounds = ([0.99, 0.0, 0.0], [1.0, 100000000.0, 1.0])

    parameters, _ = scipy.optimize.curve_fit(exponential_model, distances, correlations, bounds=bounds)
    variance, length_scale, nugget = parameters

    # print(variance, length_scale, nugget)
    # sys.exit()

    return variance, length_scale, nugget


def fit_noise_model(df1, value='value'):
    # TODO: Sort out duplicates systematically (even if only a small number)
    #print(df1.shape)
    df1.drop_duplicates(subset=['point_id', 'datetime'], inplace=True)
    #print(df1.shape)
    #sys.exit()

    # print(df1)

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

        # if ~np.isfinite(mean_diff):
        #     print(df1a)
        #     sys.exit()

        # df1a.to_csv('H:/Projects/rwgen/working/iss13/weather/noise1.csv', index=False)
        # print(df1a)
        # sys.exit()

        # result = scipy.stats.pearsonr(df1a[point1], df1a[point2])
        # correlation = result[0]  # newer scipy (1.9.0) would need result.statistic
        # print(point1, point2, distance, correlation)

        if np.isfinite(mean_diff):
            distances.append(distance)
            differences.append(mean_diff)

    # distances.append(0.0)
    # differences.append(0.0)
    distances = np.asarray(distances)
    differences = np.asarray(differences)
    # flipped_correlations = 1.0 - correlations
    # flipped_correlations[flipped_correlations < 0.0] = 0.0  # needed?

    if distances.shape[0] >= 5:

        # tmp = pd.DataFrame({'dist': distances, 'diff': differences})
        # tmp.to_csv('H:/Projects/rwgen/working/iss13/weather/noise3.csv', index=False)

        # bounds = ([0.0, 0.0], [1.0, 100000000.0])  # np.inf
        bounds = ([0.0, 0.0], [1.0, 100000000.0])  # !! increase bound on variance for noise model? !!

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
    # Adapted from rainfall/simulation.py/make_phi_interpolator
    # - df1 to include columns: easting, northing, value and optionally elevation
    # - i.e. any subsetting to be done first

    # print(df1)
    # print(df1.columns)
    # sys.exit()

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

    # --
    # from matplotlib import pyplot as plt
    # ax = variogram_model.plot(x_max=max(bin_centres))
    # ax.scatter(bin_centres, gamma)
    # plt.show()
    # print(variogram_model)
    # sys.exit()
    # --

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

    # --
    # values = interpolator(
    #     (df1[easting].values, df1[northing].values),
    #     mesh_type='unstructured',
    #     return_var=False
    # )
    # print(df1[value].values)
    # print(values)
    # print(df1[easting].values)
    # print(df1[northing].values)
    # sys.exit()
    # --

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

        # print(df1)

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


# def dict_to_df(dc):
#     df = pd.DataFrame.from_dict(dc, orient='index')
#     df.reset_index(inplace=True)
#     # factor_cols =
#     df1[['i1', 'i2']] = pd.DataFrame(df1['index'].tolist(), index=df1.index)
#     df = pd.DataFrame()


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


def read_grids(file_paths):  # ??
    raise NotImplementedError
