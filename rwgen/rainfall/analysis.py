import os
import sys  # TEMPORARY
import datetime
import itertools

import numpy as np
import pandas as pd
import scipy.stats
import scipy.interpolate

from . import utils


def main(
        spatial_model,
        season_definitions,
        statistic_definitions,
        timeseries_format,  # rename timeseries_ as input_
        start_date,  # NEW - only needed if timeseries_format='txt'
        timestep_length,  # NEW - only needed if timeseries_format='txt'
        calendar,  # NEW - only needed if timeseries_format='txt'
        timeseries_path,
        timeseries_folder,
        point_metadata,
        calculation_period,
        completeness_threshold,
        output_statistics_path,  # CHANGED - output_point_statistics_path,
        outlier_method,
        maximum_relative_difference,
        maximum_alterations,
        analysis_mode,  # 'reference_preprocessing' or 'simulation_postprocessing'
        n_years,  # specify for postprocessing or 'infer' - not needed for preprocessing...
        n_realisations,  # = 1 for reference; presume that realisations can be lumped together?
        subset_length,  # how many years to use in each subset - keep separate from calculation_period?
        output_amax_path,  # output_maxima_folder ?
        amax_durations,
        output_ddf_path,  # boolean and/or list ['annual', 'seasonal']; output_ddf_path ?
        ddf_return_periods,  # None or dict(annual=[], seasonal=[])
        write_output,
):
    """
    Make tables/dataframes of reference statistics, weights and scale factors for use in model fitting.

    Notes:
        There are two scale factors used by the model:
            * gs - Scales a given statistic for a given point by its annual mean in the objective function
            * phi - Scales the NSRP rainfall process at a given site for spatial variation in the mean and variance

    """

    # TODO: Consider what to do in the case of big dfs (reading might be OK, but aggregation...?)
    # TODO: Simulation DDFs could be maybe be done just empirically, but maybe not (especially higher RPs)
    # - and observation DDFs NEED to be done by fitting a distribution
    # TODO: Need to be able to read from text files and construct dates (current simulation output)
    # TODO: Use np.float32 as data type

    # TODO: Reinstate preprocessing using new read (chunked) approach

    # TODO: Add sub-hourly functionality
    # TODO: Accommodate csv and csvy formats

    # Unique durations based on union of statistic_definitions and maxima_durations
    durations = np.unique(statistic_definitions['duration']).tolist()
    if amax_durations is not None:
        durations.extend(list(amax_durations))
    durations = sorted(list(set(durations)))

    # Number of subsets to break timeseries into (maximum subset length of 100 years for now)
    if subset_length is None:
        n_subsets = 1
    else:
        n_subsets = int(np.ceil(n_years / subset_length))
        if (n_years % subset_length == 0) or (subset_length is None):
            last_subset_complete = True
        else:
            last_subset_complete = False

    # Dataframes for each point are stored in dictionaries using (realisation_id, subset_id) as key. Values are a list
    # of dataframes (containing point_id as a column) that are concatenated to produce one dataframe
    dfs = {}

    # Main loops - timeseries dataframes are only needed for the current realisation/subset combination
    # print('    - Point statistics')
    for realisation_id in range(1, n_realisations+1):

        # Lists of points and file paths so point and spatial models can be called in same loop
        # TODO: Account for realisation identifiers using an input argument
        file_extension = '.' + timeseries_format
        if not spatial_model:
            point_ids = [1]
            input_paths = [timeseries_path + file_extension]
        else:
            point_ids = point_metadata['point_id'].values.tolist()
            input_paths = []
            for _, row in point_metadata.iterrows():
                file_name = row['name'] + file_extension
                input_path = os.path.join(timeseries_folder, file_name)
                input_paths.append(input_path)

        # Check for realisation identifier in file names
        if analysis_mode == 'postprocessing':
            file_suffix = '_r' + str(realisation_id) + file_extension
            input_paths = [fp.replace(file_extension, file_suffix) for fp in input_paths if fp.endswith(file_extension)]

        # Readers for iteration for each point
        readers = {}
        for point_id, input_path in zip(point_ids, input_paths):
            if timeseries_format == 'txt':
                readers[point_id] = pd.read_csv(input_path, header=None, iterator=True, dtype=np.float32)
            elif timeseries_format == 'csv':
                readers[point_id] = pd.read_csv(
                    input_path, names=['value'], index_col=0, dtype={'value': np.float32}, skiprows=1, parse_dates=True,
                    infer_datetime_format=True, dayfirst=True, iterator=True
                )

        # Loop subsets
        for subset_id in range(1, n_subsets + 1):
            # print(realisation_id, subset_id)  # TODO: Update progress indicator

            # Check no references to a prior set of dataframes
            df = 0
            timeseries = 0
            timeseries = {}

            # Subset dates - using dummy to ensure fit within permitted date range
            if (analysis_mode == 'postprocessing') or (timeseries_format == 'txt'):
                if subset_id == 1:
                    subset_start_date = start_date
                else:
                    subset_start_date = subset_end_date
                if subset_start_date.year > 2100:
                    subset_start_date = datetime.datetime(
                        subset_start_date.year - 400, subset_start_date.month, subset_start_date.day,
                        subset_start_date.hour, subset_start_date.minute
                    )
                subset_end_date = datetime.datetime(
                    subset_start_date.year + subset_length, subset_start_date.month, subset_start_date.day,
                    subset_start_date.hour, subset_start_date.minute
                )
                # subset_end_date -= datetime.timedelta(seconds=1)  # if not using then need to loop etc with < (not <=)

                # Identify chunk size and index of dates
                datetime_helper = utils.make_datetime_helper(
                    subset_start_date.year, subset_end_date.year - 1, timestep_length, calendar
                )
                chunk_size = datetime_helper['end_timestep'].values[-1]
                freq_alias = str(int(timestep_length)) + 'H'
                date_index = pd.period_range(subset_start_date, periods=chunk_size, freq=freq_alias)

            # Otherwise if preprocessing and non-txt then just need chunk size
            else:
                chunk_size = 1000 * 365 * 24  # TODO: Improve this approach

            # Point loop
            for point_id in point_ids:

                df = readers[point_id].get_chunk(chunk_size)
                if timeseries_format == 'txt':
                    df.index = date_index
                    df.columns = ['value']

                # Prepare time series for analysis - join seasons, remove outliers, aggregate
                timeseries[point_id] = prepare_point_timeseries(
                    df=df,
                    calculation_period=calculation_period,
                    season_definitions=season_definitions,
                    completeness_threshold=completeness_threshold,
                    durations=durations,
                    outlier_method=outlier_method,
                    maximum_relative_difference=maximum_relative_difference,
                    maximum_alterations=maximum_alterations,
                )

                # Calculate statistics and store dataframes
                point_statistics = calculate_point_statistics(
                    statistic_definitions.loc[statistic_definitions['name'] != 'cross-correlation'],
                    timeseries[point_id]
                )
                if analysis_mode == 'preprocessing':
                    gs, point_statistics = calculate_gs(point_statistics)
                    if not spatial_model:
                        phi, point_statistics = calculate_phi(point_statistics, override_phi=True)
                    else:
                        phi, point_statistics = calculate_phi(point_statistics, override_phi=False)
                if amax_durations is not None:
                    if (start_date is not None) and (n_subsets > 1):
                        actual_start_year = start_date.year + (subset_id - 1) * subset_length
                        actual_end_year = actual_start_year + subset_length - 1
                        years = range(actual_start_year, actual_end_year+1)
                    else:
                        years = None
                    maxima = extract_maxima(timeseries[point_id], amax_durations, years)

                # Store dataframes
                id_columns = {'realisation_id': realisation_id, 'subset_id': subset_id, 'point_id': point_id}
                if (n_subsets > 1) and (subset_id == n_subsets):
                    if last_subset_complete:
                        dfs['point_statistics'] = pd.concat([
                            dfs['point_statistics'], utils.add_columns(point_statistics, id_columns)]
                        )
                else:
                    if (realisation_id == 1) and (subset_id == 1) and (point_id == min(point_ids)):
                        dfs['point_statistics'] = utils.add_columns(point_statistics, id_columns)
                    else:
                        dfs['point_statistics'] = pd.concat([
                            dfs['point_statistics'], utils.add_columns(point_statistics, id_columns)]
                        )
                if analysis_mode == 'preprocessing':
                    if (realisation_id == 1) and (subset_id == 1) and (point_id == min(point_ids)):
                        dfs['phi'] = utils.add_columns(phi, id_columns)
                        dfs['gs'] = utils.add_columns(gs, id_columns)
                    else:
                        dfs['phi'] = pd.concat([dfs['phi'], utils.add_columns(phi, id_columns)])
                        dfs['gs'] = pd.concat([dfs['gs'], utils.add_columns(gs, id_columns)])
                if amax_durations is not None:
                    if (realisation_id == 1) and (subset_id == 1) and (point_id == min(point_ids)):
                        dfs['maxima'] = utils.add_columns(maxima, id_columns)
                    else:
                        dfs['maxima'] = pd.concat([dfs['maxima'], utils.add_columns(maxima, id_columns)])

            # Calculate cross-correlations
            if 'cross-correlation' in statistic_definitions['name'].values:
                # print('    - Cross-correlations')
                unique_seasons = utils.identify_unique_seasons(season_definitions)
                cross_correlations = calculate_cross_correlations(
                    point_metadata, statistic_definitions.loc[statistic_definitions['name'] == 'cross-correlation'],
                    unique_seasons, timeseries
                )

                # Merge gs and phi into cross-correlations dataframe
                if analysis_mode == 'preprocessing':
                    phi = dfs['phi'].loc[
                        (dfs['phi']['realisation_id'] == realisation_id) & (dfs['phi']['subset_id'] == subset_id)
                    ]
                    cross_correlations['gs'] = 1.0
                    cross_correlations = pd.merge(cross_correlations, phi, how='left', on=['season', 'point_id'])
                    phi2 = phi.copy()
                    phi2.rename({'phi': 'phi2', 'point_id': 'point_id2'}, axis=1, inplace=True)
                    cross_correlations = pd.merge(cross_correlations, phi2, how='left', on=['season', 'point_id2'])

                # Store dataframe
                id_columns = {'realisation_id': realisation_id, 'subset_id': subset_id}
                if (realisation_id == 1) and (subset_id == 1):
                    dfs['cross-correlation'] = utils.add_columns(cross_correlations, id_columns)
                else:
                    dfs['cross-correlation'] = pd.concat([
                        dfs['cross-correlation'], utils.add_columns(cross_correlations, id_columns)
                    ])

        # Close file readers after realisation is processed
        for point_id in point_ids:
            readers[point_id].close()

    # Summarise statistics before merging
    if analysis_mode == 'postprocessing':

        # Point statistics
        grouping_columns = ['statistic_id', 'point_id', 'season']
        df1 = dfs['point_statistics'].groupby(grouping_columns)['value'].agg([
            'mean', utils.percentile(0.05), utils.percentile(0.25), utils.percentile(0.75), utils.percentile(0.95)
        ])
        df1 = df1.reset_index()
        df1 = pd.merge(statistic_definitions, df1)
        dfs['point_statistics'] = df1

        # Cross-correlations
        if 'cross-correlation' in statistic_definitions['name'].values:
            grouping_columns = ['statistic_id', 'point_id', 'point_id2', 'distance', 'season']
            df2 = dfs['cross-correlation'].groupby(grouping_columns)['value'].agg([
                'mean', utils.percentile(0.05), utils.percentile(0.25), utils.percentile(0.75), utils.percentile(0.95)
            ])
            df2 = df2.reset_index()
            df2 = pd.merge(statistic_definitions, df2)
            dfs['cross-correlation'] = df2

    # Merge point statistics and cross-correlations
    if 'cross-correlation' in statistic_definitions['name'].values:
        if analysis_mode == 'preprocessing':
            value_columns = ['value']
        elif analysis_mode == 'postprocessing':
            value_columns = ['mean', 'percentile_5', 'percentile_25', 'percentile_75', 'percentile_95']
        dfs['statistics'] = utils.merge_statistics(dfs['point_statistics'], dfs['cross-correlation'], value_columns)
    else:
        dfs['statistics'] = dfs['point_statistics']

    # Finalise dataframes
    if analysis_mode == 'preprocessing':
        dfs['statistics'].drop(columns=['realisation_id', 'subset_id'], inplace=True)
    if spatial_model and (analysis_mode == 'preprocessing'):
        # TODO: Check phi comes through properly in preprocessing still
        phi = dfs['phi'].loc[(dfs['phi']['realisation_id'] == 1) & (dfs['phi']['subset_id'] == 1)]
        phi = pd.merge(point_metadata, phi, how='outer', on=['point_id'])
    else:
        phi = None
    if ddf_return_periods is not None:
        ddf = calculate_ddf_statistics(dfs['maxima'], amax_durations, ddf_return_periods)

    # Write output
    if write_output:

        # Statistics and phi
        if analysis_mode == 'preprocessing':
            write_weights = True
            write_gs = True
            write_phi = True
            value_columns = 'value'
        elif analysis_mode == 'postprocessing':
            write_weights = False
            write_gs = False
            write_phi = False
            value_columns = ['mean', 'percentile_5', 'percentile_25', 'percentile_75', 'percentile_95']
        utils.write_statistics(
            dfs['statistics'], output_statistics_path, season_definitions, write_weights, write_gs, write_phi,
            value_columns
        )
        # if spatial_model and (analysis_mode == 'preprocessing'):
        #     utils.write_phi(phi, output_phi_path)

        # Maxima
        if output_amax_path is not None:
            utils.write_maxima(dfs['maxima'], output_amax_path, analysis_mode)

        # DDF statistics
        if ddf_return_periods is not None:
            utils.write_ddf(ddf, output_ddf_path)

    return dfs['statistics'], phi


def prepare_point_timeseries(
        df, calculation_period, season_definitions, completeness_threshold, durations, outlier_method,
        maximum_relative_difference, maximum_alterations,
):
    """
    Prepare point timeseries for analysis.

    Steps are: (1) subset on reference calculation period, (2) define seasons for grouping, (3) applying any trimming
    or clipping to reduce the influence of outliers, and (4) aggregating timeseries to required durations.

    """
    # Check valid or nan  # TODO: Revisit if this function gets used for non-precipitation variables
    df.loc[df['value'] < 0.0] = np.nan

    # Subset required calculation period
    if calculation_period is not None:
        start_year = calculation_period[0]
        end_year = calculation_period[1]
        df = df.loc[(df.index.year >= start_year) & (df.index.year <= end_year)]

    # Apply season definitions and make a running UID for season that goes up by one at each change in season
    # through the time series. Season definitions are needed to identify season completeness but also to apply
    # trimming or clipping
    df['season'] = df.index.month.map(season_definitions)
    df['season_uid'] = df['season'].ne(df['season'].shift()).cumsum()

    # Mask periods not meeting data completeness threshold (close approximation). There is an assumption of at
    # least one complete version of each season in dataframe
    if df['value'].isnull().any():
        df['season_count'] = df.groupby('season_uid')['value'].transform('count')
        df['season_size'] = df.groupby('season_uid')['value'].transform('size')
        df['season_size'] = df.groupby('season')['season_size'].transform('median')
        df['completeness'] = df['season_count'] / df['season_size'] * 100.0
        df['completeness'] = np.where(df['completeness'] > 100.0, 100.0, df['completeness'])
        df.loc[df['completeness'] < completeness_threshold, 'value'] = np.nan
        df = df.loc[:, ['season', 'value']]

    # Apply trimming or clipping season-wise
    if outlier_method == 'trim':
        df['value'] = df.groupby('season')['value'].transform(
            utils.trim_array(maximum_relative_difference, maximum_alterations)
        )
    elif outlier_method == 'clip':
        df['value'] = df.groupby('season')['value'].transform(
            utils.clip_array(maximum_relative_difference, maximum_alterations)
        )

    # Find timestep and convert from datetime to period index if needed
    if not isinstance(df.index, pd.PeriodIndex):
        datetime_difference = df.index[1] - df.index[0]
    else:
        datetime_difference = df.index[1].to_timestamp() - df.index[0].to_timestamp()
    timestep_length = int(datetime_difference.days * 24) + int(datetime_difference.seconds / 3600)  # hours
    period = str(timestep_length) + 'H'  # TODO: Sort out sub-hourly timestep
    if not isinstance(df.index, pd.PeriodIndex):
        df = df.to_period(period)

    # TODO: More efficient approach would be to use successive durations in aggregation
    # - e.g use 1hr to get 3hr, but then use 3hr to get 6hr, 6hr to get 12hr, etc
    # - only works if there is a neat division, otherwise need to go back to e.g. 1hr

    # Aggregate timeseries to required durations
    dfs = {}
    for duration in sorted(durations):
        resample_code = str(int(duration)) + 'H'  # TODO: Check/add sub-hourly
        expected_count = int(duration / timestep_length)
        df1 = df['value'].resample(resample_code, closed='left', label='left').sum()
        df2 = df['value'].resample(resample_code, closed='left', label='left').count()
        df1.values[df2.values < expected_count] = np.nan  # duration
        df1 = df1.to_frame()
        df1['season'] = df1.index.month.map(season_definitions)
        dfs[duration] = df1
        dfs[duration] = dfs[duration][dfs[duration]['value'].notnull()]

    return dfs


def calculate_point_statistics(statistic_definitions, dfs):
    """
    Calculate mean, variance, skewness, lag-n autocorrelation and dry probability by season and duration.

    """
    statistic_functions = {'mean': 'mean', 'variance': np.var, 'skewness': 'skew'}
    statistics = []
    for index, row in statistic_definitions.iterrows():
        statistic_name = row['name']
        duration = row['duration']
        if statistic_name in ['mean', 'variance', 'skewness']:
            values = dfs[duration].groupby('season')['value'].agg(statistic_functions[statistic_name])
        elif statistic_name == 'probability_dry':
            threshold = row['threshold']
            values = dfs[duration].groupby('season')['value'].agg(probability_dry(threshold))
        elif statistic_name == 'autocorrelation':
            lag = row['lag']
            values = dfs[duration].groupby('season')['value'].agg(autocorrelation(lag))

        values = values.to_frame('value')
        values.reset_index(inplace=True)
        for column in statistic_definitions.columns:
            if column not in values.columns:
                values[column] = row[column]
        statistics.append(values)

    statistics = pd.concat(statistics)
    ordered_columns = list(statistic_definitions.columns)
    ordered_columns.extend(['season', 'value'])
    statistics = statistics[ordered_columns]

    return statistics


def probability_dry(threshold=0.0):
    def _probability_dry(x):
        return x[x < threshold].shape[0] / x.shape[0]
    return _probability_dry


def autocorrelation(lag=1):
    def _autocorrelation(x):
        r, p = scipy.stats.pearsonr(x[lag:], x.shift(lag)[lag:])
        return r
    return _autocorrelation


def calculate_cross_correlations(metadata, statistic_definitions, unique_seasons, dfs):
    """
    Calculate cross-correlations between all pairs of points by season and duration.

    """
    # Create a table of all unique pairs of points and their separation distances
    pairs = list(itertools.combinations(list(metadata['point_id']), 2))
    id1s = []
    id2s = []
    distances = []
    for id1, id2 in pairs:
        id1_x = metadata.loc[metadata['point_id'] == id1, 'easting'].values[0]
        id1_y = metadata.loc[metadata['point_id'] == id1, 'northing'].values[0]
        id2_x = metadata.loc[metadata['point_id'] == id2, 'easting'].values[0]
        id2_y = metadata.loc[metadata['point_id'] == id2, 'northing'].values[0]
        distance = ((id1_x - id2_x) ** 2 + (id1_y - id2_y) ** 2) ** 0.5
        id1s.append(id1)
        id2s.append(id2)
        distances.append(distance / 1000.0)  # m to km
    pair_metadata = pd.DataFrame({'point_id': id1s, 'point_id2': id2s, 'distance': distances})

    # Calculations are currently done in a loop, so initialise dictionary of lists to store results
    dc = {
        'statistic_id': [], 'lag': [], 'point_id': [], 'point_id2': [], 'distance': [],
        'duration': [], 'season': [], 'value': [], 'weight': []
    }

    # Loop of statistics is really a loop of duration/lag combinations of cross-correlation
    for _, statistic_details in statistic_definitions.iterrows():

        statistic_id = statistic_details['statistic_id']
        duration = statistic_details['duration']
        lag = statistic_details['lag']

        for season in unique_seasons:
            for _, pair_details in pair_metadata.iterrows():

                # Subset on point pair and season
                id1 = pair_details['point_id']
                id2 = pair_details['point_id2']
                df1 = dfs[id1][duration]
                df2 = dfs[id2][duration]
                x = df1.loc[df1['season'] == season]
                y = df2.loc[df2['season'] == season]

                # Perform correlation and append results/metadata
                df3 = pd.merge(x, y, left_index=True, right_index=True)
                r, p = scipy.stats.pearsonr(df3['value_x'][lag:], df3['value_y'].shift(lag)[lag:])
                dc['statistic_id'].append(int(statistic_id))
                dc['lag'].append(int(lag))
                dc['point_id'].append(int(id1))
                dc['point_id2'].append(int(id2))
                dc['distance'].append(pair_details['distance'])
                dc['duration'].append(duration)
                dc['season'].append(int(season))
                dc['value'].append(r)
                dc['weight'].append(statistic_details['weight'])

    cross_correlations = pd.DataFrame(dc)
    cross_correlations['name'] = 'cross-correlation'

    return cross_correlations


def calculate_gs(statistics, merge=True):
    """
    Calculate gs scale factor as annual mean of statistic for a given site.

    """
    gs = statistics.groupby(['statistic_id', 'name'])['value'].mean()
    gs = gs.to_frame('gs')
    gs.reset_index(inplace=True)
    gs.loc[gs['name'] == 'probability_dry', 'gs'] = 1.0
    gs.loc[gs['name'] == 'autocorrelation', 'gs'] = 1.0
    gs = gs[['statistic_id', 'gs']]
    if merge:
        statistics = pd.merge(statistics, gs, how='left', on='statistic_id')
        return gs, statistics
    else:
        return gs


def calculate_phi(statistics, override_phi, merge=True):
    """
    Calculate phi scale factor for a given site/season as 24hr mean divided by 3.0 mm/day (following RainSim V3.1).

    """
    phi = statistics.loc[(statistics['name'] == 'mean') & (statistics['duration'] == 24), ['season', 'value']].copy()
    phi['value'] /= 3.0
    phi.rename({'value': 'phi'}, axis=1, inplace=True)
    if override_phi:
        phi['phi'] = 1.0
    if merge:
        statistics = pd.merge(statistics, phi, how='left', on='season')
        return phi, statistics
    else:
        return phi


def extract_maxima(dfs, durations, years=None):
    durations = sorted(list(durations))
    for duration in durations:
        df = dfs[duration].groupby(dfs[duration].index.year)['value'].max()
        df = df.to_frame('value')
        df['duration'] = duration
        if years is not None:
            df.index = years
        if durations.index(duration) == 0:
            maxima = df.copy()
        else:
            maxima = pd.concat([maxima, df])
    return maxima


def calculate_ddf_statistics(maxima, durations, return_periods):

    # This function could be modified to permit seasonal maxima/DDF calculations
    durations = list(durations)
    point_ids = list(np.unique(maxima['point_id'].values))
    for point_id, duration in itertools.product(point_ids, durations):
        df = maxima.loc[(maxima['duration'] == duration) & (maxima['point_id'] == point_id)]

        if isinstance(return_periods, list):
            return_periods = np.asarray(return_periods)

        # Depth estimates from GEV
        shape, location, scale = scipy.stats.genextreme.fit(df['value'])
        exceedance_probability = 1.0 / return_periods
        cumulative_probability = 1.0 - exceedance_probability
        gev_depths = scipy.stats.genextreme.ppf(cumulative_probability, c=shape, loc=location, scale=scale)

        # Depth estimates from empirical approach (Weibull plotting position)
        ranks = scipy.stats.rankdata(df['value'])
        pp = ranks / (ranks.shape[0] + 1)
        f = scipy.interpolate.interp1d(pp, df['value'])
        empirical_depths = f(cumulative_probability[cumulative_probability <= np.max(pp)])
        n_unestimated = cumulative_probability[cumulative_probability > np.max(pp)].shape[0]
        empirical_depths = np.concatenate([empirical_depths, np.zeros(n_unestimated) - 999])
        empirical_depths[empirical_depths == -999] = np.nan

        # Compile into output dataframes
        ddf0 = pd.DataFrame({
            'return_period': return_periods, 'depth_gev': gev_depths, 'depth_empirical': empirical_depths,
        })
        ddf0['point_id'] = point_id
        ddf0['duration'] = duration
        if (point_ids.index(point_id) == 0) and (durations.index(duration) == 0):
            ddf = ddf0.copy()
        else:
            ddf = pd.concat([ddf, ddf0])

    return ddf
