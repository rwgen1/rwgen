import os
import sys  # TEMPORARY
import datetime
import itertools

import numpy as np
import pandas as pd
import scipy.stats
import scipy.interpolate
import numba

from . import utils

# TODO: Handle case where statistics/AMAX are requested for unavailable durations (e.g. 1hr AMAX from daily data)


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
        amax_window_type,
        output_ddf_path,  # boolean and/or list ['annual', 'seasonal']; output_ddf_path ?
        ddf_return_periods,  # None or dict(annual=[], seasonal=[])
        write_output,
        simulation_name,
        # n_workers,  # currently only amax extraction has parallel option
        # !221123
        use_pooling,  # currently only for skewness
        calculate_statistics,
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
    # if amax_durations is not None:
    #     durations.extend(list(amax_durations))
    # durations = sorted(list(set(durations)))

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
            if analysis_mode == 'preprocessing':
                input_paths = [timeseries_path]  # + file_extension
            elif analysis_mode == 'postprocessing':
                file_name = simulation_name + file_extension
                input_paths = [os.path.join(timeseries_folder, file_name)]
        else:
            point_ids = point_metadata['point_id'].values.tolist()
            input_paths = []
            for _, row in point_metadata.iterrows():
                if ('file_name' in point_metadata.columns) and (analysis_mode == 'preprocessing'):
                    file_name = row['file_name']
                else:
                    file_name = row['name'] + file_extension
                input_path = os.path.join(timeseries_folder, file_name)
                input_paths.append(input_path)

        # Check for realisation identifier in file names
        if analysis_mode == 'postprocessing':
            # file_suffix = '_r' + str(realisation_id) + '_prcp' + file_extension  # TODO: Remove hardcoded variable name
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

            # Read all points
            for point_id in point_ids:

                df = readers[point_id].get_chunk(chunk_size)
                if timeseries_format == 'txt':
                    df.index = date_index
                    df.columns = ['value']

                # Prepare time series for analysis - join seasons, remove outliers, aggregate
                timeseries[point_id] = prepare_point_timeseries(
                    df=df,
                    season_definitions=season_definitions,
                    completeness_threshold=completeness_threshold,
                    durations=durations,
                    outlier_method=outlier_method,
                    maximum_relative_difference=maximum_relative_difference,
                    maximum_alterations=maximum_alterations,
                )

            # /-/
            # !221121 - Second point loop so that all series read before calculations (for pooling)

            # Calculate statistics by point (and using pooled time series if applicable)
            _point_ids = point_ids.copy()
            if spatial_model and use_pooling:
                _point_ids.append(-1)
            for point_id in _point_ids:

                # @@@
                if calculate_statistics:

                    # Prepare dataframe for calculating pooled statistics if requested
                    if (point_id == -1) and spatial_model and (analysis_mode == 'preprocessing'):
                        dfs_pooled = {}
                        for duration in statistic_definitions['duration'].unique():
                            tmp = []
                            for pid in timeseries.keys():
                                df = timeseries[pid][duration].copy()
                                # df['mean'] = df.groupby('season')['value'].transform('mean')
                                # df['value'] /= df['mean']
                                # df.drop(columns='mean', inplace=True)
                                df.index.rename('datetime', inplace=True)
                                df.reset_index(inplace=True)
                                df = df.merge(
                                    dfs['phi'].loc[dfs['phi']['point_id'] == pid, ['season', 'phi']], how='left',
                                    left_on='season', right_on='season',
                                )
                                df['scaled_value'] = df['value'] / df['phi']
                                df.drop(columns='phi', inplace=True)
                                df['point_id'] = pid
                                df.set_index('datetime', inplace=True)
                                tmp.append(df)
                            df = pd.concat(tmp)
                            dfs_pooled[duration] = df.copy()
                        point_statistics = calculate_point_statistics(
                            statistic_definitions.loc[statistic_definitions['name'] != 'cross-correlation'],
                            dfs_pooled, calculation_period, pooled_series=True,
                        )

                    # Calculate statistics for each point separately and store dataframes
                    else:
                        point_statistics = calculate_point_statistics(
                            statistic_definitions.loc[statistic_definitions['name'] != 'cross-correlation'],
                            timeseries[point_id], calculation_period
                        )

                    # Phi and gs (scaling term in fitting)
                    if analysis_mode == 'preprocessing':
                        gs, point_statistics = calculate_gs(point_statistics)
                        if not spatial_model:
                            phi, point_statistics = calculate_phi(point_statistics, override_phi=True)
                        else:
                            if point_id == -1:
                                phi, point_statistics = calculate_phi(point_statistics, override_phi=True)
                            else:
                                phi, point_statistics = calculate_phi(point_statistics, override_phi=False)
                # @@@

                # AMAX extraction
                if (amax_durations is not None) and (point_id != -1):
                    if (start_date is not None) and (n_subsets > 1):
                        actual_start_year = start_date.year + (subset_id - 1) * subset_length
                        actual_end_year = actual_start_year + subset_length - 1
                        years = range(actual_start_year, actual_end_year+1)
                    else:
                        years = None
                    maxima = extract_maxima(
                        timeseries[point_id], amax_durations, amax_window_type, analysis_mode, years=years
                    )

                # Store dataframes
                id_columns = {'realisation_id': realisation_id, 'subset_id': subset_id, 'point_id': point_id}
                # @@@
                if calculate_statistics:
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
                # @@@
                if (amax_durations is not None) and (point_id != -1):
                    if (realisation_id == 1) and (subset_id == 1) and (point_id == min(point_ids)):
                        dfs['maxima'] = utils.add_columns(maxima, id_columns)
                    else:
                        dfs['maxima'] = pd.concat([dfs['maxima'], utils.add_columns(maxima, id_columns)])

            # /-/

            # @@@
            if calculate_statistics:

                # Calculate cross-correlations
                if 'cross-correlation' in statistic_definitions['name'].values:
                    # print('    - Cross-correlations')
                    unique_seasons = utils.identify_unique_seasons(season_definitions)
                    cross_correlations = calculate_cross_correlations(
                        point_metadata, statistic_definitions.loc[statistic_definitions['name'] == 'cross-correlation'],
                        unique_seasons, timeseries
                    )

                    # Bin by distance to summarise cross-correlations - possible that more than one cross-correlation
                    # statistic specified (e.g. 1H and 24H)
                    cc_tmp = []
                    for statistic_id in np.unique(cross_correlations['statistic_id']):
                        cc1 = cross_correlations.loc[cross_correlations['statistic_id'] == statistic_id].copy()
                        nbins = min(cc1.loc[(cc1['season'] == 1)].shape[0], 20)
                        distance_bins = np.linspace(cc1['distance'].min() - 0.001, cc1['distance'].max() + 0.001, nbins)
                        bin_midpoints = (np.concatenate([np.array([0.0]), distance_bins[:-1]]) + distance_bins) / 2.0
                        cc1['distance_bin'] = np.digitize(cc1['distance'], distance_bins)
                        cc2 = cc1.groupby(['season', 'distance_bin'])['value'].mean()
                        cc2 = cc2.to_frame('value')
                        cc2.reset_index(inplace=True)
                        tmp = pd.DataFrame({
                            'distance_bin': np.arange(bin_midpoints.shape[0], dtype=int), 'distance': bin_midpoints
                        })
                        cc2 = cc2.merge(tmp)
                        cc2.sort_values(['distance_bin', 'season'], inplace=True)

                        # Fit exponential covariance model to reduce noise
                        # TODO: Minimum number of distances/bins for which to do this?
                        cc2 = get_fitted_correlations(cc2, unique_seasons)
                        cc2.sort_values(['distance_bin', 'season'], inplace=True)

                        # Format dataframe
                        cc2['statistic_id'] = statistic_id
                        cc2['lag'] = cc1['lag'].values[0]
                        cc2['point_id'] = -1
                        cc2['point_id2'] = -1
                        cc2['duration'] = cc1['duration'].values[0]
                        cc2['weight'] = cc1['weight'].values[0]
                        cc2['name'] = cc1['name'].values[0]
                        cc2.drop(columns='distance_bin', inplace=True)
                        cc_tmp.append(cc2)

                    # Concatenate pooled cross-correlations with main df
                    cc_pooled = pd.concat(cc_tmp)
                    cross_correlations = pd.concat([cross_correlations, cc_pooled])

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
            # @@@

        # Close file readers after realisation is processed
        for point_id in point_ids:
            readers[point_id].close()

    # Summarise statistics before merging
    if analysis_mode == 'postprocessing':

        # @@@
        if calculate_statistics:

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
        # @@@

    # Merge point statistics and cross-correlations
    # @@@
    if calculate_statistics:
        if 'cross-correlation' in statistic_definitions['name'].values:
            if analysis_mode == 'preprocessing':
                value_columns = ['value']
            elif analysis_mode == 'postprocessing':
                value_columns = ['mean', 'percentile_5', 'percentile_25', 'percentile_75', 'percentile_95']
            dfs['statistics'] = utils.merge_statistics(dfs['point_statistics'], dfs['cross-correlation'], value_columns)
        else:
            dfs['statistics'] = dfs['point_statistics']
    # @@@

    # Finalise dataframes
    if analysis_mode == 'preprocessing':
        dfs['statistics'].drop(columns=['realisation_id', 'subset_id'], inplace=True)
    if spatial_model and (analysis_mode == 'preprocessing'):
        # TODO: Check phi comes through properly in preprocessing still
        phi = dfs['phi'].loc[(dfs['phi']['realisation_id'] == 1) & (dfs['phi']['subset_id'] == 1)]
        phi = phi.loc[phi['point_id'] != -1]  # 061
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
        # @@@
        if calculate_statistics:
            utils.write_statistics(
                dfs['statistics'], output_statistics_path, season_definitions, write_weights, write_gs, write_phi,
                value_columns
            )
        # @@@
        # if spatial_model and (analysis_mode == 'preprocessing'):
        #     utils.write_phi(phi, output_phi_path)

        # Maxima
        if output_amax_path is not None:
            utils.write_maxima(dfs['maxima'], output_amax_path, analysis_mode)

        # DDF statistics
        if ddf_return_periods is not None:
            utils.write_ddf(ddf, output_ddf_path)

    if calculate_statistics:
        return dfs['statistics'], phi
    else:
        return None, phi


def prepare_point_timeseries(
        df, season_definitions, completeness_threshold, durations, outlier_method,
        maximum_relative_difference, maximum_alterations,
):
    """
    Prepare point timeseries for analysis.

    Steps are: (1) subset on reference calculation period, (2) define seasons for grouping, (3) applying any trimming
    or clipping to reduce the influence of outliers, and (4) aggregating timeseries to required durations.

    """
    # Check valid or nan  # TODO: Revisit if this function gets used for non-precipitation variables
    df.loc[df['value'] < 0.0] = np.nan

    # Apply season definitions and make a running UID for season that goes up by one at each change in season
    # through the time series. Season definitions are needed to identify season completeness but also to apply
    # trimming or clipping
    df['season'] = df.index.month.map(season_definitions)
    df['season_uid'] = df['season'].ne(df['season'].shift()).cumsum()

    # Mask periods not meeting data completeness threshold (close approximation). There is an assumption of at
    # least one complete version of each season in dataframe (where complete means that nans may be present - i.e.
    # fine unless only a very short (< 1 year) record is passed in)
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

    # Prepare order to process durations in, so that long durations can be calculated from daily rather than hourly
    # durations (as faster)
    duration_hours = []
    for duration in durations:
        duration_units = duration[-1]
        if duration_units == 'H':
            duration_hours.append(int(duration[:-1]))
        elif duration_units == 'D':
            duration_hours.append(int(duration[:-1]) * 24)
        elif duration_units == 'M':
            duration_hours.append(31 * 24)
    duration_hours = np.asarray(duration_hours)
    sorted_durations = np.asarray(durations)[np.argsort(duration_hours)]

    # Aggregate timeseries to required durations
    dfs = {}
    for duration in sorted_durations:
        # resample_code = str(int(duration)) + 'H'  # TODO: Check/add sub-hourly
        resample_code = duration
        duration_units = duration[-1]
        if duration_units == 'H':
            duration_hours = int(duration[:-1])
        elif duration_units == 'D':
            duration_hours = int(duration[:-1]) * 24
        elif duration_units == 'M':
            duration_hours = 31 * 24

        # Final day needed for a given aggregation
        # - relies on multiples of one day if duration exceeds 24 hours
        # - constrained to monthly
        # - maximum duration of 28 days(?)
        if duration_hours > 24:
            duration_days = int(duration_hours / 24)

            # Interim aggregation to daily to see if it speeds things up
            if '24H' in durations:
                df1 = dfs['24H'].copy()
            elif '1D' in durations:
                df1 = dfs['1D'].copy()

            n_groups = int(np.ceil(31 / duration_days))
            df1['group'] = -1
            for group in range(n_groups):
                if duration_units != 'M':
                    df1['group'] = np.where(df1.index.day >= group * duration_days + 1, group, df1['group'])
                else:
                    # df1['month'] = df1.index.month
                    # df1['group'] = df1['month'].ne(df1['month'].shift()).cumsum()
                    # df1.drop(columns=['month'], inplace=True)
                    df1['group'] = 0

            # df1 = df.groupby([df.index.year, df.index.month, 'group'])['value'].agg(['sum', 'count'])
            df1 = df1.groupby([df1.index.year, df1.index.month, 'group'])['value'].agg(['sum', 'count'])
            if df1.index.names[0] == 'datetime':  # !221025 - for dfs coming from shuffling (fitting delta)
                df1.index.rename(['level_0', 'level_1', 'group'], inplace=True)
            df1.reset_index(inplace=True)
            df1['day'] = df1['group'] * duration_days + 1
            df1.rename(columns={'level_0': 'year', 'level_1': 'month'}, inplace=True)
            df1['datetime'] = pd.to_datetime(df1[['year', 'month', 'day']])
            df1.drop(columns=['year', 'month', 'day', 'group'], inplace=True)
            df1.set_index('datetime', inplace=True)
            # print(df1)
        else:
            df1 = df['value'].resample(resample_code, closed='left', label='left').agg(['sum', 'count'])
            # df2 = df['value'].resample(resample_code, closed='left', label='left').count()

        # df1 = df['value'].resample(resample_code, closed='left', label='left').sum()
        # df2 = df['value'].resample(resample_code, closed='left', label='left').count()

        # Remove data below a duration-dependent completeness
        if duration_hours <= 24:  # TODO: Remove hardcoding of timestep requiring complete data and completeness threshold?
            expected_count = int(duration_hours / timestep_length)
        else:
            expected_count = ((duration_hours / timestep_length) / 24) * 0.9  # TODO: Remove hardcoding - user option
        # df1.values[df2.values < expected_count] = np.nan  # duration
        df1.rename(columns={'sum': 'value'}, inplace=True)
        df1.loc[df1['count'] < expected_count, 'value'] = np.nan
        df1.drop(columns=['count'], inplace=True)

        # df1 = df1.to_frame()
        # df1.reset_index(inplace=True)
        # df1.rename(columns={'level_2': 'datetime'}, inplace=True)
        # df1.set_index('datetime', inplace=True)
        df1.sort_index(inplace=True)
        # df1.drop(columns=['level_0'], inplace=True)

        df1['season'] = df1.index.month.map(season_definitions)

        dfs[duration] = df1
        dfs[duration] = dfs[duration][dfs[duration]['value'].notnull()]

    # print(dfs[672])
    # print(dfs[672].columns)
    # dfs[672].to_csv('H:/Projects/rwgen/working/iss13/df672_1.csv')
    # import sys
    # sys.exit()

    return dfs


def calculate_point_statistics(statistic_definitions, dfs, calculation_period, pooled_series=False):
    """
    Calculate mean, variance, skewness, lag-n autocorrelation and dry probability by season and duration.

    """
    # dfs is a dictionary of time series dataframes with duration as key

    # Functions for statistics other than mean, variance and skewness are custom
    statistic_functions = {'mean': 'mean', 'variance': np.var, 'skewness': 'skew'}

    statistics = []
    for index, row in statistic_definitions.iterrows():
        statistic_name = row['name']
        duration = row['duration']

        # For pooled calculations, some calculations need the unscaled values (dry probability)
        if not pooled_series:
            value_column = 'value'
        else:
            if statistic_name == 'probability_dry':
                value_column = 'value'
            else:
                value_column = 'scaled_value'

        # Subset on requested calculation period if specified
        if calculation_period is not None:
            start_year = calculation_period[0]
            end_year = calculation_period[1]
            df = dfs[duration].loc[
                (dfs[duration].index.year >= start_year) & (dfs[duration].index.year <= end_year)
            ]
        else:
            df = dfs[duration]

        # Actual calculations
        if statistic_name in ['mean', 'variance', 'skewness']:
            values = df.groupby('season')[value_column].agg(statistic_functions[statistic_name])
        elif statistic_name == 'probability_dry':
            threshold = row['threshold']
            values = df.groupby('season')[value_column].agg(probability_dry(threshold))
        elif statistic_name == 'autocorrelation':
            lag = row['lag']
            df1 = df.copy()
            for _lag in range(1, lag+1):
                df1['omit_flag'] = np.where(df1['season'].shift(_lag) != df1['season'], 1, 0)
                if 'point_id' in df.columns:
                    df1['omit_flag'] = np.where(df1['point_id'].shift(_lag) != df1['point_id'], 1, df1['omit_flag'])
                df1['tmp_value'] = np.where(df1['omit_flag'] == 0, df1[value_column], np.nan)
            values = df1.groupby('season')['tmp_value'].agg(autocorrelation(lag))
            df1.drop(columns=['omit_flag', 'tmp_value'], inplace=True)

        # Append calculations to list for subsequent conversion to dataframe
        values = values.to_frame(value_column)
        values.rename(columns={value_column: 'value', 'tmp_value': 'value'}, inplace=True)
        values.reset_index(inplace=True)
        for column in statistic_definitions.columns:
            if column not in values.columns:
                values[column] = row[column]
        statistics.append(values)

    # Formatting
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
        # r, p = scipy.stats.pearsonr(x[lag:], x.shift(lag)[lag:])
        df = pd.DataFrame({'x': x, 'x_lag': x.shift(lag)})
        df.dropna(inplace=True)
        r, p = scipy.stats.pearsonr(df['x'], df['x_lag'])
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
    if '24H' in statistics['duration'].values:
        phi = statistics.loc[
            (statistics['name'] == 'mean') & (statistics['duration'] == '24H'), ['season', 'value']
        ].copy()
    elif '1D' in statistics['duration'].values:
        phi = statistics.loc[
            (statistics['name'] == 'mean') & (statistics['duration'] == '1D'), ['season', 'value']
        ].copy()
    phi['value'] /= 3.0
    phi.rename({'value': 'phi'}, axis=1, inplace=True)
    if override_phi:
        phi['phi'] = 1.0
    if merge:
        statistics = pd.merge(statistics, phi, how='left', on='season')
        return phi, statistics
    else:
        return phi


def extract_maxima(dfs, durations, window_type, analysis_mode, years=None, completeness_threshold=90.0):
    # TODO: Consider making completeness_threshold a user option
    # - also should the outputs from here include NA where not available? - currently series is not serially complete

    duration_hours = []
    for duration in dfs.keys():
        duration_units = duration[-1]
        if duration_units == 'H':
            duration_hours.append(int(duration[:-1]))
        elif duration_units == 'D':
            duration_hours.append(int(duration[:-1]) * 24)
        elif duration_units == 'M':
            duration_hours.append(31 * 24)
    duration_hours = np.asarray(duration_hours)

    timestep = np.min(duration_hours)  # min(dfs.keys())
    sorted_durations = sorted(list(durations))

    if min(sorted_durations) > timestep:
        raise ValueError('AMAX durations must exceed time series accumulation interval.')

    # Use offsets to calculate maxima for each possible non-overlapping window if using a sliding window
    offsets = {}
    if window_type == 'fixed':
        for duration in sorted_durations:
            offsets[duration] = [0]
    elif window_type == 'sliding':
        for duration in sorted_durations:
            offsets[duration] = list(range(int(duration / timestep)))

    # Dataframe to be aggregated
    if str(timestep) + 'H' in dfs.keys():
        df0 = dfs[str(timestep) + 'H']
    elif str(int(timestep / 24)) + 'D' in dfs.keys():
        df0 = dfs[str(int(timestep / 24)) + 'D']

    # Use numba-ised function to get annual maxima for all durations - post-processing only for now
    if (analysis_mode == 'postprocessing') and (window_type == 'sliding'):
        _years = df0.index.year.values
        _values = df0['value'].values
        _durations = np.asarray(sorted_durations)
        _amax = _get_maxima(_years, _values, _durations)
        tmp = []
        for duration in sorted_durations:
            df = pd.DataFrame({'year': np.asarray(years), 'value': _amax[np.int32(duration)]})
            df['duration'] = str(duration) + 'H'  # duration
            df.set_index('year', inplace=True)
            tmp.append(df)
        maxima = pd.concat(tmp)

    # If preprocessing then use approach accounting for possible missing data (also use if fixed window maxima)
    else:
        for duration in sorted_durations:
            tmp = []
            for offset in offsets[duration]:
                df = _aggregate(duration, offset, timestep, df0, completeness_threshold, years, q=None)
                tmp.append(df)

            df1 = pd.concat(tmp, axis=1)
            df1.sort_index(inplace=True)
            df1.dropna(inplace=True)
            df1['max'] = np.max(df1.values, axis=1)
            df1 = df1[['max']]
            df1.rename(columns={'max': 'value'}, inplace=True)
            df1['duration'] = str(duration) + 'H'  # duration

            # print(df1)
            # sys.exit()

            if sorted_durations.index(duration) == 0:
                maxima = df1.copy()
            else:
                maxima = pd.concat([maxima, df1])

    return maxima


@numba.jit(nopython=True)
def _get_maxima(years, values, durations):  # sliding window maxima - assumes no missing/nan data
    n_values = values.shape[0]
    start_year = np.min(years)
    n_years = np.max(years) - np.min(years) + 1

    amax = numba.typed.Dict()
    for duration in durations:
        amax[duration] = np.zeros(n_years)

    for i in range(n_values):
        for d in durations:
            y = years[i] - start_year

            if d == 1:
                p = values[i]
            else:
                j = i + d
                j = min(j, n_values)
                p = np.sum(values[i:j])

            if p > amax[d][y]:
                amax[d][y] = p

    return amax


def _aggregate(duration, offset, timestep, df0, completeness_threshold, years, q=None):
    freq = str(duration) + 'H'  # TODO: Sort out sub-hourly implementation
    offset_in_hours = int(offset * timestep)
    # print(_offset)  # label='right', closed='right',
    df = df0['value'].resample(freq, offset=datetime.timedelta(hours=offset_in_hours)).sum()
    df = df.to_frame()
    df['n_valid'] = df0['value'].resample(freq, offset=datetime.timedelta(hours=offset_in_hours)).count()
    # df['n_total'] = dfs[timestep]['value'].resample(freq, offset=datetime.timedelta(hours=_offset)).size()

    expected_count = int(duration / timestep)
    df['value'] = np.where(df['n_valid'] < expected_count, np.nan, df['value'])

    # df = df.groupby(df.index.year)['value'].max()
    # df = df.to_frame('value')
    # df['duration'] = duration

    df = df.groupby(df.index.year).agg({'value': max, 'n_valid': sum})  # , 'n_total': sum
    df['leap_year'] = [utils.check_if_leap_year(year) for year in df.index]
    df['n_total'] = np.where(df['leap_year'], 366 * int((24 / timestep)), 365 * int((24 / timestep)))
    df['value'] = np.where(
        df['n_valid'] < ((completeness_threshold / 100.0) * df['n_total']),
        np.nan,
        df['value']
    )
    df.drop(columns=['leap_year', 'n_valid', 'n_total'], inplace=True)
    df = df.loc[df['value'].first_valid_index():df['value'].last_valid_index()]

    if years is not None:
        df.index = years  # TODO: Check how/why this is being set for post-processing
        # - it is to adjust year to simulation year from analysis year

    return df


def calculate_ddf_statistics(maxima, durations, return_periods):

    # This function could be modified to permit seasonal maxima/DDF calculations
    durations = list(durations)
    durations = [str(dur) + 'H' for dur in durations]  # TESTING
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


# --
# !221121

def calculate_point_statistics2(
        statistic_definitions, point_dfs, point_id, calculation_period, analysis_mode, use_pooling, dfs_pooled,
        spatial_model,
):
    """
    Calculate mean, variance, skewness, lag-n autocorrelation and dry probability by season and duration.

    This version is for testing pooling of skewness, dry probability and autocorrelation.

    """
    # point_dfs is a dictionary of dictionaries
    # point_dfs = {point_id: {duration: df}}
    # so dfs = point_dfs[point_id] --> df = dfs[duration]
    # dfs_pooled --> df = dfs_pooled[duration]

    statistic_functions = {'mean': 'mean', 'variance': np.var, 'skewness': 'skew'}
    statistics = []
    for index, row in statistic_definitions.iterrows():
        statistic_name = row['name']
        duration = row['duration']

        # if statistic_name in ['mean', 'variance']:  # 044
        if statistic_name in ['mean', 'variance', 'probability_dry', 'autocorrelation']:  # 045
            df = point_dfs[point_id][duration]
        else:
            if use_pooling and spatial_model and (analysis_mode == 'preprocessing'):
                df = dfs_pooled[duration]
            else:
                df = point_dfs[point_id][duration]

        if calculation_period is not None:
            start_year = calculation_period[0]
            end_year = calculation_period[1]
            df = df.loc[(df.index.year >= start_year) & (df.index.year <= end_year)]

        if statistic_name in ['mean', 'variance', 'skewness']:
            values = df.groupby('season')['value'].agg(statistic_functions[statistic_name])
        elif statistic_name == 'probability_dry':
            threshold = row['threshold']
            values = df.groupby('season')['value'].agg(probability_dry(threshold))
        elif statistic_name == 'autocorrelation':
            lag = row['lag']
            values = df.groupby('season')['value'].agg(autocorrelation(lag))

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


def exponential_model(distance, variance, length_scale, nugget=None):
    if nugget is None:
        _nugget = 1.0 - variance
    else:
        _nugget = nugget
    x = variance * np.exp(-distance / length_scale) + _nugget
    return x


def get_fitted_correlations(df, unique_seasons):  # df = cc2
    bounds = ([0.01, 0.0], [1.0, 100000000.0])
    tmp = []
    for season in unique_seasons:
        parameters, _ = scipy.optimize.curve_fit(
            exponential_model,
            df.loc[df['season'] == season, 'distance'].values,
            df.loc[df['season'] == season, 'value'].values,
            bounds=bounds
        )
        variance, length_scale = parameters

        _corrs = exponential_model(
            df.loc[df['season'] == season, 'distance'].values, variance, length_scale,
        )

        df1 = pd.DataFrame({
            'distance': df.loc[df['season'] == season, 'distance'].values,
            'value': _corrs
        })
        df1['season'] = season
        tmp.append(df1)

    df1 = pd.concat(tmp)
    df1 = df.drop(columns='value').merge(df1)

    return df1
