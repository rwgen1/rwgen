import itertools

import numpy as np
import pandas as pd
import scipy.stats

from . import utils


def prepare_point_timeseries(
        df, timeseries_format, timeseries_path, calculation_period, season_definitions, completeness_threshold,
        durations, outlier_method, maximum_relative_difference, maximum_alterations
):
    """
    Prepare point timeseries for analysis.

    Steps are: (1) subset on reference calculation period, (2) define seasons for grouping, (3) applying any trimming
    or clipping to reduce the influence of outliers, and (4) aggregating timeseries to required durations.

    """
    # Read from file if required
    if df is None:
        if timeseries_format == 'csv':
            df = utils.read_csv_timeseries(timeseries_path)
        elif timeseries_format == 'csvy':
            df = utils.read_csvy_timeseries(timeseries_path)
    df.loc[df['value'] < 0.0] = np.nan

    # Subset required calculation period
    if calculation_period is not None:
        start_year = calculation_period[0]
        end_year = calculation_period[1]
        df = df.loc[(df.index.year >= start_year) & (df.index.year <= end_year)]

    # Apply season definitions and make a running UID for season that goes up by one at each
    # change in season through the time series (needed to identify season completeness)
    df['season'] = df.index.month.map(season_definitions)
    df['season_uid'] = df['season'].ne(df['season'].shift()).cumsum()

    # Mask periods not meeting data completeness threshold (close approximation). There is an
    # assumption of at least one complete version of each season in dataframe
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

    # Convert from datetime to period index
    datetime_difference = df.index[1] - df.index[0]
    timestep_in_minutes = int(datetime_difference.seconds / 60)
    if timestep_in_minutes % 60 == 0:
        timestep_in_hours = int(timestep_in_minutes / 60)
        period = str(timestep_in_hours) + 'H'
    else:
        period = str(timestep_in_minutes) + 'T'
    df = df.to_period(period)

    # Aggregate timeseries to required durations
    dfs = {}
    for duration in durations:
        if duration % 1 == 0:
            resample_code = str(int(duration)) + 'H'
        else:
            resample_code = str(int(round(duration * 60))) + 'T'
        df1 = df['value'].resample(resample_code, closed='left', label='left').sum()
        df2 = df['value'].resample(resample_code, closed='left', label='left').count()
        df1.values[df2.values < duration] = np.nan
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


def probability_dry(threshold=0.0):
    def _probability_dry(x):
        return x[x < threshold].shape[0] / x.shape[0]
    return _probability_dry


def autocorrelation(lag=1):
    def _autocorrelation(x):
        r, p = scipy.stats.pearsonr(x[lag:], x.shift(lag)[lag:])
        return r
    return _autocorrelation
