import os
import sys
import datetime
import itertools

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.spatial
import scipy.optimize
import scipy.interpolate
import scipy.stats
import gstools
import numba

from . import simulation
from . import utils
from . import analysis


# TODO: Remove storm duration stuff if decide not to use (currently not in use - 04/10/2022)


def get_monthly_series_old(timeseries_path, spatial_model, point_metadata, calculation_period, xmin, ymin, xmax, ymax):
    # Lists of points and file paths so point and spatial models can be called in same loop
    # For spatial model, timeseries_path refers to timeseries_folder
    if not spatial_model:
        point_ids = [1]
        input_paths = [timeseries_path]
    else:
        point_ids = point_metadata['point_id'].values.tolist()
        input_paths = []
        for _, row in point_metadata.iterrows():
            file_name = row['name'] + '.csv'
            input_path = os.path.join(timeseries_path, file_name)
            input_paths.append(input_path)

    # Read data and aggregate to monthly
    # TODO: Harmonise with analysis functions used in pre-/post-processing
    dfs = []
    for point_id, input_path in zip(point_ids, input_paths):
        df = pd.read_csv(input_path, index_col=0, parse_dates=True, infer_datetime_format=True, dayfirst=True)
        df.rename(columns={'Value': 'value'}, inplace=True)
        df.loc[df['value'] < 0.0] = np.nan

        # Subset on reference period
        if calculation_period is not None:
            df = df.loc[(df.index.year >= calculation_period[0]) & df.index.year <= calculation_period[1]]

        # Find timestep and convert from datetime to period index if needed
        if not isinstance(df.index, pd.PeriodIndex):
            datetime_difference = df.index[1] - df.index[0]
        else:
            datetime_difference = df.index[1].to_timestamp() - df.index[0].to_timestamp()
        timestep_length = int(datetime_difference.days * 24) + int(datetime_difference.seconds / 3600)  # hours
        period = str(timestep_length) + 'H'  # TODO: Sort out sub-hourly timestep
        if not isinstance(df.index, pd.PeriodIndex):
            df = df.to_period(period)

        # First aggregate to daily and scale so mean is 3 mm/d to match phi definition if spatial model
        expected_count = int(24 / timestep_length)
        df1 = df['value'].resample('D', closed='left', label='left').sum()
        df2 = df['value'].resample('D', closed='left', label='left').count()
        df1.values[df2.values < expected_count] = np.nan
        df = df1.to_frame()
        df['month_mean'] = df['value'].groupby(df.index.month).transform('mean')
        if spatial_model:
            df['value'] = df['value'] * (3.0 / df['month_mean'])
        df = df.loc[:, ['value']]

        # Then aggregate to monthly
        df['month'] = df.index.month
        df['month_uid'] = df['month'].ne(df['month'].shift()).cumsum()
        df['month_count'] = df.groupby(df['month_uid'])['value'].transform('count')
        df['month_size'] = df.groupby(df['month_uid'])['value'].transform('size')
        df['month_size'] = df.groupby(df.index.month)['month_size'].transform('median')
        df['completeness'] = df['month_count'] / df['month_size'] * 100.0
        df1 = df['value'].resample('M', closed='right', label='right').sum()
        df2 = df['completeness'].resample('M', closed='right', label='right').median()
        df1.values[df2.values < 90.0] = np.nan
        df1 = df1.to_frame()
        dfs.append(df1)

    # Spatial average if required using month-wise interpolator (i.e. climatological variogram model)
    if not spatial_model:
        df = dfs[0]

    else:
        # Monthly climatologies
        monthly_means = {month: [] for month in range(1, 12 + 1)}
        for point_id, df in zip(point_ids, dfs):
            df1 = df.groupby(df.index.month)['value'].mean()
            for month in range(1, 12 + 1):
                monthly_means[month].append(df1.values[month - 1])

        # Fit variogram models
        variogram_models = {}
        for month in range(1, 12 + 1):
            variogram_models[month] = fit_variogram(point_metadata[['easting', 'northing']], monthly_means[month])

        # Grid to use in interpolation
        x_grid = np.linspace(xmin, xmax + 0.1, 200)
        y_grid = np.linspace(ymin, ymax + 0.1, 200)

        # Interpolate and apply spatial averaging to get monthly series
        df = pd.concat(dfs, axis=1)
        df.columns = point_ids
        monthly_series = []
        for date_time, row in df.iterrows():
            month = date_time.month

            # Choose non-NA values to interpolate
            x = []
            y = []
            vals = []
            for point_id in point_ids:
                if np.isfinite(row[point_id]):
                    x.append(point_metadata.loc[point_metadata['point_id'] == point_id, 'easting'].values[0])
                    y.append(point_metadata.loc[point_metadata['point_id'] == point_id, 'northing'].values[0])
                    vals.append(row[point_id])

            # Interpolate and get spatial average
            interpolator = gstools.krige.Ordinary(
                variogram_models[month], (x, y), np.asarray(vals)
            )
            month_field = interpolator((x_grid, y_grid), mesh_type='unstructured', return_var=False)
            monthly_series.append(np.mean(month_field))
        df['value'] = monthly_series
        df = df.loc[:, ['value']]

        # print(df)
        # df.to_csv('H:/Projects/rwgen/working/iss13/stnsrp/df_4.csv')

    return df


def fit_variogram(df1, vals, distance_bins=7):
    # Calculate bin edges
    max_distance = np.max(scipy.spatial.distance.pdist(np.asarray(df1[['easting', 'northing']])))
    interval = max_distance / distance_bins
    bin_edges = np.arange(0.0, max_distance + 0.1, interval)
    bin_edges[-1] = max_distance + 0.1  # ensure that all points covered

    # Estimate empirical variogram
    bin_centres, gamma, counts = gstools.vario_estimate(
        (df1['easting'].values, df1['northing'].values), vals, bin_edges, return_counts=True
    )
    bin_centres = bin_centres[counts > 0]
    gamma = gamma[counts > 0]

    # Identify best fit from exponential and spherical covariance models
    exponential_model = gstools.Exponential(dim=2)
    _, _, exponential_r2 = exponential_model.fit_variogram(bin_centres, gamma, nugget=False, return_r2=True)
    spherical_model = gstools.Spherical(dim=2)
    _, _, spherical_r2 = spherical_model.fit_variogram(bin_centres, gamma, nugget=False, return_r2=True)
    if exponential_r2 > spherical_r2:
        covariance_model = exponential_model
    else:
        covariance_model = spherical_model

    return covariance_model


def get_years(months):
    years = np.zeros(months.shape[0], dtype=int)
    year = 1
    for idx in range(months.shape[0]):
        if idx == 0:
            years[idx] = year
        else:
            if months[idx] < months[idx - 1]:
                year += 1
            years[idx] = year
    return years


def fit_delta(
        spatial_model, parameters, intensity_distribution, point_metadata, n_workers, random_seed,
        reference_statistics, reference_duration, n_divisions, use_pooling,
):
    # print('  - Shuffling')

    # !221205
    # if point_metadata is None:
    #     _point_metadata = None
    # else:
    #     _point_metadata = point_metadata.loc[point_metadata['point_id'] == point_metadata['point_id'].min()]
    _point_metadata = point_metadata

    # Bounds and reference for delta
    bounds = [0.0, 1.0]  # TODO: Confirm whether this is the range to stick with - input argument?
    # !221116 (033) - trying CV as objective function
    # [[[
    ref_var = reference_statistics.loc[
        ((reference_statistics['name'] == 'variance') & (reference_statistics['duration'] == reference_duration))
        | ((reference_statistics['name'] == 'mean') & (reference_statistics['duration'] == '24H')),
        ['point_id', 'season', 'name', 'value', 'phi', 'gs']
    ].copy()

    # if spatial_model and use_pooling:
    #     ref_var = ref_var.loc[ref_var['point_id'] == -1]
    ref_var = ref_var.loc[ref_var['point_id'] != -1]

    # ]]]
    # {{{
    # ref_var = reference_statistics.loc[
    #     (reference_statistics['name'] == 'variance') & (reference_statistics['duration'] == reference_duration),
    #     ['point_id', 'season', 'name', 'value', 'phi', 'gs']
    # ].copy()
    # }}}

    ref_var.set_index('season', inplace=True)

    # Prepare to simulate  # TODO: Consider whether anything should be an argument and tidy up
    realisation_id = 1  # -999  !221025
    simulation_length = 30  # number of years to simulate/shuffle for estimating delta
    start_year = 2000
    end_year = start_year + simulation_length - 1
    timestep_length = 24
    calendar = 'gregorian'
    datetime_helper = utils.make_datetime_helper(start_year, end_year, timestep_length, calendar)
    season_definitions = {month: month for month in range(1, 12 + 1)}
    output_types = ['point']  # None  # !221025
    discretisation_metadata = None
    catchment_metadata = None
    output_paths = None
    block_size = 30
    block_subset_size = 30
    spatial_raincell_method = 'buffer'
    spatial_buffer_factor = 15
    simulation_mode = 'shuffling_preparation'

    n_realisations = 30
    n_shuffles = 30

    if spatial_model:
        grid_metadata = None
        cell_size = None
        xmin, ymin, xmax, ymax = simulation.identify_domain_bounds(grid_metadata, cell_size, _point_metadata)
    else:
        xmin = None
        ymin = None
        xmax = None
        ymax = None

    if random_seed is None:
        seed_sequence = np.random.SeedSequence()
    else:
        seed_sequence = np.random.SeedSequence(random_seed)
    rng = np.random.default_rng(seed_sequence)

    # Fit delta - find for each realisation and then average
    deltas = np.zeros((n_realisations, 12))

    # Using multiprocessing
    if n_workers > 1:

        import multiprocessing as mp

        manager = mp.Manager()
        q = manager.Queue()
        pool = mp.Pool(n_workers)

        jobs = []

        for i in range(n_realisations):

            # TODO: Make below consistent with serial processing

            # Simulate realisation to be shuffled and fit delta
            df_wd, dc1, _ = simulation.simulate_realisation(
                realisation_id, datetime_helper, simulation_length, timestep_length, season_definitions,
                spatial_model, output_types, discretisation_metadata, _point_metadata, catchment_metadata,
                parameters, intensity_distribution, rng, xmin, xmax, ymin, ymax, output_paths, block_size,
                block_subset_size, spatial_raincell_method, spatial_buffer_factor, simulation_mode,
                weather_model=None, max_dsl=6.0, n_divisions=n_divisions, do_reordering=False,
                # TODO: Remove max_dsl - replace with n_divisions?
            )
            random_seed = rng.integers(1, 1000000000, 1)
            job = pool.apply_async(
                _fit_delta, (
                    spatial_model, bounds, df_wd, dc1, ref_var, n_shuffles, random_seed, n_divisions,
                    use_pooling, q
                )
            )
            jobs.append(job)

        i = 0
        for job in jobs:
            _delta = job.get()
            deltas[i, :] = _delta
            i += 1

        pool.close()
        pool.join()

    # Using serial processing
    else:

        for i in range(n_realisations):

            # Simulate realisation to be shuffled and fit delta
            df_wd, dc1, _ = simulation.simulate_realisation(
                realisation_id, datetime_helper, simulation_length, timestep_length, season_definitions,
                spatial_model, output_types, discretisation_metadata, _point_metadata, catchment_metadata,
                parameters, intensity_distribution, rng, xmin, xmax, ymin, ymax, output_paths, block_size,
                block_subset_size, spatial_raincell_method, spatial_buffer_factor, simulation_mode,
                weather_model=None, max_dsl=6.0, n_divisions=n_divisions, do_reordering=False,
                # TODO: Remove max_dsl - replace with n_divisions?
            )
            random_seed = rng.integers(1, 1000000000, 1)
            deltas[i, :] = _fit_delta(
                spatial_model, bounds, df_wd, dc1, ref_var, n_shuffles, random_seed, n_divisions, use_pooling
            )

    # Get final delta values - median of delta for each realisation
    deltas = ma.masked_equal(deltas, -999)
    deltas = ma.median(deltas, axis=0)
    # - use an average delta if it has not been calculated for a particular month for some reason
    tmp = np.zeros(12)
    tmp[deltas.mask] = ma.mean(deltas)
    tmp[~deltas.mask] = deltas[~deltas.mask]
    deltas = pd.DataFrame({'season': range(1, 12+1), 'delta': tmp})

    return deltas


def _fit_delta(
        spatial_model, bounds, df1, dc1, ref_var, n_shuffles, random_seed, n_divisions, use_pooling, q=None
):
    # Fit delta for one realisation
    # - df1 is df_wd (i.e. depths by window)
    # - dc1 contains a df with depths by window for each point if spatial model

    # TODO: DO NOT FORGET ABOUT PHI WHEN COMPARING OBSERVED AND SIMULATED VARIANCES

    if not spatial_model:
        n_points = 1
    else:
        n_points = len(dc1.keys())
        tmp_res = np.zeros((12, n_points))  # for calculating objective function

    # Initialisation - errors is for storing variance residuals
    rng = np.random.default_rng(random_seed)
    deltas_to_test = np.arange(bounds[0], bounds[1] + 0.00001, 0.1)  # 0.05)
    errors = np.zeros((12, n_shuffles, deltas_to_test.shape[0]))  # 12 months

    # Get window properties as arrays so numba can be used
    n_windows = df1.shape[0]
    win_id = df1['win_id'].values
    win_month = df1['month'].values
    win_length = df1['win_length'].values
    win_depth = df1['win_depth'].values

    # Loop shuffles and deltas to trial
    for i in range(n_shuffles):
        j = 0
        for _delta in deltas_to_test:
            delta = np.repeat(_delta, 12)

            # Shuffle windows
            random_numbers = rng.uniform(0.0, 1.0, n_windows)
            # !221116 (027) - test switching to float64
            # win_length = win_length.astype(np.float32)  # TESTING  # TODO: Confirm _shuffle_windows2() works in fitting
            # win_depth = win_depth.astype(np.float32)  # TESTING
            # random_numbers = random_numbers.astype(np.float32)  # TESTING
            # delta = delta.astype(np.float32)  # TESTING
            win_id2, win_depth2 = _shuffle_windows2(
                win_id, win_month, win_length, win_depth, delta, n_windows, random_numbers, n_divisions,
            )

            # Calculate variance from shuffled series and variance residual
            # if not spatial_model:  # !221205
            if not spatial_model:
                df2 = df1.copy()
                df2['win_id'] = win_id2
                df2['win_depth'] = win_depth2
                df2 = df2.groupby(['year', 'month'])['win_depth'].sum()
                df2 = df2.to_frame('win_depth')
                df2.reset_index(inplace=True)
                df2 = df2.groupby('month')['win_depth'].var()
                # TODO: Ensure objective functions are harmonised - not currently the same??
                res = ref_var.loc[ref_var['name'] == 'variance', 'value'].values - df2.values
            else:
                tmp_res.fill(0.0)
                for point_id in range(1, n_points+1):
                    # !221116 (033) - trying CV as objective function
                    # [[[
                    df2 = pd.DataFrame({'win_id': win_id2})
                    df2 = df2.merge(dc1[point_id][['win_id', 'win_depth']])
                    df2['year'] = df1['year'].values
                    df2['month'] = df1['month'].values
                    df2 = df2.groupby(['year', 'month'])['win_depth'].sum()
                    df2 = df2.to_frame('win_depth')
                    df2.reset_index(inplace=True)
                    # df2 = df2.groupby('month')['win_depth'].var()  # assume 1-12 order maintained
                    df2 = df2.groupby('month')['win_depth'].agg(['mean', 'var'])  # assume 1-12 order maintained
                    phi = ref_var.loc[(ref_var['name'] == 'variance') & (ref_var['point_id'] == point_id), 'phi'].values
                    _sim_mean = df2['mean'].values * phi
                    _sim_var = df2['var'].values * phi ** 2.0
                    # _ref_gs = ref_var.loc[
                    #     (ref_var['name'] == 'variance') & (ref_var['point_id'] == point_id), 'gs'
                    # ].values[0]

                    # if use_pooling:
                    #     ref_id = -1
                    # else:
                    #     ref_id = point_id

                    _ref_mean = ref_var.loc[
                        (ref_var['name'] == 'mean') & (ref_var['point_id'] == point_id), 'value'
                    ].values
                    _ref_mean = (
                        _ref_mean * np.array([31.0, 28.25, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0])
                    )
                    _ref_var = ref_var.loc[
                        (ref_var['name'] == 'variance') & (ref_var['point_id'] == point_id), 'value'
                    ].values
                    # !221113 - testing an objective function without scaling in 013
                    # tmp_res[:, point_id-1] = (1.0 / (_ref_gs / 100.0)) * (_ref_var - _sim_var)  # !221116 (034)
                    # tmp_res[:, point_id - 1] = _ref_var - _sim_var  # 013
                    tmp_res[:, point_id - 1] = _ref_var ** 0.5 / _ref_mean - _sim_var ** 0.5 / _sim_mean  # !221116 (033, 035)

                    # print(_delta, _ref_var[0], _sim_var[0])
                    # print(_ref_var ** 0.5 / _ref_mean)
                    # print(_sim_var ** 0.5 / _sim_mean)
                    # print()
                    # ]]]

                res = np.sum(tmp_res, axis=1)

            # Store variance residual
            errors[:, i, j] = res
            j += 1

        # sys.exit()

    # Find delta for each month
    tmp = []
    for month in range(1, 12+1):
        average_errors = np.median(errors[month-1, :, :], axis=0)  # by test delta value

        # print(month)
        # print(average_errors)
        # sys.exit()

        if np.max(average_errors) < 0:
            delta = 0.0
        else:
            smoothed_errors = np.zeros(average_errors.shape[0])  # by test delta value
            for i in range(2, average_errors.shape[0] - 1):
                smoothed_errors[i] = np.mean(average_errors[i - 2:i + 3])
            smoothed_errors[0] = average_errors[0]
            smoothed_errors[1] = np.mean([smoothed_errors[0], smoothed_errors[2]])
            smoothed_errors[-1] = average_errors[-1]
            smoothed_errors[-2] = np.mean([smoothed_errors[-1], smoothed_errors[-3]])

            # print(smoothed_errors)

            f = scipy.interpolate.interp1d(smoothed_errors, deltas_to_test)
            try:
                delta = f(0.0)
            except:
                delta = -999.0

        tmp.append(delta)

    delta = np.asarray(tmp)

    # print(delta)
    # sys.exit()

    return delta


def get_monthly_series(input_timeseries):
    # df = pd.read_csv(input_timeseries, index_col=0, parse_dates=True, infer_datetime_format=True, dayfirst=True)
    # df.reset_index(inplace=True)  # TODO: Figure out why this reset/set pattern is needed here
    # df.rename(columns={'DateTime': 'datetime'}, inplace=True)
    # df.set_index('datetime', inplace=True)
    # df.rename(columns={'Value': 'value', 'Precipitation': 'value'}, inplace=True)

    df = pd.read_csv(
        input_timeseries, names=['value'], index_col=0, skiprows=1, parse_dates=True, infer_datetime_format=True,
        dayfirst=True
    )

    df.loc[df['value'] < 0.0] = np.nan
    # df.set_index('datetime', inplace=True)
    df['season'] = df.index.month
    dfs = analysis.prepare_point_timeseries(
        df,
        season_definitions={m: m for m in range(1, 12 + 1)},
        completeness_threshold=0.0,
        durations=['24H', '1M'],  # assuming ar model is always monthly for now
        outlier_method=None,
        maximum_relative_difference=None,
        maximum_alterations=None,
    )
    df1 = dfs['1M'].copy()

    return df1


def fit_ar1(spatial_model, input_timeseries, point_metadata, calculation_period, ref_stats):

    # TODO: Consider whether to multiply reference series by phi or standardise first (spatial)?

    # Monthly reference series (for AR1 parameters)
    if not spatial_model:
        df1 = get_monthly_series(input_timeseries)
    else:
        dfs = []
        for _, row in point_metadata.iterrows():
            if 'file_name' in point_metadata.columns:
                file_name = row['file_name']
            else:
                file_name = row['name'] + '.csv'  # hardcoded file extension
            input_path = os.path.join(input_timeseries, file_name)
            df1 = get_monthly_series(input_path)
            # if len(dfs) > 0:
            # df1.drop(columns=['season'], inplace=True)
            df1.rename(columns={'value': 'value_' + str(row['point_id'])}, inplace=True)  # str(len(dfs))

            # Testing whether scaling by phi helps
            df1.reset_index(inplace=True)
            df1 = df1.merge(
                ref_stats.loc[
                    (ref_stats['point_id'] == row['point_id'])
                    & (ref_stats['statistic_id'] == ref_stats['statistic_id'].min()),
                    ['season', 'phi']
                ],
                left_on='season', right_on='season',
            )
            df1['value_' + str(row['point_id'])] /= df1['phi']
            df1.set_index('datetime', inplace=True)
            df1.drop(columns=['phi', 'season'], inplace=True)

            dfs.append(df1)

        # Concatenate and get mean of all points by month
        df1 = pd.concat(dfs, axis=1)
        df1['value'] = df1.mean(axis=1)
        df1 = df1[['value']].copy()

    # Temporal subset
    if calculation_period is not None:
        df1 = df1.loc[(df1.index.year >= calculation_period[0]) & (df1.index.year <= calculation_period[1])].copy()

    # Box-Cox transformation and standardisation
    df1['z_score'] = np.nan
    for month in range(1, 12 + 1):
        bc_value, lamda = scipy.stats.boxcox(df1.loc[df1.index.month == month, 'value'])
        df1.loc[df1.index.month == month, 'z_score'] = (bc_value - np.mean(bc_value)) / np.std(bc_value)

    # Ensure serially complete
    tmp = pd.DataFrame({
        'datetime': pd.date_range(df1.index.min(), df1.index.max(), freq='MS')},
    )
    tmp['dummy'] = np.zeros(tmp.shape[0])
    tmp.set_index('datetime', inplace=True)
    df1 = df1.merge(tmp, how='outer', left_index=True, right_index=True)
    df1.drop(columns='dummy', inplace=True)

    # Regressions
    df1['lag1_z_score'] = df1['z_score'].shift()
    df1.iloc[0, df1.columns.get_loc('lag1_z_score')] = 0.0
    slopes = []
    intercepts = []
    r2 = []
    stderr = []
    for month in range(1, 12 + 1):
        df1a = df1.loc[
            (df1.index.month == month) & (np.isfinite(df1['lag1_z_score'])) & (np.isfinite(df1['z_score']))
        ]
        result = scipy.stats.linregress(x=df1a['lag1_z_score'], y=df1a['z_score'])
        slopes.append(result.slope)
        intercepts.append(result.intercept)
        r2.append(result.rvalue ** 2)

        # Residuals - specifically standard deviation of residuals to allow simulation
        predicted = df1a['lag1_z_score'] * result.slope + result.intercept
        residuals = predicted - df1a['z_score']
        stderr.append(np.std(residuals))
        # stderr also equals (1.0 - result.rvalue**2) ** 0.5 * np.std(df1.loc[df1.index.month == month, 'z_score'])

    # Make dataframe (ultimately append delta here?)
    parameters = pd.DataFrame({
        'season': range(1, 12 + 1),
        'ar1_slope': slopes,
        'ar1_intercept': intercepts,
        'ar1_stderr': stderr,
    })

    # print(parameters)
    # sys.exit()

    return parameters


# /-/
# !221103 - Fixed window approach

# @numba.jit(nopython=True)
def _shuffle_windows2(win_id, win_month, win_length, win_depth, delta, n_windows, random_numbers, n_divisions):
    # TODO: Remove hardcoding of window sizes - n_divisions needs to be passed in accordingly
    # TODO: Try going back to float64 everywhere - may be fast enough now

    # Initialisation - careful of types if using numba
    win_id2 = np.zeros(n_windows, dtype=int) - 999
    win_depth2 = np.zeros(n_windows) - 999.0  # , dtype=np.float32  # !221116 (027) - test switching to float64
    # win_id2 = np.zeros(n_windows, dtype=numba.int64) - 999
    # win_depth2 = np.zeros(n_windows, dtype=numba.float64) - 999.0  # how does 32-bit work in numba - speed up too?

    # Prepare to jitter to avoid divide by zero etc errors
    dry_threshold = 0.01  # !221116 (030) - testing 0.1 instead of 0.01 (latter used in all previous runs)
    win_depth[win_depth < dry_threshold] = dry_threshold
    noise = random_numbers / 100.0
    # win_depth += (noise / 10.0)
    # win_depth += (noise[::-1] / 10.0)

    # First storm selection is arbitrary
    idx = 0
    win_id2[0] = win_id[idx]
    win_depth2[0] = win_depth[idx]
    prev_depth = win_depth[idx]  # np.array([win_depth[idx]], dtype=numba.float64)

    # Initialise masks - modified in loop to help guide selection
    mask = np.zeros((12, n_windows), dtype=bool)
    mask2 = np.zeros(n_windows, dtype=bool)  # for february leap years
    # mask = np.zeros((12, n_windows), dtype=numba.boolean)
    # mask2 = np.zeros(n_windows, dtype=numba.boolean)  # for february leap years
    for month in range(1, 12+1):
        if month != 2:
            mask[month-1, :] = win_month == month
        else:
            mask[month-1, :] = (win_month == month) & (win_length == ((28 * 24) / float(n_divisions)))
    mask[0, 0] = False  # i.e. first window in first month set above
    mask2[:] = (win_month == 2) & (win_length == ((29 * 24) / float(n_divisions)))

    # Then loop through storms
    for win_idx in range(n_windows):

        month = win_month[win_idx]
        len_ = win_length[win_idx]

        if win_depth2[win_idx] != -999.0:
            pass
        else:

            # Get IDs and depths of candidate windows
            if len_ != ((29 * 24) / float(n_divisions)):
                id1 = win_id[mask[month-1]]
                dep1 = win_depth[mask[month-1]]
            else:
                id1 = win_id[mask2]
                dep1 = win_depth[mask2]

            # Turning off this clause for now on the basis that the probability after jittering should be very low
            if np.any(dep1 / prev_depth == 1.0):
                prev_depth += noise[win_idx]

                # !221115 (022) - commenting out additional clause below
                # ci = 0
                # while np.any(dep1 / prev_depth == 1.0):
                #     prev_depth += (noise[win_idx] ** 2.0 / 10.0)
                #     if ci == 10:
                #         break

            # Similarity index and probability for each window
            si = (1.0 / np.absolute(np.log(dep1 / prev_depth))) ** delta[month - 1]
            pi = (1.0 / np.sum(si)) * si

            # Select a window
            idx = np.searchsorted(np.cumsum(pi), random_numbers[win_idx], side="right")  # testing
            idx = min(idx, id1.shape[0] - 1)  # out-of-bounds errors possible if moving to float32

            # Store selection
            win_id2[win_idx] = id1[idx]
            win_depth2[win_idx] = dep1[idx]

            # Update mask so selected window can no longer be chosen
            if len_ != ((29 * 24) / float(n_divisions)):
                mask[month-1, id1[idx]] = 0
            else:
                mask2[id1[idx]] = 0

            # TESTING - should no longer be needed
            # !221115 (022) - uncommenting below
            if dep1[idx] <= dry_threshold:
                prev_depth = dry_threshold + noise[win_idx]
            else:
                prev_depth = dep1[idx]

            # Prepare for next window
            # !221115 (022) - commenting below
            # prev_depth = dep1[idx]

        win_idx += 1

    return win_id2, win_depth2


def _shuffle_simulation(df, df1, parameters, datetime_helper, rng, n_divisions, do_reordering):
    # - df is raincells
    # - df1 is window depths
    # - do_reordering is currently boolean for monthly ar1 model (or not)

    # Get window properties as arrays so numba can be used
    n_windows = df1.shape[0]
    win_id = df1['win_id'].values
    win_month = df1['month'].values
    win_length = df1['win_length'].values
    win_depth = df1['win_depth'].values

    # Shuffle windows
    delta = parameters['delta'].values
    random_numbers = rng.uniform(0.0, 1.0, n_windows)
    # !221116 (027) - test switching to float64
    # win_length = win_length.astype(np.float32)  # TESTING
    # win_depth = win_depth.astype(np.float32)  # TESTING
    # random_numbers = random_numbers.astype(np.float32)  # TESTING
    # delta = delta.astype(np.float32)  # TESTING
    win_id2, win_depth2 = _shuffle_windows2(
        win_id, win_month, win_length, win_depth, delta, n_windows, random_numbers, n_divisions,
    )

    # New windows df
    df2 = pd.DataFrame({'win_id': win_id2})
    df2['new_rank'] = np.arange(df2.shape[0])
    tmp = df1[['win_id', 'year']].copy()
    tmp.rename(columns={'win_id': 'new_rank'}, inplace=True)
    df2 = df2.merge(tmp)

    # Merge new ranks into raincells df and sort on order
    df.sort_values(['win_id', 'raincell_arrival'], inplace=True)
    df.drop(columns=['year'], inplace=True)  # TESTING
    df = df.merge(df2[['win_id', 'new_rank', 'year']])
    df.sort_values(['new_rank', 'raincell_arrival'], inplace=True)
    df['season'] = df['month']

    # Merge serial window start times into df in order to adjust times after shuffling
    # - i.e. adjusted from original time relative to simulation origin to new time relative to storm origin (after
    # shuffling has taken place by reordering df)
    tmp = df1.loc[:, ['win_id', 'win_start']].copy()
    tmp.rename(columns={'win_id': 'new_rank', 'win_start': 'new_win_start'}, inplace=True)
    df = df.merge(tmp)
    time_adj = df['win_start'].values - df['new_win_start'].values
    df['storm_arrival'] -= time_adj
    df['raincell_arrival'] -= time_adj
    df['raincell_end'] -= time_adj
    df.drop(columns=['win_start'], inplace=True)
    df.rename(columns={'new_win_start': 'win_start'}, inplace=True)

    # Reordering via monthly AR1 model optional for now
    if do_reordering:

        # Simulate with AR1 model
        df3 = _simulate_ar1(parameters, df1['year'].min(), df1['year'].max(), rng)
        df3.rename(columns={'season': 'month'}, inplace=True)
        df3 = df3.merge(datetime_helper[['year', 'month', 'start_time']])
        df3.rename(columns={'start_time': 'new_month_start'}, inplace=True)
        df3['new_month_rank'] = -999
        for month in range(1, 12+1):
            df3.loc[df3['month'] == month, 'new_month_rank'] = np.arange(df1['year'].max() - df1['year'].min() + 1) + 1

        # Original month start times
        df = df.merge(
            datetime_helper.loc[
                (datetime_helper['year'] >= df1['year'].min()) & (datetime_helper['year'] <= df1['year'].max()),
                ['year', 'month', 'start_time']
            ]
        )
        df.rename(columns={'start_time': 'month_start'}, inplace=True)

        # Also need to get current rank of month (re depth) - ensuring based on serially complete set of windows
        tmp = df.groupby(['year', 'month'])['raincell_depth'].sum()
        tmp = tmp.to_frame('depth')
        tmp.reset_index(inplace=True)
        tmp = tmp.merge(
            datetime_helper.loc[
                (datetime_helper['year'] >= df1['year'].min())
                & (datetime_helper['year'] <= df1['year'].max()),
                ['year', 'month']
            ],
            how='outer'
        )
        tmp['depth'] = np.where(~np.isfinite(tmp['depth']), 0.0, tmp['depth'])
        tmp['leap_feb'] = [utils.check_if_leap_year(year) for year in tmp['year']]
        tmp.loc[tmp['month'] != 2, 'leap_feb'] = 0
        tmp['month_rank'] = tmp.groupby(['month'])['depth'].rank(method='first', ascending=False).astype(int)
        tmp.sort_values(['year', 'month'], inplace=True)

        # Adjust ranks in reordering df so leap years match up with what is available in the simulation
        tmp['ly_flag'] = 0
        for year in range(df1['year'].min(), df1['year'].max()):
            leap_feb_ranks = tmp.loc[(tmp['ly_flag'] == 0) & (tmp['leap_feb'] == 1), 'month_rank'].values
            nonleap_feb_ranks = tmp.loc[
                (tmp['ly_flag'] == 0) & (tmp['month'] == 2) & (tmp['leap_feb'] == 0), 'month_rank'
            ].values

            ar1_rank = df3.loc[(df3['year'] == year) & (df3['month'] == 2), 'month_rank'].values[0]
            sim_year = tmp.loc[(tmp['month'] == 2) & (tmp['month_rank'] == ar1_rank), 'year'].values[0]

            if utils.check_if_leap_year(year):
                if utils.check_if_leap_year(sim_year):
                    adjust = False
                else:
                    adjust = True
                    new_ar1_rank = leap_feb_ranks[(np.abs(ar1_rank - leap_feb_ranks)).argmin()]
            elif not utils.check_if_leap_year(year):
                if not utils.check_if_leap_year(sim_year):
                    adjust = False
                else:
                    adjust = True
                    new_ar1_rank = nonleap_feb_ranks[(np.abs(ar1_rank - nonleap_feb_ranks)).argmin()]

            if adjust:
                df3.loc[(df3['month'] == 2) & (df3['month_rank'] == new_ar1_rank), 'month_rank'] = ar1_rank
                df3.loc[(df3['year'] == year) & (df3['month'] == 2), 'month_rank'] = new_ar1_rank
                tmp.loc[(tmp['month'] == 2) & (tmp['month_rank'] == new_ar1_rank), 'ly_flag'] = 1

        # Join month ranks
        tmp.drop(columns=['ly_flag'], inplace=True)
        df = df.merge(tmp)
        df.drop(columns='depth', inplace=True)

        # Join AR1 ranks and then reorder raincells df
        df = df.merge(df3[['month', 'month_rank', 'new_month_rank', 'new_month_start']])  # 'year',
        df.sort_values(['month', 'new_month_rank'], inplace=True)  # , 'new_rank'
        df['year'] = (df['new_month_rank'] - 1) + df1['year'].min()
        df.sort_values(['year', 'month', 'new_rank'], inplace=True)

        # Adjust times again so all relative to block origin
        df['storm_arrival'] = df['storm_arrival'] - df['month_start'] + df['new_month_start']
        df['raincell_arrival'] = df['raincell_arrival'] - df['month_start'] + df['new_month_start']
        df['raincell_end'] = df['raincell_end'] - df['month_start'] + df['new_month_start']

    return df


def _simulate_ar1(parameters, min_year, max_year, rng):
    years = []
    seasons = []
    sa_series = []
    sa_lag1 = 0.0
    year = min_year
    parameters.sort_values('season', inplace=True)
    m = parameters['ar1_slope'].tolist()
    c = parameters['ar1_intercept'].tolist()
    stderr = parameters['ar1_stderr'].tolist()
    while year <= max_year:
        for season in range(1, 12 + 1):
            sa = sa_lag1 * m[season - 1] + c[season - 1] + rng.normal(loc=0.0, scale=stderr[season - 1])
            sa_series.append(sa)
            years.append(year)
            seasons.append(season)
            sa_lag1 = sa
        year += 1
    df_ar1 = pd.DataFrame({
        'year': years,
        'season': seasons,
        'sa': sa_series,
    })
    df_ar1['leap_feb'] = [utils.check_if_leap_year(year) for year in df_ar1['year']]
    df_ar1.loc[df_ar1['season'] != 2, 'leap_feb'] = 0
    # df_ar1['month_rank'] = df_ar1.groupby(['season', 'leap_feb'])['sa'].rank(method='first', ascending=False).astype(int)
    df_ar1['month_rank'] = df_ar1.groupby(['season'])['sa'].rank(method='first', ascending=False).astype(int)
    return df_ar1


# TESTING - incomplete attempt to interpolate variance to a specified duration based on fixed window simulations
# def _aggregate_windows(df, aggregations, ref_dur):
#     print(df)
#     print(aggregations)
#     # sys.exit()
#
#     n_ref_dur = int(np.floor(672 / ref_dur))  # number in month e.g. 2 x 14D
#
#     df['local_end'] = df.groupby(['year', 'month'])['win_length'].cumsum()  # MODIFYING DF - DROP AT END
#
#     for win_len, aggs in aggregations.items():
#         if win_len == 168.0:
#             pass
#         else:
#             for agg in (3, ):  # aggs:
#                 print(win_len, agg, win_len * agg)
#                 df1 = df.loc[df['win_length'] == win_len].copy()
#                 if agg == 1:
#                     df1['agg_id'] = np.arange(df1.shape[0])
#                 else:
#                     df1['agg_id'] = np.mod(df['local_end'], win_len * agg) == 0.0
#                     df1['agg_id'] = df1.groupby(['year', 'month'])['agg_id'].shift().cumsum()  # .astype(int)
#                     # df1['agg_id'] = np.where(~np.isfinite(df1['agg_id']), 0, df1['agg_id'])
#                 print(df1.head(10))
#                 print(df1.tail(10))
#                 print('--')
#             sys.exit()

# /-/


def fit_delta_for_realisation(
        df, ref, n_workers, random_seeds, n_shuffles, datetime_helper, variance_duration, dsl_id, real_id, q=None
):
    df['year'] = get_years(df['season'].values)

    if n_workers > 1:

        import multiprocessing as mp

        manager = mp.Manager()
        q = manager.Queue()
        pool = mp.Pool(n_workers)

        jobs = []
        for season in range(1, 12 + 1):
            df1 = df.loc[df['season'] == season]
            # storm_id = df1['storm_id'].values
            # storm_depth = df1['storm_depth'].values
            # storm_duration = df1['storm_duration'].values  # 03/10/2022
            tmp_id = np.arange(df1['storm_id'].shape[0])
            # years = df1['year'].values
            season_ref = ref.loc[ref.index == season]

            # df_s = df.loc[df['season'] == season]

            job = pool.apply_async(
                find_delta, (
                    (0.0, 1.0), tmp_id, df1, season_ref,
                    random_seeds[((season - 1) * n_shuffles):(season * n_shuffles)], n_shuffles, datetime_helper,
                    variance_duration, q
                )
            )
            jobs.append(job)

        results = []
        for job in jobs:
            result = job.get()
            results.append(result)

        pool.close()
        pool.join()

    else:

        results = []
        for season in range(1, 12 + 1):
            # print(' - season =', season)

            df1 = df.loc[df['season'] == season]
            # storm_id = df1['storm_id'].values
            # storm_depth = df1['storm_depth'].values
            # storm_duration = df1['storm_duration'].values  # 03/10/2022
            tmp_id = np.arange(df1['storm_id'].shape[0])
            # years = df1['year'].values
            season_ref = ref.loc[ref.index == season]

            # df_s = df.loc[df['season'] == season]

            result = find_delta(
                (0.0, 1.0), tmp_id, df1, season_ref,
                random_seeds[((season - 1) * n_shuffles):(season * n_shuffles)], n_shuffles, datetime_helper,
                variance_duration
            )
            results.append(result)

    return [dsl_id, real_id, np.asarray(results)]


def find_delta(
        bounds, tmp_id, df1, season_ref, random_seeds, n_shuffles, datetime_helper, variance_duration, q=None
):
    deltas_to_test = np.arange(bounds[0], bounds[1] + 0.00001, 0.1)  # 0.05)

    storm_id = df1['storm_id'].values
    storm_depth = df1['storm_depth'].values
    storm_duration = df1['storm_duration'].values
    years = df1['year'].values

    errors = np.zeros((n_shuffles, deltas_to_test.shape[0]))

    # OLD - month_ends only correct if using calendar months
    # month_starts = datetime_helper.loc[datetime_helper['month'] == season_ref.index[0], 'start_time'].values
    # month_ends = datetime_helper.loc[datetime_helper['month'] == season_ref.index[0], 'end_time'].values

    # Correcting end times in variance calculations for 28-day not monthly AND seeing if it is possible to use/test
    # a different variance duration(s) (i.e. not just hardcoded to 28 days)
    # period_starts = datetime_helper.loc[datetime_helper['month'] == season_ref.index[0], 'start_time'].values
    # period_ends = period_starts + 672.0  # assuming 28-day variance used
    if variance_duration[-1] == 'H':
        _variance_duration = int(variance_duration[:-1])
        n_groups = int(np.floor((30 * 24) / _variance_duration))  # approximation here if unequal division
    elif variance_duration[-1] == 'M':
        if season_ref.index[0] in [1, 3, 5, 7, 8, 10, 12]:
            _variance_duration = 31 * 24
        elif season_ref.index[0] in [4, 6, 9, 11]:
            _variance_duration = 30 * 24
        elif season_ref.index[0] == 2:
            _variance_duration = 28.25 * 24  # 29 * 24
        n_groups = 1
    period_starts = []
    period_ends = []
    for group in range(n_groups):
        period_starts.append(
            datetime_helper.loc[datetime_helper['month'] == season_ref.index[0], 'start_time'].values
            + (group * _variance_duration)
        )
        period_ends.append(
            datetime_helper.loc[datetime_helper['month'] == season_ref.index[0], 'start_time'].values
            + ((group + 1) * _variance_duration)
        )
    period_starts = np.sort(np.concatenate(period_starts, dtype=float))
    period_ends = np.sort(np.concatenate(period_ends, dtype=float))

    target_var = float(season_ref.values[0][0])

    # !221025 - for numba
    n_storms = storm_id.shape[0]
    storm_id2 = np.zeros(storm_id.shape, dtype=int) - 999
    storm_depth2 = np.zeros(storm_id.shape, dtype=float) - 999.0
    storm_duration2 = np.zeros(storm_id.shape, dtype=float) - 999.0
    shuffled = np.zeros(storm_id.shape, dtype=int)
    rng = np.random.default_rng(random_seeds[0])

    for i in range(n_shuffles):
        j = 0
        for delta in deltas_to_test:

            # !221025 - original - revert to this if numba not working
            # storm_id2, storm_depth2, storm_duration2 = shuffle_storms(
            #     tmp_id, storm_id, storm_depth, storm_duration, delta, random_seeds[i]
            # )

            # !221025 - numba
            storm_id2.fill(-999)
            storm_depth2.fill(-999.0)
            storm_duration2.fill(-999.0)
            shuffled.fill(0)
            idx = 0
            random_numbers = rng.uniform(0.0, 1.0, n_storms)
            storm_id2, storm_depth2, storm_duration2 = shuffle_storms_numba2(
                tmp_id, storm_id, storm_depth, storm_duration, delta, n_storms, storm_id2, storm_depth2,
                storm_duration2,
                shuffled, idx, random_numbers
            )

            df = pd.DataFrame({
                'year': years, 'storm_depth': storm_depth2, 'storm_duration': storm_duration2,
            })  # 'storm_id': storm_id2,

            df['storm_arrival'] = df1['storm_arrival'].values
            # df['storm_duration'] = df_s['storm_duration'].values
            # df['start_time'] = df_s['start_time'].values
            df['storm_end'] = df['storm_arrival'] + df['storm_duration']

            # print(df)
            # print(df.columns)
            # print(df.shape[0], df_s.shape[0])
            # print('--')
            # print(season_ref.index[0])
            # print('/')
            # print(season_ref)
            # print(season_ref.values[0][0])
            # sys.exit()

            # --
            # TESTING
            # tmp = datetime_helper.loc[
            #     datetime_helper['month'] == season_ref.index[0], ['year', 'start_time', 'end_time']
            # ].copy()
            # tmp.loc[:, 'year'] -= (tmp['year'].min() - 1)
            # df = df.merge(tmp)
            # df['storm_arrival'] -= df['start_time']
            # df['storm_end'] -= df['start_time']
            # df['end_time'] -= df['start_time']
            # df['start_time'] = 0.0
            #
            # tmp = df.loc[df['storm_end'] > df['end_time']].copy()
            # tmp['storm_arrival'] = 0.0
            # tmp['storm_end'] -= tmp['end_time']
            # tmp['storm_depth'] = tmp['storm_end'] / tmp['storm_duration'] * tmp['storm_depth']
            # tmp['storm_duration'] = tmp['storm_end']
            # tmp['year'] += 1
            # tmp = tmp.loc[tmp['year'] <= df['year'].max()]
            #
            # df['storm_depth'] = np.where(
            #     df['storm_end'] > df['end_time'],
            #     ((df['end_time'] - df['storm_arrival']) / df['storm_duration']) * df['storm_depth'],
            #     df['storm_depth']
            # )
            # df['storm_duration'] = np.where(df['storm_end'] > df['end_time'], df['end_time'], df['storm_duration'])
            # df['storm_end'] = np.where(df['storm_end'] > df['end_time'], df['end_time'], df['storm_end'])
            #
            # print(df)
            # print(df.loc[df['year'] == 1, ['year', 'storm_arrival', 'storm_end', 'start_time', 'end_time']])
            # print(df.loc[df['year'] == 2, ['year', 'storm_arrival', 'storm_end', 'start_time', 'end_time']])
            # print(tmp)
            # sys.exit()
            #
            # TODO: If pursuing, consider how it could be adapted for non-1M durations

            # MORE TESTING
            # - another option would be just to make one continuous series and split it up into the desired duration
            # and drop the final block if complete

            # Make months consecutive and times relative to origin of concatenated months
            tmp = datetime_helper.loc[
                datetime_helper['month'] == season_ref.index[0], ['year', 'start_time', 'end_time', 'n_hours']
            ].copy()
            tmp.loc[:, 'year'] -= (tmp['year'].min() - 1)
            tmp['start_time2'] = np.cumsum(tmp['n_hours']) - tmp['n_hours'].values[0]
            df = df.merge(tmp)
            df['storm_arrival'] -= df['start_time']
            df['storm_end'] -= df['start_time']
            df['end_time'] -= df['start_time']
            df['start_time'] = 0.0
            df['storm_arrival'] += df['start_time2']
            df['storm_end'] += df['start_time2']
            df['end_time'] += df['start_time2']
            df['start_time'] += df['start_time2']
            df.drop(columns=['start_time2'], inplace=True)

            # Loop through according to desired variance duration to get array of totals (from which variance can be
            # calculated)
            # TODO: Probably possible to make more efficient with numba
            period_start = 0.0
            period_end = _variance_duration
            sums = []
            storm_arrivals = df['storm_arrival'].values
            storm_ends = df['storm_end'].values
            storm_durations = df['storm_duration'].values
            storm_depths = df['storm_depth'].values
            while period_end <= df['end_time'].max():
                mask = (storm_arrivals < period_end) & (storm_ends > period_start)
                if np.sum(mask) > 0:
                    eff_arrivals = np.maximum(storm_arrivals[mask], period_start)
                    eff_ends = np.minimum(storm_ends[mask], period_end)
                    frac_coverage = (eff_ends - eff_arrivals) / storm_durations[mask]
                    eff_depths = frac_coverage * storm_depths[mask]
                    sums.append(np.sum(eff_depths))
                else:
                    sums.append(0.0)
                period_start = period_end
                period_end += _variance_duration

            sums = np.asarray(sums)
            var = np.var(sums, ddof=1)  # matching current reference

            # print(var)
            # print(sums)
            # print(tmp)
            # print(df[['start_time', 'end_time', 'storm_arrival', 'storm_end']])
            # print(df.loc[df['year'] == 1, ['start_time', 'end_time', 'storm_arrival', 'storm_end']])
            # print(df.loc[df['year'] == 2, ['start_time', 'end_time', 'storm_arrival', 'storm_end']])
            # print(df.loc[df['year'] == 3, ['start_time', 'end_time', 'storm_arrival', 'storm_end']])
            # print(df.loc[df['year'] == 4, ['start_time', 'end_time', 'storm_arrival', 'storm_end']])
            # sys.exit()

            # // overwriting df so that can see how the var numbers compare
            # df = pd.DataFrame({
            #     'year': years, 'storm_depth': storm_depth2, 'storm_duration': storm_duration2,
            # })  # 'storm_id': storm_id2,
            # df['storm_arrival'] = df1['storm_arrival'].values
            # # df['storm_duration'] = df_s['storm_duration'].values
            # # df['start_time'] = df_s['start_time'].values
            # df['storm_end'] = df['storm_arrival'] + df['storm_duration']
            # //

            # --

            # *** Below is original approach to getting variance (only works for ~1M durations) ***

            """

            # Non-looping approach - assuming no overlap possible between consecutive months as separated by a year
            # through seasonal stratification here
            # - ensure that any late storms are accounted for (i.e. remove before lead to bounds error)
            df = df.loc[df['storm_arrival'] < np.max(period_ends)]
            year_idx = np.digitize(df['storm_arrival'], period_ends, right=True)
            # year_idx = year_idx[year_idx <= (np.max(years) - 1)]

            if df.shape[0] != year_idx.shape[0]:
                print(df.shape[0], year_idx.shape[0])
                print(df['storm_arrival'])
                print(year_idx)
                sys.exit()

            df['start_time'] = period_starts[year_idx]
            df['end_time'] = period_ends[year_idx]
            df['storm_arrival'] = np.maximum(df['storm_arrival'], df['start_time'])
            df['storm_end'] = np.minimum(df['storm_end'], df['end_time'])
            df['duration_trunc'] = df['storm_end'] - df['storm_arrival']
            df['coverage'] = df['duration_trunc'] / df['storm_duration']
            df['depth_trunc'] = df['storm_depth'] * df['coverage']

            # print(df[['storm_arrival', 'storm_end', 'start_time']])
            # print(df.loc[df['year'] == 2, ['year', 'storm_arrival', 'storm_end', 'start_time', 'end_time', 'depth_trunc']])
            # print(df.columns)
            # print(df.loc[df['year'] == 2, 'depth_trunc'].sum())
            # sys.exit()

            # Original calculation of month totals - not accounting for start/end edge effects
            df = df.groupby(['year'])['storm_depth'].sum()
            var = np.var(df.values, ddof=1)  # matching current reference

            """

            # print(var)
            # sys.exit()

            # print(datetime_helper['month'].unique())
            # sys.exit()

            # Loop (by month) to get total storm depths for month (well, for 28-day period)
            # totals = []
            # for start_time in month_starts:
            #     end_time = start_time + 28.0 * 24.0  # rather than row['end_time'] to get first 28 days...
            #     df_sub = df.loc[(df['storm_end'] > start_time) & (df['storm_arrival'] < end_time)].copy()
            #     if df_sub.shape[0] > 0:
            #         df_sub['storm_arrival'] = np.maximum(df['storm_arrival'], start_time)
            #         df_sub['storm_end'] = np.minimum(df['storm_end'], end_time)
            #         df_sub['duration_sub'] = df_sub['storm_end'] - df_sub['storm_arrival']
            #         df_sub['coverage'] = df_sub['duration_sub'] / df_sub['storm_duration']
            #         df_sub['depth_sub'] = df_sub['storm_depth'] * df_sub['coverage']
            #         # if df_sub['coverage'].min() < 1.0:
            #         #     print(df_sub)
            #         #     print(df_sub['coverage'].min())
            #         #     sys.exit()
            #         totals.append(df_sub['depth_sub'].sum())
            #     else:
            #         totals.append(0.0)
            #
            # var = np.var(np.asarray(totals), ddof=1)  # matching current reference

            # error = float(season_ref) - float(var)
            error = target_var - var
            errors[i, j] = error
            j += 1

            # print(var, error)

            # if i == 5:
            #     print(delta, float(season_ref), float(var), error)
        # if i == 5:
        #     sys.exit()

    # mean_ofs = np.mean(ofs, axis=0)
    # mean_vars = np.mean(vars_, axis=0)
    average_errors = np.median(errors, axis=0)  # by test delta value

    if np.max(average_errors) < 0:
        delta = 0.0
    else:
        smoothed_errors = np.zeros(average_errors.shape[0])  # by test delta value
        for i in range(2, average_errors.shape[0] - 1):
            smoothed_errors[i] = np.mean(average_errors[i - 2:i + 3])
        smoothed_errors[0] = average_errors[0]
        smoothed_errors[1] = np.mean([smoothed_errors[0], smoothed_errors[2]])
        smoothed_errors[-1] = average_errors[-1]
        smoothed_errors[-2] = np.mean([smoothed_errors[-1], smoothed_errors[-3]])

        # for se in smoothed_errors:
        #     print(se)
        # sys.exit()

        f = scipy.interpolate.interp1d(smoothed_errors, deltas_to_test)
        try:
            delta = f(0.0)
        except:
            delta = -999

    # print(delta)

    # print(mean_ofs)
    # print(mean_vars)

    # pars, _ = scipy.optimize.curve_fit(f1, points, mean_ofs)

    # print(pars)

    # sys.exit()

    # fp = 'H:/Projects/rwgen/working/iss13/stnsrp/df_8q.csv'
    # with open(fp, 'w') as fh:
    #     for p, of, var in zip(points, mean_ofs, mean_vars):
    #         fh.write(str(p) + ',' + str(of) + ',' + str(var) + '\n')

    # print(points)
    # print(ofs)

    # sys.exit()

    return delta


def shuffle_storms(tmp_id, storm_id, storm_depth, storm_duration, delta, random_seed=None):
    # Assumes that already subset on season

    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    # ***
    # 28/09/2022 - Testing
    if storm_id.shape[0] == 0:
        print('no storms in this month')
        sys.exit()
    # ***

    # Initialise columns to store shuffled storm ID and depth - plus flag for whether already shuffled
    storm_id2 = np.zeros(storm_id.shape[0], dtype=int) - 999
    storm_depth2 = np.zeros(storm_id.shape[0]) - 999
    storm_duration2 = np.zeros(storm_id.shape[0]) - 999
    shuffled = np.zeros(storm_id.shape[0], dtype=int)

    # First storm selection is arbitrary
    idx = rng.integers(0, storm_id.shape[0] - 1, 1)
    storm_id2[0] = storm_id[idx]
    storm_depth2[0] = storm_depth[idx]
    storm_duration2[0] = storm_duration[idx]
    shuffled[idx] = 1
    prev_storm_depth = storm_depth[idx]

    # Then loop through storms
    for storm_idx in range(storm_id.shape[0]):
        # print(storm_idx)
        if storm_depth2[storm_idx] != -999:
            pass
        else:
            # 03/10/2022 - original commented out below | 04/10/2022 - restored
            mask = shuffled == 0
            tmp_id1 = tmp_id[mask]
            id1 = storm_id[mask]
            dep1 = storm_depth[mask]
            dur1 = storm_duration[mask]
            si = (1.0 / np.absolute(np.log(dep1 / prev_storm_depth))) ** delta
            pi = (1.0 / np.sum(si)) * si
            idx = rng.choice(id1.shape[0], p=pi)
            storm_id2[storm_idx] = id1[idx]
            storm_depth2[storm_idx] = dep1[idx]
            storm_duration2[storm_idx] = dur1[idx]
            shuffled[int(tmp_id1[idx])] = 1
            prev_storm_depth = dep1[idx]

            # 03/10/2022 - testing
            # dur = storm_duration[storm_idx]
            # dur_diff = np.absolute(storm_duration - dur)
            # if np.sum(shuffled == 0) >= 10:
            #     dur_threshold = np.percentile(dur_diff[shuffled == 0], 10.0)
            #     dur_threshold = max(np.min(dur_diff[(shuffled == 0) & (dur_diff > 0.0)]), dur_threshold)
            #     mask = (shuffled == 0) & (dur_diff <= dur_threshold)
            # else:
            #     mask = shuffled == 0
            # tmp_id1 = tmp_id[mask]
            # id1 = storm_id[mask]
            # dep1 = storm_depth[mask]
            # si = (1.0 / np.absolute(np.log(dep1 / prev_storm_depth))) ** delta
            # pi = (1.0 / np.sum(si)) * si
            # idx = rng.choice(id1.shape[0], p=pi)
            # storm_id2[storm_idx] = id1[idx]
            # storm_depth2[storm_idx] = dep1[idx]
            # shuffled[int(tmp_id1[idx])] = 1
            # prev_storm_depth = dep1[idx]

        storm_idx += 1

    return storm_id2, storm_depth2, storm_duration2


def shuffle_simulation(df_rc, df_sd, parameters, datetime_helper, rng):
    # need to break up into ~30-year periods
    # need to break up by season (month)

    # df_rc is full df of storms + raincells
    # df_sd is storm depths

    # print(df_sd.shape[0], df_rc.shape[0])

    # Add in month start times

    # print(df_rc[['storm_arrival', 'raincell_arrival', 'raincell_end']])
    # print(df_rc.columns)
    # sys.exit()

    # ***
    # 28/09/2022 - TESTING
    # df_sd['year_'] = get_years(df_sd['season'].values)
    # df_sd.to_csv('H:/Projects/rwgen/working/iss13/nsrp/df_sd1.csv')
    # sys.exit()
    # ***

    # Threshold for applying shuffling (if delta > delta_threshold)
    delta_threshold = 0.05

    # Get years for storm depths df first so know maximum year (to subset datetime_helper in case of >1 blocks)
    df_sd['year'] = get_years(df_sd['season'].values)

    # Prepare datetime helper - get years beginning from 1, as used in other dfs here
    df_dt = datetime_helper.copy()
    df_dt['year'] -= np.min(df_dt['year']) - 1  # to get in range 1 to n years
    df_dt.rename(columns={'month': 'season'}, inplace=True)
    df_dt = df_dt.loc[df_dt['year'] <= df_sd['year'].max()]

    # ***
    # 28/09/2022
    # print(datetime_helper['year'].min(), datetime_helper['year'].max(), datetime_helper['year'].shape[0])
    # print(df_dt['year'].min(), df_dt['year'].max(), df_dt['year'].shape[0])
    # print(df_sd.shape[0], df_rc.shape[0])
    # sys.exit()
    # ***

    # Get years and storm arrival times relative to month start times for storm depths df
    # df_sd['year'] = get_years(df_sd['season'].values)
    df_sd = pd.merge(df_sd, df_dt[['year', 'season', 'start_time']])
    df_sd['storm_arrival_mon'] = df_sd['storm_arrival'] - df_sd['start_time']

    # Get years and storm arrival times relative to month start times for full raincells df
    df_rc['year'] = get_years(df_rc['season'].values)
    df_rc = pd.merge(df_rc, df_dt[['year', 'season', 'start_time']])
    df_rc['storm_arrival_mon'] = df_rc['storm_arrival'] - df_rc['start_time']
    df_rc['raincell_arrival_mon'] = df_rc['raincell_arrival'] - df_rc['start_time']
    df_rc['raincell_end_mon'] = df_rc['raincell_end'] - df_rc['start_time']

    # print(df_sd.shape[0], df_rc.shape[0])

    # print(df_rc)
    # sys.exit()

    # !! LOOP HERE FOR ~30-YEAR PERIODS !!

    random_seeds = rng.integers(1000000, 1000000000, 12)  # TODO (221025): Remove if no longer in use

    # sd_dfs = []
    rc_dfs = []

    n_years = 1000  # 30  # TODO: Determine as input / from dataframes above

    # Line below used to keep at least 30 years in final subset - does not work if n_years is large
    # start_years = list(range(1, df_dt['year'].max() - n_years + 1, n_years))
    start_years = list(range(1, df_dt['year'].max() + 1, n_years))

    # ***
    # 28/09/2022 - Testing shuffling stuff - nsrp_06 (+...)

    # print(df_dt['year'].max())
    # sys.exit()

    # Is the way that the start_years are split up affecting the estimation of delta?
    # nsrp_06 shuffled crashes with start_years = [1, 1001], but none of the other tests have done...

    # Try just with the one start_year then... - may be worth repeating the baseline accordingly...
    start_years = [1]

    # print(start_years)
    # sys.exit()

    # ***

    # import datetime
    # t1 = datetime.datetime.now()

    # ---
    # AR1 model

    # t1 = datetime.datetime.now()

    years = []
    seasons = []
    sa_series = []
    sa_lag1 = 0.0
    year = df_dt['year'].min()
    parameters.sort_values('season', inplace=True)
    m = parameters['ar1_slope'].tolist()
    c = parameters['ar1_intercept'].tolist()
    stderr = parameters['ar1_stderr'].tolist()
    while year <= df_dt['year'].max():
        for season in range(1, 12 + 1):
            sa = sa_lag1 * m[season - 1] + c[season - 1] + rng.normal(loc=0.0, scale=stderr[season - 1])
            sa_series.append(sa)
            years.append(year)
            seasons.append(season)
            sa_lag1 = sa
        year += 1
    df_ar1 = pd.DataFrame({
        'year': years,
        'season': seasons,
        'sa': sa_series,
    })
    df_ar1['rank'] = df_ar1.groupby(['season'])['sa'].rank(method='first').astype(int)

    # t2 = datetime.datetime.now()
    # print(df_ar1)
    # print(df_ar1['rank'].min(), df_ar1['rank'].max())
    # print(df_ar1.groupby(['season'])['sa'].agg(['mean', 'std']))
    # print(t2 - t1)
    # sys.exit()

    # ---

    for start_year in start_years:
        if start_years.index(start_year) == (len(start_years) - 1):
            end_year = df_dt['year'].max()
        else:
            end_year = start_year + (n_years - 1)

        for season in range(1, 12 + 1):
            df_sd1 = df_sd.loc[
                (df_sd['season'] == season) & (df_sd['year'] >= start_year) & (df_sd['year'] <= end_year)
                ]
            storm_id = df_sd1['storm_id'].values
            storm_depth = df_sd1['storm_depth'].values
            storm_duration = df_sd1['storm_duration'].values  # 03/10/2022
            tmp_id = np.arange(storm_id.shape[0])
            years = df_sd1['year'].values

            # ***
            # 28/09/2022
            # print(start_year, end_year, season)
            # ***

            # print(df_sd1.loc[df_sd1['year']==25])
            # sys.exit()

            delta = parameters.loc[parameters['season'] == season, 'delta'].values[0]

            # if delta > delta_threshold:  # 09/10/2022 - testing not shuffling if delta ~ zero (nsrp15)
            if delta > -1.0:  # TODO: Revert to line above after tesing

                # !221025 - original
                # storm_id2, storm_depth2, storm_duration2 = shuffle_storms(
                #     tmp_id, storm_id, storm_depth, storm_duration, delta, random_seeds[season-1]
                # )

                # !221025 - numba
                # t1 = datetime.datetime.now()
                n_storms = storm_id.shape[0]
                storm_id2 = np.zeros(storm_id.shape[0], dtype=int) - 999
                storm_depth2 = np.zeros(storm_id.shape[0], dtype=float) - 999.0
                storm_duration2 = np.zeros(storm_id.shape[0], dtype=float) - 999.0
                shuffled = np.zeros(storm_id.shape[0], dtype=int)
                # rng = np.random.default_rng(random_seeds[0])  # !221025 - passed in as argument
                idx = 0
                random_numbers = rng.uniform(0.0, 1.0, n_storms)
                if delta > delta_threshold:  # TESTING - 20m/20n
                    storm_id2, storm_depth2, storm_duration2 = shuffle_storms_numba2(
                        tmp_id, storm_id, storm_depth, storm_duration, delta, n_storms, storm_id2, storm_depth2,
                        storm_duration2,
                        shuffled, idx, random_numbers
                    )
                else:  # TESTING - 20m/20n
                    storm_id2 = storm_id
                    storm_depth2 = storm_depth
                    storm_duration2 = storm_duration
                # t2 = datetime.datetime.now()
                # print(t2 - t1)

                # !221025 - numba again - initialising arrays inside numba - NO SPEED UP ACHIEVED
                # t1 = datetime.datetime.now()
                # n_storms = storm_id.shape[0]
                # # rng = np.random.default_rng(random_seeds[0])  # !221025 - passed in as argument
                # idx = 0
                # random_numbers = rng.uniform(0.0, 1.0, n_storms)
                # storm_id2, storm_depth2, storm_duration2 = shuffle_storms_numba3(
                #     tmp_id, storm_id, storm_depth, storm_duration, delta, n_storms, idx, random_numbers
                # )
                # t2 = datetime.datetime.now()
                # print(t2 - t1)

                df = pd.DataFrame({'year': years, 'storm_depth': storm_depth2})
                df = df.groupby(['year'])['storm_depth'].sum()

                # Reorder months according to a "coarse" model - original white noise approach [KEEP HERE FOR NOW]
                # df = df.to_frame()
                # df.reset_index(inplace=True)
                # df['bc_depth'], _ = scipy.stats.boxcox(df['storm_depth'])
                # df['year_rank_orig'] = scipy.stats.rankdata(df['bc_depth'], method='ordinal')
                # year_new_rank_seq = scipy.stats.rankdata(
                #     rng.normal(size=df['year_rank_orig'].shape[0]), method='ordinal'
                # )

                # TODO: Reinstate AR1 model after testing

                # Reorder months according to a "coarse" model - AR1 model approach
                # df = df.to_frame()
                # df.reset_index(inplace=True)
                # df['year_rank_orig'] = scipy.stats.rankdata(df['storm_depth'], method='ordinal')
                # year_new_rank_seq = df_ar1.loc[df_ar1['season'] == season, 'rank'].values

                # ***
                # TESTING (20p) - no monthly reordering
                df = df.to_frame()
                df.reset_index(inplace=True)
                df['year_rank_orig'] = scipy.stats.rankdata(df['storm_depth'], method='ordinal')
                year_new_rank_seq = df['year_rank_orig'].values
                # ***

                # Find new position that each year should be moved to
                x = year_new_rank_seq
                y = df['year_rank_orig'].values
                xsorted = np.argsort(x)
                ypos = np.searchsorted(x[xsorted], y)
                df['year_rank_new'] = xsorted[ypos] + 1

                # # df.sort_values('rank_2', inplace=True)
                # df['year_2'] = df['year'].values[[df['rank_2'].values - 1]]  # this is the key column
                # df['bc_depth_2'] = df['bc_depth'].values[[df['rank_2'].values - 1]]

                # sort df_sd1 / storm_id using year_2
                # - means first making a year_2 that has an entry per storm - reorder years?
                # -- can map years to a new array, put in dataframe with storm_id
                # - can we go straight to the raincells df... maybe not...

                df_sd2 = df_sd1[['year', 'season', 'storm_arrival', 'storm_arrival_mon']].copy()
                df_sd2['storm_id'] = storm_id2

                # print(df_sd2.loc[df_sd2['year'] == 25, ['storm_id', 'storm_arrival_mon']])
                # sys.exit()

                df_sd2 = pd.merge(df_sd2, df[['year', 'year_rank_orig', 'year_rank_new']])

                # print('--')
                # print(df_sd2.loc[df_sd2['year'] == 25, ['storm_id', 'storm_arrival_mon']])
                # # sys.exit()

                # Correspondence between storm_id and storm_arrival_mon is preserved if just sort on year_rank_new, but
                # order in df preserved if also include storm_arrival_mon in sort
                df_sd2.sort_values(['year_rank_new', 'storm_arrival_mon'], inplace=True)

                # print(df_sd2.loc[df_sd2['year'] == 25, ['storm_id', 'storm_arrival_mon']])
                # sys.exit()

                # df_sd2['year'] = df_sd2['year'].values[[df_sd2['year_2'].values - 1]]
                # df_sd2['storm_arrival'] = df_sd2['storm_arrival'].values[[df_sd2['year_2'].values - 1]]
                # df_sd2['storm_id2'] = df_sd2['storm_id2'].values[[df_sd2['year_2'].values - 1]]

                # TODO: Remove storm_rank_new if no longer needed...
                df_sd2['storm_rank_new'] = np.arange(1,
                                                     df_sd2.shape[0] + 1)  # only new rank for the current season/month
                # def f(x):
                #     return np.arange(1, x.shape[0] + 1)
                # df_sd2['storm_rank_new'] = df_sd2.groupby(['year'])['year'].transform(f)

                # print(df_sd2[['year', 'storm_id', 'storm_rank_new', 'storm_arrival', 'storm_arrival_mon']])
                # sys.exit()

                # ///

                # # Sorting raincell df - rearrange storms according to storm shuffling first
                # df_rc1 = df_rc.loc[df_rc['season'] == season]
                # df_rc1 = pd.merge(df_rc1, df_sd2[['storm_id', 'storm_rank_new']])
                # df_rc1.sort_values('storm_rank_new', inplace=True)
                #
                # # print(df)
                # # print(df_rc1[['storm_id', 'year', 'storm_rank_new', 'storm_arrival_mon']])
                # # print(df_sd2[['storm_id', 'year', 'storm_arrival', 'storm_arrival_mon']])
                # # print(df_sd2.head(24))
                # # print('--')
                # # print(df_sd2.tail(24))
                # # sys.exit()
                #
                # # Then rearrange years
                # df_rc1 = pd.merge(df_rc1, df[['year', 'year_rank_orig', 'year_rank_new']])
                # df_rc1.sort_values('year_rank_new', inplace=True)  # gives equivalent to year_new
                # # df_rc1['year_new'] = df_rc1['year'].ne(df_rc1['year'].shift()).cumsum()  # just use year_rank_new instead
                #
                # # print(df_rc1[['storm_id', 'year', 'year_rank_orig', 'year_rank_new', 'year_new']])
                # # sys.exit()
                #
                # # for i in range(10):
                # #     print(tmp_id[i], storm_id[i], storm_id2[i], years[i])
                # # print(df_sd1)
                # # print(df_sd1.columns)
                # # sys.exit()

                # ///

                # !! Above may be wrong because everything in df_rc1 is getting sorted - storm and raincell arrival times
                # should stay the same at this point !!
                # - would it better to do a join/merge operation?
                # -- know the final order of the original storm ids (storm_id column of df_sd2)
                # -- therefore join df_rc1 (storm id + raincell properties) to df_sd2 based on storm_id
                # -- df_sd2 needs to provide storm arrival time (relative to month origin)

                # Sorting raincell df - rearrange storms according to storm shuffling first
                df_rc1 = df_rc.loc[
                    (df_rc['season'] == season) & (df_rc['year'] >= start_year) & (df_rc['year'] <= end_year)
                    ]
                df_rc1 = pd.merge(
                    df_sd2[['year_rank_new', 'season', 'storm_id', 'storm_arrival_mon']],
                    df_rc1[
                        ['storm_id', 'raincell_arrival_mon', 'raincell_duration', 'raincell_end_mon',
                         'raincell_intensity',
                         'raincell_depth']
                    ]
                )

                # !! LATE ON 03/07/2022 TESTING !!
                df_rc1['year_rank_new'] += (start_year - 1)
                # if start_year > 1:
                #     print(start_year)
                #     print(df_rc1[['year_rank_new', 'storm_id']])
                #     sys.exit()
                # !! LATE ON 03/07/2022 TESTING !!

                # print(df_rc1[['year_rank_new', 'season', 'storm_id', 'storm_arrival_mon']].head(24))  # , 'raincell_arrival_mon'
                # print(df_rc1[['year_rank_new', 'season', 'storm_id', 'storm_arrival_mon']].tail(24))  # , 'raincell_arrival_mon'
                # sys.exit()

                # sd_dfs.append(df_sd2)
                rc_dfs.append(df_rc1)

                # print(df_rc1)
                # print(df_rc1.columns)
                # import sys
                # sys.exit()

            # if delta ~ zero
            else:
                df_rc1 = df_rc.loc[
                    (df_rc['season'] == season) & (df_rc['year'] >= start_year) & (df_rc['year'] <= end_year)
                    ].copy()
                df_rc1['year_rank_new'] = df_rc1['year'].values
                df_rc1['year_rank_new'] += (start_year - 1)
                df_rc1.drop(columns=['year'], inplace=True)
                df_rc1 = df_rc1[[
                    'year_rank_new', 'season', 'storm_id', 'storm_arrival_mon', 'raincell_arrival_mon',
                    'raincell_duration', 'raincell_end_mon', 'raincell_intensity', 'raincell_depth'
                ]]
                rc_dfs.append(df_rc1)

                # print(df_rc1)
                # print(df_rc1.columns)
                # print(start_year, end_year)
                # import sys
                # sys.exit()

                # print(df_rc1[['year', 'storm_arrival_mon']])
                # import sys
                # sys.exit()

                # for positive delta df_rc1 has the following columns:
                # 'year_rank_new', 'season', 'storm_id', 'storm_arrival_mon',
                # 'raincell_arrival_mon', 'raincell_duration', 'raincell_end_mon',
                # 'raincell_intensity', 'raincell_depth'

    # Concatenate dfs and sort out order
    df_rc2 = pd.concat(rc_dfs)
    # print(df_rc2.iloc[1000:1010, ])  # ['storm_id', 'year', 'year_rank_orig', 'year_rank_new', 'season']
    # print('--')
    # df_rc2.to_csv('H:/Projects/rwgen/working/iss13/nsrp/df_rc2.csv')

    # print(df_rc2)
    # print(df_rc2.columns)
    # print('--')
    # print(df_rc)
    # print(df_rc.columns)
    # import sys
    # sys.exit()

    # # df_rc2.sort_values(['year_rank_new', 'season'], inplace=True)
    df_rc2.sort_values(['year_rank_new', 'season', 'storm_arrival_mon', 'raincell_arrival_mon'], inplace=True)

    # print(df_rc2.iloc[1000:1010, ])  # ['storm_id', 'year', 'year_rank_orig', 'year_rank_new', 'season']
    # df_rc2.to_csv('H:/Projects/rwgen/working/iss13/nsrp/df_rc2_sorted.csv')

    # print(df_rc2[['year_rank_new', 'season', 'storm_id', 'storm_arrival_mon', 'raincell_arrival_mon']])
    # print(df_rc2.loc[df_rc2['year_rank_new'] == 10,
    #                  ['year_rank_new', 'season', 'storm_id', 'storm_arrival_mon', 'raincell_arrival_mon']])
    # sys.exit()

    # Also get storm/raincell times relative to storm origin
    # df_rc2['start_time'] = df_rc['start_time'].values  # overwrites shuffled month start times - WRONG
    df_rc2.rename(columns={'year_rank_new': 'year'}, inplace=True)
    # print(df_rc2.shape)
    df_rc2 = pd.merge(df_rc2, df_dt[['year', 'season', 'start_time']])
    df_rc2['month'] = df_rc2['season'].values  # TODO: Allow shuffling to work on seasons rather than just months

    # print(df_dt.loc[df_dt['season'] == season, ['year', 'season', 'start_time']])

    # print(df_rc2[['start_time', 'storm_arrival', 'storm_arrival_mon']])
    # sys.exit()

    df_rc2['storm_arrival'] = df_rc2['storm_arrival_mon'] + df_rc2['start_time']
    df_rc2['raincell_arrival'] = df_rc2['raincell_arrival_mon'] + df_rc2['start_time']
    df_rc2['raincell_end'] = df_rc2['raincell_end_mon'] + df_rc2['start_time']

    # print(df_rc2.shape)
    # sys.exit()

    # df_rc2.to_csv('H:/Projects/rwgen/working/iss13/nsrp/df_rc2_1b.csv')
    # import sys
    # sys.exit()

    # Drop columns not needed in discretisation
    # df_rc2.drop(columns=[
    #     'year', 'start_time', 'storm_arrival_mon', 'raincell_arrival_mon', 'raincell_end_mon', 'storm_rank_new',
    #     'year_rank_orig', 'year_rank_new'
    # ])

    # print(df_rc2.columns)

    # sys.exit()

    # t2 = datetime.datetime.now()
    # print(t2 - t1)
    # sys.exit()

    return df_rc2


# ---


# def delta_objective_func(delta, tmp_id, storm_id, storm_depth, years, season_ref):
#     storm_id2, storm_depth2 = shuffle_storms2(tmp_id, storm_id, storm_depth, delta)
#
#     # numba test
#     # storm_id2 = np.zeros(storm_id.shape[0], dtype=int) - 999
#     # storm_depth2 = np.zeros(storm_id.shape[0]) - 999
#     # shuffled = np.zeros(storm_id.shape[0], dtype=int)  # , dtype=int
#     # storm_id2, storm_depth2 = shuffle_storms3(tmp_id, storm_id, storm_depth, delta)
#
#     df = pd.DataFrame({'year': years, 'storm_depth': storm_depth2})
#     df = df.groupby(['year'])['storm_depth'].sum()
#     var = np.var(df.values, ddof=1)  # matching current reference
#
#     # TESTING - original and shuffled variance
#     # df = pd.DataFrame({'year': years, 'storm_depth1': storm_depth, 'storm_depth2': storm_depth2})
#     # df = df.groupby(['year']).agg({'storm_depth1': sum, 'storm_depth2': sum})
#     # var1 = np.var(df['storm_depth1'].values, ddof=1)  # matching current reference
#     # var2 = np.var(df['storm_depth2'].values, ddof=1)  # matching current reference
#     # print(float(season_ref), float(var1), float(var2))
#
#     # print(float(season_ref), float(var), abs(float(season_ref) - float(var)))
#
#     return float(season_ref) - float(var), var


# def test(df):
#     # df.to_csv('H:/Projects/rwgen/working/iss13/stnsrp/df_6a.csv')
#
#     dfs = []
#     for season in range(1, 12+1):
#         df1 = df.loc[df['season'] == season]
#         storm_id = df1['storm_id'].values
#         storm_depth = df1['storm_depth'].values
#         tmp_id = np.arange(storm_id.shape[0])
#         storm_id2, storm_depth2 = shuffle_storms2(tmp_id, storm_id, storm_depth, 1)
#         df2 = df1.copy()
#         df2['storm_id2'] = storm_id2
#         df2['storm_depth2'] = storm_depth2
#         dfs.append(df2)
#     df3 = pd.concat(dfs)
#     df3.sort_values('storm_arrival', inplace=True)
#
#     df3.to_csv('H:/Projects/rwgen/working/iss13/stnsrp/df_7b.csv')


# def shuffle_storms(df, delta, rng):
#     # Using dataframe operations (slower) and without month-wise stratification
#     # df.to_csv('H:/Projects/rwgen/working/iss13/stnsrp/df_5a.csv')
#
#     # Initialise columns to store shuffled storm ID and depth - plus flag for whether already shuffled
#     df['storm_id2'] = -999
#     df['storm_depth2'] = -999
#     df['shuffled'] = 0
#
#     # First storm selection is arbitrary
#     df1 = df.loc[df['season'] == df['season'].values[0]]
#     idx = rng.integers(0, df1.shape[0] - 1, 1)
#     df.iat[0, df.columns.get_loc('storm_id2')] = df1['storm_id'].values[idx]
#     df.iat[0, df.columns.get_loc('storm_depth2')] = df1['storm_depth'].values[idx]
#     # df.iat[0, df.columns.get_loc('shuffled')] = 1
#     df.iat[int(df1['storm_id'].values[idx]), df.columns.get_loc('shuffled')] = 1
#     prev_storm_depth = df1['storm_depth'].values[idx]
#
#     # Then loop through storms
#     storm_idx = 0
#     for _, row in df.iterrows():
#         # print(storm_idx)
#         if row['storm_depth2'] != -999:
#             pass
#         else:
#             df1 = df.loc[(df['season'] == row['season']) & (df['shuffled'] == 0)].copy()
#             df1['si'] = (1.0 / np.absolute(np.log(df1['storm_depth'] / prev_storm_depth))) ** delta
#             df1['pi'] = (1.0 / np.sum(df1['si'])) * df1['si']
#             idx = rng.choice(df1.shape[0], p=df1['pi'].values)
#             df.iat[storm_idx, df.columns.get_loc('storm_id2')] = df1['storm_id'].values[idx]
#             df.iat[storm_idx, df.columns.get_loc('storm_depth2')] = df1['storm_depth'].values[idx]
#             # df.iat[storm_idx, df.columns.get_loc('shuffled')] = 1
#             df.iat[int(df1['storm_id'].values[idx]), df.columns.get_loc('shuffled')] = 1
#             prev_storm_depth = df1['storm_depth'].values[idx]
#
#         storm_idx += 1
#
#     # df.to_csv('H:/Projects/rwgen/working/iss13/stnsrp/df_5b1.csv')
#     # sys.exit()
#
#     return df


@numba.jit(nopython=True)
def shuffle_storms_numba(
        tmp_id, storm_id, storm_depth, delta, n_storms, storm_id2, storm_depth2, shuffled, idx, random_numbers
):
    # Assumes that already subset on season

    # rng = np.random.default_rng()

    # ! n_storms = storm_id.shape[0]

    # storm_id = storm_id.astype(numba.int64)
    # storm_depth = storm_depth.astype(numba.float64)

    # Initialise columns to store shuffled storm ID and depth - plus flag for whether already shuffled
    # ! storm_id2 = np.zeros(n_storms, dtype=numba.int64) - 999
    # ! storm_depth2 = np.zeros(n_storms) - 999
    # ! shuffled = np.zeros(n_storms, dtype=numba.int64)  # , dtype=int

    # First storm selection is arbitrary
    # idx = rng.integers(0, storm_id.shape[0] - 1, 1)
    # ! idx = np.random.randint(0, n_storms - 1, 1)
    # ! idx = idx[0]
    storm_id2[0] = storm_id[idx]
    storm_depth2[0] = storm_depth[idx]
    shuffled[idx] = 1
    prev_storm_depth = storm_depth[idx]

    # Then loop through storms
    for storm_idx in range(n_storms):
        if storm_depth2[storm_idx] != -999:
            pass
        else:
            # print(storm_idx)
            mask = shuffled == 0
            tmp_id1 = tmp_id[mask]
            id1 = storm_id[mask]
            # print(id1)
            dep1 = storm_depth[mask]
            # print(id1[:5])
            # print(dep1[:5])
            si = (1.0 / np.absolute(np.log(dep1 / prev_storm_depth))) ** delta
            pi = (1.0 / np.sum(si)) * si
            # idx = rng.choice(id1.shape[0], p=pi)

            # ! idx = np.searchsorted(np.cumsum(pi), np.random.random(), side="right")
            idx = np.searchsorted(np.cumsum(pi), random_numbers[storm_idx], side="right")

            # print(np.searchsorted(np.cumsum(pi), 0.1, side="right"))
            # print(idx, storm_depth[idx], dep1[idx])
            # print(storm_depth[:5])
            # print(dep1[:5])
            storm_id2[storm_idx] = id1[idx]
            storm_depth2[storm_idx] = dep1[idx]
            # print(storm_id2[:5])
            # print(storm_depth2[:5])
            # print(tmp_id1[idx])
            # shuffled[int(tmp_id1[idx])] = 1
            shuffled[tmp_id1[idx]] = 1
            # print(shuffled)
            prev_storm_depth = dep1[idx]

        storm_idx += 1

    return storm_id2, storm_depth2


# !221025
@numba.jit(nopython=True)
def shuffle_storms_numba2(
        tmp_id, storm_id, storm_depth, storm_duration, delta, n_storms, storm_id2, storm_depth2, storm_duration2,
        shuffled, idx, random_numbers
):
    # Assumes that already subset on season

    # First storm selection is arbitrary
    storm_id2[0] = storm_id[idx]
    storm_depth2[0] = storm_depth[idx]
    storm_duration2[0] = storm_duration[idx]
    shuffled[idx] = 1
    prev_storm_depth = storm_depth[idx]

    # Then loop through storms
    for storm_idx in range(n_storms):
        if storm_depth2[storm_idx] != -999:
            pass
        else:
            mask = shuffled == 0
            tmp_id1 = tmp_id[mask]
            id1 = storm_id[mask]
            dep1 = storm_depth[mask]
            dur1 = storm_duration[mask]
            si = (1.0 / np.absolute(np.log(dep1 / prev_storm_depth))) ** delta
            pi = (1.0 / np.sum(si)) * si

            idx = np.searchsorted(np.cumsum(pi), random_numbers[storm_idx], side="right")

            storm_id2[storm_idx] = id1[idx]
            storm_depth2[storm_idx] = dep1[idx]
            storm_duration2[storm_idx] = dur1[idx]

            # shuffled[int(tmp_id1[idx])] = 1
            shuffled[tmp_id1[idx]] = 1

            prev_storm_depth = dep1[idx]

        storm_idx += 1

    return storm_id2, storm_depth2, storm_duration2

# !221025 - no speed up from initialising arrays inside numba function
# @numba.jit(nopython=True)
# def shuffle_storms_numba3(
#         tmp_id, storm_id, storm_depth, storm_duration, delta, n_storms, idx, random_numbers
# ):
#     # Assumes that already subset on season
#     storm_id2 = np.zeros(storm_id.shape[0], dtype=numba.int64) - 999
#     storm_depth2 = np.zeros(storm_id.shape[0], dtype=numba.float64) - 999.0
#     storm_duration2 = np.zeros(storm_id.shape[0], dtype=numba.float64) - 999.0
#     shuffled = np.zeros(storm_id.shape[0], dtype=numba.int32)
#
#     # First storm selection is arbitrary
#     storm_id2[0] = storm_id[idx]
#     storm_depth2[0] = storm_depth[idx]
#     storm_duration2[0] = storm_duration[idx]
#     shuffled[idx] = 1
#     prev_storm_depth = storm_depth[idx]
#
#     # Then loop through storms
#     for storm_idx in range(n_storms):
#         if storm_depth2[storm_idx] != -999:
#             pass
#         else:
#             mask = shuffled == 0
#             tmp_id1 = tmp_id[mask]
#             id1 = storm_id[mask]
#             dep1 = storm_depth[mask]
#             dur1 = storm_duration[mask]
#             si = (1.0 / np.absolute(np.log(dep1 / prev_storm_depth))) ** delta
#             pi = (1.0 / np.sum(si)) * si
#
#             idx = np.searchsorted(np.cumsum(pi), random_numbers[storm_idx], side="right")
#
#             storm_id2[storm_idx] = id1[idx]
#             storm_depth2[storm_idx] = dep1[idx]
#             storm_duration2[storm_idx] = dur1[idx]
#
#             # shuffled[int(tmp_id1[idx])] = 1
#             shuffled[tmp_id1[idx]] = 1
#
#             prev_storm_depth = dep1[idx]
#
#         storm_idx += 1
#
#     return storm_id2, storm_depth2, storm_duration2

# -------------------------------------------------------------------------------------------------

# !221113 - TESTING - trying to do shuffling based on si similarity metric that explicitly includes each point


def _fit_delta_si(
        spatial_model, bounds, df1, dc1, ref_var, n_shuffles, random_seed, n_divisions, ref_dur, q=None
):
    # Fit delta for one realisation
    # - df1 is df_wd (i.e. depths by window)
    # - dc1 contains a df with depths by window for each point if spatial model

    # TODO: DO NOT FORGET ABOUT PHI WHEN COMPARING OBSERVED AND SIMULATED VARIANCES

    if not spatial_model:
        n_points = 1
    else:
        n_points = len(dc1.keys())
        tmp_res = np.zeros((12, n_points))  # for calculating objective function

    # Initialisation - errors is for storing variance residuals
    rng = np.random.default_rng(random_seed)
    deltas_to_test = np.arange(bounds[0], bounds[1] + 0.00001, 0.1)  # 0.05)
    errors = np.zeros((12, n_shuffles, deltas_to_test.shape[0]))  # 12 months

    # Get window properties as arrays so numba can be used
    n_windows = df1.shape[0]
    win_id = df1['win_id'].values
    win_month = df1['month'].values
    win_length = df1['win_length'].values
    win_depth = np.zeros((n_points, n_windows))
    if not spatial_model:
        win_depth[0, :] = df1['win_depth'].values
    else:
        for point_id in range(1, n_points + 1):
            win_depth[point_id-1, :] = dc1[point_id]['win_depth'].values

    # Loop shuffles and deltas to trial
    for i in range(n_shuffles):
        j = 0
        for _delta in deltas_to_test:
            delta = np.repeat(_delta, 12)

            # Shuffle windows
            random_numbers = rng.uniform(0.0, 1.0, n_windows)
            win_length = win_length.astype(np.float32)  # TESTING
            win_depth = win_depth.astype(np.float32)  # TESTING
            random_numbers = random_numbers.astype(np.float32)  # TESTING
            delta = delta.astype(np.float32)  # TESTING
            win_id2, win_depth2 = _shuffle_windows2_si(
                win_id, win_month, win_length, win_depth, delta, n_windows, random_numbers, n_divisions,
            )

            # Calculate variance from shuffled series and variance residual
            if not spatial_model:  # TODO: If using si modifications update point model stuff too - just use below?
                df2 = df1.copy()
                df2['win_id'] = win_id2
                df2['win_depth'] = win_depth2
                df2 = df2.groupby(['year', 'month'])['win_depth'].sum()
                df2 = df2.to_frame('win_depth')
                df2.reset_index(inplace=True)
                df2 = df2.groupby('month')['win_depth'].var()
                # TODO: Update point model objective function in line with spatial (or vice versa)
                res = ref_var.loc[ref_var['name'] == 'variance', 'value'].values - df2.values
            else:
                tmp_res.fill(0.0)
                for point_id in range(1, n_points+1):
                    df2 = pd.DataFrame({'win_id': win_id2, 'win_depth': win_depth2[point_id-1, :]})
                    df2['year'] = df1['year'].values
                    df2['month'] = df1['month'].values
                    df2 = df2.groupby(['year', 'month'])['win_depth'].sum()
                    df2 = df2.to_frame('win_depth')
                    df2.reset_index(inplace=True)
                    df2 = df2.groupby('month')['win_depth'].agg(['mean', 'var'])  # assume 1-12 order maintained
                    phi = ref_var.loc[(ref_var['name'] == 'variance') & (ref_var['point_id'] == point_id), 'phi'].values
                    _sim_var = df2['var'].values * phi ** 2.0
                    _ref_gs = ref_var.loc[
                        (ref_var['name'] == 'variance') & (ref_var['point_id'] == point_id), 'gs'
                    ].values[0]
                    _ref_var = ref_var.loc[
                        (ref_var['name'] == 'variance') & (ref_var['point_id'] == point_id), 'value'
                    ].values

                    # !221113 - testing an objective function without scaling in 013
                    tmp_res[:, point_id-1] = (1.0 / (_ref_gs / 100.0)) * (_ref_var - _sim_var)  # pre-013 - RESTORE
                    # tmp_res[:, point_id - 1] = _ref_var - _sim_var  # 013

                    # !221113 - testing use of coefficient of variance as OF in 014
                    # _ref_mean = ref_var.loc[
                    #     (ref_var['name'] == 'mean') & (ref_var['point_id'] == point_id), 'value'
                    # ].values.copy()
                    # _ref_mean *= np.array([31.0, 28.25, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0])
                    # _sim_mean = df2['mean'].values * phi
                    # ref_cv = _ref_var ** 0.5 / _ref_mean
                    # sim_cv = _sim_var ** 0.5 / _sim_mean
                    # tmp_res[:, point_id - 1] = ref_cv - sim_cv

                res = np.sum(tmp_res, axis=1)
                # print(res)

            # Store variance residual
            errors[:, i, j] = res
            j += 1

    # Find delta for each month
    tmp = []
    for month in range(1, 12+1):
        average_errors = np.median(errors[month-1, :, :], axis=0)  # by test delta value

        if np.max(average_errors) < 0:
            delta = 0.0
        else:
            smoothed_errors = np.zeros(average_errors.shape[0])  # by test delta value
            for i in range(2, average_errors.shape[0] - 1):
                smoothed_errors[i] = np.mean(average_errors[i - 2:i + 3])
            smoothed_errors[0] = average_errors[0]
            smoothed_errors[1] = np.mean([smoothed_errors[0], smoothed_errors[2]])
            smoothed_errors[-1] = average_errors[-1]
            smoothed_errors[-2] = np.mean([smoothed_errors[-1], smoothed_errors[-3]])

            f = scipy.interpolate.interp1d(smoothed_errors, deltas_to_test)
            try:
                delta = f(0.0)
            except:
                delta = -999.0

        tmp.append(delta)

    delta = np.asarray(tmp)

    return delta


# @numba.jit(nopython=True)
def _shuffle_windows2_si(win_id, win_month, win_length, win_depth, delta, n_windows, random_numbers, n_divisions):
    # TODO: Remove hardcoding of window sizes - n_divisions needs to be passed in accordingly
    # TODO: Try going back to float64 everywhere - may be fast enough now

    # Initialisation - careful of types if using numba
    win_id2 = np.zeros(n_windows, dtype=int) - 999
    n_points = win_depth.shape[0]  # means that single site needs to have dimensions (1, n_windows)
    win_depth2 = np.zeros((n_points, n_windows), dtype=np.float32) - 999.0

    # Prepare to jitter to avoid divide by zero etc errors
    win_depth[win_depth < 0.01] = 0.01
    noise = random_numbers / 100.0
    win_depth += (noise / 10.0)
    win_depth += (noise[::-1] / 10.0)

    # First storm selection is arbitrary
    idx = 0
    win_id2[0] = win_id[idx]
    win_depth2[:, 0] = win_depth[:, idx]
    prev_depth = np.zeros((n_points, 1))
    prev_depth[:, 0] = win_depth[:, idx]

    # Initialise masks - modified in loop to help guide selection
    mask = np.zeros((12, n_windows), dtype=bool)
    mask2 = np.zeros(n_windows, dtype=bool)  # for february leap years
    for month in range(1, 12+1):
        if month != 2:
            mask[month-1, :] = win_month == month
        else:
            mask[month-1, :] = (win_month == month) & (win_length == ((28 * 24) / float(n_divisions)))
    mask[0, 0] = False  # i.e. first window in first month set above
    mask2[:] = (win_month == 2) & (win_length == ((29 * 24) / float(n_divisions)))

    # Then loop through storms
    for win_idx in range(n_windows):

        month = win_month[win_idx]
        len_ = win_length[win_idx]

        # if win_depth2[win_idx] != -999.0:
        if win_idx == 0:
            pass
        else:

            # TESTING
            if len_ != ((29 * 24) / float(n_divisions)):
                id1 = win_id[mask[month-1]]
                dep1 = win_depth[:, mask[month-1]]
            else:
                id1 = win_id[mask2]
                dep1 = win_depth[:, mask2]

            # Check that no identical current and previous depths - lead to divide by zero error
            if np.any(dep1 / prev_depth == 1.0):
                tmp = dep1 / prev_depth == 1.0
                tmp = np.max(tmp, axis=1)
                prev_depth[tmp, 0] += noise[win_idx]
                ci = 0
                while np.any(dep1 / prev_depth == 1.0):
                    prev_depth[tmp, 0] += (noise[win_idx] / 10.0)
                    if ci == 10:
                        break

            # Similarity index and associated probability of selection
            # si = (1.0 / np.absolute(np.log(dep1 / prev_depth))) ** delta[month - 1]  # original - _shuffle_windows2()
            si = np.sum(1.0 / np.absolute(np.log(dep1 / prev_depth)), axis=0) ** delta[month - 1]
            pi = (1.0 / np.sum(si)) * si

            # Choose an index (window)
            idx = np.searchsorted(np.cumsum(pi), random_numbers[win_idx], side="right")  # testing
            idx = min(idx, id1.shape[0] - 1)  # out-of-bounds errors possible if moving to float32

            # Store selection
            win_id2[win_idx] = id1[idx]
            win_depth2[:, win_idx] = dep1[:, idx]

            # Update mask so that the window just used can no longer be selected
            if len_ != ((29 * 24) / float(n_divisions)):
                mask[month-1, id1[idx]] = 0
            else:
                mask2[id1[idx]] = 0

            # Prepare for next window
            prev_depth[:, 0] = dep1[:, idx]

        win_idx += 1

    return win_id2, win_depth2


def _shuffle_simulation_si(spatial_model, df, df1, dc1, parameters, datetime_helper, rng, n_divisions, do_reordering):
    # - df is raincells
    # - df1 is window depths
    # - do_reordering is currently boolean for monthly ar1 model (or not)

    if not spatial_model:
        n_points = 1
    else:
        n_points = len(dc1.keys())

    # Get window properties as arrays so numba can be used
    n_windows = df1.shape[0]
    win_id = df1['win_id'].values
    win_month = df1['month'].values
    win_length = df1['win_length'].values
    win_depth = np.zeros((n_points, n_windows))
    if not spatial_model:
        win_depth[0, :] = df1['win_depth'].values
    else:
        for point_id in range(1, n_points + 1):
            win_depth[point_id - 1, :] = dc1[point_id]['win_depth'].values

    # Shuffle windows
    delta = parameters['delta'].values
    random_numbers = rng.uniform(0.0, 1.0, n_windows)
    win_length = win_length.astype(np.float32)  # TESTING
    win_depth = win_depth.astype(np.float32)  # TESTING
    random_numbers = random_numbers.astype(np.float32)  # TESTING
    delta = delta.astype(np.float32)  # TESTING
    win_id2, win_depth2 = _shuffle_windows2_si(
        win_id, win_month, win_length, win_depth, delta, n_windows, random_numbers, n_divisions,
    )

    # New windows df
    df2 = pd.DataFrame({'win_id': win_id2})
    df2['new_rank'] = np.arange(df2.shape[0])
    tmp = df1[['win_id', 'year']].copy()
    tmp.rename(columns={'win_id': 'new_rank'}, inplace=True)
    df2 = df2.merge(tmp)

    # Merge new ranks into raincells df and sort on order
    df.sort_values(['win_id', 'raincell_arrival'], inplace=True)
    df.drop(columns=['year'], inplace=True)  # TESTING
    df = df.merge(df2[['win_id', 'new_rank', 'year']])
    df.sort_values(['new_rank', 'raincell_arrival'], inplace=True)
    df['season'] = df['month']

    # Merge serial window start times into df in order to adjust times after shuffling
    # - i.e. adjusted from original time relative to simulation origin to new time relative to storm origin (after
    # shuffling has taken place by reordering df)
    tmp = df1.loc[:, ['win_id', 'win_start']].copy()
    tmp.rename(columns={'win_id': 'new_rank', 'win_start': 'new_win_start'}, inplace=True)
    df = df.merge(tmp)
    time_adj = df['win_start'].values - df['new_win_start'].values
    df['storm_arrival'] -= time_adj
    df['raincell_arrival'] -= time_adj
    df['raincell_end'] -= time_adj
    df.drop(columns=['win_start'], inplace=True)
    df.rename(columns={'new_win_start': 'win_start'}, inplace=True)

    # Reordering via monthly AR1 model optional for now
    if do_reordering:

        # TODO: If using this si approach then need to work with new win_depth2 array (n_points, n_windows)

        # Simulate with AR1 model
        df3 = _simulate_ar1(parameters, df1['year'].min(), df1['year'].max(), rng)
        df3.rename(columns={'season': 'month'}, inplace=True)
        df3 = df3.merge(datetime_helper[['year', 'month', 'start_time']])
        df3.rename(columns={'start_time': 'new_month_start'}, inplace=True)
        df3['new_month_rank'] = -999
        for month in range(1, 12+1):
            df3.loc[df3['month'] == month, 'new_month_rank'] = np.arange(df1['year'].max() - df1['year'].min() + 1) + 1

        # Original month start times
        df = df.merge(
            datetime_helper.loc[
                (datetime_helper['year'] >= df1['year'].min())
                & (datetime_helper['year'] <= df1['year'].max()),
                ['year', 'month', 'start_time']
            ]
        )
        df.rename(columns={'start_time': 'month_start'}, inplace=True)

        # Also need to get current rank of month (re depth) - ensuring based on serially complete set of windows
        tmp = df.groupby(['year', 'month'])['raincell_depth'].sum()
        tmp = tmp.to_frame('depth')
        tmp.reset_index(inplace=True)
        tmp = tmp.merge(
            datetime_helper.loc[
                (datetime_helper['year'] >= df1['year'].min())
                & (datetime_helper['year'] <= df1['year'].max()),
                ['year', 'month']
            ],
            how='outer'
        )
        tmp['depth'] = np.where(~np.isfinite(tmp['depth']), 0.0, tmp['depth'])
        tmp['leap_feb'] = [utils.check_if_leap_year(year) for year in tmp['year']]
        tmp.loc[tmp['month'] != 2, 'leap_feb'] = 0
        tmp['month_rank'] = tmp.groupby(['month'])['depth'].rank(method='first', ascending=False).astype(int)
        tmp.sort_values(['year', 'month'], inplace=True)

        # Adjust ranks in reordering df so leap years match up with what is available in the simulation
        tmp['ly_flag'] = 0
        for year in range(df1['year'].min(), df1['year'].max()):
            leap_feb_ranks = tmp.loc[(tmp['ly_flag'] == 0) & (tmp['leap_feb'] == 1), 'month_rank'].values
            nonleap_feb_ranks = tmp.loc[
                (tmp['ly_flag'] == 0) & (tmp['month'] == 2) & (tmp['leap_feb'] == 0), 'month_rank'
            ].values

            ar1_rank = df3.loc[(df3['year'] == year) & (df3['month'] == 2), 'month_rank'].values[0]
            sim_year = tmp.loc[(tmp['month'] == 2) & (tmp['month_rank'] == ar1_rank), 'year'].values[0]

            if utils.check_if_leap_year(year):
                if utils.check_if_leap_year(sim_year):
                    adjust = False
                else:
                    adjust = True
                    new_ar1_rank = leap_feb_ranks[(np.abs(ar1_rank - leap_feb_ranks)).argmin()]
            elif not utils.check_if_leap_year(year):
                if not utils.check_if_leap_year(sim_year):
                    adjust = False
                else:
                    adjust = True
                    new_ar1_rank = nonleap_feb_ranks[(np.abs(ar1_rank - nonleap_feb_ranks)).argmin()]

            if adjust:
                df3.loc[(df3['month'] == 2) & (df3['month_rank'] == new_ar1_rank), 'month_rank'] = ar1_rank
                df3.loc[(df3['year'] == year) & (df3['month'] == 2), 'month_rank'] = new_ar1_rank
                tmp.loc[(tmp['month'] == 2) & (tmp['month_rank'] == new_ar1_rank), 'ly_flag'] = 1

        # Join month ranks
        tmp.drop(columns=['ly_flag'], inplace=True)
        df = df.merge(tmp)
        df.drop(columns='depth', inplace=True)

        # Join AR1 ranks and then reorder raincells df
        df = df.merge(df3[['month', 'month_rank', 'new_month_rank', 'new_month_start']])  # 'year',
        df.sort_values(['month', 'new_month_rank'], inplace=True)  # , 'new_rank'
        df['year'] = (df['new_month_rank'] - 1) + df1['year'].min()
        df.sort_values(['year', 'month', 'new_rank'], inplace=True)

        # Adjust times again so all relative to block origin
        df['storm_arrival'] = df['storm_arrival'] - df['month_start'] + df['new_month_start']
        df['raincell_arrival'] = df['raincell_arrival'] - df['month_start'] + df['new_month_start']
        df['raincell_end'] = df['raincell_end'] - df['month_start'] + df['new_month_start']

    return df
