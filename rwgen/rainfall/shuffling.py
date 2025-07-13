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


def fit_delta(
        spatial_model, parameters, intensity_distribution, point_metadata, n_workers, random_seed,
        reference_statistics, reference_duration, n_divisions, use_pooling,
):
    _point_metadata = point_metadata

    # Bounds and reference for delta
    bounds = [0.0, 1.0]  # TODO: Confirm whether this is the range to stick with - input argument?

    ref_var = reference_statistics.loc[
        ((reference_statistics['name'] == 'variance') & (reference_statistics['duration'] == reference_duration))
        | ((reference_statistics['name'] == 'mean') & (reference_statistics['duration'] == '24H')),
        ['point_id', 'season', 'name', 'value', 'phi', 'gs']
    ].copy()

    ref_var = ref_var.loc[ref_var['point_id'] != -1]  # i.e. not pooled statistics

    ref_var.set_index('season', inplace=True)

    # Prepare to simulate
    realisation_id = 1
    simulation_length = 30
    start_year = 2000
    end_year = start_year + simulation_length - 1
    timestep_length = 24
    calendar = 'gregorian'
    datetime_helper = utils.make_datetime_helper(start_year, end_year, timestep_length, calendar)
    season_definitions = {month: month for month in range(1, 12 + 1)}
    output_types = ['point']
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

            # Simulate realisation to be shuffled and fit delta
            df_wd, dc1, _ = simulation.simulate_realisation(
                realisation_id, datetime_helper, simulation_length, timestep_length, season_definitions,
                spatial_model, output_types, discretisation_metadata, _point_metadata, catchment_metadata,
                parameters, intensity_distribution, rng, xmin, xmax, ymin, ymax, output_paths, block_size,
                block_subset_size, spatial_buffer_factor, simulation_mode,
                weather_model=None, n_divisions=n_divisions, do_reordering=False,
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
                block_subset_size, spatial_buffer_factor, simulation_mode,
                weather_model=None, n_divisions=n_divisions, do_reordering=False,
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
            win_id2, win_depth2 = _shuffle_windows(
                win_id, win_month, win_length, win_depth, delta, n_windows, random_numbers, n_divisions,
            )

            # Calculate variance from shuffled series and variance residual
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
                    df2 = pd.DataFrame({'win_id': win_id2})
                    df2 = df2.merge(dc1[point_id][['win_id', 'win_depth']])
                    df2['year'] = df1['year'].values
                    df2['month'] = df1['month'].values
                    df2 = df2.groupby(['year', 'month'])['win_depth'].sum()
                    df2 = df2.to_frame('win_depth')
                    df2.reset_index(inplace=True)
                    df2 = df2.groupby('month')['win_depth'].agg(['mean', 'var'])  # assume 1-12 order maintained
                    phi = ref_var.loc[(ref_var['name'] == 'variance') & (ref_var['point_id'] == point_id), 'phi'].values
                    _sim_mean = df2['mean'].values * phi
                    _sim_var = df2['var'].values * phi ** 2.0

                    _ref_mean = ref_var.loc[
                        (ref_var['name'] == 'mean') & (ref_var['point_id'] == point_id), 'value'
                    ].values
                    _ref_mean = (
                        _ref_mean * np.array([31.0, 28.25, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0])
                    )
                    _ref_var = ref_var.loc[
                        (ref_var['name'] == 'variance') & (ref_var['point_id'] == point_id), 'value'
                    ].values
                    tmp_res[:, point_id - 1] = _ref_var ** 0.5 / _ref_mean - _sim_var ** 0.5 / _sim_mean

                res = np.sum(tmp_res, axis=1)

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


def get_monthly_series(input_timeseries):
    df = pd.read_csv(
        input_timeseries, names=['value'], index_col=0, skiprows=1, parse_dates=True, infer_datetime_format=True,
        dayfirst=True
    )

    df.loc[df['value'] < 0.0] = np.nan
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


def _shuffle_windows(win_id, win_month, win_length, win_depth, delta, n_windows, random_numbers, n_divisions):
    # Initialisation - careful of types if using numba
    win_id2 = np.zeros(n_windows, dtype=int) - 999
    win_depth2 = np.zeros(n_windows) - 999.0

    # Prepare to jitter to avoid divide by zero etc errors
    dry_threshold = 0.01
    win_depth[win_depth < dry_threshold] = dry_threshold
    noise = random_numbers / 100.0

    # First storm selection is arbitrary
    idx = 0
    win_id2[0] = win_id[idx]
    win_depth2[0] = win_depth[idx]
    prev_depth = win_depth[idx]

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

            if dep1[idx] <= dry_threshold:
                prev_depth = dry_threshold + noise[win_idx]
            else:
                prev_depth = dep1[idx]

        win_idx += 1

    return win_id2, win_depth2


def shuffle_simulation(df, df1, parameters, datetime_helper, rng, n_divisions, do_reordering):
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
    win_id2, win_depth2 = _shuffle_windows(
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
    df_ar1['month_rank'] = df_ar1.groupby(['season'])['sa'].rank(method='first', ascending=False).astype(int)
    return df_ar1


@numba.jit(nopython=True)
def shuffle_storms(
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

            shuffled[tmp_id1[idx]] = 1

            prev_storm_depth = dep1[idx]

        storm_idx += 1

    return storm_id2, storm_depth2, storm_duration2
