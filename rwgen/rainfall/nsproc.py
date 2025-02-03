import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats


def main(
        spatial_model,
        parameters,
        simulation_length,
        month_lengths,
        season_definitions,
        intensity_distribution,
        rng,
        xmin,
        xmax,
        ymin,
        ymax,
        buffer_factor,
):
    """
    Args:
        spatial_model (bool): Flag to indicate whether spatial model (True) or point model (False).
        parameters (pandas.DataFrame): Parameters dataframe obtained from fitting.
        simulation_length (int): Number of years to simulate.
        month_lengths (int): Time series of number of hours in each month to be simulated.
        season_definitions (dict): Months (1-12) as keys and season identifiers (integers) as values.
        intensity_distribution (str): Flag indicating raincell intensity distribution.
        rng (numpy.random.Generator): Random number generator.
        xmin (float): Minimum x (easting) coordinate of domain [m].
        xmax (float): Maximum x (easting) coordinate of domain [m].
        ymin (float): Minimum y (easting) coordinate of domain [m].
        ymax (float): Maximum y (easting) coordinate of domain [m].
        method (str): Flag to use ``'buffer'`` method or Burton et al. (2010) method ``'burton'`` for spatial raincell
            simulation.
        buffer_factor (float): Number of standard deviations of raincell radius distribution to use with buffer method.

    Notes:
        The steps in simulation of the NSRP process are:  # TODO: Complete description
            1. Simulate storms as a temporal Poisson process.
            2. Simulate raincells.
            3. Simulate raincell arrival times.
            4. Simulate raincell durations.
            5. Simulate raincell intensities.

    """
    # Ensure parameters are available monthly (i.e. repeated for each month in season)
    if len(season_definitions.keys()) == 12:
        parameters = parameters.copy()
        parameters['month'] = parameters['season']
    else:
        months = []
        seasons = []
        for month, season in season_definitions.items():
            months.append(month)
            seasons.append(season)
        df_seasons = pd.DataFrame({'month': months, 'season': seasons})
        parameters = pd.merge(df_seasons, parameters, how='left', on='season')
    parameters.sort_values(by='month', inplace=True)

    # Convert coordinate units from m to km and derive domain width/length and area
    if spatial_model:
        xmin = xmin / 1000.0
        xmax = xmax / 1000.0
        ymin = ymin / 1000.0
        ymax = ymax / 1000.0

    # NSRP process simulation

    # Step 1 - Simulate storms as a temporal Poisson process
    storms, number_of_storms = simulate_storms(month_lengths, simulation_length, parameters, rng)

    # Step 2 - Simulate number of raincells associated with each storm. The resulting dataframe (df) contains one row
    # per raincell (accompanied by parent storm properties, mostly importantly storm arrival time)
    if not spatial_model:
        df = simulate_raincells_point(storms, parameters, rng)
    else:
        df = simulate_raincells_spatial(
            storms, parameters, xmin, xmax, ymin, ymax, rng, buffer_factor,
        )

    # Helper step - Merge parameters into master (row per raincell) dataframe
    df = merge_parameters(df, month_lengths, simulation_length, parameters)

    # Step 3 - Simulate raincell arrival times
    raincell_arrival_times = rng.exponential(1.0 / df['beta'])  # relative to storm origins
    df['raincell_arrival'] = df['storm_arrival'] + raincell_arrival_times  # relative to simulation period origin

    # Step 4 - Simulate raincell durations (and thus end times)
    df['raincell_duration'] = rng.exponential(1.0 / df['eta'])
    df['raincell_end'] = df['raincell_arrival'] + df['raincell_duration']

    # Step 5 - Simulate raincell intensities
    if intensity_distribution == 'exponential':
        df['raincell_intensity'] = rng.exponential(df['theta'])  # in rainsim terms, theta = 1 / xi
    elif intensity_distribution == 'weibull':
        df['raincell_intensity'] = scipy.stats.weibull_min.rvs(c=df['kappa'], scale=df['theta'], random_state=rng)
    elif intensity_distribution == 'generalised_gamma':
        df['raincell_intensity'] = scipy.stats.gengamma.rvs(
            a=(df['kappa_1'] / df['kappa_2']), c=df['kappa_2'], scale=df['theta'], random_state=rng)

    # Tidy df
    df.drop(columns=['lamda', 'beta', 'rho', 'eta', 'gamma', 'theta', 'kappa'], inplace=True, errors='ignore')

    return df


def simulate_storms(month_lengths, simulation_length, parameters, rng):
    """
    Simulate storms as a temporal Poisson process.

    """
    # Ensure that Poisson process is sampled beyond end of simulation to avoid any truncation errors
    simulation_end_time = np.cumsum(month_lengths)[-1]
    simulation_length = simulation_length
    month_lengths = month_lengths

    while True:

        # Set up simulation_length and month_lengths with buffer applied
        simulation_length += 4
        for _ in range(4):
            month_lengths = np.concatenate([month_lengths, month_lengths[-12:]])

        # Repeat each set of monthly lamda values for each year in simulation
        lamda = np.tile(parameters['lamda'].values, simulation_length)

        # Get a sample value for number of storms given simulation length
        cumulative_expected_storms = np.cumsum(lamda * month_lengths)
        cumulative_month_endtimes = np.cumsum(month_lengths)
        expected_number_of_storms = cumulative_expected_storms[-1]
        number_of_storms = rng.poisson(expected_number_of_storms)  # sampled

        # Sample storm arrival times on deformed timeline
        deformed_storm_arrival_times = (expected_number_of_storms * np.sort(rng.uniform(size=number_of_storms)))

        # Transform storm origin times from deformed to linear timeline
        cumulative_expected_storms = np.insert(cumulative_expected_storms, 0, 0.0)
        cumulative_month_endtimes = np.insert(cumulative_month_endtimes, 0, 0.0)
        interpolator = scipy.interpolate.interp1d(
            cumulative_expected_storms, cumulative_month_endtimes
        )
        storm_arrival_times = interpolator(deformed_storm_arrival_times)

        # Terminate sampling process once a storm has been simulated beyond the simulation end time (and then restrict
        # to just those storms arriving before the end time)
        if storm_arrival_times[-1] > simulation_end_time:
            storm_arrival_times = storm_arrival_times[storm_arrival_times < simulation_end_time]
            number_of_storms = storm_arrival_times.shape[0]
            storms = pd.DataFrame({
                'storm_id': np.arange(number_of_storms),
                'storm_arrival': storm_arrival_times
            })
            storms['month'] = lookup_months(month_lengths, simulation_length, storms['storm_arrival'].values)
            break

    return storms, number_of_storms


def simulate_raincells_point(storms, parameters, rng):
    """
    Simulate raincells for point model.

    """
    # Temporarily merging parameters here, but can be done before this method is called if generalise
    tmp = pd.merge(storms, parameters, how='left', on='month')
    tmp.sort_values(['storm_id'], inplace=True)  # checks that order matches self.storms

    # Generate Poisson random number of raincells for each storm
    number_of_raincells_by_storm = rng.poisson(tmp['nu'].values)

    # Make a master dataframe with one row per raincell along with the properties (ID, month, arrival time) of the
    # parent storm
    storm_ids_by_raincell, storm_arrivals_by_raincell, storm_months_by_raincell = make_storm_arrays_by_raincell(
        number_of_raincells_by_storm, storms['storm_id'].values, storms['storm_arrival'].values, storms['month'].values
    )
    df = pd.DataFrame({
        'storm_id': storm_ids_by_raincell,
        'storm_arrival': storm_arrivals_by_raincell,
        'month': storm_months_by_raincell,
    })

    return df


def simulate_raincells_spatial(storms, parameters, xmin, xmax, ymin, ymax, rng, buffer_factor):
    """
    Simulate raincells for spatial model.

    Notes:
        Requires parameters dataframe to be ordered by month (1-12).

    """
    # Loop is by unique month (1-12)
    i = 0
    for _, row in parameters.iterrows():  # could be replaced by a loop through range(1, 12+1)
        storms_in_month = storms.loc[storms['month'] == row['month']]
        month_number_of_storms = storms_in_month.shape[0]
        month_number_of_raincells_by_storm, \
            _, \
            month_raincell_x_coords, \
            month_raincell_y_coords, \
            month_raincell_radii = (
                spatial_poisson_process(
                    row['rho'], row['gamma'], month_number_of_storms, xmin, xmax, ymin,
                    ymax, rng, buffer_factor
                )
            )

        # Associate parent storm properties with each raincell
        month_storm_ids_by_raincell, month_storm_arrivals_by_raincell, _ = make_storm_arrays_by_raincell(
            month_number_of_raincells_by_storm, storms_in_month['storm_id'].values,
            storms_in_month['storm_arrival'].values, storms_in_month['month'].values
        )

        # Concatenate the arrays (appending to first month processed)
        if i == 0:
            number_of_raincells_by_storm = month_number_of_raincells_by_storm
            raincell_x_coords = month_raincell_x_coords
            raincell_y_coords = month_raincell_y_coords
            raincell_radii = month_raincell_radii
            storm_ids_by_raincell = month_storm_ids_by_raincell
            storm_arrivals_by_raincell = month_storm_arrivals_by_raincell
            months_by_raincell = np.zeros(month_storm_ids_by_raincell.shape[0]) + int(row['month'])
        else:
            number_of_raincells_by_storm = np.concatenate([
                number_of_raincells_by_storm, month_number_of_raincells_by_storm
            ])
            raincell_x_coords = np.concatenate([raincell_x_coords, month_raincell_x_coords])
            raincell_y_coords = np.concatenate([raincell_y_coords, month_raincell_y_coords])
            raincell_radii = np.concatenate([raincell_radii, month_raincell_radii])
            storm_ids_by_raincell = np.concatenate([storm_ids_by_raincell, month_storm_ids_by_raincell])
            storm_arrivals_by_raincell = np.concatenate([
                storm_arrivals_by_raincell, month_storm_arrivals_by_raincell
            ])
            months_by_raincell = np.concatenate([
                months_by_raincell, np.zeros(month_storm_ids_by_raincell.shape[0]) + int(row['month'])
            ])
        i += 1

    # Put into dataframe and then sort
    df = pd.DataFrame({
        'storm_id': storm_ids_by_raincell,
        'storm_arrival': storm_arrivals_by_raincell,
        'month': months_by_raincell,
        'raincell_x': raincell_x_coords,
        'raincell_y': raincell_y_coords,
        'raincell_radii': raincell_radii,
    })
    df.sort_values('storm_arrival', inplace=True)

    return df


# ---------------------------------------------------------------------------------------------------------------------
# Helper functions common to point and spatial models

def lookup_months(month_lengths, period_length, times):
    """
    Find calendar month associated with a set of times relative to simulation origin.

    Args:
        month_lengths (numpy.ndarray): Time series of month lengths in hours.
        period_length (int): Number of years.
        times (numpy.ndarray): Times relative to simulation origin.

    """
    end_times = np.cumsum(month_lengths)
    repeated_months = np.tile(np.arange(1, 12+1, dtype=int), period_length)
    idx = np.digitize(times, end_times)  # TODO: Check that -1 not required
    months = repeated_months[idx]
    return months


def make_storm_arrays_by_raincell(number_of_raincells_by_storm, storm_ids, storm_arrival_times, storm_months):
    """
    Repeat storm properties for each member raincell to help get arrays per raincell.

    """
    # Could be made more generic by taking df as argument and looping through columns...
    storm_ids_by_raincell = np.repeat(storm_ids, number_of_raincells_by_storm)
    storm_arrival_times_by_raincell = np.repeat(storm_arrival_times, number_of_raincells_by_storm)
    storm_months_by_raincell = np.repeat(storm_months, number_of_raincells_by_storm)
    return storm_ids_by_raincell, storm_arrival_times_by_raincell, storm_months_by_raincell


def merge_parameters(df, month_lengths, simulation_length, parameters):
    """
    Merge parameters into dataframe of all raincells.

    """
    df['month'] = lookup_months(month_lengths, simulation_length, df['storm_arrival'].values)
    parameters_subset = parameters.loc[parameters['fit_stage'] == 'final'].copy()
    parameters_subset = parameters_subset.drop([  # !221121
        'fit_stage', 'converged', 'objective_function', 'iterations', 'function_evaluations', 'delta', 'ar1_slope',
        'ar1_intercept', 'ar1_stderr'
    ], axis=1, errors='ignore')
    df = pd.merge(df, parameters_subset, how='left', on='month')
    return df


# ---------------------------------------------------------------------------------------------------------------------
# Functions required for raincell simulation for spatial model

def spatial_poisson_process(
        rho, gamma, number_of_storms, xmin, xmax, ymin, ymax, rng, buffer_factor=15,
):
    # Apply buffer to domain
    radius_variance = scipy.stats.expon.stats(moments='v', scale=(1.0 / gamma))
    buffer_distance = buffer_factor * radius_variance ** 0.5
    xmin_b = xmin - buffer_distance
    xmax_b = xmax + buffer_distance
    ymin_b = ymin - buffer_distance
    ymax_b = ymax + buffer_distance
    area_b = (xmax_b - xmin_b) * (ymax_b - ymin_b)

    # Simulate raincells (number, location and radii)
    n_raincells_by_storm = rng.poisson(rho * area_b, number_of_storms)
    n_raincells = np.sum(n_raincells_by_storm)
    x_coords = rng.uniform(xmin_b, xmax_b, n_raincells)
    y_coords = rng.uniform(ymin_b, ymax_b, n_raincells)
    radii = rng.exponential((1.0 / gamma), n_raincells)

    # Remove irrelevant raincells (and update n_raincells_by_storm)
    storm_ids_by_raincell = np.repeat(np.arange(number_of_storms, dtype=int), n_raincells_by_storm)
    relevant_flag = find_relevant_raincells(x_coords, y_coords, radii, xmin, xmax, ymin, ymax)
    df = pd.DataFrame({'storm_id': storm_ids_by_raincell, 'relevant': relevant_flag})
    df = df.groupby(['storm_id'])['relevant'].sum()
    n_raincells_by_storm = df.values
    n_raincells = np.sum(n_raincells_by_storm)
    x_coords = x_coords[relevant_flag]
    y_coords = y_coords[relevant_flag]
    radii = radii[relevant_flag]

    return n_raincells_by_storm, n_raincells, x_coords, y_coords, radii


def find_relevant_raincells(x, y, radius, xmin, xmax, ymin, ymax):
    # Distances for raincells within y-range but outside x-range
    mask_1 = ((y >= ymin) & (y <= ymax)) & ((x < xmin) | (x > xmax))
    d1 = np.abs(x - xmin)
    d2 = np.abs(x - xmax)
    distance_1 = np.minimum(d1, d2)

    # Distances for raincells within x-range but outside y-range
    mask_2 = ((x >= xmin) & (x <= xmax)) & ((y < ymin) | (y > ymax))
    d1 = np.abs(y - ymin)
    d2 = np.abs(y - ymax)
    distance_2 = np.minimum(d1, d2)

    # Distances for raincells with x greater than xmax and y outside y-range
    mask_3 = (x > xmax) & ((y < ymin) | (y > ymax))
    d1 = ((x - xmax) ** 2 + (y - ymax) ** 2) ** 0.5
    d2 = ((x - xmax) ** 2 + (y - ymin) ** 2) ** 0.5
    distance_3 = np.minimum(d1, d2)

    # Distances for raincells with x less than xmin and y outside y-range
    mask_4 = (x < xmin) & ((y < ymin) | (y > ymax))
    d1 = ((x - xmin) ** 2 + (y - ymax) ** 2) ** 0.5
    d2 = ((x - xmin) ** 2 + (y - ymin) ** 2) ** 0.5
    distance_4 = np.minimum(d1, d2)

    # To ensure all points within domain are definitely retained
    mask_5 = ((x >= xmin) & (x <= xmax)) & ((y >= ymin) & (y <= ymax))
    distance_5 = np.zeros(mask_5.shape[0])

    # Collate minimum distances
    min_distance = np.zeros(x.shape[0])
    min_distance[mask_1] = distance_1[mask_1]
    min_distance[mask_2] = distance_2[mask_2]
    min_distance[mask_3] = distance_3[mask_3]
    min_distance[mask_4] = distance_4[mask_4]
    min_distance[mask_5] = distance_5[mask_5]

    # Identify relevant raincells (i.e. radius exceeds minimum distance to domain)
    relevant_flag = np.zeros(x.shape[0], dtype=bool)
    relevant_flag[min_distance < radius] = 1

    return relevant_flag
