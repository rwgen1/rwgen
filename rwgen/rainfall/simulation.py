import os
import itertools

import psutil
import numpy as np
import scipy.stats
import scipy.spatial
import gstools
import numba

from . import nsproc
from . import utils

# TODO: Allow for varying data types (floating point precision)


def main(
        spatial_model,
        intensity_distribution,
        discretisation_method,  # 'default' or 'event_totals'
        output_types,
        output_folder,
        output_subfolders,
        output_format,
        season_definitions,
        parameters,
        points,
        catchments,
        grid,
        epsg_code,
        cell_size,
        dem,
        phi,
        simulation_length,
        number_of_realisations,
        timestep_length,
        start_year,
        calendar,
        random_seed,
        default_block_size,
        check_block_size,
        minimum_block_size,
        check_available_memory,
        maximum_memory_percentage,
        block_subset_size,
):
    print('  - Initialising')

    # Initialisations common to both point and spatial models (derived attributes)
    realisation_ids = range(1, number_of_realisations + 1)
    output_paths = make_output_paths(
        spatial_model, output_types, output_format, output_folder, output_subfolders, points, catchments,
        realisation_ids
    )
    if random_seed is None:
        seed_sequence = np.random.SeedSequence()
    else:
        seed_sequence = np.random.SeedSequence(random_seed)

    # Possible that 32-bit floats could be used in places, but times need to be tracked with 64-bit floats in the case
    # of long simulations for example. So fixed precision currently, as care needed if deviating from 64-bit
    float_precision = 64

    # Most of the preparation needed for simulation is only for a spatial model  # TODO: Check each case
    if spatial_model:

        # Set (inner) simulation domain bounds
        xmin, ymin, xmax, ymax = identify_domain_bounds(grid, cell_size, points)

        # Set up discretisation point location metadata arrays (x, y and z by point)
        discretisation_metadata = create_discretisation_metadata_arrays(points, grid, cell_size, dem)

        # Associate a phi value with each point
        unique_seasons = utils.identify_unique_seasons(season_definitions)
        discretisation_metadata = get_phi(unique_seasons, dem, phi, output_types, discretisation_metadata)

        # Get weights associated with catchments for each point
        if 'catchment' in output_types:
            discretisation_metadata = get_catchment_weights(
                grid, catchments, cell_size, epsg_code, discretisation_metadata, output_types, dem,
                unique_seasons, catchment_id_field='id'
            )

    else:
        xmin = None
        ymin = None
        xmax = None
        ymax = None
        discretisation_metadata = None

    # Date/time helper - monthly time series indicating number of hours and timesteps in month
    end_year = start_year + simulation_length - 1
    datetime_helper = utils.make_datetime_helper(start_year, end_year, timestep_length, calendar)

    # Identify block size needed to avoid memory issues
    if check_block_size:
        block_size = identify_block_size(
            datetime_helper, season_definitions, timestep_length, discretisation_metadata,
            seed_sequence, simulation_length,
            spatial_model, parameters, intensity_distribution, xmin, xmax, ymin, ymax,
            discretisation_method, output_types, points, catchments,
            float_precision, default_block_size, minimum_block_size, check_available_memory, maximum_memory_percentage,
        )
    else:
        block_size = default_block_size

    # Do simulation
    rng = np.random.default_rng(seed_sequence)
    for realisation_id in realisation_ids:
        if discretisation_method == 'default':
            simulate_realisation(
                realisation_id, datetime_helper, simulation_length, timestep_length, season_definitions,
                discretisation_method, spatial_model, output_types, discretisation_metadata, points, catchments,
                parameters, intensity_distribution, rng, xmin, xmax, ymin, ymax, output_paths, block_size,
                block_subset_size
            )
        elif discretisation_method == 'event_totals':
            df = simulate_realisation(
                realisation_id, datetime_helper, simulation_length, timestep_length, season_definitions,
                discretisation_method, spatial_model, output_types, discretisation_metadata, points, catchments,
                parameters, intensity_distribution, rng, xmin, xmax, ymin, ymax, output_paths, block_size,
                block_subset_size
            )

    # TODO: Implement additional output - phi, catchment weights, random seed


def identify_domain_bounds(grid, cell_size, points):
    """
    Set (inner) simulation domain bounds as maximum extent of output points and grid (if required).

    """
    if grid is not None:  # accounts for both catchment and grid outputs
        grid_xmin, grid_ymin, grid_xmax, grid_ymax = utils.grid_limits(grid)
    if points is not None:
        points_xmin = np.min(points['easting'])
        points_ymin = np.min(points['northing'])
        points_xmax = np.max(points['easting'])
        points_ymax = np.max(points['northing'])
        if grid is not None:
            xmin = np.minimum(points_xmin, grid_xmin)
            ymin = np.minimum(points_ymin, grid_ymin)
            xmax = np.maximum(points_xmax, grid_xmax)
            ymax = np.maximum(points_ymax, grid_ymax)
        else:
            xmin = points_xmin
            ymin = points_ymin
            xmax = points_xmax
            ymax = points_ymax
    xmin = utils.round_down(xmin, cell_size)
    ymin = utils.round_down(ymin, cell_size)
    xmax = utils.round_up(xmax, cell_size)
    ymax = utils.round_up(ymax, cell_size)
    return xmin, ymin, xmax, ymax


def create_discretisation_metadata_arrays(points, grid, cell_size, dem):
    """
    Set up discretisation point location metadata arrays (x, y and z by point).
    
    """
    # Dictionary with keys as tuples of output type and metadata attribute (values as arrays)
    discretisation_metadata = {}

    # Point metadata values are arrays of length one
    if points is not None:
        discretisation_metadata[('point', 'x')] = points['easting'].values
        discretisation_metadata[('point', 'y')] = points['northing'].values
        if 'elevation' in points.columns:
            discretisation_metadata[('point', 'z')] = points['elevation'].values

    # For a grid these arrays are flattened 2D arrays so that every point has an associated x, y pair
    if grid is not None:
        x = np.arange(
            grid['xllcorner'] + cell_size / 2.0,
            grid['xllcorner'] + grid['ncols'] * cell_size,
            cell_size
        )
        y = np.arange(
            grid['yllcorner'] + cell_size / 2.0,
            grid['yllcorner'] + grid['nrows'] * cell_size,
            cell_size
        )
        y = y[::-1]  # reverse to get north-south order

        # Meshgrid then flatten gets each xy pair
        xx, yy = np.meshgrid(x, y)
        xf = xx.flatten()
        yf = yy.flatten()
        discretisation_metadata[('grid', 'x')] = xf
        discretisation_metadata[('grid', 'y')] = yf

        # Resample DEM to grid resolution (presumed coarser) if DEM present
        if dem is not None:
            dem_cell_size = dem.x.values[1] - dem.x.values[0]
            window = int(cell_size / dem_cell_size)

            # Restrict DEM to domain of output grid
            grid_xmin, grid_ymin, grid_xmax, grid_ymax = utils.grid_limits(grid)
            mask_x = (dem.x > grid_xmin) & (dem.x < grid_xmax)
            mask_y = (dem.y > grid_ymin) & (dem.y < grid_ymax)
            dem = dem.where(mask_x & mask_y, drop=True)

            # Boundary argument required to avoid case where DEM does not match grid neatly
            resampled_dem = dem.coarsen(x=window, boundary='pad').mean(skipna=True) \
                .coarsen(y=window, boundary='pad').mean(skipna=True)
            flat_resampled_dem = resampled_dem.data.flatten()
            discretisation_metadata[('grid', 'z')] = flat_resampled_dem

    return discretisation_metadata


def get_phi(unique_seasons, dem, phi, output_types, discretisation_metadata):
    """
    Associate a phi value with each discretisation point.

    """
    # Calculate phi for each discretisation point for all output types using interpolation (unless a point is in the
    # dataframe of known phi, in which case use it directly)
    for season in unique_seasons:

        # Make interpolator (flag needed for whether phi should be log-transformed)
        if dem is not None:
            interpolator, log_transformation = make_phi_interpolator(phi.loc[phi['season'] == season])
        else:
            interpolator, log_transformation = make_phi_interpolator(
                phi.loc[phi['season'] == season], include_elevation=False
            )

        # Estimate phi for points and grid (if phi is known at point location then exact value should be preserved)
        discretisation_types = list(set(output_types) & set(['point', 'grid']))
        if ('catchment' in output_types) and ('grid' not in output_types):
            discretisation_types.append('grid')
        for output_type in discretisation_types:
            if dem is not None:
                interpolated_phi = interpolator(
                    (discretisation_metadata[(output_type, 'x')], discretisation_metadata[(output_type, 'y')]),
                    mesh_type='unstructured',
                    ext_drift=discretisation_metadata[(output_type, 'z')],
                    return_var=False
                )
            else:
                interpolated_phi = interpolator(
                    (discretisation_metadata[(output_type, 'x')], discretisation_metadata[(output_type, 'y')]),
                    mesh_type='unstructured',
                    return_var=False
                )
            if log_transformation:
                discretisation_metadata[(output_type, 'phi', season)] = np.exp(interpolated_phi)
            else:
                discretisation_metadata[(output_type, 'phi', season)] = interpolated_phi
            discretisation_metadata[(output_type, 'phi', season)] = np.where(
                discretisation_metadata[(output_type, 'phi', season)] < 0.0,
                0.0,
                discretisation_metadata[(output_type, 'phi', season)]
            )

    return discretisation_metadata


def make_phi_interpolator(df1, include_elevation=True, distance_bins=7):
    """
    Make function to interpolate phi, optionally accounting for elevation dependence if significant.

    """
    # Test for elevation-dependence of phi using linear regression (trying untransformed and log-transformed phi)
    if include_elevation:
        untransformed_regression = scipy.stats.linregress(df1['elevation'], df1['phi'])
        log_transformed_regression = scipy.stats.linregress(df1['elevation'], np.log(df1['phi']))
        if (untransformed_regression.pvalue < 0.05) or (log_transformed_regression.pvalue < 0.05):
            significant_regression = True
            if untransformed_regression.rvalue >= log_transformed_regression.rvalue:
                log_transformation = False
            else:
                log_transformation = True
        else:
            significant_regression = False
    else:
        significant_regression = False
        log_transformation = False

    # Select regression model (untransformed or log-transformed) if significant (linear) elevation dependence
    if log_transformation:
        phi = np.log(df1['phi'])
        if significant_regression:
            regression_model = log_transformed_regression
    else:
        phi = df1['phi'].values
        if significant_regression:
            regression_model = untransformed_regression

    # Remove elevation signal from data first to allow better variogram fit
    if include_elevation and significant_regression:
        detrended_phi = phi - (df1['elevation'] * regression_model.slope + regression_model.intercept)

    # Calculate bin edges
    max_distance = np.max(scipy.spatial.distance.pdist(np.asarray(df1[['easting', 'northing']])))
    interval = max_distance / distance_bins
    bin_edges = np.arange(0.0, max_distance + 0.1, interval)
    bin_edges[-1] = max_distance + 0.1  # ensure that all points covered

    # Estimate empirical variogram
    if include_elevation and significant_regression:
        bin_centres, gamma, counts = gstools.vario_estimate(
            (df1['easting'].values, df1['northing'].values), detrended_phi, bin_edges, return_counts=True
        )
    else:
        bin_centres, gamma, counts = gstools.vario_estimate(
            (df1['easting'].values, df1['northing'].values), phi, bin_edges, return_counts=True
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

    # Instantiate appropriate kriging object
    if include_elevation and significant_regression:
        phi_interpolator = gstools.krige.ExtDrift(
            covariance_model, (df1['easting'].values, df1['northing'].values), phi, df1['elevation'].values
        )
    else:
        phi_interpolator = gstools.krige.Ordinary(
            covariance_model, (df1['easting'].values, df1['northing'].values), phi
        )

    return phi_interpolator, log_transformation


def get_catchment_weights(
        grid, catchments, cell_size, epsg_code, discretisation_metadata, output_types, dem,
        unique_seasons, catchment_id_field
):
    """
    Catchment weights as contribution of each (grid) point to catchment-average.

    """
    # First get weight for every point for every catchment
    grid_xmin, grid_ymin, grid_xmax, grid_ymax = utils.grid_limits(grid)
    catchment_points = utils.catchment_weights(
        catchments, grid_xmin, grid_ymin, grid_xmax, grid_ymax, cell_size, id_field=catchment_id_field,
        epsg_code=epsg_code
    )
    for catchment_id, point_arrays in catchment_points.items():
        # Check that points are ordered in the same way
        assert np.min(point_arrays['x'] == discretisation_metadata[('grid', 'x')]) == 1
        assert np.min(point_arrays['y'] == discretisation_metadata[('grid', 'y')]) == 1
        # TODO: Replace checks on array equivalence with dataframe merge operation
        discretisation_metadata[('catchment', 'weights', catchment_id)] = point_arrays['weight']

    # Then rationalise grid discretisation points - if a point is not used by any catchment and grid output is not
    # required then no need to discretise it
    if ('catchment' in output_types) and ('grid' not in output_types):

        # Identify cells where any subcatchment is present (i.e. overall catchment mask)
        catchment_mask = np.zeros(discretisation_metadata[('grid', 'x')].shape[0], dtype=bool)
        for catchment_id in catchment_points.keys():
            subcatchment_mask = discretisation_metadata[('catchment', 'weights', catchment_id)] > 0.0
            catchment_mask[subcatchment_mask == 1] = 1

        # Subset static (non-seasonally varying) arrays - location, elevation and weights
        discretisation_metadata[('grid', 'x')] = discretisation_metadata[('grid', 'x')][catchment_mask]
        discretisation_metadata[('grid', 'y')] = discretisation_metadata[('grid', 'y')][catchment_mask]
        if dem is not None:
            discretisation_metadata[('grid', 'z')] = discretisation_metadata[('grid', 'z')][catchment_mask]
        for catchment_id in catchment_points.keys():
            discretisation_metadata[('catchment', 'weights', catchment_id)] = (
                discretisation_metadata[('catchment', 'weights', catchment_id)][catchment_mask]
            )

        # Subset seasonally varying arrays - phi
        for season in unique_seasons:
            discretisation_metadata[('grid', 'phi', season)] = (
                discretisation_metadata[('grid', 'phi', season)][catchment_mask]
            )

    return discretisation_metadata


def make_output_paths(
        spatial_model, output_types, output_format, output_folder, output_subfolders, points, catchments,
        realisation_ids
):
    output_paths = {}
    for output_type in output_types:
        if output_type == 'grid':
            paths = output_paths_helper(
                spatial_model, output_type, 'nc', output_folder, output_subfolders, points, catchments, realisation_ids
            )
        else:
            paths = output_paths_helper(
                spatial_model, output_type, output_format, output_folder, output_subfolders, points, catchments,
                realisation_ids
            )
        for key, value in paths.items():
            output_paths[key] = value
    return output_paths


def output_paths_helper(
        spatial_model, output_type, output_format, output_folder, output_subfolders, points, catchments,
        realisation_ids
):
    output_format_extensions = {'csv': '.csv', 'csvy': '.csvy', 'txt': '.txt', 'netcdf': '.nc'}

    if output_type == 'point':
        if spatial_model:
            location_ids = list(points['point_id'].values)
            location_names = list(points['name'].values)
        else:
            location_ids = [1]
            location_names = ['simulation']
        output_subfolder = os.path.join(output_folder, output_subfolders['point'])
    elif output_type == 'catchment':
        location_ids = list(catchments['id'].values)
        location_names = list(catchments['name'].values)
        output_subfolder = os.path.join(output_folder, output_subfolders['catchment'])
    elif output_type == 'grid':
        location_ids = [1]
        location_names = ['simulation']
        output_subfolder = os.path.join(output_folder, output_subfolders['grid'])

    output_paths = {}
    output_cases = itertools.product(realisation_ids, location_ids)
    for realisation_id, location_id in output_cases:
        location_name = location_names[location_ids.index(location_id)]
        output_file_name = location_name + '_r' + str(realisation_id) + output_format_extensions[output_format]
        output_path_key = (output_type, location_id, realisation_id)
        output_paths[output_path_key] = os.path.join(output_subfolder, output_file_name)

    return output_paths


def identify_block_size(
        datetime_helper, season_definitions, timestep_length, discretisation_metadata,
        seed_sequence, number_of_years,
        spatial_model, parameters, intensity_distribution, xmin, xmax, ymin, ymax,
        discretisation_method, output_types, points, catchments,
        float_precision, default_block_size, minimum_block_size, check_available_memory, maximum_memory_percentage,
):
    """Identify size of blocks (number of years) needed to avoid potential memory issues in simulations."""
    # TODO: Allow for varying data types (floating point precision)
    block_size = min(number_of_years, default_block_size)
    found_block_size = False
    while not found_block_size:

        # Simulate a few years of NSRP process to get an idea of the memory requirements of the sampling. Four years is
        # the current minimum for the NSRP process (to be changed to one year - see below)
        rng = np.random.default_rng(seed_sequence)
        sample_n_years = 30
        start_year = datetime_helper['year'].values[0]
        end_year = start_year + sample_n_years - 1
        month_lengths = datetime_helper.loc[
            (datetime_helper['year'] >= start_year) & (datetime_helper['year'] <= end_year),
            'n_hours'].values
        dummy1 = nsproc.main(
            spatial_model, parameters, sample_n_years, month_lengths, season_definitions, intensity_distribution,
            rng, xmin, xmax, ymin, ymax
        )

        # Estimate memory requirements for NSRP process for length (number of years) of block
        nsrp_memory = (dummy1.memory_usage(deep=True).sum() / sample_n_years) * block_size * 1.2  # 1.2 = safety factor
        dummy1 = 0

        # Calculate memory requirements of working discretisation arrays (independent of any sampling)
        nt = int((24 / timestep_length) * 31)
        working_memory = 0
        if 'point' in output_types:
            if spatial_model:
                working_memory += int(nt * points.shape[0] * (float_precision / 8))
            else:
                working_memory += int(nt * (float_precision / 8))
        if ('catchment' in output_types) or ('grid' in output_types):
            working_memory += int(nt * discretisation_metadata[('grid', 'x')].shape[0] * (float_precision / 8))

        # Estimate memory requirements of point/catchment output arrays for one block
        # - assuming that timing in relation to leap years will not matter (i.e. small effect)
        block_start_year = datetime_helper['year'].values[0]  # + block_id * block_size
        block_end_year = block_start_year + block_size - 1
        n_timesteps = datetime_helper.loc[
            (datetime_helper['year'] >= block_start_year) & (datetime_helper['year'] <= block_end_year),
            'n_timesteps'].sum()
        if discretisation_method == 'default':
            if spatial_model:
                if ('point' in output_types) and ('catchment' in output_types):
                    n_points = points.shape[0] * catchments.shape[0]
                elif ('point' in output_types) and ('catchment' not in output_types):
                    n_points = points.shape[0]
                elif ('point' not in output_types) and ('catchment' in output_types):
                    n_points = catchments.shape[0]
            else:
                n_points = 1
            output_memory = int((n_timesteps * n_points) * (16 / 8))  # assuming np.float16 for output arrays only
        elif discretisation_method == 'event_totals':
            raise NotImplementedError

        # Accept block size if estimated total memory is below maximum RAM percentage to use
        required_total = nsrp_memory + working_memory + output_memory
        system_total = psutil.virtual_memory().total
        estimated_percent_use = required_total / system_total * 100
        if check_available_memory:
            system_percent_available = psutil.virtual_memory().available / system_total * 100
            memory_limit = min(maximum_memory_percentage, system_percent_available)
        else:
            memory_limit = maximum_memory_percentage
        if estimated_percent_use < memory_limit:
            found_block_size = True

        # If need to try a smaller block size then ensure it stays bigger than some minimum. Currently four years is
        # used to fit in with the buffer added in the NSRP storm process.
        # TODO: Change the NSRP process so that it can work if only one year of simulation is requested
        if not found_block_size:
            if block_size == minimum_block_size:
                raise RuntimeError('Block size is at its minimum but memory availability appears to be insufficient.')
            else:
                new_block_size = int(np.floor(block_size / 2))
                block_size = max(new_block_size, minimum_block_size)

    return block_size


def simulate_realisation(
        realisation_id, datetime_helper, number_of_years, timestep_length, season_definitions,
        discretisation_method, spatial_model, output_types, discretisation_metadata, points, catchments, parameters,
        intensity_distribution, rng, xmin, xmax, ymin, ymax, output_paths, block_size, block_subset_size
):
    """
    Simulate realisation of NSRP process.

    """
    # Initialise arrays according to discretisation method
    if discretisation_method == 'default':  # one-month blocks
        discrete_rainfall = initialise_discrete_rainfall_arrays(
            spatial_model, output_types, discretisation_metadata, points, int((24 / timestep_length) * 31)
        )
    # TODO: Consider whether arrays for point or whole-domain event totals need to be initialised here

    # Simulate and discretise NSRP process by block
    n_blocks = int(np.floor(number_of_years / block_size))
    if number_of_years % block_size != 0:
        n_blocks += 1
    block_id = 0
    while block_id * block_size < number_of_years:
        print('  - Realisation =', realisation_id, '[Block =', str(block_id + 1) + '/' + str(n_blocks) + ']')
        print('    - Sampling')

        # NSRP process simulation
        block_start_year = datetime_helper['year'].values[0] + block_id * block_size
        block_end_year = block_start_year + block_size - 1
        month_lengths = datetime_helper.loc[
            (datetime_helper['year'] >= block_start_year) & (datetime_helper['year'] <= block_end_year),
            'n_hours'].values
        df = nsproc.main(
            spatial_model, parameters, block_size, month_lengths, season_definitions, intensity_distribution,
            rng, xmin, xmax, ymin, ymax
        )

        # Convert raincell coordinates and radii from km to m for discretisation
        if 'raincell_x' in df.columns:
            df['raincell_x'] *= 1000.0
            df['raincell_y'] *= 1000.0
            df['raincell_radii'] *= 1000.0

        # Discretisation
        if discretisation_method == 'default':
            discretise_by_point(
                spatial_model,
                datetime_helper.loc[
                    (datetime_helper['year'] >= block_start_year) & (datetime_helper['year'] <= block_end_year)
                ],
                season_definitions, df, output_types, timestep_length,
                discrete_rainfall,
                discretisation_metadata, points, catchments, realisation_id, output_paths, block_id, block_subset_size
            )
            # TODO: Check that slice of datetime_helper is correct
        elif discretisation_method == 'event_totals':
            events_df = discretise_by_event()  # TODO: Not yet implemented

        block_id += 1

    # Assuming that event totals etc are not being written to file but should be returned for shuffling etc
    if discretisation_method == 'event_totals':
        return events_df


def initialise_discrete_rainfall_arrays(spatial_model, output_types, discretisation_metadata, points, nt):
    dc = {}
    if 'point' in output_types:
        if spatial_model:
            dc['point'] = np.zeros((nt, points.shape[0]))
        else:
            dc['point'] = np.zeros((nt, 1))
    if ('catchment' in output_types) or ('grid' in output_types):
        dc['grid'] = np.zeros((nt, discretisation_metadata[('grid', 'x')].shape[0]))
    return dc


def discretise_by_point(
        spatial_model, datetime_helper, season_definitions, df, output_types, timestep_length, discrete_rainfall,
        discretisation_metadata, points, catchments, realisation_id, output_paths, block_id, block_subset_size
):
    print('    - Discretising = ', end='')  # TEMPORARY
    # TODO: Expecting datetime_helper just for block - check that correctly subset before argument passed

    # Adjust datetime helper so that its times and timesteps are set with reference to the beginning of the block
    # rather than the beginning of the simulation
    datetime_helper = datetime_helper.copy()
    initial_start_time = datetime_helper['start_time'].values[0]
    initial_start_timestep = datetime_helper['start_timestep'].values[0]
    datetime_helper['start_time'] -= initial_start_time
    datetime_helper['end_time'] -= initial_start_time
    datetime_helper['start_timestep'] -= initial_start_timestep
    datetime_helper['end_timestep'] -= initial_start_timestep

    # Prepare to store realisation output for block (point and catchment output only)
    output_arrays = {}

    # Month indices to use for printing progress
    print_helper = list(range(0, datetime_helper.shape[0], int(datetime_helper.shape[0] / 10)))
    if print_helper[-1] != (datetime_helper.shape[0] - 1):
        print_helper.append(datetime_helper.shape[0] - 1)

    # Use sub-blocks to speed up month-wise loop, as selection/subsetting of raincells for a given month is much faster
    # with smaller arrays
    subset_n_years = min(int(datetime_helper.shape[0] / 12), block_subset_size)
    subset_start_idx = 0
    while subset_start_idx < datetime_helper.shape[0]:
        subset_end_idx = min(subset_start_idx + subset_n_years * 12 - 1, datetime_helper.shape[0] - 1)
        subset_start_time = datetime_helper['start_time'].values[subset_start_idx]
        subset_end_time = datetime_helper['end_time'].values[subset_end_idx]
        df1 = df.loc[(df['raincell_arrival'] < subset_end_time) & (df['raincell_end'] > subset_start_time)]

        # Looping time series of months
        for month_idx in range(subset_start_idx, subset_end_idx+1):  # range(datetime_helper.shape[0]):

            if month_idx in print_helper:
                # print('    - Discretising:', str(print_helper.index(month_idx) * 10) + '%')
                if month_idx == print_helper[-1]:
                    print(str(print_helper.index(month_idx) * 10) + '%')
                else:
                    print(str(print_helper.index(month_idx) * 10) + '%', end=' ')

            year = datetime_helper['year'].values[month_idx]
            month = datetime_helper['month'].values[month_idx]
            season = season_definitions[month]

            # Perform temporal subset before discretising points (much more efficient for spatial model)
            start_time = datetime_helper['start_time'].values[month_idx]
            end_time = datetime_helper['end_time'].values[month_idx]
            temporal_mask = (df1['raincell_arrival'].values < end_time) & (df1['raincell_end'].values > start_time)
            raincell_arrival_times = df1['raincell_arrival'].values[temporal_mask]
            raincell_end_times = df1['raincell_end'].values[temporal_mask]
            raincell_intensities = df1['raincell_intensity'].values[temporal_mask]

            # Spatial model discretisation requires temporal subset of additional raincell properties
            if spatial_model:
                raincell_x = df1['raincell_x'].values[temporal_mask]
                raincell_y = df1['raincell_y'].values[temporal_mask]
                raincell_radii = df1['raincell_radii'].values[temporal_mask]

                # If both catchment and grid are in output types then the same grid is used so only need to do once
                if ('catchment' in output_types) and ('grid' in output_types):
                    _output_types = list(set(output_types) & set(['point', 'catchment']))
                else:
                    _output_types = output_types
                for output_type in _output_types:
                    if output_type == 'catchment':
                        discretisation_case = 'grid'
                    else:
                        discretisation_case = output_type

                    discretise_spatial(
                        start_time, timestep_length, raincell_arrival_times, raincell_end_times,
                        raincell_intensities, discrete_rainfall[discretisation_case],
                        raincell_x, raincell_y, raincell_radii,
                        discretisation_metadata[(discretisation_case, 'x')],
                        discretisation_metadata[(discretisation_case, 'y')],
                        discretisation_metadata[(discretisation_case, 'phi', season)],
                    )

            else:
                discretise_point(
                    start_time, timestep_length, raincell_arrival_times, raincell_end_times,
                    raincell_intensities, discrete_rainfall['point'][:, 0]
                )

            # Find number of timesteps in month to be able to subset the discretised arrays (if < 31 days in month)
            timesteps_in_month = datetime_helper.loc[
                (datetime_helper['year'] == year) & (datetime_helper['month'] == month), 'n_timesteps'
            ].values[0]

            # Put discrete rainfall in arrays ready for writing once all block available
            for output_type in output_types:
                if output_type == 'point':
                    if not spatial_model:
                        location_ids = [1]
                    else:
                        location_ids = points['point_id'].values  # self.points['point_id'].values
                elif output_type == 'catchment':
                    location_ids = catchments['id'].values  # self.catchments[self.catchment_id_field].values
                elif output_type == 'grid':
                    location_ids = [1]
                # TODO: See if output keys can be looped directly without needing to figure out location_ids again

                # TODO: Reduce dependence on list/array order
                idx = 0
                for location_id in location_ids:
                    output_key = (output_type, location_id, realisation_id)

                    if output_type == 'point':
                        output_array = discrete_rainfall['point'][:timesteps_in_month, idx]
                    elif output_type == 'catchment':
                        catchment_discrete_rainfall = np.average(
                            discrete_rainfall['grid'], axis=1,
                            weights=discretisation_metadata[('catchment', 'weights', location_id)]
                        )
                        output_array = catchment_discrete_rainfall[:timesteps_in_month]
                    elif output_type == 'grid':
                        raise NotImplementedError('Grid output not implemented yet')

                    # Appending an array to a list is faster than concatenating arrays
                    if month_idx == 0:
                        output_arrays[output_key] = [output_array.astype(np.float16)]
                    else:
                        output_arrays[output_key].append(output_array.astype(np.float16))

                    idx += 1

        # Increment subset index tracker
        subset_start_idx += (subset_n_years * 12)

    # Write output
    print('    - Writing')
    if block_id == 0:
        write_new_files = True
    else:
        write_new_files = False
    write_output(output_arrays, output_paths, write_new_files)


@numba.jit(nopython=True)
def discretise_point(
        period_start_time, timestep_length, raincell_arrival_times, raincell_end_times,
        raincell_intensities, discrete_rainfall
):
    # Modifying the discrete rainfall arrays themselves so need to ensure zeros before starting
    discrete_rainfall.fill(0.0)

    # Discretise each raincell in turn
    for idx in range(raincell_arrival_times.shape[0]):

        # Times relative to period start
        rc_arrival_time = raincell_arrival_times[idx] - period_start_time
        rc_end_time = raincell_end_times[idx] - period_start_time
        rc_intensity = raincell_intensities[idx]

        # Timesteps relative to period start
        rc_arrival_timestep = int(np.floor(rc_arrival_time / timestep_length))
        rc_end_timestep = int(np.floor(rc_end_time / timestep_length))  # timestep containing end

        # Proportion of raincell in each relevant timestep
        for timestep in range(rc_arrival_timestep, rc_end_timestep+1):
            timestep_start_time = timestep * timestep_length
            timestep_end_time = (timestep + 1) * timestep_length
            effective_start = np.maximum(rc_arrival_time, timestep_start_time)
            effective_end = np.minimum(rc_end_time, timestep_end_time)
            timestep_coverage = effective_end - effective_start

            if timestep < discrete_rainfall.shape[0]:
                discrete_rainfall[timestep] += rc_intensity * timestep_coverage


@numba.jit(nopython=True)
def discretise_spatial(
        period_start_time, timestep_length, raincell_arrival_times, raincell_end_times,
        raincell_intensities, discrete_rainfall,
        raincell_x_coords, raincell_y_coords, raincell_radii,
        point_eastings, point_northings, point_phi,  # point_ids,
):
    # Modifying the discrete rainfall arrays themselves so need to ensure zeros before starting
    discrete_rainfall.fill(0.0)

    # Subset raincells based on whether they intersect the point being discretised
    for idx in range(point_eastings.shape[0]):
        x = point_eastings[idx]
        y = point_northings[idx]
        yi = idx

        distances_from_raincell_centres = np.sqrt((x - raincell_x_coords) ** 2 + (y - raincell_y_coords) ** 2)
        spatial_mask = distances_from_raincell_centres <= raincell_radii

        discretise_point(
            period_start_time, timestep_length, raincell_arrival_times[spatial_mask],
            raincell_end_times[spatial_mask], raincell_intensities[spatial_mask], discrete_rainfall[:, yi]
        )

        discrete_rainfall[:, yi] *= point_phi[idx]


def write_output(output_arrays, output_paths, write_new_files):
    for output_key, output_array_list in output_arrays.items():
        # output_type, location_id, realisation_id = output_key
        output_path = output_paths[output_key]
        values = []
        for output_array in output_array_list:
            for value in output_array:
                values.append(('%.1f' % value).rstrip('0').rstrip('.'))  # + '\n'
        output_lines = '\n'.join(values)
        if write_new_files:
            with open(output_path, 'w') as fh:
                fh.writelines(output_lines)
        else:
            with open(output_path, 'a') as fh:
                fh.writelines(output_lines)
        # TODO: Implement other text file output options


def discretise_by_event():
    raise NotImplementedError
