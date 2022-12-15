import os
import sys
import itertools
import datetime

import psutil
import numpy as np
import scipy.stats
import scipy.spatial
import gstools
import numba
import pandas as pd

from . import nsproc
from . import utils
from . import shuffling


# TODO: Allow for varying data types (floating point precision)


def main(
        spatial_model,
        intensity_distribution,
        output_types,
        output_folder,
        output_subfolders,
        output_format,
        season_definitions,
        parameters,
        point_metadata,
        catchment_metadata,
        grid_metadata,
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
        project_name,
        spatial_raincell_method,
        spatial_buffer_factor,
        simulation_mode,  # 'no_shuffling', 'shuffling_preparation', 'with_shuffling'
        weather_model,
        max_dsl,
        n_divisions,
        do_reordering,
):
    # print('  - Initialising')

    # Initialisations common to both point and spatial models (derived attributes)
    realisation_ids = range(1, number_of_realisations + 1)
    output_paths = make_output_paths(
        spatial_model, output_types, output_format, output_folder, output_subfolders, point_metadata,
        catchment_metadata, realisation_ids, project_name
    )
    if random_seed is None:
        seed_sequence = np.random.SeedSequence()
    else:
        seed_sequence = np.random.SeedSequence(random_seed)

    # ---
    # Currently setting output paths as attribute of weather model not simulator, as simulator is not initialised yet
    if weather_model is not None:
        weather_model.set_output_paths(
            spatial_model, output_types, output_format, output_subfolders, point_metadata,  # output_folder,
            catchment_metadata, realisation_ids, project_name
        )
    # ---

    # Possible that 32-bit floats could be used in places, but times need to be tracked with 64-bit floats in the case
    # of long simulations for example. So fixed precision currently, as care needed if deviating from 64-bit
    float_precision = 64

    # Most of the preparation needed for simulation is only for a spatial model  # TODO: Check each case
    if spatial_model:

        # Set (inner) simulation domain bounds
        xmin, ymin, xmax, ymax = identify_domain_bounds(grid_metadata, cell_size, point_metadata)

        # Set up discretisation point location metadata arrays (x, y and z by point)
        discretisation_metadata = create_discretisation_metadata_arrays(point_metadata, grid_metadata, cell_size, dem)

        # Associate a phi value with each point
        unique_seasons = utils.identify_unique_seasons(season_definitions)
        discretisation_metadata = get_phi(unique_seasons, dem, phi, output_types, discretisation_metadata)

        # Get weights associated with catchments for each point
        if 'catchment' in output_types:
            discretisation_metadata = get_catchment_weights(
                grid_metadata, catchment_metadata, cell_size, epsg_code, discretisation_metadata, output_types, dem,
                unique_seasons, catchment_id_field='id'
            )

            # !221122 - TESTING
            # _fol = 'H:/Projects/rwgen/working/iss13/stnsrp/discretisation/'
            # np.savetxt(_fol + 'x_1b.txt', discretisation_metadata[('grid', 'x')], '%.2f')
            # np.savetxt(_fol + 'y_1b.txt', discretisation_metadata[('grid', 'y')], '%.2f')
            # np.savetxt(_fol + 'w_1b.txt', discretisation_metadata[('catchment', 'weights', 1)], '%.4f')
            # sys.exit()

    else:
        xmin = None
        ymin = None
        xmax = None
        ymax = None
        # discretisation_metadata = None
        discretisation_metadata = create_discretisation_metadata_arrays(point_metadata, grid_metadata, cell_size, dem)

    # Date/time helper - monthly time series indicating number of hours and timesteps in month
    end_year = start_year + simulation_length - 1
    datetime_helper = utils.make_datetime_helper(start_year, end_year, timestep_length, calendar)

    # Identify block size needed to avoid memory issues
    if check_block_size:
        block_size = identify_block_size(
            datetime_helper, season_definitions, timestep_length, discretisation_metadata,
            seed_sequence, simulation_length,
            spatial_model, parameters, intensity_distribution, xmin, xmax, ymin, ymax,
            output_types, point_metadata, catchment_metadata,
            float_precision, default_block_size, minimum_block_size, check_available_memory, maximum_memory_percentage,
            spatial_raincell_method, spatial_buffer_factor
        )
    else:
        block_size = default_block_size

    # Do simulation
    rng = np.random.default_rng(seed_sequence)
    for realisation_id in realisation_ids:
        simulate_realisation(
            realisation_id, datetime_helper, simulation_length, timestep_length, season_definitions,
            spatial_model, output_types, discretisation_metadata, point_metadata, catchment_metadata,
            parameters, intensity_distribution, rng, xmin, xmax, ymin, ymax, output_paths, block_size,
            block_subset_size, spatial_raincell_method, spatial_buffer_factor, simulation_mode,
            weather_model, max_dsl, n_divisions, do_reordering,
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
    if cell_size is not None:
        xmin = utils.round_down(xmin, cell_size)
        ymin = utils.round_down(ymin, cell_size)
        xmax = utils.round_up(xmax, cell_size)
        ymax = utils.round_up(ymax, cell_size)
    else:
        xmin = utils.round_down(xmin, 1)
        ymin = utils.round_down(ymin, 1)
        xmax = utils.round_up(xmax, 1)
        ymax = utils.round_up(ymax, 1)
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
            interpolator, log_transformation, significant_regression = make_phi_interpolator(
                phi.loc[phi['season'] == season]
            )
        else:
            interpolator, log_transformation, significant_regression = make_phi_interpolator(
                phi.loc[phi['season'] == season], include_elevation=False
            )

        # Estimate phi for points and grid (if phi is known at point location then exact value should be preserved)
        discretisation_types = list(set(output_types) & set(['point', 'grid']))
        if ('catchment' in output_types) and ('grid' not in output_types):
            discretisation_types.append('grid')
        for output_type in discretisation_types:
            if (dem is not None) and significant_regression:
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
            log_transformation = False
    else:
        significant_regression = False
        log_transformation = False

    # Select regression model (untransformed or log-transformed) if significant (linear) elevation dependence
    if include_elevation:
        if log_transformation:
            phi = np.log(df1['phi'])
            if significant_regression:
                regression_model = log_transformed_regression
        else:
            phi = df1['phi'].values
            if significant_regression:
                regression_model = untransformed_regression
    else:
        phi = df1['phi'].values

    # Remove elevation signal from data first to allow better variogram fit
    if include_elevation and significant_regression:
        detrended_phi = phi - (df1['elevation'] * regression_model.slope + regression_model.intercept)

    # Calculate bin edges
    if df1.shape[0] > 1:  # 061
        max_distance = np.max(scipy.spatial.distance.pdist(np.asarray(df1[['easting', 'northing']])))
    else:
        max_distance = 1.0
    max_distance = max(max_distance, 1.0)  # 060
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

    return phi_interpolator, log_transformation, significant_regression


def get_catchment_weights(
        grid, catchments, cell_size, epsg_code, discretisation_metadata, output_types, dem,
        unique_seasons, catchment_id_field
):
    """
    Catchment weights as contribution of each (grid) point to catchment-average.

    """
    # First get weight for every point for every catchment
    # print('!!', grid)
    # sys.exit()
    grid_xmin, grid_ymin, grid_xmax, grid_ymax = utils.grid_limits(grid)
    catchment_points = utils.catchment_weights(
        catchments, grid_xmin, grid_ymin, grid_xmax, grid_ymax, cell_size, id_field=catchment_id_field,
        epsg_code=epsg_code
    )
    # print(catchment_points[1.0]['weight'].sum())
    # print(catchment_points[2.0]['weight'].sum())
    # print(catchment_points[3.0]['weight'].sum())
    # print(catchment_points[4.0]['weight'].sum())
    # sys.exit()
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
        realisation_ids, project_name, variables=['prcp'],
):
    output_paths = {}
    for output_type in output_types:
        if output_type == 'grid':
            paths = output_paths_helper(
                spatial_model, output_type, 'nc', output_folder, output_subfolders, points, catchments, realisation_ids,
                project_name, variables,
            )
        else:
            paths = output_paths_helper(
                spatial_model, output_type, output_format, output_folder, output_subfolders, points, catchments,
                realisation_ids, project_name, variables,
            )
        for key, value in paths.items():
            output_paths[key] = value
    return output_paths


def output_paths_helper(
        spatial_model, output_type, output_format, output_folder, output_subfolders, points, catchments,
        realisation_ids, project_name, variables,
):
    output_format_extensions = {'csv': '.csv', 'csvy': '.csvy', 'txt': '.txt', 'netcdf': '.nc'}

    if output_type == 'point':
        if spatial_model:
            location_ids = list(points['point_id'].values)
            location_names = list(points['name'].values)
        else:
            location_ids = [1]
            location_names = [project_name]
        output_subfolder = os.path.join(output_folder, output_subfolders['point'])
    elif output_type == 'catchment':
        location_ids = list(catchments['id'].values)
        location_names = list(catchments['name'].values)
        output_subfolder = os.path.join(output_folder, output_subfolders['catchment'])
    elif output_type == 'grid':
        location_ids = [1]
        location_names = [project_name]
        output_subfolder = os.path.join(output_folder, output_subfolders['grid'])

    output_paths = {}
    output_cases = itertools.product(realisation_ids, location_ids, variables)
    for realisation_id, location_id, variable in output_cases:
        location_name = location_names[location_ids.index(location_id)]
        output_file_name = (
            # location_name + '_r' + str(realisation_id) + '_' + variable + output_format_extensions[output_format]
                location_name + '_r' + str(realisation_id) + output_format_extensions[output_format]
        )
        output_path_key = (output_type, location_id, realisation_id, variable)
        if variable == 'prcp':
            _output_subfolder = output_subfolder
        else:
            _output_subfolder = os.path.join(output_subfolder, variable)
        output_paths[output_path_key] = os.path.join(_output_subfolder, output_file_name)

        if not os.path.exists(_output_subfolder):
            os.makedirs(_output_subfolder)

    return output_paths


def identify_block_size(
        datetime_helper, season_definitions, timestep_length, discretisation_metadata,
        seed_sequence, number_of_years,
        spatial_model, parameters, intensity_distribution, xmin, xmax, ymin, ymax,
        output_types, points, catchments,
        float_precision, default_block_size, minimum_block_size, check_available_memory, maximum_memory_percentage,
        spatial_raincell_method, spatial_buffer_factor
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
        dth = datetime_helper.loc[
            (datetime_helper['year'] >= start_year) & (datetime_helper['year'] <= end_year),
        ].copy()
        dth['month_id'] = np.arange(dth.shape[0])
        month_lengths = dth['n_hours'].values
        # !221120
        # dummy1 = nsproc.main(
        #     spatial_model, parameters, sample_n_years, month_lengths, season_definitions, intensity_distribution,
        #     rng, xmin, xmax, ymin, ymax, spatial_raincell_method, spatial_buffer_factor,
        # )
        # !221120
        dummy1 = nsproc.main(  # main2
            spatial_model, parameters, sample_n_years, month_lengths, season_definitions, intensity_distribution,
            rng, xmin, xmax, ymin, ymax, spatial_raincell_method, spatial_buffer_factor,  # dth,
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
        spatial_model, output_types, discretisation_metadata, points, catchments, parameters,
        intensity_distribution, rng, xmin, xmax, ymin, ymax, output_paths, block_size, block_subset_size,
        spatial_raincell_method, spatial_buffer_factor, simulation_mode, weather_model, max_dsl=6.0,
        n_divisions=8, do_reordering=True,
        # TODO: Drop max_dsl if not using and drop default from n_divisions if using
        # - only needed in fitting, as part of parameters df otherwise
):
    """
    Simulate realisation of NSRP process.

    """
    # simulation_mode as 'no_shuffling', 'shuffling_preparation', 'with_shuffling'

    # Create a separate rng for shuffling to ensure consistency for a given input seed regardless of whether shuffling
    # is required
    rng2 = np.random.default_rng(rng.integers(1000000, 1000000000))

    # Initialise arrays according to discretisation method
    if simulation_mode != 'shuffling_preparation':
        discrete_rainfall = initialise_discrete_rainfall_arrays(
            spatial_model, output_types, discretisation_metadata, points, int((24 / timestep_length) * 31)
        )
    # TODO: Consider whether arrays for point or whole-domain event totals need to be initialised here

    # For shuffling preparation store event total dfs in a list for later recombination
    if simulation_mode == 'shuffling_preparation':
        storm_dfs = []  # !221025 - changed dfs to storm_dfs
        raincell_dfs = []  # !221025 - changed dfs to storm_dfs
        # dc1 = {}  # !221112 - contains fixed window aggregations for each point if spatial model
    dc1 = {}  # !221113 - testing si based on all points

    # In fitting max_dsl will be passed in explicitly, but otherwise it can be left as default (if no shuffling is
    # going to take place) or it can be taken from the parameters dataframe. Using value per season
    if 'max_dsl' in parameters.columns:  # !221025
        max_dsls = parameters['max_dsl'].values
    elif not isinstance(max_dsl, np.ndarray):
        max_dsls = np.repeat(max_dsl, len(season_definitions.keys()))
    else:
        max_dsls = max_dsl

    # Simulate and discretise NSRP process by block
    n_blocks = int(np.floor(number_of_years / block_size))
    if number_of_years % block_size != 0:
        n_blocks += 1
    block_id = 0
    # ***
    # 28/09/2022 - TESTING ONLY
    # block_id = 1
    # print(datetime_helper.columns)
    # sys.exit()
    # ***
    while block_id * block_size < number_of_years:
        if simulation_mode != 'shuffling_preparation':
            pass
            # print('  - Realisation =', realisation_id, '[Block =', str(block_id + 1) + '/' + str(n_blocks) + ']')
            # print('    - Sampling')

        # NSRP process simulation - allow for the final block to be less than the full block size
        block_start_year = datetime_helper['year'].values[0] + block_id * block_size
        block_end_year = block_start_year + block_size - 1
        block_end_year = min(block_end_year, datetime_helper['year'].values[-1])
        actual_block_size = min(block_size, block_end_year - block_start_year + 1)
        # !221120
        # month_lengths = datetime_helper.loc[
        #     (datetime_helper['year'] >= block_start_year) & (datetime_helper['year'] <= block_end_year),
        #     'n_hours'].values
        # df = nsproc.main(
        #     spatial_model, parameters, actual_block_size, month_lengths, season_definitions, intensity_distribution,
        #     rng, xmin, xmax, ymin, ymax, spatial_raincell_method, spatial_buffer_factor
        # )
        # !221120
        dth = datetime_helper.loc[
            (datetime_helper['year'] >= block_start_year) & (datetime_helper['year'] <= block_end_year),
        ].copy()
        dth['month_id'] = np.arange(dth.shape[0])
        month_lengths = dth['n_hours'].values
        df = nsproc.main(  # main2
            spatial_model, parameters, actual_block_size, month_lengths, season_definitions, intensity_distribution,
            rng, xmin, xmax, ymin, ymax, spatial_raincell_method, spatial_buffer_factor,  # dth,
        )

        # Convert raincell coordinates and radii from km to m for discretisation
        if 'raincell_x' in df.columns:
            df['raincell_x'] *= 1000.0
            df['raincell_y'] *= 1000.0
            df['raincell_radii'] *= 1000.0

        # /-/
        # !221103 - Fixed window approach in shuffling

        # +++
        # ! 221111 - shuffling with spatial model - coverage of domain needed by raincell for spatial mean

        if spatial_model:
            # Coverage of raincells of whole domain (to help get a spatial mean for the domain below)
            # - use 100x100 grid in discretisation
            df['raincell_coverage'] = get_raincell_coverage2(
                df['raincell_x'].values, df['raincell_y'].values, df['raincell_radii'].values, xmin, xmax, ymin, ymax,
                100, 100,
            )
            # !221112 - New bit for testing 06 - identify just those cells intersecting a (central) indicator location
            # xi = np.mean([xmin, xmax])
            # yi = np.mean([ymin, ymax])
            # df['raincell_coverage'] = np.where(
            #     ((df['raincell_x'] - xi) ** 2 + (df['raincell_y'] - yi) ** 2) ** 0.5 <= df['raincell_radii'],
            #     1.0,
            #     0.0
            # )
        else:
            df['raincell_coverage'] = 1.0

        # +++

        # Rationalised dataframe of raincells and total depths by window
        dth = datetime_helper.loc[
            (datetime_helper['year'] >= block_start_year) & (datetime_helper['year'] <= block_end_year)
        ]
        # df, df1 = rationalise_storms2(dth, df, n_divisions)
        df = rationalise_storms2(dth, df, n_divisions)

        # Spatial mean (domain-average) aggregated into fixed windows
        df1 = aggregate_windows(dth, df, n_divisions)

        # !221115 - TESTING
        # df1.to_csv('H:/Projects/rwgen/working/iss13/stnsrp/shuffling/df1_3a.csv', index=False)
        # sys.exit()

        # Dataframes aggregated into fixed windows for each point
        # !221115 (026) - commented out trying to reproduce 010
        if simulation_mode == 'shuffling_preparation':  # !221113 - commented out to test si based on all points
        # if simulation_mode != 'no_shuffling':
            if spatial_model:
                for _, row in points.iterrows():
                    xi = row['easting']
                    yi = row['northing']
                    df['raincell_coverage'] = np.where(
                        ((df['raincell_x'] - xi) ** 2 + (df['raincell_y'] - yi) ** 2) ** 0.5 <= df['raincell_radii'],
                        1.0,
                        0.0
                    )
                    dc1[row['point_id']] = aggregate_windows(dth, df, n_divisions)
                    # TODO: Remove dependence on there only being one block simulated during shuffling

        # /-/

        # Total storm depths if shuffling preparation - calculated here in case df is large
        # if simulation_mode != 'no_shuffling':
        #     df1 = get_storm_depths(df, spatial_model, rng2, xmin, xmax, ymin, ymax)
        if simulation_mode == 'shuffling_preparation':
            # storm_dfs.append(df1)  # !221025 - storm_dfs to dfs
            storm_dfs.append(df1)  # !221025 - storm_dfs to dfs
            raincell_dfs.append(df)  # !221025

            # Discretisation
        if simulation_mode != 'shuffling_preparation':

            # !221025 - moved from above
            # Shuffling storms
            if simulation_mode == 'with_shuffling':
                df = shuffling._shuffle_simulation(  # !221113 - original
                    df, df1, parameters, datetime_helper, rng2, n_divisions, do_reordering
                )
                # df = shuffling._shuffle_simulation_si(  # !221113 - testing si based on all points
                #     spatial_model, df, df1, dc1, parameters, datetime_helper, rng2, n_divisions, do_reordering
                # )

            discretise_by_point(
                spatial_model,
                datetime_helper.loc[
                    (datetime_helper['year'] >= block_start_year) & (datetime_helper['year'] <= block_end_year)
                    ],
                season_definitions, df, output_types, timestep_length,
                discrete_rainfall,
                discretisation_metadata, points, catchments, realisation_id, output_paths, block_id, block_subset_size,
                weather_model, simulation_mode,
            )
            # TODO: Check that slice of datetime_helper is correct

        block_id += 1

    # sys.exit()

    # Pass storm depths dataframe to shuffling fitting
    if simulation_mode == 'shuffling_preparation':
        storm_depths = pd.concat(storm_dfs)  # !221025
        raincells = pd.concat(raincell_dfs)
        # return storm_depths, raincells  # !221025
        return storm_depths, dc1, raincells  # !221112


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
        discretisation_metadata, points, catchments, realisation_id, output_paths, block_id, block_subset_size,
        weather_model, simulation_mode  # !221025 - added simulation_mode
):
    # print('    - Discretising = ', end='')  # TEMPORARY COMMENTING OUT  # TODO: Uncomment once weather generator working
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

    t1 = datetime.datetime.now()

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
        for month_idx in range(subset_start_idx, subset_end_idx + 1):  # range(datetime_helper.shape[0]):

            # if month_idx in print_helper:
            #     # print('    - Discretising:', str(print_helper.index(month_idx) * 10) + '%')  # OLD NOW?
            #     percent_complete = print_helper.index(month_idx) * 10
            #     if percent_complete <= 100:
            #         if percent_complete == 100:
            #             print(str(percent_complete) + '%')
            #         else:
            #             pass
            #             print(str(percent_complete) + '%', end=' ')  # TODO: Uncomment once weather generator working

            year = datetime_helper['year'].values[month_idx]
            month = datetime_helper['month'].values[month_idx]
            # !221025 - added clause to pass on season for now for shuffling development, but reconsider  # TODO
            # season = season_definitions[month]  # !221025 - commented out for now
            if month in season_definitions.keys():
                season = season_definitions[month]
            else:
                season = -999

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

                    # !221122 - ORIGINAL
                    discretise_spatial(
                        start_time, timestep_length, raincell_arrival_times, raincell_end_times,
                        raincell_intensities, discrete_rainfall[discretisation_case],
                        raincell_x, raincell_y, raincell_radii,
                        discretisation_metadata[(discretisation_case, 'x')],
                        discretisation_metadata[(discretisation_case, 'y')],
                        discretisation_metadata[(discretisation_case, 'phi', season)],
                    )

                    # !221122 - TESTING grid cell average discretisation
                    # if discretisation_case == 'point':
                    #     discretise_spatial(
                    #         start_time, timestep_length, raincell_arrival_times, raincell_end_times,
                    #         raincell_intensities, discrete_rainfall[discretisation_case],
                    #         raincell_x, raincell_y, raincell_radii,
                    #         discretisation_metadata[(discretisation_case, 'x')],
                    #         discretisation_metadata[(discretisation_case, 'y')],
                    #         discretisation_metadata[(discretisation_case, 'phi', season)],
                    #     )
                    # else:
                    #     cell_size = (
                    #         discretisation_metadata[(discretisation_case, 'x')][1]
                    #         - discretisation_metadata[(discretisation_case, 'x')][0]
                    #     )
                    #     discretise_spatial_with_gridcell_averages(
                    #         start_time, timestep_length, raincell_arrival_times, raincell_end_times,
                    #         raincell_intensities, discrete_rainfall[discretisation_case],
                    #         raincell_x, raincell_y, raincell_radii,
                    #         discretisation_metadata[(discretisation_case, 'x')],
                    #         discretisation_metadata[(discretisation_case, 'y')],
                    #         discretisation_metadata[(discretisation_case, 'phi', season)],
                    #         cell_size,
                    #     )

            else:
                discretise_point(
                    start_time, timestep_length, raincell_arrival_times, raincell_end_times,
                    raincell_intensities, discrete_rainfall['point'][:, 0]
                )

            # Find number of timesteps in month to be able to subset the discretised arrays (if < 31 days in month)
            timesteps_in_month = datetime_helper.loc[
                (datetime_helper['year'] == year) & (datetime_helper['month'] == month), 'n_timesteps'
            ].values[0]

            # ---
            # Need to simulate weather prior to doing catchment-averages
            if weather_model is not None:
                weather_model.simulate(
                    rainfall=discrete_rainfall,  # key = 'point' or 'grid', value = array (n timesteps, n points)
                    # rainfall_timestep=timestep_length,
                    n_timesteps=timesteps_in_month,
                    year=year,
                    month=month,
                    discretisation_metadata=discretisation_metadata,
                    output_types=output_types,
                    timestep=timestep_length,
                )
                # sys.exit()
            # ---

            # ///
            collate_output_arrays(
                output_types, spatial_model, points, catchments, realisation_id, discrete_rainfall, 'prcp',
                0, timesteps_in_month, discretisation_metadata, month_idx, output_arrays,
            )
            if weather_model is not None:
                weather_model.simulator.collate_outputs(
                    output_types, spatial_model, points, catchments, realisation_id, timesteps_in_month,
                    discretisation_metadata, month_idx,
                )
            # ///

            # ***
            # Put discrete rainfall in arrays ready for writing once all block available
            # for output_type in output_types:
            #     if output_type == 'point':
            #         if not spatial_model:
            #             location_ids = [1]
            #         else:
            #             location_ids = points['point_id'].values  # self.points['point_id'].values
            #     elif output_type == 'catchment':
            #         location_ids = catchments['id'].values  # self.catchments[self.catchment_id_field].values
            #     elif output_type == 'grid':
            #         location_ids = [1]
            #     # TODO: See if output keys can be looped directly without needing to figure out location_ids again
            #
            #     # TODO: Reduce dependence on list/array order
            #     idx = 0
            #     for location_id in location_ids:
            #         output_key = (output_type, location_id, realisation_id, 'prcp')
            #
            #         if output_type == 'point':
            #             output_array = discrete_rainfall['point'][:timesteps_in_month, idx]
            #         elif output_type == 'catchment':
            #             catchment_discrete_rainfall = np.average(
            #                 discrete_rainfall['grid'], axis=1,
            #                 weights=discretisation_metadata[('catchment', 'weights', location_id)]
            #             )
            #             output_array = catchment_discrete_rainfall[:timesteps_in_month]
            #         elif output_type == 'grid':
            #             raise NotImplementedError('Grid output not implemented yet')
            #
            #         # Appending an array to a list is faster than concatenating arrays
            #         if month_idx == 0:
            #             output_arrays[output_key] = [output_array.astype(np.float16)]
            #         else:
            #             output_arrays[output_key].append(output_array.astype(np.float16))
            #
            #         idx += 1
            # ***

        # Increment subset index tracker
        subset_start_idx += (subset_n_years * 12)

    # testing/checking
    # print(len(output_arrays.keys()))

    t2 = datetime.datetime.now()
    # print(t2 - t1)
    # sys.exit()

    # Write output
    if simulation_mode != 'shuffling_preparation':  # !221025
        # print('    - Writing')
        if block_id == 0:
            write_new_files = True
        else:
            write_new_files = False
        write_output(output_arrays, output_paths, write_new_files)

        # --
        if weather_model is not None:
            if weather_model.simulator.output_paths is None:
                weather_model.simulator.set_output_paths(weather_model.output_paths)
            weather_model.simulator.write_output(write_new_files)
        # --
    else:  # !221025
        return output_arrays


@numba.jit(nopython=True)  # , fastmath=True
def discretise_point(
        period_start_time, timestep_length, raincell_arrival_times, raincell_end_times,
        raincell_intensities, discrete_rainfall
):
    # Not parallelisable in numba as is, as could try to write to same timestep concurrently for > 1 raincell

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
        for timestep in range(rc_arrival_timestep, rc_end_timestep + 1):
            timestep_start_time = timestep * timestep_length
            timestep_end_time = (timestep + 1) * timestep_length
            effective_start = np.maximum(rc_arrival_time, timestep_start_time)
            effective_end = np.minimum(rc_end_time, timestep_end_time)
            timestep_coverage = effective_end - effective_start

            if timestep < discrete_rainfall.shape[0]:
                discrete_rainfall[timestep] += rc_intensity * timestep_coverage


@numba.jit(nopython=True)  # , parallel=True , fastmath=True
def discretise_spatial(
        period_start_time, timestep_length, raincell_arrival_times, raincell_end_times,
        raincell_intensities, discrete_rainfall,
        raincell_x_coords, raincell_y_coords, raincell_radii,
        point_eastings, point_northings, point_phi,  # point_ids,
):
    # Point loop should be parallelisable in numba, as each point has its own index in the second dimension of the
    # discrete_rainfall array

    # Modifying the discrete rainfall arrays themselves so need to ensure zeros before starting
    discrete_rainfall.fill(0.0)

    # Subset raincells based on whether they intersect the point being discretised
    for idx in range(point_eastings.shape[0]):
    # for idx in numba.prange(point_eastings.shape[0]):
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


def write_output(output_arrays, output_paths, write_new_files, number_format=None):
    for output_key, output_array_list in output_arrays.items():
        # output_type, location_id, realisation_id, variable = output_key

        if number_format is not None:
            _number_format = '%.1f'
        else:
            variable = output_key[3]
            if variable in ['prcp', 'tas']:
                _number_format = '%.1f'
            elif variable == 'pet':
                _number_format = '%.2f'
            else:
                _number_format = '%.2f'

        output_path = output_paths[output_key]
        values = []
        for output_array in output_array_list:
            for value in output_array:
                values.append((_number_format % value).rstrip('0').rstrip('.'))  # + '\n'
        output_lines = '\n'.join(values)
        if write_new_files:
            with open(output_path, 'w') as fh:
                fh.write(output_lines)
        else:
            with open(output_path, 'a') as fh:
                fh.write('\n' + output_lines)
        # TODO: Implement other text file output options


def collate_output_arrays(
        output_types, spatial_model, points, catchments, realisation_id, values, variable, timesteps_to_skip,
        timesteps_in_month, discretisation_metadata, month_idx, output_arrays,
):
    # - make more generic by not just working off discrete_rainfall - generalised to values
    # - also use arguments for beginning and end of time slice (i.e. not just end)
    # -- timesteps_to_skip will be zero for prcp, but two for other variables (currently)
    # - modifying output_arrays variable directly

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
            output_key = (output_type, location_id, realisation_id, variable)

            if output_type == 'point':
                if variable == 'prcp':
                    output_array = values['point'][:, idx]
                else:
                    output_array = values[('point', variable)][:, idx]
                output_array = output_array[timesteps_to_skip:timesteps_to_skip + timesteps_in_month]
            elif output_type == 'catchment':
                if variable == 'prcp':
                    catchment_discrete_rainfall = np.average(
                        values['grid'], axis=1,
                        weights=discretisation_metadata[('catchment', 'weights', location_id)]
                    )
                else:
                    catchment_discrete_rainfall = np.average(
                        values[('grid', variable)], axis=1,
                        weights=discretisation_metadata[('catchment', 'weights', location_id)]
                    )
                output_array = catchment_discrete_rainfall[timesteps_to_skip:timesteps_to_skip + timesteps_in_month]
            elif output_type == 'grid':
                raise NotImplementedError('Grid output not implemented yet')

            # Appending an array to a list is faster than concatenating arrays
            if month_idx == 0:
                output_arrays[output_key] = [output_array.astype(np.float16)]
            else:
                output_arrays[output_key].append(output_array.astype(np.float16))

            idx += 1


def get_storm_depths(df, spatial_model, rng, xmin, xmax, ymin, ymax):
    df['raincell_depth'] = df['raincell_duration'] * df['raincell_intensity']

    # Depth needs to be factored by fractional coverage of domain in spatial model
    if spatial_model:
        points_x = rng.uniform(xmin, xmax, 10000)
        points_y = rng.uniform(ymin, ymax, 10000)
        df['raincell_coverage'] = get_raincell_coverage(
            df['raincell_x'].values, df['raincell_y'].values, df['raincell_radii'].values, xmin, xmax, ymin, ymax,
            points_x, points_y
        )
        df['raincell_depth'] *= df['raincell_coverage']

    # ***
    # 03/10/2022 - Testing inclusion of storm duration

    # Sum raincell depths to get storm depths
    # ! df1 = df.groupby(['storm_id']).agg({'storm_arrival': min, 'season': min, 'raincell_depth': sum})  # 03/10/2022
    df1 = df.groupby(['storm_id']).agg({
        'storm_arrival': min, 'month': min, 'season': min, 'raincell_depth': sum, 'raincell_end': max
    })
    df1.reset_index(inplace=True)
    # ! df1.rename(columns={'raincell_depth': 'storm_depth'}, inplace=True)  # 03/10/2022
    df1.rename(columns={'raincell_depth': 'storm_depth', 'raincell_end': 'storm_end'}, inplace=True)

    # 03/10/2022
    df1['storm_duration'] = df1['storm_end'] - df1['storm_arrival']
    df1.drop(columns=['storm_end'], inplace=True)

    return df1


@numba.jit(nopython=True)
def get_raincell_coverage(rc_xs, rc_ys, rc_rads, xmin, xmax, ymin, ymax, points_x, points_y):
    domain_area = (xmax - xmin) * (ymax - ymin)

    # Initial area assuming all relevant - to be overwritten for partially overlapping cells
    areas = np.pi * rc_rads ** 2

    # Raincell loop to check/modify raincell areas array if only partly covering domain
    for i in range(rc_xs.shape[0]):
        rc_x = rc_xs[i]
        rc_y = rc_ys[i]
        rad = rc_rads[i]

        # Find minimum distance of centre to domain bounds for centres within domain
        if (rc_x >= xmin) and (rc_x <= xmax) and (rc_y >= ymin) and (rc_y <= ymax):
            min_distance = min(
                np.abs(rc_x - xmin),
                np.abs(rc_x - xmax),
                np.abs(rc_y - ymin),
                np.abs(rc_y - ymax)
            )
            centre_inside = True

        # Find minimum distance of centre to domain bounds for centres outside domain
        else:
            # Raincells within y-range but outside x-range
            if ((rc_y >= ymin) and (rc_y <= ymax)) and ((rc_x < xmin) or (rc_x > xmax)):
                min_distance = np.minimum(np.abs(rc_x - xmin), np.abs(rc_x - xmax))
            # Raincells within x-range but outside y-range
            elif ((rc_x >= xmin) and (rc_x <= xmax)) and ((rc_y < ymin) or (rc_y > ymax)):
                min_distance = np.minimum(np.abs(rc_y - ymin), np.abs(rc_y - ymax))
            # Raincells with x greater than xmax and y outside y-range
            elif (rc_x > xmax) and ((rc_y < ymin) or (rc_y > ymax)):
                min_distance = np.minimum(
                    ((rc_x - xmax) ** 2 + (rc_y - ymax) ** 2) ** 0.5, ((rc_x - xmax) ** 2 + (rc_y - ymin) ** 2) ** 0.5
                )
            # Raincells with x less than xmin and y outside y-range
            elif (rc_x < xmin) and ((rc_y < ymin) or (rc_y > ymax)):
                min_distance = np.minimum(
                    ((rc_x - xmin) ** 2 + (rc_y - ymax) ** 2) ** 0.5, ((rc_x - xmin) ** 2 + (rc_y - ymin) ** 2) ** 0.5
                )
            centre_inside = False

        # Calculate area inside domain if relevant
        if centre_inside and (rad <= min_distance):
            pass
        else:
            inside_count = 0
            outside_count = 0
            for x, y in zip(points_x, points_y):
                d = ((x - rc_x) ** 2 + (y - rc_y) ** 2) ** 0.5
                if d <= rad:
                    inside_count += 1
                else:
                    outside_count += 1
            areas[i] = (inside_count / (inside_count + outside_count)) * domain_area

    frac_areas = areas / domain_area

    return frac_areas


@numba.jit(nopython=True)
def rationalise_storms(n_raincells, arrivals, ends, seasons, block_end_time, max_dsls):
    # arrivals = df.iloc[arrival_col].values
    # ends = df.iloc[end_col].values

    # If called during fitting max_dsl will be a single value, but during simulation it will be an array with a value
    # per season (currently hardcoded as monthly). Perhaps easiest to bring in a season array to help determine the
    # relevant max_dsl to use

    # new_ids = np.zeros(n_raincells, dtype=int)  # TESTING ONLY
    new_ids = np.zeros(n_raincells, dtype=numba.int64)
    storm_id = 0
    prev_end = -999.0
    for i in range(n_raincells):
        arrival = arrivals[i]
        end = ends[i]
        season = seasons[i]
        max_dsl = max_dsls[season - 1]
        if i == 0:
            prev_end = end
        else:
            if arrival > prev_end:
                if arrival < block_end_time:
                    if (arrival - prev_end) > max_dsl:  # !221025
                        storm_id += 1
                    else:
                        pass
                prev_end = end
            else:
                prev_end = max(prev_end, end)
        new_ids[i] = storm_id

    return new_ids


def rationalise_storms2(datetime_helper, df, n_divisions):
    # df is df of raincells - operating on it...
    # print(df)
    # print(df.columns)
    # sys.exit()

    # print(datetime_helper)
    # print(datetime_helper.columns)
    # sys.exit()

    # Make a datetime_helper containing the starts and ends of the windows
    # print(datetime_helper)
    tmp = pd.concat([datetime_helper for _ in range(n_divisions)])
    tmp['window'] = np.repeat(np.arange(n_divisions), datetime_helper.shape[0])
    tmp.sort_values(['year', 'month', 'window'], inplace=True)
    tmp['window_start'] = tmp['n_hours'] / float(n_divisions) * tmp['window'] + tmp['start_time'] - tmp['start_time'].min()
    tmp['window_end'] = tmp['window_start'].shift(-1)
    tmp.iloc[-1, tmp.columns.get_loc('window_end')] = tmp['end_time'].values[-1] - tmp['start_time'].min()
    # print(tmp)
    # print(datetime_helper.columns)
    # sys.exit()

    # Ensure that minimum arrival time is greater than zero (for binning) and maximum end if less than block
    # df['raincell_arrival'] = np.where(df['raincell_arrival'] == 0.0, 0.00001, df['raincell_arrival'])
    df = df.loc[df['raincell_arrival'] < tmp['window_end'].max()].copy()  # hopefully no memory issues...
    df['raincell_end'] = np.where(
        df['raincell_end'] > tmp['window_end'].max(), tmp['window_end'].max(), df['raincell_end']
    )

    # For each raincell, identify the window in which the arrival occurs
    # year_idx = np.digitize(df['storm_arrival'], period_ends, right=True)
    df['start_win_idx'] = np.digitize(df['raincell_arrival'], tmp['window_end'], right=True)

    # For each raincell, identify the window in which the end occurs
    df['end_win_idx'] = np.digitize(df['raincell_end'], tmp['window_end'], right=True)

    # print(np.unique(df['start_win_idx']).shape[0], tmp.shape[0])
    # print(np.min(df['start_win_idx']), np.max(df['start_win_idx']))
    # sys.exit()

    # Check for a difference between the arrival and end windows
    # - if so the raincell needs to be "duplicated" that number of times
    # - only a very long raincell might span enough days to get n_windows > 2
    df['n_windows'] = df['end_win_idx'] - df['start_win_idx'] + 1
    df['duplicate_id'] = 0
    if df['n_windows'].max() > 1:
        df_subs = [df]
        for n_wins in np.unique(df['n_windows'][df['n_windows'] > 1]):
            # print(n_wins)
            for dup_id in range(1, n_wins):
                df_sub = df.loc[df['n_windows'] == n_wins].copy()
                df_sub['duplicate_id'] = dup_id
                df_subs.append(df_sub)
        df = pd.concat(df_subs)
    df.sort_values(['storm_id', 'raincell_arrival'], inplace=True)  # sort on raincell_arrival / window?

    # CHECKING - should be ok now that ensure only raincells arriving before period end are considered
    if df['n_windows'].min() < 1:
        print(df.loc[df['n_windows'] < 1])
        sys.exit()
    # assert df['n_windows'].min() > 0

    # print(df)
    # print(df.loc[df['n_windows'] > 1, ['raincell_arrival', 'raincell_end', 'start_win_idx', 'end_win_idx', 'duplicate_id']])
    # print(df['n_windows'].min())
    # print(df['n_windows'].max())
    # print(df['duplicate_id'].max())
    # print(df.columns)
    # print(df.loc[df['n_windows'] > 1, ['raincell_arrival', 'raincell_end', 'start_win_idx', 'end_win_idx']])
    # sys.exit()

    # For "duplicated" raincells, update starts/ends in line with window starts/ends
    df['start_win_idx2'] = df['start_win_idx'] + df['duplicate_id']
    df['end_win_idx2'] = df['start_win_idx2']
    df['start_win'] = tmp['window_start'].values[df['start_win_idx2']]
    df['end_win'] = tmp['window_end'].values[df['end_win_idx2']]
    df['raincell_arrival'] = np.where(df['raincell_arrival'] < df['start_win'], df['start_win'], df['raincell_arrival'])
    df['raincell_end'] = np.where(df['raincell_end'] > df['end_win'], df['end_win'], df['raincell_end'])
    df['raincell_duration'] = df['raincell_end'] - df['raincell_arrival']
    df['win_length'] = df['end_win'] - df['start_win']
    # df['raincell_depth'] = df['raincell_intensity'] * df['raincell_duration']
    df['year'] = tmp['year'].values[df['start_win_idx2']]
    df['month'] = tmp['month'].values[df['start_win_idx2']]
    df.drop(columns=[
        'start_win_idx', 'end_win_idx', 'n_windows', 'duplicate_id', 'end_win_idx2', 'end_win'],  # 'start_win',
        inplace=True
    )
    df.rename(columns={'start_win_idx2': 'win_id', 'start_win': 'win_start'}, inplace=True)

    # print(df.loc[:, ['raincell_arrival', 'raincell_end', 'start_win', 'end_win']])
    # print(df.loc[:, ['raincell_arrival', 'raincell_end', 'start_win_idx2', 'end_win_idx2']])
    # print(df)
    # print(np.unique(df['win_length']))
    # print(df.columns)

    # print(df[['year', 'month', 'win_start', 'raincell_arrival', 'win_id']])
    # sys.exit()

    # # ///
    #
    # # Depths for windows (equivalent to storm depths but for fixed windows)
    # df_win = df.groupby(['win_id'])['year', 'month', 'win_start', 'win_length', 'raincell_depth'].agg({
    #     'year': min, 'month': min, 'win_start': min, 'win_length': min, 'raincell_depth': sum
    # })
    # df_win.reset_index(inplace=True)
    # df_win.rename(columns={'raincell_depth': 'win_depth'}, inplace=True)
    #
    # # Ensure serially complete
    # # - assuming only window depth df (not raincell df) needs to be serially complete for now
    # # print(df_win.shape[0])
    # tmp1 = pd.DataFrame({
    #     'win_id': np.arange(tmp.shape[0]), 'year': tmp['year'], 'month': tmp['month'], 'win_start': tmp['window_start'],
    #     'win_length': tmp['window_end'] - tmp['window_start']}
    # )
    # df_win = df_win.merge(tmp1, how='outer')
    # df_win['win_depth'] = np.where(~np.isfinite(df_win['win_depth']), 0.0, df_win['win_depth'])
    # df_win.sort_values('win_id', inplace=True)
    #
    # # ///

    # print(df_win)
    # sys.exit()

    return df  # , df_win


def aggregate_windows(datetime_helper, df, n_divisions):
    # Make a datetime_helper containing the starts and ends of the windows
    # print(datetime_helper)
    tmp = pd.concat([datetime_helper for _ in range(n_divisions)])
    tmp['window'] = np.repeat(np.arange(n_divisions), datetime_helper.shape[0])
    tmp.sort_values(['year', 'month', 'window'], inplace=True)
    tmp['window_start'] = tmp['n_hours'] / float(n_divisions) * tmp['window'] + tmp['start_time'] - tmp[
        'start_time'].min()
    tmp['window_end'] = tmp['window_start'].shift(-1)
    tmp.iloc[-1, tmp.columns.get_loc('window_end')] = tmp['end_time'].values[-1] - tmp['start_time'].min()

    # Depths for windows (equivalent to storm depths but for fixed windows)
    df['raincell_depth'] = df['raincell_intensity'] * df['raincell_duration'] * df['raincell_coverage']  # !221111
    df_win = df.groupby(['win_id'])['year', 'month', 'win_start', 'win_length', 'raincell_depth'].agg({
        'year': min, 'month': min, 'win_start': min, 'win_length': min, 'raincell_depth': sum
    })
    df_win.reset_index(inplace=True)
    df_win.rename(columns={'raincell_depth': 'win_depth'}, inplace=True)

    # Ensure serially complete
    # - assuming only window depth df (not raincell df) needs to be serially complete for now
    # print(df_win.shape[0])
    tmp1 = pd.DataFrame({
        'win_id': np.arange(tmp.shape[0]), 'year': tmp['year'], 'month': tmp['month'], 'win_start': tmp['window_start'],
        'win_length': tmp['window_end'] - tmp['window_start']}
    )
    df_win = df_win.merge(tmp1, how='outer')
    df_win['win_depth'] = np.where(~np.isfinite(df_win['win_depth']), 0.0, df_win['win_depth'])
    df_win.sort_values('win_id', inplace=True)

    return df_win


# @numba.jit(nopython=True)
@numba.jit(nopython=True, parallel=True)
def get_raincell_coverage2(rc_xs, rc_ys, rc_rads, xmin, xmax, ymin, ymax, nx, ny):
    # assumes everything is coming in with units of metres

    domain_area = (xmax - xmin) * (ymax - ymin)

    # Initial area assuming all relevant - to be overwritten for partially overlapping cells
    areas = np.pi * rc_rads ** 2

    # Identify cells with centres inside domain
    # centre_inside = (rc_xs >= xmin) & (rc_xs <= xmax) & (rc_ys >= ymin) & (rc_ys <= ymax)

    # Identify if entirely within domain (maximum distance to domain bounds less than radius)
    #  TODO: See notes - need to check bounding x and y range
    # - x and y range etc can be calculated in a vectorised way upfront - reduce amount of stuff calculated in the loop
    rc_xmin = rc_xs - rc_rads
    rc_xmax = rc_xs + rc_rads
    rc_ymin = rc_ys - rc_rads
    rc_ymax = rc_ys + rc_rads
    inside_domain = (rc_xmin >= xmin) & (rc_xmax <= xmax) & (rc_ymin >= ymin) & (rc_ymax <= ymax)

    # Identify corners of intersecting rectangle of domain and the square that bounds the raincell circle
    sub_xmin = np.maximum(rc_xmin, xmin)
    sub_xmax = np.minimum(rc_xmax, xmax)
    sub_ymin = np.maximum(rc_ymin, ymin)
    sub_ymax = np.minimum(rc_ymax, ymax)

    # Raincell loop to check/modify raincell areas array if only partly covering domain
    # for i in range(rc_xs.shape[0]):
    for i in numba.prange(rc_xs.shape[0]):
        inside = inside_domain[i]

        if not inside:
            rc_x = rc_xs[i]
            rc_y = rc_ys[i]
            rad = rc_rads[i]
            _sub_xmin = sub_xmin[i]
            _sub_xmax = sub_xmax[i]
            _sub_ymin = sub_ymin[i]
            _sub_ymax = sub_ymax[i]

            # Identify area covered by one grid cell in the intersecting rectangle
            # - use less points in grid spacing is very small
            # nx = 100  # 1000
            # ny = 100  # 1000
            dx = (_sub_xmax - _sub_xmin) / nx  # float(nx)
            dy = (_sub_ymax - _sub_ymin) / ny  # float(ny)
            # if dx < 1.0:
            #     nx = 100
            #     dx = (sub_xmax - sub_xmin) / float(nx)
            # if dy < 1.0:
            #     ny = 100
            #     dy = (sub_ymax - sub_ymin) / float(ny)
            # dxdy = dx * dy

            # Count grid cells within raincell
            grid_x = np.linspace(_sub_xmin + dx / 2.0, _sub_xmax - dx / 2.0, nx)
            grid_y = np.linspace(_sub_ymin + dy / 2.0, _sub_ymax - dy / 2.0, ny)
            # grid_count = np.sum((((_x - rc_x) ** 2.0 + (_y - rc_y) ** 2.0) ** 0.5) < rad)
            grid_count = 0
            for x in grid_x:
                for y in grid_y:
                    d = ((x - rc_x) ** 2 + (y - rc_y) ** 2) ** 0.5
                    if d < rad:
                        grid_count += 1
            areas[i] = grid_count * dx * dy

    frac_areas = areas / domain_area

    return frac_areas


# ---
# !221122 - TESTING USE OF GRID CELL AVERAGES IN DISCRETISATION

@numba.jit(nopython=True)
def discretise_gridcell(
        period_start_time, timestep_length, raincell_arrival_times, raincell_end_times,
        raincell_intensities, spatial_frac_cover, discrete_rainfall
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
        for timestep in range(rc_arrival_timestep, rc_end_timestep + 1):
            timestep_start_time = timestep * timestep_length
            timestep_end_time = (timestep + 1) * timestep_length
            effective_start = np.maximum(rc_arrival_time, timestep_start_time)
            effective_end = np.minimum(rc_end_time, timestep_end_time)
            timestep_coverage = effective_end - effective_start

            if timestep < discrete_rainfall.shape[0]:
                discrete_rainfall[timestep] += rc_intensity * timestep_coverage * spatial_frac_cover[idx]


@numba.jit(nopython=True)
def discretise_spatial_with_gridcell_averages(
        period_start_time, timestep_length, raincell_arrival_times, raincell_end_times,
        raincell_intensities, discrete_rainfall,
        raincell_x_coords, raincell_y_coords, raincell_radii,
        point_eastings, point_northings, point_phi, cell_size  # i.e. grid cell size
):
    # Assuming that grid cell width and length are equal

    # Buffer term for identifying potentially relevant raincells
    buffer_length = (cell_size ** 2 + cell_size ** 2) ** 0.5
    buffered_raincell_radii = raincell_radii + buffer_length

    # Modifying the discrete rainfall arrays themselves so need to ensure zeros before starting
    discrete_rainfall.fill(0.0)

    # Subset raincells based on whether they intersect the point being discretised
    for idx in range(point_eastings.shape[0]):
        x = point_eastings[idx]
        y = point_northings[idx]
        yi = idx

        # Find raincells that intersect the gridcell
        distances_from_raincell_centres = np.sqrt((x - raincell_x_coords) ** 2 + (y - raincell_y_coords) ** 2)
        spatial_mask = distances_from_raincell_centres <= buffered_raincell_radii

        # Find fractional coverage of gridcell by each raincell
        xmin = x - cell_size * 0.5
        xmax = x + cell_size * 0.5
        ymin = y - cell_size * 0.5
        ymax = y + cell_size * 0.5
        frac_cover = get_raincell_coverage2(
            raincell_x_coords[spatial_mask], raincell_y_coords[spatial_mask], raincell_radii[spatial_mask], xmin, xmax,
            ymin, ymax, 10, 10,
        )

        # Discretise grid cell obtaining average
        discretise_gridcell(
            period_start_time, timestep_length, raincell_arrival_times[spatial_mask],
            raincell_end_times[spatial_mask], raincell_intensities[spatial_mask], frac_cover, discrete_rainfall[:, yi]
        )

        discrete_rainfall[:, yi] *= point_phi[idx]
