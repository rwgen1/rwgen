import os
import sys
import itertools
import datetime

import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
import scipy.interpolate
import scipy.spatial
import geopandas
import gstools
import numba

from . import utils
from . import properties


class Preprocessor(object):

    def __init__(
            self,
            season_definitions=None,
            statistic_definitions=None,
            statistic_definitions_path=None,
            timeseries_format=None,
            calculation_period=None,
            completeness_threshold=80.0,
            output_folder=None,
            point_statistics_filename=None,
            cross_correlation_filename=None,
            phi_filename=None,
            override_phi=False,
            outlier_method=None,
            maximum_relative_difference=2.0,
            maximum_alterations=5
    ):
        self.season_definitions = utils.parse_season_definitions(season_definitions)

        if statistic_definitions is not None:
            self.statistic_definitions = statistic_definitions
        elif statistic_definitions_path is not None:
            self.statistic_definitions = utils.read_statistic_definitions(statistic_definitions_path)
        else:
            self.statistic_definitions = None
        
        self.timeseries_format = timeseries_format
        self.calculation_period = calculation_period
        self.completeness_threshold = completeness_threshold
        self.output_folder = output_folder
        self.point_statistics_filename = point_statistics_filename
        self.cross_correlation_filename = cross_correlation_filename
        self.phi_filename = phi_filename
        self.override_phi = override_phi

        self.outlier_method = outlier_method
        self.maximum_relative_difference = maximum_relative_difference
        self.maximum_alterations = maximum_alterations
        
        self.statistics = None  # ~output/return
        self.phi = None  # ~output/return
    
    def check_statistics_include_24hr_mean(self):
        includes_24hr_mean = self.statistic_definitions.loc[
            (self.statistic_definitions['name'] == 'mean')
            & (self.statistic_definitions['duration'] == 24)
        ].shape[0]
        if not includes_24hr_mean:
            tmp = pd.DataFrame({
                'statistic_id': [int(np.max(self.statistic_definitions['statistic_id'])) + 1],
                'weight': [0], 'duration': [24], 'name': ['mean'], 'lag': ['NA'],
                'threshold': ['NA']
            })
            self.statistic_definitions = pd.concat([self.statistic_definitions, tmp])

    def run(self):
        raise NotImplementedError
    
    @property
    def unique_seasons(self):
        if self.season_definitions is not None:
            return list(set(self.season_definitions.values()))
        else:
            return None

    def write_statistics(self):
        if self.point_statistics_filename is not None:
            point_path = os.path.join(self.output_folder, self.point_statistics_filename)
        else:
            point_path = os.path.join(self.output_folder, 'reference_point_statistics.csv')

        if 'cross-correlation' in self.statistic_definitions['name'].tolist():
            if self.cross_correlation_filename is not None:
                cross_correlation_path = os.path.join(self.output_folder, self.cross_correlation_filename)
            else:
                cross_correlation_path = os.path.join(self.output_folder, 'reference_cross_correlations.csv')
        else:
            cross_correlation_path = None

        utils.write_statistics(self.statistics, point_path, self.season_definitions, cross_correlation_path)

        # If spatial case (more than one point) then also write phi
        if 'point_id' in self.statistics.columns:
            if self.phi_filename is not None:
                phi_path = os.path.join(self.output_folder, self.phi_filename)
            else:
                phi_path = os.path.join(self.output_folder, 'phi.csv')
            phi = pd.merge(self.metadata, self.phi, how='outer', on=['point_id'])
            utils.write_phi(phi, phi_path)


class Fitter(object):

    # - need to adjust so that it can take point statistics and cross-correlations from file as inputs
    # - also options to do two-step with smoothing
    # - plus set bounds
    # - plus begin from a set of starting parameters in two-step
    # - plus option for some parameters to remain fixed

    def __init__(
            self,
            season_definitions=None,
            reference_statistics=None,
            reference_statistics_path=None,
            output_folder=None,
            parameters_filename=None,
            point_statistics_filename=None,
            cross_correlation_filename=None
            # fitting_method='by_season',  # 'harmonics'
            # optimisation_algorithm='differential_evolution',  # ...
            # dry_probability_resolution_correction=True,
            # dry_probability_bias_correction=False,
    ):
        self.season_definitions = utils.parse_season_definitions(season_definitions)
        if reference_statistics is not None:
            self.reference_statistics = reference_statistics
        else:
            self.reference_statistics = utils.read_statistics(reference_statistics_path)

        self.output_folder = output_folder
        if self.output_folder is not None:
            if parameters_filename is not None:
                self.parameters_path = os.path.join(self.output_folder, parameters_filename)
            else:
                self.parameters_path = os.path.join(self.output_folder, 'parameters.csv')
            if point_statistics_filename is not None:
                self.point_statistics_path = os.path.join(self.output_folder, point_statistics_filename)
            else:
                self.point_statistics_path = os.path.join(self.output_folder, 'fitted_point_statistics.csv')
            if cross_correlation_filename is not None:
                self.cross_correlation_path = os.path.join(self.output_folder, cross_correlation_filename)
            else:
                self.cross_correlation_path = os.path.join(self.output_folder, 'fitted_cross_correlations.csv')
        else:
            self.parameters_path = None
            self.point_statistics_path = None
            self.cross_correlation_path = None

        self.parameter_output_renaming = {
            'number_of_iterations': 'iterations',
            'number_of_evaluations': 'function_evaluations',
            'fit_success': 'converged',
        }

        # Subclass attributes
        self.parameter_names = None
        self.parameter_bounds = None
        self.parameter_output_columns = None

        self.parameters = None

        # fitting_method='by_season',  # 'harmonics'
        # optimisation_algorithm='differential_evolution',  # ...
        # dry_probability_resolution_correction=True,
        # dry_probability_bias_correction=False,

    def run(self):
        t1 = datetime.datetime.now()
        # self.fit_by_season()
        self.fit_with_smoothing()
        t2 = datetime.datetime.now()
        print(' ', t2 - t1)

    def fit_by_season(self, seasonal_parameter_bounds=None):
        results = {}
        fitted_statistics = []
        for season in self.unique_seasons:
            if len(self.unique_seasons) == 12:
                print('    - Month =', season)
            else:
                print('    - Season =', season)

            if seasonal_parameter_bounds is None:
                parameter_bounds = self.parameter_bounds  # should be self.default_parameter_bounds
            else:
                parameter_bounds = seasonal_parameter_bounds[season]

            season_reference_statistics = self.reference_statistics.loc[
                self.reference_statistics['season'] == season
            ].copy()

            statistic_ids, fitting_data, ref, weights, gs = self.prepare(season_reference_statistics)

            result = scipy.optimize.differential_evolution(
                func=self.fitting_wrapper,
                bounds=parameter_bounds,  # self.parameter_bounds,
                args=(
                    statistic_ids,
                    fitting_data,
                    ref,
                    weights,
                    gs
                ),
                tol=0.001,  # 0.01,  #
                updating='deferred',
                workers=6  # 4
            )
            for idx in range(len(self.parameter_names)):
                results[(self.parameter_names[idx], season)] = result.x[idx]
            results[('fit_success', season)] = result.success
            results[('objective_function', season)] = result.fun
            results[('number_of_iterations', season)] = result.nit
            results[('number_of_evaluations', season)] = result.nfev

            # Get and store statistics associated with optimised parameters
            dfs = []
            parameters = []
            for parameter in self.parameter_names:
                parameters.append(results[(parameter, season)])
            mod_stats = self.analytical_properties_wrapper(parameters, statistic_ids, fitting_data)
            for statistic_id in statistic_ids:
                tmp = fitting_data[(statistic_id, 'df')].copy()
                dfs.append(tmp)
            df = pd.concat(dfs)
            df['value'] = mod_stats
            df['season'] = season
            fitted_statistics.append(df)

        # Format results and write output tables (parameter values and fitted statistics)
        self.parameters = self.format_results(results)
        if self.output_folder is not None:
            df = self.parameters[self.parameter_output_columns]
            utils.write_csv_(df, self.parameters_path, self.season_definitions, self.parameter_output_renaming)

            fitted_statistics = pd.concat(fitted_statistics)
            utils.write_statistics(
                fitted_statistics, self.point_statistics_path, self.season_definitions, self.cross_correlation_path,
                write_weights=False, write_gs=False, write_phi=False
            )

    # ---
    # Experimental fitting methods

    def fit_with_smoothing(self):
        # - do we want to write the results of the first step? yes - might be useful to inspect
        # - also useful to be able to start from a parameter file, rather than necessarily having to do the first step
        # - should we be operating on self.parameter_bounds or not?

        # First step is a normal season-wise fit
        # self.fit_by_season()

        # ---
        # Read starting parameters from here for now
        self.parameters = utils.read_csv_('H:/Projects/rwgen/examples/stnsrp/output/parameters.csv')
        # self.parameters = utils.read_csv_('H:/Projects/rwgen/examples/nsrp/output2/parameters.csv')
        # ---

        # Then smooth the parameter values using a +/-1 month (season?) weighted moving average

        # Insert (repeat) final season at beginning of df and first season at end to avoid boundary effects
        tmp1 = self.parameters.loc[self.parameters['season'] == 12].copy()
        tmp1.loc[:, 'season'] = 0
        tmp2 = self.parameters.loc[self.parameters['season'] == 1].copy()
        tmp2.loc[:, 'season'] = max(self.unique_seasons) + 1
        df = pd.concat([self.parameters, tmp1, tmp2])
        df.sort_values('season', inplace=True)

        # Weighted moving average
        def weighted_average(x):
            return (x.values[0] * 0.5 + x.values[1] + x.values[2] * 0.5) / 2.0
        df1 = df.rolling(window=3, center=True, on='season').apply(weighted_average)
        df1 = df1.loc[(df1['season'] >= min(self.unique_seasons)) & (df1['season'] <= max(self.unique_seasons))]

        # # Define new bounds for optimisation by season
        # seasonal_parameter_bounds = {}
        # for season in self.unique_seasons:
        #     seasonal_parameter_bounds[season] = []
        #     for parameter in self.parameter_names:
        #         parameter_idx = self.parameter_names.index(parameter)
        #         smoothed_initial_value = df1.loc[df1['season'] == season, parameter].values[0]
        #         offset = smoothed_initial_value * 0.25
        #         lower_bound = max(smoothed_initial_value - offset, self.parameter_bounds[parameter_idx][0])
        #         upper_bound = min(smoothed_initial_value + offset, self.parameter_bounds[parameter_idx][1])
        #         seasonal_parameter_bounds[season].append((lower_bound, upper_bound))

        # ALTERNATIVE Define new bounds for optimisation by season using fraction of annual mean of smoothed parameter
        seasonal_parameter_bounds = {}
        for season in self.unique_seasons:
            seasonal_parameter_bounds[season] = []
            for parameter in self.parameter_names:
                parameter_idx = self.parameter_names.index(parameter)
                parameter_mean = df1[parameter].mean()
                offset = parameter_mean * 0.15  # 0.25
                smoothed_initial_value = df1.loc[df1['season'] == season, parameter].values[0]
                lower_bound = max(smoothed_initial_value - offset, self.parameter_bounds[parameter_idx][0])
                upper_bound = min(smoothed_initial_value + offset, self.parameter_bounds[parameter_idx][1])
                seasonal_parameter_bounds[season].append((lower_bound, upper_bound))

        # Refit by season with refined bounds
        # - temporarily adjust output file names here - needs to be rationalised
        self.parameters_path = self.parameters_path.replace('.csv', '3_15.csv')
        self.point_statistics_path = self.point_statistics_path.replace('.csv', '3_15.csv')
        self.cross_correlation_path = self.cross_correlation_path.replace('.csv', '3_15.csv')
        self.fit_by_season(seasonal_parameter_bounds)

    # ---

    @staticmethod
    def format_results(results):
        df = pd.DataFrame.from_dict(results, orient='index', columns=['value'])
        df.index = pd.MultiIndex.from_tuples(df.index, names=['field', 'season'])
        df.reset_index(inplace=True)
        df = df.pivot(index='season', columns='field', values='value')
        df.reset_index(inplace=True)
        return df

    @property
    def unique_seasons(self):
        if self.season_definitions is not None:
            return list(set(self.season_definitions.values()))
        else:
            return None

    @classmethod
    def fitting_wrapper(cls, parameters, statistic_ids, fitting_data, ref_stats, weights, gs):
        mod_stats = cls.analytical_properties_wrapper(parameters, statistic_ids, fitting_data)
        obj_fun = Fitter.calculate_objective_function(ref_stats, mod_stats, weights, gs)
        return obj_fun

    @classmethod
    def analytical_properties_wrapper(cls, parameters, statistic_ids, fitting_data):
        pass

    @staticmethod
    def prepare(statistics):
        statistic_ids = sorted(list(set(statistics['statistic_id'])))

        fitting_data = {}
        reference_statistics = []
        weights = []
        gs = []
        for statistic_id in statistic_ids:
            df = statistics.loc[statistics['statistic_id'] == statistic_id].copy()

            fitting_data[(statistic_id, 'name')] = df['name'].values[0]
            fitting_data[(statistic_id, 'duration')] = df['duration'].values[0]
            fitting_data[(statistic_id, 'lag')] = df['lag'].values[0]
            fitting_data[(statistic_id, 'threshold')] = df['threshold'].values[0]
            fitting_data[(statistic_id, 'df')] = df

            reference_statistics.append(df['value'].values)
            weights.append(df['weight'].values)
            gs.append(df['gs'].values)

        reference_statistics = np.concatenate(reference_statistics)
        weights = np.concatenate(weights)
        gs = np.concatenate(gs)

        return statistic_ids, fitting_data, reference_statistics, weights, gs

    @staticmethod
    def calculate_analytical_properties(statistic_ids, fitting_data, lamda, beta, eta, xi, nu, gamma=None):
        mu_1 = 1.0 / xi
        mu_2 = 2.0 / xi ** 2.0
        mu_3 = 6.0 / xi ** 3.0

        statistic_arrays = []
        for statistic_id in statistic_ids:
            name = fitting_data[(statistic_id, 'name')]
            duration = fitting_data[(statistic_id, 'duration')]
            phi = fitting_data[(statistic_id, 'df')]['phi'].values

            if name in ['autocorrelation', 'cross-correlation']:
                lag = fitting_data[(statistic_id, 'lag')]
                if name == 'cross-correlation':
                    phi2 = fitting_data[(statistic_id, 'df')]['phi2'].values
                    distances = fitting_data[(statistic_id, 'df')]['distance'].values
            elif name == 'probability_dry':
                threshold = fitting_data[(statistic_id, 'threshold')]

            if name == 'mean':
                values = properties.calculate_mean(duration, lamda, nu, mu_1, eta, phi)
            elif name == 'variance':
                values = properties.calculate_variance(
                    duration, eta, beta, lamda, nu, mu_1, mu_2, phi
                )
            elif name == 'skewness':
                values = properties.calculate_skewness(
                    duration, eta, beta, lamda, nu, mu_1, mu_2, mu_3, phi
                )
            elif name == 'autocorrelation':
                values = properties.calculate_autocorrelation(
                    duration, lag, eta, beta, lamda, nu, mu_1, mu_2, phi
                )
            elif name == 'probability_dry':
                values = properties.calculate_probability_dry(
                    duration, nu, beta, eta, lamda, phi, threshold
                )
            elif name == 'cross-correlation':
                values = properties.calculate_cross_correlation(
                    duration, lag, eta, beta, lamda, nu, mu_1, mu_2, gamma, distances, phi, phi2
                )

            statistic_arrays.append(values)

        return np.concatenate(statistic_arrays)

    @staticmethod
    def calculate_objective_function(ref, mod, w, sf):
        obj_fun = np.sum((w ** 2 / sf ** 2) * (ref - mod) ** 2)
        return obj_fun


class Simulator(object):

    # - need phi for each month / point to be discretised
    # -- option to interpolate if not a gauge point
    # -- option to use / interpolate a field
    # -- best handled in simulator rather than preprocessor? - or not?
    # --- simulator has to sort out stuff re catchments etc, so why not phi too!
    # --- similar thing with weighting grids for catchment-average output...
    # --- is preprocessor really just a fitting preprocessor? - probably ok though...

    # - seed and rng for random number generation

    # ! probably next need to sort out point/catchment/grid details to finalise discretisation calls !
    # ! also test all discretisation !

    # - need to know catchment id field

    # - perhaps no defaults are needed if no expectation of instanstiating superclass?

    # - should xmin, max etc be set in here as defaults? - perhaps yes...

    # - perhaps some of init could go into helper functions etc

    def __init__(
            self,
            output_types=None,
            output_folder=None,  # could also be dictionary for paths to point, catchment, grid folders...
            output_format=None,  # either string or dictionary? {'point': 'csv', 'catchment': 'txt'} ?
            output_prefix=None,
            season_definitions=None,
            process_class=None,
            parameters=None,
            points=None,
            catchments=None,
            catchment_id_field=None,
            grid=None,  # dictionary {'ncols': 10, 'nrow': 10, ...}
            cell_size=None,  # of grid for discretisation
            dem=None,  # path to ascii raster [or xarray data array]
            phi=None,  # phi df, path to phi df [or xarray data array]
            number_of_years=30,  # stick to <= 1000 for now?
            number_of_realisations=1,
            concatenate_output=False,
            equal_length_output=False,
            timestep_length=1,  # hrs
            start_year=2000,
            calendar='gregorian',  # gregorian or 365-day
    ):
        # Output folder/format details
        self.output_types = output_types  # ['point', 'catchment', 'grid']

        if isinstance(output_format, dict):
            self.output_format = output_format
        else:
            self.output_format = {output_type: output_format for output_type in self.output_types}
            if 'grid' in self.output_types:
                self.output_format['grid'] = 'nc'

        if isinstance(output_folder, dict):
            self.output_folder = output_folder
        else:
            self.output_folder = {output_type: output_folder for output_type in self.output_types}

        if output_prefix is None:
            self.output_prefix = {'point': 'point', 'catchment': 'catchment', 'grid': 'grid'}
        elif isinstance(output_prefix, dict):
            self.output_prefix = output_prefix
        else:
            self.output_prefix = {output_type: output_prefix for output_type in self.output_types}

        # Model details
        self.season_definitions = utils.parse_season_definitions(season_definitions)
        self.process_class = process_class
        if os.path.exists(parameters):
            self.parameters = utils.read_csv_(parameters)
        else:
            self.parameters = parameters
        if dem is not None:  # moved up here so change order of arguments - attribute?
            if os.path.exists(dem):
                dem = utils.read_ascii_raster(dem)
            dem_cell_size = dem.x.values[1] - dem.x.values[0]

        # Output location details
        if points is not None:
            if os.path.exists(points):
                self.points = utils.read_csv_(points)
            else:
                self.points = points
        else:
            self.points = points

        if catchments is not None:
            if os.path.exists(catchments):
                self.catchments = geopandas.read_file(catchments)
            else:
                self.catchments = catchments
        else:
            self.catchments = catchments
        self.catchment_id_field = catchment_id_field

        if 'grid' in self.output_types:
            self.grid = grid
            self.cell_size = grid['cellsize']
            # TODO: Make it so that grid does not need to be specified for gridded output (i.e. infer from shapefile)
        elif 'catchment' in self.output_types:
            if isinstance(grid, dict):
                self.grid = grid
                self.cell_size = grid['cellsize']
            else:
                xmin, ymin, xmax, ymax = utils.geodataframe_bounding_box(self.catchments, round_extent=False)
                if dem is not None:
                    new_xmin = dem.x.values[0] - dem_cell_size / 2.0
                    new_ymin = dem.y.values[-1] - dem_cell_size / 2.0
                    x_offset = new_xmin - utils.round_down(xmin, cell_size)
                    y_offset = new_ymin - utils.round_down(ymin, cell_size)
                    new_xmax = utils.round_up(xmax, cell_size) - (cell_size - x_offset)
                    new_ymax = utils.round_up(ymax, cell_size) - (cell_size - y_offset)
                    if new_xmax < xmax:
                        xmax = new_xmax + cell_size
                    else:
                        xmax = new_xmax
                    if new_ymax < ymax:
                        ymax = new_ymax + cell_size
                    else:
                        ymax = new_ymax
                    xmin = new_xmin
                    ymin = new_ymin
                self.grid = utils.ascii_grid_headers_from_extent(xmin, ymin, xmax, ymax, cell_size)
                self.cell_size = cell_size
        else:
            self.grid = grid
            self.cell_size = cell_size

        # Set (inner) simulation domain bounds
        if self.grid is not None:  # accounts for both catchment and grid outputs
            grid_xmin = self.grid['xllcorner']
            grid_ymin = self.grid['yllcorner']
            grid_xmax = grid_xmin + self.grid['ncols'] * self.cell_size
            grid_ymax = grid_ymin + self.grid['nrows'] * self.cell_size
        if self.points is not None:
            points_xmin = np.min(self.points['easting'])
            points_ymin = np.min(self.points['northing'])
            points_xmax = np.max(self.points['easting'])
            points_ymax = np.max(self.points['northing'])
            if self.grid is not None:
                xmin = np.minimum(points_xmin, grid_xmin)
                ymin = np.minimum(points_ymin, grid_ymin)
                xmax = np.maximum(points_xmax, grid_xmax)
                ymax = np.maximum(points_ymax, grid_ymax)
            else:
                xmin = points_xmin
                ymin = points_ymin
                xmax = points_xmax
                ymax = points_ymax
        if ('point' in self.output_types) and (self.points is None):  # point/nsrp simulation
            self.xmin = None
            self.ymin = None
            self.xmax = None
            self.ymax = None
            self.xrange = None
            self.yrange = None
            self.area = None
        else:
            self.xmin = utils.round_down(xmin, self.cell_size)
            self.ymin = utils.round_down(ymin, self.cell_size)
            self.xmax = utils.round_up(xmax, self.cell_size)
            self.ymax = utils.round_up(ymax, self.cell_size)
            self.xrange = xmax - xmin
            self.yrange = ymax - ymin
            self.area = self.xrange * self.yrange

        # Set up discretisation point location metadata arrays
        self.discretisation_metadata = {}
        if self.points is not None:
            self.discretisation_metadata[('point', 'x')] = self.points['easting'].values
            self.discretisation_metadata[('point', 'y')] = self.points['northing'].values
            if 'elevation' in self.points.columns:
                self.discretisation_metadata[('point', 'z')] = self.points['elevation'].values

        # For a grid these arrays are flattened 2D arrays so that every point has an associated x, y pair
        if self.grid is not None:
            x = np.arange(
                self.grid['xllcorner'] + self.cell_size / 2.0,
                self.grid['xllcorner'] + self.grid['ncols'] * self.cell_size,
                self.cell_size
            )
            y = np.arange(
                self.grid['yllcorner'] + self.cell_size / 2.0,
                self.grid['yllcorner'] + self.grid['nrows'] * self.cell_size,
                self.cell_size
            )
            y = y[::-1]

            xx, yy = np.meshgrid(x, y)
            xf = xx.flatten()
            yf = yy.flatten()
            self.discretisation_metadata[('grid', 'x')] = xf
            self.discretisation_metadata[('grid', 'y')] = yf

            if dem is not None:
                window = int(self.cell_size / dem_cell_size)

                # Restrict DEM to domain of output grid
                mask_x = (dem.x > grid_xmin) & (dem.x < grid_xmax)
                mask_y = (dem.y > grid_ymin) & (dem.y < grid_ymax)
                dem = dem.where(mask_x & mask_y, drop=True)

                # resampled_dem = dem.coarsen(x=window).mean().coarsen(y=window).mean()
                resampled_dem = dem.coarsen(x=window, boundary='pad').mean(skipna=True)\
                    .coarsen(y=window, boundary='pad').mean(skipna=True)
                flat_resampled_dem = resampled_dem.data.flatten()
                self.discretisation_metadata[('grid', 'z')] = flat_resampled_dem

        # Associate a phi value with each discretisation point

        # Dataframe of known phi values at given locations
        if phi is not None:
            if os.path.exists(phi):
                phi = utils.read_csv_(phi)
            else:
                phi = phi

        # Calculate phi for each discretisation point for all output types (unless a point is in the dataframe of
        # known phi, in which case use it directly)
        if phi is not None:
            for season in self.unique_seasons:

                if dem is not None:
                    interpolator, log_transformation = self.make_phi_interpolator(phi, season)
                else:
                    interpolator, log_transformation = self.make_phi_interpolator(phi, season, include_elevation=False)

                if self.points is not None:
                    if dem is not None:
                        interpolated_phi = interpolator(
                            (self.discretisation_metadata[('point', 'x')],
                             self.discretisation_metadata[('point', 'y')]),
                            mesh_type='unstructured',
                            ext_drift=self.discretisation_metadata[('point', 'z')],
                            return_var=False
                        )
                    else:
                        interpolated_phi = interpolator(
                            (self.discretisation_metadata[('point', 'x')],
                             self.discretisation_metadata[('point', 'y')]),
                            mesh_type='unstructured',
                            return_var=False
                        )
                    if log_transformation:
                        self.discretisation_metadata[('point', 'phi', season)] = np.exp(interpolated_phi)
                    else:
                        self.discretisation_metadata[('point', 'phi', season)] = interpolated_phi
                    self.discretisation_metadata[('point', 'phi', season)] = np.where(
                        self.discretisation_metadata[('point', 'phi', season)] < 0.0,
                        0.0,
                        self.discretisation_metadata[('point', 'phi', season)]
                    )

                if self.grid is not None:
                    if dem is not None:
                        interpolated_phi = interpolator(
                            (self.discretisation_metadata[('grid', 'x')], self.discretisation_metadata[('grid', 'y')]),
                            mesh_type='unstructured',
                            ext_drift=self.discretisation_metadata[('grid', 'z')],
                            return_var=False
                        )
                    else:
                        interpolated_phi = interpolator(
                            (self.discretisation_metadata[('grid', 'x')], self.discretisation_metadata[('grid', 'y')]),
                            mesh_type='unstructured',
                            return_var=False
                        )
                    if log_transformation:
                        self.discretisation_metadata[('grid', 'phi', season)] = np.exp(interpolated_phi)
                    else:
                        self.discretisation_metadata[('grid', 'phi', season)] = interpolated_phi
                    self.discretisation_metadata[('grid', 'phi', season)] = np.where(
                        self.discretisation_metadata[('grid', 'phi', season)] < 0.0,
                        0.0,
                        self.discretisation_metadata[('grid', 'phi', season)]
                    )

        # Catchment weights
        if 'catchment' in self.output_types:
            catchment_points = utils.catchment_weights(
                self.catchments, grid_xmin, grid_ymin, grid_xmax, grid_ymax, self.cell_size,
                id_field=self.catchment_id_field, epsg_code=32632
            )
            for catchment_id, point_arrays in catchment_points.items():
                # Check that points are ordered in the same way
                assert np.min(point_arrays['x'] == self.discretisation_metadata[('grid', 'x')]) == 1
                assert np.min(point_arrays['y'] == self.discretisation_metadata[('grid', 'y')]) == 1
                # TODO: Replace checks on array equivalence with dataframe merge operation
                self.discretisation_metadata[('catchment', 'weights', catchment_id)] = point_arrays['weight']

            # Rationalise grid discretisation points
            # - if a point is not used by any catchment and grid output is not required then no need to discretise it
            if ('catchment' in self.output_types) and ('grid' not in self.output_types):

                # Identify cells where any subcatchment is present (i.e. overall catchment mask)
                catchment_mask = np.zeros(self.discretisation_metadata[('grid', 'x')].shape[0], dtype=bool)
                for catchment_id in catchment_points.keys():
                    subcatchment_mask = self.discretisation_metadata[('catchment', 'weights', catchment_id)] > 0.0
                    catchment_mask[subcatchment_mask == 1] = 1

                # Subset static (non-seasonally varying) arrays - location, elevation and weights
                self.discretisation_metadata[('grid', 'x')] = (
                    self.discretisation_metadata[('grid', 'x')][catchment_mask]
                )
                self.discretisation_metadata[('grid', 'y')] = (
                    self.discretisation_metadata[('grid', 'y')][catchment_mask]
                )
                if dem is not None:
                    self.discretisation_metadata[('grid', 'z')] = (
                        self.discretisation_metadata[('grid', 'z')][catchment_mask]
                    )
                for catchment_id in catchment_points.keys():
                    self.discretisation_metadata[('catchment', 'weights', catchment_id)] = (
                        self.discretisation_metadata[('catchment', 'weights', catchment_id)][catchment_mask]
                    )

                # Subset seasonally varying arrays - phi
                for season in self.unique_seasons:
                    self.discretisation_metadata[('grid', 'phi', season)] = (
                        self.discretisation_metadata[('grid', 'phi', season)][catchment_mask]
                    )

        # Simulation configuration settings
        self.number_of_years = number_of_years
        self.number_of_realisations = number_of_realisations
        self.concatenate_output = concatenate_output
        if self.concatenate_output:
            self.equal_length_output = False
        else:
            self.equal_length_output = equal_length_output
        self.timestep_length = timestep_length
        self.start_year = start_year
        self.calendar = calendar

        # Derived attributes
        self.realisation_ids = range(1, self.number_of_realisations + 1)
        if not self.equal_length_output:
            self.start_years = [
                start_year + number_of_years * i for i in range(self.number_of_realisations)
            ]
        else:
            self.start_years = [start_year for _ in range(self.number_of_realisations)]
        self.output_paths = self.make_output_paths()

        # Other
        self.rng = np.random.default_rng()

    @staticmethod
    def make_phi_interpolator(df, season, include_elevation=True, distance_bins=7):
        df1 = df.loc[df['season'] == season]

        # Test for elevation-dependence of phi
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
        max_distance = np.max(scipy.spatial.distance.pdist(np.asarray(df[['easting', 'northing']])))
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

    def run(self):
        for realisation_id in self.realisation_ids:
            self.simulate_realisation(realisation_id, self.start_years[realisation_id - 1])
        # pass  # raise NotImplementedError

    def simulate_all(self):
        # initially serial but could be a multi-processing call to simulate_realisation
        pass

    def simulate_realisation(self, realisation_id, start_year):
        print('    - Realisation =', realisation_id)
        # ---
        # - consider splitting into chunks for simulation purposes, as chance that generating quite large arrays
        # -- could simulate say all storms but then raincells in batches
        # -- or say 100 years and then have a "join"
        # -- just requires splitting the realisation length and then a loop (possibly a time counter/index?)
        # - get all of the discretisation, output and concatenation working first though
        # ---

        # Array for month of discretised rainfall
        rainfall_arrays = self._initialise_discrete_rainfall_arrays(int((24 / self.timestep_length) * 31))

        # Get datetime series and end year
        end_year = start_year + self.number_of_years - 1
        datetimes = utils.datetime_series(
            start_year, end_year, self.timestep_length, self.season_definitions, self.calendar
        )

        # Helper dataframe with month end timesteps and times
        datetime_helper = datetimes.groupby(['year', 'month'])['hour'].agg('size')
        datetime_helper = datetime_helper.to_frame('n_timesteps')
        datetime_helper.reset_index(inplace=True)
        datetime_helper['end_timestep'] = datetime_helper['n_timesteps'].cumsum()  # beginning timestep of next month
        datetime_helper['start_timestep'] = datetime_helper['end_timestep'].shift()
        datetime_helper.iloc[0, datetime_helper.columns.get_loc('start_timestep')] = 0
        datetime_helper['start_time'] = datetime_helper['start_timestep'] * self.timestep_length
        datetime_helper['end_time'] = datetime_helper['end_timestep'] * self.timestep_length
        datetime_helper['n_hours'] = datetime_helper['end_time'] - datetime_helper['start_time']

        # Instantiate NSRP process and simulate storms/raincells
        nsrp_process = self._initialise_process(datetime_helper['n_hours'].values)
        nsrp_process.simulate()

        # Convert raincell coordinates and radii from km to m for discretisation
        if 'raincell_x' in nsrp_process.df.columns:
            nsrp_process.df['raincell_x'] *= 1000.0
            nsrp_process.df['raincell_y'] *= 1000.0
            nsrp_process.df['raincell_radii'] *= 1000.0

        # Prepare to store realisation output (point and catchment)
        output_lists = {}

        # Discretise simulation and write to output files
        number_of_months = datetime_helper.shape[0]

        t1 = datetime.datetime.now()

        # print('       - 0%')
        for month_idx in range(number_of_months):
            # progress = month_idx / number_of_months * 100.0
            # if (progress >= 10) and (progress % 10 == 0):
            #     print('       -', str(int(np.around(progress))) + '%')

            year = datetime_helper['year'].values[month_idx]
            month = datetime_helper['month'].values[month_idx]
            season = self.season_definitions[month]
            self.discretise(
                datetime_helper['start_time'][month_idx], datetime_helper['end_time'][month_idx], self.timestep_length,
                month, season,
                # datetime_helper['start_timestep'][month_idx], datetime_helper['end_timestep'][month_idx],
                nsrp_process.df, self.discretisation_metadata, rainfall_arrays
            )
            month_datetimes = datetimes.loc[(datetimes['year'] == year) & (datetimes['month'] == month)]
            timesteps_in_month = month_datetimes.shape[0]

            # Format discretised point/catchment series and store for later output (write grid output now)
            for output_type in self.output_types:
                if output_type == 'point':
                    if self.points is None:
                        location_ids = [1]
                    else:
                        location_ids = self.points['point_id'].values
                elif output_type == 'catchment':
                    location_ids = self.catchments[self.catchment_id_field].values
                elif output_type == 'grid':
                    location_ids = [1]

                idx = 0
                for location_id in location_ids:
                    output_key = (output_type, location_id, realisation_id)

                    # output_path = self.output_paths[(output_type, location_id, realisation_id)]
                    if output_type == 'point':
                        output_array = rainfall_arrays['point'][:timesteps_in_month, idx]
                    elif output_type == 'catchment':
                        catchment_discrete_rainfall = np.average(
                            rainfall_arrays['grid'], axis=1,
                            weights=self.discretisation_metadata[('catchment', 'weights', location_id)]
                        )
                        output_array = catchment_discrete_rainfall[:timesteps_in_month]

                    if output_type in ['point', 'catchment']:
                        formatted_output = []
                        for value in output_array:
                            formatted_output.append(('%.1f' % value).rstrip('0').rstrip('.'))  # + '\n'
                        if output_key not in output_lists.keys():
                            output_lists[output_key] = formatted_output.copy()
                        else:
                            output_lists[output_key].extend(formatted_output)

                    elif output_type == 'grid':
                        raise NotImplementedError

                    idx += 1

        self.write_output(realisation_id, output_lists)

        t2 = datetime.datetime.now()
        print(t2 - t1)

    def make_output_paths(self):
        output_paths = {}
        if 'point' in self.output_types:
            paths = self._output_paths_helper('point', self.output_format['point'])
            for key, value in paths.items():
                output_paths[key] = value
        if 'catchment' in self.output_types:
            paths = self._output_paths_helper('catchment', self.output_format['catchment'])
            for key, value in paths.items():
                output_paths[key] = value
        if 'grid' in self.output_types:
            paths = self._output_paths_helper('grid', 'nc')
            for key, value in paths.items():
                output_paths[key] = value
        return output_paths

    def _output_paths_helper(self, output_type, output_format):
        output_format_extensions = {'csv': '.csv', 'csvy': '.csvy', 'txt': '.txt', 'netcdf': '.nc'}

        if output_type == 'point':
            if self.points is not None:
                location_ids = self.points['point_id'].values
            else:
                location_ids = [1]
            output_folder = self.output_folder['point']
        elif output_type == 'catchment':
            location_ids = self.catchments[self.catchment_id_field].values
            output_folder = self.output_folder['catchment']
        elif output_type == 'grid':
            location_ids = [1]
            output_folder = self.output_folder['grid']

        output_paths = {}
        output_cases = itertools.product(self.realisation_ids, location_ids)
        for realisation_id, location_id in output_cases:
            _realisation_id = utils.format_with_leading_zeros(realisation_id)
            if self.points is not None:
                _location_id = utils.format_with_leading_zeros(location_id)
            else:
                _location_id = ''
            output_file_name = self.output_prefix[output_type] + _location_id + '_' + _realisation_id + \
                output_format_extensions[output_format]
            output_path_key = (output_type, location_id, realisation_id)
            output_paths[output_path_key] = os.path.join(output_folder, output_file_name)

        return output_paths

    def _initialise_discrete_rainfall_arrays(self, nt):
        dc = {}
        if 'point' in self.output_types:
            if self.points is not None:
                dc['point'] = np.zeros((nt, self.points.shape[0]))
            else:
                dc['point'] = np.zeros((nt, 1))
        if ('catchment' in self.output_types) or ('grid' in self.output_types):
            dc['grid'] = np.zeros((nt, self.discretisation_metadata[('grid', 'x')].shape[0]))
        return dc

    def _initialise_process(self, month_lengths):
        raise NotImplementedError

    def discretise(
            self, start_time, end_time, timestep_length, month, season, nsrp_df, discretisation_metadata,
            discrete_rainfall
    ):
        raise NotImplementedError

    def write_output(self, realisation_id, output_lists):
        for output_type in self.output_types:
            if output_type == 'point':
                if self.points is None:
                    location_ids = [1]
                else:
                    location_ids = self.points['point_id'].values
            elif output_type == 'catchment':
                location_ids = self.catchments[self.catchment_id_field].values
            elif output_type == 'grid':
                location_ids = [1]

            for location_id in location_ids:
                output_path = self.output_paths[(output_type, location_id, realisation_id)]
                output_lines = '\n'.join(output_lists[(output_type, location_id, realisation_id)])
                with open(output_path, 'w') as fh:
                    fh.writelines(output_lines)

    @staticmethod
    def spatial_mean():
        pass  # this might be handled by Simulator (just STNSRP) - i.e. not here in base

    def concatenate_output_files(self):  # utils function?
        pass

    @property
    def unique_seasons(self):
        if self.season_definitions is not None:
            return list(set(self.season_definitions.values()))
        else:
            return None


class Process(object):

    def __init__(
            self,
            parameters,
            simulation_length,  # years initially
            month_lengths,  # number of hours in each month
            season_definitions,
            xmin, xmax, ymin, ymax, xrange, yrange, area
    ):
        # - could either use parameters in dataframe (maybe do for now as quicker and easier) or set parameters as
        # attributes in another way (but e.g. dictionary keys now becoming (parameter, season) so may as well just
        # have a dataframe at this point in some ways!)

        # - what if seasonal simulation? repeat parameters for months in season... do this in calling method though

        # - !! need to ensure no long dry periods at the end again though... apply buffer or iterate (keep appending
        # after final storm origin until get a storm origin after the end of the required period)? !!

        # - currently assuming parameters indicate the months (i.e. parameters df in wide format)

        # Ensure parameters are available monthly (i.e. repeated for each month in season)
        if len(season_definitions.keys()) == 12:
            self.parameters = parameters.copy()
            self.parameters['month'] = self.parameters['season']
        else:
            months = []
            seasons = []
            for month, season in season_definitions.items():
                months.append(month)
                seasons.append(season)
            df_seasons = pd.DataFrame({'month': months, 'season': seasons})
            self.parameters = pd.merge(df_seasons, parameters, how='left', on='season')
        self.parameters.sort_values(by='month', inplace=True)

        self.simulation_length = simulation_length  # years
        self.month_lengths = month_lengths

        # season_definitions ?

        # !! CONVERT UNITS M TO KM !!
        if xmin is not None:
            self.xmin = xmin / 1000.0
            self.xmax = xmax / 1000.0
            self.ymin = ymin / 1000.0
            self.ymax = ymax / 1000.0
            self.xrange = xrange / 1000.0
            self.yrange = yrange / 1000.0
            self.area = area / (1000.0 * 1000.0)
        else:
            self.xmin = None
            self.xmax = None
            self.ymin = None
            self.ymax = None
            self.xrange = None
            self.yrange = None
            self.area = None

        self.number_of_months = self.parameters['month'].shape[0]  # useful helper? - no, always 12 at the moment

        self.number_of_storms = None
        # self.storm_arrival_times = None
        self.number_of_raincells = None
        self.number_of_raincells_by_storm = None  # still needed if using merge rather than repeat?

        self.df = None  # master dataframe by raincell  # to be replaced by self.raincells?

        self.storms = None  # replacing self.storm_arrival_times and storm_ids
        # self.raincells = None  # replacing self.df ultimately?

        self.rng = np.random.default_rng()

    def simulate(self):
        self.simulate_storms()
        self.simulate_raincells()
        self._merge_parameters()
        self.simulate_raincell_arrivals()
        self.simulate_raincell_intensity()
        self.simulate_raincell_duration()

    def simulate_storms(self):
        # Ensure that Poisson process is sampled beyond end of simulation to avoid any truncation errors
        simulation_end_time = np.cumsum(self.month_lengths)[-1]
        simulation_length = self.simulation_length
        month_lengths = self.month_lengths
        while True:

            # Set up simulation_length and month_lengths with buffer applied
            simulation_length += 4
            idx = 4 * self.number_of_months
            month_lengths = np.concatenate([month_lengths, month_lengths[-idx:]])

            # Repeat each set of monthly lamda values for each year in simulation
            lamda = np.tile(self.parameters['lamda'].values, simulation_length)

            # Get a sample value for number of storms given simulation length
            cumulative_expected_storms = np.cumsum(lamda * month_lengths)
            cumulative_month_endtimes = np.cumsum(month_lengths)
            expected_number_of_storms = cumulative_expected_storms[-1]
            number_of_storms = self.rng.poisson(expected_number_of_storms)  # sampled

            # Sample storm arrival times on deformed timeline
            deformed_storm_arrival_times = (
                    expected_number_of_storms * np.sort(self.rng.uniform(size=number_of_storms))
            )

            # Transform storm origin times from deformed to linear timeline
            cumulative_expected_storms = np.insert(cumulative_expected_storms, 0, 0.0)
            cumulative_month_endtimes = np.insert(cumulative_month_endtimes, 0, 0.0)
            interpolator = scipy.interpolate.interp1d(
                cumulative_expected_storms, cumulative_month_endtimes
            )
            storm_arrival_times = interpolator(deformed_storm_arrival_times)

            if storm_arrival_times[-1] > simulation_end_time:
                storm_arrival_times = storm_arrival_times[storm_arrival_times < simulation_end_time]
                self.number_of_storms = storm_arrival_times.shape[0]
                self.storms = pd.DataFrame({
                    'storm_id': np.arange(self.number_of_storms),
                    'storm_arrival': storm_arrival_times
                })
                self.storms['month'] = self._lookup_months(
                    self.parameters['month'].values, self.month_lengths, self.simulation_length,
                    self.storms['storm_arrival'].values
                )
                break

    def simulate_raincells(self):
        # - assumption that this method generates the master df (i.e. per raincell attributes)
        # - for nsrp this can just be a dummy dataframe of correct length
        # - for stnsrp this dataframe contains location and radius details AND it needs to contain storm arrival times
        # as raincells are simulated month-wise - so add storm arrival times to df in nsrp too
        raise NotImplementedError

    def simulate_raincell_arrivals(self):
        # Raincell arrival times relative to storm origins
        raincell_arrival_times = self.rng.exponential(1.0 / self.df['beta'])

        # Raincell arrival times relative to simulation period origin
        self.df['raincell_arrival'] = self.df['storm_arrival'] + raincell_arrival_times

    def simulate_raincell_intensity(self):
        self.df['raincell_intensity'] = self.rng.exponential(1.0 / self.df['xi'])

    def simulate_raincell_duration(self):
        self.df['raincell_duration'] = self.rng.exponential(1.0 / self.df['eta'])
        self.df['raincell_end'] = self.df['raincell_arrival'] + self.df['raincell_duration']

    def _merge_parameters(self):
        # - does parameters df need to be subset on columns?
        # - also parameters df needs to be monthly, even if simulation is seasonal - input thing

        # self._lookup_months(
        #     self.parameters['month'].values, self.month_lengths, self.simulation_length, self.df, 'storm_arrival'
        # )
        self.df['month'] = self._lookup_months(
            self.parameters['month'].values, self.month_lengths, self.simulation_length,
            self.df['storm_arrival'].values
        )
        parameters = self.parameters.drop(
            ['converged', 'objective_function', 'iterations', 'function_evaluations'], axis=1
        )
        self.df = pd.merge(self.df, parameters, how='left', on='month')

    @staticmethod
    def _lookup_months(unique_months, month_lengths, period_length, times):
        end_times = np.cumsum(month_lengths)
        # start_times = np.insert(np.cumsum(month_lengths)[:-1], 0, 0.0)
        repeated_months = np.tile(unique_months, period_length)
        idx = np.digitize(times, end_times)  # -1 not required... check
        months = repeated_months[idx]
        return months

    @staticmethod
    def _storm_arrays_by_raincell(number_of_raincells_by_storm, storm_ids, storm_arrival_times, storm_months):
        # Repeat storm arrival times for each member raincell to help get an array/dataframe per raincell
        # - could be made more generic by taking df as argument and looping through columns...
        storm_ids_by_raincell = np.repeat(storm_ids, number_of_raincells_by_storm)
        storm_arrival_times_by_raincell = np.repeat(storm_arrival_times, number_of_raincells_by_storm)
        storm_months_by_raincell = np.repeat(storm_months, number_of_raincells_by_storm)
        return storm_ids_by_raincell, storm_arrival_times_by_raincell, storm_months_by_raincell


class Model(object):

    # - should probably keep track of seasons at least and flag if some error on user input / inconsistency?
    # - also the place to specify intensity distribution

    def __init__(
            self,
            season_definitions=None,
            preprocessor_class=None,
            fitter_class=None,
            process_class=None,
            simulator_class=None
    ):
        """
        Basic NSRP model class for pre-processing, fitting and simulation.

        Override this class with a specific model subclass (e.g. for a NSRP or STNSRP model).

        Args:
            season_definitions (dict): Month identifier (as integer 1-12) as key and season identifier (integer) as
                value
            preprocessor_class (class): Preprocessor class for model
            fitter_class (class): Fitter class for model
            process_class (class): NSRP Process class for model
            simulator_class (class): Simulator class for model

        """
        if season_definitions is not None:
            self.season_definitions = utils.parse_season_definitions(season_definitions)
        else:
            self.season_definitions = {}
            for month in range(1, 12 + 1):
                self.season_definitions[month] = month

        self.preprocessor_class = preprocessor_class
        self.fitter_class = fitter_class
        self.process_class = process_class
        self.simulator_class = simulator_class
        self.preprocessor = None
        self.fitter = None
        self.simulator = None

    def preprocess(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def simulate(self):
        raise NotImplementedError