import os
import sys
import itertools

import numpy as np
import scipy.stats
import scipy.optimize
import pandas as pd

from . import base
from . import utils
from . import nsrp
from . import properties


class Preprocessor(base.Preprocessor):

    def __init__(  # !! Order here can be whatever best (e.g. putting metadata_path higher up) !!
            self,
            season_definitions=None,
            statistic_definitions=None,
            statistic_definitions_path=None,
            timeseries_format='csv',
            calculation_period=None,
            completeness_threshold=80.0,
            metadata_path=None,
            timeseries_folder=None,
            output_folder=None,
            point_statistics_filename=None,
            cross_correlation_filename=None,
            phi_filename=None,
            outlier_method=None,
            maximum_relative_difference=2.0,
            maximum_alterations=5
    ):
        super().__init__(
            season_definitions,
            statistic_definitions,
            statistic_definitions_path,
            timeseries_format,
            calculation_period,
            completeness_threshold,
            output_folder,
            point_statistics_filename,
            cross_correlation_filename,
            phi_filename,
            False,  # override_phi
            outlier_method,
            maximum_relative_difference,
            maximum_alterations
        )
        
        if self.statistic_definitions is None:
            dc = {
                1: {'weight': 3.0, 'duration': 1,  'name': 'variance'},
                2: {'weight': 3.0, 'duration': 1,  'name': 'skewness'},
                3: {'weight': 5.0, 'duration': 1,  'name': 'probability_dry', 'threshold': 0.2},
                4: {'weight': 5.0, 'duration': 24, 'name': 'mean'},
                5: {'weight': 2.0, 'duration': 24, 'name': 'variance'},
                6: {'weight': 2.0, 'duration': 24, 'name': 'skewness'},
                7: {'weight': 6.0, 'duration': 24, 'name': 'probability_dry', 'threshold': 0.2},
                8: {'weight': 3.0, 'duration': 24, 'name': 'autocorrelation', 'lag': 1},
                9: {'weight': 2.0, 'duration': 24, 'name': 'cross-correlation', 'lag': 0}
            }
            id_name = 'statistic_id'
            non_id_columns = ['name', 'duration', 'lag', 'threshold', 'weight']
            self.statistic_definitions = utils.nested_dictionary_to_dataframe(
                dc, id_name, non_id_columns
            )

        self.check_statistics_include_24hr_mean()
        
        self.metadata_path = metadata_path
        self.timeseries_folder = timeseries_folder
        
        self.metadata = pd.read_csv(metadata_path)  # df could be passed directly
        self.metadata.columns = [cn.lower() for cn in self.metadata.columns]
        
        self.point_preprocessors = None
        self.pair_metadata = None
        self.cross_correlations = None

    def run(self):
        self.preprocess_points()
        self.merge_phi()
        self.create_pair_metadata()
        self.calculate_cross_correlation()
        self.merge_statistics()
        if self.output_folder is not None:
            self.write_statistics()

    def preprocess_points(self):
        self.point_preprocessors = {}
        for index, row in self.metadata.iterrows():
            if os.path.exists(row['file_path']):
                timeseries_path = row['file_path']
            else:
                timeseries_path = os.path.join(self.timeseries_folder, row['file_path'])

            point_statistic_definitions = self.statistic_definitions.loc[
                self.statistic_definitions['name'] != 'cross-correlation'
            ]
            
            point_preprocessor = nsrp.Preprocessor(
                season_definitions=self.season_definitions,
                statistic_definitions=point_statistic_definitions,
                statistic_definitions_path=None,
                timeseries_format=self.timeseries_format,
                calculation_period=self.calculation_period,
                completeness_threshold=self.completeness_threshold,
                timeseries_path=timeseries_path,
                outlier_method=self.outlier_method,
                maximum_relative_difference=self.maximum_relative_difference,
                maximum_alterations=self.maximum_alterations
            )
            point_preprocessor.run()
            
            self.point_preprocessors[row['point_id']] = point_preprocessor
    
    # def merge_point_statistics(self):
    #     dfs = []
    #     for point_id, point_preprocessor in self.point_preprocessors.items():
    #         df = point_preprocessor.statistics.copy()
    #         df['point_id'] = point_id
    #         dfs.append(df)
    #     self.statistics = pd.concat(dfs)
    
    def merge_phi(self):
        dfs = []
        for point_id, point_preprocessor in self.point_preprocessors.items():
            df = point_preprocessor.phi.copy()
            df['point_id'] = point_id
            dfs.append(df)
        self.phi = pd.concat(dfs)
    
    # static method / function?
    def create_pair_metadata(self):
        pairs = list(itertools.combinations(list(self.metadata['point_id']), 2))
        id1s = []
        id2s = []
        distances = []
        for id1, id2 in pairs:
            id1_x = self.metadata.loc[self.metadata['point_id'] == id1, 'easting'].values[0]
            id1_y = self.metadata.loc[self.metadata['point_id'] == id1, 'northing'].values[0]
            id2_x = self.metadata.loc[self.metadata['point_id'] == id2, 'easting'].values[0]
            id2_y = self.metadata.loc[self.metadata['point_id'] == id2, 'northing'].values[0]
            distance = ((id1_x - id2_x) ** 2 + (id1_y - id2_y) ** 2) ** 0.5
            id1s.append(id1)
            id2s.append(id2)
            distances.append(distance / 1000.0)  # m to km
        self.pair_metadata = pd.DataFrame({
            'point_id': id1s, 'point_id2': id2s, 'distance': distances
        })
    
    # static method / function?
    def calculate_cross_correlation(self):
        # - another method could implement the "reduced pairs" approach say
        # -- another column (factor level) could act as a flag etc
        # -- or separate dataframe...
        # - or a function that operates on the dataframe...
        
        cross_correlation_definitions = self.statistic_definitions.loc[
            self.statistic_definitions['name'] == 'cross-correlation'
        ]
        
        dc = {
            'statistic_id': [], 'lag': [], 'point_id': [], 'point_id2': [], 'distance': [],
            'duration': [], 'season': [], 'value': [], 'weight': []
        }

        number_of_points = np.unique(self.pair_metadata['point_id']).shape[0]
        number_of_pairs = self.pair_metadata.shape[0]
        weight_factor = 1.0 #  (number_of_points + 1) / number_of_pairs
        
        for _, statistic_details in cross_correlation_definitions.iterrows():
            statistic_id = statistic_details['statistic_id']
            duration = statistic_details['duration']
            lag = statistic_details['lag']
            
            for season in self.unique_seasons:
                
                for _, pair_details in self.pair_metadata.iterrows():
                    id1 = pair_details['point_id']
                    id2 = pair_details['point_id2']
                    
                    df1 = self.point_preprocessors[id1].dfs[duration]
                    df2 = self.point_preprocessors[id2].dfs[duration]
                    x = df1.loc[df1['season'] == season]
                    y = df2.loc[df2['season'] == season]
                    
                    df3 = pd.merge(x, y, left_index=True, right_index=True)
                    r, p = scipy.stats.pearsonr(df3['value_x'][lag:],
                                                df3['value_y'].shift(lag)[lag:])
                    
                    dc['statistic_id'].append(int(statistic_id))
                    dc['lag'].append(int(lag))
                    dc['point_id'].append(int(id1))
                    dc['point_id2'].append(int(id2))
                    dc['distance'].append(pair_details['distance'])
                    dc['duration'].append(duration)
                    dc['season'].append(int(season))
                    dc['value'].append(r)
                    dc['weight'].append(statistic_details['weight'] * weight_factor)
                    
        self.cross_correlations = pd.DataFrame(dc)
        self.cross_correlations['name'] = 'cross-correlation'
        self.cross_correlations['gs'] = 1.0

        # Merge phi
        self.cross_correlations = pd.merge(
            self.cross_correlations, self.phi, how='left', on=['season', 'point_id']
        )
        phi2 = self.phi.copy()
        phi2.rename({'phi': 'phi2', 'point_id': 'point_id2'}, axis=1, inplace=True)
        self.cross_correlations = pd.merge(
            self.cross_correlations, phi2, how='left', on=['season', 'point_id2']
        )
    
    def merge_statistics(self):
        dfs = []
        for point_id, point_preprocessor in self.point_preprocessors.items():
            df = point_preprocessor.statistics.copy()
            df['point_id'] = point_id
            dfs.append(df)
        self.statistics = pd.concat(dfs)

        self.statistics['point_id2'] = pd.NA
        self.statistics['distance'] = np.nan
        self.statistics = pd.concat([self.statistics, self.cross_correlations])

        column_order = [
            'point_id', 'point_id2', 'distance', 'statistic_id', 'name', 'duration', 'lag',
            'threshold', 'weight', 'season', 'value', 'gs', 'phi', 'phi2'
        ]
        self.statistics = self.statistics[column_order]


class Fitter(base.Fitter):

    def __init__(
            self,
            season_definitions,
            reference_statistics,
            reference_statistics_path=None,  # ?
            output_folder=None,
            parameters_filename=None,
            statistics_filename=None,
            cross_correlation_filename=None,
            parameter_bounds=None
    ):
        super().__init__(
            season_definitions, reference_statistics, reference_statistics_path,
            output_folder, parameters_filename, statistics_filename, cross_correlation_filename
        )

        self.parameter_names = ['lamda', 'beta', 'rho', 'eta', 'xi', 'gamma']
        if parameter_bounds is None:
            self.parameter_bounds = [
                (0.001, 0.05),      # lamda
                (0.02, 0.5),        # beta
                (0.0001, 2.0),      # rho
                (0.1, 12.0),        # eta
                (0.01, 4.0),        # xi
                (0.01, 500.0)       # gamma
            ]
        else:
            self.parameter_bounds = parameter_bounds
        self.parameter_output_columns = [
            'season', 'lamda', 'beta', 'rho', 'eta', 'xi', 'gamma', 'fit_success', 'objective_function',
            'number_of_iterations', 'number_of_evaluations'
        ]

    @classmethod
    def analytical_properties_wrapper(cls, parameters, statistic_ids, fitting_data):
        lamda, beta, rho, eta, xi, gamma = parameters
        nu = 2.0 * np.pi * rho / gamma ** 2.0
        mod_stats = cls.calculate_analytical_properties(
            statistic_ids, fitting_data, lamda, beta, eta, xi, nu, gamma
        )
        return mod_stats


class Process(base.Process):

    def simulate_raincells(self):
        i = 0
        for _, row in self.parameters.iterrows():
            storms_in_month = self.storms.loc[self.storms['month'] == row['month']]
            month_number_of_storms = storms_in_month.shape[0]
            month_number_of_raincells_by_storm, \
                month_raincell_x_coords, \
                month_raincell_y_coords, \
                month_raincell_radii = (
                    self.simulate_raincells_for_month(
                        row['month'], row['rho'], row['gamma'], month_number_of_storms, self.xmin, self.xmax, self.ymin,
                        self.ymax, self.xrange, self.yrange, self.area, self.rng
                    )
                )
            month_storm_ids_by_raincell, month_storm_arrivals_by_raincell, _ = self._storm_arrays_by_raincell(
                month_number_of_raincells_by_storm, storms_in_month['storm_id'].values,
                storms_in_month['storm_arrival'].values, storms_in_month['month'].values
            )
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
        # - inner and outer raincells already concatenated in month-wise call... check all!
        self.number_of_raincells_by_storm = number_of_raincells_by_storm
        self.df = pd.DataFrame({
            'storm_id': storm_ids_by_raincell,
            'storm_arrival': storm_arrivals_by_raincell,
            'month': months_by_raincell,
            'raincell_x': raincell_x_coords,
            'raincell_y': raincell_y_coords,
            'raincell_radii': raincell_radii,
        })
        self.df.sort_values('storm_arrival', inplace=True)

    @staticmethod
    def simulate_raincells_for_month(
            month, rho, gamma, number_of_storms, xmin, xmax, ymin, ymax, xrange, yrange, area, rng
    ):
        # Inner region - "standard" spatial Poisson process
        inner_number_of_raincells_by_storm = rng.poisson(rho * area, number_of_storms)
        inner_number_of_raincells = np.sum(inner_number_of_raincells_by_storm)
        inner_x_coords = rng.uniform(xmin, xmax, inner_number_of_raincells)
        inner_y_coords = rng.uniform(ymin, ymax, inner_number_of_raincells)
        inner_radii = rng.exponential((1.0 / gamma), inner_number_of_raincells)

        # Outer region

        # Construct CDF lookup function for distances of relevant raincells occurring in outer
        # region - Burton et al. (2010) equation A8
        distance_from_quantile_func = Process.construct_outer_raincells_inverse_cdf(gamma, xrange, yrange)

        # Density of relevant raincells in outer region - Burton et al. (2010) equation A9
        rho_y = 2 * rho / gamma ** 2 * (gamma * (xrange + yrange) + 4)

        # Number of relevant raincells in outer region
        outer_number_of_raincells_by_storm = rng.poisson(rho_y, number_of_storms)  # check rho=mean
        outer_number_of_raincells = np.sum(outer_number_of_raincells_by_storm)

        # Sample from CDF of distances of relevant raincells occurring in outer region
        outer_raincell_distance_quantiles = rng.uniform(0.0, 1.0, outer_number_of_raincells)
        outer_raincell_distances = distance_from_quantile_func(outer_raincell_distance_quantiles)

        # Sample eastings and northings from uniform distribution given distance from domain
        # boundaries
        outer_x_coords, outer_y_coords = Process.sample_outer_locations(
            outer_raincell_distances, xrange, yrange, xmin, xmax, ymin, ymax, rng
        )

        # Sample raincell radii - for outer region raincells the radii need to exceed the distance
        # of the cell centre from the domain boundary (i.e. conditional)
        min_quantiles = scipy.stats.expon.cdf(outer_raincell_distances, scale=(1.0 / gamma))
        quantiles = rng.uniform(min_quantiles, np.ones(min_quantiles.shape[0]))
        outer_radii = scipy.stats.expon.ppf(quantiles, scale=(1.0 / gamma))

        # Combiner inner and outer region raincells (concatenate arrays)
        # - what about ordering? -- no need to worry because all independent sampling? -- check this
        number_of_raincells_by_storm = inner_number_of_raincells_by_storm + outer_number_of_raincells_by_storm
        raincell_x_coords = np.concatenate([inner_x_coords, outer_x_coords])
        raincell_y_coords = np.concatenate([inner_y_coords, outer_y_coords])
        raincell_radii = np.concatenate([inner_radii, outer_radii])

        return number_of_raincells_by_storm, raincell_x_coords, raincell_y_coords, raincell_radii

    @staticmethod
    def outer_raincells_cdf(x, gamma, xrange, yrange, q=0):
        # Burton et al. (2010) - equation A8
        # x = distance from domain boundaries, xrange is w and yrange is z in Burton et al.
        # returns y = cdf of distance of relevant raincells occurring in the outer region
        # additionally subtracting q (in range 0-1) to enable solving for x given a desired y
        return 1 - (1 + (4 * x * gamma) / (gamma * (xrange + yrange) + 4)) * np.exp(-gamma * x) - q

    @staticmethod
    def construct_outer_raincells_inverse_cdf(gamma, xrange, yrange):
        # So that x (distance) can be looked up from (sampled) y (cdf quantile)
        y1 = np.arange(0.0, 0.01, 0.0001)
        y2 = np.arange(0.01, 0.99, 0.001)
        y3 = np.arange(0.99, 1.0+0.00001, 0.0001)
        y = np.concatenate([y1, y2, y3])
        x = []
        i = 1
        for q in y:
            # r = scipy.optimize.fsolve(Process.outer_raincells_cdf, 0, args=(gamma, xrange, yrange, q))
            r, info, ier, msg = scipy.optimize.fsolve(
                Process.outer_raincells_cdf, 0, args=(gamma, xrange, yrange, q), full_output=True
            )
            x.append(r[0])

            # Final quantile at ~1 may be subject to convergence issues, so use previous value of x
            if ier != 1:
                if i == y.shape[0] and ier == 5:
                    pass
                else:
                    raise RuntimeError('Convergence error in construction of inverse CDF for outer raincells')

            i += 1

        x = np.asarray(x)
        y[-1] = 1.0
        # cdf = scipy.interpolate.interp1d(x, y)
        inverse_cdf = scipy.interpolate.interp1d(y, x)
        return inverse_cdf

    @staticmethod
    def sample_outer_locations(d, xrange, yrange, xmin, xmax, ymin, ymax, rng):
        # d = distance to raincell centre = x in Burton et al. (2010)
        # vectorised so perimeter array contains a perimeter for each raincell's distance d

        # Perimeter is the sum of the domain perimeter and four quarter-circle arc lengths
        perimeter = 2 * xrange + 2 * yrange
        perimeter += 2 * np.pi * d

        # Sample along each perimeter
        uniform_sample = rng.uniform(0.0, 1.0, perimeter.shape[0])
        position_1d = uniform_sample * perimeter

        # Identify which of the eight line segments that the sampled lengths correspond to using
        # the lower left as a reference point (xmin-d, ymin). Also identify the length relative to
        # the segment origin (first point reached moving clockwise from lower left)
        corner_length = (2.0 * np.pi * d) / 4.0  # quarter-circle arc length
        segment_id = np.zeros(perimeter.shape[0], dtype=int)
        segment_position = np.zeros(perimeter.shape[0])  # i.e. length relative to segment origin
        for i in range(1, 8+1):
            if i == 1:
                min_length = np.zeros(perimeter.shape[0])
                max_length = xrange
            elif i == 2:
                min_length = np.zeros(perimeter.shape[0]) + xrange
                max_length = xrange + corner_length
            elif i == 3:
                min_length = xrange + corner_length
                max_length = xrange + corner_length + yrange
            elif i == 4:
                min_length = xrange + corner_length + yrange
                max_length = xrange + 2 * corner_length + yrange
            elif i == 5:
                min_length = xrange + 2 * corner_length + yrange
                max_length = 2 * xrange + 2 * corner_length + yrange
            elif i == 6:
                min_length = 2 * xrange + 2 * corner_length + yrange
                max_length = 2 * xrange + 3 * corner_length + yrange
            elif i == 7:
                min_length = 2 * xrange + 3 * corner_length + yrange
                max_length = 2 * xrange + 3 * corner_length + 2 * yrange
            elif i == 8:
                min_length = 2 * xrange + 3 * corner_length + 2 * yrange
                max_length = perimeter  # = 2 * xrange + 4 * corner_length + 2 * yrange

            segment_id[(position_1d >= min_length) & (position_1d < max_length)] = i

            segment_position[segment_id == i] = (
                    position_1d[segment_id == i] - min_length[segment_id == i]
            )

        # Identify eastings and northings for straight-line segments first (1, 3, 5, 7)
        x = np.zeros(perimeter.shape[0])
        y = np.zeros(perimeter.shape[0])
        x[segment_id == 1] = xmin - d[segment_id == 1]
        y[segment_id == 1] = ymin + segment_position[segment_id == 1]
        x[segment_id == 3] = xmin + segment_position[segment_id == 3]
        y[segment_id == 3] = ymax + d[segment_id == 3]
        x[segment_id == 5] = xmax + d[segment_id == 5]
        y[segment_id == 5] = ymax - segment_position[segment_id == 5]
        x[segment_id == 7] = xmax - segment_position[segment_id == 7]
        y[segment_id == 7] = ymin - d[segment_id == 7]

        # Identify eastings and northings for corner segments (2, 4, 6, 8)
        theta = np.zeros(perimeter.shape[0])  # angle of sector corresponding with arc length

        theta[segment_id == 2] = segment_position[segment_id == 2] / d[segment_id == 2]
        x[segment_id == 2] = xmin + d[segment_id == 2] * np.cos(np.pi - theta[segment_id == 2])
        y[segment_id == 2] = ymax + d[segment_id == 2] * np.sin(np.pi - theta[segment_id == 2])

        theta[segment_id == 4] = segment_position[segment_id == 4] / d[segment_id == 4]
        x[segment_id == 4] = xmax + d[segment_id == 4] * np.cos(np.pi / 2.0 - theta[segment_id == 4])
        y[segment_id == 4] = ymax + d[segment_id == 4] * np.sin(np.pi / 2.0 - theta[segment_id == 4])

        theta[segment_id == 6] = segment_position[segment_id == 6] / d[segment_id == 6]
        x[segment_id == 6] = xmax + d[segment_id == 6] * np.cos(2.0 * np.pi - theta[segment_id == 6])
        y[segment_id == 6] = ymin + d[segment_id == 6] * np.sin(2.0 * np.pi - theta[segment_id == 6])

        theta[segment_id == 8] = segment_position[segment_id == 8] / d[segment_id == 8]
        x[segment_id == 8] = xmin + d[segment_id == 8] * np.cos(3.0 / 2.0 * np.pi - theta[segment_id == 8])
        y[segment_id == 8] = ymin + d[segment_id == 8] * np.sin(3.0 / 2.0 * np.pi - theta[segment_id == 8])

        return x, y


class Simulator(base.Simulator):

    def _initialise_process(self, month_lengths):
        process = self.process_class(
            self.parameters, self.number_of_years, month_lengths, self.season_definitions,
            self.xmin, self.xmax, self.ymin, self.ymax, self.xrange, self.yrange, self.area
        )
        return process

    def discretise(
            self, start_time, end_time, timestep_length, month, season, nsrp_df, discretisation_metadata,
            discrete_rainfall
    ):
        temporal_mask = (nsrp_df['raincell_arrival'].values < end_time) & (nsrp_df['raincell_end'].values > start_time)
        raincell_arrival_times = nsrp_df['raincell_arrival'].values[temporal_mask]
        raincell_end_times = nsrp_df['raincell_end'].values[temporal_mask]
        raincell_intensities = nsrp_df['raincell_intensity'].values[temporal_mask]
        raincell_x = nsrp_df['raincell_x'].values[temporal_mask]
        raincell_y = nsrp_df['raincell_y'].values[temporal_mask]
        raincell_radii = nsrp_df['raincell_radii'].values[temporal_mask]
        
        for output_type in self.output_types:
            if output_type == 'catchment':
                discretisation_case = 'grid'
            else:
                discretisation_case = output_type

            # self.discretise_multiple_points(
            utils.discretise_multiple_points(
                start_time,
                end_time,
                timestep_length,
                raincell_arrival_times,
                raincell_end_times,
                raincell_intensities,
                discrete_rainfall[discretisation_case],
                raincell_x,
                raincell_y,
                raincell_radii,
                # discretisation_metadata[('point', 'id')],  # ! does not exist currently - needs to be added or removed !
                discretisation_metadata[(discretisation_case, 'x')],
                discretisation_metadata[(discretisation_case, 'y')],
                discretisation_metadata[(discretisation_case, 'phi', season)],  # ! how to handle season... take as arg for now !
                # point_indexes_axis_1, point_indexes_axis_2=None
            )


class Model(base.Model):

    def __init__(
            self,
            season_definitions=None,
            preprocessor_class=Preprocessor,
            fitter_class=Fitter,
            process_class=Process,
            simulator_class=Simulator
    ):
        super().__init__(season_definitions, preprocessor_class, fitter_class, process_class, simulator_class)

    def preprocess(
            self,
            statistic_definitions=None,
            statistic_definitions_path=None,
            timeseries_format='csv',
            calculation_period=None,
            completeness_threshold=80.0,
            metadata_path=None,
            timeseries_folder=None,
            output_folder=None,
            point_statistics_filename=None,
            cross_correlation_filename=None,
            phi_filename=None,
            outlier_method=None,
            maximum_relative_difference=2.0,
            maximum_alterations=5
    ):
        print('  Preprocessing')
        self.preprocessor = self.preprocessor_class(
            self.season_definitions,
            statistic_definitions,
            statistic_definitions_path,
            timeseries_format,
            calculation_period,
            completeness_threshold,
            metadata_path,
            timeseries_folder,
            output_folder,
            point_statistics_filename,
            cross_correlation_filename,
            phi_filename,
            outlier_method,
            maximum_relative_difference,
            maximum_alterations
        )
        self.preprocessor.run()

    def fit(
            self,
            preprocessor=None,
            reference_statistics=None,
            reference_statistics_path=None,
            output_folder=None,
            parameters_filename=None,
            statistics_filename=None,
            cross_correlation_filename=None,
            parameter_bounds=None
    ):
        print('  Fitting')
        if preprocessor is not None:
            reference_statistics = preprocessor.statistics
            reference_statistics_path = reference_statistics_path
        elif self.preprocessor is not None:
            reference_statistics = self.preprocessor.statistics
            reference_statistics_path = None

        self.fitter = self.fitter_class(
            self.season_definitions,
            reference_statistics,
            reference_statistics_path,
            output_folder,
            parameters_filename,
            statistics_filename,
            cross_correlation_filename,
            parameter_bounds
        )
        self.fitter.run()

    def simulate(
            self,
            fitter=None,
            output_types=None,
            output_folder=None,
            output_format=None,
            parameters=None,
            output_prefix=None,
            points=None,
            catchments=None,
            catchment_id_field=None,
            grid=None,
            cell_size=None,
            dem=None,
            phi=None,
            number_of_years=30,
            number_of_realisations=1,
            concatenate_output=False,
            equal_length_output=False,
            timestep_length=1,
            start_year=2000,
            calendar='gregorian',
    ):
        print('  Simulating')
        if fitter is not None:
            parameters = fitter.parameters
        elif self.fitter is not None:
            parameters = self.fitter.parameters

        self.simulator = self.simulator_class(
            output_types=output_types,
            output_folder=output_folder,
            output_format=output_format,
            output_prefix=output_prefix,
            season_definitions=self.season_definitions,
            process_class=self.process_class,
            parameters=parameters,
            points=points,
            catchments=catchments,
            catchment_id_field=catchment_id_field,
            grid=grid,
            cell_size=cell_size,
            dem=dem,
            phi=phi,
            number_of_years=number_of_years,
            number_of_realisations=number_of_realisations,
            concatenate_output=concatenate_output,
            equal_length_output=equal_length_output,
            timestep_length=timestep_length,
            start_year=start_year,
            calendar=calendar,
        )

        self.simulator.run()




