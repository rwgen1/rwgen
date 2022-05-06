import sys
import datetime  # temporary

import numpy as np
import pandas as pd

from . import base
from . import utils
from . import properties


class Preprocessor(base.Preprocessor):

    def __init__(
            self,
            season_definitions=None,
            statistic_definitions=None,
            statistic_definitions_path=None,
            timeseries_format='csv',
            calculation_period=None,
            completeness_threshold=80.0,
            timeseries_path=None,
            output_folder=None,
            point_statistics_filename=None,
            override_phi=False,
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
            None,  # cross_correlation_filename
            None,  # phi_filename
            override_phi,
            outlier_method,
            maximum_relative_difference,
            maximum_alterations
        )

        if self.statistic_definitions is None:
            dc = {
                1: {'weight': 1.0, 'duration': 1,  'name': 'variance'},
                2: {'weight': 2.0, 'duration': 1,  'name': 'skewness'},
                3: {'weight': 7.0, 'duration': 1,  'name': 'probability_dry', 'threshold': 0.2},
                4: {'weight': 6.0, 'duration': 24, 'name': 'mean'},
                5: {'weight': 2.0, 'duration': 24, 'name': 'variance'},
                6: {'weight': 3.0, 'duration': 24, 'name': 'skewness'},
                7: {'weight': 7.0, 'duration': 24, 'name': 'probability_dry', 'threshold': 0.2},
                8: {'weight': 6.0, 'duration': 24, 'name': 'autocorrelation', 'lag': 1},
            }

            # ---
            # TESTING
            # dc = {
            #     1: {'weight': 1.0, 'duration': 1, 'name': 'variance'},
            #     2: {'weight': 4.0, 'duration': 1, 'name': 'skewness'},
            #     3: {'weight': 2.0, 'duration': 1, 'name': 'probability_dry', 'threshold': 0.2},
            #     4: {'weight': 6.0, 'duration': 24, 'name': 'mean'},
            #     5: {'weight': 1.0, 'duration': 24, 'name': 'variance'},
            #     6: {'weight': 4.0, 'duration': 24, 'name': 'skewness'},
            #     7: {'weight': 2.0, 'duration': 24, 'name': 'probability_dry', 'threshold': 0.2},
            #     8: {'weight': 6.0, 'duration': 24, 'name': 'autocorrelation', 'lag': 1},
            # }
            # ---

            id_name = 'statistic_id'
            non_id_columns = ['name', 'duration', 'lag', 'threshold', 'weight']
            self.statistic_definitions = utils.nested_dictionary_to_dataframe(
                dc, id_name, non_id_columns
            )

        self.check_statistics_include_24hr_mean()

        self.timeseries_path = timeseries_path

        self.df = None  # could be passed in
        self.dfs = None  # possibly could be passed in

    def run(self):
        self.read_timeseries()
        self.prepare_data()
        self.aggregate_data()
        self.calculate_statistics()
        self.calculate_gs()
        self.calculate_phi()
        if self.output_folder is not None:
            self.write_statistics()

    def read_timeseries(self):
        if self.timeseries_format == 'csv':
            self.df = utils.read_csv_timeseries(self.timeseries_path)
        elif self.timeseries_format == 'csvy':
            self.df = utils.read_csvy_timeseries(self.timeseries_path)

    def prepare_data(self):
        df = self.df
        df.loc[df['value'] < 0.0] = np.nan

        # Subset required calculation period
        if self.calculation_period is not None:
            start_year = self.calculation_period[0]
            end_year = self.calculation_period[1]
            df = df.loc[(df.index.year >= start_year) & (df.index.year <= end_year)]

        # Apply season definitions and make a running UID for season that goes up by one at each
        # change in season through the time series (needed to identify season completeness)
        df['season'] = df.index.month.map(self.season_definitions)
        df['season_uid'] = df['season'].ne(df['season'].shift()).cumsum()

        # Mask periods not meeting data completeness threshold (close approximation). There is an
        # assumption of at least one complete version of each season in dataframe
        df['season_count'] = df.groupby('season_uid')['value'].transform('count')
        df['season_size'] = df.groupby('season_uid')['value'].transform('size')
        df['season_size'] = df.groupby('season')['season_size'].transform('median')
        df['completeness'] = df['season_count'] / df['season_size'] * 100.0
        df['completeness'] = np.where(df['completeness'] > 100.0, 100.0, df['completeness'])
        df.loc[df['completeness'] < self.completeness_threshold, 'value'] = np.nan
        df = df.loc[:, ['season', 'value']]

        # Apply trimming or clipping season-wise
        if self.outlier_method == 'trim':
            df['value'] = df.groupby('season')['value'].transform(
                utils.trim_array(self.maximum_relative_difference, self.maximum_alterations)
            )
        elif self.outlier_method == 'clip':
            df['value'] = df.groupby('season')['value'].transform(
                utils.clip_array(self.maximum_relative_difference, self.maximum_alterations)
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
        self.df = df

    def aggregate_data(self):
        self.dfs = {}
        for duration in np.unique(self.statistic_definitions['duration']):
            if duration % 1 == 0:
                resample_code = str(int(duration)) + 'H'
            else:
                resample_code = str(int(round(duration * 60))) + 'T'
            df1 = self.df['value'].resample(resample_code, closed='left', label='left').sum()
            df2 = self.df['value'].resample(resample_code, closed='left', label='left').count()
            df1.values[df2.values < duration] = np.nan
            df1 = df1.to_frame()
            df1['season'] = df1.index.month.map(self.season_definitions)
            self.dfs[duration] = df1
            self.dfs[duration] = self.dfs[duration][self.dfs[duration]['value'].notnull()]

    def calculate_statistics(self):
        statistic_functions = {'mean': 'mean', 'variance': np.var, 'skewness': 'skew'}
        statistics = []
        for index, row in self.statistic_definitions.iterrows():
            statistic_name = row['name']
            duration = row['duration']
            if statistic_name in ['mean', 'variance', 'skewness']:
                values = self.dfs[duration].groupby('season')['value'].agg(statistic_functions[statistic_name])
            elif statistic_name == 'probability_dry':
                threshold = row['threshold']
                values = self.dfs[duration].groupby('season')['value'].agg(utils.probability_dry(threshold))
            elif statistic_name == 'autocorrelation':
                lag = row['lag']
                values = self.dfs[duration].groupby('season')['value'].agg(utils.autocorrelation(lag))

            values = values.to_frame('value')
            values.reset_index(inplace=True)
            for column in self.statistic_definitions.columns:
                if column not in values.columns:
                    values[column] = row[column]
            statistics.append(values)

        self.statistics = pd.concat(statistics)
        ordered_columns = list(self.statistic_definitions.columns)
        ordered_columns.extend(['season', 'value'])
        self.statistics = self.statistics[ordered_columns]

    def calculate_gs(self):
        gs = self.statistics.groupby(['statistic_id', 'name'])['value'].mean()
        gs = gs.to_frame('gs')
        gs.reset_index(inplace=True)
        gs.loc[gs['name'] == 'probability_dry', 'gs'] = 1.0
        gs.loc[gs['name'] == 'autocorrelation', 'gs'] = 1.0
        gs = gs[['statistic_id', 'gs']]
        self.statistics = pd.merge(self.statistics, gs, how='left', on='statistic_id')

    def calculate_phi(self):
        self.phi = self.statistics.loc[
            (self.statistics['name'] == 'mean') & (self.statistics['duration'] == 24),
            ['season', 'value']
        ].copy()
        self.phi['value'] /= 3.0
        self.phi.rename({'value': 'phi'}, axis=1, inplace=True)
        if self.override_phi:
            self.phi['phi'] = 1.0
        self.statistics = pd.merge(self.statistics, self.phi, how='left', on='season')


class Fitter(base.Fitter):

    def __init__(
            self,
            season_definitions,
            reference_statistics,
            reference_statistics_path=None,  # ?
            output_folder=None,
            parameters_filename=None,
            point_statistics_filename=None,
            parameter_bounds=None
    ):
        super().__init__(
            season_definitions, reference_statistics, reference_statistics_path, output_folder, parameters_filename,
            point_statistics_filename
        )
        self.cross_correlation_path = None  # override

        self.parameter_names = ['lamda', 'beta', 'nu', 'eta', 'xi']
        if parameter_bounds is None:
            self.parameter_bounds = [
                (0.00001, 0.02),    # lamda
                (0.02, 1.0),        # beta
                (0.1, 30),          # nu
                (0.1, 60.0),        # eta
                (0.01, 4.0),        # xi
            ]
        else:
            self.parameter_bounds = parameter_bounds
        self.parameter_output_columns = [
            'season', 'lamda', 'beta', 'nu', 'eta', 'xi', 'fit_success', 'objective_function', 'number_of_iterations',
            'number_of_evaluations'
        ]

    @classmethod
    def analytical_properties_wrapper(cls, parameters, statistic_ids, fitting_data):
        lamda, beta, nu, eta, xi = parameters

        # lamda = 0.012907
        # beta = 0.043389
        # nu = 9.021041
        # eta = 1.515229
        # xi = 0.895981

        mod_stats = cls.calculate_analytical_properties(
            statistic_ids, fitting_data, lamda, beta, eta, xi, nu
        )
        return mod_stats


class Process(base.Process):

    def simulate_raincells(self):
        # Temporarily merging parameters here, but can be done before this method is called if generalise
        # _storm_arrays_by_raincell() method to work using all columns
        tmp = pd.merge(self.storms, self.parameters, how='left', on='month')
        tmp.sort_values(['storm_id'], inplace=True)  # checks that order matches self.storms

        self.number_of_raincells_by_storm = self.rng.poisson(tmp['nu'].values)
        # self.number_of_raincells_by_storm = self.rng.poisson(self.parameters['nu'], self.number_of_storms)

        self.number_of_raincells = np.sum(self.number_of_raincells_by_storm)
        storm_ids_by_raincell, storm_arrivals_by_raincell, storm_months_by_raincell = self._storm_arrays_by_raincell(
            self.number_of_raincells_by_storm, self.storms['storm_id'].values, self.storms['storm_arrival'].values,
            self.storms['month'].values
        )
        self.df = pd.DataFrame({
            'storm_id': storm_ids_by_raincell,
            'storm_arrival': storm_arrivals_by_raincell,
            'month': storm_months_by_raincell,
            # 'raincell_arrival': np.zeros(self.number_of_raincells)  # dummy - remove?
        })


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
        # period_rc_x = raincell_x_coords[temporal_mask]
        # period_rc_y = raincell_y_coords[temporal_mask]
        # period_rc_radii = raincell_radii[temporal_mask]

        # self.discretise_point(
        utils.discretise_point(
            start_time,
            end_time,
            timestep_length,
            raincell_arrival_times,
            raincell_end_times,
            raincell_intensities,
            discrete_rainfall['point'][:, 0]  # discrete_rainfall
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
            timeseries_path=None,
            output_folder=None,
            point_statistics_filename=None,
            override_phi=True,
            outlier_method=None,
            maximum_relative_difference=2.0,
            maximum_alterations=5
    ):
        """
        Calculate statistics required in model fitting.

        Args:
            statistic_definitions (dict): Month identifier (as integer 1-12) as key and season identifier (integer) as
                value
            statistic_definitions_path:
            timeseries_format:
            calculation_period:
            completeness_threshold:
            timeseries_path:
            output_folder:
            point_statistics_filename:
            override_phi:
            outlier_method:
            maximum_relative_difference:
            maximum_alterations:

        """
        print('  Preprocessing')
        self.preprocessor = self.preprocessor_class(
            self.season_definitions,
            statistic_definitions,
            statistic_definitions_path,
            timeseries_format,
            calculation_period,
            completeness_threshold,
            timeseries_path,
            output_folder,
            point_statistics_filename,
            override_phi,
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
            parameter_bounds=None
    ):
        """
        Optimise model parameters.

        Args:
            preprocessor:
            reference_statistics:
            reference_statistics_path:
            output_folder:
            parameters_filename:
            statistics_filename:
            parameter_bounds:

        """
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
            parameter_bounds
        )
        self.fitter.run()

    def simulate(
            self,
            fitter=None,
            output_folder=None,
            output_format='txt',
            parameters=None,
            process_class=Process,
            output_prefix='point',
            number_of_years=30,
            number_of_realisations=1,
            concatenate_output=False,
            equal_length_output=False,
            timestep_length=1,
            start_year=2000,
            calendar='gregorian',
    ):
        """
        Simulate rainfall time series.

        Args:
            fitter:
            output_folder:
            output_format:
            parameters:
            process_class:
            output_prefix:
            number_of_years:
            number_of_realisations:
            concatenate_output:
            equal_length_output:
            timestep_length:
            start_year:
            calendar:

        """
        print('  Simulating')
        if fitter is not None:
            parameters = fitter.parameters
        elif self.fitter is not None:
            parameters = self.fitter.parameters

        self.simulator = self.simulator_class(
            output_types=['point'],
            output_folder=output_folder,
            output_format=output_format,
            output_prefix=output_prefix,
            season_definitions=self.season_definitions,
            process_class=self.process_class,
            parameters=parameters,
            number_of_years=number_of_years,
            number_of_realisations=number_of_realisations,
            concatenate_output=concatenate_output,
            equal_length_output=equal_length_output,
            timestep_length=timestep_length,
            start_year=start_year,
            calendar=calendar,
        )

        # self.simulator.simulate_realisation(1, 2000)
        self.simulator.run()




