import os
import sys
import datetime
import itertools

import numpy as np
import pandas as pd
import scipy.special
import scipy.interpolate
import numba
import gstools

from ..rainfall import simulation as rainfall_simulation
from . import fao56
from ..rainfall import utils


# TODO: Implement option for elevation in interpolation - also needs to be implemented in preprocessing
# - also useful to know if more/less bins can help - number of bins varying with number of stations?

# - assuming that always 31 days of rainfall coming in - fine if runs are always jan-dec, as dec has 31 days, but it
# could be made more flexible


class Simulator:

    def __init__(
            self,
            spatial_model,
            wet_threshold,
            predictors,
            input_variables,  # i.e. what needs to be simulated using autoregressive models
            output_variables,  # what to write out (including pet)
            timestep,  # timestep or rainfall input and weather output (weather simulated daily then disaggregated)
            season_length,

            raw_statistics,
            transformed_statistics,
            transformations,
            # regressions,
            parameters,
            r2,
            standard_errors,  # new
            statistics_variograms,
            residuals_variograms,  # covariance model parameters - to be replaced by noise_models?
            r2_variograms,
            se_variograms,  # new
            noise_models,

            discretisation_metadata,
            output_types,

            random_seed,

            dem,

            residual_method,

            wind_height,

            latitude,  # assume input is decimal degrees (negative for southern hemisphere)
            longitude,  # assume input is decimal degrees east of 0 (i.e. needs to be converted for subdaily ra)

            bc_offset,  # used in Box-Cox back-transformations

            # output_folder,  # ?? or can this be handled by rainfall model simulator - just pass rainfall back ??
            # output subfolders / output file naming?
            # output_paths,

            ###

            # ! spatial domain could also be set/updated after initialisation ! - so defaults as none
            # grid_metadata,  # needed? - or just xmin, ymin, ... ?
            # xmin,
            # xmax,
            # ymin,
            # ymax,
            # cell_size,
            # mask,  # indicating discretisation points

    ):
        self.spatial_model = spatial_model
        self.wet_threshold = wet_threshold
        self.predictors = predictors
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.timestep = timestep
        self.season_length = season_length

        if self.season_length == 'month':
            self.seasons = list(range(1, 12+1))
        elif self.season_length == 'half-month':
            self.seasons = list(range(1, 24 + 1))

        self.raw_statistics = raw_statistics
        self.transformed_statistics = transformed_statistics
        self.transformations = transformations
        # self.regressions = regressions
        self.parameters = parameters
        self.r2 = r2
        self.standard_errors = standard_errors
        self.statistics_variograms = statistics_variograms
        self.residuals_variograms = residuals_variograms
        self.r2_variograms = r2_variograms
        self.se_variograms = se_variograms
        self.noise_models = noise_models

        self.discretisation_metadata = discretisation_metadata

        self.final_output_types = output_types  # actual output types ['point', 'catchment', 'grid]
        self.output_types = []  # output types for internal use - 'point' and/or 'grid'
        for output_type in self.final_output_types:
            if output_type == 'catchment':
                if 'grid' not in self.output_types:
                    self.output_types.append('grid')
            if output_type == 'grid':
                if 'grid' not in self.output_types:
                    self.output_types.append(output_type)
            if output_type == 'point':
                self.output_types.append(output_type)

        if random_seed is None:
            self.seed_sequence = np.random.SeedSequence()
        else:
            self.seed_sequence = np.random.SeedSequence(random_seed)
        self.rng = np.random.default_rng(self.seed_sequence)

        self.dem = dem

        self.residual_method = residual_method

        self.wind_height = wind_height

        self.latitude = latitude * np.pi / 180.0  # assume input is decimal degrees and convert to radians

        # Assume longitude input is decimal degrees east of 0 and convert to west of 0
        # =IF(D2=0,0,IF(AND(D2>0,D2<=180),360-D2,IF(AND(D2<0,D2>=-180),D2*-1)))
        if longitude == 0.0:
            self.longitude = longitude
        elif (longitude > 0.0) and (longitude <= 180.0):
            self.longitude = 360.0 - longitude
        elif (longitude < 0.0) and (longitude >= -180.0):
            self.longitude = longitude * -1

        self.bc_offset = bc_offset  # used in box-cox back-transformations

        # ---

        self.covariance_models = None
        self.field_generators = None
        if self.spatial_model:
            self.covariance_models = {}  # for residuals
            self.field_generators = {}  # for residuals
            for season, variable in itertools.product(self.seasons, self.input_variables):
                # print(season, variable, self.residuals_variograms[(season, variable)])

                # TODO: Handle case where residuals_variogram is None (i.e. insufficient points...)
                # - try to use a neighbouring season?
                # - try to guess using (1) explained variance and (2) length scale from other variable(s)
                # - apply a constant random residual/error term across whole domain?
                # -- this option might be needed anyway in cases where e.g. only one station is available

                # replacing residuals_variograms with noise_models
                # !221210 if self.noise_models[(season, variable)] is not None:
                if len(self.noise_models.keys()) > 0:  # !221210
                    self.covariance_models[(season, variable)] = gstools.Exponential(
                        dim=2,
                        var=self.noise_models[(season, variable)][0],
                        len_scale=self.noise_models[(season, variable)][1]
                    )
                    self.field_generators[(season, variable)] = gstools.SRF(self.covariance_models[(season, variable)])

        ###

        self.simulation_variables = self.input_variables.copy()
        self.simulation_variables.append('prcp')

        # needs to be stratified by point/grid discretisation type too...
        self.lag_z_scores = {}
        self.lag_values = {}  # this will be lag of daily values, as that is what underpins regressions
        self.z_scores = {}
        self.values = {}  # daily weather values for current month
        self.n_points = {}
        for output_type in self.output_types:
            if self.spatial_model:
                self.n_points[output_type] = self.discretisation_metadata[(output_type, 'x')].shape[0]
            else:
                self.n_points[output_type] = 1
            for variable in self.simulation_variables:
                # for lag in [1, 2]:
                self.lag_z_scores[(output_type, variable)] = np.zeros((2, self.n_points[output_type]))  # daily
                self.lag_values[(output_type, variable)] = np.zeros((2, self.n_points[output_type]))  # daily
                self.z_scores[(output_type, variable)] = np.zeros((33, self.n_points[output_type]))  # daily
                self.values[(output_type, variable)] = np.zeros((33, self.n_points[output_type]))  # daily

        # Rather than e.g. pred_ddd, can loop transitions and put in e.g. dict of arrays
        self.transition_key = {
            1: 'DDD',
            2: 'DD',
            3: 'DW',
            4: 'WD',
            5: 'WW',
        }

        # self.rainfall = {}
        # for output_type in self.output_types:
        #     n_points = self.discretisation_metadata[(output_type, 'x')].shape[0]
        #     self.rainfall[output_type] = np.zeros((31 * 24, n_points))  # for storing daily rainfall

        # initialisation of output variables

        # !! interpolation of stuff to discretisation locations !!

        self.interpolated_parameters = {}
        if self.spatial_model:
            # self.interpolated_parameters = {}
            # - includes statistics and r2 currently
            # -- raw vs transformed statistics - just raw for now
            # - could ultimately include parameters
            self.interpolate_parameters()
        else:
            tmp = []  # temporary list for looking at r2  # TODO: Remove
            tmp_pars = []
            transitions = ['DDD', 'DD', 'DW', 'WD', 'WW']
            for season, variable, in itertools.product(self.seasons, self.simulation_variables):
                # raw statistics
                for statistic in ['mean', 'std']:
                    self.interpolated_parameters[('raw_statistics', 'point', variable, season, statistic)] = (
                        self.raw_statistics.loc[
                            (self.raw_statistics['variable'] == variable) & (self.raw_statistics['season'] == season),
                            statistic
                        ].values
                    )

                # r2 and se
                point_id = 1
                for transition in transitions:
                    key = (point_id, season, variable, transition)
                    if key in self.r2.keys():
                        self.interpolated_parameters[('r2', 'point', variable, season, transition)] = self.r2[key]
                        self.interpolated_parameters[('se', 'point', variable, season, transition)] = (
                            self.standard_errors[key]
                        )
                        # TODO: Why not storing r2 per point? - WRONG??

                        # TEMP/TESTING  # TODO: Remove
                        tmp.append([variable, season, transition, self.r2[key]])
                    _ = [variable, season, transition]
                    key = (1, season, variable, transition)
                    if key in self.parameters.keys():
                        _.extend(self.parameters[(1, season, variable, transition)].tolist())
                        tmp_pars.append(_)

            # self.interpolated_parameters[('raw_statistics', output_type, variable, season, statistic)] = values
            # - values contains one value per point (ordered according to discretisation metadata)
            # - so for single site statistics can just be one number - or array of size 1...
            # - access from self.raw_statistics df and (2) self.r2 dictionary

            # self.interpolated_parameters[('r2', output_type, variable, season, transition)] = values
            # self.r2[(row['point_id'], season, variable, transition)]  # one number (i.e. point_id in key)

        # print(self.r2)
        # print(self.interpolated_parameters)
        # sys.exit()

        # TESTING  # TODO: Remove
        # fp = 'Z:/DP/Work/ER/rwgen/testing/weather/r2.csv'
        # with open(fp, 'w') as fh:
        #     fh.write('variable,season,transition,r2\n')
        #     for case in tmp:
        #         ol = ','.join(str(item) for item in case)
        #         fh.write(ol + '\n')
        # sys.exit()
        # fp = 'Z:/DP/Work/ER/rwgen/testing/weather/pars.csv'
        # with open(fp, 'w') as fh:
        #     fh.write('variable,season,transition,par\n')
        #     for case in tmp_pars:
        #         ol = ','.join(str(item) for item in case)
        #         fh.write(ol + '\n')
        # sys.exit()

        # TODO: RESUME HERE - issue is that point model needs to look up parameters in interpolated_parameters
        # - otherwise attempting to find parameters in a None object currently
        # - for non-spatial model can simply set "interpolated" parameters as equal to "uninterpolated" parameters

        # !! adjustments to match input means... !!

        self.output_paths = None  # {}
        self.output_arrays = {}

        for output_type in self.output_types:
            if 'pet' in self.output_variables:
                self.values[(output_type, 'pet')] = np.zeros((33, self.n_points[output_type]))  # daily

        self.temp_profiles = {}
        self.pet_profiles = {}
        if self.timestep < 24:
            self.create_disaggregation_profiles()

        self.disaggregated_values = {}
        for output_type in self.output_types:
            for variable in self.output_variables:
                self.disaggregated_values[(output_type, variable)] = (
                    np.zeros((int(31 * (24 / self.timestep)), self.n_points[output_type]))
                )

        # !221209 - transformed statistics as dictionary
        # print(self.transformed_statistics.shape)
        # sys.exit()
        self.transformed_statistics_dict = {}
        for _, row in self.transformed_statistics.iterrows():
            self.transformed_statistics_dict[(row['pool_id'], row['variable'], row['season'])] = [  # , row['transition']
                row['bc_mean'], row['bc_std']
            ]
            # print(row)

        # !221212 - create interpolator function for inverse cdf of beta distribution (i.e. see if faster than ppf)
        self.sundur_beta_ppf_funcs = {}
        pool_id = 1
        x = np.arange(0.0, 1.0+0.0001, 0.001)
        for season in self.seasons:
            y = scipy.stats.beta.ppf(
                x,
                self.transformations[(pool_id, 'sun_dur', season, 'a')],
                self.transformations[(pool_id, 'sun_dur', season, 'b')],
                self.transformations[(pool_id, 'sun_dur', season, 'loc')],
                self.transformations[(pool_id, 'sun_dur', season, 'scale')],
            )
            f = scipy.interpolate.interp1d(x, y, bounds_error=False)
            self.sundur_beta_ppf_funcs[(pool_id, season)] = f

    def interpolate_parameters(self):  # need to know all output points (point points and grid points)
        # ! if no variogram just try IDW? !
        # - include statistics and r2 as "parameters"
        # - need to interpolate raw statistics - transformed statistics same at each point in pool(?)

        # Raw statistics - mean and standard deviation by variable
        for variable, season, statistic in itertools.product(self.simulation_variables, self.seasons, ['mean', 'std']):
            interpolation_type, interpolator = self.statistics_variograms[(season, variable, statistic)]
            for output_type in self.output_types:
                if interpolation_type == 'kriging':
                    if self.dem is not None:
                        values = interpolator(
                            (self.discretisation_metadata[(output_type, 'x')],
                             self.discretisation_metadata[(output_type, 'y')]),
                            mesh_type='unstructured',
                            ext_drift=self.discretisation_metadata[(output_type, 'z')],
                            return_var=False
                        )
                    else:
                        values = interpolator(
                            (self.discretisation_metadata[(output_type, 'x')],
                             self.discretisation_metadata[(output_type, 'y')]),
                            mesh_type='unstructured',
                            return_var=False
                        )
                elif interpolation_type == 'idw':
                    values = interpolator(
                        (self.discretisation_metadata[(output_type, 'x')],
                         self.discretisation_metadata[(output_type, 'y')])
                    )
                self.interpolated_parameters[('raw_statistics', output_type, variable, season, statistic)] = values

                # --
                # if output_type == 'grid':
                #     print(variable, season, statistic, values)
                #     print(
                #         self.raw_statistics.loc[
                #             (self.raw_statistics['variable'] == variable) & (self.raw_statistics['season'] == season)
                #         ]
                #     )
                #     print(interpolator.model)
                #     tmp = pd.DataFrame({
                #         'x': self.discretisation_metadata[('grid', 'x')],
                #         'y': self.discretisation_metadata[('grid', 'y')],
                #         'val': values,
                #     })
                #     tmp.to_csv('H:/Projects/rwgen/working/iss13/weather/interp1.csv', index=False)
                #     sys.exit()
                #     # !! important to test whether accounting for elevation influence improves interpolation !!
                #     # !! also useful to know if more/less bins can help !!
                #     # - number of bins varying with number of stations?
                # --

        # Transformation parameters
        # - maybe one day...

        # Regression parameters
        # - maybe one day...

        # R-squared
        for variable, season, transition in itertools.product(
                self.input_variables, self.seasons, self.transition_key.values()
        ):
            interpolation_type, interpolator = self.r2_variograms[(season, variable, transition)]
            for output_type in self.output_types:
                if interpolation_type == 'kriging':
                    if self.dem is not None:
                        values = interpolator(
                            (self.discretisation_metadata[(output_type, 'x')],
                             self.discretisation_metadata[(output_type, 'y')]),
                            mesh_type='unstructured',
                            ext_drift=self.discretisation_metadata[(output_type, 'z')],
                            return_var=False
                        )
                    else:
                        values = interpolator(
                            (self.discretisation_metadata[(output_type, 'x')],
                             self.discretisation_metadata[(output_type, 'y')]),
                            mesh_type='unstructured',
                            return_var=False
                        )
                elif interpolation_type == 'idw':
                    values = interpolator(
                        (self.discretisation_metadata[(output_type, 'x')],
                         self.discretisation_metadata[(output_type, 'y')])
                    )
                self.interpolated_parameters[('r2', output_type, variable, season, transition)] = values

                # print(variable, season, transition, values)

        # --
        # COPIED FROM ABOVE  # TODO: Rationalise
        # Standard errors
        for variable, season, transition in itertools.product(
                self.input_variables, self.seasons, self.transition_key.values()
        ):
            interpolation_type, interpolator = self.se_variograms[(season, variable, transition)]
            for output_type in self.output_types:
                if interpolation_type == 'kriging':
                    if self.dem is not None:
                        values = interpolator(
                            (self.discretisation_metadata[(output_type, 'x')],
                             self.discretisation_metadata[(output_type, 'y')]),
                            mesh_type='unstructured',
                            ext_drift=self.discretisation_metadata[(output_type, 'z')],
                            return_var=False
                        )
                    else:
                        values = interpolator(
                            (self.discretisation_metadata[(output_type, 'x')],
                             self.discretisation_metadata[(output_type, 'y')]),
                            mesh_type='unstructured',
                            return_var=False
                        )
                elif interpolation_type == 'idw':
                    values = interpolator(
                        (self.discretisation_metadata[(output_type, 'x')],
                         self.discretisation_metadata[(output_type, 'y')])
                    )
                self.interpolated_parameters[('se', output_type, variable, season, transition)] = values
        # --

    def adjust_means(self):  # e.g. to match haduk-grid statistics
        pass

    def simulate(
            self,
            rainfall,
            n_timesteps,  # in month - expressed in rainfall timestep (e.g. not necessarily days)
            year,
            month,
    ):
        # print(rainfall['point'].shape, n_timesteps, month)  # rainfall['grid'].shape,
        # simulation of transformed+standardised variables
        #

        # assume that may need to aggregate rainfall to daily timestep
        # - which dimension is time and which is point?
        # -- first = time and second = point - check though

        # - keep hold of previous two timesteps (days) so they can be used in the transitions
        # - so rainfall_lag1 and rainfall_lag2 can be initialised earlier and then updated here
        # - needs to be done for non-rainfall variables too... (only lag-1 needed though)

        # - careful with different month lengths though...
        # -- better literally just to store the required 1-2 lags per point
        # TODO: Careful here - not currently accounting for different month lengths !!

        # - transitions:
        #   1 - DDD
        #   2 - DD
        #   3 - DW
        #   4 - WD
        #   5 - WW

        # self.regressions[(half_month, variable, transition, 'parameters')]

        # --
        # for k, v in self.parameters.items():
        #     print(k, v)
        # # print(self.parameters[(pool_id, season, variable, transition_name)])
        # sys.exit()
        # --

        # --
        # !221211 - testing numba dictionaries

        # self.transition_key_nb, self.z_scores_nb, self.parameters_nb, self.predictors_nb,
        # self.interpolated_parameters_nb,

        self.transition_key_nb = numba.typed.Dict.empty(numba.types.int64, numba.types.string)  # numba.types.unicode_type
        self.z_scores_nb = numba.typed.Dict.empty(numba.types.UniTuple(numba.types.string, 2), numba.float64[:,:])
        self.parameters_nb = numba.typed.Dict.empty(
            numba.types.Tuple([numba.types.int64, numba.types.int64, numba.types.string, numba.types.string]),
            numba.types.float64[:]
        )
        self.predictors_nb = numba.typed.Dict.empty(
            numba.types.UniTuple(numba.types.string, 2), numba.types.UniTuple(numba.types.string, 4)
        )
        self.interpolated_parameters_nb = numba.typed.Dict.empty(
            numba.types.Tuple([numba.types.string, numba.types.string, numba.types.string, numba.types.int64,
                               numba.types.string]),
            numba.types.float64[:]
        )
        # self.transformed_statistics_dict_nb = numba.typed.Dict.empty(
        #     numba.types.Tuple([numba.types.int64, numba.types.string, numba.types.int64, numba.types.string]),
        #     numba.float64[:]
        # )

        for k, v in self.transition_key.items():
            self.transition_key_nb[k] = v
            # print(k, v)
            # break
        # for k, v in self.z_scores.items():
        #     self.z_scores_nb[k] = v
            # print(k, v)
            # break
        for k, v in self.parameters.items():
            self.parameters_nb[k] = v
            # print(k, v)
            # break
        for k, vs in self.predictors.items():
            tmp = []
            i = 0
            for v in vs:
                tmp.append(v)
                i += 1
            while i < 4:
                tmp.append('na')
                i += 1
            self.predictors_nb[k] = tuple(tmp)
            # print(k, v)
            # break
        for k, v in self.interpolated_parameters.items():
            if isinstance(v, float):
                v_ = np.asarray([v])
            else:
                v_ = v
            self.interpolated_parameters_nb[k] = v_
            # print(k, v)
            # break
        # for k, v in self.transformed_statistics_dict.items():
        #     self.transformed_statistics_dict_nb[k] = np.asarray(v)
        #     # print(k, v)
        #     # break

        # sys.exit()

        # --

        # --
        # !221212 - testing vectorisation (helped by using transition id rather than name in some dictionary keys)
        # destandardise_
        # tmp = {}
        # for k, v in self.transformed_statistics_dict.items():
        #     new_k = [j for j in k[:-1]]
        #     new_k.append(self.transition_key[k[-1]])
        #     new_k = tuple(new_k)
        #     tmp[new_k] = v
        # self.transformed_statistics_dict = tmp

        # --

        # ---

        # Number of days in month / half-month
        n_days = int(n_timesteps / (24 / self.timestep))

        # Random residual/error component
        # - series of numbers per variable for point model
        # - series of fields per variable for spatial model
        # - if generate all upfront could be more efficient perhaps

        # !221209 - trying simulation from standard normal upfront
        _n = n_days * len(self.input_variables) * len(self.output_types)
        sn_sample = self.rng.standard_normal(_n)  # standard normal sample
        ri = 0  # counter for residual - for indexing sn_sample (increment after each day+variable combination)

        # Everything needs to be stratified by output type
        for output_type in self.output_types:

            # print(output_type)
            # t1 = datetime.datetime.now()

            # Ensure arrays of values are reset to zero
            for variable in self.simulation_variables:
                self.z_scores[(output_type, variable)].fill(0.0)
                self.values[(output_type, variable)].fill(0.0)
            if 'pet' in self.output_variables:
                self.values[(output_type, 'pet')].fill(0.0)

            # Construct arrays with space for first two lags at beginning (requiring lags from previous months)
            # TODO: Check that lag arrays have first position as lag-1 and second position as lag-2
            for variable in self.simulation_variables:
                self.z_scores[(output_type, variable)][0, :] = self.lag_z_scores[(output_type, variable)][0, :]
                self.z_scores[(output_type, variable)][1, :] = self.lag_z_scores[(output_type, variable)][1, :]
                self.values[(output_type, variable)][0, :] = self.lag_values[(output_type, variable)][0, :]
                self.values[(output_type, variable)][1, :] = self.lag_values[(output_type, variable)][1, :]

            # Aggregate input rainfall (current month) to daily timestep
            if self.timestep != 24:
                # t99a = datetime.datetime.now()
                self.values[(output_type, 'prcp')][2:,:] = aggregate_rainfall(
                    rainfall[output_type], self.n_points[output_type], int(24 / self.timestep),
                )
                # t99b = datetime.datetime.now()
                # tmp = aggregate_rainfall_TEST(
                #     rainfall[output_type], self.n_points[output_type], int(24 / self.timestep),
                # )
                # t99c = datetime.datetime.now()
                # if output_type == 'grid':
                #     print(self.values[(output_type, 'prcp')][-2,:])
                #     print(tmp[-2, :])
                #     print('agg1', t99b - t99a)
                #     print('agg2', t99c - t99b)
                #     # sys.exit()
            else:
                self.values[(output_type, 'prcp')][2:,:] = rainfall[output_type][:]

            # TODO: Rainfall needs to be standardised for regression (but un-standardised for identifying transitions)
            # - undertaken below

            # Identify transition states
            transitions = np.zeros((31, self.n_points[output_type]), dtype=int)

            # Order of assignment such that DDD can overwrite DD
            transitions = np.where(  # DD
                (self.values[(output_type, 'prcp')][2:, :] < self.wet_threshold)
                & (self.values[(output_type, 'prcp')][1:-1, :] < self.wet_threshold),
                2,
                transitions
            )
            transitions = np.where(  # DDD
                (self.values[(output_type, 'prcp')][2:, :] < self.wet_threshold)
                & (self.values[(output_type, 'prcp')][1:-1, :] < self.wet_threshold)
                & (self.values[(output_type, 'prcp')][:-2, :] < self.wet_threshold),
                1,
                transitions
            )
            transitions = np.where(  # DW
                (self.values[(output_type, 'prcp')][1:-1, :] < self.wet_threshold)
                & (self.values[(output_type, 'prcp')][2:, :] >= self.wet_threshold),
                3,
                transitions
            )
            transitions = np.where(  # WD
                (self.values[(output_type, 'prcp')][1:-1, :] >= self.wet_threshold)
                & (self.values[(output_type, 'prcp')][2:, :] < self.wet_threshold),
                4,
                transitions
            )
            transitions = np.where(  # WW
                (self.values[(output_type, 'prcp')][1:-1, :] >= self.wet_threshold)
                & (self.values[(output_type, 'prcp')][2:, :] >= self.wet_threshold),
                5,
                transitions
            )

            # print(self.values[(output_type, 'rainfall')][:,0])
            # print(transitions)
            # print(transitions.shape)
            # sys.exit()

            # Could check here that no zeros remaining in transitions - should not be needed

            # Standardise rainfall - if half-months then each half needs to be standardised separately
            # - requires interpolation of mean and sd to each output point to have been completed first
            # - assuming rainfall is not transformed, only standardised
            # - remember that number of points needs to be considered (i.e. standardisation by point)
            # !! making explicit that this is prcp - otherwise a bit fragile as depends on sim variable list order
            variable = 'prcp'
            if self.season_length == 'month':
                season = month
                rainfall_mean = self.interpolated_parameters[('raw_statistics', output_type, variable, season, 'mean')]
                rainfall_stdev = self.interpolated_parameters[('raw_statistics', output_type, variable, season, 'std')]
                rainfall_sa = (self.values[(output_type, 'prcp')] - rainfall_mean) / rainfall_stdev
            elif self.season_length == 'half-month':
                season = (month - 1) * 2 + 1
                rainfall_mean = self.interpolated_parameters[('raw_statistics', output_type, variable, season, 'mean')]
                rainfall_stdev = self.interpolated_parameters[('raw_statistics', output_type, variable, season, 'std')]
                rainfall_sa1 = (self.values[(output_type, 'prcp')] - rainfall_mean) / rainfall_stdev
                season = (month - 1) * 2 + 2
                rainfall_mean = self.interpolated_parameters[('raw_statistics', output_type, variable, season, 'mean')]
                rainfall_stdev = self.interpolated_parameters[('raw_statistics', output_type, variable, season, 'std')]
                rainfall_sa2 = (self.values[(output_type, 'prcp')] - rainfall_mean) / rainfall_stdev
                rainfall_sa = np.zeros(self.values[(output_type, 'prcp')].shape[0])
                rainfall_sa[:2] = rainfall_sa2[:2]  # TODO: Check that this is done correctly
                rainfall_sa[2:2+15] = rainfall_sa1[2:2+15]  # TODO: Check that this is done correctly
                rainfall_sa[15:] = rainfall_sa2[15:]  # TODO: Check that this is done correctly

            # !! PUTTING STANDARDISED ANOMALIES IN VALUES ARRAYS
            # - now switching to z-scores arrays
            self.z_scores[(output_type, 'prcp')][:] = rainfall_sa[:]

            # print(rainfall_sa)
            # sys.exit()

            # TODO: Case where regression coefficients vary by location vs constant across domain
            # - may just need to implement constant across domain for now... i.e. pool_id = 1
            # - unless adjust/broadcast regression parameter array so one number per output location
            # -- this might permit vectorised calculations

            # TODO: Remove hardcoding of pool_id and inflexibility on having only one pool
            pool_id = 1

            t1 = datetime.datetime.now()

            # !221212 *** TESTING ***
            for k, v in self.z_scores.items():
                self.z_scores_nb[k] = v
            residuals_dummy = np.zeros(self.n_points[output_type])
            for variable in self.input_variables:
                self.z_scores[(output_type, variable)][2:, :], ri = regressions(
                    n_days, self.season_length, month, variable, sn_sample, ri, self.transition_key_nb,
                    self.z_scores_nb, output_type, transitions, self.parameters_nb, pool_id, self.predictors_nb,
                    self.interpolated_parameters_nb, residuals_dummy,
                )
            """

            # Regressions
            # - maximum number of predictors is 4 - could always loop to 4, but if final terms are not required then
            # these coefficients could be zero (i.e. no contribution)
            # - still need to map the relevant coefficients out in array (spatially) though?
            # - or do regression for each transition state and then select the relevant one for each point
            for day in range(1, n_days+1):

                # Identify season based on month (argument) and day of month if using half-months
                if self.season_length == 'month':
                    season = month
                elif self.season_length == 'half-month':
                    if day <= 15:  # TODO: Check looping days here and getting half-months correct (hardcoded)
                        season = (month - 1) * 2 + 1
                    else:
                        season = (month - 1) * 2 + 2

                for variable in self.input_variables:

                    # Prepare (simulate) standard normal residual/error term
                    # residuals = self.rng.standard_normal(1)[0]  # !221209
                    residuals = sn_sample[ri]  # !221209

                    # t1a = datetime.datetime.now()

                    # Prediction of standardised anomalies
                    # - day loop starts with one
                    # - only 31 days in transitions array (i.e. current month)
                    # - first value to store is in position 2
                    # TODO: Check all indexing here
                    # !221211 - commented out below to try numba version (essential code!)
                    # for idx in range(self.n_points[output_type]):
                    #     transition_id = transitions[day-1,idx]
                    #     transition_name = self.transition_key[transition_id]
                    #
                    #     # Intercept
                    #     self.z_scores[(output_type, variable)][day+1,idx] = (
                    #         self.parameters[(pool_id, season, variable, transition_name)][0]
                    #     )
                    #
                    #     # Multiplicative terms
                    #     i = 1
                    #     for predictor in self.predictors[(variable, transition_name)]:
                    #         # either (predictor, lag) tuples in self.predictors or parse here
                    #         # e.g. something like predictor, lag = predictor_variable.split('_')
                    #         if predictor.endswith('_lag1'):
                    #             predictor_variable = predictor.replace('_lag1', '')
                    #             lag = 1
                    #         else:
                    #             predictor_variable = predictor
                    #             lag = 0
                    #         self.z_scores[(output_type, variable)][day+1,idx] += (
                    #             self.parameters[(pool_id, season, variable, transition_name)][i]
                    #             * self.z_scores[(output_type, predictor_variable)][day+1-lag,idx]
                    #         )
                    #         i += 1

                    # # !221211 - trying out numba version of above code (initially modifying z_scores in-place)
                    # # t99a = datetime.datetime.now()
                    # self.z_scores[(output_type, variable)][day+1, :] = predict(
                    #     self.n_points[output_type], transitions, day, self.transition_key_nb, self.z_scores_nb,
                    #     output_type, variable, self.parameters_nb, pool_id, season, self.predictors_nb, residuals,
                    #     self.interpolated_parameters_nb,
                    # )
                    # t99b = datetime.datetime.now()
                    # print('prediction', t99b - t99a)

                    # !221210 - moved block in one tab
                    # Residual/error (random) component
                    # - sampled residual/error needs to be scaled according to 1 - r2
                    # - !! note that this applies the same random number to each point but scaled by spatially
                    #      varying 1 - r2
                    # residuals are sampled from standard normal distribution above now
                    # residuals *= (
                    #     1.0 - self.interpolated_parameters[('r2', output_type, variable, season, transition_name)]
                    # )

                    # --
                    # !221212 - testing vectorisation by transition state
                    for transition_id, transition_name in self.transition_key.items():

                        # Intercept
                        self.z_scores[(output_type, variable)][day+1, :] = np.where(
                            transitions[day-1, :] == transition_id,
                            self.parameters[(pool_id, season, variable, transition_name)][0],
                            self.z_scores[(output_type, variable)][day+1, :]
                        )

                        # Multiplicative terms
                        i = 1
                        for predictor in self.predictors[(variable, transition_name)]:
                            # either (predictor, lag) tuples in self.predictors or parse here
                            # e.g. something like predictor, lag = predictor_variable.split('_')
                            if predictor.endswith('_lag1'):  # TODO: Can this parsing be replaced with a lookup?
                                predictor_variable = predictor.replace('_lag1', '')
                                lag = 1
                            else:
                                predictor_variable = predictor
                                lag = 0
                            self.z_scores[(output_type, variable)][day+1, :] += np.where(
                                transitions[day-1, :] == transition_id,
                                self.parameters[(pool_id, season, variable, transition_name)][i]
                                * self.z_scores[(output_type, predictor_variable)][day+1-lag, :],
                                0.0
                            )
                            i += 1
                    # --

                        # !221212 - bringing inside loop so that transition state is referenced properly
                        # - previously it was just interiting from final transition state from loop above (i.e. error)
                        residuals *= np.where(
                            transitions[day-1, :] == transition_id,
                            self.interpolated_parameters[('se', output_type, variable, season, transition_name)],
                            1.0
                        )

                    #
                    # if variable == 'temp_avg':
                    #     print(residuals,
                    #           self.interpolated_parameters[('r2', output_type, variable, season, transition_name)])

                    # ***
                    # Spatial noise  # TODO: MORE WORK NEEDED
                    # if (self.residual_method == 'geostatistical') and (self.noise_models[(season, variable)] is not None):
                    #     # TODO: Set up field generator object outside of loop - SEE INIT - ALREADY DONE?
                    #     covariance_model = gstools.Exponential(
                    #         dim=2,
                    #         var=self.noise_models[(season, variable)][0],  # , transition_name
                    #         len_scale=self.noise_models[(season, variable)][1]
                    #     )
                    #     # print(self.noise_models[(season, variable)])
                    #     field_generator = gstools.SRF(covariance_model)
                    #     noise = field_generator(
                    #         (self.discretisation_metadata[(output_type, 'x')],
                    #          self.discretisation_metadata[(output_type, 'y')]),
                    #         seed=self.rng.integers(100000, 100000000),
                    #         mesh_type='unstructured'
                    #     )

                    # !221210 - moved block in one tab
                    # --
                    # if day == 1:
                    #     print(variable, self.noise_models[(season, variable)])
                    # --
                    #
                    # print(noise)
                    # residuals += noise  # !! COMMENTED OUT FOR TESTING ONLY !!
                    # ***
                    #
                    # print(residuals)
                    # print(self.values[(output_type, variable)][day+1,:])
                    # field_generator.plot()

                    # !221210 - moved block in one tab
                    # TESTING - omission of residuals  # TODO: Remove (undo commenting of line below)
                    # !221211 - TESTING - commented out line below while trying numba stuff - REVERT IF NOT USING NUMBA
                    self.z_scores[(output_type, variable)][day+1, :] += residuals

                    # !221210 - moved block in one tab
                    # print(self.values[(output_type, variable)][day+1, :])
                    # sys.exit()

                    # !221210 - moved block in one tab
                    # TESTING ONLY - set value as z-score to check it  # TODO: REMOVE THIS!!
                    # self.values[(output_type, variable)][day+1, :] = (  # !221212 - commented out (z-scores ok)
                    #     self.z_scores[(output_type, variable)][day+1, :]
                    # )

                    # self.interpolated_parameters[('r2', output_type, variable, season, transition)]

                    # Put into one dataframe to allow merging for conversion of standardised values to values
                    # !! not yet attempted !!
                    # for variable in self.input_variables:
                    #     self.z_scores[(output_type, variable)]

                    # !! *** RESUME HERE *** !!
                    # TODO: Check that simulated z-scores end up with mean=0 and sd=1 over a full simulation
                    # TODO: Check conversion of standardised variables to values
                    # - if z-scores are ok it indicates a problem in the destandardisation/transformation

                    # TESTING ONLY - comment out transformation  # TODO: REMOVE THIS!!
                    # 

                    # !221209 - conversion of z-scores to values moved from here

                    # Increment counter (index place) for standard normal sample
                    ri += 1

                    #   # TODO: REMOVE THIS!!

            t1b = datetime.datetime.now()
            # print('sa prediction', t1b - t1)
            # sys.exit()

            """
            # !221212 *** TESTING ***

            # ---
            # !221209 - trying to move conversion of z-scores to values outside of day+variable loop

            # TODO: Remove loops in favour of vectorised approach
            # Convert standardised variables to values
            # - destandardise once using mean and sd following transformation
            # - reverse transformations
            # - destandardise again using raw (interpolated) mean and sd
            # - checks on bounds
            # print(self.transformed_statistics)
            # print(self.transformed_statistics.columns)

            # !221210 - TESTING
            # """

            # First destandardisation
            # !221212 - testing vectorisation so commented out block below
            # for day in range(1, n_days+1):
            #     for variable in self.input_variables:
            #         # !221211 - testing numba so commented out below
            #         for idx in range(self.n_points[output_type]):
            #             transition_id = transitions[day - 1, idx]
            #             transition_name = self.transition_key[transition_id]
            #
            #             # Destandardise once using mean and sd following transformation
            #             mean_1 = self.transformed_statistics_dict[(pool_id, variable, season, transition_name)][0]
            #             sd_1 = self.transformed_statistics_dict[(pool_id, variable, season, transition_name)][1]
            #             self.values[(output_type, variable)][day + 1, idx] = (
            #                     self.z_scores[(output_type, variable)][day + 1, idx] * sd_1 + mean_1
            #             )
            #         # !221211 - testing numba
            #         # self.values[(output_type, variable)][day + 1, :] = destandardise_1(
            #         #     self.n_points[output_type], transitions, day, self.transition_key_nb,
            #         #     self.transformed_statistics_dict_nb, pool_id, variable, season,
            #         #     self.z_scores[(output_type, variable)][day + 1, :],
            #         # )
            # !221212 - testing vectorised destandardisation 1
            for variable in self.input_variables:
                # for transition_id, transition_name in self.transition_key.items():
                #     mean_1 = self.transformed_statistics_dict[(pool_id, variable, season, transition_name)][0]  #
                #     sd_1 = self.transformed_statistics_dict[(pool_id, variable, season, transition_name)][1]  #
                #     self.values[(output_type, variable)][2:, :] = np.where(
                #         transitions[:, :] == transition_id,
                #         self.z_scores[(output_type, variable)][2:, :] * sd_1 + mean_1,
                #         self.values[(output_type, variable)][2:, :]
                #     )
                mean_1 = self.transformed_statistics_dict[(pool_id, variable, season)][0]  # , transition_name
                sd_1 = self.transformed_statistics_dict[(pool_id, variable, season)][1]  # , transition_name
                self.values[(output_type, variable)][2:, :] = (
                    self.z_scores[(output_type, variable)][2:, :] * sd_1 + mean_1
                )

                # print(variable, season)
                # print(self.transformed_statistics_dict[(pool_id, variable, season)])
                # print(self.z_scores[(output_type, variable)][2:, 0])
                # print(self.values[(output_type, variable)][2:, 0])
                # sys.exit()

            # TODO: Need to check back through where _nb is used for arrays that are updated
            # - this should mainly be self.z_scores_nb currently
            # - may want just one attribute for simplicity, otherwise possibility that self.z_scores and
            # self.z_scores_nb are not in sync

            t1c = datetime.datetime.now()
            # print('destandardisation 1', t1c - t1b)
            # print(dt)
            # sys.exit()

            # Reverse transformation
            for variable in self.input_variables:
                # for idx in range(self.n_points[output_type]):
                # t99a = datetime.datetime.now()
                if variable in ['temp_avg', 'dtr', 'vap_press', 'wind_speed']:
                    self.values[(output_type, variable)][2:, :] = scipy.special.inv_boxcox(
                        self.values[(output_type, variable)][2:, :],
                        self.transformations[(pool_id, variable, season, 'lamda')]
                    )
                    # TESTING offset reversal  # TODO: Check this!!
                    self.values[(output_type, variable)][2:, :] -= self.bc_offset
                elif variable == 'sun_dur':
                    # TODO: Double-check that variable is standard normal after first destandardisation
                    # - cdf to get probabilities associated with the (standard normal) value
                    # - values less than or equal to zero threshold become zero
                    # - ppf to transform other probabilities to beta-distributed values
                    # [- use min/max bounds to convert from 0-1 to min-max range]  # TODO: Consider implementing
                    # p98a = datetime.datetime.now()
                    p0 = self.transformations[(pool_id, variable, season, 'p0')]
                    zero_threshold = scipy.stats.norm.ppf(p0)  # !221212 - this is effectively a parameter so calculate upfront and store in dictionary
                    # p98b = datetime.datetime.now()
                    # ~~
                    # if self.values[(output_type, variable)][:, idx] <= zero_threshold:
                    #     self.values[(output_type, variable)][:, idx] = 0.0
                    # else:
                    #     p = scipy.stats.norm.cdf(self.values[(output_type, variable)][:, idx])
                    #     p = (p - p0) / (1.0 - p0)
                    #     self.values[(output_type, variable)][:, idx] = scipy.stats.beta.ppf(
                    #         p,
                    #         self.transformations[(pool_id, variable, season, 'a')],
                    #         self.transformations[(pool_id, variable, season, 'b')],
                    #         self.transformations[(pool_id, variable, season, 'loc')],
                    #         self.transformations[(pool_id, variable, season, 'scale')],
                    #     )
                    # ~~
                    p = scipy.stats.norm.cdf(self.values[(output_type, variable)][2:, :])
                    p = (p - p0) / (1.0 - p0)
                    # p98c = datetime.datetime.now()
                    # !221212 - original call to beta.ppf - restore if issues with interpolator approach
                    # self.values[(output_type, variable)][2:, :] = scipy.stats.beta.ppf(
                    #     p,
                    #     self.transformations[(pool_id, variable, season, 'a')],
                    #     self.transformations[(pool_id, variable, season, 'b')],
                    #     self.transformations[(pool_id, variable, season, 'loc')],
                    #     self.transformations[(pool_id, variable, season, 'scale')],
                    # )
                    # p98c1 = datetime.datetime.now()

                    # !221212 - interpolator for beta inverse cdf (faster and same to 5+ decimal places)
                    f = self.sundur_beta_ppf_funcs[(pool_id, season)]
                    self.values[(output_type, variable)][2:, :] = f(p)

                    # testing
                    # tmp = scipy.special.btdtri(
                    #     self.transformations[(pool_id, variable, season, 'a')],
                    #     self.transformations[(pool_id, variable, season, 'b')],
                    #     p,
                    # )
                    # tmp *= self.transformations[(pool_id, variable, season, 'scale')]
                    # tmp += self.transformations[(pool_id, variable, season, 'loc')]
                    # p98c2 = datetime.datetime.now()
                    # if output_type == 'grid':
                    #     print(self.values[(output_type, variable)][2, :])
                    #     print(tmp[0, :])
                    #     print(p98c1 - p98c)
                    #     print(p98c2 - p98c1)
                    #     sys.exit()
                    # testing

                    # p98d = datetime.datetime.now()
                    self.values[(output_type, variable)][2:, :] = np.where(
                        (self.values[(output_type, variable)][2:, :] <= zero_threshold)
                        | (~np.isfinite(self.values[(output_type, variable)][2:, :])),
                        0.0,
                        self.values[(output_type, variable)][2:, :]
                    )
                    # p98e = datetime.datetime.now()

                    # if output_type == 'grid':
                    #     print(p98b - p98a)
                    #     print(p98c - p98b)
                    #     print(p98d - p98c)
                    #     print(p98e - p98d)
                    #     sys.exit()

                # t99b = datetime.datetime.now()
                # print(variable, t99b - t99a)

            t1d = datetime.datetime.now()
            # print('detransformation', t1d - t1c)

            # Destandardise again using raw (interpolated) mean and sd for all variables except sunshine
            # duration, which is rescaled based on min/max
            # !221212 - testing - commented out below below to try vectorised approach
            # for day in range(1, n_days+1):
            #     for variable in self.input_variables:
            #         for idx in range(self.n_points[output_type]):
            #             mean_2 = self.interpolated_parameters[('raw_statistics', output_type,
            #                                                    variable, season, 'mean')][idx]
            #             sd_2 = self.interpolated_parameters[('raw_statistics', output_type,
            #                                                  variable, season, 'std')][idx]
            #             if variable != 'sun_dur':
            #                 self.values[(output_type, variable)][day + 1, idx] = (
            #                         self.z_scores[(output_type, variable)][day + 1, idx] * sd_2 + mean_2
            #                 )
            #             else:
            #                 min_ = self.transformations[(pool_id, variable, season, 'obs_min')]
            #                 max_ = self.transformations[(pool_id, variable, season, 'obs_max')]
            #                 self.values[(output_type, variable)][day + 1, idx] = (
            #                         self.values[(output_type, variable)][day + 1, idx] * (max_ - min_) + min_
            #                 )

            # !221212 - vectorised approach
            for variable in self.input_variables:
                if variable != 'sun_dur':
                    mean_2 = self.interpolated_parameters[('raw_statistics', output_type, variable, season, 'mean')]
                    sd_2 = self.interpolated_parameters[('raw_statistics', output_type, variable, season, 'std')]
                    # self.values[(output_type, variable)][2:, :] = (  # !221212 - why is this based on z-scores?
                    #         self.z_scores[(output_type, variable)][2:, :] * sd_2 + mean_2
                    # )
                    self.values[(output_type, variable)][2:, :] = (  # !221212 - testing based on values - CHECK!!
                            self.values[(output_type, variable)][2:, :] * sd_2 + mean_2
                    )
                else:
                    min_ = self.transformations[(pool_id, variable, season, 'obs_min')]
                    max_ = self.transformations[(pool_id, variable, season, 'obs_max')]
                    self.values[(output_type, variable)][2:, :] = (
                            self.values[(output_type, variable)][2:, :] * (max_ - min_) + min_
                    )

            t1e = datetime.datetime.now()
            # print('destandardisation 2', t1e - t1d)

            # Checks on bounds
            if variable == 'dtr':
                self.values[(output_type, variable)] = np.maximum(
                    self.values[(output_type, variable)], 0.1
                )
            elif variable == 'vap_press':
                self.values[(output_type, variable)] = np.maximum(
                    self.values[(output_type, variable)], 0.01
                )
            elif variable == 'wind_speed':
                self.values[(output_type, variable)] = np.maximum(
                    self.values[(output_type, variable)], 0.01
                )
            elif variable == 'sun_dur':
                # TODO: Check/update these bounds once decided if using 0-1 or min-max range
                self.values[(output_type, variable)] = np.maximum(
                    self.values[(output_type, variable)], 0.0
                )
                # self.values[(output_type, variable)][day + 1, idx] = np.minimum(
                #     self.values[(output_type, variable)][day + 1, idx], 1.0
                # )
            
            # """
            # !221210 - TESTING

            # ~~
            # if np.any(~np.isfinite(self.values[(output_type, variable)])):
            #     print(self.values[(output_type, variable)])
            #     print(variable)
            #     sys.exit()
            # ~~

            # t1f = datetime.datetime.now()
            # print('bounds', t1f - t1e)

            # ---

            # t2 = datetime.datetime.now()
            # print(t2 - t1)
            # sys.exit()

            # print(np.mean(self.z_scores[(output_type, 'temp_avg')], axis=0),
            #       np.std(self.z_scores[(output_type, 'temp_avg')], axis=0))
            # sys.exit()

            # for variable in self.input_variables:
            #     # print(variable)
            #     fp = 'H:/Projects/rwgen/working/iss13/weather/test9_' + variable + '.csv'
            #     np.savetxt(fp, self.values[(output_type, variable)], fmt='%.2f', delimiter=',')
            # sys.exit()

            # TODO: Further checks/corrections on spatial noise simulation needed
            # - spatial noise seems to be too large in magnitude (e.g. looking at tavg)
            # -- try simulating fields of residuals directly (transform residuals to standard normal and then
            # back-transform based on 1-r2)
            # -- different approach to noise...??

            # Update lagged arrays with final days of month (first position is lag-1, second position is lag-2)
            # - take varying month lengths into account - e.g if n_days is 31, need to get value from position 32 (i.e.
            # starting from index 0), given that the values and z-scores arrays have (time) length 33 - i.e. 33rd value
            # is in position 32
            self.lag_z_scores[(output_type, variable)][0, :] = self.z_scores[(output_type, variable)][n_days+1,:]
            self.lag_z_scores[(output_type, variable)][1, :] = self.z_scores[(output_type, variable)][n_days, :]
            self.lag_values[(output_type, variable)][0, :] = self.values[(output_type, variable)][n_days+1, :]
            self.lag_values[(output_type, variable)][1, :] = self.values[(output_type, variable)][n_days, :]

            # PET calculations
            if 'pet' in self.output_variables:
                self.calculate_pet(year, month)

            # Disaggregate
            # t99a = datetime.datetime.now()
            self.disaggregate(n_days, season)
            # t99b = datetime.datetime.now()
            # print('disagg', t99b - t99a)

            t2 = datetime.datetime.now()
            # print(t2 - t1)
            # sys.exit()

    def calculate_pet(self, year, month):
        # - self.values has 33 days (i.e. two lags at beginning)
        # - do calculations based on actual month length
        # - then just take relevant part from pet array that gets stored in self.values

        # Array (1d) of days of year corresponding with month for potential/clear-sky radiation calculations
        doy = day_of_year(year, month)
        n_days = doy.shape[0]
        # doy = doy[:, None]

        for output_type in self.output_types:

            # Derive minimum and maximum temperatures and convert temperatures from [C] to [K]
            tmax = (
                self.values[(output_type, 'temp_avg')][2:2+n_days, :]
                + 0.5 * self.values[(output_type, 'dtr')][2:2+n_days, :]
            )
            tmin = (
                self.values[(output_type, 'temp_avg')][2:2+n_days, :]
                - 0.5 * self.values[(output_type, 'dtr')][2:2+n_days, :]
            )
            tmax += 273.15
            tmin += 273.15
            tavg = self.values[(output_type, 'temp_avg')][2:2+n_days, :] + 273.15

            # Estimate pressure [kPa] from elevation [m]
            # TODO: Need to pass in elevation of points / grid points
            # - is this available in discretisation metadata? - spatial model only...
            # - required for both point and spatial models - otherwise need to assume a pressure
            pres = fao56.atmos_pressure(self.discretisation_metadata[(output_type, 'z')])

            # Actual vapour pressure is already in correct units [kPa]
            avp = self.values[(output_type, 'vap_press')][2:2+n_days, :]

            # !221209 - nan occurred once possibly related to avp**0.5 - so try to ensure positive at least
            avp = np.maximum(avp, 0.000001)

            # Saturation vapour pressure [kPa]
            svp = fao56.mean_svp(tmin, tmax)

            # Slope of saturation vapour pressure curve [kPa C-1]
            delta_svp = fao56.delta_svp(tavg)

            # Extra-terrestrial radiation [MJ m-2 day-1]
            # - need to be able to broadcast time- and location-dependent terms to 2d arrays of variables, which have
            # first dimension as time and second dimension as a location i.e. a point or grid point
            dr = fao56.earth_sun_distance(doy)  # inverse relative earth-sun distance [-]
            dec = fao56.solar_declination(doy)  # solar declination [rad]
            dr = dr[:, None]  # adds a second dimension
            dec = dec[:, None]
            # lat = self.latitude
            lat = np.zeros((doy.shape[0], self.n_points[output_type]))  # self.latitude[None, :]
            lat.fill(self.latitude)
            omega = fao56.omega_(lat, dec)
            ra = fao56.extraterrestrial_radiation(dr, omega, lat, dec)  # extraterrestrial radiation [MJ m-2 day-1]

            # Solar radiation at the surface
            # - ra goes in with shape (n_times, n_points) and sun_dur has (n_times, n_points) - so rs should end up
            # as (n_times, n_points)
            # - omega is (n_times, 1) so N should be the same and thus broadcast
            N = fao56.daylight_hours(omega)  # potential / clear-sky
            rs = fao56.solar_radiation(ra, self.values[(output_type, 'sun_dur')][2:2+n_days, :], N)  # downwards solar
            rns = fao56.net_solar_radiation(rs)  # net downwards solar radiation

            # Net longwave radiation requires clear-sky solar radiation at the surface if estimating
            # TODO: Sort out where elevation is coming from
            rso = fao56.clear_sky_solar_radiation(self.discretisation_metadata[(output_type, 'z')], ra)
            rnl = fao56.net_longwave_radiation(tmin, tmax, avp, rs, rso)

            # Net radiation [MJ m-2 day-1]
            netrad = fao56.net_radiation(rns, rnl)

            # Wind speed height adjustment from 10m to 2m [m s-1]
            # - adjustment from 10 to 2m is required for MIDAS data
            if self.wind_height != 2.0:
                ws2 = fao56.windspeed_2m(self.values[(output_type, 'wind_speed')][2:2+n_days, :], self.wind_height)
            else:
                ws2 = self.values[(output_type, 'wind_speed')][2:2+n_days, :]

            # Assume soil heat flux of zero at daily timestep (FAO56 equation 42)
            shf = 0.0

            # Psychrometric constant
            psy = fao56.psy_const(pres)

            # Calculate ET0 [mm day-1]
            et0 = fao56.fao56_et0(delta_svp, netrad, shf, psy, tavg, ws2, svp, avp)
            self.values[(output_type, 'pet')][2:2+n_days, :] = et0

    # def untransform(self):
    #     pass

    def create_disaggregation_profiles(self):
        # Representative days of year for which to calculate profiles (one per month or half-month)
        # - ignore leap and non-leap for now - should not come out with much difference (i.e. +/- one day)
        doys = {}
        if self.season_length == 'month':
            for month in range(1, 12+1):
                doys[month] = datetime.datetime(2001, month, 15).timetuple().tm_yday
        elif self.season_length == 'half-month':
            for half_month in range(1, 24+1):
                month = int(np.ceil(half_month / 2))
                if half_month % 2 == 1:
                    day = 8
                if half_month % 2 == 0:
                    day = 23
                doys[half_month] = datetime.datetime(2001, month, day).timetuple().tm_yday

        # Create profiles
        for season in self.seasons:

            # Extra-terrestrial radiation for representative days at hourly or shorter period - min(1, timestep)
            # - better to stick with timesteps >= 1-hour for the moment
            doy = doys[season]
            dt = min(self.timestep, 1.0)
            t1 = np.arange(0, 24, self.timestep)
            t2 = t1 + self.timestep
            t = (t1 + t2) / 2.0  # midpoint time of timestep (period) - e.g. for 14:00-15:00, t = 14.5
            dr = fao56.earth_sun_distance(doy)
            dec = fao56.solar_declination(doy)
            lat = self.latitude
            Lm = self.longitude
            if Lm >= 352.5:  # inferring central meridian of time zones based on 15-degree increments
                Lz = 0.0
            else:
                Lz = 15.0 * np.around(Lm / 15.0)
            ra = fao56.subdaily_extraterrestrial_radiation(doy, dt, t, dr, dec, lat, Lm, Lz)

            # Normalised diurnal cycle for extra-terrestrial radiation (to be used for pet for now)
            # - fraction of cumulative ra
            ra_norm = ra / np.sum(ra)

            # Normalised diurnal cycle for temperature
            # - each hour as proportion of range
            # - offset puts max at 1500 and min at 0300 - hardcoded currently
            nt = int(24 / dt)
            x = np.arange(nt) * np.pi / (nt / 2.0)
            y = np.sin(x)
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
            offset = int((3.0 + 6.0) / dt)  # +6 needed to enable 1500 max and 0300 min
            temp_norm = np.zeros(y.shape)
            temp_norm[offset:] = y_norm[:-offset]
            temp_norm[:offset] = y_norm[-offset:]

            # Store
            temp_norm = np.tile(temp_norm, 31)
            temp_norm = np.expand_dims(temp_norm, 1)
            pet_norm = np.tile(ra_norm, 31)
            pet_norm = np.expand_dims(pet_norm, 1)
            self.temp_profiles[season] = temp_norm
            self.pet_profiles[season] = pet_norm

    def disaggregate(self, n_days, season):  # daily to sub-daily
        # - focus on temperature and pet
        # - other variables can just be uniform disaggregation for now at least
        # - remember that first two timesteps of self.values are lags from previous month or half-month

        # TODO: Rationalise internal and user-facing variable names
        # - tasmin and tasmax
        # - tertiary variables
        internal_variables = {
            'tas': 'temp_avg', 'dtr': 'dtr', 'vap': 'vap_press', 'sundur': 'sun_dur', 'wind_speed': 'ws10', 'pet': 'pet'
        }

        nt = int(24 / self.timestep)
        for output_type, variable in itertools.product(self.output_types, self.output_variables):

            self.disaggregated_values[(output_type, variable)].fill(0.0)

            if self.timestep == 24:
                if variable in ['tasmin', 'tasmax']:
                    dtr = self.values[(output_type, 'dtr')][2:2 + n_days, :]
                    tavg = self.values[(output_type, 'temp_avg')][2:2 + n_days, :]
                    if variable == 'tasmin':
                        var = tavg - 0.5 * dtr
                    elif variable == 'tasmax':
                        var = tavg + 0.5 * dtr
                    self.disaggregated_values[(output_type, variable)][:n_days, :] = var
                else:
                    self.disaggregated_values[(output_type, variable)][:n_days, :] = (
                        self.values[(output_type, internal_variables[variable])][2:2 + n_days, :]
                    )

            else:
                if variable == 'pet':
                    pet = np.repeat(self.values[(output_type, variable)][2:2 + n_days, :], nt, axis=0)
                    self.disaggregated_values[(output_type, variable)][:n_days*nt, :] = (
                        pet * self.pet_profiles[season][:n_days*nt, :]
                    )
                elif variable == 'tas':
                    dtr = np.repeat(self.values[(output_type, 'dtr')][2:2 + n_days, :], nt, axis=0)
                    tavg = np.repeat(self.values[(output_type, 'temp_avg')][2:2 + n_days, :], nt, axis=0)
                    tmin = tavg - 0.5 * dtr
                    self.disaggregated_values[(output_type, variable)][:n_days*nt, :] = (
                        self.temp_profiles[season][:n_days*nt, :] * dtr + tmin
                    )
                else:
                    var = np.repeat(
                        self.values[(output_type, internal_variables[variable])][2:2 + n_days, :], nt, axis=0
                    )
                    self.disaggregated_values[(output_type, variable)][:n_days*nt, :] = var

    # - output paths are currently figured out in the weather model (rather than simulator)
    # - this is because it is easiest to get the relevant information when it is being prepared for the rainfall
    # simulation (i.e. near the top of its main function)
    # - so for now the weather simulator needs to "feed off" the weather model to get the relevant paths
    # - may be useful to move things around...
    # def set_output_paths(
    #         self, spatial_model, output_types, output_format, output_folder, output_subfolders, point_metadata,
    #         catchment_metadata, realisation_ids, project_name
    # ):
    #     self.output_paths = rainfall_simulation.make_output_paths(
    #         spatial_model,
    #         output_types,  # !! need to distinguish discretisation types
    #         output_format,
    #         output_folder,
    #         output_subfolders,
    #         point_metadata,
    #         catchment_metadata,
    #         realisation_ids,
    #         project_name,
    #         variables=self.output_variables
    #     )

    def set_output_paths(self, output_paths):
        self.output_paths = output_paths

    def collate_outputs(
            self, output_types, spatial_model, points, catchments, realisation_id, timesteps_in_month,
            discretisation_metadata, month_idx,
    ):
        # TODO: Rationalise what is drawn from arguments vs attributes
        for variable in self.output_variables:
            rainfall_simulation.collate_output_arrays(
                output_types=output_types,
                spatial_model=spatial_model,
                points=points,
                catchments=catchments,
                realisation_id=realisation_id,
                values=self.disaggregated_values,
                variable=variable,
                timesteps_to_skip=0,  # 0 for self.disaggregated_values (would be 2 for self.values currently)
                timesteps_in_month=timesteps_in_month,
                discretisation_metadata=discretisation_metadata,
                month_idx=month_idx,
                output_arrays=self.output_arrays,
            )

    def write_output(self, write_new_files):
        rainfall_simulation.write_output(
            output_arrays=self.output_arrays,
            output_paths=self.output_paths,
            write_new_files=write_new_files,
            # number_format='%.2f',
        )

    # def set_domain(
    #         self,
    #         grid_metadata,  # needed? - or just xmin, ymin, ... ?
    #         xmin,
    #         xmax,
    #         ymin,
    #         ymax,
    #         cell_size,
    #         mask,  # indicating discretisation points
    # ):
    #     pass


# def aggregate_rainfall(x, n_points, window_size):
#     # assumes that array has dimensions (n_timesteps, n_points)
#     n_days = int(x.shape[0] / window_size)
#     y = np.zeros((n_days, n_points))
#     # j = 0
#     for j in range(n_points):
#         i = 0
#         for d in range(n_days):
#             y[d,j] = np.sum(x[i:i+window_size, j])
#             i += window_size
#         # j += 1
#     return y


def aggregate_rainfall(x, n_points, window_size):
    # assumes that array has dimensions (n_timesteps, n_points)
    # much faster than commented out version above
    n_days = int(x.shape[0] / window_size)
    y = np.zeros((n_days, n_points))
    i = 0
    for d in range(n_days):
        y[d, :] = np.sum(x[i:i+window_size, :], axis=0)
        i += window_size
    return y


def day_of_year(year, month):
    if utils.check_if_leap_year(year):
        pseudo_year = 2000
    else:
        pseudo_year = 2001

    doy_list = []
    d = datetime.datetime(pseudo_year, month, 1, 0)
    while (d.year == pseudo_year) and (d.month == month):
        doy_list.append(d.timetuple().tm_yday)
        d += datetime.timedelta(days=1)
    doy_array = np.asarray(doy_list)

    return doy_array


# ---

@numba.jit(nopython=True)
def regressions(
        n_days, season_length, month, variable, sn_sample, ri, transition_key, z_scores, output_type, transitions,
        parameters, pool_id, predictors, interpolated_parameters, residuals,
):
    for day in range(1, n_days + 1):

        # Identify season based on month (argument) and day of month if using half-months
        if season_length == 'month':
            season = month
        elif season_length == 'half-month':
            if day <= 15:  # TODO: Check looping days here and getting half-months correct (hardcoded)
                season = (month - 1) * 2 + 1
            else:
                season = (month - 1) * 2 + 2

        # Prepare (simulate) standard normal residual/error term
        # residuals = self.rng.standard_normal(1)[0]  # !221209
        residuals[:] = sn_sample[ri]  # !221209

        # Prediction of standardised anomalies
        # - day loop starts with one
        # - only 31 days in transitions array (i.e. current month)
        # - first value to store is in position 2
        for transition_id, transition_name in transition_key.items():

            # Intercept
            z_scores[(output_type, variable)][day + 1, :] = np.where(
                transitions[day - 1, :] == transition_id,
                parameters[(pool_id, season, variable, transition_name)][0],
                z_scores[(output_type, variable)][day + 1, :]
            )

            # Multiplicative terms
            i = 1
            for predictor in predictors[(variable, transition_name)]:
                # either (predictor, lag) tuples in self.predictors or parse here
                # e.g. something like predictor, lag = predictor_variable.split('_')
                if predictor.endswith('_lag1'):  # TODO: Can this parsing be replaced with a lookup?
                    predictor_variable = predictor.replace('_lag1', '')
                    lag = 1
                else:
                    predictor_variable = predictor
                    lag = 0
                if predictor_variable != 'na':
                    z_scores[(output_type, variable)][day + 1, :] += np.where(
                        transitions[day - 1, :] == transition_id,
                        parameters[(pool_id, season, variable, transition_name)][i]
                        * z_scores[(output_type, predictor_variable)][day + 1 - lag, :],
                        0.0
                    )
                i += 1

            # Scale residual/error term by standard error
            residuals *= np.where(
                transitions[day - 1, :] == transition_id,
                interpolated_parameters[('se', output_type, variable, season, transition_name)],
                1.0
            )

        # Add residual/error component
        z_scores[(output_type, variable)][day + 1, :] += residuals

        # Increment counter (index place) for standard normal sample
        ri += 1

    return z_scores[(output_type, variable)][2:, :], ri


# ---
# Unused functions


@numba.jit(nopython=True)
def predict(
        n_points, transitions, day, transition_key, z_scores, output_type, variable, parameters, pool_id, season,
        predictors, residuals, interpolated_parameters,
):
    for idx in range(n_points):
        transition_id = transitions[day - 1, idx]
        transition_name = transition_key[transition_id]

        # Intercept
        z_scores[(output_type, variable)][day + 1, idx] = parameters[(pool_id, season, variable, transition_name)][0]

        # Multiplicative terms
        i = 1
        for predictor in predictors[(variable, transition_name)]:
            # either (predictor, lag) tuples in self.predictors or parse here
            # e.g. something like predictor, lag = predictor_variable.split('_')
            if predictor.endswith('_lag1'):
                predictor_variable = predictor.replace('_lag1', '')
                lag = 1
            else:
                predictor_variable = predictor
                lag = 0
            if predictor_variable == 'na':
                pass
            else:
                z_scores[(output_type, variable)][day + 1, idx] += (
                        parameters[(pool_id, season, variable, transition_name)][i]
                        * z_scores[(output_type, predictor_variable)][day + 1 - lag, idx]
                )
            i += 1

        # Error/noise/random term
        z_scores[(output_type, variable)][day + 1, idx] += (
            residuals * interpolated_parameters[('se', output_type, variable, season, transition_name)][idx]
        )

    return z_scores[(output_type, variable)][day + 1, :]


@numba.jit(nopython=True)
def destandardise_1(
        n_points, transitions, day, transition_key, transformed_statistics_dict, pool_id, variable, season, z_scores,
):
    values = np.zeros(z_scores.shape[0], dtype=numba.float64)
    for idx in range(n_points):
        transition_id = transitions[day - 1, idx]
        transition_name = transition_key[transition_id]

        mean_1 = transformed_statistics_dict[(pool_id, variable, season, transition_name)][0]
        sd_1 = transformed_statistics_dict[(pool_id, variable, season, transition_name)][1]
        values[idx] = z_scores[idx] * sd_1 + mean_1

    return values
