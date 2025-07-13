import datetime
import itertools

import numpy as np
import scipy.special
import scipy.interpolate
import numba
import gstools

from ..rainfall import simulation as rainfall_simulation
from . import fao56
from ..rainfall import utils


class Simulator:

    def __init__(
            self,
            spatial_model,
            wet_threshold,
            predictors,
            input_variables,  # i.e. what needs to be simulated using autoregressive models
            output_variables,  # what to write out (including pet)
            timestep,  # timestep of rainfall input and weather output (weather simulated daily then disaggregated)
            season_length,
            raw_statistics,
            transformed_statistics,
            transformations,
            # regressions,
            parameters,
            r2,
            standard_errors,
            statistics_variograms,
            residuals_variograms,
            r2_variograms,
            se_variograms,
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
        self.parameters = parameters
        self.r2 = r2
        self.standard_errors = standard_errors
        self.statistics_variograms = statistics_variograms
        self.residuals_variograms = residuals_variograms
        self.r2_variograms = r2_variograms
        self.se_variograms = se_variograms
        self.noise_models = noise_models

        self.discretisation_metadata = discretisation_metadata

        self.final_output_types = output_types  # actual output types ['point', 'catchment', 'grid']
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
        if longitude == 0.0:
            self.longitude = longitude
        elif (longitude > 0.0) and (longitude <= 180.0):
            self.longitude = 360.0 - longitude
        elif (longitude < 0.0) and (longitude >= -180.0):
            self.longitude = longitude * -1

        self.bc_offset = bc_offset  # used in box-cox back-transformations

        self.covariance_models = None
        self.field_generators = None
        if self.spatial_model:
            self.covariance_models = {}  # for residuals
            self.field_generators = {}  # for residuals

            for season, variable in itertools.product(self.seasons, self.input_variables):
                if len(self.noise_models.keys()) > 0:
                    self.covariance_models[(season, variable)] = gstools.Exponential(
                        dim=2,
                        var=self.noise_models[(season, variable)][0],
                        len_scale=self.noise_models[(season, variable)][1]
                    )
                    self.field_generators[(season, variable)] = gstools.SRF(self.covariance_models[(season, variable)])

        self.simulation_variables = self.input_variables.copy()
        self.simulation_variables.append('prcp')

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
                self.lag_z_scores[(output_type, variable)] = np.zeros((2, self.n_points[output_type]))
                self.lag_values[(output_type, variable)] = np.zeros((2, self.n_points[output_type]))
                self.z_scores[(output_type, variable)] = np.zeros((33, self.n_points[output_type]))
                self.values[(output_type, variable)] = np.zeros((33, self.n_points[output_type]))

        # Rather than e.g. pred_ddd, can loop transitions and put in e.g. dict of arrays
        self.transition_key = {
            1: 'DDD',
            2: 'DD',
            3: 'DW',
            4: 'WD',
            5: 'WW',
        }

        self.interpolated_parameters = {}
        if self.spatial_model:
            self.interpolate_parameters()
        else:
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
                        # TODO: Check why not storing r2 per point

        self.output_paths = None
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

        self.transformed_statistics_dict = {}
        for _, row in self.transformed_statistics.iterrows():
            self.transformed_statistics_dict[(row['pool_id'], row['variable'], row['season'])] = [
                row['bc_mean'], row['bc_std']
            ]

        # Create interpolator function for inverse cdf of beta distribution
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

    def simulate(
            self,
            rainfall,
            n_timesteps,  # in month - expressed in rainfall timestep (e.g. not necessarily days)
            year,
            month,
    ):
        # transitions:
        #   1 - DDD
        #   2 - DD
        #   3 - DW
        #   4 - WD
        #   5 - WW

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

        for k, v in self.transition_key.items():
            self.transition_key_nb[k] = v

        for k, v in self.parameters.items():
            self.parameters_nb[k] = v

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

        for k, v in self.interpolated_parameters.items():
            if isinstance(v, float):
                v_ = np.asarray([v])
            else:
                v_ = v
            self.interpolated_parameters_nb[k] = v_

        # Number of days in month / half-month
        n_days = int(n_timesteps / (24 / self.timestep))

        # Random residual/error component
        # - series of numbers per variable for point model
        # - series of fields per variable for spatial model
        # - if generate all upfront could be more efficient perhaps

        _n = n_days * len(self.input_variables) * len(self.output_types)
        sn_sample = self.rng.standard_normal(_n)  # standard normal sample
        ri = 0  # counter for residual - for indexing sn_sample (increment after each day+variable combination)

        # Everything needs to be stratified by output type
        for output_type in self.output_types:

            # Ensure arrays of values are reset to zero
            for variable in self.simulation_variables:
                self.z_scores[(output_type, variable)].fill(0.0)
                self.values[(output_type, variable)].fill(0.0)
            if 'pet' in self.output_variables:
                self.values[(output_type, 'pet')].fill(0.0)

            # Construct arrays with space for first two lags at beginning (requiring lags from previous months)
            for variable in self.simulation_variables:
                self.z_scores[(output_type, variable)][0, :] = self.lag_z_scores[(output_type, variable)][0, :]
                self.z_scores[(output_type, variable)][1, :] = self.lag_z_scores[(output_type, variable)][1, :]
                self.values[(output_type, variable)][0, :] = self.lag_values[(output_type, variable)][0, :]
                self.values[(output_type, variable)][1, :] = self.lag_values[(output_type, variable)][1, :]

            # Aggregate input rainfall (current month) to daily timestep
            if self.timestep != 24:
                self.values[(output_type, 'prcp')][2:,:] = aggregate_rainfall(
                    rainfall[output_type], self.n_points[output_type], int(24 / self.timestep),
                )
            else:
                self.values[(output_type, 'prcp')][2:,:] = rainfall[output_type][:]

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

            # Standardise rainfall - if half-months then each half needs to be standardised separately
            # - requires interpolation of mean and sd to each output point to have been completed first
            # - assuming rainfall is not transformed, only standardised
            # - remember that number of points needs to be considered (i.e. standardisation by point)
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
                rainfall_sa[:2] = rainfall_sa2[:2]
                rainfall_sa[2:2+15] = rainfall_sa1[2:2+15]
                rainfall_sa[15:] = rainfall_sa2[15:]

            # now switching to z-scores arrays
            self.z_scores[(output_type, 'prcp')][:] = rainfall_sa[:]

            # TODO: Remove hardcoding of pool_id and inflexibility on having only one pool
            pool_id = 1

            for k, v in self.z_scores.items():
                self.z_scores_nb[k] = v
            residuals_dummy = np.zeros(self.n_points[output_type])
            for variable in self.input_variables:
                self.z_scores[(output_type, variable)][2:, :], ri = regressions(
                    n_days, self.season_length, month, variable, sn_sample, ri, self.transition_key_nb,
                    self.z_scores_nb, output_type, transitions, self.parameters_nb, pool_id, self.predictors_nb,
                    self.interpolated_parameters_nb, residuals_dummy,
                )

            for variable in self.input_variables:
                mean_1 = self.transformed_statistics_dict[(pool_id, variable, season)][0]
                sd_1 = self.transformed_statistics_dict[(pool_id, variable, season)][1]
                self.values[(output_type, variable)][2:, :] = (
                    self.z_scores[(output_type, variable)][2:, :] * sd_1 + mean_1
                )

            # Reverse transformation
            for variable in self.input_variables:
                if variable in ['temp_avg', 'dtr', 'vap_press', 'wind_speed']:
                    self.values[(output_type, variable)][2:, :] = scipy.special.inv_boxcox(
                        self.values[(output_type, variable)][2:, :],
                        self.transformations[(pool_id, variable, season, 'lamda')]
                    )
                    self.values[(output_type, variable)][2:, :] -= self.bc_offset
                elif variable == 'sun_dur':
                    p0 = self.transformations[(pool_id, variable, season, 'p0')]
                    zero_threshold = scipy.stats.norm.ppf(p0)
                    p = scipy.stats.norm.cdf(self.values[(output_type, variable)][2:, :])
                    p = (p - p0) / (1.0 - p0)

                    # interpolator for beta inverse cdf (faster and same to 5+ decimal places)
                    f = self.sundur_beta_ppf_funcs[(pool_id, season)]
                    self.values[(output_type, variable)][2:, :] = f(p)

                    self.values[(output_type, variable)][2:, :] = np.where(
                        (self.values[(output_type, variable)][2:, :] <= zero_threshold)
                        | (~np.isfinite(self.values[(output_type, variable)][2:, :])),
                        0.0,
                        self.values[(output_type, variable)][2:, :]
                    )

            for variable in self.input_variables:
                if variable != 'sun_dur':
                    mean_2 = self.interpolated_parameters[('raw_statistics', output_type, variable, season, 'mean')]
                    sd_2 = self.interpolated_parameters[('raw_statistics', output_type, variable, season, 'std')]
                    self.values[(output_type, variable)][2:, :] = (
                            self.values[(output_type, variable)][2:, :] * sd_2 + mean_2
                    )
                else:
                    min_ = self.transformations[(pool_id, variable, season, 'obs_min')]
                    max_ = self.transformations[(pool_id, variable, season, 'obs_max')]
                    self.values[(output_type, variable)][2:, :] = (
                            self.values[(output_type, variable)][2:, :] * (max_ - min_) + min_
                    )

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

            self.lag_z_scores[(output_type, variable)][0, :] = self.z_scores[(output_type, variable)][n_days+1,:]
            self.lag_z_scores[(output_type, variable)][1, :] = self.z_scores[(output_type, variable)][n_days, :]
            self.lag_values[(output_type, variable)][0, :] = self.values[(output_type, variable)][n_days+1, :]
            self.lag_values[(output_type, variable)][1, :] = self.values[(output_type, variable)][n_days, :]

            # PET calculations
            if 'pet' in self.output_variables:
                self.calculate_pet(year, month)

            # Disaggregate
            self.disaggregate(n_days, season)

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
            pres = fao56.atmos_pressure(self.discretisation_metadata[(output_type, 'z')])

            # Actual vapour pressure is already in correct units [kPa]
            avp = self.values[(output_type, 'vap_press')][2:2+n_days, :]

            # nan occurred once possibly related to avp**0.5 - so try to ensure positive at least
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
            lat = np.zeros((doy.shape[0], self.n_points[output_type]))
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
        # - other variables just uniform disaggregation for now
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
        )


def aggregate_rainfall(x, n_points, window_size):
    # assumes that array has dimensions (n_timesteps, n_points)
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
        residuals[:] = sn_sample[ri]

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
                if predictor.endswith('_lag1'):
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
