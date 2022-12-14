import os
import sys

from . import preprocessing
from . import simulation
from ..rainfall import simulation as rainfall_simulation


class WeatherModel:

    def __init__(
            self,
            spatial_model,
            input_timeseries,
            output_folder,
            latitude,  # decimal degrees (negative for southern hemisphere)
            longitude,  # decimal degrees east of greenwich (negative for western hemisphere)
            predictors='default',
            input_variables='default',
            output_variables='default',
            season_length='month',  # 'month' or 'half-month'
            wet_threshold=0.2,
            # timestep=1,
            # random_seed=None,
            dem=None,
            residual_method='default',  # default is 'geostatistical' for spatial, other option is 'uniform'
            wind_height=10.0,
            xmin=None,
            xmax=None,
            ymin=None,
            ymax=None,
            point_metadata=None,
    ):
        print('Weather model initialisation')

        self.spatial_model = spatial_model
        self.input_timeseries = input_timeseries
        self.output_folder = output_folder
        self.latitude = latitude
        self.longitude = longitude

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if predictors == 'default':
            self.predictors = {
                ('temp_avg', 'DDD'): ['temp_avg_lag1'],
                ('temp_avg', 'DD'): ['temp_avg_lag1'],
                ('temp_avg', 'DW'): ['temp_avg_lag1', 'prcp'],
                ('temp_avg', 'WD'): ['temp_avg_lag1', 'prcp_lag1'],
                ('temp_avg', 'WW'): ['temp_avg_lag1'],
                ('dtr', 'DDD'): ['dtr_lag1'],
                ('dtr', 'DD'): ['dtr_lag1'],
                ('dtr', 'DW'): ['dtr_lag1', 'prcp'],
                ('dtr', 'WD'): ['dtr_lag1', 'prcp_lag1'],
                ('dtr', 'WW'): ['dtr_lag1'],
                ('vap_press', 'DDD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
                ('vap_press', 'DD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
                ('vap_press', 'DW'): ['vap_press_lag1', 'prcp', 'temp_avg', 'dtr'],
                ('vap_press', 'WD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
                ('vap_press', 'WW'): ['vap_press_lag1', 'prcp', 'temp_avg', 'dtr'],
                ('wind_speed', 'DDD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
                ('wind_speed', 'DD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
                ('wind_speed', 'DW'): ['wind_speed_lag1', 'prcp', 'temp_avg', 'dtr'],
                ('wind_speed', 'WD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
                ('wind_speed', 'WW'): ['wind_speed_lag1', 'prcp', 'temp_avg', 'dtr'],
                ('sun_dur', 'DDD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
                ('sun_dur', 'DD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
                ('sun_dur', 'DW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr'],
                ('sun_dur', 'WD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
                ('sun_dur', 'WW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr'],
            }
        else:
            self.predictors = predictors

        if input_variables == 'default':
            self.input_variables = ['temp_avg', 'dtr', 'vap_press', 'wind_speed', 'sun_dur']
        else:
            self.input_variables = input_variables

        self.season_length = season_length
        self.wet_threshold = wet_threshold
        # self.timestep = timestep

        if output_variables == 'default':
            self.output_variables = ['tas', 'pet']  # arguably defaults could depend on timestep
        else:
            self.output_variables = output_variables
        # 24hr - pet, tas, tasmin, tasmax, dtr, sundur, vap, ws10
        # <24hr - pet, tas, sundur, vap, ws10

        self.random_seed = None  # random_seed

        self.dem = dem

        if residual_method == 'default':
            if self.spatial_model:
                self.residual_method = 'geostatistical'
            else:
                self.residual_method = 'uniform'
        else:
            if self.spatial_model:
                self.residual_method = residual_method
            else:
                self.residual_method = 'uniform'

        self.wind_height = wind_height

        # ---

        self.offset = 10  # add to standardised variables before transformation

        self.xmin = xmin  # only used in preprocessing - simulation should be based on what rainfall model uses
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.point_metadata = point_metadata

        self.preprocessor = None
        self.simulator = None

        self.output_paths = None

        print('  - Completed')

    def preprocess(
            self,
            climatology_grids=None,
            spatial_method='pool',
            max_buffer=150.0,  # km
            min_points=5,
            use_neighbours=True,  # use neighbouring precipitation record
            neighbour_radius=20.0,  # km - but easting/northing in m so convert at relevant point
            calculation_period=(1991, 2020),
            completeness_threshold=33.0,
    ):
        print('Weather model preprocessing')

        self.preprocessor = preprocessing.Preprocessor(
            spatial_model=self.spatial_model,
            input_timeseries=self.input_timeseries,  # infer input variables | file path or folder of files
            point_metadata=self.point_metadata,  # optional if single site | rename as point_metadata in line with rainfall model?
            climatology_grids=climatology_grids,  # dict of file paths (or opened files)
            output_folder=self.output_folder,
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.ymin,
            ymax=self.ymax,
            spatial_method=spatial_method,  # use one station, pool multiple stations or interpolate multiple stations
            max_buffer=max_buffer,  # km - but easting/northing in m so convert - could be set to zero
            # min_years,  # minimum number of years data in pooled series (e.g. 30 * 365.25 - no missing)
            min_points=min_points,  # minimum number of stations if using interpolation or pooling
            wet_threshold=self.wet_threshold,
            use_neighbours=use_neighbours,  # use neighbouring precipitation record to try to infill wet days
            neighbour_radius=neighbour_radius,  # km - but easting/northing in m so convert
            calculation_period=calculation_period,
            completeness_threshold=completeness_threshold,
            predictors=self.predictors,
            input_variables=self.input_variables,  # list of variables to work with sunshine duration vs incoming SW...
            season_length=self.season_length,  # 'month' or 'half-month'
            offset=self.offset,
        )
        self.preprocessor.preprocess()

        # print('--')
        # df = self.preprocessor.raw_statistics
        # print(df.loc[df['variable'] == 'temp_avg'])
        # print()
        # # print(self.preprocessor.r2[(1, 1, 'temp_avg', 'DDD')])
        # # print(self.preprocessor.r2[(1, 1, 'temp_avg', 'DD')])
        # # print(self.preprocessor.r2[(1, 1, 'temp_avg', 'DW')])
        # # print(self.preprocessor.r2[(1, 1, 'temp_avg', 'WD')])
        # # print(self.preprocessor.r2[(1, 1, 'temp_avg', 'WW')])
        # print(self.preprocessor.r2)
        # sys.exit()

        print('  - Completed')

    def fit(self):
        print('Weather model fitting')
        self.preprocessor.fit()
        print('  - Completed')

    # def set_outputs(
    #         self,
    #         precipitation_paths,  # to "hack"
    # ):
    #     pass

    def simulate(
            self,
            rainfall,  # array or list of arrays
            n_timesteps,
            year,
            month,
            discretisation_metadata,
            output_types,
            timestep,  # hours
    ):
        if self.simulator is None:
            self.simulator = simulation.Simulator(
                spatial_model=self.spatial_model,
                wet_threshold=self.wet_threshold,
                predictors=self.predictors,
                input_variables=self.input_variables,
                output_variables=self.output_variables,
                timestep=timestep,
                season_length=self.season_length,

                raw_statistics=self.preprocessor.raw_statistics,  # transferring attributes from preprocessor
                transformed_statistics=self.preprocessor.transformed_statistics,
                transformations=self.preprocessor.transformations,
                # regressions=self.preprocessor.regressions,
                parameters=self.preprocessor.parameters,
                r2=self.preprocessor.r2,
                standard_errors=self.preprocessor.standard_errors,  # new
                statistics_variograms=self.preprocessor.statistics_variograms,
                residuals_variograms=self.preprocessor.residuals_variograms,  # covariance model parameters
                r2_variograms=self.preprocessor.r2_variograms,
                se_variograms=self.preprocessor.se_variograms,  # new
                noise_models=self.preprocessor.noise_models,

                discretisation_metadata=discretisation_metadata,
                output_types=output_types,

                random_seed=self.random_seed,

                dem=self.dem,

                residual_method=self.residual_method,

                wind_height=self.wind_height,

                latitude=self.latitude,
                longitude=self.longitude,

                bc_offset=self.offset
            )

        self.simulator.simulate(
            rainfall=rainfall,
            n_timesteps=n_timesteps,
            year=year,
            month=month,
        )

    def set_output_paths(
            self, spatial_model, output_types, output_format, output_subfolders, point_metadata,  # output_folder,
            catchment_metadata, realisation_ids, project_name
    ):
        self.output_paths = rainfall_simulation.make_output_paths(
            spatial_model,
            output_types,  # !! need to distinguish discretisation types
            output_format,
            # output_folder,
            self.output_folder,
            output_subfolders,
            point_metadata,
            catchment_metadata,
            realisation_ids,
            project_name,
            variables=self.output_variables
        )
