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
            latitude,
            longitude,
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
        """
        Initialise weather model to simulate temperature, potential evapotranspiration and other weather variables.

        Args:
            spatial_model (bool): Flag to indicate whether point or spatial model.
            input_timeseries (str): Path to file containing timeseries data (for point model) or folder containing
                timeseries data files (for spatial model). Currently compulsory.
            output_folder (str): Root folder for model output.
            latitude (int or float): Latitude in decimal degrees (negative for southern hemisphere).
            longitude (int or float): Longitude in decimal degrees (negative for western hemisphere).
            predictors (dict or str): Predictors for each variable for each (wet/dry) transition state. Leave as
                'default' for now.
            input_variables (list of str): List of input variables for weather/PET simulation (corresponding with
                column names in input file(s)). Leave as 'default' for now, which is
                ``['temp_avg', 'dtr', 'vap_press', 'wind_speed', 'sun_dur']``.
            output_variables (list or str): List of output variables. If 'default' then the list is ['pet', 'tas'].
            season_length (str): Default is to run the weather model on a monthly basis, with 'half-month' under
                development as an option.
            wet_threshold (float): Threshold used to identify wet days (in mm/day).
            wind_height (int or float): Measurement height for wind speed data (metres above ground).
            dem (str): Only None acceptable currently.
            residual_method (str): Only 'default' available currently.
            xmin (int or float): Minimum easting (in metres) of domain bounding box for spatial model.
            xmax (int or float): Maximum easting (in metres) of domain bounding box for spatial model.
            ymin (int or float): Minimum northing (in metres) of domain bounding box for spatial model.
            ymax (int or float): Maximum northing (in metres) of domain bounding box for spatial model.
            point_metadata (pandas.DataFrame or str): Metadata (or path to metadata file) on point (site/gauge)
                locations to use for fitting, simulation and/or evaluation for a spatial model only.

        """
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
            max_buffer=150.0,  # km
            min_points=5,
            use_neighbours=True,  # use neighbouring precipitation record
            neighbour_radius=20.0,  # km - but easting/northing in m so convert at relevant point
            calculation_period=(1991, 2020),
            completeness_threshold=33.0,
    ):
        """
        Read and transform reference series in preparation for model fitting.

        Args:
            max_buffer (int or float): Maximum distance (km) to look away from domain edges for weather stations.
                Defaults to 150km.
            min_points (int): Minimum number of weather stations required for spatial model. Defaults to 5.
            use_neighbours (bool): Flag to use neighbouring precipitation record to infill wet/dry state series for above
                weather station (i.e. if precipitation data wholly or partly missing). Default is True.
            neighbour_radius (int or float): Maximum distance for using a neighbouring gauge to infill wet/dry state
                for a weather station (i.e. if precipitation data missing). Defaults to 20km.
            calculation_period (list or tuple): Start and end year for calculating reference statistics. Defaults to
                ``(1991, 2020)``.
            completeness_threshold (int or float): Percentage completeness threshold for using a reference weather
                series (default of 33%).

        """
        print('Weather model preprocessing')

        # Things that might become options/arguments
        climatology_grids = None
        spatial_method = 'pool'

        self.preprocessor = preprocessing.Preprocessor(
            spatial_model=self.spatial_model,
            input_timeseries=self.input_timeseries,  # infer input variables | file path or folder of files
            point_metadata=self.point_metadata,
            # optional if single site | rename as point_metadata in line with rainfall model?
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

        print('  - Completed')

    def fit(self):
        """
        Fit regression models for weather variables.

        """
        print('Weather model fitting')
        self.preprocessor.fit()
        print('  - Completed')

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
        """
        Simulate weather and potential evapotranspiration variables for one month.

        Args:
            rainfall (numpy.ndarray or list of numpy.ndarray): Rainfall/precipitation values in mm/timestep.
            n_timesteps (int): Number of timesteps in month (matching rainfall model timestep).
            year (int): Calendar year.
            month (int): Calendar month to simulate (1-12).
            discretisation_metadata (dict): Metadata of locations for which output is required. See RainfallModel.
            output_types (str or list of str): Types of output (discretised) rainfall required. Options are ``'point'``,
                ``'catchment'`` and ``'grid'``.
            timestep (int): Timestep for weather model output in hours.

        """
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
