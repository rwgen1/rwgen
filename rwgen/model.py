import os
import sys

import numpy as np

from .rainfall.model import RainfallModel
from .weather.model import WeatherModel


class WeatherGenerator:

    def __init__(
            self,
            spatial_model,
            project_name,
            output_folder,
            latitude,
            longitude,
            easting=None,
            northing=None,
            elevation=None,
            easting_min=None,  # domain extent could ultimately be inferred - perhaps from rainfall model...? / dem
            easting_max=None,
            northing_min=None,
            northing_max=None,
    ):
        """
        Weather generator to simulate rainfall, temperature, potential evapotranspiration and other weather variables.

        More explanation and an example.

        Notes:
            The ``easting``, ``northing`` and ``elevation`` arguments are only needed for a point (gauge/site) model.

            The ``easting_min``, ``easting_max``, ``northing_min`` and ``northing_max`` arguments apply only to above
            spatial model.

        Args:
            spatial_model (bool): Flag to indicate whether point or spatial model.
            project_name (str): A name for the gauge/site location, domain or catchment.
            output_folder (str): Root folder for model output.
            latitude (int or float): Latitude in decimal degrees (negative for southern hemisphere).
            longitude (int or float): Longitude in decimal degrees (negative for western hemisphere).
            easting (int or float): Easting (in metres) of gauge/site for a point model.
            northing (int or float): Northing (in metres) of gauge/site for a point model.
            elevation (int or float): Elevation (in metres above sea level) of gauge/site for a point model.
            easting_min (int or float): Minimum easting (in metres) of domain bounding box for spatial model.
            easting_max (int or float): Maximum easting (in metres) of domain bounding box for spatial model.
            northing_min (int or float): Minimum northing (in metres) of domain bounding box for spatial model.
            northing_max (int or float): Maximum northing (in metres) of domain bounding box for spatial model.

        """
        print('Weather generator initialisation')

        self.spatial_model = spatial_model
        self.project_name = project_name  # ? droppable ?
        self.output_folder = output_folder
        self.latitude = latitude
        self.longitude = longitude
        self.easting = easting
        self.northing = northing
        self.elevation = elevation

        self.easting_min = easting_min
        self.easting_max = easting_max
        self.northing_min = northing_min
        self.northing_max = northing_max

        #: rwgen.RainfallModel: Instance of NSRP rainfall model
        self.rainfall_model = None

        #: rwgen.WeatherModel: Instance of regression-based weather model
        self.weather_model = None

        print('  - Completed')

    def initialise_rainfall_model(
            self,
            input_timeseries=None,
            point_metadata=None,
            season_definitions='monthly',  # currently need to match defaults in RainfallModel.__init__()
            intensity_distribution='exponential',
            statistic_definitions=None,
    ):
        """
        Initialise rainfall model.

        Link to rwgen.RainfallModel docs for more details.

        Args:
            input_timeseries (str): Path to file containing timeseries data (for point model) or folder containing
                timeseries data files (for spatial model). Needed if running pre-processing or fitting steps.
            point_metadata (pandas.DataFrame or str): Metadata (or path to metadata file) on point (site/gauge)
                locations to use for fitting, simulation and/or evaluation for a spatial model only. See Notes for
                details.
            season_definitions (str, list or dict): The model works on a monthly basis by default, but this argument
                allows for user-defined seasons (see Notes).
            intensity_distribution (str): Flag to indicate the type of probability distribution for raincell
                intensities. Defaults to ``'exponential'`` (with ``'weibull'`` also available currently).
            statistic_definitions (pandas.DataFrame or str): Definitions (descriptions) of statistics to use in fitting
                and/or evaluation (or path to file of definitions). See Notes for explanation of format.

        """
        self.rainfall_model = RainfallModel(
            spatial_model=self.spatial_model,
            project_name=self.project_name,
            input_timeseries=input_timeseries,
            point_metadata=point_metadata,
            season_definitions=season_definitions,
            intensity_distribution=intensity_distribution,
            output_folder=os.path.join(self.output_folder, 'rainfall_model'),
            statistic_definitions=statistic_definitions,
            easting=self.easting,
            northing=self.northing,
            elevation=self.elevation,
        )

    def initialise_weather_model(
            self,
            input_timeseries,
            point_metadata=None,
            output_variables='default',
            season_length='month',
            wet_threshold=0.2,
            wind_height=10.0,
    ):
        """
        Initialise weather model.

        Link to rwgen.WeatherModel docs for more details.

        Args:
            input_timeseries (str): Path to file containing timeseries data (for point model) or folder containing
                timeseries data files (for spatial model). Currently compulsory.
            point_metadata (pandas.DataFrame or str): Metadata (or path to metadata file) on point (site/gauge)
                locations to use for fitting, simulation and/or evaluation for a spatial model only.
            output_variables (list or str): List of output variables. If 'default' then the list is ['pet', 'tas'].
            season_length (str): Default is to run the weather model on a monthly basis, with 'half-month' under
                development as an option.
            wet_threshold (float): Threshold used to identify wet days (in mm/day).
            wind_height (int or float): Measurement height for wind speed data (metres above ground).


        """
        # Things that could be potentially be arguments (but are not easily changed by the user currently)
        predictors = 'default'
        input_variables = 'default'
        dem = None  # currently no dem reading in weather model stuff, so could only be taken as xarray dataset (at the moment)
        residual_method = 'default'

        self.weather_model = WeatherModel(
            spatial_model=self.spatial_model,
            input_timeseries=input_timeseries,
            output_folder=os.path.join(self.output_folder, 'weather_model'),
            latitude=self.latitude,
            longitude=self.longitude,
            predictors=predictors,
            input_variables=input_variables,
            output_variables=output_variables,
            season_length=season_length,
            wet_threshold=wet_threshold,
            # timestep=timestep,
            # random_seed=random_seed,
            dem=dem,
            residual_method=residual_method,
            wind_height=wind_height,
            xmin=self.easting_min,
            xmax=self.easting_max,
            ymin=self.northing_min,
            ymax=self.northing_max,
            point_metadata=point_metadata,
        )

    def simulate(  # matching rainfall model args/defaults
            self,
            output_types='point',
            output_subfolders='default',
            output_format='txt',
            catchment_metadata=None,
            grid_metadata=None,
            epsg_code=None,
            cell_size=None,
            dem=None,
            simulation_length=30,
            n_realisations=1,
            timestep_length=1,
            start_year=2000,
            calendar='gregorian',
            random_seed=None,  # pass on initialisation? one seed for weather generator?
            run_simulation=True,
            apply_shuffling=False,
    ):
        """
        Simulate rainfall and other weather (temperature, PET, ...) variables.

        Link to rgwen.RainfallModel.simulate() for more details.

        Args:
            output_types (str or list of str): Types of output (discretised) rainfall required. Options are ``'point'``,
                ``'catchment'`` and ``'grid'``.
            output_subfolders (str or dict): Sub-folder in which to place each output type. If ``'default'`` then
                ``dict(point='point', catchment='catchment', grid='grid')`` is used for a spatial model and
                ``dict(point='')`` for a point model (i.e. output to ``self.output_folder``). If None then all output
                files are written to ``self.output_folder``.
            output_format (str): Flag indicating output file format for point and catchment output. Current
                option is ``txt``. Gridded output will be written in NetCDF format.
            catchment_metadata (geopandas.GeoDataFrame or str): Geodataframe containing catchments for which output is
                required (or path to catchments shapefile). Optional.
            grid_metadata (dict or str): Specification of output grid to use for both gridded output (optional). This
                grid is also used to support catchment output. Dictionary keys use ascii raster header keywords, e.g.
                ``dict(ncols=10, nrow=10, ...)``. Use ``xllcorner`` and ``yllcorner``, as well as lowercase for each
                keyword. If None then a grid is defined to encompass catchment locations using the ``cell_size``
                argument. The path to an ascii raster file to use as a template for the grid can be given instead.
            epsg_code (int): EPSG code for projected coordinate system used for domain (required if catchment or
                grid output is requested).
            cell_size (float): Cell size to use if grid is None but a grid is needed for gridded output and/or catchment
                output.
            dem (xarray.DataArray or str): Digital elevation model (DEM) [m] as data array or ascii raster file path.
                Optional but recommended.
            simulation_length (int): Number of years to simulate in one realisation (minimum of 1).
            n_realisations (int): Number of realisations to simulate.
            timestep_length (int): Timestep of output [hr]. Default is 1 (hour).
            start_year (int): Start year of simulation.
            calendar (str): Flag to indicate whether ``gregorian`` (default accounting for leap years) or ``365-day``
                calendar should be used.
            random_seed (int): Seed to use in random number generation.
            run_simulation (bool): Flag for whether to run simulation. Setting to False may be used to update
                ``self.simulation_args`` to allow ``self.postprocess()`` to be run without ``self.simulate()``
                having been run first (i.e. reading from existing simulation output files).
            apply_shuffling (bool): Indicates whether to run model with or without shuffling following Kim and Onof
                (2020) method.

        """
        if self.weather_model is not None:
            if self.weather_model.preprocessor is None:
                raise ValueError('Run preprocess and fit methods of weather_model before simulating.')

        if random_seed is not None:
            if self.weather_model.random_seed is None:
                rng = np.random.default_rng(random_seed)
                self.weather_model.random_seed = rng.integers(1000000, 100000000)

        self.rainfall_model.simulate(
            output_types=output_types,
            output_subfolders=output_subfolders,
            output_format=output_format,
            catchment_metadata=catchment_metadata,
            grid_metadata=grid_metadata,
            epsg_code=epsg_code,
            cell_size=cell_size,
            dem=dem,
            simulation_length=simulation_length,
            n_realisations=n_realisations,
            timestep_length=timestep_length,
            start_year=start_year,
            calendar=calendar,
            random_seed=random_seed,
            run_simulation=run_simulation,
            apply_shuffling=apply_shuffling,
            weather_model=self.weather_model
        )

# Baseline usage idea
# wg = WeatherGenerator()
# wg.initialise_rainfall_model()
# wg.rainfall_model.preprocess()
# wg.rainfall_model.fit()
# wg.initialise_weather_model()
# wg.weather_model.preprocess()
# wg.weather_model.fit()
# wg.simulate()
# wg.rainfall_model.postprocess()
# # wg.weather_model.postprocess()
# # wg.rainfall_model.plot(['reference', 'fitted'], ...)

# Climate change usage idea
# ... initialise ...
# wg.rainfall_model.set_statistics()
# wg.rainfall_model.perturb_statistics()
# wg.rainfall_model.fit()
# wg.weather_model.set_statistics()
# wg.weather_model.perturb_statistics()
# wg.weather_model.set_parameters()
# wg.simulate()
