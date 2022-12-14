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
            latitude,  # decimal degrees (negative for southern hemisphere)
            longitude,  # decimal degrees east of greenwich (negative for western hemisphere)
            easting=None,  # needed for CC perturbation but nowhere else in WG?
            northing=None,  # needed for CC perturbation but nowhere else in WG?
            elevation=None,  # needed to estimate atmospheric pressure in PET calculations
            # !! ELEVATION AS ARGUMENT HERE (MANDATORY)? - SCALAR FOR POINT AND DEM (PATH/XARRAY) FOR SPATIAL? !!
            # - needs to propagate into discretisation_metadata as 'z'
            # - rainfall model could check if a dem has been assigned to the weather generator
            easting_min=None,  # domain extent could ultimately be inferred - perhaps from rainfall model...?
            easting_max=None,
            northing_min=None,
            northing_max=None,
            # - if inferring extent from rainfall model then would need to simulation preparation earlier, as this is
            # where domain extent is figured out
    ):
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

        self.rainfall_model = None
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
            predictors='default',
            input_variables='default',
            output_variables='default',
            season_length='month',
            wet_threshold=0.2,
            # timestep=1,
            random_seed=None,  # passed on initialisation to weather model but simulation to rainfall model currently
            dem=None,
            residual_method='default',
            wind_height=10.0,
    ):
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
        # Weather model preprocessing needs to be run before simulation [could also check rainfall model attributes...]
        if self.weather_model is not None:
            if self.weather_model.preprocessor is None:
                raise ValueError('Run preprocess and fit methods of weather_model before simulating.')

        if random_seed is not None:
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
