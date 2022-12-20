"""
Example of spatial rainfall and weather modelling based on part of the Thames basin.

"""
import rwgen

# Boilerplate line needed to use multiprocessing in fitting on Windows OS
if __name__ == '__main__':
    # Initialise weather generator
    wg = rwgen.WeatherGenerator(
        spatial_model=True,
        project_name='thames',
        output_folder='./output',
        latitude=51.6,
        longitude=-1.1,
        easting_min=375000,
        easting_max=545000,
        northing_min=118000,
        northing_max=270000,
    )

    # Initialise rainfall model component
    wg.initialise_rainfall_model(
        input_timeseries='./input/hourly_rainfall/time_series',
        point_metadata='./input/hourly_rainfall/gauge_metadata.csv',
        intensity_distribution='weibull',
    )

    # At this point we could carry out the preprocessing and fitting for the rainfall model using the (commented out)
    # lines below

    # # Calculate rainfall reference statistics from gauge time series
    # wg.rainfall_model.preprocess()
    #
    # # Fit rainfall model parameters using reference statistics
    # wg.rainfall_model.fit()

    # However, instead we will set reference statistics and parameters directly (i.e. using outputs from prior
    # preprocessing/fitting)
    wg.rainfall_model.set_statistics(
        reference_statistics='./input/reference_statistics.csv',
        fitted_statistics='./input/fitted_statistics.csv',
    )
    wg.rainfall_model.set_parameters(
        parameters='./input/parameters.csv',
    )

    # Initialise weather model component
    wg.initialise_weather_model(
        input_timeseries='./input/daily_weather/time_series',
        point_metadata='./input/daily_weather/weather_metadata.csv',
    )

    # Calculate weather reference statistics from station time series
    wg.weather_model.preprocess()

    # Fit weather model parameters using reference statistics
    wg.weather_model.fit()

    # Simulate one realisation of 30 years at an hourly timestep
    wg.simulate(
        output_types=['point', 'catchment'],
        catchment_metadata='./input/catchments/test_catchments.shp',
        epsg_code=27700,
        cell_size=1000.0,
        dem='./input/thames_dem.asc',
        simulation_length=30,
    )

    # Calculate/extract statistics from simulated rainfall time series (e.g. AMAX, DDF)
    wg.rainfall_model.postprocess(
        amax_durations=[1, 6, 24],  # durations in hours
        ddf_return_periods=[2, 5, 10],  # return periods in years
    )
