"""
Example of point rainfall and weather modelling based on the Brize Norton hourly gauge record.

"""
import rwgen

# Boilerplate line needed to use multiprocessing in fitting on Windows OS
if __name__ == '__main__':
    # Initialise weather generator
    wg = rwgen.WeatherGenerator(
        spatial_model=False,
        project_name='brize-norton',
        output_folder='./output',
        latitude=51.758,
        longitude=-1.578,
        easting=429124,
        northing=206725,
        elevation=82.0,
    )

    # Initialise rainfall model component
    wg.initialise_rainfall_model(
        input_timeseries='./input/hourly_rainfall.csv',
    )

    # At this point we could carry out the preprocessing and fitting for the rainfall model using the (commented out)
    # lines below

    # # Calculate rainfall reference statistics from gauge time series
    # wg.rainfall_model.preprocess(dayfirst=True)
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
        input_timeseries='./input/daily_weather.csv',
    )

    # Calculate weather reference statistics from station time series
    wg.weather_model.preprocess()

    # Fit weather model parameters using reference statistics
    wg.weather_model.fit()

    # Simulate one realisation of 100 years at an hourly timestep
    wg.simulate(
        simulation_length=100,
        n_realisations=1,
        timestep_length=1,
    )

    # Calculate/extract statistics from simulated rainfall time series (e.g. AMAX, DDF)
    wg.rainfall_model.postprocess(
        amax_durations=[1, 6, 24],  # durations in hours
        ddf_return_periods=[2, 5, 10],  # return periods in years
    )
