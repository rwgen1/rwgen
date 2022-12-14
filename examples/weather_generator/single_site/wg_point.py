"""
Example of point weather generator based on the MIDAS-Open data for Brize Norton.

Script version of wg_point.ipynb notebook.

"""
import sys

import pandas as pd

from rwgen import WeatherGenerator


if __name__ == '__main__':

    # Initialise weather generator
    wg = WeatherGenerator(
        spatial_model=False,
        project_name='vals-pet',  # 'brize-norton',
        output_folder='Z:/DP/Work/ER/rwgen/testing/weather',  # './output',
        latitude=51.758,
        longitude=-1.578,
    )

    # Initialise rainfall model component of weather generator
    wg.initialise_rainfall_model(
        point_metadata=pd.DataFrame({'Easting': 429124, 'Northing': 206725, 'Elevation': 82.0}, index=[0]),
        intensity_distribution='weibull',
    )

    # Now we could run the rainfall model preprocessing and fitting (using the commented out lines below)...
    # wg.rainfall_model.preprocess(
    #     input_timeseries='./input/brize-norton_hourly.csv',
    # )
    # wg.rainfall_model.fit(
    #     n_workers=6  # for e.g. a 6-core computer
    # )

    # But we will set reference statistics and parameters directly (i.e. using outputs from prior preprocessing/fitting)
    wg.rainfall_model.set_statistics(
        reference_statistics='./input/reference_statistics.csv',
        fitted_statistics='./input/fitted_statistics.csv',
    )
    wg.rainfall_model.set_parameters(
        parameters='./input/parameters.csv',
    )

    # Initialise weather (non-rainfall) model component of weather generator
    wg.initialise_weather_model(output_variables=['tas', 'pet'])

    # Read data, calculate statistics and perform data transformations
    wg.weather_model.preprocess(
        input_timeseries='./input/brize-norton_daily.csv',
    )

    # Fit regression parameters of weather model
    wg.weather_model.fit()

    # Simulate with the weather generator
    wg.simulate(
        simulation_length=100,
        timestep_length=24,  # TODO: Test daily and sub-daily disaggregation
    )

    # Postprocessing - !! implement disaggregation first? !!
    # wg.rainfall_model.postprocess(
    #     amax_durations=[1, 6, 24],  # durations in hours
    #     ddf_return_periods=[20, 50, 100],  # return periods in years
    # )
