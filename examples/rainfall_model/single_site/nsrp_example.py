"""
Example of point NSRP modelling based on the Brize Norton hourly gauge record.

Script version of nsrp_example.ipynb notebook.

"""
import rwgen

# Boilerplate line needed to use multiprocessing in fitting on Windows OS
if __name__ == '__main__':

    # Initialise model
    rainfall_model = rwgen.RainfallModel(
        spatial_model=False,
        project_name='brize_norton',
        statistic_definitions='./input/statistic_definitions.csv'
    )

    # Calculate observed/reference statistics from gauge time series file
    rainfall_model.preprocess(
        input_timeseries='./input/brize_norton.csv',
    )

    # Fit model and save parameters to file
    rainfall_model.fit(
        n_workers=6  # for e.g. a 6-core computer
    )

    # Simulate three realisations of 1000 years with a 1hr timestep
    rainfall_model.simulate(
        simulation_length=1000,
        n_realisations=3,
        timestep_length=1
    )

    # Postprocessing
    rainfall_model.postprocess(
        amax_durations=[1, 6, 24],  # durations in hours
        ddf_return_periods=[20, 50, 100],  # return periods in years
    )
    
    # Plotting
    rainfall_model.plot()
