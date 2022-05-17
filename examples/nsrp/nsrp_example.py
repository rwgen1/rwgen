"""
Example of point NSRP modelling based on the Brize Norton hourly gauge record.

"""
from rwgen.rainfall import model

# Boilerplate line needed to use multiprocessing in fitting on Windows OS
if __name__ == '__main__':

    # Initialise model object with defaults (single site + monthly + 1hr and 24hr statistics used in fitting)
    m = model.Model()

    # # Calculate observed/reference statistics from gauge time series file
    # m.preprocess(
    #     timeseries_path='./input/brize_norton.csv',
    #     output_folder='Z:/DP/Work/ER/rwgen/testing/examples/nsrp'  # './output'
    # )
    #
    # # Fit model and save parameters to file
    # m.fit(
    #     output_folder='Z:/DP/Work/ER/rwgen/testing/examples/nsrp',  # './output'
    #     n_workers=6
    # )

    # Simulate five realisations of 1000 years at an hourly timestep (the default)
    m.simulate(
        output_folder='Z:/DP/Work/ER/rwgen/testing/examples/nsrp',  # './output'
        parameters='Z:/DP/Work/ER/rwgen/testing/examples/nsrp/parameters.csv',
        simulation_length=1000,
        number_of_realisations=5,
    )
