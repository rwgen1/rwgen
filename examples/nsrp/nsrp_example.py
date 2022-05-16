"""
Example of point NSRP modelling based on the Brize Norton hourly gauge record.

"""
from rwgen.rainfall import model

# Boilerplate line needed to use multiprocessing in fitting on Windows OS
if __name__ == '__main__':

    # Initialise model object with defaults (single site + monthly + 1hr and 24hr statistics used in fitting)
    m = model.Model()

    # Calculate observed/reference statistics from gauge time series file
    m.preprocess(
        timeseries_path='./input/brize_norton.csv',
        output_folder='Z:/DP/Work/ER/rwgen/testing/examples/nsrp'  # './output'
    )

    import sys
    sys.exit()

    # Fit model and save parameters to file
    m.fit(
        output_folder='Z:/DP/Work/ER/rwgen/testing/examples/nsrp',  # './output'
        n_workers=6
    )

    import sys
    sys.exit()

    # Simulate three realisations of 100 years at an hourly timestep (the default) and concatenate the
    # resulting output into one file
    model1.simulate(
        output_folder='./output',
        output_filename_root='brize-norton_sim',
        number_of_years=100,
        number_of_realisations=3,
        concatenated_output_path='./output/brize-norton_sim_concat.csvy',
    )
