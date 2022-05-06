"""Examples of point NSRP modelling based on the Brize Norton hourly gauge record.

"""

import sys

from rwgen.rainfall import nsrp

# ------------------------------------------------------------------------------------------------
# Example 1: Pre-processing, fitting and simulations

if __name__ == '__main__':
    print('Example 1')

    # Initialise model object with defaults (monthly setup using 1hr and 24hr statistics in fitting)
    model1 = nsrp.Model()

    # # Calculate observed/reference statistics from gauge time series file
    # model1.preprocess(
    #     # season_definitions=['ONDJFM', 'AMJJAS'],
    #     timeseries_path='./input/brize-norton.csv',
    #     timeseries_format='csv',
    #     # output_statistics_path='./output/brize-norton_statistics1.csv'
    #     output_folder='./output2'
    # )
    #
    # # Fit model and save parameters to file
    # model1.fit(
    #     output_folder='./output2'
    #     # parameters_output_path='./output/parameters1.csv'
    # )
    #
    # sys.exit()

    # ---

    model1.simulate(
        output_folder='./output3',
        output_format='txt',
        parameters='./output/parameters.csv',
        output_prefix='brize-norton',
        number_of_years=500,
        number_of_realisations=20,
        equal_length_output=True
    )

    # ---

    # # Simulate three realisations of 100 years at an hourly timestep (the default) and concatenate the
    # # resulting output into one file
    # model1.simulate(
    #     output_folder='./output',
    #     output_filename_root='brize-norton_sim',
    #     number_of_years=100,
    #     number_of_realisations=3,
    #     concatenated_output_path='./output/brize-norton_sim_concat.csvy',
    # )

    sys.exit()

# # ------------------------------------------------------------------------------------------------
# # Example 2: User-specified season definitions and statistic definitions/weights for fitting
#
# print()
# print('Example 2')
#
# # Initialise model object with user-specified seasons and statistics/weights
# model2 = nsrp.Model()
#
# # Calculate observed/reference statistics from gauge time series file
# model2.preprocess(
#     statistic_definitions_path='./input/statistic_definitions.csv',
#     # season_definitions={
#     #     # month: season identifier
#     #     12: 1,
#     #     1: 1,
#     #     2: 1,
#     #     3: 2,
#     #     4: 2,
#     #     5: 2,
#     #     6: 3,
#     #     7: 3,
#     #     8: 3,
#     #     9: 4,
#     #     10: 4,
#     #     11: 4
#     # },
#     timeseries_path='./input/brize-norton.csv',
#     timeseries_format='csv',
#     output_folder='./output'
# )
#
# # print(model2.preprocessor.statistics)
#
# # Then could carry on with fitting and simulation ...


