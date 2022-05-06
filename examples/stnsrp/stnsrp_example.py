import os
import sys

import numpy as np
import pandas as pd

from rwgen.rainfall import stnsrp

# ---



# ---
# dc = {
#     'month': [1, 2],
#     'lamda': [0.01, 0.02],
#     'beta': [0.1, 0.2],
#     'eta': [1.0, 2.0],
#     'gamma': [1.0, 2.0],
#     'rho': [1.0, 2.0],
#     'xi': [1.0, 2.0]
# }
# parameters = pd.DataFrame(dc)
# simulation_length = 1000
# month_lengths = np.zeros(simulation_length * 2) + 24 * 31
# xmin = 0
# xmax = 10
# ymin = 0
# ymax = 10
# xrange = xmax - xmin
# yrange = ymax - ymin
# area = xrange * yrange
# p = stnsrp.Process(parameters, simulation_length, month_lengths, xmin, xmax, ymin, ymax, xrange, yrange, area)
# p.simulate()
# print(p.df)

# ---
# from rwgen.rainfall import nsrp
# dc = {
#     'month': [1],
#     'lamda': [0.01],
#     'beta': [0.1],
#     'eta': [1.0],
#     'nu': [1.0],
#     'xi': [1.0]
# }
# parameters = pd.DataFrame(dc)
# simulation_length = 1000
# month_lengths = np.zeros(simulation_length) + 24 * 31
# p = nsrp.Process(parameters, simulation_length, month_lengths)
# p.simulate()
# print(p.df)
# print(p.df.columns)

# ---
# from rwgen.rainfall import base
#
# dc = {'month': range(1, 12+1), 'lamda': [0.01,0.02,0.03,0.04,0.045,0.05,0.045,0.035,0.025,0.015,0.01,0.05]}
# parameters = pd.DataFrame(dc)
# simulation_length = 1000
# month_lengths = np.zeros(simulation_length * 12) + 24 * 30  # equal length months for initial testing
# p = base.Process(parameters, simulation_length, month_lengths)
# p.simulate_storms()

# ---
# gamma = 1.0
# xrange = 10.0
# yrange = 10.0
# stnsrp.Process.construct_outer_raincells_inverse_cdf(gamma, xrange, yrange)

# ---
# d = np.ones(2000)
# xmin = 0
# xmax = 10
# ymin = 0
# ymax = 10
# xrange = xmax - xmin
# yrange = ymax - ymin
# rng = np.random.default_rng()
# x, y = stnsrp.Process.sample_outer_locations(d, xrange, yrange, xmin, xmax, ymin, ymax, rng)
# with open('C:/Temp/spp_test3.csv', 'w') as fh:
#     for idx in range(x.shape[0]):
#         line = [x[idx], y[idx]]
#         line = ','.join(str(item) for item in line)
#         fh.write(line + '\n')
#
# sys.exit()
# ---

##os.chdir('H:/Projects/rwgen/examples/stnsrp')

if __name__ == '__main__':

    model = stnsrp.Model()

    # model.preprocess(
    #     # season_definitions=['ONDJFM', 'AMJJAS'],
    #     timeseries_format='csv',
    #     metadata_path='./input/metadata.csv',
    #     timeseries_folder='./input',
    #     output_folder='./output',
    #     completeness_threshold=0.0,
    #     outlier_method='trim'
    # )

    # sys.exit()

    # # print(model.preprocessor.statistics)
    # # print(model.preprocessor.phi)
    #
    # # model.preprocessor.statistics.to_csv('C:/Temp/rwgen_test1.csv')
    #
    # # sys.exit()

    # model.fit(
    #     output_folder='./output'
    # )
    #
    # sys.exit()

    # model.simulate(
    #     output_types=['point', 'catchment'],  # , 'grid'
    #     output_folder='./output2',
    #     output_format='txt',
    #     parameters='./output/parameters.csv',
    #     # output_prefix={'point': 's', 'catchment': 'c', 'grid': 'g'},
    #     points='./input/metadata.csv',
    #     catchments='./input/catchments.shp',
    #     catchment_id_field='ID',
    #     # grid=dict(
    #     #     ncols=75,
    #     #     nrows=81,
    #     #     xllcorner=632000,
    #     #     yllcorner=5518000,
    #     #     cellsize=1000
    #     # ),
    #     cell_size=1000.0,  # should be unused if grid is defined...
    #     dem='./input/srtm_dem.asc',  # None,  #
    #     phi='./output/phi.csv',
    #     number_of_years=500,
    #     number_of_realisations=20,
    #     equal_length_output=True
    # )
    #
    # sys.exit()

    # Testing spatial Poisson process
    model.simulate(
        output_types=['point'],
        output_folder='./spp1',
        output_format='txt',
        parameters='./output/parameters.csv',
        points='./spp1/points.csv',
        cell_size=1000.0,  # ! used to define xmin etc - line 648 in base.py - so currently required !
        # dem='./input/srtm_dem.asc',
        phi='./spp1/phi.csv',
        number_of_years=500,
        number_of_realisations=20,
        equal_length_output=True
    )

    sys.exit()
