"""
Example of spatial NSRP modelling based on a catchment in the eastern Rhine basin.

"""
from rwgen.rainfall import model

# Boilerplate line needed to use multiprocessing in fitting on Windows OS
if __name__ == '__main__':

    # Initialise spatial model object with defaults (monthly + 1hr and 24hr statistics used in fitting)
    m = model.Model(spatial_model=True)

    # Calculate observed/reference statistics from gauge time series files
    m.preprocess(
        metadata_path='./input/metadata.csv',
        timeseries_folder='./input',
        output_folder='Z:/DP/Work/ER/rwgen/testing/examples/stnsrp',
        outlier_method='trim'
    )

    # Fit model and save parameters to file
    m.fit(
        output_folder='Z:/DP/Work/ER/rwgen/testing/examples/stnsrp',
        n_workers=6
    )

    import sys
    sys.exit()

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
