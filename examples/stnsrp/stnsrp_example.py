"""
Example of spatial NSRP modelling based on a catchment in the eastern Rhine basin.

"""
from rwgen.rainfall import model

# Boilerplate line needed to use multiprocessing in fitting on Windows OS
if __name__ == '__main__':

    # Initialise spatial model object with defaults (monthly + 1hr and 24hr statistics used in fitting)
    m = model.Model(spatial_model=True)

    # # Calculate observed/reference statistics from gauge time series files
    # m.preprocess(
    #     metadata_path='./input/metadata.csv',
    #     timeseries_folder='./input',
    #     output_folder='Z:/DP/Work/ER/rwgen/testing/examples/stnsrp',  # './output'
    #     outlier_method='trim'
    # )
    #
    # # Fit model and save parameters to file
    # m.fit(
    #     output_folder='Z:/DP/Work/ER/rwgen/testing/examples/stnsrp',  # './output'
    #     n_workers=6
    # )

    # Simulate three realisations of 1000 years at an hourly timestep (the default)
    m.simulate(
        output_types=['point', 'catchment'],
        output_folder='Z:/DP/Work/ER/rwgen/testing/examples/stnsrp',  # './output'
        parameters='Z:/DP/Work/ER/rwgen/testing/examples/stnsrp/parameters.csv',  # './output/parameters.csv'
        points='./input/metadata.csv',
        catchments='./input/catchments.shp',
        epsg_code=32632,
        cell_size=1000.0,
        dem='./input/srtm_dem.asc',
        phi='Z:/DP/Work/ER/rwgen/testing/examples/stnsrp/phi.csv',  # './output/phi.csv',
        simulation_length=1000,
        number_of_realisations=3,
    )
