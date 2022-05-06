from rwgen.rainfall import utils


def test_catchment_weights():
    shapefile_path = '../tests/data/stnsrp_catchments.shp'
    output_grid_resolution = 1000  # m
    id_field = 'ID_STRING'
    epsg_code = 27700
    utils.catchment_weights(shapefile_path, output_grid_resolution, id_field, epsg_code)
