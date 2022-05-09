


def main(
            output_types=None,
            output_folder=None,  # could also be dictionary for paths to point, catchment, grid folders...
            output_format=None,  # either string or dictionary? {'point': 'csv', 'catchment': 'txt'} ?
            output_prefix=None,
            season_definitions=None,
            process_class=None,
            parameters=None,
            points=None,
            catchments=None,
            catchment_id_field=None,
            grid=None,  # dictionary {'ncols': 10, 'nrow': 10, ...}
            cell_size=None,  # of grid for discretisation
            dem=None,  # path to ascii raster [or xarray data array]
            phi=None,  # phi df, path to phi df [or xarray data array]
            number_of_years=30,  # stick to <= 1000 for now?
            number_of_realisations=1,
            concatenate_output=False,
            equal_length_output=False,
            timestep_length=1,  # hrs
            start_year=2000,
            calendar='gregorian',  # gregorian or 365-day
):
    pass