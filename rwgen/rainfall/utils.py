import os
import datetime
import zipfile
import re
import inspect
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import numpy.ma as ma
import xarray as xr
import pandas as pd
import geocube.api.core


def get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs


def parse_season_definitions(user_input):
    month_abbreviations = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
        'oct': 10, 'nov': 11, 'dec': 12
    }

    season_definitions = {}

    # Construct from keywords
    if isinstance(user_input, str):
        if user_input.lower() == 'monthly':
            for month in range(1, 12+1):
                season_definitions[month] = month

        elif (user_input.lower().startswith('quarterly')) \
                or (user_input.lower().startswith('half-years')):
            if '_' in user_input:
                start_month = month_abbreviations[user_input.lower().split('_')[1]]
            elif user_input.lower().startswith('quarterly'):
                start_month = 12
            elif user_input.lower().startswith('half-years'):
                start_month = 10

            month = start_month
            season = 1
            counter = 1
            if user_input.lower().startswith('quarterly'):
                max_counter = 3
            elif user_input.lower().startswith('half-years'):
                max_counter = 6

            for _ in range(12):
                if counter <= max_counter:
                    season_definitions[month] = season
                if month < 12:
                    month += 1
                elif month == 12:
                    month = 1
                if counter == max_counter:
                    season += 1
                    counter = 1
                else:
                    counter += 1

        elif (user_input.lower() == 'none') or (user_input.lower() == 'annual'):
            for month in range(1, 12+1):
                season_definitions[month] = 1

    elif isinstance(user_input, list):
        # Keys are offsets relative to the case that January is month 1 and values are the
        # corresponding month numbers
        start_months = {
            0: 1, 1: 12, 2: 11, 3: 10, 4: 9, 5: 8, 6: 7, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2
        }

        # Identify offset of first month relative to January
        concatenated = ''.join(user_input)
        concatenated = concatenated.lower()
        if len(concatenated) != 12:
            raise ValueError('Season abbreviations do not make up a year (too few/many months')
        for offset in range(0, 11+1):
            reordered = concatenated[offset:] + concatenated[:offset]
            if reordered == 'jfmamjjasond':
                break
            else:
                if offset < 11:
                    pass
                else:
                    raise ValueError('Unable to identify seasons - definitions may be incorrect')

        start_month = start_months[offset]
        month = start_month
        season = 1
        for season_abbreviation in user_input:
            for _ in season_abbreviation:  # looping each "month letter"
                season_definitions[month] = season
                if month < 12:
                    month += 1
                elif month == 12:
                    month = 1
            season += 1

    elif isinstance(user_input, dict):
        season_definitions = user_input

    return season_definitions


def identify_unique_seasons(season_definitions):
    return list(set(season_definitions.values()))


def check_if_leap_year(year):
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                leap_year = True
            else:
                leap_year = False
        else:
            leap_year = True
    else:
        leap_year = False
    return leap_year


def make_datetime_list(start_date, timestep, end_date=None, periods=None, calendar='gregorian'):
    date_series = []
    d = start_date
    if end_date is not None:
        while d <= end_date:
            date_series.append(d)
            d += datetime.timedelta(seconds=timestep * 60 * 60)
            if (calendar == '365-day') and (d.month == 2) and (d.day == 29):
                d = datetime.datetime(d.year, 3, 1)
    elif periods is not None:
        i = 1
        while i <= periods:
            date_series.append(d)
            d += datetime.timedelta(seconds=timestep * 60 * 60)
            if (calendar == '365-day') and (d.month == 2) and (d.day == 29):
                d = datetime.datetime(d.year, 3, 1)
            i += 1
    return date_series


def add_columns(df, dc):
    for key, value in dc.items():
        df[key] = value
    return df


def make_datetime_helper(start_year, end_year, timestep_length, calendar):
    # Construct dataframe of core date information
    unique_years = np.arange(start_year, end_year+1)
    years = np.repeat(unique_years, 12)
    months = np.tile(np.arange(1, 12+1), unique_years.shape[0])
    leap_year = (
        ((np.mod(years, 4) == 0) & (np.mod(years, 100) == 0) & (np.mod(years, 400) == 0))
        | ((np.mod(years, 4) == 0) & (np.mod(years, 100) != 0))
    )
    days = np.tile(np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), unique_years.shape)
    if calendar == 'gregorian':
        days[leap_year & (months == 2)] = 29
    hours = days * 24
    timesteps = hours / timestep_length
    timesteps = timesteps.astype(int)
    df = pd.DataFrame(dict(year=years, month=months, n_days=days, n_hours=hours, n_timesteps=timesteps))

    # Add helpers
    df['end_timestep'] = df['n_timesteps'].cumsum()  # beginning timestep of next month
    df['end_timestep'] = df['end_timestep'].astype(int)
    df['start_timestep'] = df['end_timestep'].shift()
    df.iloc[0, df.columns.get_loc('start_timestep')] = 0
    df['start_timestep'] = df['start_timestep'].astype(int)
    df['start_time'] = df['start_timestep'] * timestep_length
    df['end_time'] = df['end_timestep'] * timestep_length
    # df['n_hours'] = df['end_time'] - df['start_time']

    return df


def read_statistics(filepath):
    # Suitable for both statistics definitions and reference statistics files
    df = pd.read_csv(filepath)
    df.columns = [column_name.lower() for column_name in df.columns.tolist()]
    df.rename(columns={'month': 'season'}, inplace=True)
    df['threshold'] = np.nan  # 0.0
    df['lag'] = pd.NA  # 0
    for idx in range(df.shape[0]):
        name = df.iat[idx, df.columns.get_loc('name')]
        duration = df.iat[idx, df.columns.get_loc('duration')]
        if 'probability_dry' in name:
            threshold = name.replace('probability_dry_', '')
            threshold = threshold.replace('_probability_dry', '')
            if '.' in threshold:
                threshold = float(threshold.replace('mm', ''))
                if duration == '1H':
                    if threshold not in [0.0, 0.1, 0.2]:
                        raise ValueError('1-hour probability dry threshold must be 0.0, 0.1 or 0.2')
                elif duration == '24H':
                    if threshold not in [0.0, 0.2, 1.0]:
                        raise ValueError('24-hour probability dry threshold must be 0.0, 0.2 or 1.0')
                else:
                    threshold = 0.0
            else:
                threshold = 0.0
            df.iat[idx, df.columns.get_loc('threshold')] = threshold
            df.iat[idx, df.columns.get_loc('name')] = 'probability_dry'
        elif 'autocorrelation' in name:
            if 'lag' in name:
                lag = name.replace('autocorrelation_', '')
                lag = lag.replace('_autocorrelation', '')
                lag = int(lag.replace('lag', ''))
            else:
                lag = 1
            df.iat[idx, df.columns.get_loc('lag')] = lag
            df.iat[idx, df.columns.get_loc('name')] = 'autocorrelation'
        elif 'cross-correlation' in name:
            if 'lag' in name:
                lag = name.replace('cross-correlation_', '')
                lag = lag.replace('_cross-correlation', '')
                lag = int(lag.replace('lag', ''))
            else:
                lag = 0
            df.iat[idx, df.columns.get_loc('lag')] = lag
            df.iat[idx, df.columns.get_loc('name')] = 'cross-correlation'
    return df


def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = "percentile_{:d}".format(int(n * 100))  # TODO: Flexibility for non-integers
    return percentile_


def make_column_names_lowercase(df):
    column_names = [name.lower() for name in df.columns]
    df.columns = column_names
    return df


def merge_statistics(point_statistics, cross_correlations, value_columns='value'):
    # Ensure consistent columns and then merge
    point_statistics['point_id2'] = pd.NA
    point_statistics['distance'] = np.nan
    if 'phi' in cross_correlations.columns:
        point_statistics['phi2'] = np.nan
    cross_correlations['threshold'] = np.nan
    statistics = pd.concat([point_statistics, cross_correlations])

    # Construct core column order
    column_order = ['point_id', 'point_id2', 'distance', 'statistic_id', 'name', 'duration', 'weight', 'season']
    if not isinstance(value_columns, list):
        value_columns = list(value_columns)
    column_order.extend(value_columns)
    column_order.extend(['lag', 'threshold'])  # lag and threshold will be removed before write

    # Additional columns relevant in some cases
    if 'gs' in point_statistics.columns:
        column_order.append('gs')
    if 'phi' in cross_correlations.columns:
        column_order.extend(['phi', 'phi2'])
    if 'realisation_id' in statistics.columns:
        column_order.append('realisation_id')
    if 'subset_id' in statistics.columns:
        column_order.append('subset_id')

    statistics = statistics[column_order]

    return statistics


def _columns_to_write(columns_present, write_weights, write_gs, write_phi_, value_columns):  # statistic_names,
    if isinstance(value_columns, str):
        value_columns = [value_columns]

    # Core columns
    columns = ['realisation_id', 'subset_id', 'point_id', 'statistic_id', 'name', 'duration', 'season']
    columns.extend(value_columns)

    # Additional columns relevant in some cases
    if write_weights:
        columns.append('weight')
    if write_gs:
        columns.append('gs')
    if write_phi_:
        columns.append('phi')
    columns.extend(['point_id2', 'distance', 'phi2'])

    # Check columns are available - realisation_id and subset_id plus cross-correlation columns
    columns = [c for c in columns if c in columns_present]

    return columns


def _concise_statistic_names(df):
    for idx in range(df.shape[0]):
        name = df.iat[idx, df.columns.get_loc('name')]
        lag = df.iat[idx, df.columns.get_loc('lag')]
        threshold = df.iat[idx, df.columns.get_loc('threshold')]
        if name == 'autocorrelation':
            df.iat[idx, df.columns.get_loc('name')] = 'autocorrelation' + '_lag' + str(lag)
        if name == 'cross-correlation':
            df.iat[idx, df.columns.get_loc('name')] = 'cross-correlation' + '_lag' + str(lag)
        if name == 'probability_dry':
            df.iat[idx, df.columns.get_loc('name')] = 'probability_dry' + '_' + "{:.1f}".format(threshold) + 'mm'
    return df


def write_statistics(
        df, output_path, season_definitions, write_weights=True, write_gs=True, write_phi_=True, value_columns='value'
):
    # TODO: Rationalise into a loop to remove effectively duplicated code
    # Point statistics
    columns = _columns_to_write(df.columns, write_weights, write_gs, write_phi_, value_columns)
    df1 = df.copy()
    if 'point_id' not in df1.columns:
        df1['point_id'] = 1

    sort_columns = ['realisation_id', 'subset_id', 'point_id', 'statistic_id', 'season']
    sort_columns = [c for c in sort_columns if c in df1.columns]
    df1.sort_values(sort_columns, inplace=True)

    df1 = _concise_statistic_names(df1)
    df1 = df1[columns]
    if max(season_definitions.values()) == 12:
        df1.rename({'season': 'month'}, axis=1, inplace=True)
    df1.columns = [name.capitalize() for name in df1.columns]
    df1.columns = [name.replace('_id', '_ID') for name in df1.columns]

    df1.to_csv(output_path, index=False, na_rep='NA')


def write_phi(df, file_path):
    df1 = df.copy()
    df1.sort_values(['point_id', 'season'], inplace=True)
    if 'elevation' in df1.columns:
        df1 = df1[['point_id', 'easting', 'northing', 'elevation', 'season', 'phi']]
    else:
        df1 = df1[['point_id', 'easting', 'northing', 'season', 'phi']]
    if max(df1['season']) == 12:
        df1.rename({'season': 'month'}, axis=1, inplace=True)
    df1.columns = [name.capitalize() for name in df1.columns]
    df1.columns = [name.replace('_id', '_ID') for name in df1.columns]
    df1.to_csv(file_path, index=False)


def write_maxima(df, output_path, analysis_mode):
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'year', 'datetime': 'year'}, inplace=True)
    df = df.loc[:, ['realisation_id', 'point_id', 'duration', 'year', 'value']]

    df['dur_tmp'] = [int(duration[:-1]) for duration in df['duration']]
    df.sort_values(['realisation_id', 'point_id', 'dur_tmp', 'year'], inplace=True)
    df.drop(columns=['dur_tmp'], inplace=True)

    df.columns = ['Realisation_ID', 'Point_ID', 'Duration', 'Year', 'Value']
    if analysis_mode == 'preprocessing':
        df.drop(columns=['Realisation_ID'], inplace=True)
    df.to_csv(output_path, index=False)


def write_ddf(df, output_path):
    df = df[['point_id', 'duration', 'return_period', 'depth_gev', 'depth_empirical']]
    df.columns = ['Point_ID', 'Duration', 'Return_Period', 'Depth_GEV', 'Depth_Empirical']
    df.to_csv(output_path, na_rep='NA', index=False)


def resample(df, timestep_length, duration):
    expected_count = int(duration / timestep_length)
    if df.shape[0] % expected_count == 0:
        periods = int(df.shape[0] / expected_count)
    else:
        periods = int(df.shape[0] / expected_count) + 1
    mask = np.repeat(np.arange(periods, dtype=int), expected_count)
    mask = mask[:df.shape[0]]
    df['mask'] = mask
    df1 = df.groupby(['mask']).agg({'datetime': 'min', 'value': 'sum'})
    df1.index = pd.PeriodIndex(df1['datetime'])
    df1.drop(columns='datetime', inplace=True)

    return df1


def nested_dictionary_to_dataframe(dc, id_name, non_id_columns):
    ids = sorted(list(dc.keys()))
    data = {}
    for non_id_column in non_id_columns:
        data[non_id_column] = []
        for id_ in ids:
            values = dc[id_]
            data[non_id_column].append(
                values[non_id_column] if non_id_column in values.keys() else 'NA'
            )
    dc1 = {}
    dc1[id_name] = ids
    for non_id_column in non_id_columns:
        dc1[non_id_column] = data[non_id_column]
    df = pd.DataFrame(dc1)
    return df


def read_csv_(file_path):
    df = pd.read_csv(file_path)
    df.columns = [name.lower() for name in df.columns]
    df.rename({'month': 'season'}, axis=1, inplace=True)
    return df


def write_csv_(df, file_path, season_definitions, renaming=None, write_index=False):
    df1 = df.copy()
    if max(season_definitions.values()) == 12:
        df1.rename({'season': 'month'}, axis=1, inplace=True)
    if renaming is not None:
        df1.rename(renaming, axis=1, inplace=True)

    column_names = []
    for name in df1.columns:
        name_components = name.split('_')
        new_name = []
        for component in name_components:
            if component not in ['a', 'the', 'in', 'of', 'at', 'with']:
                new_name.append(component.capitalize())
            elif component == 'id':
                new_name.append('ID')
            else:
                new_name.append(component)
        new_name = '_'.join(new_name)
        column_names.append(new_name)
    df1.columns = column_names
    df1.to_csv(file_path, index=write_index)


def read_ascii_raster(file_path, data_type=float):
    # Headers
    dc = {}
    with open(file_path, 'r') as fh:
        for i in range(6):
            line = fh.readline()
            key, val = line.rstrip().split()
            key = key.lower()
            dc[key] = val

    nx = int(dc['ncols'])
    ny = int(dc['nrows'])
    cell_size = float(dc['cellsize'])
    if ('xllcorner' in dc.keys()) and ('yllcorner' in dc.keys()):
        xll = float(dc['xllcorner']) + cell_size / 2.0
        yll = float(dc['yllcorner']) + cell_size / 2.0
    elif ('xllcenter' in dc.keys()) and ('yllcenter' in dc.keys()):
        xll = float(dc['xllcenter'])
        yll = float(dc['yllcenter'])
    if data_type == float:
        nodata_flag = float(dc['nodata_value'])
    else:
        nodata_flag = int(dc['nodata_value'])

    # Values array
    arr = np.loadtxt(file_path, dtype=data_type, skiprows=6)
    arr = ma.masked_values(arr, nodata_flag)

    # Convert to xarray data array
    x = np.arange(xll, xll + cell_size * nx, cell_size)
    y = np.arange(yll, yll + cell_size * ny, cell_size)
    y = y[::-1]  # check
    da = xr.DataArray(
        data=arr,
        dims=['y', 'x'],
        coords={
            'x': x,
            'y':  y,
        },
    )

    return da


def round_down(x, base):
    return base * int(np.floor(x/base))


def round_up(x, base):
    return base * int(np.ceil(x/base))


def geodataframe_bounding_box(gdf, round_extent=False, resolution=None):
    xmin = np.min(gdf.bounds.minx)
    ymin = np.min(gdf.bounds.miny)
    xmax = np.max(gdf.bounds.maxx)
    ymax = np.max(gdf.bounds.maxy)
    if round_extent:
        xmin = round_down(xmin, resolution)
        ymin = round_down(ymin, resolution)
        xmax = round_up(xmax, resolution)
        ymax = round_up(ymax, resolution)
    return xmin, ymin, xmax, ymax


def ascii_grid_headers_from_extent(xmin, ymin, xmax, ymax, cell_size, nodata_value=-999):
    # Assuming xmin, ymin, xmax, ymax provide outer bounds (not cell centres)
    nx = ((xmax - cell_size / 2.0) - (xmin + cell_size / 2.0)) / cell_size + 1
    ny = ((ymax - cell_size / 2.0) - (ymin + cell_size / 2.0)) / cell_size + 1
    dc = {
        'ncols': nx,
        'nrows': ny,
        'xllcorner': xmin,
        'yllcorner': ymin,
        'cellsize': cell_size,
        'nodata_value': nodata_value
    }
    return dc


def grid_definition_from_ascii(filepath):
    dc = {}
    with open(filepath, 'r') as fh:
        for _ in range(6):
            line = fh.readline()
            line = line.rstrip().split()
            if line[0] in ['ncols', 'nrows']:
                dc[line[0]] = int(line[1])
            else:
                dc[line[0]] = float(line[1])
    return dc


def define_grid_extent(catchments, cell_size, dem):
    """
    Identify grid extent that fits in catchments and aligns with DEM if present.

    """
    xmin, ymin, xmax, ymax = geodataframe_bounding_box(catchments, round_extent=False)
    if dem is not None:
        new_xmin = round_down(xmin, cell_size)
        new_ymin = round_down(ymin, cell_size)
        new_xmax = round_up(xmax, cell_size)
        new_ymax = round_up(ymax, cell_size)

    grid = ascii_grid_headers_from_extent(new_xmin, new_ymin, new_xmax, new_ymax, cell_size)

    return grid


def grid_limits(grid):
    xmin = grid['xllcorner']
    ymin = grid['yllcorner']
    xmax = xmin + grid['ncols'] * grid['cellsize']
    ymax = ymin + grid['nrows'] * grid['cellsize']
    return xmin, ymin, xmax, ymax


def catchment_weights(
        catchment_polygons, xmin, ymin, xmax, ymax, output_grid_resolution, id_field, epsg_code,
        shapefile_grid_resolution=200
):
    # Fractional coverage of catchment in each grid cell being used in output discretisation
    # in the spatial-temporal model
    # - careful with assumptions on ordering of arrays etc

    # catchment_polygons = geopandas.read_file(catchment_polygons)
    catchment_polygons['Dummy'] = 1  # helps get fractional coverage when coarsening
    number_of_catchments = catchment_polygons.shape[0]

    discretisation_points = {}

    for index in range(number_of_catchments):

        # Duplicate the required polygon to keep the shape of the geodataframe
        # - double square brackets to make sure iloc returns a dataframe not a series
        catchment_polygon = pd.concat([catchment_polygons, catchment_polygons.iloc[[index]]])
        catchment_polygon.index = range(number_of_catchments + 1)
        catchment_polygon = catchment_polygon.loc[
            (catchment_polygon.index == index) | (catchment_polygon.index == np.max(catchment_polygon.index))
        ]
        catchment_polygon.crs = epsg_code

        # Discretisation on high resolution grid
        cube = geocube.api.core.make_geocube(
            catchment_polygon,
            measurements=["Dummy"],
            resolution=(shapefile_grid_resolution, -shapefile_grid_resolution),
            fill=0,
            geom=(
                '{"type": "Polygon", '
                + '"crs": {"properties": {"name": "EPSG:' + str(epsg_code) + '"}}, '
                + '"coordinates": [['
                + '[' + str(xmin) + ', ' + str(ymin) + '], '
                + '[' + str(xmin) + ', ' + str(ymax) + '], '
                + '[' + str(xmax) + ', ' + str(ymax) + '], '
                + '[' + str(xmax) + ', ' + str(ymin) + ']'
                + ']]'
                + '}'
            )
        )

        # Swap easting coordinates so go from low to high (east-west)
        cube = cube.reindex(x=cube.x[::-1])

        # Swap northing coordinates so go from high to low (north-south)
        cube = cube.reindex(y=cube.y[::-1])

        # Resample (coarsen) to output grid resolution
        window = int(output_grid_resolution / shapefile_grid_resolution)
        cube = cube.coarsen(x=window).mean().coarsen(y=window).mean()

        # Add all points to output dictionary
        xx, yy = np.meshgrid(cube.x, cube.y)
        xf = xx.flatten()
        yf = yy.flatten()
        weights = cube.Dummy.data.flatten()
        catchment_id = catchment_polygon.loc[catchment_polygon.index == index, id_field].values[0]
        discretisation_points[catchment_id] = {'x': xf, 'y': yf, 'weight': weights}

    return discretisation_points


def trim_array(max_relative_difference, max_removals):
    def f(x):
        y = x.copy()
        removals = 0
        while True:
            y_max = np.max(y)
            y_max_count = np.sum(y == y_max)
            y_next_largest = np.max(y[y < y_max])
            if y_max / y_next_largest > max_relative_difference:
                if removals + y_max_count <= max_removals:
                    y = y[y < y_max]
                    removals += y_max_count
                else:
                    break
            else:
                break
        return y
    return f


def clip_array(max_relative_difference, max_clips):
    # - assuming working with zero-bounded values
    def f(x):
        y = x.copy()
        clips = 0
        clip_flag = -999
        while True:
            y_max = np.max(y)
            y_max_count = np.sum(y == y_max)
            y_next_largest = np.max(y[y < y_max])
            if y_max / y_next_largest > max_relative_difference:
                if clips + y_max_count <= max_clips:
                    y[y == y_max] = clip_flag
                    clips += y_max_count
                else:
                    break
            else:
                break
        y[y == clip_flag] = np.max(y)
        return y
    return f


def define_parameter_bounds(parameter_bounds, fixed_parameters, required_parameters, default_bounds, unique_seasons):
    """
    Set up required parameter name lists and bounds.

    """
    # Prepare parameter bounds dataframe for internal use
    if parameter_bounds is not None:
        df = parameter_bounds.copy()
    else:
        df = pd.DataFrame({'season': [], 'parameter': []})
    df.columns = [column_name.lower() for column_name in df.columns.tolist()]
    df.rename(columns={'month': 'season'}, inplace=True)
    df['parameter'] = df['parameter'].apply(lambda x: x.lower())
    df['season'] = df['season'].apply(lambda x: -1 if x in ['All', 'all'] else int(x))  # lambda x: f(x)
    df.loc[(df['parameter'] == 'lambda') | (df['parameter'] == 'lamda'), 'parameter'] = 'lamda'

    # Prepare fixed parameters dataframe for internal use
    if fixed_parameters is not None:
        df1 = fixed_parameters.copy()
    else:
        df1 = pd.DataFrame({'season': []})
    df1.columns = [column_name.lower() for column_name in df1.columns.tolist()]
    df1.rename(columns={'month': 'season', 'lambda': 'lamda'}, inplace=True)
    df1['season'] = df1['season'].apply(lambda x: -1 if x in ['All', 'all'] else int(x))  # lambda x: f(x)

    # Give fixed values precedence in the event that both fixed values and bounds have been provided
    fixed_parameter_names = [name for name in df1.columns.tolist() if name != 'season']
    df = df.loc[~df['parameter'].isin(fixed_parameter_names)]

    # Reshape fixed parameters df so that it can be merged with bounds df
    df1 = pd.melt(df1, id_vars='season', var_name='parameter', value_name='fixed_value')

    # Problems with merge for empty dataframes - so just copy over if empty, otherwise merge
    if len(df) == 0 and len(df1) == 0:
        df = df1
    else:
        df = pd.merge(df, df1, how='outer', on=['season', 'parameter'])

    # Parameters to fit (and their bounds) need to be ordered for optimisation (more flexibility on fixed parameters)
    fixed_parameters = {}  # key = tuple (season, parameter), values = parameter value
    parameters_to_fit = []  # list of parameters that need to be fitted
    parameter_bounds = {}  # key = season, values = list of tuples (lower bound, upper bound)
    for season in unique_seasons:
        parameter_bounds[season] = []

    # Keep in required order in loop
    for parameter in required_parameters:
        df1 = df.loc[df['parameter'] == parameter]

        # Use default bounds if no entry for parameter in df
        if df1.shape[0] == 0:
            parameters_to_fit.append(parameter)
            for season in unique_seasons:
                parameter_bounds[season].append((
                    default_bounds.loc[default_bounds['parameter'] == parameter, 'lower_bound'].values[0],
                    default_bounds.loc[default_bounds['parameter'] == parameter, 'upper_bound'].values[0]
                ))
        else:
            for season in unique_seasons:
                if df1.shape[0] == 1:
                    if ('fixed_value' in df1.columns) and np.isfinite(df1['fixed_value'].values[0]):
                        fixed_value = df1['fixed_value'].values[0]
                    else:
                        fixed_value = np.nan
                        lower_bound = df1['lower_bound'].values[0]
                        upper_bound = df1['upper_bound'].values[0]
                else:
                    if ('fixed_value' in df1.columns) and np.isfinite(df1['fixed_value'].values[0]):
                        fixed_value = df1.loc[df1['season'] == season, 'fixed_value'].values[0]
                    else:
                        fixed_value = np.nan
                        lower_bound = df1.loc[df1['season'] == season, 'lower_bound'].values[0]
                        upper_bound = df1.loc[df1['season'] == season, 'upper_bound'].values[0]

                if np.isfinite(fixed_value):
                    fixed_parameters[(season, parameter)] = fixed_value
                else:
                    if season == unique_seasons[0]:
                        parameters_to_fit.append(parameter)
                    parameter_bounds[season].append((lower_bound, upper_bound))

    return parameters_to_fit, fixed_parameters, parameter_bounds


def zip_files(folder, file_extension='.txt', delete_uncompressed=False):
    # Patterns to look for in file names - hardcoded realisation convention (e.g. _r1) and assuming that no more than
    # 9999 realisations likely
    patterns = [r'_r\d', r'_r\d\d', r'_r\d\d\d', r'_r\d\d\d\d']
    patterns = [p + file_extension for p in patterns]

    for root, dirs, file_names in os.walk(folder):
        for file_name in file_names:

            # Check for time series file realisation convention and extension (e.g. _r1.txt)
            zip_this_file = False
            for pattern in patterns:
                match = re.search(pattern, file_name)
                if match:
                    zip_this_file = True
                    break

            # Zip then optionally delete the uncompressed file
            if zip_this_file:
                src = os.path.join(root, file_name)
                dst = os.path.join(root, file_name.replace(file_extension, '.zip'))
                if not os.path.exists(dst):
                    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.write(src, arcname=file_name)
                else:
                    raise FileExistsError('Zip file already exists: ' + dst)
                if delete_uncompressed:
                    os.remove(src)
