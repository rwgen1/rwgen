import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import time
import datetime

import numpy as np
import numpy.ma as ma
import xarray as xr
import pandas as pd
import scipy.stats
import yaml
import geocube.api.core
import numba


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


def datetime_series(start_year, end_year, timestep, season_definitions, calendar='gregorian'):
    # - timestep minimum = 1hr currently
    # - put in pandas dataframe...?

    dc = {
        'year': [],
        'month': [],
        'day': [],
        'hour': [],
        'season': [],
        # 'timestep': [],
    }

    year = start_year
    while year <= end_year:

        if calendar == 'gregorian':
            is_leap_year = check_if_leap_year(year)
        elif calendar == '365-day':
            is_leap_year = False

        if is_leap_year:
            sdt = datetime.datetime(2000, 1, 1)
        else:
            sdt = datetime.datetime(2001, 1, 1)

        edt = datetime.datetime(sdt.year, 12, 31, 23, 59, 59)
        dt = sdt
        # timestep_start = 0
        while dt <= edt:
            dc['year'].append(year)
            dc['month'].append(dt.month)
            dc['day'].append(dt.day)
            dc['hour'].append(dt.hour)
            dc['season'].append(season_definitions[dt.month])
            # dc['timestep'].append(timestep_start)
            dt += datetime.timedelta(hours=timestep)
            # timestep_start += timestep

        year += 1

    df = pd.DataFrame(dc)

    df['season_uid'] = df['season'].ne(df['season'].shift()).cumsum()

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


# def statistic_definitions_from_reference_statistics():
#     pass
#
#
# def parse_statistic_definitions():
#     pass


def read_statistic_definitions(filepath):
    df = pd.read_csv(filepath)
    df.columns = [column_name.lower() for column_name in df.columns.tolist()]
    df.rename({'id': 'statistic_id'})
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
                if duration == 1:
                    if threshold not in [0.0, 0.1, 0.2]:
                        raise ValueError('1-hour probability dry threshold must be 0.0, 0.1 or 0.2')
                elif duration == 24:
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
            else:
                lag = 0
            df.iat[idx, df.columns.get_loc('lag')] = lag
            df.iat[idx, df.columns.get_loc('name')] = 'autocorrelation'
    return df


def write_statistic_definitions():
    pass


def make_column_names_lowercase(df):
    column_names = [name.lower() for name in df.columns]
    df.columns = column_names
    return df


def merge_statistics(point_statistics, cross_correlations):
    point_statistics['point_id2'] = pd.NA
    point_statistics['distance'] = np.nan
    if 'phi' in cross_correlations.columns:
        point_statistics['phi2'] = np.nan
    cross_correlations['threshold'] = np.nan
    statistics = pd.concat([point_statistics, cross_correlations])
    column_order = [
        'point_id', 'point_id2', 'distance', 'statistic_id', 'name', 'duration', 'weight', 'season', 'value',
        'lag', 'threshold'  # lag and threshold will be removed before write
    ]
    if 'gs' in point_statistics.columns:
        column_order.append('gs')
    if 'phi' in cross_correlations.columns:
        column_order.extend(['phi', 'phi2'])
    statistics = statistics[column_order]
    return statistics


def read_statistics(point_statistics_path, cross_correlations_path=None):
    # TODO: Fix to parse lag and threshold
    statistics = pd.read_csv(point_statistics_path)
    statistics = make_column_names_lowercase(statistics)
    statistics.rename({'month': 'season'}, axis=1, inplace=True)
    if cross_correlations_path is not None:
        cross_correlations = pd.read_csv(cross_correlations_path)
        cross_correlations = make_column_names_lowercase(cross_correlations)
        cross_correlations.rename({'month': 'season'}, axis=1, inplace=True)
        statistics = merge_statistics(statistics, cross_correlations)
    return statistics


def _columns_to_write(statistic_type, write_weights, write_gs, write_phi):
    if statistic_type == 'point':
        columns = ['point_id', 'statistic_id', 'name', 'duration', 'season', 'value']
    elif statistic_type == 'cross-correlation':
        columns = ['point_id', 'point_id2', 'distance', 'statistic_id', 'name', 'duration', 'season', 'value']
    if write_weights:
        columns.append('weight')
    if write_gs:
        columns.append('gs')
    if write_phi:
        columns.append('phi')
        if statistic_type == 'cross-correlation':
            columns.append('phi2')
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
        df, point_path, season_definitions, cross_correlation_path=None, write_weights=True, write_gs=True,
        write_phi=True
):
    # Point statistics
    columns = _columns_to_write('point', write_weights, write_gs, write_phi)
    df1 = df.copy()
    df1 = df1.loc[df1['name'] != 'cross-correlation']
    if 'point_id' not in df1.columns:
        df1['point_id'] = 1
    df1.sort_values(['point_id', 'statistic_id', 'season'], inplace=True)
    df1 = _concise_statistic_names(df1)
    df1 = df1[columns]
    if max(season_definitions.values()) == 12:
        df1.rename({'season': 'month'}, axis=1, inplace=True)
    df1.columns = [name.capitalize() for name in df1.columns]
    df1.columns = [name.replace('_id', '_ID') for name in df1.columns]
    df1.to_csv(point_path, index=False)

    # Cross-correlation statistics
    if cross_correlation_path is not None:
        columns = _columns_to_write('cross-correlation', write_weights, write_gs, write_phi)
        df1 = df.copy()
        df1 = df1.loc[df1['name'] == 'cross-correlation']
        df1.sort_values(['point_id', 'statistic_id', 'season'], inplace=True)
        df1 = _concise_statistic_names(df1)
        df1 = df1[columns]
        if max(season_definitions.values()) == 12:
            df1.rename({'season': 'month'}, axis=1, inplace=True)
        df1.columns = [name.capitalize() for name in df1.columns]
        df1.columns = [name.replace('_id', '_ID') for name in df1.columns]
        df1.to_csv(cross_correlation_path, index=False)


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


def read_csv_timeseries(input_path):
    df = pd.read_csv(
        input_path, index_col=0, parse_dates=True, infer_datetime_format=True, dayfirst=True
    )
    df.columns = ['value']
    return df


def read_csvy_timeseries(input_path):
    with open(input_path, 'r') as fh:
        fh.readline()
        number_of_headers = 1
        headers = ''
        for line in fh:
            if line.rstrip() in ['---', '...']:
                number_of_headers += 1
                break
            else:
                headers = headers + line
                number_of_headers += 1

    metadata = yaml.safe_load(headers)

    df = pd.read_csv(input_path, skiprows=number_of_headers, names=['value'])
    start_datetime = datetime.datetime.strptime(
        metadata['start_datetime'], metadata['datetime_format']
    )
    datetime_series = pd.date_range(
        start_datetime, periods=df.shape[0], freq=str(metadata['interval_in_hours']) + 'H'
    )
    df['date_time'] = datetime_series
    df.set_index('date_time', inplace=True)
    #df.loc[df['value'] < 0.0] = np.nan

    return df


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


def format_with_leading_zeros(x, min_string_length=3):  # TODO: Check if still used anywhere
    format_string = '0' + str(max(min_string_length, (len(str(x)))))
    return format(x, format_string)


def read_csv_(file_path):
    df = pd.read_csv(file_path)
    df.columns = [name.lower() for name in df.columns]
    df.rename({'month': 'season'}, axis=1, inplace=True)
    return df


def write_csv_(df, file_path, season_definitions, renaming, write_index=False):
    df1 = df.copy()
    if max(season_definitions.values()) == 12:
        df1.rename({'season': 'month'}, axis=1, inplace=True)
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
        dem_cell_size = dem.x.values[1] - dem.x.values[0]
        new_xmin = dem.x.values[0] - dem_cell_size / 2.0
        new_ymin = dem.y.values[-1] - dem_cell_size / 2.0
        x_offset = new_xmin - round_down(xmin, cell_size)
        y_offset = new_ymin - round_down(ymin, cell_size)
        new_xmax = round_up(xmax, cell_size) - (cell_size - x_offset)
        new_ymax = round_up(ymax, cell_size) - (cell_size - y_offset)
        if new_xmax < xmax:
            xmax = new_xmax + cell_size
        else:
            xmax = new_xmax
        if new_ymax < ymax:
            ymax = new_ymax + cell_size
        else:
            ymax = new_ymax
        xmin = new_xmin
        ymin = new_ymin
    grid = ascii_grid_headers_from_extent(xmin, ymin, xmax, ymax, cell_size)
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
        catchment_polygon = catchment_polygons.append(catchment_polygons.iloc[index])
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

        # Adjust data array itself to match (only change in y-direction required)
        cube.Dummy.data = np.flipud(cube.Dummy.data)

        # Resample (coarsen) to output grid resolution
        window = int(output_grid_resolution / shapefile_grid_resolution)
        cube = cube.coarsen(x=window).mean().coarsen(y=window).mean()

        # Add relevant points to output dictionary
        # xx, yy = np.meshgrid(cube.x, cube.y)
        # x = xx[cube.Dummy.data > 0.0]
        # y = yy[cube.Dummy.data > 0.0]
        # weights = cube.Dummy.data[cube.Dummy.data > 0.0]
        # discretisation_points[id_field] = {'easting': x, 'northing': y, 'weight': weights}

        # Add all points to output dictionary
        xx, yy = np.meshgrid(cube.x, cube.y)
        xf = xx.flatten()
        yf = yy.flatten()
        weights = cube.Dummy.data.flatten()
        catchment_id = catchment_polygon.loc[catchment_polygon.index == index, id_field].values[0]
        discretisation_points[catchment_id] = {'x': xf, 'y': yf, 'weight': weights}

    return discretisation_points


# -----------------------------------------------------------------------------

# def trim_array(x, max_relative_difference, max_removals):
#     y = x.copy()
#     removals = 0
#     while True:
#         y_max = np.max(y)
#         y_max_count = np.sum(y == y_max)
#         y_next_largest = np.max(y[y < y_max])
#         if y_max / y_next_largest > max_relative_difference:
#             if removals + y_max_count <= max_removals:
#                 y = y[y < y_max]
#                 removals += y_max_count
#             else:
#                 break
#         else:
#             break
#     return y  # , removals

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
        return y  # , removals
    return f


# def clip_array(x, max_relative_difference, max_clips):
#     # - assuming working with zero-bounded values
#     y = x.copy()
#     clips = 0
#     clip_flag = -999
#     while True:
#         y_max = np.max(y)
#         y_max_count = np.sum(y == y_max)
#         y_next_largest = np.max(y[y < y_max])
#         if y_max / y_next_largest > max_relative_difference:
#             if clips + y_max_count <= max_clips:
#                 y[y == y_max] = clip_flag
#                 clips += y_max_count
#             else:
#                 break
#         else:
#             break
#     y[y == clip_flag] = np.max(y)
#     return y  # , clips

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
        return y  # , clips
    return f


# -----------------------------------------------------------------------------
# TODO: Remove discretisation functions as now part of simulation module

@numba.jit(nopython=True)
def discretise_point(
        period_start_time, period_end_time, timestep_length, raincell_arrival_times, raincell_end_times,
        raincell_intensities, discrete_rainfall
):
    # Modifying the discrete rainfall arrays themselves so need to ensure zeros before starting
    discrete_rainfall.fill(0.0)

    # Discretise each raincell in turn
    for idx in range(raincell_arrival_times.shape[0]):

        # Times relative to period start
        rc_arrival_time = raincell_arrival_times[idx] - period_start_time
        rc_end_time = raincell_end_times[idx] - period_start_time
        rc_intensity = raincell_intensities[idx]

        # Timesteps relative to period start
        rc_arrival_timestep = int(np.floor(rc_arrival_time / timestep_length))
        rc_end_timestep = int(np.floor(rc_end_time / timestep_length))  # timestep containing end

        # Proportion of raincell in each relevant timestep
        for timestep in range(rc_arrival_timestep, rc_end_timestep+1):
            timestep_start_time = timestep * timestep_length
            timestep_end_time = (timestep + 1) * timestep_length
            effective_start = np.maximum(rc_arrival_time, timestep_start_time)
            effective_end = np.minimum(rc_end_time, timestep_end_time)
            timestep_coverage = effective_end - effective_start

            if timestep < discrete_rainfall.shape[0]:
                discrete_rainfall[timestep] += rc_intensity * timestep_coverage


@numba.jit(nopython=True)  # parallel=True
def discretise_multiple_points(
        period_start_time, period_end_time, timestep_length, raincell_arrival_times, raincell_end_times,
        raincell_intensities, discrete_rainfall,
        raincell_x_coords, raincell_y_coords, raincell_radii,
        point_eastings, point_northings, point_phi,  # point_ids,
):
    # - id array needed?

    # t0 = datetime.datetime.now()

    # Modifying the discrete rainfall arrays themselves so need to ensure zeros before starting
    discrete_rainfall.fill(0.0)

    # t1 = datetime.datetime.now()

    # Subset raincells based on whether they intersect the point being discretised
    for idx in range(point_eastings.shape[0]):
    # for idx in numba.prange(point_eastings.shape[0]):
        x = point_eastings[idx]
        y = point_northings[idx]
        yi = idx

        # t2 = datetime.datetime.now()

        # TO BE RESTORED
        # distances_from_raincell_centres = np.sqrt((x - raincell_x_coords) ** 2 + (y - raincell_y_coords) ** 2)
        # mask = distances_from_raincell_centres <= raincell_radii
        #
        # discretise_point(
        #     period_start_time, period_end_time, timestep_length, raincell_arrival_times[mask],
        #     raincell_end_times[mask], raincell_intensities[mask], discrete_rainfall[:, yi]
        # )

        # TESTING
        distances_from_raincell_centres = np.sqrt((x - raincell_x_coords) ** 2 + (y - raincell_y_coords) ** 2)
        spatial_mask = distances_from_raincell_centres <= raincell_radii

        discretise_point(
            period_start_time, period_end_time, timestep_length, raincell_arrival_times[spatial_mask],
            raincell_end_times[spatial_mask], raincell_intensities[spatial_mask], discrete_rainfall[:, yi]
        )

        discrete_rainfall[:, yi] *= point_phi[idx]

        # t5 = datetime.datetime.now()

        # print(t5-t2)

    # print()
    # print(t5-t1)
    # sys.exit()

# -----------------------------------------------------------------------------

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

    # print(parameters_to_fit)
    # print(parameter_bounds)
    # print(fixed_parameters)

    return parameters_to_fit, fixed_parameters, parameter_bounds

