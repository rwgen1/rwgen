import os
import sys

import numpy as np
import pandas as pd

from rwgen.rainfall import preproc


def test_read_statistic_definitions():
    input_path = './data/nsrp_statistic_definitions.csv'
    statistic_definitions, weights = preproc.read_statistic_definitions(input_path)
    expected_statistic_definitions = {
        1: {'duration': 1, 'name': 'variance'},
        2: {'duration': 1, 'name': 'skewness'},
        3: {'duration': 1, 'name': 'probability_dry', 'threshold': 0.2},
        4: {'duration': 24, 'name': 'mean'},
        5: {'duration': 24, 'name': 'variance'},
        6: {'duration': 24, 'name': 'skewness'},
        7: {'duration': 24, 'name': 'probability_dry', 'threshold': 0.2},
        8: {'duration': 24, 'name': 'autocorrelation', 'lag': 1},
    }
    expected_weights = {
        1: 1.0,
        2: 2.0,
        3: 7.0,
        4: 6.0,
        5: 2.0,
        6: 3.0,
        7: 7.0,
        8: 6.0,
    }
    assert statistic_definitions == expected_statistic_definitions
    assert weights == expected_weights

def test_read_csvy():
    input_path = './data/brize-norton.csv'
    df1 = pd.read_csv(
        input_path, index_col=0, parse_dates=True,
        infer_datetime_format=True, dayfirst=True
    )
    df1.columns = ['Value']

    input_path = './data/brize-norton.csvy'
    df2 = preproc.read_csvy(input_path)

    assert np.all(df2.index.values == df1.index.values)
    assert np.all(df2['Value'].values == df1['Value'].values)

def test_read_statistics():
    input_path = './data/brize-norton_rainsim_statistics.csv'
    expected_statistics = {
        (1, 1): np.array([0.133675]),
        (2, 7): np.array([20.936564])

    }
    expected_statistic_definitions = {
        1: {'duration': 1, 'name': 'variance'},
        2: {'duration': 1, 'name': 'skewness'},
        3: {'duration': 1, 'name': 'probability_dry', 'threshold': 0.2},
        4: {'duration': 24, 'name': 'mean'},
        5: {'duration': 24, 'name': 'variance'},
        6: {'duration': 24, 'name': 'skewness'},
        7: {'duration': 24, 'name': 'probability_dry', 'threshold': 0.2},
        8: {'duration': 24, 'name': 'autocorrelation', 'lag': 1},
    }
    expected_weights = {}
    statistics, scale_factors, statistic_definitions, weights = preproc.read_statistics(
        input_path
    )
    assert statistics[(1, 1)] == expected_statistics[(1, 1)]
    assert statistics[(2, 7)] == expected_statistics[(2, 7)]
    assert statistic_definitions == expected_statistic_definitions
    assert weights == expected_weights

def test_calculate_statistics():
    # - check against rainsim statistics to be improved
    # - issue is that variance calculations in python are unbiased, whereas rainsim appears to be
    # using biased statistic

    timeseries_path = './data/brize-norton.csv'
    output_path = './output/brize-norton_statistics.csv'

    statistic_definitions = {
        1: {'duration': 1, 'name': 'variance'},
        2: {'duration': 1, 'name': 'skewness'},
        3: {'duration': 1, 'name': 'probability_dry', 'threshold': 0.2},
        4: {'duration': 24, 'name': 'mean'},
        5: {'duration': 24, 'name': 'variance'},
        6: {'duration': 24, 'name': 'skewness'},
        7: {'duration': 24, 'name': 'probability_dry', 'threshold': 0.2},
        8: {'duration': 24, 'name': 'autocorrelation', 'lag': 1},
    }
    season_definitions = {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12
    }

    statistics, scale_factors = preproc.calculate_point_statistics(
        timeseries_path=timeseries_path,
        timeseries_format='simple-csv',
        calculation_period=(1981,2020),
        completeness_threshold=100.0,
        statistic_definitions=statistic_definitions,
        season_definitions=season_definitions,
        output_statistics_path=output_path
    )

    rainsim_statistics_path = './data/brize-norton_rainsim_statistics.csv'
    rainsim_statistics, rainsim_scale_factors, _, _ = preproc.read_statistics(
        rainsim_statistics_path
    )

    calculated = [statistics[k] for k in sorted(statistics.keys())]
    rainsim = [rainsim_statistics[k] for k in sorted(statistics.keys())]
    calculated = np.asarray(calculated)
    rainsim = np.asarray(rainsim)
    percent_difference = (calculated - rainsim) / rainsim * 100.0
    assert np.percentile(percent_difference, 0.75) < 1.0

    calculated = [scale_factors[k] for k in sorted(scale_factors.keys())]
    rainsim = [rainsim_scale_factors[k] for k in sorted(scale_factors.keys())]
    calculated = np.asarray(calculated)
    rainsim = np.asarray(rainsim)
    percent_difference = (calculated - rainsim) / rainsim * 100.0
    assert np.max(np.abs(percent_difference)) < 1.0


