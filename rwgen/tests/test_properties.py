import numpy as np
import pandas as pd

from rwgen.rainfall import properties


def test_calculate_properties():

    # ---
    # Site properties

    df = pd.DataFrame({'phi': [1]})
    statistic_definitions = {
        1: {'duration': 1,  'df': df, 'name': 'variance'},
        2: {'duration': 1,  'df': df, 'name': 'skewness'},
        3: {'duration': 1,  'df': df, 'name': 'probability_dry', 'threshold': 0.2},
        4: {'duration': 24, 'df': df, 'name': 'mean'},
        5: {'duration': 24, 'df': df, 'name': 'variance'},
        6: {'duration': 24, 'df': df, 'name': 'skewness'},
        7: {'duration': 24, 'df': df, 'name': 'probability_dry', 'threshold': 0.2},
        8: {'duration': 24, 'df': df, 'name': 'autocorrelation', 'lag': 1},
    }
    lamda = 0.016024
    beta = 0.051436
    nu = 5.717468
    eta = 1.166798
    xi = 0.915212

    calculated_statistics = properties.calculate_properties(
        range(1, 8+1), statistic_definitions, lamda, beta, eta, xi, nu
    )

    # rainsim_statistics = {
    #     (1, 1): 0.142966,
    #     (2, 1): 7.558288,
    #     (3, 1): 0.892626,
    #     (4, 1): 2.059041,
    #     (5, 1): 12.118274,
    #     (6, 1): 2.948261,
    #     (7, 1): 0.435684,
    #     (8, 1): 0.196921,
    # }

    rainsim_statistics = np.array([
        0.142966,
        7.558288,
        0.892626,
        2.059041,
        12.118274,
        2.948261,
        0.435684,
        0.196921,
    ])

    percent_difference = (
        (calculated_statistics - rainsim_statistics) / rainsim_statistics * 100.0
    )
    assert np.max(percent_difference) < 0.1

    # for key in calculated_statistics.keys():
    #     percent_difference = (
    #         (calculated_statistics[key][0] - rainsim_statistics[key])
    #         / rainsim_statistics[key] * 100.0
    #     )
    #     assert abs(percent_difference) < 0.1

    # ---
    # Cross-correlation

    df = pd.DataFrame({
        'statistic_id': [1],
        'phi':  [0.543226],
        'phi2': [0.637527],
        'distance': [57.4543296888929]})
    statistic_definitions = {
        #1: {'duration': 24, 'df': df, 'name': 'variance'},
        #1: {'duration': 24, 'df': df, 'name': 'skewness'},
        1: {'duration': 24, 'df': df, 'name': 'cross-correlation', 'lag': 0}
    }

    lamda = 0.015202
    beta = 0.062486
    rho = 0.000460
    eta = 0.971669
    xi = 0.712291
    gamma = 0.022540

    nu = 2.0 * np.pi * rho / gamma ** 2.0

    calculated_statistics = properties.calculate_properties(
        [1], statistic_definitions, lamda, beta, eta, xi, nu, gamma
    )

    rainsim_statistics = np.array([0.783749])

    percent_difference = (
            (calculated_statistics - rainsim_statistics) / rainsim_statistics * 100.0
    )
    assert np.max(percent_difference) < 0.1




test_calculate_properties()