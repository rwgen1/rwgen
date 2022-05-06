import numpy as np

from rwgen.rainfall import nsrp


def test_Model_init():

    model1 = nsrp.Model()

    model2 = nsrp.Model(
        statistic_definitions_path='./data/nsrp_statistic_definitions.csv'
    )
    assert model2.statistic_definitions == model1.statistic_definitions
    assert model2.weights == model1.weights

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
    weights = {
        1: 1.0,
        2: 2.0,
        3: 7.0,
        4: 6.0,
        5: 2.0,
        6: 3.0,
        7: 7.0,
        8: 6.0,
    }
    model3 = nsrp.Model(
        statistic_definitions=statistic_definitions,
        weights=weights
    )
    assert model3.statistic_definitions == model1.statistic_definitions
    assert model3.weights == model1.weights

    season_definitions = {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12
    }
    model4 = nsrp.Model(season_definitions=season_definitions)
    assert model4.season_definitions == model1.season_definitions

def test_Model_preprocess():
    model = nsrp.Model()
    model.preprocess(
        timeseries_path='./data/brize-norton.csv',
        timeseries_format='simple-csv',
        write_output=False,
    )
    # TODO: Add checks for each preprocess option

def test_Model_fit():
    model = nsrp.Model()
    model.preprocess(
        timeseries_path='./data/brize-norton.csv',
        timeseries_format='simple-csv',
        completeness_threshold=100.0,
        write_output=False,
    )
    # model.unique_seasons = [1]
    model.fit(
        parameters_output_path='./output/parameters.csv'
    )
    for season in model.unique_seasons:
        assert model.optimisation_results[season][0]
    # TODO: Check objective function against RainSim reference

def test_Model_simulate():
    pass

def test_calculate_objective_function():
    model = nsrp.Model()
    model.preprocess(
        timeseries_path='./data/brize-norton.csv',
        timeseries_format='simple-csv',
        completeness_threshold=100.0,
        write_output=False,
    )

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
    season = 1

    model_statistics = model.reference_statistics
    objective_function = nsrp.calculate_objective_function(
        statistic_definitions, season, model_statistics, model.reference_statistics,
        model.weights, model.fitting_scale_factors
    )
    assert abs(objective_function) < 0.000001

def test_fit_nsrp():
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
    season = 1
    intensity_distribution = 'exponential'

    lamda = 0.016024
    beta = 0.051436
    nu = 5.717468
    eta = 1.166798
    xi = 0.915212
    parameters = [lamda, beta, nu, eta, xi]

    # I.e. rainsim fitted statistics corresponding with this set of parameters
    reference_statistics = {
        (1, 1): np.asarray(0.142966),
        (2, 1): np.asarray(7.558288),
        (3, 1): np.asarray(0.892626),
        (4, 1): np.asarray(2.059041),
        (5, 1): np.asarray(12.118274),
        (6, 1): np.asarray(2.948261),
        (7, 1): np.asarray(0.435684),
        (8, 1): np.asarray(0.196921),
    }

    # These parameters should give model statistics matching the reference, so weights and scale
    # factors should not matter - objective function should be ~0 anyway
    weights = {
        1: 1.0,
        2: 2.0,
        3: 7.0,
        4: 6.0,
        5: 2.0,
        6: 3.0,
        7: 7.0,
        8: 6.0,
    }
    scale_factors = {
        1: 1.0,
        2: 1.0,
        3: 1.0,
        4: 1.0,
        5: 1.0,
        6: 1.0,
        7: 1.0,
        8: 1.0,
    }

    objective_function = nsrp.fit_nsrp(
        parameters, statistic_definitions, season, reference_statistics, weights, scale_factors,
        phi=np.array([1.0])
    )
    assert abs(objective_function) < 0.0001

def test_discretise_point():
    pass