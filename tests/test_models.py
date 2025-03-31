import os

import numpy as np
import pandas as pd
import pytest

import rwgen


def _dataframes_match(df1, df2, cols_to_ignore=None):
    if cols_to_ignore is None:
        cols_to_ignore = []

    for col in df1.columns:
        if col not in cols_to_ignore:
            if isinstance(df1[col][0], int):
                assert np.all(df1[col] == df2[col])
            elif isinstance(df1[col][0], float):
                assert np.all(np.isfinite(df1[col]) == np.isfinite(df2[col]))
                assert np.allclose(
                    df1.loc[np.isfinite(df1[col]), col],
                    df2.loc[np.isfinite(df2[col]), col]
                )
            else:
                assert np.all(df1[col] == df2[col])

    return True


@pytest.fixture
def nsrp_output_folder():
    return './tests/output/point'


@pytest.fixture
def nsrp_model(nsrp_output_folder):
    model = rwgen.RainfallModel(
        spatial_model=False,
        project_name='brize_norton',
        input_timeseries='./tests/data/point/inputs/00605_brize-norton.csv',
        intensity_distribution='weibull',
        output_folder=nsrp_output_folder,
    )
    return model


@pytest.fixture
def stnsrp_output_folder():
    return './tests/output/spatial'


@pytest.fixture
def stnsrp_model(stnsrp_output_folder):
    input_folder = './tests/data/spatial/inputs/hourly_rainfall'
    model = rwgen.RainfallModel(
        spatial_model=True,
        project_name='thames',
        input_timeseries=f'{input_folder}/time_series',
        point_metadata=f'{input_folder}/gauge_metadata.csv',
        intensity_distribution='weibull',
        output_folder=stnsrp_output_folder,
    )
    return model


@pytest.fixture
def wg_site_output_folder():
    return './tests/output/wg_site'


@pytest.fixture
def wg_site(wg_site_output_folder):
    pass


@pytest.fixture
def wg_spatial_output_folder():
    return './tests/output/wg_spatial'


@pytest.fixture
def wg_spatial(wg_spatial_output_folder):
    pass


def test_preprocess__nsrp(nsrp_model):
    nsrp_model.preprocess()

    df1 = pd.read_csv(
        os.path.join(nsrp_model.output_folder, 'reference_statistics.csv')
    )
    df2 = pd.read_csv('./tests/data/point/outputs/reference_statistics.csv')
    assert _dataframes_match(df1, df2)


def test_preprocess__stnsrp(stnsrp_model):
    stnsrp_model.preprocess()

    df1 = pd.read_csv(
        os.path.join(stnsrp_model.output_folder, 'reference_statistics.csv')
    )
    df2 = pd.read_csv('./tests/data/spatial/outputs/reference_statistics.csv')
    assert _dataframes_match(df1, df2)


def test_fit__nsrp(nsrp_model):
    nsrp_model.set_statistics(
        reference_statistics='./tests/data/point/outputs/reference_statistics.csv',
    )
    nsrp_model.fit(
        n_workers=8,
        pdry_iterations=0,
        parameter_bounds='./tests/data/point/inputs/parameter_bounds.csv',
        fixed_parameters='./tests/data/point/inputs/fixed_parameters.csv',
    )


def test_fit__stnsrp(stnsrp_model):
    stnsrp_model.set_statistics(
        reference_statistics='./tests/data/spatial/outputs/reference_statistics.csv',
    )
    stnsrp_model.fit(
        n_workers=8,
        pdry_iterations=0,
        parameter_bounds='./tests/data/spatial/inputs/parameter_bounds.csv',
        fixed_parameters='./tests/data/spatial/inputs/fixed_parameters.csv',
    )


def test_simulate__nsrp(nsrp_model):
    nsrp_model.set_parameters('./tests/data/point/outputs/parameters.csv')
    nsrp_model.simulate()
    # TODO: Check against reference simulation


def test_simulate__stnsrp(stnsrp_model):
    stnsrp_model.set_statistics(
        reference_statistics='./tests/data/spatial/outputs/reference_statistics.csv'
    )
    stnsrp_model.set_parameters('./tests/data/spatial/outputs/parameters.csv')
    stnsrp_model.simulate()
    # TODO: Check against reference simulation


# TODO: Test rainfall model postprocessing
# TODO: Test fitting with shuffling
# TODO: Test simulation with shuffling
# TODO: Tidy up test output files (also separate subfolders for rainfall and WG)
