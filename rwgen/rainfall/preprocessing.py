import os

import numpy as np
import pandas as pd

from . import analysis
from . import utils


def main(
        spatial_model,
        season_definitions,
        statistic_definitions,
        timeseries_format,
        timeseries_path,
        timeseries_folder,
        metadata,
        calculation_period,
        completeness_threshold,
        output_point_statistics_path,
        output_cross_correlation_path,
        output_phi_path,
        outlier_method,
        maximum_relative_difference,
        maximum_alterations,

):
    """
    Make tables/dataframes of reference statistics, weights and scale factors for use in model fitting.

    Notes:
        There are two scale factors used by the model:
            * gs - Scales a given statistic for a given point by its annual mean in the objective function
            * phi - Scales the NSRP rainfall process at a given site for spatial variation in the mean and variance

    """
    if not spatial_model:
        dfs = analysis.prepare_point_timeseries(
            df=None,
            timeseries_format=timeseries_format,
            timeseries_path=timeseries_path,
            calculation_period=calculation_period,
            season_definitions=season_definitions,
            completeness_threshold=completeness_threshold,
            durations=np.unique(statistic_definitions['duration']),
            outlier_method=outlier_method,
            maximum_relative_difference=maximum_relative_difference,
            maximum_alterations=maximum_alterations
        )
        statistics = analysis.calculate_point_statistics(statistic_definitions, dfs)
        gs, statistics = calculate_gs(statistics)
        phi, statistics = calculate_phi(statistics, override_phi=True)

    else:

        # Calculate statistics, gs and phi for each point
        statistics_dfs = {}
        gs_dfs = {}
        phi_dfs = {}
        for index, row in metadata.iterrows():
            if os.path.exists(row['file_path']):
                timeseries_path = row['file_path']
            else:
                timeseries_path = os.path.join(timeseries_folder, row['file_path'])
            point_statistic_defs = statistic_definitions.loc[statistic_definitions['name'] != 'cross-correlation']
            dfs = analysis.prepare_point_timeseries(
                df=None,
                timeseries_format=timeseries_format,
                timeseries_path=timeseries_path,
                calculation_period=calculation_period,
                season_definitions=season_definitions,
                completeness_threshold=completeness_threshold,
                durations=np.unique(point_statistic_defs['duration']),
                outlier_method=outlier_method,
                maximum_relative_difference=maximum_relative_difference,
                maximum_alterations=maximum_alterations
            )
            statistics = analysis.calculate_point_statistics(statistic_definitions, dfs)
            gs, statistics = calculate_gs(statistics)
            phi, statistics = calculate_phi(statistics, override_phi=False)
            statistics_dfs[row['point_id']] = statistics
            gs_dfs[row['point_id']] = gs
            phi_dfs[row['point_id']] = phi

        # Merge phi to get one table containing all points for output
        dfs = []
        for point_id, df in phi_dfs.items():
            df = df.copy()
            df['point_id'] = point_id
            dfs.append(df)
        phi = pd.concat(dfs)

        # Calculate cross-correlations
        xcorr_statistic_defs = statistic_definitions.loc[statistic_definitions['name'] == 'cross-correlation']
        unique_seasons = utils.identify_unique_seasons(season_definitions)
        cross_correlations = analysis.calculate_cross_correlations(xcorr_statistic_defs, unique_seasons, statistics_dfs)

        # Merge gs and phi into cross-correlations dataframe
        cross_correlations['gs'] = 1.0
        cross_correlations = pd.merge(cross_correlations, phi, how='left', on=['season', 'point_id'])
        phi2 = phi.copy()
        phi2.rename({'phi': 'phi2', 'point_id': 'point_id2'}, axis=1, inplace=True)
        cross_correlations = pd.merge(cross_correlations, phi2, how='left', on=['season', 'point_id2'])

        # Merge point statistics and cross-correlations
        dfs = []
        for point_id, df in statistics_dfs.items():
            df = df.copy()
            df['point_id'] = point_id
            dfs.append(df)
        point_statistics = pd.concat(dfs)
        statistics = utils.merge_statistics(point_statistics, cross_correlations)

    # Write statistics to output files
    utils.write_statistics(
        statistics, output_point_statistics_path, season_definitions, output_cross_correlation_path
    )
    if spatial_model:
        phi_ = pd.merge(metadata, phi, how='outer', on=['point_id'])
        utils.write_phi(phi_, output_phi_path)

    return statistics


def calculate_gs(statistics, merge=True):
    """
    Calculate gs scale factor as annual mean of statistic for a given site.

    """
    gs = statistics.groupby(['statistic_id', 'name'])['value'].mean()
    gs = gs.to_frame('gs')
    gs.reset_index(inplace=True)
    gs.loc[gs['name'] == 'probability_dry', 'gs'] = 1.0
    gs.loc[gs['name'] == 'autocorrelation', 'gs'] = 1.0
    gs = gs[['statistic_id', 'gs']]
    if merge:
        statistics = pd.merge(statistics, gs, how='left', on='statistic_id')
        return gs, statistics
    else:
        return gs


def calculate_phi(statistics, override_phi, merge=True):
    """
    Calculate phi scale factor for a given site/season as 24hr mean divided by 3.0 mm/day (following RainSim V3.1).

    """
    phi = statistics.loc[(statistics['name'] == 'mean') & (statistics['duration'] == 24), ['season', 'value']].copy()
    phi['value'] /= 3.0
    phi.rename({'value': 'phi'}, axis=1, inplace=True)
    if override_phi:
        phi['phi'] = 1.0
    if merge:
        statistics = pd.merge(statistics, phi, how='left', on='season')
        return phi, statistics
    else:
        return phi
