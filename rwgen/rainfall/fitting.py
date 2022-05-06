import numpy as np
import pandas as pd
import scipy.optimize

from . import properties
from . import utils


def main(
        season_definitions,
        spatial_model,
        intensity_distribution,
        fitting_method,
        reference_statistics,
        parameter_names,
        parameter_bounds,
        n_workers,
        output_parameters_path,
        output_point_statistics_path,
        output_cross_correlation_path,
        initial_parameters,
        initial_parameters_path,
        smoothing_tolerance
):
    """
    Fit model parameters.

    """
    # Helper for renaming columns in parameter output table
    parameter_output_renaming = {
        'number_of_iterations': 'iterations',
        'number_of_evaluations': 'function_evaluations',
        'fit_success': 'converged',
    }

    # Derived helper variables
    unique_seasons = list(set(season_definitions.values()))
    parameter_output_columns = ['fit_stage', 'season']
    for parameter_name in parameter_names:
        parameter_output_columns.append(parameter_name)
    parameter_output_columns.extend([
        'fit_success', 'objective_function', 'number_of_iterations', 'number_of_evaluations'
    ])

    # Select and run fitting method
    if fitting_method == 'default':
        parameters, fitted_statistics = fit_by_season(
            unique_seasons, reference_statistics, parameter_bounds, spatial_model, intensity_distribution, n_workers,
            parameter_names
        )
    elif fitting_method == 'nonparametric_smoothing':
        # parameters, fitted_statistics = fit_with_nonparametric_smoothing()
        pass

    # Write outputs
    df = parameters[parameter_output_columns]
    utils.write_csv_(df, output_parameters_path, season_definitions, parameter_output_renaming)
    utils.write_statistics(
        fitted_statistics, output_point_statistics_path, season_definitions, output_cross_correlation_path,
        write_weights=False, write_gs=False, write_phi=False
    )

    return parameters, fitted_statistics


def fit_by_season(
        unique_seasons, reference_statistics, parameter_bounds, spatial_model, intensity_distribution, n_workers,
        parameter_names, stage='final'
):
    """
    Optimise parameters for each season independently.

    """
    results = {}
    fitted_statistics = []
    for season in unique_seasons:
        if len(unique_seasons) == 12:
            print('    - Month =', season)
        else:
            print('    - Season =', season)

        # Gather relevant data, weights and objective function scaling terms
        season_reference_statistics = reference_statistics.loc[reference_statistics['season'] == season].copy()
        statistic_ids, fitting_data, ref, weights, gs = prepare(season_reference_statistics)

        # Run optimisation
        result = scipy.optimize.differential_evolution(
            func=fitting_wrapper,
            bounds=parameter_bounds[season],
            args=(
                spatial_model,
                intensity_distribution,
                statistic_ids,
                fitting_data,
                ref,
                weights,
                gs
            ),
            tol=0.001,
            updating='deferred',
            workers=n_workers
        )

        # Store optimisation results for season
        for idx in range(len(parameter_names)):
            results[(parameter_names[idx], season)] = result.x[idx]
        results[('fit_success', season)] = result.success
        results[('objective_function', season)] = result.fun
        results[('number_of_iterations', season)] = result.nit
        results[('number_of_evaluations', season)] = result.nfev

        # Get and store statistics associated with optimised parameters
        dfs = []
        parameters = []
        for parameter in parameter_names:
            parameters.append(results[(parameter, season)])
        mod_stats = calculate_analytical_properties(
            spatial_model, intensity_distribution, parameters, statistic_ids, fitting_data
        )
        for statistic_id in statistic_ids:
            tmp = fitting_data[(statistic_id, 'df')].copy()
            dfs.append(tmp)
        df = pd.concat(dfs)
        df['value'] = mod_stats
        df['season'] = season
        fitted_statistics.append(df)

    # Format results for output
    parameters = format_results(results)
    fitted_statistics = pd.concat(fitted_statistics)
    parameters['fit_stage'] = stage
    fitted_statistics['fit_stage'] = stage

    return parameters, fitted_statistics


# TO BE UPDATED
def fit_with_nonparametric_smoothing(self):
    # - do we want to write the results of the first step? yes - might be useful to inspect
    # - also useful to be able to start from a parameter file, rather than necessarily having to do the first step
    # - should we be operating on self.parameter_bounds or not?

    # First step is a normal season-wise fit
    # self.fit_by_season()

    # ---
    # Read starting parameters from here for now
    self.parameters = utils.read_csv_('H:/Projects/rwgen/examples/stnsrp/output/parameters.csv')
    # self.parameters = utils.read_csv_('H:/Projects/rwgen/examples/nsrp/output2/parameters.csv')
    # ---

    # Then smooth the parameter values using a +/-1 month (season?) weighted moving average

    # Insert (repeat) final season at beginning of df and first season at end to avoid boundary effects
    tmp1 = self.parameters.loc[self.parameters['season'] == 12].copy()
    tmp1.loc[:, 'season'] = 0
    tmp2 = self.parameters.loc[self.parameters['season'] == 1].copy()
    tmp2.loc[:, 'season'] = max(self.unique_seasons) + 1
    df = pd.concat([self.parameters, tmp1, tmp2])
    df.sort_values('season', inplace=True)

    # Weighted moving average
    def weighted_average(x):
        return (x.values[0] * 0.5 + x.values[1] + x.values[2] * 0.5) / 2.0

    df1 = df.rolling(window=3, center=True, on='season').apply(weighted_average)
    df1 = df1.loc[(df1['season'] >= min(self.unique_seasons)) & (df1['season'] <= max(self.unique_seasons))]

    # # Define new bounds for optimisation by season
    # seasonal_parameter_bounds = {}
    # for season in self.unique_seasons:
    #     seasonal_parameter_bounds[season] = []
    #     for parameter in self.parameter_names:
    #         parameter_idx = self.parameter_names.index(parameter)
    #         smoothed_initial_value = df1.loc[df1['season'] == season, parameter].values[0]
    #         offset = smoothed_initial_value * 0.25
    #         lower_bound = max(smoothed_initial_value - offset, self.parameter_bounds[parameter_idx][0])
    #         upper_bound = min(smoothed_initial_value + offset, self.parameter_bounds[parameter_idx][1])
    #         seasonal_parameter_bounds[season].append((lower_bound, upper_bound))

    # ALTERNATIVE Define new bounds for optimisation by season using fraction of annual mean of smoothed parameter
    seasonal_parameter_bounds = {}
    for season in self.unique_seasons:
        seasonal_parameter_bounds[season] = []
        for parameter in self.parameter_names:
            parameter_idx = self.parameter_names.index(parameter)
            parameter_mean = df1[parameter].mean()
            offset = parameter_mean * 0.15  # 0.25
            smoothed_initial_value = df1.loc[df1['season'] == season, parameter].values[0]
            lower_bound = max(smoothed_initial_value - offset, self.parameter_bounds[parameter_idx][0])
            upper_bound = min(smoothed_initial_value + offset, self.parameter_bounds[parameter_idx][1])
            seasonal_parameter_bounds[season].append((lower_bound, upper_bound))

    # Refit by season with refined bounds
    # - temporarily adjust output file names here - needs to be rationalised
    self.parameters_path = self.parameters_path.replace('.csv', '3_15.csv')
    self.point_statistics_path = self.point_statistics_path.replace('.csv', '3_15.csv')
    self.cross_correlation_path = self.cross_correlation_path.replace('.csv', '3_15.csv')
    self.fit_by_season(seasonal_parameter_bounds)


def format_results(results):
    df = pd.DataFrame.from_dict(results, orient='index', columns=['value'])
    df.index = pd.MultiIndex.from_tuples(df.index, names=['field', 'season'])
    df.reset_index(inplace=True)
    df = df.pivot(index='season', columns='field', values='value')
    df.reset_index(inplace=True)
    return df


def fitting_wrapper(
        parameters, spatial_model, intensity_distribution, statistic_ids, fitting_data, ref_stats, weights, gs
):
    mod_stats = calculate_analytical_properties(
        spatial_model, intensity_distribution, parameters, statistic_ids, fitting_data
    )
    obj_fun = calculate_objective_function(ref_stats, mod_stats, weights, gs)
    return obj_fun


def prepare(statistics):
    statistic_ids = sorted(list(set(statistics['statistic_id'])))

    fitting_data = {}
    reference_statistics = []
    weights = []
    gs = []
    for statistic_id in statistic_ids:
        df = statistics.loc[statistics['statistic_id'] == statistic_id].copy()

        fitting_data[(statistic_id, 'name')] = df['name'].values[0]
        fitting_data[(statistic_id, 'duration')] = df['duration'].values[0]
        fitting_data[(statistic_id, 'lag')] = df['lag'].values[0]
        fitting_data[(statistic_id, 'threshold')] = df['threshold'].values[0]
        fitting_data[(statistic_id, 'df')] = df

        reference_statistics.append(df['value'].values)
        weights.append(df['weight'].values)
        gs.append(df['gs'].values)

    reference_statistics = np.concatenate(reference_statistics)
    weights = np.concatenate(weights)
    gs = np.concatenate(gs)

    return statistic_ids, fitting_data, reference_statistics, weights, gs


def calculate_analytical_properties(spatial_model, intensity_distribution, parameters, statistic_ids, fitting_data):
    if not spatial_model:
        if intensity_distribution == 'exponential':
            lamda, beta, nu, eta, xi = parameters
    else:
        if intensity_distribution == 'exponential':
            lamda, beta, rho, eta, gamma, xi = parameters  # ! ORDER OF gamma AND xi SWAPPED HERE !
        nu = 2.0 * np.pi * rho / gamma ** 2.0

    if intensity_distribution == 'exponential':
        mu_1 = 1.0 / xi
        mu_2 = 2.0 / xi ** 2.0
        mu_3 = 6.0 / xi ** 3.0

    statistic_arrays = []
    for statistic_id in statistic_ids:
        name = fitting_data[(statistic_id, 'name')]
        duration = fitting_data[(statistic_id, 'duration')]
        phi = fitting_data[(statistic_id, 'df')]['phi'].values

        if name in ['autocorrelation', 'cross-correlation']:
            lag = fitting_data[(statistic_id, 'lag')]
            if name == 'cross-correlation':
                phi2 = fitting_data[(statistic_id, 'df')]['phi2'].values
                distances = fitting_data[(statistic_id, 'df')]['distance'].values
        elif name == 'probability_dry':
            threshold = fitting_data[(statistic_id, 'threshold')]

        if name == 'mean':
            values = properties.calculate_mean(duration, lamda, nu, mu_1, eta, phi)
        elif name == 'variance':
            values = properties.calculate_variance(
                duration, eta, beta, lamda, nu, mu_1, mu_2, phi
            )
        elif name == 'skewness':
            values = properties.calculate_skewness(
                duration, eta, beta, lamda, nu, mu_1, mu_2, mu_3, phi
            )
        elif name == 'autocorrelation':
            values = properties.calculate_autocorrelation(
                duration, lag, eta, beta, lamda, nu, mu_1, mu_2, phi
            )
        elif name == 'probability_dry':
            values = properties.calculate_probability_dry(
                duration, nu, beta, eta, lamda, phi, threshold
            )
        elif name == 'cross-correlation':
            values = properties.calculate_cross_correlation(
                duration, lag, eta, beta, lamda, nu, mu_1, mu_2, gamma, distances, phi, phi2
            )

        statistic_arrays.append(values)

    return np.concatenate(statistic_arrays)


def calculate_objective_function(ref, mod, w, sf):
    obj_fun = np.sum((w ** 2 / sf ** 2) * (ref - mod) ** 2)
    return obj_fun

