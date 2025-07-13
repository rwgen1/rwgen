import os
import datetime

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from . import properties
from . import utils
from . import simulation
from . import analysis


def main(
        season_definitions,
        spatial_model,
        intensity_distribution,
        fitting_method,
        reference_statistics,
        all_parameter_names,
        parameters_to_fit,
        parameter_bounds,
        fixed_parameters,
        n_workers,
        output_parameters_path,
        output_statistics_path,
        write_output,
        n_iterations,  # for prebiasing (dry probability)
        output_folder,
        point_metadata,
        phi,
        statistic_definitions,
        random_seed,
        use_pooling,
):
    """
    Fit model parameters.

    """
    # Derived helper variables
    unique_seasons = list(set(season_definitions.values()))
    parameter_output_columns = ['fit_stage', 'season']
    for parameter_name in all_parameter_names:
        parameter_output_columns.append(parameter_name)
    parameter_output_columns.extend([
        'converged', 'objective_function', 'iterations', 'function_evaluations'
    ])

    # Month statistics not used in NSRP fitting
    reference_statistics = reference_statistics.loc[reference_statistics['duration'] != '1M'].copy()

    # Subset reference statistics further if pooling is being used
    if spatial_model and use_pooling:
        reference_statistics = reference_statistics.loc[reference_statistics['point_id'] == -1]

    # For now no initial parameters or modified parameter bounds during pre-biasing iterations
    par_bounds = parameter_bounds.copy()
    initial_parameters = None  # formerly an argument to functino

    # Do an initial fit
    ref_stats = reference_statistics.copy()
    _ref_stats = ref_stats.copy()
    _ref_stats['duration'] = [int(dur_code[:-1]) for dur_code in _ref_stats['duration']]
    if fitting_method == 'default':
        parameters, fitted_statistics = fit_by_season(
            unique_seasons, _ref_stats, spatial_model, intensity_distribution, n_workers,
            all_parameter_names, parameters_to_fit, par_bounds, fixed_parameters,
            initial_parameters=initial_parameters, use_pooling=use_pooling,
        )
    elif fitting_method == 'empirical_smoothing':
        raise NotImplementedError('Empirical smoothing method is not fully implemented in fitting yet.')

    # Iterations of NSRP fitting to allow for "pre-biasing" of reference statistics to account for (1) bias in
    # analytical vs simulated dry probability and (2) apparent bias in skewness
    rng = np.random.default_rng()
    tmp_folder = os.path.join(output_folder, 'tmp-' + str(rng.integers(100000, 999999)))
    for iteration in range(n_iterations):

        # Get simulated statistics and carry out "pre-biasing" calculations
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        stat_defs = statistic_definitions.loc[
            (statistic_definitions['name'] == 'skewness')
            | ((statistic_definitions['name'] == 'probability_dry') & (statistic_definitions['duration'] == '24H'))
        ]
        sim_stats = _get_simulated_statistics(
            intensity_distribution, tmp_folder, season_definitions, parameters, phi, stat_defs, random_seed,
            spatial_model, point_metadata,
        )
        ref_stats = _prebias_reference_statistics(
            reference_statistics,
            ref_stats,
            sim_stats,
            spatial_model,
            use_pooling,  # for ensuring point_id matches in merge
            iteration,
        )

        # And then proceed to fitting (which takes duration as an integer in hours)
        _ref_stats = ref_stats.copy()
        _ref_stats['duration'] = [int(dur_code[:-1]) for dur_code in _ref_stats['duration']]
        if fitting_method == 'default':
            parameters, fitted_statistics = fit_by_season(
                unique_seasons, _ref_stats, spatial_model, intensity_distribution, n_workers,
                all_parameter_names, parameters_to_fit, par_bounds, fixed_parameters,
                initial_parameters=initial_parameters, use_pooling=use_pooling,
            )
        elif fitting_method == 'empirical_smoothing':
            raise NotImplementedError('Empirical smoothing method is not fully implemented in fitting yet.')

    # Get rid of any temporary folders or files
    if os.path.exists(tmp_folder):
        os.rmdir(tmp_folder)

    # Update needed to work with reference duration codes
    fitted_statistics = fitted_statistics.loc[fitted_statistics['duration'] != '1M'].copy()
    fitted_statistics['duration'] = [str(dur_code) + 'H' for dur_code in fitted_statistics['duration']]

    # Write outputs
    if write_output:
        df = parameters[parameter_output_columns]  # TODO: Check that fixed parameters are present by this point
        utils.write_csv_(df, output_parameters_path, season_definitions)
        utils.write_statistics(
            fitted_statistics, output_statistics_path, season_definitions, write_weights=False, write_gs=False,
            write_phi_=False
        )

    return parameters, fitted_statistics


def fit_by_season(
        unique_seasons, reference_statistics, spatial_model, intensity_distribution, n_workers,
        all_parameter_names, parameters_to_fit, parameter_bounds, fixed_parameters, stage='final',
        initial_parameters=None, use_pooling=False,
):
    """
    Optimise parameters for each season independently.

    """
    results = {}
    fitted_statistics = []
    for season in unique_seasons:

        # TODO: Check that both rho and gamma are fixed if one of them is (move check to model.fit too perhaps)

        # Gather relevant data, weights and objective function scaling terms
        # - if using pooled statistics in a spatial model then first fit is just a point model
        season_ref_stats = reference_statistics.loc[reference_statistics['season'] == season].copy()
        if spatial_model:
            if use_pooling:
                season_ref_stats = season_ref_stats.loc[season_ref_stats['name'] != 'cross-correlation']
                if 'rho' in parameters_to_fit:
                    _spatial_model = False  # fit first as point model using nu
                else:
                    _spatial_model = True  # fit as spatial model using rho and gamma
            else:
                _spatial_model = True
        else:
            _spatial_model = False
        statistic_ids, fitting_data, ref, weights, gs = prepare(season_ref_stats)

        # Also need to replace rho/gamma in parameter lists and bounds with nu if using pooled statistics
        if spatial_model and use_pooling and ('rho' in parameters_to_fit):

            # Replace rho and gamma with nu in parameter name lists (order needed for fitting)
            _all_parameter_names = [pn for pn in all_parameter_names if pn not in ['rho', 'gamma']]
            _all_parameter_names.append('nu')
            _parameters_to_fit = [pn for pn in parameters_to_fit if pn not in ['rho', 'gamma']]
            _parameters_to_fit.append('nu')
            _fixed_parameters = fixed_parameters

            # Replace rho and gamma bounds with nu bounds (if they are being fitted)
            rho_idx = parameters_to_fit.index('rho')
            gamma_idx = parameters_to_fit.index('gamma')
            _parameter_bounds = []
            for idx in range(len(parameter_bounds[season])):
                if (idx != rho_idx) and (idx != gamma_idx):
                    _parameter_bounds.append(parameter_bounds[season][idx])
            rho_min = parameter_bounds[season][rho_idx][0]
            rho_max = parameter_bounds[season][rho_idx][1]
            gamma_min = parameter_bounds[season][gamma_idx][0]
            gamma_max = parameter_bounds[season][gamma_idx][1]
            nu_min = 2.0 * np.pi * rho_min / gamma_max ** 2.0
            nu_max = 2.0 * np.pi * rho_max / gamma_min ** 2.0
            _parameter_bounds.append((nu_min, nu_max))

        # If point model or not pooling then no changes needed
        else:
            _all_parameter_names = all_parameter_names
            _parameters_to_fit = parameters_to_fit
            _fixed_parameters = fixed_parameters
            _parameter_bounds = parameter_bounds[season]

        # Option for initial parameters estimate
        if initial_parameters is not None:
            x0 = initial_parameters[season]
        else:
            x0 = None

        # Run optimisation
        result = scipy.optimize.differential_evolution(
            func=fitting_wrapper,
            bounds=_parameter_bounds,
            args=(
                _spatial_model,
                intensity_distribution,
                statistic_ids,
                fitting_data,
                ref,
                weights,
                gs,
                _all_parameter_names,
                _parameters_to_fit,
                _fixed_parameters,
                season
            ),
            tol=0.001,
            updating='deferred',
            workers=n_workers,
            x0=x0,
        )

        # Store optimisation results for season
        for idx in range(len(_parameters_to_fit)):
            results[(_parameters_to_fit[idx], season)] = result.x[idx]
        results[('converged', season)] = result.success
        results[('objective_function', season)] = result.fun
        results[('iterations', season)] = result.nit
        results[('function_evaluations', season)] = result.nfev

        # Additional fit for rho and gamma if using pooled statistics with spatial model
        # - optimise rho and back-calculate gamma from rho and nu
        if spatial_model and use_pooling and ('rho' in parameters_to_fit):

            # Get reference statistics etc ready
            season_ref_stats = reference_statistics.loc[
                (reference_statistics['season'] == season) & (reference_statistics['name'] == 'cross-correlation')
            ].copy()
            statistic_ids, fitting_data, ref, weights, gs = prepare(season_ref_stats)

            # Parameter lists and bounds etc for rho
            _spatial_model = True
            _all_parameter_names = ['rho']
            _parameters_to_fit = ['rho']
            _fixed_parameters = {}
            rho_idx = parameters_to_fit.index('rho')
            _parameter_bounds = [parameter_bounds[season][rho_idx]]

            # Need to pass nu (as known) plus other parameters too (as basically all are needed for analytical
            # calculation of cross-correlations)
            nu = results[('nu', season)]
            if 'lamda' in parameters_to_fit:
                lamda = results[('lamda', season)]
            else:
                lamda = fixed_parameters[(season, 'lamda')]
            if 'beta' in parameters_to_fit:
                beta = results[('beta', season)]
            else:
                beta = fixed_parameters[(season, 'beta')]
            if 'eta' in parameters_to_fit:
                eta = results[('eta', season)]
            else:
                eta = fixed_parameters[(season, 'eta')]
            if 'theta' in parameters_to_fit:
                theta = results[('theta', season)]
            else:
                theta = fixed_parameters[(season, 'theta')]
            if intensity_distribution == 'weibull':
                if 'kappa' in parameters_to_fit:
                    kappa = results[('kappa', season)]
                else:
                    kappa = fixed_parameters[(season, 'kappa')]
            else:
                kappa = None

            # TODO: Sort out initial parameters
            x0 = None

            # Run optimisation
            result = scipy.optimize.differential_evolution(
                func=fitting_wrapper,
                bounds=_parameter_bounds,
                args=(
                    _spatial_model,
                    intensity_distribution,
                    statistic_ids,
                    fitting_data,
                    ref,
                    weights,
                    gs,
                    _all_parameter_names,
                    _parameters_to_fit,
                    _fixed_parameters,
                    season,
                    nu,
                    lamda,
                    beta,
                    eta,
                    theta,
                    kappa
                ),
                tol=0.001,
                updating='deferred',
                workers=n_workers,
                x0=x0,
            )

            # Store optimisation results for season
            # - add in gamma, as needs to be back-calculated from rho and nu
            for idx in range(len(_parameters_to_fit)):
                results[(_parameters_to_fit[idx], season)] = result.x[idx]
            results[('gamma', season)] = (2.0 * np.pi * results[('rho', season)] / results[('nu', season)]) ** 0.5

            # Combine optimisation information for two steps for now
            if result.success and results[('converged', season)]:
                pass
            else:
                results[('converged', season)] = False
            results[('objective_function', season)] += result.fun
            results[('iterations', season)] += result.nit
            results[('function_evaluations', season)] += result.nfev

            # TODO: Consider whether to drop nu from results dictionary - may be removed anyway in storage step below
            results.pop(('nu', season))

        # Get parameters into dictionary ready for formatting
        parameters_dict = {}
        for parameter_name in all_parameter_names:
            if parameter_name in parameters_to_fit:
                parameters_dict[parameter_name] = results[(parameter_name, season)]
            else:
                parameters_dict[parameter_name] = fixed_parameters[(season, parameter_name)]

        # Get and store statistics (all) associated with optimised parameters
        dfs = []
        statistic_ids, fitting_data, ref, weights, gs = prepare(
            reference_statistics.loc[reference_statistics['season'] == season]
        )
        mod_stats = calculate_analytical_properties(
            spatial_model, intensity_distribution, parameters_dict, statistic_ids, fitting_data
        )
        for statistic_id in statistic_ids:
            tmp = fitting_data[(statistic_id, 'df')].copy()
            dfs.append(tmp)
        df = pd.concat(dfs)
        df['value'] = mod_stats
        df['season'] = season
        fitted_statistics.append(df)

    # Format results for output
    parameters = format_results(results, all_parameter_names, parameters_to_fit, fixed_parameters, unique_seasons)
    fitted_statistics = pd.concat(fitted_statistics)
    parameters['fit_stage'] = stage
    fitted_statistics['fit_stage'] = stage

    return parameters, fitted_statistics


def format_results(results, all_parameter_names, parameters_to_fit, fixed_parameters, unique_seasons):
    # Insert fixed parameters into results dictionary
    dc = results.copy()
    for parameter_name in all_parameter_names:
        if parameter_name in parameters_to_fit:
            pass
        else:
            for season in unique_seasons:
                dc[(parameter_name, season)] = fixed_parameters[(season, parameter_name)]

    # Format as dataframe for output
    df = pd.DataFrame.from_dict(dc, orient='index', columns=['value'])
    df.index = pd.MultiIndex.from_tuples(df.index, names=['field', 'season'])
    df.reset_index(inplace=True)
    df = df.pivot(index='season', columns='field', values='value')
    df.sort_index(inplace=True)
    df.reset_index(inplace=True)

    # Types need to be set after unpivotting (mixed column type comes through as object dtype)
    for parameter_name in all_parameter_names:
        df = df.astype({parameter_name: float})
    df = df.astype({
        'season': int, 'converged': bool, 'function_evaluations': int, 'iterations': int,
        'objective_function': float
    })

    return df


def fitting_wrapper(
        parameters, spatial_model, intensity_distribution, statistic_ids, fitting_data, ref_stats, weights, gs,
        all_parameter_names, parameters_to_fit, fixed_parameters, season, nu=None, lamda=None, beta=None, eta=None,
        theta=None, kappa=None
):
    # List of parameters from optimisation can be converted to a dictionary for easier comprehension in analytical
    # property calculations. Fixed parameters can also be included
    parameters_dict = {}
    for parameter_name in all_parameter_names:
        if parameter_name in parameters_to_fit:
            parameters_dict[parameter_name] = parameters[parameters_to_fit.index(parameter_name)]
        else:
            parameters_dict[parameter_name] = fixed_parameters[(season, parameter_name)]

    # If nu is passed then assume that rho is being optimised and gamma should be back-calculated
    # - this will be the second step of fitting a spatial model when the first step is fitting a point model via nu,
    # i.e. typically using a pooled approach to spatial model fitting
    # - fixed parameters should not change in this case - empty dictionary
    # - also need then to add other parameters to dictionary for calculation of analytical properties
    if nu is not None:
        parameters_dict['gamma'] = (2 * np.pi * parameters[0] / nu) ** 0.5
        parameters_dict['lamda'] = lamda
        parameters_dict['beta'] = beta
        parameters_dict['eta'] = eta
        parameters_dict['theta'] = theta
        if intensity_distribution == 'weibull':
            parameters_dict['kappa'] = kappa

    # Calculate properties and objective function
    mod_stats = calculate_analytical_properties(
        spatial_model, intensity_distribution, parameters_dict, statistic_ids, fitting_data
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


def calculate_analytical_properties(
        spatial_model, intensity_distribution, parameters_dict, statistic_ids, fitting_data
):
    # Unpack parameter values common to point and spatial models
    lamda = parameters_dict['lamda']
    beta = parameters_dict['beta']
    eta = parameters_dict['eta']
    theta = parameters_dict['theta']

    # Get or calculate nu
    if not spatial_model:
        nu = parameters_dict['nu']
    else:
        rho = parameters_dict['rho']
        gamma = parameters_dict['gamma']
        nu = 2.0 * np.pi * rho / gamma ** 2.0

    # Shape parameters are only relevant to non-exponential intensity distributions
    if intensity_distribution == 'weibull':
        kappa = parameters_dict['kappa']
    elif intensity_distribution == 'generalised_gamma':
        kappa_1 = parameters_dict['kappa_1']
        kappa_2 = parameters_dict['kappa_2']

    # Calculate raw moments (1-3) of intensity distribution
    moments = []
    for n in [1, 2, 3]:
        if intensity_distribution == 'exponential':
            moments.append(scipy.stats.expon.moment(n, scale=theta))
        elif intensity_distribution == 'weibull':
            moments.append(scipy.stats.weibull_min.moment(n, c=kappa, scale=theta))
        elif intensity_distribution == 'generalised_gamma':
            moments.append(scipy.stats.gengamma.moment(n, a=(kappa_1 / kappa_2), c=kappa_2, scale=theta))
    mu_1, mu_2, mu_3 = moments

    # Main loop to get each required statistic
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


def _get_simulated_statistics(
        intensity_distribution, tmp_folder, season_definitions, parameters, phi, statistic_definitions, random_seed,
        spatial_model, point_metadata=None, write_statistics=False, output_folder=None,
):
    _parameters = parameters.copy()

    if spatial_model:
        _point_metadata = point_metadata.loc[point_metadata['point_id'] == point_metadata['point_id'].min()]

    else:
        _point_metadata = None

    simulation.main(
        spatial_model=spatial_model,
        intensity_distribution=intensity_distribution,
        output_types=['point'],
        output_folder=tmp_folder,
        output_subfolders=dict(point=''),
        output_format='txt',
        season_definitions=season_definitions,
        parameters=_parameters,
        point_metadata=_point_metadata,
        catchment_metadata=None,
        grid_metadata=None,
        epsg_code=None,
        cell_size=None,
        dem=None,
        phi=phi,
        simulation_length=10000,
        number_of_realisations=1,
        timestep_length=1,
        start_year=2000,
        calendar='gregorian',
        random_seed=random_seed,
        default_block_size=1000,
        check_block_size=True,
        minimum_block_size=10,
        check_available_memory=True,
        maximum_memory_percentage=75,
        block_subset_size=50,
        project_name='tmp',
        spatial_buffer_factor=15,
        simulation_mode='no_shuffling',
        weather_model=None,
        n_divisions=4,
        do_reordering=False,
    )

    _stat_defs = statistic_definitions.loc[statistic_definitions['name'] != 'cross-correlation']

    sim_stats, _ = analysis.main(
            spatial_model=spatial_model,
            season_definitions=season_definitions,
            statistic_definitions=_stat_defs,
            timeseries_format='txt',
            start_date=datetime.datetime(2000, 1, 1),
            timestep_length=1,
            calendar='gregorian',
            timeseries_path=None,
            timeseries_folder=tmp_folder,
            point_metadata=_point_metadata,
            calculation_period=None,
            completeness_threshold=0.0,
            output_statistics_path=None,
            outlier_method=None,
            maximum_relative_difference=None,
            maximum_alterations=None,
            analysis_mode='postprocessing',
            n_years=10000,
            n_realisations=1,
            subset_length=200,  # ??
            output_amax_path=None,
            amax_durations=None,
            amax_window_type=None,
            output_ddf_path=None,
            ddf_return_periods=None,
            write_output=False,
            simulation_name='tmp',
            use_pooling=False,
            calculate_statistics=True,
            dayfirst=False,
        )

    if write_statistics:
        output_path = os.path.join(output_folder, 'simulated_statistics_DEBUGGING.csv')
        if not os.path.exists(output_path):
            sim_stats.to_csv(output_path, index=False)

    # Remove temporary simulation files
    files_to_delete = os.listdir(tmp_folder)
    for fn in files_to_delete:
        fp = os.path.join(tmp_folder, fn)
        os.remove(fp)

    return sim_stats


def _prebias_reference_statistics(orig_ref_stats, curr_ref_stats, sim_stats, spatial_model, use_pooling, iteration):  # TESTING - just so can save each iteration for now
    # Prepare to merge
    curr_ref_stats.rename(columns={'value': 'curr_ref_value'}, inplace=True)
    sim_stats.rename(columns={'mean': 'sim_value'}, inplace=True)

    if spatial_model and use_pooling:
        sim_stats['point_id'] = -1

    # Merge simulated and reference statistics
    orig_ref_stats_sub = orig_ref_stats.loc[orig_ref_stats['name'] != 'cross-correlation']
    if spatial_model:
        new_ref_stats = orig_ref_stats_sub.merge(
            curr_ref_stats[['point_id', 'statistic_id', 'point_id2', 'season', 'curr_ref_value']], how='left'
        )
    else:
        new_ref_stats = orig_ref_stats_sub.merge(
            curr_ref_stats[['point_id', 'statistic_id', 'season', 'curr_ref_value']], how='left'
        )
    new_ref_stats = new_ref_stats.merge(
        sim_stats[['point_id', 'statistic_id', 'season', 'sim_value']], how='left',
    )

    # Bring cross-correlations back in and ensure order matches original
    if spatial_model:
        orig_ref_stats_sub = orig_ref_stats.loc[orig_ref_stats['name'] == 'cross-correlation']
        new_ref_stats = pd.concat([new_ref_stats, orig_ref_stats_sub])
    if spatial_model:
        new_ref_stats.sort_values(['statistic_id', 'point_id', 'season', 'distance'], inplace=True)
    else:
        new_ref_stats.sort_values(['statistic_id', 'point_id', 'season'], inplace=True)

    # Calculate current bias in reference statistics
    new_ref_stats['curr_bias'] = new_ref_stats['value'] - new_ref_stats['curr_ref_value']

    # Calculate required additional bias
    new_ref_stats['additional_bias'] = new_ref_stats['sim_value'] - new_ref_stats['value']

    # Add required increments to original reference values
    new_ref_stats['new_value'] = np.where(
        (new_ref_stats['name'] == 'probability_dry') & (new_ref_stats['duration'] == '24H'),
        new_ref_stats['value'] - new_ref_stats['curr_bias'] - new_ref_stats['additional_bias'],
        new_ref_stats['value']
    )
    new_ref_stats['new_value'] = np.where(
        (new_ref_stats['name'] == 'skewness') & (new_ref_stats['value'] > new_ref_stats['sim_value']),
        new_ref_stats['value'] - new_ref_stats['curr_bias'] - new_ref_stats['additional_bias'],
        new_ref_stats['new_value']
    )

    # Tidy up df for subsequent fitting
    new_ref_stats.drop(columns=['value', 'curr_ref_value', 'sim_value', 'curr_bias', 'additional_bias'], inplace=True)
    new_ref_stats.rename(columns={'new_value': 'value'}, inplace=True)

    return new_ref_stats
