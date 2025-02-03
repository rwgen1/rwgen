import os
import datetime

import numpy as np
import pandas as pd
import geopandas

from . import analysis
from . import fitting
from . import simulation
from . import utils
from . import plotting
from . import shuffling
from . import perturbation


class RainfallModel:
    """
    Neyman-Scott Rectangular Pulse (NSRP) rainfall model for point/site and spatial simulations.

    Args:
        spatial_model (bool): Flag to indicate whether point or spatial model.
        project_name (str): A name for the gauge/site location, domain or catchment.
        input_timeseries (str): Path to file containing timeseries data (for point model) or folder containing
            timeseries data files (for spatial model). Needed if running pre-processing or fitting steps.
        point_metadata (pandas.DataFrame or str): Metadata (or path to metadata file) on point (site/gauge)
            locations to use for fitting, simulation and/or evaluation for a spatial model only. See Notes for
            details.
        season_definitions (str, list or dict): The model works on a monthly basis by default, but this argument
            allows for user-defined seasons (see Notes).
        intensity_distribution (str): Flag to indicate the type of probability distribution for raincell
            intensities. Defaults to ``'exponential'`` (with ``'weibull'`` also available currently).
        output_folder (str): Root folder for model output. Defaults to ``'./output'``.
        statistic_definitions (pandas.DataFrame or str): Definitions (descriptions) of statistics to use in fitting
            and/or evaluation (or path to file of definitions). See Notes for explanation of format.
        easting (int or float): Easting of gauge/site for a point simulation. Required if planning to simulate other
            variables (temperature, PET, ...) or apply climate change perturbations via the ``perturb_statistics()``
            method. Not required for a spatial model.
        northing (int or float): Northing of gauge/site for a point simulation. See easting entry above.
        elevation (int or float): Elevation of gauge/site for a point simulation. See easting entry above.

    Notes:
        Seasons can be specified through the ``season_definitions`` argument in several ways:

         * As descriptive strings (``'monthly'``, ``'quarterly'``, ``'half-years'``, ``'annual'``). More
           control can be gained using e.g. ``'quarterly_dec'`` to make Dec-Jan-Feb the first season and so on.
           Specifying annual will lead to the whole year being considered together, i.e. no seasonality.
         * As a list of strings indicating season abbreviations, e.g. ``['DJF', 'MAM', 'JJA', 'SON']``.
         * As a dictionary whose keys are the months of the year (integers 1-12) and whose values represent a
           season identifier, e.g. ``dict(12=1, 1=1, 2=1, 3=2, 4=2, 5=2, 6=3, 7=3, 8=3, 9=4, 10=4, 11=4)`` would
           give quarterly seasons beginning in December.

        Statistic definitions are required primarily for model fitting and evaluation. The default
        ``statistic_definitions`` are taken largely from RainSim V3.1 and can be changed with a similarly
        structured ``.csv`` file. Note that:

         * Thresholds can be specified for ``probability_dry`` for 1hr (``0.1mm`` or ``0.2mm``) and 24hr
           (``0.2mm`` or ``1.0mm``) durations.
         * Lag can be specified for ``autocorrelation`` and ``cross-correlation`` (if not specified then
           defaults of 1 and 0 will be used, respectively).

        For a point model the default ``statistic_definitions`` are:

        ============  =====================  ========  ======
        Statistic_ID  Name                   Duration  Weight
        ============  =====================  ========  ======
        1             variance               1H        1
        2             skewness               1H        2
        3             probability_dry_0.2mm  1H        7
        4             mean                   24H       6
        5             variance               24H       2
        6             skewness               24H       3
        7             probability_dry_0.2mm  24H       7
        8             autocorrelation_lag1   24H       6
        9             variance               72H       3
        ============  =====================  ========  ======

        For a spatial model the default ``statistic_definitions`` are:

        ============  ======================  ========  ======
        Statistic_ID  Name                    Duration  Weight
        ============  ======================  ========  ======
        1             variance                1H        3
        2             skewness                1H        3
        3             probability_dry_0.2mm   1H        5
        4             mean                    24H       5
        5             variance                24H       2
        6             skewness                24H       2
        7             probability_dry_0.2mm   24H       6
        8             autocorrelation_lag1    24H       3
        9             cross-correlation_lag0  24H       2
        10            variance                72H       3
        ============  ======================  ========  ======

        For a spatial model, metadata for point (gauge/site) locations to be used in fitting, simulation or
        evaluation must be specified through the ``point_metadata`` argument. This should be a table like the one
        below (or a path to a ``.csv`` file containing such a table):

        ========  =======  ========  ============  =========
        Point_ID  Easting  Northing  Name          Elevation
        ========  =======  ========  ============  =========
        1         659493   5556905   Burgkunstadt  277.5
        2         640130   5574573   Lautertal     343.79
        3         688073   5524669   Creussen      449.5
        ========  =======  ========  ============  =========

        ``Elevation`` is good to include if available.

    """

    def __init__(
            self,
            spatial_model,
            project_name,  # TODO: Check whether needed now that variable name is used in output files (for WG)
            input_timeseries=None,
            point_metadata=None,  # TODO: Compulsory or not for rainfall model? - ideally not
            season_definitions='monthly',
            intensity_distribution='exponential',
            output_folder='./output',
            statistic_definitions=None,
            easting=None,
            northing=None,
            elevation=None,
    ):
        print('Rainfall model initialisation')

        # Set key options/attributes needed throughout model
        self.season_definitions = utils.parse_season_definitions(season_definitions)
        self.spatial_model = spatial_model
        self.intensity_distribution = intensity_distribution
        self.output_folder = output_folder
        self.project_name = project_name
        self.input_timeseries = input_timeseries

        # Spatial model requires a table of metadata for points
        # if self.spatial_model:
        if isinstance(point_metadata, pd.DataFrame):
            self.point_metadata = point_metadata
            self.point_metadata.columns = [name.lower() for name in self.point_metadata.columns]
            # TODO: Other dataframe arguments need to be converted to lowercase column headings (e.g. statistic defs...)
        elif isinstance(point_metadata, str):
            self.point_metadata = utils.read_csv_(point_metadata)
        else:
            if not spatial_model and (easting is not None):
                self.point_metadata = pd.DataFrame(dict(
                    point_id=[1], easting=[easting], northing=[northing], name=[project_name], elevation=[elevation],
                ))
            else:
                self.point_metadata = None

        # Default statistic definitions (and weights) are taken largely from RainSim V3.1 documentation
        if isinstance(statistic_definitions, pd.DataFrame):
            self.statistic_definitions = statistic_definitions
        elif isinstance(statistic_definitions, str):
            self.statistic_definitions = utils.read_statistics(statistic_definitions)
        else:
            if self.spatial_model:
                dc = {
                    1: {'weight': 3.0, 'duration': '1H', 'name': 'variance'},
                    2: {'weight': 3.0, 'duration': '1H', 'name': 'skewness'},
                    3: {'weight': 5.0, 'duration': '1H', 'name': 'probability_dry', 'threshold': 0.2},
                    4: {'weight': 5.0, 'duration': '24H', 'name': 'mean'},
                    5: {'weight': 2.0, 'duration': '24H', 'name': 'variance'},
                    6: {'weight': 2.0, 'duration': '24H', 'name': 'skewness'},
                    7: {'weight': 6.0, 'duration': '24H', 'name': 'probability_dry', 'threshold': 0.2},
                    8: {'weight': 3.0, 'duration': '24H', 'name': 'autocorrelation', 'lag': 1},
                    9: {'weight': 2.0, 'duration': '24H', 'name': 'cross-correlation', 'lag': 0},
                    10: {'weight': 3.0, 'duration': '72H', 'name': 'variance'},
                    11: {'weight': 0.0, 'duration': '1M', 'name': 'variance'},
                }
            else:
                dc = {
                    1: {'weight': 1.0, 'duration': '1H', 'name': 'variance'},
                    2: {'weight': 2.0, 'duration': '1H', 'name': 'skewness'},
                    3: {'weight': 7.0, 'duration': '1H', 'name': 'probability_dry', 'threshold': 0.2},
                    4: {'weight': 6.0, 'duration': '24H', 'name': 'mean'},
                    5: {'weight': 2.0, 'duration': '24H', 'name': 'variance'},
                    6: {'weight': 3.0, 'duration': '24H', 'name': 'skewness'},
                    7: {'weight': 7.0, 'duration': '24H', 'name': 'probability_dry', 'threshold': 0.2},
                    8: {'weight': 6.0, 'duration': '24H', 'name': 'autocorrelation', 'lag': 1},
                    9: {'weight': 3.0, 'duration': '72H', 'name': 'variance'},
                    10: {'weight': 0.0, 'duration': '1M', 'name': 'variance'},
                }
            id_name = 'statistic_id'
            non_id_columns = ['name', 'duration', 'lag', 'threshold', 'weight']
            self.statistic_definitions = utils.nested_dictionary_to_dataframe(dc, id_name, non_id_columns)

        # Check that statistics include 24hr mean, as it is currently required for calculating phi (add in if absent)
        includes_24hr_mean = self.statistic_definitions.loc[
            (self.statistic_definitions['name'] == 'mean')
            & ((self.statistic_definitions['duration'] == '24H') | (self.statistic_definitions['duration'] == '1D'))
            ].shape[0]
        if not includes_24hr_mean:
            df = pd.DataFrame({
                'statistic_id': [int(np.max(statistic_definitions['statistic_id'])) + 1], 'weight': [0],
                'duration': ['24H'], 'name': ['mean'], 'lag': ['NA'], 'threshold': ['NA']
            })
            self.statistic_definitions = pd.concat([statistic_definitions, df])

        # Default configuration settings for simulation (primarily memory management) - see
        # ``update_simulation_config()`` method docstring
        self.simulation_config = self.update_simulation_config()

        # For sharing simulation arguments with post-processing method
        self.simulation_args = None

        # Calculated during model use and relevant across more than one method

        #: pandas.DataFrame: Statistics to use as reference in fitting, simulation and/or evaluation
        self.reference_statistics = None

        self.phi = None  # for convenience - same information as in self.reference_statistics

        #: pandas.DataFrame: Parameters to use in simulation
        self.parameters = None

        #: pandas.DataFrame: Statistics of the NSRP process given current parameters
        self.fitted_statistics = None

        #: pandas.DataFrame: Statistics of simulated time series
        self.simulated_statistics = None

        # Default configuration settings for shuffling - see ``update_shuffling_config()`` method docstring
        self.shuffling_config = self.update_shuffling_config()

        #: list: Start year and end year of record to use in calculations based on reference time series
        self.calculation_period = None

        print('  - Completed')

    def preprocess(
            self,
            calculation_period='full_record',
            completeness_threshold=0.0,
            outlier_method=None,
            maximum_relative_difference=2.0,
            maximum_alterations=5,
            amax_durations=None,
            amax_window_type='sliding',
            output_filenames='default',
            use_pooling=True,
            dayfirst=False,
    ):
        """
        Prepare reference statistics, weights and scale factors for use in model fitting, simulation and evaluation.

        Updates ``self.reference_statistics`` and ``self.phi`` attributes and writes a ``reference_statistics`` output
        file.

        Args:
            calculation_period (str or list of int): Start year and end year of calculation period as list. If
                string ``'full_record'`` is passed (default) then all available data will be used.
            completeness_threshold (float): Percentage completeness for a month or season to be included in statistics
                calculations. Default is 0.0, i.e. any completeness (or missing data) percentage is acceptable.
            outlier_method (str): Flag indicating which (if any) method should be to reduce the influence of outliers.
                Options are None (default), ``'trim'`` (remove outliers) or ``'clip'`` (Winsorise). See Notes.
            maximum_relative_difference (float): Maximum relative difference to allow between the two largest values
                in a timeseries. Used only if ``outlier_method`` is not None.
            maximum_alterations (int): Maximum number of trimming or clipping alterations permitted. Used only if
                ``outlier_method`` is not None.
            amax_durations (int or list of int): Durations (in hours) for which annual maxima (AMAX) should be
                identified (default is None).
            amax_window_type (str): Use a ``'sliding'`` (default) or ``'fixed'`` window in AMAX extraction.
            output_filenames (str or dict): Either key/value pairs indicating output file names, ``'default'`` to use
                {'statistics': 'reference_statistics.csv', 'amax': 'reference_amax.csv'} or ``None`` to indicate that
                no output files should be written.
            use_pooling (bool): Indicates whether to pool (scaled) point series for calculating statistics for
                a spatial model. If True (default), cross-correlations are also "averaged" for a set of separation
                distance bins.
            dayfirst (bool). Whether dates are formatted as dd/mm/yyyy[ hh:mm]. Default False (i.e. yyyy-mm-dd hh:mm).

        Notes:
            Currently ``.csv`` files are used for time series inputs. These files are expected to contain a
            ``DateTime`` column using ``dd/mm/yyyy hh:mm`` format ('%d/%m/%Y %H:%M') or ``yyyy-mm-dd hh:mm``
            ('%Y-%m-%d %H:%M'). They should also contain a ``Value`` column using units of mm/timestep.

        """
        print('Rainfall model preprocessing')

        # Infer timeseries data format
        if not self.spatial_model:
            timeseries_format = self.input_timeseries.split('.')[-1]
        else:
            file_names = os.listdir(self.input_timeseries)
            if 'file_name' in self.point_metadata.columns:
                test_file = self.point_metadata['file_name'].values[0]
                test_file, _ = test_file.split('.')
            else:
                test_file = self.point_metadata['name'].values[0]
            for file_name in file_names:
                file_name, extension = file_name.split('.')
                if file_name == test_file:
                    timeseries_format = extension
                    break

        # Input paths as required by analysis function
        if self.spatial_model:
            timeseries_path = None
            timeseries_folder = self.input_timeseries
        else:
            timeseries_path = self.input_timeseries
            timeseries_folder = None

        # Construct output paths
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if output_filenames == 'default':
            output_statistics_filename = 'reference_statistics.csv'
            output_amax_filename = 'reference_amax.csv'
        else:
            if 'statistics' in output_filenames:
                output_statistics_filename = output_filenames['statistics']
            if 'amax' in output_filenames:
                output_amax_filename = output_filenames['amax']
        output_statistics_path = os.path.join(self.output_folder, output_statistics_filename)
        if amax_durations is not None:
            output_amax_path = os.path.join(self.output_folder, output_amax_filename)
        else:
            output_amax_path = None

        # Check (partially) that arguments are suitable for analysis call
        if amax_durations is not None:
            if not isinstance(amax_durations, list):
                amax_durations = [amax_durations]
        if output_filenames is None:
            write_output = False
        else:
            write_output = True
        if calculation_period == 'full_record':
            self.calculation_period = None
        else:
            self.calculation_period = calculation_period

        # AMAX durations using resample codes
        # _amax_durations = [str(dur) + 'H' for dur in amax_durations]

        # Do preprocessing
        self.reference_statistics, self.phi = analysis.main(
            spatial_model=self.spatial_model,
            season_definitions=self.season_definitions,
            statistic_definitions=self.statistic_definitions,
            timeseries_format=timeseries_format,
            start_date=None,
            timestep_length=None,
            calendar=None,
            timeseries_path=timeseries_path,
            timeseries_folder=timeseries_folder,
            point_metadata=self.point_metadata,
            calculation_period=self.calculation_period,
            completeness_threshold=completeness_threshold,
            output_statistics_path=output_statistics_path,
            outlier_method=outlier_method,
            maximum_relative_difference=maximum_relative_difference,
            maximum_alterations=maximum_alterations,
            analysis_mode='preprocessing',
            n_years=None,
            n_realisations=1,
            subset_length=None,
            output_amax_path=output_amax_path,
            amax_durations=amax_durations,
            amax_window_type=amax_window_type,
            output_ddf_path=None,
            ddf_return_periods=None,
            write_output=write_output,
            simulation_name=None,
            use_pooling=use_pooling,
            calculate_statistics=True,
            dayfirst=dayfirst,
        )

        print('  - Completed')

    def fit(
            self,
            fitting_method='default',
            parameter_bounds=None,
            fixed_parameters=None,
            n_workers=1,
            output_filenames='default',
            fit_nsrp=True,
            fit_shuffling=False,
            random_seed=None,
            pdry_iterations=2,
            use_pooling=True,
    ):
        """
        Fit model parameters.

        Depends on ``self.reference_statistics`` attribute. Sets ``self.parameters`` and ``self.fitted_statistics``.
        Also writes ``parameters`` and `fitted_statistics`` output files.

        Args:
            fitting_method (str): Flag to indicate fitting method. Using ``'default'`` will fit each month or season
                independently. Other options under development.
            parameter_bounds (dict or str or pandas.DataFrame): Dictionary containing tuples of upper and lower
                parameter bounds by parameter name. Alternatively the path to a parameter bounds file or an equivalent
                dataframe (see Notes).
            fixed_parameters (dict or str or pandas.DataFrame): Dictionary containing fixed parameter values by
                parameter name. Alternatively the path to a parameters file or an equivalent dataframe (see Notes).
            n_workers (int): Number of workers (cores/processes) to use in fitting. Default is 1.
            output_filenames (str or dict): Either key/value pairs indicating output file names, ``'default'`` to use
                {'statistics': 'fitted_statistics.csv', 'parameters': 'parameters.csv'} or ``None`` to indicate that
                no output files should be written.
            fit_nsrp (bool): Indicates whether to fit NSRP parameters.
            fit_shuffling (bool): Indicates whether to fit the "delta" parameter that controls the probability of
                selecting more/less similar storms during shuffling, as well the parameters of the periodic
                monthly AR1 model.
            random_seed (int or numpy.random.SeedSequence): For reproducibility in fitting (currently for delta only).
            pdry_iterations (int): Number of iterations to use to correct bias between fitted (analytical) dry
                probability and simulated dry probability. Default is 2.
            use_pooling (bool): Indicates whether to used pooled statistics in NSRP fitting for a spatial model.

        Notes:
            The parameters used by the model are:

                * lamda - reciprocal of the mean waiting time between adjacent storm origins [h-1]
                * beta - reciprocal of the mean waiting time for raincell origins after storm origin [h-1]
                * eta - reciprocal of the mean duration of raincells [h-1]
                * nu - mean number of raincells per storm (specified for point model only) [-]
                * theta - mean intensity of raincells [h mm-1]
                * gamma - reciprocal of mean radius of raincells (spatial model) [km-1]
                * rho - spatial density of raincell centres (spatial model) [km-2]

            Note also that:

                * For a spatial model, the mean number of raincells overlapping a given location is related to rho
                  and gamma, such that nu can be inferred.
                * If using ``intensity_distribution='weibull'``, theta is the scale parameter and an additional
                  parameter (kappa) is introduced as the shape parameter.

            The ``parameter_bounds`` argument can be specified by a dictionary like ``dict(beta=(0.02, 1.0))``. For
            more control a dataframe (or ``.csv`` file) can be passed. For example, if a model has two (6-month)
            seasons (using arbitrary example numbers):

            ======  =========  ===========  ===========
            Season  Parameter  Lower_Bound  Upper_Bound
            ======  =========  ===========  ===========
            1       Beta       0.02         0.1
            2       Beta       0.1          1.0
            ======  =========  ===========  ===========

            If a parameter(s) should be fixed across all seasons then it can be set as e.g.
            ``dict(beta=0.1, theta=1)``. Otherwise a table can be provided like:

            ======  =====  ====
            Season  Lamda  Beta
            ======  =====  ====
            1       0.015  0.05
            2       0.012  0.04
            ======  =====  ====

            Note that if a parameter is not found in the ``parameter_bounds`` argument then default bounds will be
            used. Similarly, if it is not found in ``fixed_parameters`` it is assumed that the parameter must be
            fitted.

            The current default parameter values for a point (gauge/site) model are  below. ``-1`` indicates that they
            are applied across all months/seasons). The values are largely from the RainSim V3.1 documentation:

            ======  =========  ===========  ===========
            Season  Parameter  Lower_Bound  Upper_Bound
            ======  =========  ===========  ===========
            -1      lamda      0.00001      0.02
            -1      beta       0.02         1.0
            -1      nu         0.1          30.0
            -1      eta        0.1          60.0
            -1      theta      0.25         100.0
            -1      kappa      0.5          1.0
            ======  =========  ===========  ===========

            And for a spatial model:

            ======  =========  ===========  ===========
            Season  Parameter  Lower_Bound  Upper_Bound
            ======  =========  ===========  ===========
            -1      lamda      0.001        0.05
            -1      beta       0.02         0.5
            -1      rho        0.0001       0.05
            -1      eta        0.1          12.0
            -1      gamma      0.01         500.0
            -1      theta      0.25         100.0
            -1      kappa      0.5          1.0
            ======  =========  ===========  ===========

            Fitting can be speeded up significantly with ``n_workers > 1``. The maximum ``n_workers`` should be less
            than or equal to the number of cores or logical processors available.

        """
        # TODO: Check upper bound for rho in spatial model - originally 2.0
        print('Rainfall model fitting')

        # Get parameter bounds into a dataframe
        if isinstance(parameter_bounds, dict):
            dc = {key.lower(): value for key, value in parameter_bounds.items()}
            parameter_bounds = pd.DataFrame.from_dict(dc, orient='index', columns=['lower_bound', 'upper_bound'])
            parameter_bounds.reset_index(inplace=True)
            parameter_bounds.rename(columns={'index': 'parameter'}, inplace=True)
            parameter_bounds['season'] = -1
            parameter_bounds.loc[parameter_bounds['parameter'] == 'lambda', 'parameter'] = 'lamda'
        elif isinstance(parameter_bounds, str):
            parameter_bounds = pd.read_csv(parameter_bounds)

        # Get fixed parameters into a dataframe
        if isinstance(fixed_parameters, dict):
            dc = {key.lower(): value for key, value in fixed_parameters.items()}
            dc1 = {}
            for key, value in dc.items():
                if isinstance(value, list):
                    dc1[key] = value
                else:
                    dc1[key] = [value]
            fixed_parameters = pd.DataFrame.from_dict(dc1)
            fixed_parameters['season'] = -1
            fixed_parameters.rename(columns={'lambda': 'lamda'}, inplace=True)
        elif isinstance(fixed_parameters, str):
            fixed_parameters = pd.read_csv(fixed_parameters)

        # Default bounds if required
        if not self.spatial_model:
            default_bounds = pd.DataFrame.from_dict({
                'lamda': (0.00001, 0.02), 'beta': (0.02, 1.0), 'nu': (0.1, 30), 'eta': (0.1, 60.0),  # 'xi': (0.01, 4.0)
                'theta': (0.25, 100), 'kappa': (0.5, 1.0), 'kappa_1': (0.5, 1.0), 'kappa_2': (0.5, 1.0)
            }, orient='index', columns=['lower_bound', 'upper_bound'])
        else:
            default_bounds = pd.DataFrame.from_dict({  # TODO: Check rho - originally 2.0
                'lamda': (0.001, 0.05), 'beta': (0.02, 0.5), 'rho': (0.0001, 0.05), 'eta': (0.1, 12.0),
                'gamma': (0.01, 500.0),
                'theta': (0.25, 100), 'kappa': (0.5, 1.0), 'kappa_1': (0.5, 1.0), 'kappa_2': (0.5, 1.0)
            }, orient='index', columns=['lower_bound', 'upper_bound'])
        default_bounds.reset_index(inplace=True)
        default_bounds.rename(columns={'index': 'parameter'}, inplace=True)

        # Identify parameters to fit (or not fit) and set parameter bounds
        parameters_to_fit, fixed_parameters, parameter_bounds = utils.define_parameter_bounds(
            parameter_bounds, fixed_parameters, self.parameter_names, default_bounds, self.unique_seasons
        )

        # Construct output paths
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if output_filenames == 'default':
            output_parameters_filename = 'parameters.csv'
            output_statistics_filename = 'fitted_statistics.csv'
        else:
            if 'parameters' in output_filenames:
                output_parameters_filename = output_filenames['parameters']
            if 'statistics' in output_filenames:
                output_statistics_filename = output_filenames['statistics']
        output_parameters_path = os.path.join(self.output_folder, output_parameters_filename)
        output_statistics_path = os.path.join(self.output_folder, output_statistics_filename)
        if output_filenames is None:
            write_output = False
        else:
            write_output = True

        if random_seed is None:
            rng = np.random.default_rng()
            random_seed = rng.integers(1000000, 1000000000)

        # Do fitting
        if fit_nsrp:
            self.parameters, self.fitted_statistics = fitting.main(
                season_definitions=self.season_definitions,
                spatial_model=self.spatial_model,
                intensity_distribution=self.intensity_distribution,
                fitting_method=fitting_method,
                reference_statistics=self.reference_statistics.loc[
                    self.reference_statistics['duration'] != self.shuffling_config['target_duration']
                ],  # TODO: This subset may no longer be needed, as all weights except nsrp can potentially go to zero
                all_parameter_names=self.parameter_names,  # RENAMED
                parameters_to_fit=parameters_to_fit,  # NEW
                parameter_bounds=parameter_bounds,  # SAME
                fixed_parameters=fixed_parameters,  # NEW
                n_workers=n_workers,
                output_parameters_path=output_parameters_path,
                output_statistics_path=output_statistics_path,
                write_output=write_output,  # NEW
                # !221123 - for pre-biasing
                n_iterations=pdry_iterations,
                output_folder=self.output_folder,
                point_metadata=self.point_metadata,
                phi=self.phi,
                statistic_definitions=self.statistic_definitions,
                random_seed=random_seed,
                use_pooling=use_pooling,
            )

        if fit_shuffling:
            ar1_parameters = shuffling.fit_ar1(
                self.spatial_model, self.input_timeseries, self.point_metadata, self.calculation_period,
                self.reference_statistics,
            )
            delta = shuffling.fit_delta(
                spatial_model=self.spatial_model,
                parameters=self.parameters,
                intensity_distribution=self.intensity_distribution,
                point_metadata=self.point_metadata,
                n_workers=n_workers,
                random_seed=random_seed,
                reference_statistics=self.reference_statistics,
                reference_duration=self.shuffling_config['target_duration'],
                n_divisions=self.shuffling_config['month_divisions'],
                use_pooling=use_pooling,
            )
            delta = delta.merge(ar1_parameters)
            delta = delta[['season', 'delta', 'ar1_slope', 'ar1_intercept', 'ar1_stderr']]
            self.parameters = pd.merge(self.parameters, delta)

            # TODO: Sort out how to write out delta - merge into fit method?
            lines = []
            with open(output_parameters_path, 'r') as fh:
                for line in fh:
                    lines.append(line.rstrip())
            with open(output_parameters_path, 'w') as fh:
                i = 0
                for line in lines:
                    if i == 0:
                        # line = line + ',Delta\n'  # Max_DSL,
                        line = line + ',Delta,AR1_Slope,AR1_Intercept,AR1_STDERR\n'
                    else:
                        line = (
                            line
                            # + ',' + str(self.parameters['max_dsl'].values[i-1])
                            + ',' + str(self.parameters['delta'].values[i - 1])
                            + ',' + str(self.parameters['ar1_slope'].values[i - 1])
                            + ',' + str(self.parameters['ar1_intercept'].values[i - 1])
                            + ',' + str(self.parameters['ar1_stderr'].values[i - 1])
                            + '\n'
                        )
                    fh.write(line)
                    i += 1

        print('  - Completed')

    def simulate(
            self,
            output_types='point',
            output_subfolders='default',
            output_format='txt',
            catchment_metadata=None,
            grid_metadata=None,
            epsg_code=None,
            cell_size=None,
            dem=None,
            simulation_length=30,
            n_realisations=1,
            timestep_length=1,
            start_year=2000,
            calendar='gregorian',
            random_seed=None,
            run_simulation=True,
            apply_shuffling=False,
            weather_model=None,
    ):
        """
        Simulate stochastic time series realisation(s) of NSRP process.

        Args:
            output_types (str or list of str): Types of output (discretised) rainfall required. Options are ``'point'``,
                ``'catchment'`` and ``'grid'``.
            output_subfolders (str or dict): Sub-folder in which to place each output type. If ``'default'`` then
                ``dict(point='point', catchment='catchment', grid='grid')`` is used for a spatial model and
                ``dict(point='')`` for a point model (i.e. output to ``self.output_folder``). If None then all output
                files are written to ``self.output_folder``.
            output_format (str): Flag indicating output file format for point and catchment output. Current
                option is ``txt``. Gridded output will be written in NetCDF format.
            catchment_metadata (geopandas.GeoDataFrame or str): Geodataframe containing catchments for which output is
                required (or path to catchments shapefile). Optional.
            grid_metadata (dict or str): Specification of output grid to use for both gridded output (optional). This
                grid is also used to support catchment output. Dictionary keys use ascii raster header keywords, e.g.
                ``dict(ncols=10, nrow=10, ...)``. Use ``xllcorner`` and ``yllcorner``, as well as lowercase for each
                keyword. If None then a grid is defined to encompass catchment locations using the ``cell_size``
                argument. The path to an ascii raster file to use as a template for the grid can be given instead.
            epsg_code (int): EPSG code for projected coordinate system used for domain (required if catchment or
                grid output is requested).
            cell_size (float): Cell size to use if grid is None but a grid is needed for gridded output and/or catchment
                output.
            dem (xarray.DataArray or str): Digital elevation model (DEM) [m] as data array or ascii raster file path.
                Optional but recommended.
            simulation_length (int): Number of years to simulate in one realisation (minimum of 1).
            n_realisations (int): Number of realisations to simulate.
            timestep_length (int): Timestep of output [hr]. Default is 1 (hour).
            start_year (int): Start year of simulation.
            calendar (str): Flag to indicate whether ``gregorian`` (default accounting for leap years) or ``365-day``
                calendar should be used.
            random_seed (int): Seed to use in random number generation.
            run_simulation (bool): Flag for whether to run simulation. Setting to False may be used to update
                ``self.simulation_args`` to allow ``self.postprocess()`` to be run without ``self.simulate()``
                having been run first (i.e. reading from existing simulation output files).
            apply_shuffling (bool): Indicates whether to run model with or without shuffling following Kim and Onof
                (2020) method.  # TODO: Provide explanation of method
            weather_model (object): Instance of WeatherModel.  # TODO: Expand explanation

        Notes:
            Though gridded output is calculated (if ``output_types`` includes ``'grid'``) it is not yet available to
            write (under development).

            All locations in ``self.point_metadata`` will be written as output currently.  # TODO: To be changed

            Catchment shapefile or geodataframe assumed to have an ``ID`` field that can be used to identify catchments.
            Both point and catchment metadata are assumed to have a ``Name`` field for use as a prefix in file naming.

            Updates ``self.simulation_args`` in preparation for post-processing.

        """
        # Make output folders if required
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if output_subfolders == 'default':
            if self.spatial_model:
                output_subfolders = dict(point='point', catchment='catchment', grid='grid')
            else:
                output_subfolders = dict(point='')
        if isinstance(output_types, str):
            output_types = [output_types]
        if isinstance(output_subfolders, dict):
            for output_type, output_subfolder in output_subfolders.items():
                if output_type in output_types:
                    if not os.path.exists(os.path.join(self.output_folder, output_subfolder)):
                        os.mkdir(os.path.join(self.output_folder, output_subfolder))

        # Ensure valid output types
        if not self.spatial_model:
            output_types = ['point']

        # Get DEM if required
        if isinstance(dem, str):
            dem = utils.read_ascii_raster(dem)

        # Output location details (grid must be defined or derived for catchment output)
        if self.spatial_model:
            if 'catchment' in output_types:
                if isinstance(catchment_metadata, str):
                    catchment_metadata = geopandas.read_file(catchment_metadata)
                    catchment_metadata.columns = [column_name.lower() for column_name in catchment_metadata.columns]
            if ('grid' in output_types) or ('catchment' in output_types):
                if isinstance(grid_metadata, str):
                    grid_metadata = utils.grid_definition_from_ascii(grid_metadata)
                else:
                    grid_metadata = utils.define_grid_extent(catchment_metadata, cell_size, dem)
                cell_size = grid_metadata['cellsize']

        # Check simulation length is long enough
        simulation_length = max(simulation_length, 1)

        # Give a default name for output if needed
        if self.project_name is None:
            output_name = 'simulation'
        else:
            output_name = self.project_name  # TODO: Sort out inclusion of variable name...

        # Ensure only "final" parameters are used in simulation (in case intermediate parameters were recorded during
        # fitting)
        if 'fit_stage' in self.parameters.columns:
            parameters = self.parameters.loc[self.parameters['fit_stage'] == 'final']
        else:
            parameters = self.parameters

        # Capture simulation arguments for use in post-processing
        self.simulation_args = dict(
            simulation_format=output_format,
            start_year=start_year,
            timestep_length=timestep_length,
            calendar=calendar,
            simulation_subfolders=output_subfolders,
            simulation_length=simulation_length,
            n_realisations=n_realisations,
            simulation_name=output_name,
        )

        # Get input timeseries data file/folder paths to help with shuffling
        if self.spatial_model:
            timeseries_path = None
            timeseries_folder = self.input_timeseries
        else:
            timeseries_path = self.input_timeseries
            timeseries_folder = None

        if apply_shuffling:
            simulation_mode = 'with_shuffling'
        else:
            simulation_mode = 'no_shuffling'

        # Stop here if only the arguments needed for post-processing (of existing files) are required
        if run_simulation:
            print('Simulating')
            simulation.main(
                spatial_model=self.spatial_model,
                intensity_distribution=self.intensity_distribution,
                output_types=output_types,
                output_folder=self.output_folder,
                output_subfolders=output_subfolders,
                output_format=output_format,
                season_definitions=self.season_definitions,
                parameters=parameters,
                point_metadata=self.point_metadata,
                catchment_metadata=catchment_metadata,
                grid_metadata=grid_metadata,
                epsg_code=epsg_code,
                cell_size=cell_size,
                dem=dem,
                phi=self.phi,
                simulation_length=simulation_length,
                number_of_realisations=n_realisations,
                timestep_length=timestep_length,
                start_year=start_year,
                calendar=calendar,
                random_seed=random_seed,
                default_block_size=self.simulation_config['default_block_size'],
                check_block_size=self.simulation_config['check_block_size'],
                minimum_block_size=self.simulation_config['minimum_block_size'],
                check_available_memory=self.simulation_config['check_available_memory'],
                maximum_memory_percentage=self.simulation_config['maximum_memory_percentage'],
                block_subset_size=self.simulation_config['block_subset_size'],
                project_name=output_name,
                spatial_buffer_factor=self.simulation_config['spatial_buffer_factor'],
                simulation_mode=simulation_mode,
                weather_model=weather_model,
                n_divisions=self.shuffling_config['month_divisions'],
                do_reordering=self.shuffling_config['reorder_months'],
            )

            print('  - Completed')

    def postprocess(
            self,
            amax_durations=None,
            ddf_return_periods=None,
            amax_window_type='sliding',
            subset_length=200,  # 50,
            output_filenames='default',
            calculate_statistics=True,
            simulation_format=None,
            start_year=None,
            timestep_length=None,
            calendar=None,
            simulation_subfolders=None,
            simulation_length=None,
            n_realisations=None,
            # n_workers=1,  # TODO: Define n_workers in __init__()
    ):
        """
        Post-processing to calculate statistics from simulated point output.

        Calculates statistics given in ``self.statistic_definitions`` by default, with options to extract annual
        maxima (AMAX) at multiple duration and to estimate depth-duration-frequency (DDF) statistics.

        Args:
            amax_durations (int or list of int): Durations (in hours) for which annual maxima (AMAX) should be
                identified.
            ddf_return_periods (int or list of int): Return periods (in years) for which depths should be estimated,
                given the durations specified by ``amax_durations``.
            amax_window_type (str): Use a ``'sliding'`` (default) or ``'fixed'`` window in AMAX extraction.
            subset_length (int): For splitting a realisation into ``subset_length`` years for calculating (seasonal)
                statistics (e.g. mean, variance, etc.). Does not affect AMAX extraction or DDF calculations.
            output_filenames (str or dict): Either key/value pairs indicating output file names, ``'default'`` to use
                ``{'statistics': 'simulated_statistics.csv', 'amax': 'simulated_amax.csv', 'ddf': 'simulated_ddf.csv}``
                or ``None`` to indicate that no output files should be written.
            calculate_statistics (bool): Indicates whether to calculate statistics (e.g. mean, variance, etc) for
                comparison with reference and/or fitted statistics. Default is True.
            simulation_format (str): Flag indicating point output file format (current option is ``txt``).
            start_year (int): See ``self.simulate()`` arguments.
            timestep_length (int): See ``self.simulate()`` arguments.
            calendar (str): See ``self.simulate()`` arguments.
            simulation_subfolders (dict): See ``self.simulate()`` arguments (``output_subfolders``).
            simulation_length (int): See ``self.simulate()`` arguments.
            n_realisations (int): See ``self.simulate()`` arguments.

        Notes:
            If the ``self.simulate()`` method has been run in the same session then the following arguments are not
            required: ``simulation_format, start_year, timestep_length, calendar, simulation_subfolders,
            simulation_length, n_realisations``.

        """
        print('Rainfall simulation post-processing')

        # Check that either arguments of self.simulation_args are available
        postprocessing_args = utils.get_kwargs()
        keys_to_ignore = ['amax_durations', 'ddf_return_periods', 'subset_length', 'output_filenames']
        for key, value in postprocessing_args.items():
            if key not in keys_to_ignore:
                if value is None:
                    if self.simulation_args is None:
                        raise ValueError(key, 'not set in self.simulation_args or provided as argument.')
                    if key in self.simulation_args.keys():
                        if self.simulation_args[key] is None:
                            raise ValueError(key, 'not set in self.simulation_args or provided as argument.')

        # Get simulation arguments needed for post-processing
        if simulation_format is None:
            simulation_format = self.simulation_args['simulation_format']
        if start_year is None:
            start_year = self.simulation_args['start_year']
        if timestep_length is None:
            timestep_length = self.simulation_args['timestep_length']
        if calendar is None:
            calendar = self.simulation_args['calendar']
        if simulation_subfolders is None:
            simulation_subfolders = self.simulation_args['simulation_subfolders']
        if simulation_length is None:
            simulation_length = self.simulation_args['simulation_length']
        if n_realisations is None:
            n_realisations = self.simulation_args['n_realisations']

        # Construct output paths
        output_subfolder = os.path.join(self.output_folder, simulation_subfolders['point'])
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        if output_filenames == 'default':
            output_statistics_filename = 'simulated_statistics.csv'
            output_amax_filename = 'simulated_amax.csv'
            output_ddf_filename = 'simulated_ddf.csv'
        else:
            if 'statistics' in output_filenames:
                output_statistics_filename = output_filenames['statistics']
            if 'amax' in output_filenames:
                output_amax_filename = output_filenames['amax']
            if 'ddf' in output_filenames:
                output_ddf_filename = output_filenames['ddf']
        output_statistics_path = os.path.join(output_subfolder, output_statistics_filename)
        if amax_durations is not None:
            output_amax_path = os.path.join(output_subfolder, output_amax_filename)
        else:
            output_amax_path = None
        if ddf_return_periods is not None:
            output_ddf_path = os.path.join(output_subfolder, output_ddf_filename)
        else:
            output_ddf_path = None

        # Check (partially) that arguments are suitable for analysis call
        if amax_durations is not None:
            if not isinstance(amax_durations, list):
                amax_durations = [amax_durations]
        if ddf_return_periods is not None:
            if not isinstance(ddf_return_periods, list):
                ddf_return_periods = [ddf_return_periods]

        # Construct paths to simulation (point) output
        if simulation_subfolders == 'default':
            if self.spatial_model:
                simulation_subfolders = dict(point='point', catchment='catchment', grid='grid')
            else:
                simulation_subfolders = dict(point='')
        timeseries_path = None
        timeseries_folder = os.path.join(output_subfolder)

        # Subset on points to use in post-processing
        # if 'postprocess' in self.point_metadata.columns:
        #     pass

        self.simulated_statistics, _ = analysis.main(
            spatial_model=self.spatial_model,
            season_definitions=self.season_definitions,
            statistic_definitions=self.statistic_definitions,
            timeseries_format=simulation_format,
            start_date=datetime.datetime(start_year, 1, 1),
            timestep_length=timestep_length,
            calendar=calendar,
            timeseries_path=timeseries_path,
            timeseries_folder=timeseries_folder,
            point_metadata=self.point_metadata,
            calculation_period=None,
            completeness_threshold=0.0,
            output_statistics_path=output_statistics_path,
            outlier_method=None,
            maximum_relative_difference=None,
            maximum_alterations=None,
            analysis_mode='postprocessing',
            n_years=simulation_length,
            n_realisations=n_realisations,
            subset_length=subset_length,
            output_amax_path=output_amax_path,
            amax_durations=amax_durations,
            amax_window_type=amax_window_type,
            output_ddf_path=output_ddf_path,
            ddf_return_periods=ddf_return_periods,
            write_output=True,
            simulation_name=self.project_name,
            # n_workers=n_workers,
            use_pooling=False,
            calculate_statistics=calculate_statistics,
            dayfirst=False,
        )

        print('  - Completed')

    def set_statistics(
            self,
            point_metadata=None,
            reference_statistics=None,
            fitted_statistics=None,
            simulated_statistics=None,
    ):
        """
        Set statistics and related attributes.

        Args:
            point_metadata (pandas.DataFrame or str): Required for a spatial model. See ``RainfallModel`` class
                docstring for explanation.
            reference_statistics (pandas.DataFrame or str): Reference statistics for model fitting and/or evaluation as
                dataframe (or path to file). Also used in simulation for a spatial model. Optional (depending on
                subsequent workflow).
            fitted_statistics: Optional.
            simulated_statistics: Optional.

        Notes:
            If dataframe is passed it is currently assumed that lag and threshold columns are present.

            Setting statistics with this method updates the ``self.statistic_definitions``, ``self.point_metadata`` and
            ``self.phi`` attributes, as well as the relevant statistics attribute (e.g. ``self.reference_statistics``).

            No checks are currently made on the different statistics arguments or their consistency with the
            ``point_metadata`` argument.

            For identifying (updating) ``self.statistic_definitions``, order of priority is reference > fitted >
            simulated.

            Strongly recommended that ``self.reference_statistics`` is set either here or via ``self.preprocess()``, as
            it is required for both fitting and simulation.

        """
        print('Setting rainfall statistics')

        # Point metadata
        # if self.spatial_model and point_metadata is None:
        #     raise ValueError('point_metadata must be supplied for a spatial model.')
        if (self.point_metadata is None) and (point_metadata is None):
            # TODO: Confirm whether point_metadata is needed/compulsory for rainfall model - ideally not
            # raise ValueError('point_metadata must be supplied.')
            pass
        if isinstance(point_metadata, pd.DataFrame):
            self.point_metadata = point_metadata
        elif isinstance(point_metadata, str):
            self.point_metadata = utils.read_csv_(point_metadata)
        # if not self.spatial_model:
        #     self.point_metadata = None

        # Reference statistics
        if reference_statistics is not None:
            if isinstance(reference_statistics, pd.DataFrame):
                self.reference_statistics = reference_statistics
            elif isinstance(reference_statistics, str):
                self.reference_statistics = utils.read_statistics(reference_statistics)
            phi = self.reference_statistics.loc[
                (self.reference_statistics['point_id'] == self.reference_statistics['point_id'].min())
                & (self.reference_statistics['statistic_id'] == self.reference_statistics['statistic_id'].min()),
                ['point_id', 'season', 'phi']
            ]
            if self.spatial_model:
                self.phi = pd.merge(phi, self.point_metadata)
            else:
                self.phi = phi

        # Fitted statistics
        if fitted_statistics is not None:
            if isinstance(fitted_statistics, pd.DataFrame):
                self.fitted_statistics = fitted_statistics
            elif isinstance(fitted_statistics, str):
                self.fitted_statistics = utils.read_statistics(fitted_statistics)

        # Simulated statistics
        if simulated_statistics is not None:
            if isinstance(simulated_statistics, pd.DataFrame):
                self.simulated_statistics = simulated_statistics
            elif isinstance(simulated_statistics, str):
                self.simulated_statistics = utils.read_statistics(simulated_statistics)

        # Derive statistics definitions
        columns = ['statistic_id', 'name', 'duration', 'lag', 'threshold', 'weight']
        if self.reference_statistics is not None:
            self.statistic_definitions = self.reference_statistics[columns].drop_duplicates(subset=columns)
        elif self.fitted_statistics is not None:
            self.statistic_definitions = self.fitted_statistics[columns].drop_duplicates(subset=columns)
        elif self.simulated_statistics is not None:
            self.statistic_definitions = self.simulated_statistics[columns].drop_duplicates(subset=columns)

        # Create phi dataframe
        if self.spatial_model and (self.reference_statistics is not None):
            columns = ['point_id', 'season', 'phi']
            phi = self.reference_statistics[columns].drop_duplicates(subset=columns)
            self.phi = pd.merge(self.point_metadata, phi)

        print('  - Completed')

    def set_parameters(self, parameters):
        """
        Set parameters attribute.

        Args:
            parameters (pd.DataFrame or str): Dataframe (or path to file) containing parameters for each month/season.

        """
        print('Setting rainfall model parameters')

        if isinstance(parameters, pd.DataFrame):
            self.parameters = parameters
        elif isinstance(parameters, str):
            self.parameters = utils.read_csv_(parameters)

        print('  - Completed')

    def perturb_statistics(
            self,
            change_factors,
            change_factor_names='default',
            easting_name='projection_x_coordinate',
            northing_name='projection_y_coordinate',
            month_variable='month_number',
            write_output=True,
            output_filename='perturbed_statistics.csv',
    ):
        """
        Perturb reference statistics using a set of change factors.

        Updates self.reference_statistics in preparation for fitting.

        Args:
            change_factors (dict): Paths to NetCDF files containing change factors for each required duration
                (e.g. 1H, 24H).
            change_factor_names (dict): Mapping of statistic names used in rwgen to change factor variable names in
                NetCDF files. See notes for 'default' mapping.
            easting_name (str): Name of easting variable in NetCDF files.
            northing_name (str): Name of northing variable in NetCDF files.
            month_variable (str): Name of variable indicating month number associated with each time in the NetCDF
                files.
            write_output (bool): Flag to write output file containing perturbed statistics.
            output_filename (str): Name of output file

        Notes:
             Change factors are read from NetCDF files. The files should be specified via the ``change_factors``
             argument using e.g.:
             ``change_factors={'1H': 'C:/Path/To/1H_ChangeFactors.nc', '24H': 'C:/Path/To/24H_ChangeFactors.nc'}``

             The default names for mapping rwgen statistics to the change factors in the NetCDF files are specified by
             the following dictionary:
             ``change_factor_names = {
             'mean': 'mean_changefactor_ts1_to_ts3',
             'variance': 'variance_changefactor_ts1_to_ts3',
             'skewness': 'skewness_changefactor_ts1_to_ts3',
             'probability_dry_0.2mm': 'pd_0p2_changefactor_ts1_to_ts3',
             'autocorrelation_lag1': 'l1ac_changefactor_ts1_to_ts3',
             }``

             Other points to note are:

                 * Perturbation currently works for the point (single site) version of the model only.
                 * Change factors can also only be specified monthly at present.
                 * Only a 0.2mm threshold for dry probability is supported currently.
                 * Only lag-1 autocorrelation is supported.

        """
        print('Perturbing reference statistics')

        if self.spatial_model:
            raise TypeError('Statistic perturbation is only implemented for the single site model currently.')

        if change_factor_names == 'default':
            change_factor_names = {
                'mean': 'mean_changefactor_ts1_to_ts3',
                'variance': 'variance_changefactor_ts1_to_ts3',
                'skewness': 'skewness_changefactor_ts1_to_ts3',
                'probability_dry_0.2mm': 'pd_0p2_changefactor_ts1_to_ts3',
                'autocorrelation_lag1': 'l1ac_changefactor_ts1_to_ts3',
            }

        durations = list(change_factors.keys())
        statistic_names = list(change_factor_names.keys())

        ref_stats = self.reference_statistics.copy()

        easting = self.point_metadata.loc[self.point_metadata['point_id'] == 1, 'easting'].values[0]
        northing = self.point_metadata.loc[self.point_metadata['point_id'] == 1, 'northing'].values[0]

        stat_defs = self.statistic_definitions.loc[self.statistic_definitions['duration'].isin(durations)]

        self.reference_statistics = perturbation.perturb_statistics(
            stat_defs=stat_defs,
            statistic_names=statistic_names,
            durations=durations,
            change_factors=change_factors,
            month_variable=month_variable,
            easting_name=easting_name,
            northing_name=northing_name,
            easting=easting,
            northing=northing,
            change_factor_names=change_factor_names,
            ref_stats=ref_stats,
        )

        if write_output:
            utils.write_statistics(
                self.reference_statistics,
                os.path.join(self.output_folder, output_filename),
                self.season_definitions,
            )

        print('  - Completed')

    def update_output_folder(self, output_folder):
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def update_simulation_config(
            self,
            default_block_size=1000,
            check_block_size=True,
            minimum_block_size=10,
            check_available_memory=True,
            maximum_memory_percentage=75,
            block_subset_size=50,
            spatial_buffer_factor=15,
    ):
        """
        Update default configuration settings for simulation.

        Args:
            default_block_size (int): Number of years (maximum) to simulate at once to avoid memory issues.
            check_block_size (bool): Flag to indicate whether code should automatically check whether the
                default_block_size (probably) needs to be reduced to avoid memory issues (see
                ``check_available_memory`` and ``maximum_memory_percentage``).
            minimum_block_size (int): Minimum number of years to simulate at once.
            check_available_memory (bool): Flag to indicate whether current system memory usage should be checked to
                limit the maximum amount of memory assigned in simulation.
            maximum_memory_percentage (int or float): Maximum percentage of system memory that may be assigned by a
                simulation. If estimated memory usage exceeds this percentage then a smaller block size will be tried
                until the ``minimum_block_size`` is reached.
            block_subset_size (int): Block subset size (number of years) for internal use in discretisation (as it is
                much faster to discretise subsets of each block).
            spatial_buffer_factor (float): Number of standard deviations of raincell radius distribution to use with
                buffer method of spatial raincell simulation.

        """
        simulation_config = dict(
            default_block_size=default_block_size,
            check_block_size=check_block_size,
            minimum_block_size=minimum_block_size,
            check_available_memory=check_available_memory,
            maximum_memory_percentage=maximum_memory_percentage,
            block_subset_size=block_subset_size,
            spatial_buffer_factor=spatial_buffer_factor,
        )
        if hasattr(self, 'simulation_config'):
            self.simulation_config = simulation_config
        else:
            return simulation_config

    def update_shuffling_config(
            self,
            target_duration='1M',
            month_divisions=4,  # TODO: Could be 8 for single site?
            reorder_months=True
    ):
        """
        Update shuffling settings for fitting and simulation using modified Kim and Onof (2020) method.

        Args:
            target_duration (str): Duration to use for reference variance in fitting (cannot be changed currently).
            month_divisions (int): Number of windows to divide month into when shuffling. This needs to be an even
                number currently (4, 6 or 8). Possible to use 8 with a single site model, but 4 better for a spatial
                model.
            reorder_months (bool): If True (default) then apply both shuffling algorithm and then further reordering
                based on an AR1 model. These stages correspond to modules 2 and 3 in Kim and Onof (2020).

        """
        shuffling_config = dict(
            target_duration=target_duration,
            month_divisions=month_divisions,
            reorder_months=reorder_months,
        )
        if hasattr(self, 'shuffling_config'):
            self.shuffling_config = shuffling_config
        else:
            return shuffling_config

    def plot(self, plot_type='annual_cycle', data_types='all', point_id=1):
        """
        Plot reference, fitted and/or simulated statistics.

        Args:
            plot_type (str): Flag to plot ``'annual_cycle'`` or ``'cross-correlation'`` statistics.
            data_types (str or list of str): Indicates which of ``'reference', 'fitted', 'simulated'`` to plot.
            point_id (int): Point to plot for annual cycle statistics.

        """
        if data_types == 'all':
            data_types = ['reference', 'fitted', 'simulated']
        elif isinstance(data_types, str):
            data_types = list(data_types)

        if ('reference' in data_types) and (self.reference_statistics is not None):
            ref = self.reference_statistics
            if plot_type == 'annual_cycle':
                ref = ref.loc[ref['point_id'] == point_id]
        else:
            ref = None

        if ('fitted' in data_types) and (self.fitted_statistics is not None):
            fit = self.fitted_statistics
            if plot_type == 'annual_cycle':
                fit = fit.loc[fit['point_id'] == point_id]
        else:
            fit = None

        if ('simulated' in data_types) and (self.simulated_statistics is not None):
            sim = self.simulated_statistics
            if plot_type == 'annual_cycle':
                sim = sim.loc[sim['point_id'] == point_id]
        else:
            sim = None

        # Plot per statistic definition (except cross-correlation)
        if plot_type == 'annual_cycle':
            plots = []
            for _, row in self.statistic_definitions.iterrows():
                sid = row['statistic_id']
                name = row['name']
                duration = row['duration']

                if name != 'cross-correlation':

                    p = plotting.plot_annual_cycle(sid, name, duration, ref, fit, sim)
                    plots.append(p)

        # Plot per season/month (and duration) if cross-correlation
        if plot_type == 'cross-correlation':
            plots = []
            for _, row in self.statistic_definitions.iterrows():
                sid = row['statistic_id']
                name = row['name']
                duration = row['duration']

                if name == 'cross-correlation':
                    for season in self.unique_seasons:
                        if ref is not None:
                            ref_sub = ref.loc[ref['season'] == season]
                        else:
                            ref_sub = None
                        if fit is not None:
                            fit_sub = fit.loc[fit['season'] == season]
                        else:
                            fit_sub = None
                        if sim is not None:
                            sim_sub = sim.loc[sim['season'] == season]
                        else:
                            sim_sub = None
                        p = plotting.plot_cross_correlation(sid, name, duration, season, ref_sub, fit_sub, sim_sub)
                        plots.append(p)

        # Construct grid plot and show
        g = plotting.construct_gridplot(plots, 3)
        plotting.show_plot(g)

    def zip_output(self, file_extension='.txt', delete_uncompressed=False):  # essentially duplicated in weather model
        """
        Zip and compress output files of a specified extension (optionally deleting uncompressed files).

        Args:
            file_extension (str): Delete files with this extension (default is '.txt').
            delete_uncompressed (bool): Delete uncompressed files after zipping complete (default is False).

        """
        print('Rainfall model output zipping')
        utils.zip_files(self.output_folder, file_extension=file_extension, delete_uncompressed=delete_uncompressed)
        print('  - Completed')

    @property
    def parameter_names(self):
        """list of str: Parameter names."""
        if self.spatial_model:
            parameter_names = ['lamda', 'beta', 'rho', 'eta', 'gamma']
        else:
            parameter_names = ['lamda', 'beta', 'nu', 'eta']
        if self.intensity_distribution == 'exponential':
            parameter_names.append('theta')  # theta = 1 / xi
        elif self.intensity_distribution == 'weibull':
            parameter_names.extend(['theta', 'kappa'])
        elif self.intensity_distribution == 'generalised_gamma':
            parameter_names.extend(['theta', 'kappa_1', 'kappa_2'])
        return parameter_names

    @property
    def unique_seasons(self):
        """list of int: Unique season identifiers."""
        return list(set(self.season_definitions.values()))


