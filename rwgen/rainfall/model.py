import os
import datetime

import numpy as np
import pandas as pd
import geopandas

from . import analysis
from . import fitting
from . import simulation
from . import utils


class RainfallModel:
    """
    Neyman-Scott Rectangular Pulse (NSRP) rainfall model for point/site and spatial simulations.

    Args:
        spatial_model (bool): Flag to indicate whether point or spatial model.
        project_name (str): A name for the gauge/site location, domain or catchment.
        season_definitions (str, list or dict): The model works on a monthly basis by default, but this argument
            allows for user-defined seasons (see Notes).
        intensity_distribution (str): Flag to indicate the type of probability distribution for raincell
            intensities. Defaults to ``'exponential'`` (with ``'weibull'`` also available currently).
        output_folder (str): Root folder for model output. Defaults to ``'./output'``.
        statistic_definitions (pandas.DataFrame or str): Definitions (descriptions) of statistics to use in fitting
            and/or evaluation (or path to file of definitions). See Notes for explanation of format.
        point_metadata (pandas.DataFrame or str): Metadata (or path to metadata file) on point (site/gauge)
            locations to use for fitting, simulation and/or evaluation for a spatial model only. See Notes for
            details.

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
        1             variance               1         1
        2             skewness               1         2
        3             probability_dry_0.2mm  1         7
        4             mean                   24        6
        5             variance               24        2
        6             skewness               24        3
        7             probability_dry_0.2mm  24        7
        8             autocorrelation_lag1   24        6
        ============  =====================  ========  ======

        For a spatial model the default ``statistic_definitions`` are:

        ============  ======================  ========  ======
        Statistic_ID  Name                    Duration  Weight
        ============  ======================  ========  ======
        1             variance                1         3
        2             skewness                1         3
        3             probability_dry_0.2mm   1         5
        4             mean                    24        5
        5             variance                24        2
        6             skewness                24        2
        7             probability_dry_0.2mm   24        6
        8             autocorrelation_lag1    24        3
        9             cross-correlation_lag0  24        2
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
            project_name,
            season_definitions='monthly',
            intensity_distribution='exponential',
            output_folder='./output',
            statistic_definitions=None,
            point_metadata=None,
    ):
        self.season_definitions = utils.parse_season_definitions(season_definitions)
        self.spatial_model = spatial_model
        self.intensity_distribution = intensity_distribution
        self.output_folder = output_folder
        self.project_name = project_name

        # Spatial model requires a table of metadata for points
        if self.spatial_model:
            if isinstance(point_metadata, pd.DataFrame):
                self.point_metadata = point_metadata
            elif isinstance(point_metadata, str):
                self.point_metadata = utils.read_csv_(point_metadata)
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
                    1: {'weight': 3.0, 'duration': 1, 'name': 'variance'},
                    2: {'weight': 3.0, 'duration': 1, 'name': 'skewness'},
                    3: {'weight': 5.0, 'duration': 1, 'name': 'probability_dry', 'threshold': 0.2},
                    4: {'weight': 5.0, 'duration': 24, 'name': 'mean'},
                    5: {'weight': 2.0, 'duration': 24, 'name': 'variance'},
                    6: {'weight': 2.0, 'duration': 24, 'name': 'skewness'},
                    7: {'weight': 6.0, 'duration': 24, 'name': 'probability_dry', 'threshold': 0.2},
                    8: {'weight': 3.0, 'duration': 24, 'name': 'autocorrelation', 'lag': 1},
                    9: {'weight': 2.0, 'duration': 24, 'name': 'cross-correlation', 'lag': 0}
                }
            else:
                dc = {
                    1: {'weight': 1.0, 'duration': 1, 'name': 'variance'},
                    2: {'weight': 2.0, 'duration': 1, 'name': 'skewness'},
                    3: {'weight': 7.0, 'duration': 1, 'name': 'probability_dry', 'threshold': 0.2},
                    4: {'weight': 6.0, 'duration': 24, 'name': 'mean'},
                    5: {'weight': 2.0, 'duration': 24, 'name': 'variance'},
                    6: {'weight': 3.0, 'duration': 24, 'name': 'skewness'},
                    7: {'weight': 7.0, 'duration': 24, 'name': 'probability_dry', 'threshold': 0.2},
                    8: {'weight': 6.0, 'duration': 24, 'name': 'autocorrelation', 'lag': 1},
                }
            id_name = 'statistic_id'
            non_id_columns = ['name', 'duration', 'lag', 'threshold', 'weight']
            self.statistic_definitions = utils.nested_dictionary_to_dataframe(dc, id_name, non_id_columns)

        # Check that statistics include 24hr mean, as it is currently required for calculating phi (add in if absent)
        includes_24hr_mean = self.statistic_definitions.loc[
            (self.statistic_definitions['name'] == 'mean') & (self.statistic_definitions['duration'] == 24)
            ].shape[0]
        if not includes_24hr_mean:
            df = pd.DataFrame({
                'statistic_id': [int(np.max(statistic_definitions['statistic_id'])) + 1], 'weight': [0],
                'duration': [24], 'name': ['mean'], 'lag': ['NA'], 'threshold': ['NA']
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

    def preprocess(
            self,
            input_timeseries,
            calculation_period='full_record',
            completeness_threshold=0.0,
            outlier_method=None,
            maximum_relative_difference=2.0,
            maximum_alterations=5,
            amax_durations=None,
            output_filenames='default',
    ):
        """
        Prepare reference statistics, weights and scale factors for use in model fitting, simulation and evaluation.

        Updates ``self.reference_statistics`` and ``self.phi`` attributes and writes a ``reference_statistics`` output
        file.

        Args:
            input_timeseries (str): Path to file containing timeseries data (for point model) or folder containing
                timeseries data files (for spatial model).
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
            output_filenames (str or dict): Either key/value pairs indicating output file names, ``'default'`` to use
                {'statistics': 'reference_statistics.csv', 'amax': 'reference_amax.csv'} or ``None`` to indicate that
                no output files should be written.

        Notes:
            Currently ``.csv`` files are used for time series inputs. These files are expected to contain a
            ``DateTime`` column using ``dd/mm/yyyy hh:mm`` format, i.e. '%d/%m/%Y %H:%M'. They should also contain
            a ``Value`` column using units of mm/timestep.

        """
        print('Preprocessing')

        # Infer timeseries data format
        if not self.spatial_model:
            timeseries_format = input_timeseries.split('.')[-1]
        else:
            file_names = os.listdir(input_timeseries)
            test_file = self.point_metadata['name'].values[0]
            for file_name in file_names:
                file_name, extension = file_name.split('.')
                if file_name == test_file:
                    timeseries_format = extension
                    break

        # Input paths as required by analysis function
        if self.spatial_model:
            timeseries_path = None
            timeseries_folder = input_timeseries
        else:
            timeseries_path = input_timeseries
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
            calculation_period = None

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
            calculation_period=calculation_period,
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
            output_ddf_path=None,
            ddf_return_periods=None,
            write_output=write_output,
            simulation_name=None,
        )

        print('  - Completed')

    def fit(
            self,
            fitting_method='default',
            parameter_bounds=None,
            fixed_parameters=None,
            n_workers=1,
            initial_parameters=None,
            smoothing_tolerance=0.2,
            output_filenames='default',
    ):
        """
        Fit model parameters.

        Depends on ``self.reference_statistics`` attribute. Sets ``self.parameters`` and ``self.fitted_statistics``.
        Also writes ``parameters`` and `fitted_statistics`` output files.

        Args:
            fitting_method (str): Flag to indicate fitting method. Using ``'default'`` will fit each month or season
                independently. Other options (including ``'empirical_smoothing'``) under development.
            parameter_bounds (dict or str or pandas.DataFrame): Dictionary containing tuples of upper and lower
                parameter bounds by parameter name. Alternatively the path to a parameter bounds file or an equivalent
                dataframe (see Notes).
            fixed_parameters (dict or str or pandas.DataFrame): Dictionary containing fixed parameter values by
                parameter name. Alternatively the path to a parameters file or an equivalent dataframe (see Notes).
            n_workers (int): Number of workers (cores/processes) to use in fitting. Default is 1.
            initial_parameters (pandas.DataFrame or str): Initial parameter values to use if ``fitting_method`` is
                ``'empirical_smoothing'``. If not specified then initial parameter values will be obtained using
                the default fitting method (for which no initial values are currently required).
            smoothing_tolerance (float): Permitted deviation in smoothed annual cycle of parameter values (only used
                if ``fitting_method`` is ``'empirical_smoothing'``). Expressed as fraction of annual mean parameter
                value, e.g. 0.2 allows a +/- 20% deviation from the smoothed annual cycle for a given parameter.
            output_filenames (str or dict): Either key/value pairs indicating output file names, ``'default'`` to use
                {'statistics': 'fitted_statistics.csv', 'parameters': 'parameters.csv'} or ``None`` to indicate that
                no output files should be written.

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
                * If using ``intensity_distribution=='weibull'``, theta is the scale parameter and an additional
                  parameter (kappa) is introduced as the shape parameter.

            The ``parameter_bounds`` argument can be specified by a dictionary like ``dict('beta': (0.02, 1.0)``. For
            more control a dataframe (or ``.csv`` file) can be passed. For example, if a model has two (6-month)
            seasons (using arbitrary example numbers):

            ======  =========  ===========  ===========
            Season  Parameter  Lower_Bound  Upper_Bound
            ======  =========  ===========  ===========
            1       Beta       0.02         0.1
            2       Beta       0.1          1.0
            ======  =========  ===========  ===========

            If a parameter(s) should be fixed across all seasons then it can be set as e.g.
            ``dict('beta'=0.1, 'theta'=1)``. Otherwise a table can be provided like:

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
            -1      rho        0.0001       2.0
            -1      eta        0.1          12.0
            -1      gamma      0.01         500.0
            -1      theta      0.25         100.0
            -1      kappa      0.5          1.0
            ======  =========  ===========  ===========

            Fitting can be speeded up significantly with ``n_workers > 1``. The maximum ``n_workers`` should be less
            than or equal to the number of cores or logical processors available.

        """
        print('Fitting')

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
            default_bounds = pd.DataFrame.from_dict({
                'lamda': (0.001, 0.05), 'beta': (0.02, 0.5), 'rho': (0.0001, 2.0), 'eta': (0.1, 12.0),
                'gamma': (0.01, 500.0),
                'theta': (0.25, 100), 'kappa': (0.5, 1.0), 'kappa_1': (0.5, 1.0), 'kappa_2': (0.5, 1.0)
            }, orient='index', columns=['lower_bound', 'upper_bound'])
        default_bounds.reset_index(inplace=True)
        default_bounds.rename(columns={'index': 'parameter'}, inplace=True)

        # Identify parameters to fit (or not fit) and set parameter bounds
        parameters_to_fit, fixed_parameters, parameter_bounds = utils.define_parameter_bounds(
            parameter_bounds, fixed_parameters, self.parameter_names, default_bounds, self.unique_seasons
        )

        # Initial parameters are currently only relevant to empirical smoothing method
        if initial_parameters is not None:
            if isinstance(initial_parameters, pd.DataFrame):
                pass
            elif isinstance(initial_parameters, str):
                initial_parameters = utils.read_csv_(initial_parameters)

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

        # Do fitting
        self.parameters, self.fitted_statistics = fitting.main(
            season_definitions=self.season_definitions,
            spatial_model=self.spatial_model,
            intensity_distribution=self.intensity_distribution,
            fitting_method=fitting_method,
            reference_statistics=self.reference_statistics,
            all_parameter_names=self.parameter_names,  # RENAMED
            parameters_to_fit=parameters_to_fit,  # NEW
            parameter_bounds=parameter_bounds,  # SAME
            fixed_parameters=fixed_parameters,  # NEW
            n_workers=n_workers,
            output_parameters_path=output_parameters_path,
            output_statistics_path=output_statistics_path,
            initial_parameters=initial_parameters,
            smoothing_tolerance=smoothing_tolerance,
            write_output=write_output,  # NEW
        )

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

        Notes:
            Though gridded output is calculated (if ``output_types`` includes ``'grid'``) it is not yet available to
            write (under development).

            All locations in ``self.point_metadata`` will be written as output currently.  # TODO: CHANGE THIS

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
            output_name = self.project_name

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
                spatial_raincell_method=self.simulation_config['spatial_raincell_method'],
                spatial_buffer_factor=self.simulation_config['spatial_buffer_factor'],
            )

        print('  - Completed')

    def postprocess(
            self,
            amax_durations=None,
            ddf_return_periods=None,
            subset_length=50,
            output_filenames='default',
            simulation_format=None,
            start_year=None,
            timestep_length=None,
            calendar=None,
            simulation_subfolders=None,
            simulation_length=None,
            n_realisations=None,
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
            subset_length (int): For splitting a realisation into ``subset_length`` years for calculating (seasonal)
                statistics (e.g. mean, variance, etc.). Does not affect AMAX extraction or DDF calculations.
            output_filenames (str or dict): Either key/value pairs indicating output file names, ``'default'`` to use
                ``{'statistics': 'simulated_statistics.csv', 'amax': 'simulated_amax.csv', 'ddf': 'simulated_ddf.csv}``
                or ``None`` to indicate that no output files should be written.
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
        print('Post-processing')

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
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
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
        output_statistics_path = os.path.join(self.output_folder, output_statistics_filename)
        if amax_durations is not None:
            output_amax_path = os.path.join(self.output_folder, output_amax_filename)
        else:
            output_amax_path = None
        if ddf_return_periods is not None:
            output_ddf_path = os.path.join(self.output_folder, output_ddf_filename)
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
        timeseries_folder = os.path.join(self.output_folder, simulation_subfolders['point'])

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
            output_ddf_path=output_ddf_path,
            ddf_return_periods=ddf_return_periods,
            write_output=True,
            simulation_name=self.project_name,
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
            point_metadata: Required for a spatial model.
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
        # Point metadata
        if self.spatial_model and point_metadata is None:
            raise ValueError('point_metadata must be supplied for a spatial model.')
        if isinstance(point_metadata, pd.DataFrame):
            self.point_metadata = point_metadata
        elif isinstance(point_metadata, str):
            self.point_metadata = utils.read_csv_(point_metadata)
        if not self.spatial_model:
            self.point_metadata = None

        # Reference statistics
        if reference_statistics is not None:
            if isinstance(reference_statistics, pd.DataFrame):
                self.reference_statistics = reference_statistics
            elif isinstance(reference_statistics, str):
                self.reference_statistics = utils.read_statistics(reference_statistics)

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

    def set_parameters(self, parameters):
        """
        Set parameters attribute.

        Args:
            parameters (pd.DataFrame or str): Dataframe (or path to file) containing parameters for each month/season.

        """
        if isinstance(parameters, pd.DataFrame):
            self.parameters = parameters
        elif isinstance(parameters, str):
            self.parameters = utils.read_csv_(parameters)

    def update_output_folder(self, output_folder):
        self.output_folder = output_folder

    def update_simulation_config(
            self,
            default_block_size=1000,
            check_block_size=True,
            minimum_block_size=10,
            check_available_memory=True,
            maximum_memory_percentage=75,
            block_subset_size=50,
            spatial_raincell_method='buffer',
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
            spatial_raincell_method (str): Flag to use ``'buffer'`` method or Burton et al. (2010) method ``'burton'``
                for spatial raincell simulation.
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
            spatial_raincell_method=spatial_raincell_method,
            spatial_buffer_factor=spatial_buffer_factor,
        )
        if hasattr(self, 'simulation_config'):
            self.simulation_config = simulation_config
        else:
            return simulation_config

    def plot(self):
        raise NotImplementedError

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


