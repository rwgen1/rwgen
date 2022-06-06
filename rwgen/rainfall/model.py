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
    Point and spatial Neyman-Scott Rectangular Pulse models.

    This class contains methods for data pre-processing, model fitting, simulation and post-processing.

    Attributes:


    """

    # TODO: Clarify which arguments are optional and which are needed for point vs spatial model
    # TODO: Consider breaking up into smaller methods
    # - e.g. pre-processing for NSRP fitting, NSRP simulation, SARIMA fitting, etc
    # - plus e.g. simulation constructing output paths, doing normal vs shuffling simulations
    # - any value in setting more stuff as attributes to allow calling etc from different places?
    # TODO: Swap order of precedence so that input arguments are given precedence over existing model attributes
    # - should e.g. reference statistics passed into fitting be used to update reference statistics and statistics
    # definitions? - this would probably make sense
    # TODO: Move defaults into a config.py, as it will be easier for users to edit?
    # TODO: Check use of lambda vs lamda now - only referred to in dataframes?
    # TODO: Check switch from xi to theta throughout
    # TODO: Flag in metadata (points) file to indicate whether to use in fitting or just simulation
    # TODO: Put attributes in class docstring once decided on how to get relevant defaults to post-processing etc
    # TODO: Check consistency of cross-correlation vs cross-correlations, especially in user-related arguments
    # TODO: Ensure only final stage parameters are used in simulation

    def __init__(
            self,
            season_definitions='monthly',
            spatial_model=False,
            intensity_distribution='exponential',
            output_folder='./output',
            statistic_definitions=None,
            point_metadata=None,
            project_name=None,
    ):
        """
        Args:
            season_definitions (str, list or dict): The model works on a monthly basis by default, but this argument
                allows for user-defined seasons. The seasons can be specified in several ways (see Notes).
            spatial_model (bool): Flag to indicate whether point or spatial model. Default is ``False`` (point model).
            intensity_distribution (str): Flag to indicate the type of probability distribution for raincell
                intensities. Defaults to ``'exponential'`` (with ``weibull`` also available currently).
            output_folder (str): Root folder for model output.
            statistic_definitions (pandas.DataFrame or str): Definitions (descriptions) of statistics to calculate or
                path to file of definitions. See Notes for explanation of DataFrame contents and file format.
            point_metadata (pandas.DataFrame or str): Metadata (or path to metadata file) on point locations for which
                preprocessing should be carried out for a spatial model (or path to metadata file). The dataframe should
                contain identifiers (integers) and coordinates - see Notes.
            project_name (str): A name for e.g. the gauge/site location or catchment may be specified (optional). If
                given this name will be used in the file names of point simulations (i.e. when ``spatial_model`` is set
                to ``False``).

        Notes:
            Seasons can be specified through the season_definitions argument in several ways:
                * As descriptive strings (``'monthly'``, ``'quarterly'``, ``'half-years'``, ``'annual'``). Further
                  control can be gained using e.g. ``'quarterly_dec'`` to make Dec-Jan-Feb the first season and so on).
                  Specifying annual will lead to the whole year being considered together, i.e. no seasonality.
                * As a list of strings indicating season abbreviations, e.g. ``['DJF', 'MAM', 'JJA', 'SON']``.
                * As a dictionary whose keys are the months of the year (integers 1-12) and whose values represent a
                  season identifier, e.g. ``dict(12=1, 1=1, 2=1, 3=2, 4=2, 5=2, 6=3, 7=3, 8=3, 9=4, 10=4, 11=4)`` would
                  give quarterly seasons beginning in December.

            statistic_definitions

            point_metadata - Useful to include points relevant to fitting and simulation if want to calculate
                reference statistics for those points (e.g. for use in evaluation).

        """
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
        if statistic_definitions is not None:
            self.statistic_definitions = statistic_definitions
        elif isinstance(statistic_definitions, str):
            self.statistic_definitions = utils.read_statistic_definitions(statistic_definitions)
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

        # TODO: Rationalise config and args - only needed for simulation?

        # Default configuration settings for simulation (primarily memory management) - see
        # ``update_preprocessing_config()`` method docstring
        self.simulation_config = self.update_simulation_config()

        # For sharing simulation arguments with post-processing method
        self.simulation_args = None

        # Calculated during model use and relevant across more than one method
        self.reference_statistics = None
        self.phi = None  # TODO: When phi is needed figure it out from reference_statistics?
        self.parameters = None
        self.fitted_statistics = None
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
        Prepare reference statistics, weights and scale factors for use in model fitting and evaluation.

        Updates ``self.reference_statistics`` and ``self.phi``attributes.

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
                identified.
            output_filenames (str or dict): Either key/value pairs indicating output file names, ``'default'`` to use
                {'statistics': 'reference_statistics.csv', 'amax': 'reference_amax.csv'} or ``None`` to indicate that
                no output files should be written.

        Notes:

            outlier_method

        """
        print('Preprocessing')

        # TODO: Document each item calculated (e.g. definitions of scale factors)
        # TODO: Expand notes section of docstring

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

        Args:
            fitting_method (str): Flag to indicate fitting method. Using ``'default'`` will fit each month or season
                independently. Option for ``'empirical_smoothing'`` under development (see Notes).
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
            If ``self.reference_statistics`` is not ``None`` it will be given priority for use in fitting. Otherwise the
            reference statistics can be passed in as an argument or read from file(s).

            Lists of parameter bounds need to be passed in the order required by the model. For the point model this
            order is: lamda, beta, nu, eta, xi. For the spatial model this order is: lamda, beta, rho, eta, gamma,
            xi. This approach will be replaced.  # TODO: Update

            Fitting can be speeded up significantly with ``n_workers > 1``. The maximum ``n_workers`` should be less
            than or equal to the number of cores or logical processors available.

            Empirical smoothing.  # TODO: Explain method so far

            Parameter bounds setting / file and fixed parameters setting / file.  # TODO: Expand

        """
        print('Fitting')

        # TODO: Can subset to remove statistics where weight = 0
        # TODO: Option to subset point_metadata so only flagged points are used

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

    def simulate(
            self,
            output_types='point',
            output_subfolders='default',
            output_format='txt',  # TODO: Add in csvy functionality
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
        Simulate realisation(s) of NSRP process.

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
            # TODO: Implement additional output and include also random seed (entropy attribute of SeedSequence)

        Notes:
            Though gridded output is calculated (if ``output_types`` includes ``'grid'``) it is not yet available to
            write (under development).

            Dataframe of metadata for points should contain fields (columns) for ...  # TODO: Complete description

            The code currently calculates catchment weights and performs interpolation of phi. Features could be added
            for these variables to be passed directly as arguments.

            Point metadata dataframe assumed to have a ``Point_ID`` field that can be sued to identify points.
            Catchment shapefile or geodataframe assumed to have an ``ID`` field that can be used to identify catchments.
            Both point and catchment metadata are assumed to have a ``Name`` field for use as a prefix in file naming.
            Point (single site) simulations and grid output are assumed not to need a prefix.

            Updates ``self.simulation_args`` in preparation for post-processing.

        """

        # TODO: Implement output for catchment_weights_output_folder and phi_output_path - currently not implemented
        # TODO: Ensure that 'final' parameters are used e.g. parameters.loc[parameters['stage'] == 'final']

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
        # TODO: Expand checks on user input arguments
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
            simulation_name=None,
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
                statistics (e.g. mean, variance, etc.). Does not affect AMAX extraction or DDF calculations. See Notes.
            output_filenames (str or dict): Either key/value pairs indicating output file names, ``'default'`` to use
                ``{'statistics': 'simulated_statistics.csv', 'amax': 'simulated_amax.csv', 'ddf': 'simulated_ddf.csv}``
                or ``None`` to indicate that no output files should be written.
            simulation_format:
            start_year:
            timestep_length:
            calendar:
            simulation_subfolders:
            simulation_length:
            n_realisations:
            simulation_name:

        Notes:
            subset_length - why? Should be less than 100 (for now at least) and less than simulation_length

            passing arguments vs using self.simulation_args

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
        if simulation_name is None:
            simulation_name = self.simulation_args['simulation_name']

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
            simulation_name=simulation_name,
        )

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
                dataframe (or path to file). Optional.
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
            parameters:

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

    # TODO: Implement method
    def plot_statistics(
            self,
            plot_type='annual_cycle',  # 'cross-correlation'
            include_reference=True,
            include_fitted=True,
            include_simulated=True,
    ):
        raise NotImplementedError

    # TODO: Implement method - what is the most useful plot(s) for AMAX/DDF? Include external reference
    def plot_amax(self):
        raise NotImplementedError

    @property
    def parameter_names(self):
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


