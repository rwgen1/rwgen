import os

import numpy as np
import pandas as pd
import geopandas

from . import preprocessing
from . import fitting
from . import simulation
from . import utils


class Model:
    """
    Pre-processing, fitting and simulation of point and spatial Neyman-Scott Rectangular Pulse (NSRP) models.

    """

    # TODO: Clarify which arguments are optional and which are needed for point vs spatial model
    # TODO: Consider breaking up into smaller methods
    # - e.g. pre-processing for NSRP fitting, NSRP simulation, SARIMA fitting, etc
    # - plus e.g. simulation constructing output paths, doing normal vs shuffling simulations
    # - any value in setting more stuff as attributes to allow calling etc from different places?

    def __init__(
            self,
            season_definitions=None,
            spatial_model=False,
            intensity_distribution='exponential'
    ):
        """
        Args:
            season_definitions (str, list or dict): The model works on a monthly basis by default, but this argument
                allows for user-defined seasons. The seasons can be specified in several ways (see Notes).
            spatial_model (bool): Flag to indicate whether point or spatial model. Default is False (point model).
            intensity_distribution (str): Flag to indicate the type of probability distribution for raincell
                intensities. Defaults to exponential (only option currently).

        Notes:
            Seasons can be specified through the season_definitions argument in several ways:
                * As descriptive strings (monthly, quarterly, half-years, annual), optionally specifying the first month
                  of one of the quarterly or half-years seasons (e.g. quarterly_dec would make Dec-Jan-Feb the first
                  season and so on). Specifying annual will lead to the whole year being considered together, i.e.
                  no seasonality.
                * As a list of strings indicating season abbreviations, e.g. ['DJF', 'MAM', 'JJA', 'SON'].
                * As a dictionary whose keys are the months of the year (integers 1-12) and whose values represent a
                  season identifier, e.g. dict(12=1, 1=1, 2=1, 3=2, 4=2, 5=2, 6=3, 7=3, 8=3, 9=4, 10=4, 11=4) would
                  give quarterly seasons beginning in December.

        """
        print()

        if season_definitions is not None:
            self.season_definitions = utils.parse_season_definitions(season_definitions)
        else:
            self.season_definitions = {}
            for month in range(1, 12 + 1):
                self.season_definitions[month] = month
        self.spatial_model = spatial_model
        self.intensity_distribution = intensity_distribution

        #: pandas.DataFrame: Statistics for fitting model parameters or evaluating model fit / simulated statistics.
        self.reference_statistics = None

        #: pandas.DataFrame: Scale factor phi at point locations (used in spatial model simulation).
        self.phi = None

        #: pandas.DataFrame: Parameters for time series simulation.
        self.parameters = None

    def preprocess(
            self,
            output_folder,
            statistic_definitions=None,
            statistic_definitions_path=None,
            timeseries_format='csv',
            timeseries_path=None,
            timeseries_folder=None,
            metadata=None,
            metadata_path=None,
            calculation_period=None,
            completeness_threshold=0.0,
            output_point_statistics_filename='reference_point_statistics.csv',
            output_cross_correlations_filename='reference_cross_correlations.csv',
            output_phi_filename='phi.csv',
            outlier_method=None,
            maximum_relative_difference=2.0,
            maximum_alterations=5
    ):
        """
        Prepare reference statistics, weights and scale factors for use in model fitting.

        # TODO: Document each item calculated (e.g. definitions of scale factors)

        Args:
            output_folder (str): Path to folder in which output statistics (and scale factors) should be written.
            statistic_definitions (pandas.DataFrame): Definitions (descriptions) of statistics to calculate. See Notes
                for explanation of DataFrame contents.
            statistic_definitions_path (str): Path to file containing statistic definitions (see Notes for details).
            timeseries_format (str): Flag indicating format of timeseries inputs. Use `csv` for now.
            timeseries_path (str): Path to file containing timeseries data (only required for point model).
            timeseries_folder (str): Path to folder containing timeseries data files (only required for spatial model).
            metadata (pandas.DataFrame): Metadata on point locations for which preprocessing should be carried out for
                a spatial model. The dataframe should contain identifiers (integers) and coordinates - see Notes.
            metadata_path (str): Path to file containing metadata for a spatial model.
            calculation_period (list of int): Start year and end year of calculation period. If not specified then all
                available data will be used in statistics calculations.
            completeness_threshold (float): Percentage completeness for a month or season to be included in statistics
                calculations. Default is 0.0, i.e. any completeness (or missing data) percentage is acceptable.
            output_point_statistics_filename (str): Name of output file for point statistics.
            output_cross_correlations_filename (str): Name of output file for cross-correlations (spatial model only).
            output_phi_filename (str): Name of output file for phi scale factors (spatial model only).
            outlier_method (str): Flag indicating which (if any) method should be to reduce the influence of outliers.
                Options are None (default), 'trim' (remove outliers) or 'clip' (Winsorise). See Notes for details.
            maximum_relative_difference (float): Maximum relative difference to allow between the two largest values
                in a timeseries. Used only if outlier_method is not None.
            maximum_alterations (int): Maximum number of trimming or clipping alterations permitted. Used only if
                outlier_method is not None.

        Notes:  # TODO: Expand notes
            statistic_definitions

            metadata

            outlier_method

        """
        print('Preprocessing')

        # Set default statistic definitions (and weights) if needed (taken largely from RainSim V3.1 documentation)
        if statistic_definitions is not None:
            pass
        elif statistic_definitions_path is not None:
            statistic_definitions = utils.read_statistic_definitions(statistic_definitions_path)
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
            statistic_definitions = utils.nested_dictionary_to_dataframe(dc, id_name, non_id_columns)

        # Check that statistics include 24hr mean, as it is currently required for calculating phi (add in if absent)
        includes_24hr_mean = statistic_definitions.loc[
            (statistic_definitions['name'] == 'mean') & (statistic_definitions['duration'] == 24)
        ].shape[0]
        if not includes_24hr_mean:
            df = pd.DataFrame({
                'statistic_id': [int(np.max(statistic_definitions['statistic_id'])) + 1], 'weight': [0],
                'duration': [24], 'name': ['mean'], 'lag': ['NA'], 'threshold': ['NA']
            })
            statistic_definitions = pd.concat([statistic_definitions, df])

        # Spatial model requires a table of metadata for points
        if self.spatial_model:
            if metadata is not None:
                pass
            else:
                metadata = pd.read_csv(metadata_path)
            metadata.columns = [column_name.lower() for column_name in metadata.columns]

        # Construct output paths
        output_point_statistics_path = os.path.join(output_folder, output_point_statistics_filename)
        if 'cross-correlation' in statistic_definitions['name'].tolist():
            output_cross_correlation_path = os.path.join(output_folder, output_cross_correlations_filename)
        else:
            output_cross_correlation_path = None
        if self.spatial_model:
            output_phi_path = os.path.join(output_folder, output_phi_filename)
        else:
            output_phi_path = None

        # Do preprocessing
        self.reference_statistics, self.phi = preprocessing.main(
            spatial_model=self.spatial_model,
            season_definitions=self.season_definitions,
            statistic_definitions=statistic_definitions,
            timeseries_format=timeseries_format,
            timeseries_path=timeseries_path,
            timeseries_folder=timeseries_folder,
            metadata=metadata,
            calculation_period=calculation_period,
            completeness_threshold=completeness_threshold,
            output_point_statistics_path=output_point_statistics_path,
            output_cross_correlation_path=output_cross_correlation_path,
            output_phi_path=output_phi_path,
            outlier_method=outlier_method,
            maximum_relative_difference=maximum_relative_difference,
            maximum_alterations=maximum_alterations,
        )

    def fit(
            self,
            output_folder,
            fitting_method='default',
            reference_statistics=None,
            reference_point_statistics_path=None,
            reference_cross_correlations_path=None,
            parameter_bounds=None,  # TODO: Use a dictionary to remove dependence on list order
            n_workers=1,
            output_parameters_filename='parameters.csv',
            output_point_statistics_filename='fitted_point_statistics.csv',
            output_cross_correlation_filename='fitted_cross_correlations.csv',
            initial_parameters=None,
            initial_parameters_path=None,
            smoothing_tolerance=0.2
    ):
        """
        Fit model parameters.

        Args:
            output_folder (str): Folder in which to save output parameters and fitted statistics files.
            fitting_method (str): Flag to indicate fitting method. Using `default` will fit each month or season
                independently. Option for `empirical_smoothing` under development (see Notes).
            reference_statistics (pandas.DataFrame): Statistics for fitting model parameters.
            reference_point_statistics_path (str): Path to file containing point statistics.
            reference_cross_correlations_path (str): Path to file containing cross-correlation statistics.
            parameter_bounds (list or dict): List of tuples of upper and lower parameter bounds in order required by
                model. If these bounds should vary seasonally then a dictionary of lists can be passed, with each key
                corresponding with a season identifier.
            n_workers (int): Number of workers (cores/processes) to use in fitting. Default is 1.
            output_parameters_filename (str): Name of output parameters file.
            output_point_statistics_filename (str): Name of output point statistics file.
            output_cross_correlation_filename (str): Name of output cross-correlation statistics file.
            initial_parameters (pandas.DataFrame): Initial parameter values to use if fitting_method is
                empirical_smoothing. If not specified then initial parameter values will be obtained using
                the default fitting method (for which no initial values are currently required).
            initial_parameters_path (str): Path to file from which initial_parameters should be read if desired.
            smoothing_tolerance (float): Permitted deviation in smoothed annual cycle of parameter values (only used
                if fitting_method is empirical_smoothing). Expressed as fraction of annual mean parameter value,
                such that 0.2 allows a +/- 20% deviation from the smoothed annual cycle for a given parameter.

        Notes:
            If `self.reference_statistics` is not None it will be given priority for use in fitting. Otherwise the
            reference statistics can be passed in as an argument or read from file(s).

            Lists of parameter bounds need to be passed in the order required by the model. For the point model this
            order is: lamda, beta, nu, eta, xi. For the spatial model this order is: lamda, beta, rho, eta, gamma,
            xi. This approach will be replaced.

            Fitting can be speeded up significantly with n_workers > 1. The maximum n_workers should be less than or
            equal to the number of cores or logical processors available.

            Empirical smoothing.  # TODO: Explain method so far

        """
        print('Fitting')

        # Read reference statistics if not available from preprocessing or passed directly
        if self.reference_statistics is not None:
            reference_statistics = self.reference_statistics
        elif reference_statistics is not None:
            pass
        else:
            # TODO: Fix read of reference statistics in utils.read_statistics (parsing of lag and threshold)
            # reference_statistics = utils.read_statistics(
            #     reference_point_statistics_path, reference_cross_correlations_path
            # )
            raise NotImplementedError

        # If bounds are passed as a list assume that they should be applied to each season
        if parameter_bounds is not None:
            if isinstance(parameter_bounds, list):
                dc = {}
                for season in self.unique_seasons:
                    dc[season] = parameter_bounds
                parameter_bounds = dc

        # Identify relevant parameters and set default bounds by season if required
        if not self.spatial_model:
            parameter_names = ['lamda', 'beta', 'nu', 'eta']
            if parameter_bounds is None:
                parameter_bounds = {}
                for season in self.unique_seasons:
                    parameter_bounds[season] = [
                        (0.00001, 0.02),    # lamda
                        (0.02, 1.0),        # beta
                        (0.1, 30),          # nu
                        (0.1, 60.0),        # eta
                    ]
            if self.intensity_distribution == 'exponential':
                parameter_names.append('xi')
                for season in self.unique_seasons:
                    parameter_bounds[season].append((0.01, 4.0))  # xi
        else:
            parameter_names = ['lamda', 'beta', 'rho', 'eta', 'gamma']  # ! ORDER OF gamma AND xi SWAPPED HERE !
            if parameter_bounds is None:
                parameter_bounds = {}
                for season in self.unique_seasons:
                    parameter_bounds[season] = [
                        (0.001, 0.05),      # lamda
                        (0.02, 0.5),        # beta
                        (0.0001, 2.0),      # rho
                        (0.1, 12.0),        # eta
                        (0.01, 500.0)       # gamma
                    ]
            if self.intensity_distribution == 'exponential':
                parameter_names.append('xi')
                for season in self.unique_seasons:
                    parameter_bounds[season].append((0.01, 4.0))  # xi

        # Construct output paths
        output_parameters_path = os.path.join(output_folder, output_parameters_filename)
        output_point_statistics_path = os.path.join(output_folder, output_point_statistics_filename)
        if self.spatial_model:
            output_cross_correlation_path = os.path.join(output_folder, output_cross_correlation_filename)
        else:
            output_cross_correlation_path = None

        # Do fitting
        parameters, fitted_statistics = fitting.main(
            season_definitions=self.season_definitions,
            spatial_model=self.spatial_model,
            intensity_distribution=self.intensity_distribution,
            fitting_method=fitting_method,
            reference_statistics=reference_statistics,
            parameter_names=parameter_names,
            parameter_bounds=parameter_bounds,
            n_workers=n_workers,
            output_parameters_path=output_parameters_path,
            output_point_statistics_path=output_point_statistics_path,
            output_cross_correlation_path=output_cross_correlation_path,
            initial_parameters=initial_parameters,
            initial_parameters_path=initial_parameters_path,
            smoothing_tolerance=smoothing_tolerance
        )
        self.parameters = parameters

    def simulate(
            self,
            discretisation_method='default',  # TODO: Probably more like simulation_type
            output_types=None,
            output_folder=None,
            output_subfolders='default',
            output_format='txt',
            parameters=None,
            points=None,
            catchments=None,
            grid=None,  # TODO: Consider whether to add grid_output_prefix
            epsg_code=None,
            cell_size=None,
            dem=None,
            phi=None,
            simulation_length=30,
            number_of_realisations=1,
            timestep_length=1,
            start_year=2000,
            calendar='gregorian',
            random_seed=None,
            additional_output=True,
    ):
        """
        Simulate realisation(s) of NSRP process.

        Args:
            discretisation_method (str): Flag indicating whether to discretise rainfall series for output (`default`)
                or to calculate total depth for each event (`event_totals`), as required by Kim and Onof (2020)
                shuffling method (not yet fully implemented).
            output_types (list of str): Types of output (discretised) rainfall required. Options are `point`,
                `catchment` and `grid`.
            output_folder (str): Path to folder in which output files should be written.
            output_subfolders (str or dict): Sub-folder in which to place each output type. If `default` then the
                following dictionary is used: dict(point='point', catchment='catchment', grid='grid'). If None then
                all output files are written to output_folder.
            output_format (str): Flag indicating output file format for point and catchment output. Current
                option is `txt`. Gridded output will be written in NetCDF format.
            parameters (pandas.DataFrame or str): Dataframe of parameters to use in simulation (or path to file
                containing parameters). Optional, as self.parameters will be used by default (and take precedence) if it
                is not None (i.e. if the self.fitting() method has been run).
            points (pandas.DataFrame or str): Metadata dataframe of points for which output is required (or path to
                file). See Notes.
            catchments (geopandas.GeoDataFrame or str): Geodataframe containing catchments for which output is required
                (or path to catchments shapefile).
            grid (dict or str): Specification of output grid to use for both gridded output (if required). This grid
                is also used to support catchment output. Dictionary keys use ascii raster header keywords, e.g.
                dict(ncols=10, nrow=10, ...). Use xllcorner and yllcorner, as well as lowercase for each keyword.
                If None then a grid is defined to encompass catchment locations using cell_size argument. Path to an
                ascii raster file to use as a template for the grid can be given instead.
            epsg_code (int): EPSG code for projected coordinate system used for domain (used explicitly if catchment or
                grid output is required).
            cell_size (float): Cell size to use if grid is None but a grid is needed for gridded output and/or catchment
                output.
            dem (xarray.DataArray or str): Digital elevation model (DEM) [m] as data array or ascii raster file path.
            phi (pandas.DataFrame or str): Dataframe containing phi scale factor at point locations
                (from self.preprocessing() method). Path to phi file can be passed. If self.phi is not None (i.e.
                preprocessing method has been run) then self.phi is given precedence.
            simulation_length (int): Number of years to simulate in one realisation.
            number_of_realisations (int): Number of realisations to simulate.
            timestep_length (int): Timestep of output [hr]. Default is 1 (hour).
            start_year (int): Start year of simulation.
            calendar (str): Flag to indicate whether `gregorian` (default accounting for leap years) or `365-day`
                calendar should be used.
            random_seed (int): Seed to use in random number generation.
            additional_output (bool): Flag to write additional output files to output_folder. These files are
                catchment weights as ascii rasters and phi grid as ascii raster.
            # TODO: Implement additional output and include also random seed (entropy attribute of SeedSequence)

        Notes:
            Though gridded output is calculated (if output_types includes `grid`) it is not yet available to write (i.e.
            under development).

            Dataframe of metadata for points should contain fields (columns) for ...  # TODO: Complete description

            The code currently calculates catchment weights and performs interpolation of phi. Features could be added
            for these variables to be passed directly as arguments.

            Point metadata dataframe assumed to have a `Point_ID` field that can be sued to identify points.
            Catchment shapefile or geodataframe assumed to have an `ID` field that can be used to identify catchments.
            Both point and catchment metadata are assumed to have a `Name` field for use as a prefix in file naming.
            Point (single site) simulations and grid output are assumed not to need a prefix.

        """
        print('Simulating')

        # TODO: Implement output for catchment_weights_output_folder and phi_output_path - currently not implemented
        # TODO: Ensure that 'final' parameters are used e.g. parameters.loc[parameters['stage'] == 'final']

        # Make output folders if required
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if output_subfolders == 'default':
            if self.spatial_model:
                output_subfolders = dict(point='point', catchment='catchment', grid='grid')
            else:
                output_subfolders = dict(point='')
        if isinstance(output_subfolders, dict):
            for output_type, output_subfolder in output_subfolders.items():
                if not os.path.exists(os.path.join(output_folder, output_subfolder)):
                    os.mkdir(os.path.join(output_folder, output_subfolder))

        # Ensure valid output types
        # TODO: Expand checks on user input arguments
        if not self.spatial_model:
            output_types = ['point']

        # Get parameters if required
        if self.parameters is not None:
            parameters = self.parameters
        elif isinstance(parameters, str):
            parameters = utils.read_csv_(parameters)

        # Get DEM if required
        if isinstance(dem, str):
            dem = utils.read_ascii_raster(dem)

        # Output location details (grid must be defined or derived for catchment output)
        if self.spatial_model:
            if 'point' in output_types:
                if isinstance(points, str):
                    points = utils.read_csv_(points)
            if 'catchment' in output_types:
                if isinstance(catchments, str):
                    catchments = geopandas.read_file(catchments)
                    catchments.columns = [column_name.lower() for column_name in catchments.columns]
            if ('grid' in output_types) or ('catchment' in output_types):
                if isinstance(grid, str):
                    grid = utils.grid_definition_from_ascii(grid)
                else:
                    grid = utils.define_grid_extent(catchments, cell_size, dem)
                cell_size = grid['cellsize']

        # Known phi values at point locations
        if self.spatial_model:
            if self.phi is not None:
                phi = self.phi
            elif isinstance(phi, str):
                phi = utils.read_csv_(phi)

        # Do simulation
        simulation.main(
            spatial_model=self.spatial_model,
            intensity_distribution=self.intensity_distribution,
            discretisation_method=discretisation_method,
            output_types=output_types,
            output_folder=output_folder,
            output_subfolders=output_subfolders,
            output_format=output_format,
            season_definitions=self.season_definitions,
            parameters=parameters,
            points=points,
            catchments=catchments,
            grid=grid,
            epsg_code=epsg_code,
            cell_size=cell_size,
            dem=dem,
            phi=phi,
            simulation_length=simulation_length,
            number_of_realisations=number_of_realisations,
            timestep_length=timestep_length,
            start_year=start_year,
            calendar=calendar,
            random_seed=random_seed,
            additional_output=additional_output
        )

    @property
    def unique_seasons(self):
        """list of int: Unique season identifiers."""
        return list(set(self.season_definitions.values()))


