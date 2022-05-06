import os

from . import fitting
from . import utils


class Model:
    """
    Pre-processing, fitting and simulation of point and spatial Neyman-Scott Rectangular Pulse (NSRP) models.

    """

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

        #: pandas.DataFrame: Parameters for time series simulation.
        self.parameters = None

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
                independently. Option for `nonparametric_smoothing` under development (see Notes).
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
                nonparametric_smoothing. If not specified then initial parameter values will be obtained using
                the default fitting method (for which no initial values are currently required).
            initial_parameters_path (str): Path to file from which initial_parameters should be read
            smoothing_tolerance (float): Permitted deviation in smoothed annual cycle of parameter values (only used
                if fitting_method is nonparametric_smoothing). Expressed as fraction of annual mean parameter value,
                such that 0.2 allows a +/- 20% deviation from the smoothed annual cycle for a given parameter.

        Notes:
            If `self.reference_statistics` is not None it will be given priority for use in fitting. Otherwise the
            reference statistics can be passed in as an argument or read from file(s).

            Lists of parameter bounds need to be passed in the order required by the model. For the point model this
            order is: lamda, beta, nu, eta, xi. For the spatial model this order is: lamda, beta, rho, eta, gamma,
            xi. This approach will be replaced.

            Fitting can be speeded up significantly with n_workers > 1. The maximum n_workers should be less than or
            equal to the number of cores or logical processors available.

            Non-parametric smoothing.  # TODO: Explain method so far

        """
        # Read reference statistics if not available from preprocessing or passed directly
        if self.reference_statistics is not None:
            reference_statistics = self.reference_statistics
        elif reference_statistics is not None:
            pass
        else:
            reference_statistics = utils.read_statistics(
                reference_point_statistics_path, reference_cross_correlations_path
            )

        # If bounds are passed as a list assume that they should be applied to each season
        if parameter_bounds is not None:
            if isinstance(parameter_bounds, list):
                _ = {}
                for season in self.unique_seasons:
                    _[season] = parameter_bounds
                parameter_bounds = _

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
        output_cross_correlation_path = os.path.join(output_folder, output_cross_correlation_filename)

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

    @property
    def unique_seasons(self):
        """list of int: Unique season identifiers."""
        return list(set(self.season_definitions.values()))


