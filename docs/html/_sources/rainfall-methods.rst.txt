Method Options
==============

This page provides an overview of the options available in each of the key
methods (functions) of the rainfall model. For detailed information on each of
the method arguments, please see the RainfallModel API documentation pages. For
an overview of the methods, take a look at the rainfall model
:doc:`rainfall-workflow` page.

Initialisation
--------------

The initialisation step creates an instance of the rainfall model and sets some
basic options.

When initialising the rainfall model it is compulsory to specify:

    - Whether the model is for a single site (point) or a spatial model
    - A name for the model (e.g. gauge/site or catchment name) for output
      purposes
    - The path to an input time series file (or a folder containing input
      series for the spatial model).

Note also that we have to specify a metadata file detailing the input
gauges/sites when initialising a spatial model.

Additional options are available for:

    - Season definitions - default is a monthly model
    - Probability distribution to use for rain cell intensity (defaults to
      exponential
    - Path to output folder
    - Whether or not we wish to override the default statistic definitions
      to be used in fitting

The :doc:`rainfall-api` provides further details of all defaults, including
tables of the statistics and weights that the model uses in fitting.

Preprocessing
-------------

The preprocessing step calculates reference statistics against which the model
parameters should be fitted.

Options available in this step include:

    - Using a subset of the record period (default is to use full record)
    - Using only months/seasons exceeding a completeness threshold (default is
      to use all data)
    - Trimming or clipping time series to reduce the influence of outliers
      (default is not to trim/clip)
    - Annual maxima (AMAX) extraction for a set of durations
    - Output file names (default is reference_statistics.csv)

In addition, there is an option to calculate "pooled" statistics for a spatial
model. If this option is applied (which it is by default in a spatial model)
then input time series are scaled to have a uniform mean and then pooled before
calculation of reference statistics. This helps to speed up the fitting step,
as fewer calculations are required. A simple exponential function is also
fitted to the reference spatial cross-correlations to help further reduce the
number of calculations in fitting.

Fitting
-------

The fitting step finds a parameter set that should give as good a fit as
possible between a model simulation and the reference statistics.

Several options can be specified when fitting:

    - Parameter bounds (if you wish to change the defaults)
    - Fixed parameters (if you wish to specify any)
    - Output file names
    - Whether to fit parameters needed for the `Kim and Onof (2020)`_
      "shuffling" method (see rainfall model :doc:`rainfall-overview`)
    - Whether to iteratively correct for residual bias between simulated and
      fitted (analytical) 24H dry probability (see below)

There can be a residual bias in the simulated dry probability compared with
the fitted (analytical) equivalent in the Neyman-Scott Rectangular Pulse (NSRP)
rainfall model. This is also apparent in other implementations of the NSRP
model and requires further investigation. For now, an iterative correction
method has been implemented to reduce this bias (using simulation to identify
the approximate bias and then adjusting the reference statistics before
re-estimating the parameters).

.. _Kim and Onof (2020): https://doi.org/10.1016/j.jhydrol.2020.125150

Simulation
----------

General options available in time series simulation include:

    - Simulation length (in years)
    - Number of realisations to simulate
    - Length of output timestep (in hours)
    - Calendar (i.e. account for leap years or not - default yes) and notional
      start year (default is 2000)
    - Random number seed (optional but can be used for reproducibility)
    - Whether or not to apply the `Kim and Onof (2020)`_ "shuffling" method
      (see also rainfall model :doc:`rainfall-overview`)

There are some additional options for a spatial model:

    - Output types: point/site or catchment-average (gridded output under
      development)
    - Resolution at which to discretise the domain for catchment-average or
      gridded outputs
    - Whether to use a DEM to help interpolate the “scale factor” Φ (phi) to
      ungauge locations
