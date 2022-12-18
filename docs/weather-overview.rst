Weather Generator Overview
==========================

There are three key "objects" available within RWGEN:

    1. Weather generator
    2. Rainfall model
    3. Weather model

The weather generator is used to perform coupled simulations of rainfall and
other weather variables (e.g. temperature). To do this, the weather generator
uses the other two objects as its component models:

    1. Rainfall model - Neyman-Scott Rectangular Pulse (NSRP) process
    2. Weather model - regression equations

Regressions in the weather model are conducted according to wet/dry
transition states. This means that the weather model depends on the output of
the rainfall model (whereas the rainfall model does not depend on the weather
model). The rainfall model may be used as a standalone model, but the weather
model will typically be used as a component of the weather generator.

The rest of this page explains the core concepts of the weather model
component. The rainfall model is described on the :doc:`rainfall-overview` page,
while an overview of usage of the weather generator is provided on the
:doc:`weather-workflow` page.

Single Site Weather Model
-------------------------

The weather model largely follows the structure described by
`Kilsby et al. (2007)`_ and `Jones et al. (2010)`_.

.. _Kilsby et al. (2007): https://doi.org/10.1016/j.envsoft.2007.02.005
.. _Jones et al. (2010): https://doi.org/10.5281/zenodo.7357057

Transformations
~~~~~~~~~~~~~~~

Before fitting the regression models, transformations are used to help
non-normal variables better approximate a normal distribution. The Box-Cox
transformation is used, apart from for sunshine duration (for which a Beta
distribution was selected). The weather input time series are additionally
standardised/scaled to follow a standard normal distribution (mean of 0 and
standard deviation of 1).

Regressions
~~~~~~~~~~~

Regression equations are used to model temperature, vapour pressure, sunshine
duration and wind speed based on their previous values and the precipitation
"transition state". The transition state describes whether a day and the
preceding day are both wet, dry or different to each other. Five transition
states are used in the model:

    - Dry to dry (DD)
    - Dry to wet (DW)
    - Wet to dry (WD)
    - Wet to wet (WW)
    - Dry to dry to dry (DDD)

The final state listed above (DDD) considers the previous two days, rather than
just the preceding day. This helps to better simulate longer dry spells.

.. note::

    While the weather model runs on a daily basis, simple sub-daily
    disaggregation methods are included in the package. The timestep of the
    weather model output can therefore match that of the rainfall model.

Both average temperature and diurnal temperature range are simulated. Daily
minimum and maximum temperatures can be derived from these variables (and
written as outputs). Temperature is simulated first (after precipitation),
with the other weather variables following.

The precise form of the regression equation used varies depending on the
variable and transition state. The equations all include an autoregressive
(lag-1) term and sometimes a term related to another variable. For example,
when simulating average temperature, a term depending on precipitation is
included in the regression equation if either the current or previous day
are classified as wet.

Regression coefficients are identified using ordinary least squares.

An error/noise term adds the random component to the regression equations.
This random component is simulated from a standard normal distribution and
scaled according to the standard error of the regression equation.

Spatial Weather Model
---------------------

The spatial weather model is very similar to its single site counterpart. The
form of the regression equations is the same, but it is possible for some of
the parameters to vary spatially. It is also possible to get weather model
output for any location in the domain, even if the location does not correspond
to an input weather station.

However, at present, only the standard error (used to scale the error/noise
term) can vary spatially for a given application. This means that, currently,
the model only uses uniform regression coefficients across the domain. This
will be updated in future.

Future versions will also include the ability to simulate spatial fields for
the error/noise term. Currently the model uses a single random number across
the domain, although this number is scaled according to the standard error at
each location.
