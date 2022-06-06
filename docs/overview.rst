Overview
========

RWGEN is a stochastic spatiotemporal Rainfall and Weather GENerator package. 

RWGEN is based on the Neyman-Scott Rectangular Pulse (NSRP) rainfall model. 
Weather variables (e.g. temperature) are simulated using autoregressive models. 
RWGEN calculates potential evapotranspiration (PET) using the FAO56 
Penman-Moneith method.

The RWGEN package supports both single site (point) and spatial simulations. 
It contains functionality to support data pre-processing, model fitting, 
simulation and model evaluation.


Structure
---------

The core of RWGEN is the NSRP rainfall model. The NSRP model can be run in a
standalone mode.

The weather model (with integrated PET calculation) can only be run alongside 
the NSRP rainfall model.

Simple diagram/figure to show this relationship.

Could also show as a workflow:
    * Initialise rainfall model
    * Rainfall model data preprocessing (for fitting)
    * ...

Model Options
-------------

Rainfall model:
    * Season definitions
    * Intensity distribution

    * Data completeness and outliers
    * Statistics (and weights)

    * Fitting methods
    * Parameter bounds and fixing

    * Output types
    * DEM
    * Simulation length, timestep and calendar (start)
    * Number of realisations
    * Random seed (reproducibility)

    * Simulation statistics, maxima (AMAX) and DDFs

    * Plotting...

Using the NSRP rainfall model
-----------------------------

The main way:

.. ipython:: python

    import rwgen

    rainfall_model = rwgen.RainfallModel()
    
    rainfall_model.preprocess()
    rainfall_model.fit()
    rainfall_model.simulate()
    rainfall_model.postprocess()
    rainfall_model.plot() 

e.g.
rainfall_model.fit()
rainfall_model.plot_fit()
rainfall_model.fit()
rainfall_model.plot_fit()
rainfall_model.simulate()
...

Workflows
Initialisation options
Main way to do things
Setting what is needed if skipping a step(s)

Using the full weather generator
--------------------------------

For example:

.. ipython:: python

    import rwgen

    wg = rwgen.WeatherGenerator()
    
    wg.initialise_rainfall_model()  # convenient if a model or pickled model could be passed
    # wg.set_rainfall_model(rainfall_model)
    
    wg.intialise_weather_model()
    wg.weather_model.fit()
    # wg.set_weather_model(weather_model)
    
    wg.simulate()

    wg.rainfall_model.postprocess()
    wg.rainfall_model.plot()
    wg.weather_model.postprocess()
    wg.weather_model.plot() 









