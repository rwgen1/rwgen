Workflow
========

The weather generator workflow is for performing coupled simulations of 
rainfall and other weather variables (e.g. temperature). Potential
evapotranspiration can also be calculated.

It is best to read the rainfall model :doc:`rainfall-workflow` page before
reading this page.

Example notebooks are under development.

Basic Workflow
--------------

The basic workflow for the weather generator consists of the following steps:

    1. Initialise a weather generator
    2. Initialise the rainfall model component of the weather generator
    3. Perform preprocessing and fitting for the rainfall model
    4. Initialise the weather model component of the weather generator
    5. Perform preprocessing and fitting for the weather model
    6. Simulate rainfall and weather variables
    7. Postprocess simulation output to calculate/extract relevant quantities

.. note::

    Preprocessing and fitting for the rainfall model can be replaced by
    setting the required reference statistics and/or parameters. This
    functionality is not yet implemented for the weather model.

.. note::

    Postprocessing is currently only available for the rainfall model.

Example
-------

The basic workflow can be carried out using a script like the following (for a
single site model)::

    import rwgen
    
    # Initialise weather generator
    wg = rwgen.WeatherGenerator(
        spatial_model=False,
        project_name='brize-norton',
        latitude=51.758,
        longitude=-1.578,
        easting=429124,
        northing=206725,
        elevation=82.0,
    )

    # Initialise rainfall model component
    wg.initialise_rainfall_model(
        input_timeseries='/path/to/input_timeseries.csv',
    )

    # Calculate rainfall reference statistics from gauge time series
    wg.rainfall_model.preprocess()

    # Fit rainfall model parameters using reference statistics
    wg.rainfall_model.fit()
    
    # Initialise weather model component
    wg.initialise_weather_model(
        input_timeseries='/path/to/input_timeseries',
    )
    
    # Calculate weather reference statistics from station time series
    wg.weather_model.preprocess()
    
    # Fit weather model parameters using reference statistics
    wg.weather_model.fit()

    # Simulate one realisation of 1000 years at an hourly timestep
    wg.simulate(
        simulation_length=1000,
        n_realisations=1,
        timestep_length=1,
    )

    # Calculate/extract statistics from simulated rainfall time series (e.g. AMAX, DDF)
    wg.rainfall_model.postprocess(
        amax_durations=[1, 6, 24],  # durations in hours
        ddf_return_periods=[20, 50, 100]  # return periods in years
    )

Explanation
-----------

In the example above we begin by initialising a weather generator object. Some
basic options need to be specified at this point, including whether or not the
model is for a single site and its coordinates.

Next we initialise the rainfall model as a component of the weather generator.
The ``wg`` weather generator object now has a ``rainfall_model`` attribute.
All of the rainfall model methods and attributes can be accessed as usual, but
now via e.g. ``wg.rainfall_model.preprocess()`` rather than just
``rainfall_model.preprocess()``.

.. note::

    If we want to use pre-existing reference statistics or parameters then
    they can be loaded via ``wg.rainfall_model.set_statistics()`` and
    ``wg.rainfall_model.set_parameters()`` calls.

Now that the rainfall model component is ready we can initialise the weather
model component. For a single site model, the only compulsory argument is a
path to an input ``.csv`` file containing weather time series.

The weather model has ``preprocess()`` and ``fit()`` methods, as per the
rainfall model. The former method calculates statistics and performs
transformations on the input weather series, while the latter fits the
regression models.

At this point we are now ready to simulate via a call to ``wg.simulate()``. By
default the model will try to simulate temperature, humidity, sunshine duration,
wind speed and potential evapotranspiration. See the
:doc:`weather-generator-api` and :doc:`weather-model-api` pages for further
details.

