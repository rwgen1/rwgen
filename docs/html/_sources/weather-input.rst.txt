Input Data
==========

This page is under development.

Please see the example notebooks scripts (see :doc:`examples`) for the daily
time series file format (and the metadata file format for a spatial model).

Abbreviated variable names used for key column headers and associated units in
the time series files are:

    - Precipitation (mm/day): ``prcp``
    - Daily minimum temperature (°C): ``temp_min``
    - Daily maximum temperature (°C): ``temp_max``
    - Daily average temperature (°C): ``temp_avg``
    - Vapour pressure (kPa): ``vap_press``
    - Sunshine duration (hours): ``sun_dur``
    - Wind speed (m/s): ``wind_speed``

In the pre-processed :doc:`uk-data`, a couple of additional columns are present
but not used currently:

    - Relative humidity (%): ``rel_hum``
    - Daily mean temperature (°C): ``temp_mean``

.. note::

    Daily mean temperature (``temp_mean``) is the mean of an hourly
    temperature series if available. Daily average temperature
    (``temp_avg``) is the mean of the daily minimum and maximum temperatures.
    The latter (``temp_avg``) is currently used in the model.

Further details will be added.
