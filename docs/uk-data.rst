UK Input Data
=============

Open gauge/station data have been collated to support stochastic rainfall and
weather modelling in the UK. The data are formatted ready for use with the
RWGEN modelling package. Details of the data available so far are given below.

.. note::

    Please consider the data to be preliminary at present. Changes may be made
    if any issues are found in the data processing (none so far) and following
    further quality checks.

Available Data
--------------

Currently available data are gauge/station time series data processed from the
UKMO `MIDAS-Open`_ collection:

    - Hourly rainfall time series for 331 gauges
    - Daily weather time series for 1000 stations

.. _MIDAS-Open: https://catalogue.ceda.ac.uk/uuid/dbd451271eb04662beade68da43546e1

The data have been compiled and reformatted using the *qc-version-1* data in
the `MIDAS-Open`_ datasets. No further quality control has been applied so far.

Note that the periods of record and completeness vary notably.

The daily weather time series include precipitation, temperature, vapour
pressure, wind speed and sunshine duration. Not all variables are available at
all stations or throughout the entire record periods. The data have been
compiled from different datasets within the `MIDAS-Open`_ collection - a
mixture of daily and (aggregated) hourly time series.

A metadata file summarises basic information (location, UKMO identifier etc)
for the gauges/stations.

Downloads
---------

The formatted `MIDAS-Open`_ data are now available for download from the UKCEH
Environmental Information Data Centre (EIDC). Click on the "download data"
button on the `dataset landing page`_ or see the following direct links:

    - `Hourly rainfall data`_
    - `Daily weather data`_
    - `Gauge/station metadata`_

.. _dataset landing page: https://catalogue.ceh.ac.uk/documents/44c577d3-665f-40de-adce-74ecad7b304a
.. _Hourly rainfall data: https://catalogue.ceh.ac.uk/datastore/eidchub/44c577d3-665f-40de-adce-74ecad7b304a/historical_hourly-rainfall/
.. _Daily weather data: https://catalogue.ceh.ac.uk/datastore/eidchub/44c577d3-665f-40de-adce-74ecad7b304a/historical_daily-weather/
.. _Gauge/station metadata: https://catalogue.ceh.ac.uk/datastore/eidchub/44c577d3-665f-40de-adce-74ecad7b304a/station_metadata.csv

.. note::

    The EIDC dataset entry contains an archived version of the model code, but
    this has been superseded. Instead, please refer to the GitHub repository,
    which contains the most recent code.

Future Data
-----------

Additional data to be added in the future include:

    - Updated gauge/station time series supplemented by point extractions from
      `HadUK-Grid`_ to supplement missing precipitation and temperature data
    - Change factors based on `UKCP18-Local`_ to facilitate climate change scenarios

.. _HadUK-Grid: https://catalogue.ceda.ac.uk/uuid/4dc8450d889a491ebb20e724debe2dfb

.. _UKCP18-Local: https://catalogue.ceda.ac.uk/uuid/ad2ac0ddd3f34210b0d6e19bfc335539

Once the above are available and stable, the data will be moved to a data repository
and a DOI assigned.

Licence Information
-------------------

Contains public sector information licensed under the Open Government Licence v3.0.
