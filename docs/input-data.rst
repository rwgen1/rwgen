Input Data
==========

Example files for a spatiotemporal NSRP model are given in the
``examples/stnsrp/input`` folder in the RWGEN root folder. In the description
below it is assumed that we are located in this folder (current directory).

The minimal requirements for setting up a spatiotemporal rainfall model
using gauge data are:

    1. A csv file containing gauge metadata - see ``gauge_metadata.csv``
    2. Files containing gauge time series (one file per gauge) - see
       gauge_data directory

A DEM is optional but good if available.

A polygon shapefile containing catchment/sub-catchment boundaries is needed if 
catchment-average outputs are requested.

Gauge Metadata File
-------------------

The gauge metadata csv file has mandatory columns (data types) of:

    1. ``Point_ID`` (integer) - unique integer for each gauge
    2. ``Easting`` (integer or float) - units = metres
    3. ``Northing`` (integer or float) - units = metres
    4. ``Name`` (string or integer) - to help identify the relevant gauge time 
       series file

It is assumed that the ``Name`` field corresponds with a gauge time series file
name (after ``.csv`` is added to the ``Name entry``). For example,
``Burgkunstadt`` corresponds with the file ``./gauge_data/Burgkunstadt.csv``.
Avoid spaces and special characters.

``Elevation`` (integer or float) in metres is not technically mandatory but
good to include if possible (depending on region).

Other fields in the gauge metadata csv file are not currently needed or used 
(but can be added for reference).

Gauge Time Series Files
-----------------------

The example uses csv format for time series files. Required columns are:

    1. ``DateTime`` - ``dd/mm/yyyy hh:mm`` format (i.e.
       ``'%d/%m/%Y %H:%M'``), e.g. ``01/01/2005 00:00``
    2. ``Value`` (float) - mm/timestep

Missing data can be represented as -999 (the code will set any negative values 
as missing).

All gauge time series files should be placed in the same directory, e.g. 
``./gauge_data``

DEM
---

Supplying a DEM is optional but useful if "ungauged" locations are to be 
sampled or grid-based / catchment-average outputs are required (depending on 
region to some extent).

If supplying a DEM, the gauge metadata file should contain an ``Elevation``
column.

The DEM should be in ascii raster format and it should not contain any missing 
values (for now at least).

DEM resolution can be relatively high (e.g. 100 m in this example of a 
~6000 km**2 overall domain). While a 100 m resolution DEM might be supplied as
input, internally the model simulation can take place (and produce outputs) at
a coarser resolution (e.g. 1 km).

Unless output on a specific grid is required, precise DEM extent does not 
matter but it should not be much larger than it really needs to be.

Catchment(s) Shapefile
----------------------

A polygon shapefile of catchment(s) is needed if catchment-average output is
requested (otherwise it is not required).

Each polygon is treated independently by the code, so the shapefile can contain 
any polygons of interest (i.e. it does not matter if they overlap). Output time 
series will be generated for each polygon in the shapefile.

It is currently assumed that the shapefile contains the following fields:
    1. ``ID`` (integer) - identifier for catchment / sub-catchment
    2. ``Name`` (string) - for use in writing output files

Other fields may be present in the shapefile - they will be ignored.
