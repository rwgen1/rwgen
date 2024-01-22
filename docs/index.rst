RWGEN
=====

Welcome to the RWGEN documentation!

RWGEN is a stochastic spatiotemporal Rainfall and Weather GENerator built
around the Neyman-Scott Rectangular Pulse (NSRP) rainfall model. The package
provides an open Python implementation of the NSRP model coupled with (i) an
autoregressive model for other weather variables and (ii) FAO56 Penman-Monteith
potential evapotranspiration calculations.

RWGEN can currently conduct data pre-processing, model fitting, simulation and
selected post-processing. Both single site (point) and spatial simulations are
supported. The model and documentation are new and under active development -
please check back regularly for updates.


.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    installation.rst
    examples.rst
    uk-data.rst


.. toctree::
    :maxdepth: 1
    :caption: Rainfall Model

    rainfall-overview.rst
    rainfall-workflow.rst
    rainfall-methods.rst
    rainfall-input.rst


.. toctree::
    :maxdepth: 1
    :caption: Weather Generator

    weather-overview.rst
    weather-workflow.rst
    weather-input.rst


.. toctree::
    :maxdepth: 1
    :caption: Reference

    weather-generator-api.rst
    rainfall-api.rst
    weather-model-api.rst
