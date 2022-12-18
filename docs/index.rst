RWGEN
=====

RWGEN is a stochastic spatiotemporal Rainfall and Weather GENerator based on
the Neyman-Scott Rectangular Pulse (NSRP) rainfall model.

RWGEN supports both single site (point) and spatial simulations. It can
conduct data pre-processing, model fitting, simulation and post-processing.

A `demo`_ version of the rainfall model component can be tried in the browser
without the need for any Python knowledge or installation. To run the model
with your own data, please see the :doc:`installation` page.

.. _demo: https://mybinder.org/v2/gh/davidpritchard1/rwgen-demo/HEAD

The model and documentation are still under active development - please check
back regularly for updates. The package is new, so please also check the model
behaviour and outputs when using it in a new application.


.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    installation.rst
    examples.rst


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
