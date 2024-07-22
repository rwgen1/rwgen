RWGEN
=====

Welcome to the RWGEN repository!

RWGEN is a stochastic spatiotemporal Rainfall and Weather GENerator built
around the Neyman-Scott Rectangular Pulse (NSRP) rainfall model. The package
provides an open Python implementation of the NSRP model coupled with (i) an
autoregressive model for other weather variables and (ii) FAO56 Penman-Monteith
potential evapotranspiration calculations.

RWGEN can currently conduct data pre-processing, model fitting, simulation and
selected post-processing. Both single site (point) and spatial simulations are
supported. See below for more details on the package status, getting started,
documentation and ways to get in touch!

|

.. image:: ./docs/_static/amax_example.png
  :width: 600
  :align: center

*Example of observed and simulated annual maximum rainfall for different
durations at Heathrow, UK.*

Status
------

The package is new and still undergoing development. It should be considered
as a pre-release for testing at present. Some features may not work properly,
but the core functionality should behave reasonably in most cases. Development
of the rainfall model component is ahead of the weather model component, but
the outputs and behaviour of both should be checked carefully if used in an
application. Parts of the interface may still change.

Getting Started
---------------

Documentation
~~~~~~~~~~~~~

Package `documentation`_ is available that explains the core features of RWGEN
and typical use. Please check back for updates, as work on the documentation
is ongoing.

.. _documentation: https://rwgen1.github.io/rwgen/html/index.html

Installation
~~~~~~~~~~~~

Installation is described in the `documentation`_.

Examples
~~~~~~~~

Examples are available in the GitHub repository (see the ``examples`` folder
above and the `documentation`_ for further explanation). Both Jupyter notebooks
and scripts are available.

Demo Notebooks
~~~~~~~~~~~~~~

A `demo`_ version of the rainfall model component can be tried in the browser
without the need for any Python knowledge or installation. To run the model
with your own data, please see the `documentation`_.

.. _demo: https://mybinder.org/v2/gh/davidpritchard1/rwgen-demo/HEAD

Following the link above should load Jupyter in the default browser. If you
then navigate into the ``examples`` folder you should find sub-folders
containing the Jupyter notebooks. These notebooks contain explanations and
example usages of the rainfall model.

Note that the load times for the demo notebook can vary, as it is hosted on the
free `mybinder`_ service. If it is being slow (more than a few minutes), try
closing the tab and reopening (or try again later).

.. _mybinder: https://mybinder.org/

Note also that the demo version currently uses an older version of the rainfall
model with some small differences in the interface.

UK Input Data
~~~~~~~~~~~~~

Selected open data have been pre-processed to help facilitate RWGEN
applications in the UK. Please see the `documentation`_ for details and
download links.

Contributing
------------

Please feel free to get in touch if you would like to discuss ways to
contribute to the code. All questions, feedback, ideas, problems, bug reports
and other comments etc are also much appreciated. For most things (including
questions and problems) it is best to start a conversation by raising an issue
on GitHub if possible.

Acknowledgments
---------------

RWGEN was largely developed during an `Embedded Researcher project`_ as part
of the UK Climate Resilience Programme funded by UK Research and Innovation
(UKRI). The authors are also grateful to the researchers whose work over many
years has formed the basis for RWGEN, as well as to the developers who have
contributed to the code on which RWGEN depends.

.. _Embedded Researcher project: https://www.ukclimateresilience.org/projects/facilitating-stochastic-simulation-for-uk-climate-resilience/

License
-------

Distributed under the GNU GPL V3 license. See LICENSE for more details.
