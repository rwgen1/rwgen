.. |rwgen_version| replace:: ``rwgen-0.0.6``

.. |example_folder| replace:: ``H:/RWGEN/rwgen-0.0.6``

.. |example_folder_nonliteral| replace:: H:/RWGEN/rwgen-0.0.6

.. _zip: https://github.com/davidpritchard1/rwgen/archive/refs/tags/v0.0.6.zip

Installation
============

The instructions below are for installing RWGEN as a developer on a Windows
operating system. Some steps will not be needed once the package has been
released on PyPI. A note on updating an existing RWGEN version is also given
below - see :ref:`Installing RWGEN`.

Python and the RWGEN dependencies can be installed using the `conda`_ package
manager. `Miniconda`_ provides a lightweight implementation.

.. _conda: https://conda.io/projects/conda/en/latest/user-guide/index.html
.. _Miniconda: http://conda.pydata.org/miniconda.html

Installing Miniconda
--------------------

The conda website provides `instructions`_ for installing Miniconda. 
Installation on Windows involves downloading and running an `installer`_ (use 
the installer for the most recent Python version).

.. _instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
.. _installer: https://docs.conda.io/en/latest/miniconda.html#windows-installers

.. note::

    The second step of the instructions (verify your installer hashes) can
    typically be skipped.

Updating and Configuring Conda
------------------------------

A few settings need to be adjusted after installing conda. To do this open the 
Anaconda Prompt (press the Windows button and then start typing 
``anaconda prompt``). This should give you a prompt that looks similar to the
Windows Command Prompt.

Update conda by typing the following (enter ``y`` when prompted)::

    conda update -n base -c defaults conda

Next add the `conda-forge`_ channel with the two commands::

    conda config --add channels conda-forge
    conda config --set channel_priority strict

The conda-forge channel is needed to help manage dependencies.

.. _conda-forge: https://conda-forge.org/

Getting the RWGEN Files
-----------------------

Download the |rwgen_version| files as a `zip`_ file (if you do not have it
already) and unzip. It is recommended to create a ``RWGEN`` folder somewhere
- e.g. ``H:/RWGEN``.

Note that Windows ``Extract All...`` adds the zip file name (e.g.
|rwgen_version|) to the default location for unzipping. This can be deleted, as
the zip file already contains an |rwgen_version| folder housing all other
files.

If the target folder is ``H:/RWGEN`` then we should end up with a folder
|example_folder|, which contains sub-folders ``docs, examples, rwgen``
and some files (e.g. ``setup.py, environment.yml, ...``).

Creating the Conda Environment
------------------------------

In the Anaconda Prompt (or open a new one if closed - see the
`Updating and Configuring Conda`_ section) and navigate (at the prompt) to the
unzipped |rwgen_version| folder.

For example, to change from the directory ``C:/`` to the directory 
|example_folder|, first switch drives by typing::

    H:

Then to change to |example_folder| enter

.. parsed-literal::

    cd\  |example_folder_nonliteral|\

This folder should contain a file called ``environment.yml``, which lists all
of the dependencies required by RWGEN.

.. note::
    
    If you already have an environment for rwgen installed that you want to
    update, it may be easiest to remove the existing environment and reinstall.
    The existing environment can be removed at this point by using 
    ``conda env remove --name rwgen`` (if your existing environment name is 
    ``rwgen``. You can now proceed with the rest of the instructions.

Create a specific conda `environment`_ for RWGEN by typing::

    conda env create --name rwgen --file environment.yml

The environment name can be set to something other than ``rwgen`` if preferred.
It may take a few minutes for the environment to be created as conda installs
the dependencies that we need.

Activate the `environment`_ before continuing::

    conda activate rwgen

.. _environment: https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html

.. note::

    If we want to deactivate the rwgen environment at any point we can just
    enter ``conda deactivate``.

Installing RWGEN
----------------

.. note::

    If you already have a version of RWGEN installed in the activated
    environment but would like to update it to a newer version, first
    uninstall the current version by entering ``pip uninstall rwgen``. Then
    proceed to install the new version as below.

At the Anaconda Prompt (while still in the |rwgen_version| folder), type the
following (including the ``.``) to install RWGEN in developer mode::

    pip install -e .

The installation can be initially tested by opening an interactive Python 
session, which is done by entering::

    python

In the Python session type::

    import rwgen

This command should not return anything (i.e. anything that is printed to 
screen will be a warning or an error).

Close the Python session by typing::

    exit()

