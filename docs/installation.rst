Installation
============

The instructions below are for installing RWGEN as a developer on a Windows
operating system. Some steps will not be needed once the package has been
released.

Python and the RWGEN dependencies can be installed using the `conda`_ package
manager. `Miniconda`_ provides a lightweight implementation.

.. _conda: https://conda.io/projects/conda/en/latest/user-guide/index.html
.. _Miniconda: http://conda.pydata.org/miniconda.html

Installing Miniconda
--------------------

The conda website provides `instructions`_ for installing Miniconda. 
Installation on Windows involves downloading and running an `installer`_ (use 
the installer for the most recent Python version - currently 3.9).

.. _instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
.. _installer: https://docs.conda.io/en/latest/miniconda.html#windows-installers

.. note::

    The second step of the instructions (verify your installer hashes) can be
    skipped.

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

Getting the RWGEN Code
----------------------

Download the code as a `zip`_ file and unzip (choose somewhere other than the
``Downloads`` folder).

Note that Windows ``Extract All...`` adds the zip file name (e.g.
``rwgen-0.0.0``) to the default location for unzipping. This can be deleted, as
the zip file already contains an ``rwgen-0.0.0`` folder housing all other
files.

If the target folder is ``H:/RWGEN`` then we should end up with a folder
``H:/RWGEN/rwgen-0.0.0``, which contains sub-folders ``docs, examples, rwgen``
and some files (e.g. ``setup.py, environment.yml, ...``).

.. _zip: https://github.com/davidpritchard1/rwgen/archive/refs/tags/v0.0.0.zip

Creating the Conda Environment
------------------------------

Open the Anaconda Prompt - see `Updating and Configuring Conda`_ section - and
navigate (at the prompt) to the unzipped ``rwgen-0.0.0`` folder.

For example, to change from the directory ``C:/`` to the directory 
``H:/RWGEN/rwgen-0.0.0``, first switch drives by typing::

    H:

Then to change to ``H:/RWGEN/rwgen-0.0.0`` enter::

    cd H:/RWGEN/rwgen-0.0.0

This folder should contain a file called ``environment.yml``, which lists all
of the dependencies required by RWGEN.

Create a specific conda `environment`_ for RWGEN by typing::

    conda env create --name rwgen --file environment.yml

The environment name can be set to something other than ``rwgen`` if preferred.

Activate the `environment`_ before continuing::

    conda activate rwgen

.. _environment: https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html

.. note::

    If conda fails to create the environment try replacing ``environment.yml``
    with ``environment_unversioned.yml`` in the command above.

Installing RWGEN
----------------

At the Anaconda Prompt type the following to install RWGEN in developer mode::

    pip install -e .

The installation can be initially tested by opening an interactive Python 
session, which is done by entering::

    python

In the python session type::

    import rwgen

This command should not return anything (i.e. anything that is printed to 
screen will be a warning or an error).

Close the Python session by typing::

    exit()

