Installation
============

Python and the RWGEN dependencies can be installed using the `conda`_ package 
manager. If you are new to `conda`_ then `miniconda`_ provides a lightweight
implementation.

.. _conda: https://conda.io/projects/conda/en/latest/user-guide/index.html
.. _miniconda: http://conda.pydata.org/miniconda.html

.. note::

   The instructions below are for installing RWGEN as a developer on a Windows
   operating system. Some steps will not be needed once the package has been
   released.


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
``anaconda prompt``. This should give you a prompt that looks similar to the
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

`Download`_ the code as a zip file and unzip (choose somewhere other than the
``Downloads`` folder).

Note that Windows ``Extract All...`` adds the zip file name (i.e.
``rwgen-main``) to the default location for unzipping - this can be deleted, as
the zip file already contains an ``rwgen-main`` folder housing all other files.

Rename the unzipped folder from ``rwgen-main`` to ``rwgen``.

PUT IMAGE etc TO SHOW HOW IT SHOULD END UP (no nested rwgen-main/rwgen-main)

.. _Download: https://github.com/davidpritchard1/rwgen


Creating the Conda Environment
------------------------------

Open the Anaconda Prompt - see `Updating and Configuring Conda`_ section - and
navigate (at the prompt) to the unzipped ``rwgen`` folder. 

For example, to change from the directory ``C:/`` to the directory 
``H:/Python/rwgen``, first switch drives by typing::

    H:

Then to change to ``H:/Python/rwgen`` enter::

    cd H:/Python/rwgen

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



