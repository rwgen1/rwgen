.. |rwgen_version| replace:: ``rwgen-0.0.6``

.. |example_path| replace:: ``H:/RWGEN/rwgen-0.0.6/examples/rainfall_model/single_site/nsrp_example.ipynb``

.. |example_folder_nonliteral| replace:: H:/RWGEN/rwgen-0.0.6/examples/rainfall_model/single_site

Examples
========

For outline examples and explanations, see the rainfall model
:doc:`rainfall-workflow` and weather generator :doc:`weather-workflow` pages.

More detailed examples are provided as Jupyter notebooks (just for the rainfall
model currently) and scripts. The notebooks contain relatively detailed
explanations of the examples/code and provide a more interactive way of trying
things out.

The examples can be found in the ``examples`` folder within the RWGEN root
folder (e.g. |rwgen_version|).

.. note::

    If preferred the examples folder can be copied to another location before
    use to avoid modifying files in (or adding files to) the RWGEN root folder.

To look at the examples, first open an Anaconda Prompt (press the Windows
button and then start typing ``anaconda prompt``) if not already open. Switch to
the RWGEN environment created during installation if not already active - for
example::

    conda activate rwgen

Then navigate to the folder containing the notebook
that you would like to run. For example, if the notebook is located at
|example_path| then type

.. parsed-literal::

    H:
    cd\  |example_folder_nonliteral|\

Next launch JupyterLab by typing at the prompt::

    jupyter-lab

It may take a moment, but this should launch JupyterLab in the default browser.
In the left hand panel you should see the notebook ``nsrp_example.ipynb`` -
double click on this to launch it.

The notebooks contain a mixture of explanation and "code cells" that can be run
by clicking them and pressing ``Ctrl + Enter``. The cells should generally be
run in order (although it is possible to do things like change something and
rerun a cell). Various options are available through the ``Run`` menu. It is
sometimes also useful to clear the outputs printed by a code cell, which can
be done via ``Clear Output`` or ``Clear All Outputs`` in the ``Edit`` menu.

When finished JupyterLab can be closed via ``File > Shut Down``. The browser
tab can then be closed. If the Anaconda Prompt does not return it is OK to
press ``Ctrl + C``.

.. note::

    The examples are also available as ``.py`` scripts, which can be run at
    the prompt by entering e.g. ``python nsrp_example.py``. The scripts
    contain all of the core steps but not every call made in the notebooks.
    They also include a "boilerplate" line currently needed on Windows. The
    scripts can be inspected in an IDE, text editor or JupyterLab.
