Examples
========

Examples are provide as Jupyter notebooks. These examples can be found in the
``examples`` folder within the RWGEN root folder.

To run these examples, first copy the whole examples folder to another location
(i.e. outside of RWGEN root folder). This can be a local or temporary folder.
This helps us to avoid modifying files within the RWGEN root folder.

Open an Anaconda Prompt (press the Windows button and then start typing
``anaconda prompt``). Switch to the RWGEN environment created during
installation - for example::

    conda activate rwgen

Then navigate to the folder containing the notebook
that you would like to run. For example, if the notebook is located at
``H:/RWGEN/examples/nsrp_example.ipynb`` then type::

    H:
    cd H:\RWGEN\examples

Next launch JupyterLab by typing at the prompt::

    jupyter-lab

It may take a moment, but this should launch JupyterLab in the default browser.
In the left hand panel you should see the notebook ``nsrp_example.ipynb`` -
double click on this to launch it.

The code cells in the notebooks can be run using ``Ctrl + Enter``.

To close JupyterLab, the browser tab can be closed but it also needs to be
stopped in the Anaconda Prompt. This can be done by entering ``Ctrl + C``
(sometimes twice).
