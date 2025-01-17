.. highlight:: shell

=============
Installation
=============

Prerequisites
-------------

1. Familiarity with ``xarray``

   We highly recommend visiting the `xarray tutorial`_ and `xarray documentation`_
   pages if you aren't familiar with ``xarray``.

2. xCDAT is distributed through conda, which is available through Anaconda and Miniconda.

   We recommend following the `Quick command line install`_ steps in the Anaconda docs
   to install Miniconda. Those steps are also provided below for convenience.

   .. code-block:: bash

      >>> # Linux
      >>> mkdir -p ~/miniconda3
      >>> curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
      >>> bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
      >>> rm -rf ~/miniconda3/miniconda.sh

   .. code-block:: bash

      >>> # MacOS
      >>> mkdir -p ~/miniconda3
      >>> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
      >>> bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
      >>> rm -rf ~/miniconda3/miniconda.sh

   Then follow the instructions for installation. To have conda added to
   your path you will need to type ``yes`` in response to ``"Do you wish the
   installer to initialize Miniconda3 by running conda init?"`` (we recommend
   that you do this). Note that this will modify your shell profile (e.g.,
   ``~/.bashrc``) to add ``conda`` to your path.

   .. note::
      After installation completes you may need to type ``bash`` to
      restart your shell (if you use bash). Alternatively, you can log out and
      log back in.

3. Add the ``conda-forge`` channel.

   xCDAT is hosted on the `conda-forge`_ channel, which is the standard channel for
   most scientific Python packages.

   .. code-block:: bash

      >>> conda config --add channels conda-forge
      >>> conda config --set channel_priority strict

.. _xarray tutorial: https://tutorial.xarray.dev/intro.html
.. _xarray documentation: https://docs.xarray.dev/en/stable/getting-started-guide/index.html
.. _Quick command line install: https://docs.anaconda.com/free/miniconda/#quick-command-line-install
.. _conda-forge: https://anaconda.org/conda-forge/xcdat

Instructions
------------

1. Create a Conda environment from scratch with ``xcdat`` (`conda create`_)

   We recommend using the Conda environment creation procedure to install ``xcdat``.
   The advantage with following this approach is that Conda will attempt to resolve
   dependencies for compatibility.

   To create an ``xcdat`` Conda environment, run:

   .. code-block:: bash

       >>> conda create -n <ENV_NAME> -c conda-forge xcdat
       >>> conda activate <ENV_NAME>

2. Install ``xcdat`` in an existing Conda environment (`conda install`_)

   You can also install ``xcdat`` in an existing Conda environment, granted that Conda
   is able to resolve the compatible dependencies.

   .. code-block:: bash

       >>> conda activate <ENV_NAME>
       >>> conda install -c conda-forge xcdat

3. [Optional] Some packages that are commonly used with ``xcdat`` can be installed
   either in step 1 or step 2 above:

   - ``jupyterlab``: a web-based interactive development environment for notebooks,
     code, and data. This package also includes ``ipykernel``.
   - ``matplotlib``: a library for creating visualizations in Python.
   - ``nc-time-axis`` is an optional dependency required for ``matplotlib`` to plot ``cftime`` coordinates
   - ``cartopy``: an add-on package for ``matplotlib`` and specialized for geospatial data processing.

.. _conda create: https://docs.conda.io/projects/conda/en/latest/commands/create.html
.. _conda install: https://docs.conda.io/projects/conda/en/latest/commands/install.html

Updating
--------

New versions of ``xcdat`` will be released periodically. We recommend you use the
latest stable version of ``xcdat`` for the latest features and bug fixes.

.. code-block:: bash

   >>> conda activate <ENV_NAME>
   >>> conda update xcdat

To update to a specific version of ``xcdat``:

.. code-block:: bash

   >>> conda activate <ENV_NAME>
   >>> conda update xcdat=<MAJOR.MINOR.PATCH>
   >>> # Example: conda update xcdat=0.6.1

Jupyter Users set ``ESMFMKFILE`` env variable
---------------------------------------------

If you are a Jupyter user, the ``ESMFMKFILE`` environment variable will need to be set
either directly on the machine or through your Jupyter Notebook.

This env variable is normally set when calling ``conda activate`` with the conda
environment that has ``xesmf``. However, Jupyter does not run ``conda activate`` when using
the Python kernel associated with the environment so ``ESMFMKFILE`` is not set, resulting 
in ``ImportError: The ESMFMKFILE environment variable is not available.`` (related `GitHub
Issue <https://github.com/xCDAT/xcdat/issues/574>`_).

To set the ``ESMFMKFILE`` in a Jupyter Notebook add:

.. code-block:: python

   >>> import os
   >>> os.environ['ESMFMKFILE'] = 'conda-envs/xcdat/lib/esmf.mk'
   >>>
   >>> import xcdat
