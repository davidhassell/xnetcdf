.. _Installation:

Installation
============

The only library needed to import `xnetcdf`, besides Python, is
`numpy` (version 2.0.0 or later). Each of the backend libraries
`pyfive`, `netCDF4`, `zarr`, `umfive`, `earthkit-data`, `xarray`,
`h5py`, and `scipy.io.netcdf_file` is only imported when that backend
is used to read a dataset, so it is not a problem, in general, if a
backend library is not importable -- it just reduces the size of the
pool of backends that are available for reading a dataset.

Nevertheless, installing `xnetcdf` also installs all of the backend
libraries, so that any supported format can be read without further
steps. `xnetcdf` can be installed using ``pip`` using the command:

.. code-block:: console

    $ pip install xnetcdf

A ``conda`` package, which also installs all of the backend
libraries, is available from conda-forge:

.. code-block:: console

    $ conda install -c conda-forge xnetcdf

The library can also be imported directly from the `xnetcdf` source
root directory:

.. code-block:: console

    $ cd xnetcdf
    $ pip install -e . 
