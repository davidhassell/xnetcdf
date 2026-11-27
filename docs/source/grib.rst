.. _GRIB:

GRIB datasets
=============

GRIB edition 1 and 2 datasets are read with the ``earthkit`` backend,
which uses `earthkit-data <https://earthkit-data.readthedocs.io>`_.
The fields of the dataset are combined into N-d variables, described
with CF metadata, so that the result is a netCDF-like view of the
GRIB.

Nothing needs to be specified to open a GRIB dataset, since the
backends that can't read it will fail and ``earthkit`` will then be
tried. To skip the other backends, use ``backend='earthkit'``:

.. code-block:: python

   >>> import xnetcdf
   >>> nc = xnetcdf.Dataset("era5.grib", backend="earthkit")
   >>> nc['t']

The CF metadata
---------------

The coordinates, bounds, cell methods, grid mappings, valid time and
time encoding are created by `xnetcdf`, from the GRIB metadata, so
that they follow the CF conventions as far as possible. Nothing is
invented: if the GRIB does not describe a coordinate or bounds, then
they are not created.

Options
-------

Options for reading GRIB are provided with the ``earthkit_options``
parameter of `xnetcdf.Dataset`, in separate sections for
`earthkit.data.from_source` (``'from_source'``), for the
`earthkit.data` xarray engine profile that decides which GRIB keys
define the dimensions (``'profile'``), and for the mapping to CF
(``'mapping'``). For example:

.. code-block:: python

   >>> nc = xnetcdf.Dataset(
   ...     "era5.grib",
   ...     earthkit_options={
   ...         "profile": {"name": "mars", "extra_dims": ["step"]},
   ...         "mapping": {"calendar": "365_day"},
   ...     },
   ... )

See `xnetcdf.Dataset` for all of the options.

Using ``grfive`` directly
-------------------------

The conversion of GRIB to a netCDF-4 like view is provided by the
``xnetcdf.grfive`` package, whose `xnetcdf.grfive.File` has the same
read-only API as `pyfive.File` and `umfive.File`. It can be used on
its own, without `xnetcdf.Dataset`:

.. code-block:: python

   >>> f = xnetcdf.grfive.File("era5.grib")
   >>> f["t"].shape

Limitations
-----------

* GRIB1 is supported on a best-effort basis, since it has less
  metadata to build CF from.

* Reduced Gaussian grids are not yet supported, because they await
  CF-1.14.
