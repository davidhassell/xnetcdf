"""A read-only, `pyfive`-like view of GRIB datasets.

`grfive` deliberately mirrors the API of `pyfive` and `umfive`: a
`File` presents the contents of a GRIB file as the groups, dimensions
and variables of a netCDF-4 (HDF5) dataset, with CF metadata. The
GRIB is read with `earthkit`.

It is kept inside `xnetcdf` as a stand-in for what could become a
standalone ``grfive`` library, which `xnetcdf` would then use as a
backend in place of `earthkit` directly. It is available as
``xnetcdf.grfive``, and nothing in it depends on the rest of
`xnetcdf`.

"""

from .high_level import File
