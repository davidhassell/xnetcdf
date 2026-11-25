from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .grib_to_cf import grib_to_flat_netcdf_spec
from .variable import DataVariable, DimensionScale, Variable


class File(Mapping):
    """Read a GRIB file.

    2-d "slices" within a single file are always combined, where
    possible, into fields with N-d data

    :CF mappings:

    The contents of the dataset are mapped to CF dimensions and
    coordinate variables (as `DimensionScale` objects); auxiliary
    coordinate, domain ancillary, bounds and grid mapping variables
    (as `Variable` objects); and data variables (as `DataVariable`
    objects).

    :Performance:

    The read of the dataset is lazy in that only the metadata
    (i.e. the lookup headers and any extra data) are accessed during
    the initial read. A data array in the dataset is then accessed on
    demand, and then only for the part of the data array requested by
    the indexing.

    :Interoperability:

    This class is has enough of the `pyfive` API to allow us to use
    the ``'pyfive'`` backed API route through `xnetcdf.Dataset`.

    :Initialisation:

        filename:
            The definition of the PP or fields file dataset to be
            read. One of:

        mode: `str`
            The data access mode. Only ``'r'`` (read-only) is allowed.

        profile : str or Profile, default "auto"
             ``earthkit.data`` ``xr_engine`` profile name or object
             (e.g. ``"auto"``, ``"grib"``, ``"mars"``, or
             ``"earthkit"``). When ``"auto"``, selects ``"mars"`` if
             MARS namespace metadata is present on the first field and
             ``"grib"`` otherwise.

         calendar : str, default "proleptic_gregorian"
             CF calendar name used when encoding datetime coordinates
             into numeric ``'<unit> since YYYY-01-01'``
             values. Non-default calendars use `cftime` for
             conversion.

         decreasing_atmos_pressure: `bool`
             If True, 1-d atmospheric pressure coordinates that are
             strictly increasing (e.g. ``300 -> 1000`` hPa) are
             reversed to decreasing order (surface-to-top, ``1000 ->
             300`` hPa), and all variables sharing that dimension are
             flipped lazily along the corresponding axis.

         source_type: `str`, optional
             Explicit ``earthkit.data.from_source`` source type (such
             as ``"file"`` or ``"stream"``) to bypass automatic input
             type detection.

         squeezed: `bool`, optional
             If True (the default) then drop size 1, non-horizontal
             dimensions (such as time, step, level, number, etc.) from
             the data variables data array. If False then keep them in
             the data array.

         earthkit_options: `str`
             Keyword arguments that are passed to
             `earthkit.data.from_source`. Setting to `None` (the
             default) is equivalent to providing an empty dictionary.

    """

    def __init__(
        self,
        filename,
        mode="r",
        profile="auto",
        source_type=None,
        calendar="proleptic_gregorian",
        decreasing_atmos_pressure=True,
        squeeze=True,
        **earthkit_options,
    ):
        import earthkit.data as ekd

        if mode != "r":
            raise ValueError(
                f"{self.__class__.__name__} only supports read-only mode='r'"
            )

        try:
            self.filename = str(Path(filename))
        except TypeError:
            self.filename = filename.path

        options = {
            "profile": profile,
            "source_type": source_type,
            "calendar": calendar,
            "decreasing_atmos_pressure": decreasing_atmos_pressure,
            "squeeze": squeeze,
            "source_kwargs": earthkit_options,
        }

        ds = ekd.from_source("file", self.filename)
        d = grib_to_flat_netcdf_spec(ds, **options)

        self.options = options

        import pprint

        pprint.pprint(d)

        self.mode = mode
        self.parent = None
        self.name = "/"
        self.path = "/"
        self.groups = {}
        self.dimensions = {}

        # Initialise the _Netcdf4Dimid attribute of DimensionScale
        # instances. This list get updated in-place during each
        # DimensionScale initialisation.
        Netcdf4Dimid = [np.int32(0)]

        # Initialise the dictionary of all (i.e. data and metadata)
        # variables, keyed by their variable names.
        all_variables = {}

        # Initialise the list of data variable names
        data_variable_names = []

        self.attrs = d.global_attrs

        for name, (dims, data, attrs) in d.coord_vars.items():
            all_variables[name] = DimensionScale(
                name, data, attrs=attrs, Netcdf4Dimid=Netcdf4Dimid
            )

        for name, size in d.dimensions.items():
            if name in all_variables:
                continue

            all_variables[name] = DimensionScale(
                name, size=size, Netcdf4Dimid=Netcdf4Dimid
            )

        for name, (dims, data, attrs) in d.aux_vars.items():
            DIMENSION_LIST = tuple((dim,) for dim in dims)
            all_variables[name] = Variable(
                name, data, attrs=attrs, DIMENSION_LIST=DIMENSION_LIST
            )

        for name, (dims, data, attrs) in d.data_vars.items():
            DIMENSION_LIST = tuple((dim,) for dim in dims)
            all_variables[name] = DataVariable(
                name, data, attrs=attrs, DIMENSION_LIST=DIMENSION_LIST
            )

            data_variable_names.append(name)

        self._all_variables = all_variables
        self._data_variable_names = data_variable_names

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()

    def __getitem__(self, path: str):
        """Get a data or metadata variable or the File from its path.

        Paths may start with ``.``, ``/``, or ``./``, and a trailing
        ``/`` is ignored. A path may be an empty string.

        """
        if not isinstance(path, str):
            raise TypeError("path must be a string")

        key = path
        if key.startswith("."):
            key = key[1:]

        if key.startswith("/"):
            key = key[1:]

        if key.endswith("/"):
            key = key[:-1]

        if not key:
            return self

        if "/" in key:
            raise KeyError(f"Nested paths are not supported: {path!r}")

        return self.variables[key]

    def __iter__(self):
        """Return an iterator over the variable mapping keys."""
        return iter(self.variables)

    def __len__(self):
        """The number of data and metadata variables."""
        return len(self.variables)

    @property
    def variables(self):
        """All (data and metadata) variables.

        .. seealso:: `dimension_variables`, `data_variables`,
                     `metadata_variables`

        :Returns:

            `dict`
                The variables, keyed by their names.

        :Examples:

        >>> u.variables
        TODOx

        """
        return self._all_variables

    @property
    def consolidated_metadata(self):
        """Whether the metadata are in a contiguous block.

        Metadata in this context comprises the lookup headers and any
        extra data.

        :Returns:

            `bool`

        """
        return False

    @property
    def userblock_size(self):
        """Size of the user block in bytes (currently always 0).

        Provided for compatibility with the `pyfive` API.

        :Returns:

            `int`

        """
        return

    def close(self):
        """Close the underlying dataset reader.

        However, the reader is not closed if it was opened externally,
        i.e. if the *filename* argument to `File` was a file-like
        object.

        :Returns:

            `None`

        """
        pass

    def items(self):
        """A set-like object providing a view on the dataset's items."""
        return self.variables.items()
