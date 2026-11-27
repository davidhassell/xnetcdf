from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .grib_to_cf import grib_to_flat_netcdf_spec
from .variable import DataVariable, DimensionScale, Variable


class File(Mapping):
    """Read a GRIB file.

    The fields of the file are combined, where possible, into
    variables with N-d data, and described with CF metadata. See
    `grib_to_flat_netcdf_spec` for the details of how.

    :CF mappings:

    The contents of the dataset are mapped to CF dimensions and
    coordinate variables (as `DimensionScale` objects); auxiliary
    coordinate, domain ancillary, bounds and grid mapping variables
    (as `Variable` objects); and data variables (as `DataVariable`
    objects).

    :Performance:

    The read of the dataset is lazy in that only the metadata are
    read during the initial read. A data array in the dataset is then
    read on demand, and then only the fields needed for the part of
    the data array requested by the indexing.

    :Interoperability:

    This class has enough of the `pyfive` API to allow it to be used
    by `xnetcdf.Dataset` through its ``'earthkit'`` backend.

    :Initialisation:

        filename:
            The GRIB dataset to be read. One of a local path or
            ``file://`` URI, an ``http(s)://`` or ``ftp://`` URL, an
            ``fsspec`` URI or ``OpenFile`` object, raw bytes, an open
            binary stream, an `earthkit.data.FieldList` or ``Source``,
            or a sequence of ``Field`` objects.

        mode: `str`
            The data access mode. Only ``'r'`` (read-only) is allowed.

        profile: `str` or `Profile`, default "auto"
            The `earthkit.data` profile name or object (e.g.
            ``'auto'``, ``'grib'``, ``'mars'``, or ``'earthkit'``)
            that decides which GRIB keys become dimensions. When
            ``'auto'`` (the default), selects ``'mars'`` if MARS keys
            (``mars.class`` or ``levtype``) can be read from the first
            field, which is almost always the case, even for files not
            made with MARS, and ``'grib'`` otherwise.

        source_type: `str`, optional
            Explicit `earthkit.data.from_source` source type (such as
            ``'file'`` or ``'stream'``) to bypass automatic input type
            detection.

        calendar: `str`, default "proleptic_gregorian"
            CF calendar name used when encoding datetime coordinates
            into numeric ``'<unit> since YYYY-01-01'`` values.
            Non-default calendars use `cftime` for conversion.

        decreasing_atmos_pressure: `bool`, default True
            If True, 1-d atmospheric pressure coordinates that are
            strictly increasing (e.g. ``300 -> 1000`` hPa) are
            reversed to decreasing order (surface-to-top, ``1000 ->
            300`` hPa), and all variables sharing that dimension are
            flipped lazily along the corresponding axis.

        squeeze: `bool` or `None`, default True
            If True then drop size 1, non-horizontal dimensions (such
            as ``forecast_reference_time``, ``step``, ``level`` or
            ``member``) from the data variables' data arrays, and
            keep their coordinates as scalars. If False then keep
            them in the data arrays. If `None`, resolves to False for
            the ``'earthkit'`` profile and True otherwise.

        verbose: `int`, optional
            Set the verbosity. If *verbose* is ``0`` there is no
            verbose output, and more output is produced for
            progressively larger values of *verbose*. Values of ``1``
            and higher (or the value ``-1``) produce the same
            maximally verbose output.

        source_kwargs: `dict`, optional
            Keyword arguments that are passed to
            `earthkit.data.from_source`. Setting to `None` (the
            default) is equivalent to providing an empty dictionary.

        profile_kwargs: `dict`, optional
            Keyword arguments for the `earthkit.data` profile, such
            as ``lazy_load=False``, ``allow_holes=True``,
            ``rename_dims=...``, ``extra_dims=...`` or
            ``add_earthkit_attrs=True`` (which defaults to False
            here). Do not pass ``'squeeze'`` here; use the dedicated
            *squeeze* parameter instead. Setting to `None` (the
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
        verbose=0,
        source_kwargs=None,
        profile_kwargs=None,
    ):
        if mode != "r":
            raise ValueError(
                f"{self.__class__.__name__} only supports read-only mode='r'"
            )

        try:
            self.filename = str(Path(filename))
        except TypeError:
            try:
                self.filename = str(filename.path)
            except AttributeError:
                self.filename = "<file-like>"

        options = {
            "profile": profile,
            "source_type": source_type,
            "calendar": calendar,
            "decreasing_atmos_pressure": decreasing_atmos_pressure,
            "squeeze": squeeze,
            "source_kwargs": source_kwargs,
            "profile_kwargs": profile_kwargs,
        }
        self.options = options

        d = grib_to_flat_netcdf_spec(filename, **options)
        self._spec = d

        if verbose > 0 or verbose == -1:
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

        :Returns:

            `dict`
                The variables, keyed by their names.

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
        return 0

    def close(self):
        """Close the dataset.

        The data of the data variables are released, so they can't be
        read once the dataset is closed. This does not close any
        underlying GRIB file, because that is managed by `earthkit`.

        :Returns:

            `None`

        """
        self._spec.close()

    def items(self):
        """A set-like object providing a view on the dataset's items."""
        return self.variables.items()
