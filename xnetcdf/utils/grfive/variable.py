from math import prod

import numpy as np

# np.printoptions parameters for `dump` output (17 significant digits
# is enough to fully represent every possible bit of precision for
# 64-bit floats).
_printoptions = {"precision": 17, "floatmode": "maxprec"}


def _sorted_dict(d):
    """Return a dictionary sorted by its keys."""
    return {name: value for name, value in sorted(d.items())}


class _Mixin:
    """Mixin class for dataset variables."""

    def __getitem__(self, key):
        """Return a subspace of the data array."""
        data = self._data
        if data is None:
            raise ValueError(
                "Can't index a DimensionScale that is a netCDF dimension "
                "but not a netCDF variable."
            )

        return data[key]

    def __len__(self):
        shape = self.shape
        if not shape:
            raise TypeError(f"len() of unsized object: {self!r}")

        return shape[0]

    def __repr__(self):
        dimensions = self.dimensions
        if dimensions is None:
            dimensions = ""
        else:
            dims = ", ".join(dim for dim in dimensions)
            dimensions = f", dimensions=({dims})"

        return (
            f"<{__package__}.{self.__class__.__name__}: "
            f"{self.name}, shape={self.shape}{dimensions}>"
        )

    def __str__(self):
        return self.dump(display=False, data=False)

    def __array__(self, dtype=None, copy=None):
        """The numpy array interface.

        :Parameters:

            dtype: data-type, optional
                Typecode or data-type to which the array is cast.

            copy: `None` or `bool`
                Included to match the `numpy.ndarray.__array__` API,
                but ignored. The returned numpy array is always
                independent.

        :Returns:

            `numpy.ndarray`
                An independent numpy array of the data.

        """
        if dtype is not None:
            raise NotImplementedError("TODO")

        return self[...]

    @property
    def __orthogonal_indexing__(self):
        """Flag to indicate whether indexing is orthogonal."""
        return True

    @property
    def dtype(self):
        """The format of the elements in the array."""
        return self._dtype

    @property
    def ndim(self):
        """The array's number of dimensions.

        :Returns:

            `int`

        """
        return len(self.shape)

    @property
    def shape(self):
        """Shape of the data array.

        :Returns:

            `tuple`

        """
        return self._shape

    @property
    def size(self):
        """Number of elements in the array.

        :Returns:

            `int`

        """
        return prod(self.shape)

    @property
    def chunks(self):
        """The chunk shape.

        :Returns:

            `tuple`

        :Examples:

        >>> d.chunks
        (1, 1, 73, 96)

        """
        return self._chunks

    @property
    def dimensions(self):
        """The dimension names.

        :Returns:

            `tuple` or `None`
                The dimension names, or `None` if they are undefined.

        :Examples:

        >>> d.dimensions
        ('time', 'air_pressure', 'latitude', 'longitude')

        """
        DIMENSION_LIST = self.attrs.get("DIMENSION_LIST")
        if DIMENSION_LIST is None:
            return

        return tuple(dim[0] for dim in DIMENSION_LIST)

    @property
    def maxshape(self):
        """Maximum shape of the data.

        Always the same as the `shape`.

        :Returns:

            `tuple`

        :Examples:

        >>> d.maxshape
        (12, 19, 73, 96)
        >>> d.maxshape == d.shape
        True

        """
        return self.shape

    def dump(self, display=True, data=False, _level=0):
        """A full description of the variable.

        :Parameters:

            display: `bool`, optional
                If False then return the description as a string. By
                default the description is printed.

            data: `bool`, optional
                If True then include a summary of the variable's data
                array. If False (the default) then don't include a
                data summary.

        :Returns:

            `None` or `str`
                The description. If *display* is True then the
                description is printed and `None` is
                returned. Otherwise the description is returned as a
                string.

        """
        indent = "    "
        i0 = indent * _level
        i1 = indent * (_level + 1)
        i2 = indent * (_level + 2)

        lines = [f"{i0}{self.name}: {self!r}"]

        printoptions = _printoptions
        if data and "linewidth" not in printoptions:
            # Set the numpy linewidth
            printoptions = printoptions | {"linewidth": len(lines[0])}

        with np.printoptions(**printoptions):
            if self.attrs:
                lines.append(f"{i1}Attributes:")
                lines.extend(
                    f"{i2}{name}: {value!r}"
                    for name, value in self.attrs.items()
                )

            if data:
                try:
                    array = self[...]
                except Exception:
                    array = None

                if array is not None:
                    lines.append(f"{i1}Data {self.dtype.name}:")
                    data_string = np.array2string(
                        array, separator=", ", prefix=i2
                    )
                    lines.append(f"{i2}{data_string}")

        out = "\n".join(lines)
        if not display:
            return out

        print(out)

    @property
    def compression(self):
        """Returns `None`.

        Provided for compatibility with the `pyfive` API.

        """
        return

    @property
    def compression_opts(self):
        """Returns `None`.

        Provided for compatibility with the `pyfive` API.

        """
        return

    @property
    def dims(self):
        """Returns `None`.

        Provided for compatibility with the `pyfive` API.

        """
        return

    @property
    def fillvalue(self):
        """Fillvalue of the data."""
        mdi = self.attrs.get("missing_value")
        if mdi is None:
            mdi = self.attrs.get("_FillValue")

        return mdi

    @property
    def fletcher32(self):
        """Boolean indicator if the fletcher32 filter was applied.

        Provided for compatibility with the `pyfive` API.

        """
        return False

    @property
    def scaleoffset(self):
        """Returns `None`.

        Provided for compatibility with the `pyfive` API.

        """
        return

    @property
    def shuffle(self):
        """Boolean indicator if shuffle filter was applied.

        Provided for compatibility with the `pyfive` API.

        """
        return False


class DimensionScale(_Mixin):
    """A CF dimension coordinate variable or dimension.

    A CF dimension coordinate variable and a CF dimension are
    distinguished by the latter having no data array, as well as
    different value for the ``NAME`` attribute.

    **Performance**

    A dimension coordinate variable's data array is derived from the
    lookup headers of the associated data variable, and so is always
    stored in memory.

    **Interoperability**

    This class is registered as a virtual subclass of
    `pyfive.Dataset`, meaning that it implements the core abstract
    methods required to safely mimic a native `pyfive.Dataset`
    layout. Therefore runtime type-checking using
    ``isinstance(dimension_scale_instance, pyfive.Dataset)`` will
    evaluate to `True`.

    **Initialisation**

    :Parameters:

        name: `str` or `None`, optional
            The dimension name.

        data: `np.ndarray` or `None`, optional
            The 1-d data, or `None` if dimension has no data.

        size: `int` or `None`, optional
            The size of the dimension. Ignored if *data* is set to an
            array.

        attrs: `dict` or `None`, optional
            The dimension coordinate attributes.

        Netcdf4Dimid: `list` or `None`, optional
            A single-element list containing the next available
            ``_NetCDF4Dimid`` attribute value. The list is updated
            in-place. If `None` (the default) then the
            ``_NetCDF4Dimid`` attribute will be assigned a value of
            ``0``.

    """

    def __init__(
        self,
        name=None,
        data=None,
        size=None,
        attrs=None,
        Netcdf4Dimid=None,
    ):
        if attrs is None:
            attrs = {}

        if data is None:
            self._data = None
            self._shape = (int(size),)
            self._dtype = np.dtype("float32")
        else:
            data = np.asanyarray(data)
            if data.ndim != 1:
                raise ValueError("Dimension scale data must be 1-d")

            self._data = data
            self._shape = data.shape
            self._dtype = data.dtype

        self.name = name
        self._chunks = None
        self.attrs = attrs

        if Netcdf4Dimid is None:
            Netcdf4Dimid = [np.int32(0)]

        hdf5_attrs = {
            "CLASS": b"DIMENSION_SCALE",
            "_Netcdf4Dimid": Netcdf4Dimid[0],
        }

        # Increment Netcdf4Dimid in-place
        Netcdf4Dimid[0] += 1

        if data is None:
            hdf5_attrs["NAME"] = (
                b"This is a netCDF dimension but not a netCDF variable."
            )
        else:
            hdf5_attrs["NAME"] = b"netCDF dimension coordinate variable"

        self.attrs.update(hdf5_attrs)

        self.attrs = _sorted_dict(self.attrs)

    def __repr__(self):
        out = f"<{__package__}.{self.__class__.__name__}: {self.name}, "
        if self._data is None:
            out += f"size={self.shape[0]}>"
        else:
            out += f"shape={self.shape}>"

        return out

    @property
    def dimensions(self):
        """The dimension name.

        :Returns:

            `tuple`
                The dimension name.

        :Examples:

        >>> d.dimensions
        ('time',)

        """
        return (self.name,)

    @property
    def has_coordinates(self):
        """Whether the dimension has a coordinate data array.

        :Returns:

            `bool`

        """
        return self._data is not None


class Variable(_Mixin):
    """A CF metadata variable in the dataset.

    Any CF variable that is not a dimension coordinate variable nor a
    data variable is represented by a `Variable` instance. This
    includes, but is not restricted to, coordinate bounds, auxiliary
    coordinate, domain ancillary, and grid mapping variables.

    A CF dimension coordinate variable or CF dimension must be
    represented by a `DimensionScale` instance, and a CF data variable
    must represented by a `DataVariable` instance.

    **Performance**

    A metadata variable's data array is derived from the lookup
    headers of the associated data variable, and so is always stored
    in memory.

    **Interoperability**

    This class is registered as a virtual subclass of
    `pyfive.Dataset`, meaning that it implements the core abstract
    methods required to safely mimic a native `pyfive.Dataset`
    layout. Therefore runtime type-checking using
    ``isinstance(variable_instance, pyfive.Dataset)`` will evaluate to
    `True`.

    """

    def __init__(
        self,
        name: str | None = None,
        data=None,
        attrs: dict | None = None,
        DIMENSION_LIST: tuple | None = None,
    ):
        """**Initialisation**

        :Parameters:

            name: `str` or `None`, optional
                The variable name.

            data: `np.ndarray` or `None`, optional
                The data.

            attrs: `dict` or `None`, optional
                The variable attributes.

            DIMENSION_LIST: `tuple` or `None`, optional
                The dimension names for the data, e.g. ``()``,
                ``(('time',),)``, ``(('latitude',), ('longitude',))``

        """
        if attrs is None:
            attrs = {}

        self.name = name
        self._data = data
        self._shape = data.shape
        self._dtype = data.dtype
        self._chunks = None
        self.attrs = attrs

        if DIMENSION_LIST is None and data is not None and not self.shape:
            DIMENSION_LIST = ()

        if DIMENSION_LIST is None:
            raise ValueError(
                "Must provide DIMENSION_LIST when instantiating a "
                f"non-scalar Variable instance: {name}({self.shape})"
            )

        if len(DIMENSION_LIST) != len(self.shape):
            raise ValueError(
                "DIMENSION_LIST must have the same number of elements as "
                "there are data dimensions"
            )

        self.attrs["DIMENSION_LIST"] = DIMENSION_LIST
        self.attrs = _sorted_dict(self.attrs)


class DataVariable(Variable):
    """A CF data variable in the dataset.

    A data variable comprises a multi-dimensional data array of a
    single physical quantity, that is defined in the dataset by one
    or more two-dimensional chunks, each of which is described by
    metadata stored in a lookup header.

    **Performance**

    Accessing the data array is lazy. The data array in the file is
    accessed on demand, and then only for the part of the data array
    requested by the indexing. Data reads are parallelised over the
    2-d slices stored for each lookup header (see `get_parallelism`
    and `set_parallelism` methods).

    **Interoperability**

    This class is registered as a virtual subclass of
    `pyfive.Dataset`, meaning that it implements the core abstract
    methods required to safely mimic a native `pyfive.Dataset`
    layout. Therefore runtime type-checking using
    ``isinstance(data_variable_instance, pyfive.Dataset)`` will
    evaluate to `True`.

    **Initialisation**

    :Parameters:

        name: `str`
            The name of the data variable, e.g.
            ``'UM_m01s30i201_vn1100:'``.

        attrs: `dict`
            The data variable attributes.

        shape: `tuple` of `int`
            The shape of the data array, e.g. ``(12, 19, 110, 106)``.

        dtype: data-type, optional
            The data type of the data array.

        chunk_shape: `tuple` of `int`
            The shape of each data chunk, e.g. ``(1, 1, 110, 106)``.

        file: `File`
            The parent `File` object.

        DIMENSION_LIST: `tuple` or `None`, optional
            The dimension names for the data, e.g. ``()``,
            ``(('time',),)``, ``(('latitude',),
            ('longitude',))``. This are stored as an attribute in the
            `attrs` dictionary.

    """

    def __init__(
        self,
        name: str | None = None,
        data=None,
        attrs: dict | None = None,
        DIMENSION_LIST: tuple | None = None,
    ):
        """**Initialisation**

        :Parameters:

            name: `str` or `None`, optional
                The variable name.

            data: `np.ndarray` or `None`, optional
                The data.

            attrs: `dict` or `None`, optional
                The variable attributes.

            DIMENSION_LIST: `tuple` or `None`, optional
                The dimension names for the data, e.g. ``()``,
                ``(('time',),)``, ``(('latitude',), ('longitude',))``

        """
        super().__init__(
            name=name, data=data, attrs=attrs, DIMENSION_LIST=DIMENSION_LIST
        )

        try:
            self._chunks = data.chunks
        except AttributeError:
            self._chunks = self.shape
