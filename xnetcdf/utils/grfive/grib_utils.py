"""Small shared helpers for the GRIB to netCDF conversion."""

from __future__ import annotations

import numpy as np




def _is_time_units(units):
    """Whether a CF ``units`` attribute string denotes a time unit.

    :Parameters:

        units:
            Candidate attribute value to inspect.

    :Returns:

        `bool`
            True if *units* is a string matching a CF time unit (with
            or without a ``"since <epoch>"`` reference), False
            otherwise.

    """
    return isinstance(units, str) and "since" in units


def _unique_name(name, used):
    """Return a collision-free identifier and register it in *used*.

    :Parameters:

        name: `str`
            Preferred variable or dimension `name.

        used: `set`
            Mutable `set` of names already claimed in the target
            namespace. Updated in place with the returned name.

    :Returns:

        `str`
            ``name`` if not already present in ``used``, otherwise
            ``"<name>_<n>"`` for the smallest positive integer ``n``
            not in ``used``.

    """
    if name not in used:
        used.add(name)
        return name

    n = 1
    while True:
        candidate = f"{name}_{n}"
        if candidate not in used:
            used.add(candidate)
            return candidate

        n += 1


def _field_meta(field, key):
    """Retrieve a metadata key from an `earthkit.data` GRIB field.

    :Parameters:

        field: `earthkit.data.Field`
            GRIB message wrapper exposing ``get`` and/or ``metadata``.

        key: `str`
            ecCodes / MARS metadata key to look up.

    :Returns:

            Metadata value if found, or ``None`` if missing or unreadable.

    """
    if hasattr(field, "get"):
        try:
            val = field.get(key, None)
            if val is not None:
                return val

        except Exception:
            pass

    try:
        return field.metadata(key)
    except Exception:
        pass


def _grid_type(field):
    """Return the grid type of an `earthkit.data` field.

    Uses `earthkit`'s parsed ``field.geography.grid_type()``, falling
    back to the GRIB ``gridType`` key.

    :Returns:

        `str` or `None`

    """
    try:
        return field.geography.grid_type()
    except Exception:
        return _field_meta(field, "gridType")


def _default_fill_value(dtype):
    """Return the default netCDF `_FillValue` for a floating dtype.

    :Returns:

        scalar or `None`
            The netCDF default fill value (``NC_FILL_FLOAT`` or
            ``NC_FILL_DOUBLE``), or `None` for non-floating dtypes.

    """
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        return None

    return dtype.type(9.969209968386869e36)
