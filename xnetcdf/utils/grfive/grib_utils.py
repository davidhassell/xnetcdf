"""Small shared helpers and the `FlatNetCDFSpec` container."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any




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


@dataclass(frozen=True, slots=True)
class FlatNetCDFSpec:
    """Flat, CF compliant representation of a netCDF-style dataset.

    :Attributes:

        dimensions: dict of {str: int}
            Mapping of dimension names to their sizes.

        coord_vars: dict of {str: tuple of (tuple of `str`, numpy.ndarray, dict)}
            1-d numeric dimension coordinate variables keyed by name,
            each stored as ``(dims, values, attrs)``.

        data_vars: dict of {str: tuple of (tuple of `str`, Any, dict)}
            Primary data variables keyed by name, each stored as
            ``(dims, array_like, attrs)`` where ``array_like`` is
            either a `LazyTensorArray` or a `numpy.ndarray`.

        aux_vars: dict of {str: tuple of (tuple of `str`, numpy.ndarray, dict)}
            Scalar, multi-dimensional, string, and ``*_bnds``
            auxiliary variables keyed by name, each stored as ``(dims,
            values, attrs)``.

        global_attrs: dict of {str: Any}
            Sanitised dataset-level netCDF global attributes.

    """

    dimensions: dict[str, int]
    coord_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, Any]]]
    data_vars: dict[str, tuple[tuple[str, ...], Any, dict[str, Any]]]
    aux_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, Any]]]
    global_attrs: dict[str, Any]

    def close(self):
        """Close all `LazyTensorArray` instances in `data_vars`."""
        for _, data, _ in self.data_vars.values():
            # Lazy arrays (`LazyTensorArray`) are closable
            close = getattr(data, "close", None)
            if callable(close):
                close()

    def __enter__(self):
        """Enter the runtime context for this specification."""
        return self

    def __exit__(self, *exc):
        """Exit and close all lazy variable arrays."""
        self.close()
