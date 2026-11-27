"""The `NetCDFDataset` container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class NetCDFDataset:
    """In-memory, CF compliant representation of a netCDF dataset.

    The dataset is described by plain dictionaries of dimensions,
    variables and global attributes, as for a netCDF file with no
    groups. Every variable is stored as a ``(dims, values, attrs)``
    tuple, keyed by its name. The variables are linked by name in the
    usual CF way, i.e. by the ``coordinates``, ``bounds`` and
    ``grid_mapping`` attributes.

    The dataclass is frozen, so its attributes can't be reassigned,
    but the dictionaries themselves can be modified.

    The data of the data variables are lazy (see `LazyTensorArray`)
    and so hold a reference to the source of the data. Close the
    instance, or use it as a context manager, to release them.

    :Attributes:

        dimensions: dict of {str: int}
            Mapping of dimension names to their sizes.

        coord_vars: dict of {str: tuple of (tuple of `str`, numpy.ndarray, dict)}
            1-d numeric dimension coordinate variables keyed by name,
            each stored as ``(dims, values, attrs)``.

        data_vars: dict of {str: tuple of (tuple of `str`, Any, dict)}
            Primary data variables keyed by name, each stored as
            ``(dims, array_like, attrs)`` where ``array_like`` is
            either a `LazyTensorArray` or a `numpy.ndarray` (the
            latter if the data were read at once, with
            ``lazy_load=False``).

        aux_vars: dict of {str: tuple of (tuple of `str`, numpy.ndarray, dict)}
            All other coordinate and metadata variables, keyed by
            name, each stored as ``(dims, values, attrs)``. These are
            scalar, multi-dimensional and string coordinates; bounds
            (``*_bnds``) variables; and grid mapping variables.

        global_attrs: dict of {str: Any}
            Sanitised dataset-level netCDF global attributes.

    :Examples:

    >>> with grib_to_netcdf_dataset("tuv_pl.grib") as spec:
    ...     dims, data, attrs = spec.data_vars["t"]
    ...     first_level = data[0]

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
        """Enter the runtime context for this dataset."""
        return self

    def __exit__(self, *exc):
        """Exit and close all lazy variable arrays."""
        self.close()
