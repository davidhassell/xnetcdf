"""Convert GRIB datasets into CF compliant netCDF specifications.

This module wraps `earthkit.data`'s ``xr_engine`` builder to extract
GRIB metadata, coordinates, and deferred `numpy` array views
(`LazyTensorArray`) into a `FlatNetCDFSpec` without importing
``xarray`` or ``dask``.

"""

from __future__ import annotations

import datetime
import io
import math
import os
import threading
import uuid
import weakref
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np

CF_CONVENTIONS = "CF-1.14"
CF_CALENDAR = "proleptic_gregorian"

NC_FILL_DOUBLE = np.float64(9.969209968386869e36)
NC_FILL_INT64 = np.int64(-9223372036854775806)

_NUMERIC_DTYPE_KINDS = frozenset({"b", "i", "u", "f"})
_PRESSURE_UNITS = {"pa", "hpa", "mbar", "millibar", "bar"}
_VERTICAL_NAMES = {
    "level",
    "levelist",
    "isobaricInhPa",
    "isobaricInPa",
    "hybrid",
    "heightAboveGround",
    "depthBelowLandLayer",
    "potentialVorticity",
    "theta",
}
_LEVEL_TYPE_NAMES = ("level_type", "levtype")
_HORIZONTAL_1D_BOUNDS_NAMES = {"latitude", "longitude", "x", "y"}

_GRIB_STEP_UNITS_TO_SECONDS = {
    0: 60,  # minute
    1: 3600,  # hour
    2: 86400,  # day
    10: 10800,  # 3 hours
    11: 21600,  # 6 hours
    12: 43200,  # 12 hours
    13: 1,  # second
}

_GRIB_STEP_TYPE_TO_CF_METHOD = {
    "instant": "point",
    "avg": "mean",
    "avgd": "mean",
    "avgua": "mean",
    "accum": "sum",
    "min": "minimum",
    "max": "maximum",
    "rms": "root_mean_square",
    "sd": "standard_deviation",
}

_GRIB2_STAT_PROC_TO_CF_METHOD = {
    0: "mean",
    1: "sum",
    2: "maximum",
    3: "minimum",
    5: "root_mean_square",
    6: "standard_deviation",
    11: "sum",
    13: "median",
}

_GRIB2_DERIVED_FORECAST_TO_CF_METHOD = {
    0: "realization: mean",
    1: "realization: mean",
    2: "realization: standard_deviation",
    3: "realization: standard_deviation",
    4: "realization: standard_deviation",
    6: "realization: mean",
    8: "realization: minimum",
    9: "realization: maximum",
}

_LOCK_REGISTRY = weakref.WeakValueDictionary()
_LOCK_REGISTRY_GUARD = threading.Lock()


class _PicklableLock:
    """A re-entrant lock that survives pickling.

    The underlying `threading.RLock` is not pickled. Instead each
    instance carries a token, and unpickling in a process re-uses any
    lock already registered for that token, so all copies of a lock
    within one process exclude each other.

    """

    def __init__(self, token=None):
        self._token = token or uuid.uuid4().hex
        with _LOCK_REGISTRY_GUARD:
            self._lock = _LOCK_REGISTRY.setdefault(
                self._token, threading.RLock()
            )

    def acquire(self, *args, **kwargs):
        """Acquire the lock."""
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        """Release the lock."""
        self._lock.release()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()

    def __getstate__(self):
        return {"token": self._token}

    def __setstate__(self, state):
        self.__init__(state["token"])


def _get_source_lock(ds):
    """Return the lock shared by everything reading from *ds*.

    The lock is cached on *ds* so that all arrays derived from the
    same `earthkit.data.FieldList` (and so the same GRIB file handles
    and caches) are serialised against each other, whereas arrays
    from different sources can be read in parallel.

    :Parameters:

        ds: `earthkit.data.FieldList`

    :Returns:

        `_PicklableLock`

    """
    with _LOCK_REGISTRY_GUARD:
        lock = getattr(ds, "_xnetcdf_lock", None)

    if lock is None:
        lock = _PicklableLock()
        with _LOCK_REGISTRY_GUARD:
            lock = getattr(ds, "_xnetcdf_lock", None) or lock
            try:
                ds._xnetcdf_lock = lock
            except AttributeError:
                pass

    return lock


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


def _as_fieldlist(ds, source_type, source_kwargs):
    """Return a `FieldList`.

    Coerces a supported GRIB input object into an
    `earthkit.data.FieldList`.

    :Parameters:

        ds: `str`, `os.PathLike`, `bytes`, file_like, `FieldList`, `Source`, or sequence of `Field`
            Input GRIB source to normalise.

        source_type: `str`, optional
            Explicit `earthkit.data.from_source` source type
            (e.g. ``'file'`` or ``'stream'``) to bypass automatic
            input type detection.

        source_kwargs: `dict`, optional
            Additional keyword arguments forwarded to
            `earthkit.data.from_source`.

    :Returns:

        `earthkit.data.FieldList`
            Normalised field list backing *ds*.

    """
    import earthkit.data as ekd

    if isinstance(ds, ekd.FieldList):
        return ds

    kw = dict(source_kwargs or {})

    # Explicit source override (e.g. source_type="stream" for custom
    # file/byte wrappers)
    if source_type is not None:
        if isinstance(ds, (bytes, bytearray, memoryview)):
            ds = io.BytesIO(ds)
        elif (
            not hasattr(ds, "read")
            and hasattr(ds, "open")
            and hasattr(ds, "fs")
        ):
            ds = ds.open()

        return ekd.from_source(source_type, ds, **kw).to_fieldlist()

    source_cls = getattr(ekd, "Source", None)
    is_ek_source = source_cls is not None and isinstance(ds, source_cls)

    if not is_ek_source:
        # Local path or remote URI string / PathLike
        if isinstance(ds, (str, os.PathLike)):
            raw = os.fspath(ds)
            parsed = urlparse(raw)
            scheme = parsed.scheme.lower()

            # Single-letter schemes (e.g. "C:\\...") are Windows drive
            # paths, not URIs
            if len(scheme) <= 1 or scheme == "file":
                if scheme == "file":
                    local_path = url2pathname(parsed.path)
                else:
                    local_path = raw

                return ekd.from_source("file", local_path, **kw).to_fieldlist()

            if scheme in ("http", "https", "ftp"):
                return ekd.from_source("url", raw, **kw).to_fieldlist()

            try:
                import fsspec
            except ImportError as exc:
                raise ImportError(
                    f"fsspec is required to open '{scheme}://' URIs"
                ) from exc

            return ekd.from_source(
                "stream", fsspec.open(raw, "rb").open(), **kw
            ).to_fieldlist()

        # Raw in-memory bytes
        if isinstance(ds, (bytes, bytearray, memoryview)):
            return ekd.from_source(
                "stream", io.BytesIO(ds), **kw
            ).to_fieldlist()

        # Already-open binary file-like object (fsspec file, open(...,
        # "rb"), BytesIO, etc.)
        if hasattr(ds, "read"):
            return ekd.from_source("stream", ds, **kw).to_fieldlist()

        # Unopened fsspec.OpenFile
        if hasattr(ds, "open") and hasattr(ds, "fs"):
            return ekd.from_source("stream", ds.open(), **kw).to_fieldlist()

    # Sequence of individual Field objects (e.g. [ds[0], ds[2]])
    field_cls = getattr(ekd, "Field", None)
    if (
        field_cls is not None
        and isinstance(ds, (list, tuple))
        and ds
        and all(isinstance(f, field_cls) for f in ds)
    ):
        return ekd.FieldList.from_fields(ds)

    # Existing earthkit Source or wrapper (GribData, etc.)
    if callable(getattr(ds, "to_fieldlist", None)):
        return ds.to_fieldlist()

    for attr in (
        "as_fieldlist",
        "fields",
        "fieldlist",
        "_fields",
        "_fieldlist",
        "ds",
        "_ds",
    ):
        try:
            candidate = getattr(ds, attr)
        except Exception:
            continue

        if callable(candidate):
            try:
                candidate = candidate()
            except TypeError:
                continue

        if isinstance(candidate, ekd.FieldList):
            return candidate

    raise TypeError(
        f"Cannot convert {type(ds).__name__} to an earthkit.data.FieldList "
        "(available attributes: "
        f"{[a for a in dir(ds) if not a.startswith('__')]})"
    )


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


def _extract_grib_cell_method_hints(source):
    """Infer CF cell_methods from a variable's GRIB field list.

    Inspects the first and last fields of ``source`` for ``stepType``,
    ``typeOfStatisticalProcessing``, and ``derivedForecast`` keys.

    :Parameters:

        source: `earthkit.data.FieldList`
            Field list belonging to a single variable tensor.

    :Returns:

        `tuple` of (`str` or `None`, `str` or `None`)
            ``(temporal_cf_method, ensemble_cf_method)``, where either
            element is ``None`` if absent or inconsistent across
            sampled fields.

    """
    try:
        n = len(source)
    except Exception:
        return None, None

    if n == 0:
        return None, None

    if n == 1:
        samples = [source[0]]
    else:
        samples = [source[0], source[-1]]

    temporal_methods: set[str | None] = set()
    ensemble_methods: set[str | None] = set()

    for fld in samples:
        t_method = None
        step_type = _field_meta(fld, "stepType")
        if isinstance(step_type, str):
            t_method = _GRIB_STEP_TYPE_TO_CF_METHOD.get(step_type.lower())

        if t_method is None:
            stat_proc = _field_meta(fld, "typeOfStatisticalProcessing")
            if isinstance(stat_proc, (int, np.integer)):
                t_method = _GRIB2_STAT_PROC_TO_CF_METHOD.get(int(stat_proc))

        temporal_methods.add(t_method)

        ens_method = None
        derived = _field_meta(fld, "derivedForecast")
        if isinstance(derived, (int, np.integer)):
            ens_method = _GRIB2_DERIVED_FORECAST_TO_CF_METHOD.get(int(derived))

        ensemble_methods.add(ens_method)

    if len(temporal_methods) == 1:
        temporal = next(iter(temporal_methods))
    else:
        temporal = None

    if len(ensemble_methods) == 1:
        ensemble = next(iter(ensemble_methods))
    else:
        ensemble = None

    return temporal, ensemble


def _extract_grib_interval_hints(ds):
    """Get temporal and verticl interval bounds from GRIB fields.

    Extracts forecast step and vertical layer interval bounds from
    GRIB fields.

    :Parameters:

        ds: `earthkit.data.FieldList`
            Full GRIB field list to scan for non-instantaneous step
            intervals (``startStep``/``endStep``) and multi-level
            layer bounds (``topLevel``/``bottomLevel``).

    :Returns:

        (`dict`, `dict`)
            ``(step_intervals_s, level_intervals)`` mapping end-step
            seconds to ``(start_seconds, end_seconds)`` and level
            coordinate values to ``(top_level, bottom_level)``.

    """
    step_intervals_s = {}
    level_intervals = {}

    try:
        fields = list(ds)
    except Exception:
        return step_intervals_s, level_intervals

    for fld in fields:
        step_type = _field_meta(fld, "stepType")
        stat_proc = _field_meta(fld, "typeOfStatisticalProcessing")
        is_interval_type = (
            isinstance(step_type, str) and step_type.lower() != "instant"
        ) or isinstance(stat_proc, (int, np.integer))

        start_step = _field_meta(fld, "startStep")
        end_step = _field_meta(fld, "endStep")
        if (
            is_interval_type
            and isinstance(start_step, (int, float, np.number))
            and isinstance(end_step, (int, float, np.number))
        ):
            s_val, e_val = float(start_step), float(end_step)
            if s_val != e_val:
                su = _field_meta(fld, "stepUnits")
                if isinstance(su, (int, np.integer)):
                    scale = _GRIB_STEP_UNITS_TO_SECONDS.get(int(su), 3600)
                else:
                    scale = 3600

                e_sec = int(round(e_val * scale))
                s_sec = int(round(s_val * scale))
                step_intervals_s.setdefault(e_sec, (s_sec, e_sec))

        top_lvl = _field_meta(fld, "topLevel")
        bot_lvl = _field_meta(fld, "bottomLevel")
        lvl = _field_meta(fld, "level")
        if (
            isinstance(top_lvl, (int, float, np.number))
            and isinstance(bot_lvl, (int, float, np.number))
            and float(top_lvl) != float(bot_lvl)
        ):
            if isinstance(lvl, (int, float, np.number)):
                key_lvl = float(lvl)
            else:
                key_lvl = float(top_lvl)

            level_intervals.setdefault(
                key_lvl, (float(top_lvl), float(bot_lvl))
            )

    return step_intervals_s, level_intervals


class LazyTensorArray:
    """Pure `numpy` lazy array backed by an `earthkit.data` Tensor.

    Supports indexing, while tracking virtual squeezed and reversed
    axes lazily.

    :Parameters:

        tensor:
            Underlying `earthkit.data` field tensor object.

        dims: iterable of `str`
            Exposed dimension names matching the unsqueezed axes.

        shape: iterable of `int`
            Raw shape of `tensor` prior to applying `squeeze_axes`.

        array_namespace:
            Array API namespace passed to `tensor.to_array`.

        dtype: `numpy.dtype` or type
            Target `numpy` data type of materialised slices.

        var_name: `str`
            Variable name used in `__repr__` and closed-array error
            messages.

        squeeze_axes: `tuple` of `int`, default ()
            Raw axis indices of singleton dimensions to drop from the
            exposed view.

        flip_axes: `tuple` of `int`, default ()
            Raw axis indices along which indexing should be reversed.

        lock: `_PicklableLock`, optional
            Lock guarding `earthkit` tensor reads. It should be
            shared by all arrays backed by the same GRIB source. If
            not provided, a new lock private to this array is used.

    .. note:: The lock pickles safely (see `_PicklableLock`), but
              whether a `LazyTensorArray` as a whole can be pickled,
              such as for use with a multiprocessing or distributed
              scheduler, has not been tested. It also depends on
              whether the underlying `earthkit` tensor can be
              pickled, which is governed by `earthkit.data`'s
              ``grib-file-serialisation-policy`` configuration. This
              is currently not something that is handled here.

    """

    def __init__(
        self,
        tensor,
        dims,
        shape,
        array_namespace,
        dtype,
        var_name,
        squeeze_axes: tuple[int, ...] = (),
        flip_axes: tuple[int, ...] = (),
        lock: _PicklableLock | None = None,
    ):
        self.tensor = tensor
        self._raw_shape = tuple(shape)
        self.squeeze_axes = tuple(sorted(squeeze_axes))
        self.flip_axes = tuple(sorted(flip_axes))
        squeezed = set(self.squeeze_axes)
        self.dims = tuple(dims)
        self.shape = tuple(
            s for i, s in enumerate(self._raw_shape) if i not in squeezed
        )
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape)
        self.dtype = np.dtype(dtype)
        self.array_namespace = array_namespace
        self._var_name = var_name
        self.lock = lock if lock is not None else _PicklableLock()

    @property
    def nbytes(self):
        """Total uncompressed size in bytes of the full array."""
        return self.size * self.dtype.itemsize

    @property
    def chunks(self):
        """Natural GRIB field-aligned chunk shape for this array."""
        field_shape = tuple(
            int(n)
            for n in getattr(self.tensor, "field_shape", self.shape[-2:])
        )
        n_outer = max(0, self.ndim - len(field_shape))
        return (1,) * n_outer + field_shape[-self.ndim :]

    def with_squeezed_axis(self, axis):
        """Return a new `LazyTensorArray` view with *axis* squeezed out.

        :Parameters:

            axis: `int`
                Exposed axis index (of size 1) to remove.

        :Returns:

            `LazyTensorArray`
                New lazy view sharing the underlying tensor with
                *axis* hidden.

        """
        squeezed = set(self.squeeze_axes)
        exposed_to_raw = [
            i for i in range(len(self._raw_shape)) if i not in squeezed
        ]
        raw_axis = exposed_to_raw[axis]
        new_dims = tuple(d for i, d in enumerate(self.dims) if i != axis)
        return LazyTensorArray(
            self.tensor,
            new_dims,
            self._raw_shape,
            self.array_namespace,
            self.dtype,
            self._var_name,
            squeeze_axes=(*self.squeeze_axes, raw_axis),
            flip_axes=self.flip_axes,
            lock=self.lock,
        )

    def with_flipped_axis(self, axis):
        """Return a new `LazyTensorArray` view reversed along *axis*.

        :Parameters:

            axis: `int`
                Exposed axis index whose coordinate order should be
                toggled.

        :Returns:

            `LazyTensorArray`
                New lazy view sharing the underlying tensor with
                *axis* flipped.

        """
        squeezed = set(self.squeeze_axes)
        exposed_to_raw = [
            i for i in range(len(self._raw_shape)) if i not in squeezed
        ]
        raw_axis = exposed_to_raw[axis]
        flipped = set(self.flip_axes)
        if raw_axis in flipped:
            flipped.remove(raw_axis)
        else:
            flipped.add(raw_axis)

        return LazyTensorArray(
            self.tensor,
            self.dims,
            self._raw_shape,
            self.array_namespace,
            self.dtype,
            self._var_name,
            squeeze_axes=self.squeeze_axes,
            flip_axes=tuple(flipped),
            lock=self.lock,
        )

    def close(self):
        """Release the reference to the underlying `earthkit` tensor."""
        with self.lock:
            self.tensor = None

    def __enter__(self):
        """Enter the runtime context for this lazy array."""
        return self

    def __exit__(self, *exc):
        """Exit and release the underlying tensor."""
        self.close()

    def _prepare_tensor_key(self, key):
        """Translate an index expression into raw tensor coordinates.

        :Parameters:

            key: `int`, `slice`, `Ellipsis`, array_like, or `tuple`
                Indexing expression relative to the exposed
                (squeezed/flipped) view. Integers and integer
                arrays may be negative. Boolean arrays must be 1-d
                and span a whole axis. ``None`` is not supported.
                Multiple array indices are applied independently to
                their own axes (orthogonal indexing, as in
                `netCDF4`), not pointwise as in `numpy`. An
                `IndexError` is raised for out of bounds indices.

        :Returns:

            `tuple` of (`tuple`, `tuple` of `int`) or `numpy.ndarray`
                ``(raw_key, post_flip_axes)`` ready for
                ``self.tensor[raw_key]``, or an empty `numpy.ndarray`
                directly when any slice/index is zero length.

        """
        if not isinstance(key, tuple):
            key = (key,)

        if any(k is None for k in key):
            raise IndexError(
                "numpy.newaxis (None) is not supported when indexing "
                f"{self!r}"
            )

        n_ellipsis = sum(k is Ellipsis for k in key)
        if n_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis")

        if n_ellipsis:
            idx = next(i for i, k in enumerate(key) if k is Ellipsis)
            n_missing = self.ndim - (len(key) - 1)
            key = key[:idx] + (slice(None),) * n_missing + key[idx + 1 :]

        if len(key) > self.ndim:
            raise IndexError(
                f"too many indices for {self!r}: array is "
                f"{self.ndim}-dimensional, but {len(key)} were indexed"
            )

        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))

        norm_exposed = []
        for axis, (k, dim_size) in enumerate(zip(key, self.shape)):
            if not isinstance(k, slice):
                if not isinstance(k, (int, np.integer)):
                    k = np.asarray(k)
                    if k.ndim == 0 and k.dtype.kind in "iu":
                        k = int(k)

            if isinstance(k, (int, np.integer)):
                k = int(k)
                if not -dim_size <= k < dim_size:
                    raise IndexError(
                        f"index {k} is out of bounds for axis {axis} "
                        f"with size {dim_size}"
                    )

                if k < 0:
                    k += dim_size
            elif isinstance(k, np.ndarray):
                if k.dtype.kind == "b":
                    if k.ndim != 1 or k.shape[0] != dim_size:
                        raise IndexError(
                            f"boolean index for axis {axis} must be 1-d "
                            f"with length {dim_size}, got shape {k.shape}"
                        )

                    k = np.flatnonzero(k)
                elif k.dtype.kind in "iu":
                    if k.size and (
                        k.min() < -dim_size or k.max() >= dim_size
                    ):
                        raise IndexError(
                            f"index array for axis {axis} has values out "
                            f"of bounds for size {dim_size}"
                        )

                    k = np.where(k < 0, k + dim_size, k)
                elif k.size or k.dtype.kind != "f":
                    raise IndexError(
                        "only integers, slices, Ellipsis, and integer or "
                        f"boolean arrays are valid indices, not {k.dtype}"
                    )
                else:
                    # Empty list: np.asarray([]) is float64
                    k = k.astype(int)

            norm_exposed.append(k)

        it = iter(norm_exposed)
        squeezed = set(self.squeeze_axes)
        flipped = set(self.flip_axes)
        raw_key = []
        post_flip_axes: list[int] = []
        empty_shape: list[int] = []
        has_empty = False
        out_ax = 0

        for raw_idx, raw_size in enumerate(self._raw_shape):
            if raw_idx in squeezed:
                raw_key.append(0)
                continue

            k = next(it)
            is_flipped = raw_idx in flipped

            if isinstance(k, int):
                if is_flipped:
                    raw_k = raw_size - 1 - k
                else:
                    raw_k = k

                raw_key.append(raw_k)
            elif isinstance(k, slice):
                if is_flipped:
                    base = range(raw_size)[::-1]
                else:
                    base = range(raw_size)

                rng = base[k]
                empty_shape.append(len(rng))
                if len(rng) == 0:
                    has_empty = True
                    raw_key.append(slice(0, 0, 1))
                elif rng.step > 0:
                    raw_key.append(slice(rng.start, rng.stop, rng.step))
                else:
                    rev = rng[::-1]
                    raw_key.append(slice(rev.start, rev.stop, rev.step))
                    post_flip_axes.append(out_ax)

                out_ax += 1
            else:
                idx_arr = np.asarray(k)
                if is_flipped:
                    idx_arr = (raw_size - 1) - idx_arr

                raw_key.append(idx_arr)
                empty_shape.extend(idx_arr.shape)
                if idx_arr.size == 0:
                    has_empty = True

                out_ax += idx_arr.ndim

        if has_empty:
            return np.empty(tuple(empty_shape), dtype=self.dtype)

        return tuple(raw_key), tuple(post_flip_axes)

    def __getitem__(self, key):
        """Return a slice from the backing GRIB tensor.

        :Parameters:

            key: `int`, `slice`, `Ellipsis`, array_like, or `tuple`
                Index expression.

        :Returns:

            `numpy.ndarray`
                Materialised array slice with singleton
                integer-indexed axes squeezed and negative-step or
                flipped axes oriented properly.

        """
        prepared = self._prepare_tensor_key(key)
        if isinstance(prepared, np.ndarray):
            return prepared

        norm_key, post_flip_axes = prepared
        with self.lock:
            if self.tensor is None:
                raise ValueError(
                    "I/O operation on closed LazyTensorArray "
                    f"({self._var_name!r})."
                )

            r = self.tensor[norm_key]
            field_index = r.field_indexes(norm_key)
            if self.tensor.is_full_field(field_index):
                field_index = None

            result = r.to_array(
                index=field_index,
                array_namespace=self.array_namespace,
                dtype=self.dtype,
            )

            singles = [
                i
                for i in range(len(r.user_shape))
                if isinstance(norm_key[i], int)
            ]
            if singles:
                result = result.squeeze(axis=tuple(singles))

            arr = np.asarray(result)
            if post_flip_axes:
                arr = np.flip(arr, axis=post_flip_axes)

            return arr

    def __array__(self, dtype=None, copy=None):
        """Materialise the entire array via `numpy.asarray(self)`.

        :Parameters:

            dtype: `numpy.dtype` or type, optional
                Optional target dtype requested by `numpy`.

            copy: `bool`, optional
                Accepted for `numpy` 2.x `__array__` protocol
                compatibility.

        :Returns:

            `numpy.ndarray`
                Full materialised data array.

        """
        arr = self[tuple(slice(None) for _ in self.shape)]
        if dtype is not None:
            return np.asarray(arr, dtype=dtype)

        return arr

    def __repr__(self):
        """Return a concise string representation of the lazy array."""
        return (
            f"LazyTensorArray(name={self._var_name!r}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )


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
            if isinstance(data, LazyTensorArray):
                data.close()

    def __enter__(self):
        """Enter the runtime context for this specification."""
        return self

    def __exit__(self, *exc):
        """Exit and close all lazy variable arrays."""
        self.close()


def _resolve_profile(ds, profile):
    """Resolve the GRIB profile.

    Resolves ``profile="auto"`` into a concrete `earthkit.data` profile
    name.

    :Parameters:

        ds: `earthkit.data.FieldList`
            Input field list inspected when *profile* is ``'auto'``.

        profile: `str` or `Profile`
            Requested profile name or object.

    :Returns:

        `str` or `Profile`
            Resolved profile name (``'mars'`` if MARS namespace
            metadata is detected, ``'grib'`` otherwise) or the
            original *profile* value if not ``'auto'``.

    """
    if not isinstance(profile, str) or profile != "auto":
        return profile

    from earthkit.data.xr_engine import profile as ek_profile

    for fn_name in ("auto_profile", "guess_profile", "resolve_profile"):
        fn = getattr(ek_profile, fn_name, None)
        if callable(fn):
            try:
                resolved = fn(ds)
                if resolved and resolved != "auto":
                    return resolved

            except Exception:
                pass

    profile_cls = getattr(ek_profile, "Profile", None)
    if profile_cls is not None:
        for meth_name in ("auto", "guess", "from_ds", "from_fieldlist"):
            meth = getattr(profile_cls, meth_name, None)
            if callable(meth):
                try:
                    resolved = meth(ds)
                    if resolved and resolved != "auto":
                        return resolved

                except Exception:
                    pass

    profiles_dir = os.path.join(
        os.path.dirname(ek_profile.__file__), "profiles"
    )
    if os.path.exists(os.path.join(profiles_dir, "mars.yaml")):
        try:
            if len(ds) > 0:
                fld = ds[0]
                if callable(getattr(fld, "as_namespace", None)):
                    mars_meta = fld.as_namespace("mars")
                    if mars_meta:
                        return "mars"

                if (
                    _field_meta(fld, "mars.class") is not None
                    or _field_meta(fld, "levtype") is not None
                ):
                    return "mars"

        except Exception:
            pass

    return "grib"


@lru_cache(maxsize=1)
def _get_dataset_builder_cls():
    """Construct and cache the `FlatNetCDFDatasetBuilder` class.

    :Returns:

            Cached `FlatNetCDFDatasetBuilder` subclass of
            `earthkit.data.xr_engine.builder.SingleDatasetBuilder`.

    """
    # Initialise eckit before importing
    # `earthkit.data.xr_engine.builder` to prevent C++ library
    # initialisation order conflicts with `pyproj`.
    try:
        import eckit  # noqa: #401
    except ImportError:
        pass

    from earthkit.data.xr_engine.accessor import EarthkitAttrsBuilder
    from earthkit.data.xr_engine.builder import (
        BackendDataBuilder,
        LevelPerTypeDim,
        SingleDatasetBuilder,
    )

    class FlatNetCDFBackendDataBuilder(BackendDataBuilder):
        """Backend builder.

        Backend builder producing plain dictionaries and
        `LazyTensorArray`s.

        """

        def __init__(
            self, *args, squeeze_singletons=False, lock=None, **kwargs
        ):
            """Initialise the flat netCDF backend data builder.

            :Parameters:

                *args
                    Positional arguments forwarded to
                    `BackendDataBuilder`.

                squeeze_singletons: `bool`, default False
                    Whether to drop size 1, non-horizontal dimensions
                    after dimension renaming and ownership tracking.

                lock: `_PicklableLock`, optional
                    The lock shared by the `LazyTensorArray`
                    instances that are created.

                **kwargs
                    Keyword arguments forwarded to
                    `BackendDataBuilder`.

            """
            super().__init__(*args, **kwargs)
            self._squeeze_singletons = squeeze_singletons
            self._lock = lock

        def _make_field_coords(self):
            """Make coordinates.

            Builds horizontal grid coordinate entries from `self.grid`.

            :Returns:

                `dict`
                    Grid coordinate variables mapped to ``(dims,
                    values, attrs)``.

            """
            r = {}
            for k, v in self.grid.coords.items():
                dims = tuple(self.grid.coords_dim[k])
                attrs = dict(self.profile.attrs.coord_attrs.get(k, {}))
                r[k] = (dims, np.asarray(v), attrs)

            return r

        def coords(self):
            """Collect and convert all tensor and field coordinates.

            :Returns:

                `dict`
                    Combined coordinate dictionary mapped to ``(dims,
                    values, attrs)``.

            """
            r = {}
            used: set[str] = set()
            for k, coord in self.tensor_coords.items():
                # convert() must run before attrs() because StepCoord
                # sets self.resolution in convert()
                vals = coord.convert(self.profile)
                attrs = dict(coord.attrs(self.profile))
                r[_unique_name(k, used)] = (tuple(coord.dims), vals, attrs)

            for k, entry in self.field_coords.items():
                r[_unique_name(k, used)] = entry

            return r

        def build_values(self, tensor, var_dims, name):
            """Wrap or materialise a variable's `earthkit` field tensor.

            :Parameters:

                tensor:
                    `earthkit.data` tensor object for a single
                    variable.

                var_dims: iterable of `str`
                    Dimension names corresponding to *tensor*'s axes.

                name: `str`
                    Variable name.

            :Returns:

                `LazyTensorArray` or `numpy.ndarray`
                   `LazyTensorArray` when `self.profile.lazy_load` is
                    True, or a concrete `numpy.ndarray` otherwise.

            """
            tensor.source = tensor.source.unwrap()
            if not self.profile.lazy_load:
                return np.asarray(
                    tensor.to_array(
                        array_namespace=self.array_namespace, dtype=self.dtype
                    )
                )

            return LazyTensorArray(
                tensor,
                var_dims,
                tensor.full_shape,
                self.array_namespace,
                self.dtype,
                name,
                lock=self._lock,
            )

        @staticmethod
        def _apply_dim_rename(mapping, dimensions, all_coords, data_vars):
            """Rename dimensions.

            Renames dimensions across *dimensions*, *all_coords*, and
            *data_vars*.

            :Parameters:

                mapping: `dict`
                    Source-to-target dimension name mapping.

                dimensions: `dict`
                    Current dimension sizes.

                all_coords: `dict`
                    Current coordinate entries.

                data_vars: `dict`
                    Current data variable entries.

            :Returns:

                (`dict`, `dict`, `dict`)
                    Updated ``(dimensions, all_coords, data_vars)``.

            """
            if not mapping:
                return dimensions, all_coords, data_vars

            used_dims = set()
            dim_map = {}
            new_dims = {}
            for k, v in dimensions.items():
                new_k = _unique_name(mapping.get(k, k), used_dims)
                dim_map[k] = new_k
                new_dims[new_k] = v

            used_coords = set()
            new_coords = {}
            for k, (dims, vals, attrs) in all_coords.items():
                mapped_dims = tuple(dim_map.get(d, d) for d in dims)
                desired_k = dim_map.get(k, mapping.get(k, k))
                new_k = _unique_name(desired_k, used_coords)
                new_coords[new_k] = (mapped_dims, vals, attrs)

            new_vars = {
                k: (tuple(dim_map.get(d, d) for d in dims), data, attrs)
                for k, (dims, data, attrs) in data_vars.items()
            }
            return new_dims, new_coords, new_vars

        @staticmethod
        def _squeeze_singleton_dim(
            dim_name, dimensions, all_coords, data_vars
        ):
            """Drop a size 1 dimension.

            Drops a size 1 dimension from ``dimensions``,
            ``all_coords``, and ``data_vars``.

            :Parameters:

                dim_name: `str`
                    Name of the singleton dimension to squeeze.

                dimensions: `dict`
                    Current dimension sizes.

                all_coords: `dict`
                    Current coordinate entries.

                data_vars: `dict`
                    Current data variable entries.

                :Returns:

                    (`dict`, `dict`, dict)
                        Updated ``(dimensions, all_coords,
                        data_vars)`` with ``dim_name`` removed.

            """
            if dimensions.get(dim_name) != 1:
                return dimensions, all_coords, data_vars

            new_dims = {k: v for k, v in dimensions.items() if k != dim_name}

            new_coords = {}
            for k, (dims, vals, attrs) in all_coords.items():
                if dim_name in dims:
                    ax = dims.index(dim_name)
                    squeezed_dims = tuple(d for d in dims if d != dim_name)
                    squeezed_vals = np.squeeze(np.asarray(vals), axis=ax)
                    new_coords[k] = (squeezed_dims, squeezed_vals, attrs)
                else:
                    new_coords[k] = (dims, vals, attrs)

            new_vars = {}
            for k, (dims, data, attrs) in data_vars.items():
                if dim_name in dims:
                    ax = dims.index(dim_name)
                    squeezed_dims = tuple(d for d in dims if d != dim_name)
                    if isinstance(data, LazyTensorArray):
                        squeezed_data = data.with_squeezed_axis(ax)
                    else:
                        squeezed_data = np.squeeze(data, axis=ax)

                    new_vars[k] = (squeezed_dims, squeezed_data, attrs)
                else:
                    new_vars[k] = (dims, data, attrs)

            return new_dims, new_coords, new_vars

        def build(self):  # noqa: D205
            """Extract raw dimensions, coordinates, variables, and GRIB
            metadata hints.

            :Returns:

                `tuple`
                     Nine-element `tuple` ``(dimensions, coord_vars,
                     dedup_data_vars, dedup_aux_vars, global_attrs,
                     var_orig_dims, coord_orig_dims, raw_cell_methods,
                     (step_intervals_s, level_intervals))`` ready for
                     `_apply_cf_polish`.

            """
            if self.profile.allow_holes:
                global_tensor_dims, self.raw_global_tensor_coords, _ = (
                    self.prepare_tensor(self.ds, self.dims, "<ALL VARIABLES>")
                )
                for d in global_tensor_dims:
                    if isinstance(d, LevelPerTypeDim):
                        raise NotImplementedError(
                            "level_dim_mode='level_per_type' not yet "
                            "supported when allow_holes=True"
                        )

                    k, c = d.as_coord(
                        self.raw_global_tensor_coords[d.key], self.ds
                    )
                    self.tensor_coords[k] = c

            var_builders = self.pre_build_variables()

            if (
                self.profile.add_valid_time_coord
                and "valid_time" not in [d.name for d in self.dims]
                and "time.valid_datetime" not in self.tensor_coords
            ):
                time_dim_names = self.profile.dims.active_time_dim_names
                if time_dim_names:
                    self.profile.aux_coords.setdefault(
                        "valid_time", ("time.valid_datetime", time_dim_names)
                    )

            self.collect_aux_coords()

            global_attrs = dict(
                self.profile.attrs.builder.build(
                    self.ds, var_builders, rename=True
                )
            )
            all_coords = self.coords()

            data_vars = {}
            raw_cell_methods = {}
            used_raw_vars = set()
            for k, vb in var_builders.items():
                attrs = dict(vb.attrs)
                if self.profile.add_earthkit_attrs:
                    ek_attrs = EarthkitAttrsBuilder().build(
                        vb.tensor.source[0]
                    )
                    if ek_attrs:
                        attrs.update(ek_attrs)

                attrs.update(vb.fixed_local_attrs)
                data = vb.data_maker(vb.tensor, vb.var_dims, vb.name)
                vname = _unique_name(
                    self.profile.variable.rename(k), used_raw_vars
                )
                data_vars[vname] = (tuple(vb.var_dims), data, attrs)
                raw_cell_methods[vname] = _extract_grib_cell_method_hints(
                    vb.tensor.source
                )

            step_intervals_s, level_intervals = _extract_grib_interval_hints(
                self.ds
            )

            dimensions = {}
            for dims, data, _ in data_vars.values():
                for d, n in zip(dims, data.shape):
                    dimensions.setdefault(d, int(n))

            for dims, vals, _ in all_coords.values():
                for d, n in zip(dims, np.shape(vals)):
                    dimensions.setdefault(d, int(n))

            dim_handler = self.profile.dims
            pass1 = {
                d.key: d.get_simple_name()
                for d in dim_handler.dims.values()
                if d.key in dimensions and d.get_simple_name() != d.key
            }
            dimensions, all_coords, data_vars = self._apply_dim_rename(
                pass1, dimensions, all_coords, data_vars
            )

            if dim_handler.rename_dims_map:
                pass2 = {
                    k: v
                    for k, v in dim_handler.rename_dims_map.items()
                    if k in dimensions
                }
                dimensions, all_coords, data_vars = self._apply_dim_rename(
                    pass2, dimensions, all_coords, data_vars
                )

            # Record pre-squeeze dimension ownership so scalar coords
            # only link to their parent vars
            var_orig_dims = {
                k: set(dims) for k, (dims, _, _) in data_vars.items()
            }
            coord_orig_dims = {
                k: set(dims) for k, (dims, _, _) in all_coords.items()
            }

            grid_dims = {
                d for dims in self.grid.coords_dim.values() for d in dims
            }
            if self._squeeze_singletons:
                to_squeeze = [
                    d
                    for d, n in dimensions.items()
                    if n == 1 and d not in grid_dims
                ]
            else:
                to_squeeze = [
                    d for d in _LEVEL_TYPE_NAMES if dimensions.get(d) == 1
                ]

            for dim_name in to_squeeze:
                dimensions, all_coords, data_vars = (
                    self._squeeze_singleton_dim(
                        dim_name, dimensions, all_coords, data_vars
                    )
                )

            # Fold scalar level_type/levtype (MARS) onto the vertical
            # coordinate
            for lt_key in _LEVEL_TYPE_NAMES:
                if lt_key in all_coords and all_coords[lt_key][0] == ():
                    lt_val = str(np.asarray(all_coords[lt_key][1]).item())
                    rename_map = dim_handler.rename_dims_map or {}
                    preferred = [
                        rename_map.get("level", "level"),
                        rename_map.get("levelist", "levelist"),
                        "level",
                        "levelist",
                    ]
                    target_vert = next(
                        (k for k in preferred if k in all_coords),
                        next(
                            (
                                k
                                for k in all_coords
                                if k in _VERTICAL_NAMES
                                and k not in _LEVEL_TYPE_NAMES
                            ),
                            None,
                        ),
                    )
                    if target_vert is not None:
                        v_dims, v_vals, v_attrs = all_coords[target_vert]
                        v_attrs = dict(v_attrs)
                        v_attrs.setdefault("level_type", lt_val)
                        all_coords[target_vert] = (v_dims, v_vals, v_attrs)
                        del all_coords[lt_key]
                        coord_orig_dims.pop(lt_key, None)

            coord_vars = {}
            raw_aux_vars = {}
            for name, entry in all_coords.items():
                dims, vals, _ = entry
                is_numeric_or_temporal = (
                    np.asarray(vals).dtype.kind in _NUMERIC_DTYPE_KINDS
                    or _is_datetime_values(vals)
                    or _is_timedelta_values(vals)
                )
                if (
                    dims == (name,)
                    and name in dimensions
                    and is_numeric_or_temporal
                ):
                    coord_vars[name] = entry
                else:
                    raw_aux_vars[name] = entry

            # Enforce a single collision-free variable namespace
            # across coord_vars, data_vars, and aux_vars
            used_var_names: set[str] = set(coord_vars.keys())
            dim_names = set(dimensions.keys())

            dedup_data_vars = {}
            for vname, entry in data_vars.items():
                forbidden = used_var_names | dim_names
                new_vname = _unique_name(vname, forbidden)
                used_var_names.add(new_vname)
                dedup_data_vars[new_vname] = entry
                if new_vname != vname:
                    if vname in var_orig_dims:
                        var_orig_dims[new_vname] = var_orig_dims.pop(vname)

                    if vname in raw_cell_methods:
                        raw_cell_methods[new_vname] = raw_cell_methods.pop(
                            vname
                        )

            dedup_aux_vars = {}
            for aname, entry in raw_aux_vars.items():
                adims = entry[0]
                is_own_dim_aux = adims == (aname,) and aname in dim_names
                if is_own_dim_aux:
                    forbidden = used_var_names
                else:
                    forbidden = used_var_names | dim_names

                new_aname = _unique_name(aname, forbidden)
                used_var_names.add(new_aname)
                dedup_aux_vars[new_aname] = entry
                if new_aname != aname and aname in coord_orig_dims:
                    coord_orig_dims[new_aname] = coord_orig_dims.pop(aname)

            return (
                dimensions,
                coord_vars,
                dedup_data_vars,
                dedup_aux_vars,
                global_attrs,
                var_orig_dims,
                coord_orig_dims,
                raw_cell_methods,
                (step_intervals_s, level_intervals),
            )

    class FlatNetCDFDatasetBuilder(SingleDatasetBuilder):  # noqa :D209
        """`SingleDatasetBuilder` subclass wired to
        `FlatNetCDFBackendDataBuilder`."""

        def __init__(
            self, ds, profile="auto", backend_kwargs=None, lock=None, **kwargs
        ):
            """Initialise the dataset builder .

            Initialises the dataset builder with profile resolution
            and squeeze handling.

            :Parameters:

                ds: `earthkit.data.FieldList`
                    Input field list.

                profile: `str` or `Profile`, default "auto"
                    Profile name or object; ``"auto"`` resolves via
                    `_resolve_profile`.

                backend_kwargs: `dict`, optional
                    Backend configuration keyword arguments.

                lock: `_PicklableLock`, optional
                    The lock shared by all arrays read from *ds*.

                **kwargs
                    Additional backend keyword arguments merged over
                    ``backend_kwargs``.

            """
            merged = dict(backend_kwargs or {})
            merged.update(kwargs)
            user_squeeze = merged.get("squeeze", None)
            merged["squeeze"] = False
            resolved_profile = _resolve_profile(ds, profile)
            super().__init__(ds, resolved_profile, backend_kwargs=merged)
            profile_name = getattr(
                getattr(self, "profile", None), "name", resolved_profile
            )
            if user_squeeze is not None:
                squeeze_singletons = bool(user_squeeze)
            else:
                squeeze_singletons = profile_name != "earthkit"

            self.builder = lambda *a, **kw: FlatNetCDFBackendDataBuilder(
                *a,
                squeeze_singletons=squeeze_singletons,
                lock=lock,
                **kw,
            )

    return FlatNetCDFDatasetBuilder


# ---------------------------------------------------------------------------
# CF polish pass
# ---------------------------------------------------------------------------


def _is_cftime_datetime(x):
    """Check whether *x*x is a datetime-like object.

    :Parameters:

        x:
            The object to inspect.

    :Returns:

        `bool`
            True if *x* exposes ``calendar``, ``year``, ``month``, and
            ``day`` attributes, False otherwise.

    """
    return (
        hasattr(x, "calendar")
        and hasattr(x, "year")
        and hasattr(x, "month")
        and hasattr(x, "day")
    )


def _is_datetime_values(vals):
    """Check whether *vals* contains datetime-like elements.

    :Parameters:
    vals: array_like
        Coordinate array or sequence to inspect.

    :Returns:

    bool
        True if ``vals`` has ``datetime64`` dtype or contains
        `datetime.datetime`, `numpy.datetime64`, or
        ``cftime.datetime`` elements.

    """
    if isinstance(vals, np.ndarray) and np.issubdtype(
        vals.dtype, np.datetime64
    ):
        return True

    flat = np.asarray(vals, dtype=object).ravel()
    return any(
        isinstance(x, (datetime.datetime, np.datetime64))
        or _is_cftime_datetime(x)
        for x in flat
    )


def _is_timedelta_values(vals):
    """Check whether ``vals`` contains timedelta-like elements.

    :Parameters:

        vals: array_like
            Coordinate array or sequence to inspect.

    :Returns:

        bool
            True if ``vals`` has ``timedelta64`` dtype or contains
            `datetime.timedelta` or `numpy.timedelta64` elements.

    """
    if isinstance(vals, np.ndarray) and np.issubdtype(
        vals.dtype, np.timedelta64
    ):
        return True

    flat = np.asarray(vals, dtype=object).ravel()
    return any(
        isinstance(x, (datetime.timedelta, np.timedelta64)) for x in flat
    )


def _to_dt64_us(x):
    """Convert a scalar datetime or missing marker to `datetime64[us]`.

    :Parameters:

        x: `datetime.datetime`, `numpy.datetime64`, `str`, `None`, or `float`
            Scalar timestamp or ``None``/``NaN`` missing value.

    :Returns:

        `numpy.datetime64`
            Microsecond-precision UTC-naive `numpy.datetime64` value
            (or ``NaT``).

    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.datetime64("NaT", "us")

    if isinstance(x, datetime.datetime):
        if x.tzinfo is not None:
            x = x.astimezone(datetime.timezone.utc).replace(tzinfo=None)

        return np.datetime64(x, "us")

    return np.datetime64(x, "us")


def _to_cftime_dt(x, calendar, cftime_mod):
    """Convert a timestamp into a `cftime.datetime` for ``calendar``.

    :Parameters:

        x: `numpy.datetime64`, `datetime.datetime`, `cftime.datetime`, `None`, or `float`
            Scalar timestamp or ``None``/``NaN`` missing value.

        calendar: `str`
            Target CF calendar name.

        cftime_mod: module
            Imported ``cftime`` module.

    :Returns:

        `cftime.datetime` or `None`
            Converted ``cftime`` timestamp, or ``None`` if ``x`` is
            missing/``NaT``.

    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None

    if isinstance(x, np.datetime64):
        if np.isnat(x):
            return None

        dt_us = x.astype("datetime64[us]")
        days_since_epoch = dt_us.astype("datetime64[D]")
        year = (
            int(days_since_epoch.astype("datetime64[Y]").astype(np.int64))
            + 1970
        )
        month = (
            int(days_since_epoch.astype("datetime64[M]").astype(np.int64)) % 12
            + 1
        )
        day = (
            int(
                (days_since_epoch - days_since_epoch.astype("datetime64[M]"))
                / np.timedelta64(1, "D")
            )
            + 1
        )
        rem_us = int(
            (dt_us - days_since_epoch.astype("datetime64[us]"))
            / np.timedelta64(1, "us")
        )
        hour, rem_us = divmod(rem_us, 3_600_000_000)
        minute, rem_us = divmod(rem_us, 60_000_000)
        second, microsecond = divmod(rem_us, 1_000_000)
        return cftime_mod.datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond,
            calendar=calendar,
        )

    if isinstance(x, datetime.datetime):
        if x.tzinfo is not None:
            x = x.astimezone(datetime.timezone.utc).replace(tzinfo=None)

        return cftime_mod.datetime(
            x.year,
            x.month,
            x.day,
            x.hour,
            x.minute,
            x.second,
            x.microsecond,
            calendar=calendar,
        )

    if _is_cftime_datetime(x):
        if getattr(x, "calendar", None) == calendar:
            return x

        return cftime_mod.datetime(
            x.year,
            x.month,
            x.day,
            x.hour,
            x.minute,
            x.second,
            getattr(x, "microsecond", 0),
            calendar=calendar,
        )

    raise TypeError(f"Unsupported datetime value for cftime encoding: {x!r}")


def _select_cf_time_unit_from_seconds(seconds):
    """Choose the coarsest CF time unit.

    Chooses ``'days'``, ``'hours'``, or ``'seconds'``.

    :Parameters:

        seconds: `numpy.ndarray`
            1-d array of elapsed seconds relative to the reference
            epoch.

    :Returns:

        `str`
            ``"days"``, ``"hours"``, or ``"seconds"`` based on the
            minimum spacing between unique timestamps or exact
            divisibility for single timestamps.

    """
    if seconds.size == 0:
        return "days"

    unique_seconds = np.unique(seconds)
    if unique_seconds.size >= 2:
        min_delta_s = float(np.min(np.diff(unique_seconds)))
        if min_delta_s >= 86400.0:
            return "days"

        if min_delta_s >= 60.0:
            return "hours"

        return "seconds"

    offset_s = float(unique_seconds[0])
    if offset_s.is_integer():
        offset_int = int(offset_s)
        if offset_int % 86400 == 0:
            return "days"

        if offset_int % 3600 == 0:
            return "hours"

    return "seconds"


def _format_ref_year(year):
    """Format a year for a CF time epoch.

    Formats a four-digit (or signed four-digit) year for a CF time
    epoch.

    :Parameters:

        year: `int`
            Calendar year of the reference origin.

    :Returns:

        `str`
            Zero-padded year string (e.g. ``'2026'`` or ``'-0050'``).

    """
    if year >= 0:
        return f"{year:04d}"

    return f"-{abs(year):04d}"


def _encode_cf_datetime(vals, attrs, calendar=CF_CALENDAR):
    """Encode datetime values into CF numeric offsets and attributes.

    :Parameters:

        vals: array_like
            Datetime values to encode.

        attrs: `dict`
            Existing coordinate attributes to copy and update.

        calendar: `str`, default "proleptic_gregorian"
            Target CF calendar name.

    :Returns:

        `numpy.ndarray`, `dict`
            ``(numeric_offsets, updated_attrs)`` with ``float64``
            offsets relative to ``YYYY-01-01``, ``NC_FILL_DOUBLE`` for
            missing entries, and CF ``units`` and ``calendar``
            attributes populated.

    """
    raw = np.asarray(vals, dtype=object)
    has_cftime = any(_is_cftime_datetime(x) for x in raw.ravel())
    use_cftime = has_cftime or calendar.lower() != CF_CALENDAR

    if use_cftime:
        try:
            import cftime
        except ImportError as exc:
            raise ImportError(
                "cftime is required to encode timestamps with "
                f"calendar={calendar!r}"
            ) from exc

        converted = [_to_cftime_dt(x, calendar, cftime) for x in raw.ravel()]
        nat_mask = np.array(
            [x is None for x in converted], dtype=bool
        ).reshape(raw.shape)
        valid = [x for x in converted if x is not None]

        if valid:
            year = valid[0].year
        else:
            year = 1970

        year_str = _format_ref_year(year)

        if valid:
            secs = np.asarray(
                cftime.date2num(
                    valid, f"seconds since {year_str}-01-01", calendar=calendar
                ),
                dtype=np.float64,
            )
            unit_name = _select_cf_time_unit_from_seconds(secs)
            valid_num = np.asarray(
                cftime.date2num(
                    valid,
                    f"{unit_name} since {year_str}-01-01",
                    calendar=calendar,
                ),
                dtype=np.float64,
            )
        else:
            unit_name = "days"
            valid_num = np.empty(0, dtype=np.float64)

        numeric = np.empty(raw.shape, dtype=np.float64)
        if valid_num.size > 0:
            numeric[~nat_mask] = valid_num
    else:
        if isinstance(vals, np.ndarray) and np.issubdtype(
            vals.dtype, np.datetime64
        ):
            dt64 = vals.astype("datetime64[us]")
        else:
            converted = [_to_dt64_us(x) for x in raw.ravel()]
            dt64 = np.array(converted, dtype="datetime64[us]").reshape(
                raw.shape
            )

        nat_mask = np.isnat(dt64)
        valid = dt64[~nat_mask]

        if valid.size > 0:
            year = int(str(valid[0].astype("datetime64[Y]")))
        else:
            year = 1970

        year_str = _format_ref_year(year)
        ref_dt64 = np.datetime64(f"{year_str}-01-01T00:00:00", "us")

        if valid.size > 0:
            secs = ((valid - ref_dt64) / np.timedelta64(1, "s")).astype(
                np.float64
            )
            unit_name = _select_cf_time_unit_from_seconds(secs)
            if unit_name == "days":
                scale_s = 86400.0
            elif unit_name == "hours":
                scale_s = 3600.0
            else:
                scale_s = 1.0

            valid_num = secs / scale_s
        else:
            unit_name = "days"
            valid_num = np.empty(0, dtype=np.float64)

        numeric = np.empty(dt64.shape, dtype=np.float64)
        if valid_num.size > 0:
            numeric[~nat_mask] = valid_num

    out_attrs = dict(attrs)
    if nat_mask.any():
        numeric[nat_mask] = NC_FILL_DOUBLE
        out_attrs.setdefault("_FillValue", NC_FILL_DOUBLE)

    out_attrs["units"] = f"{unit_name} since {year_str}-01-01"
    out_attrs["calendar"] = calendar
    return numeric, out_attrs


def _encode_cf_timedelta(vals, attrs):
    """Encode timedelta values into integer CF ``hours`` or ``seconds``.

    :Parameters:

        vals: array_like
            Timedelta values to encode.

        attrs: `dict`
            Existing coordinate attributes to copy and update.

    :Returns:

        (`numpy.ndarray`, `dict`)
            ``(numeric_durations, updated_attrs)`` with ``int64``
            values (in ``"hours"`` when all valid entries are exact
            multiples of 3600 s, otherwise ``"seconds"``) and
            ``NC_FILL_INT64`` for missing entries.

    """
    if isinstance(vals, np.ndarray) and np.issubdtype(
        vals.dtype, np.timedelta64
    ):
        td64 = vals.astype("timedelta64[s]")
    else:
        raw = np.asarray(vals, dtype=object)
        converted = [
            (
                np.timedelta64("NaT", "s")
                if (x is None or (isinstance(x, float) and math.isnan(x)))
                else (
                    np.timedelta64(int(x.total_seconds()), "s")
                    if isinstance(x, datetime.timedelta)
                    else np.timedelta64(x, "s")
                )
            )
            for x in raw.ravel()
        ]
        td64 = np.array(converted, dtype="timedelta64[s]").reshape(raw.shape)

    nat_mask = np.isnat(td64)
    secs = td64.astype(np.int64)
    valid = secs[~nat_mask]
    use_hours = valid.size > 0 and np.all(valid % 3600 == 0)

    if use_hours:
        numeric = (secs // 3600).astype(np.int64)
        unit = "hours"
    else:
        numeric = secs
        unit = "seconds"

    out_attrs = dict(attrs)
    if nat_mask.any():
        numeric[nat_mask] = NC_FILL_INT64
        out_attrs.setdefault("_FillValue", NC_FILL_INT64)

    out_attrs.setdefault("units", unit)
    return numeric, out_attrs


def _sanitize_nc_attrs(attrs):
    """Coerce attribute values into netCDF-serializable primitive types.

    Drops ``None`` and nested ``dict`` entries, formats datetime
    objects as ISO strings, and casts ``bool`` values to ``int``.

    :Parameters:

        attrs: `dict`
            Raw attribute dictionary.

    :Returns:

        `dict`
            Sanitized attribute dictionary safe for netCDF
            serialization.

    """
    clean = {}
    for k, v in attrs.items():
        if v is None or isinstance(v, dict):
            continue

        if isinstance(v, datetime.datetime):
            clean[k] = v.strftime("%Y-%m-%dT%H:%M:%S")
        elif _is_cftime_datetime(v):
            clean[k] = (
                f"{v.year:04d}-{v.month:02d}-{v.day:02d}T{v.hour:02d}:{v.minute:02d}:{v.second:02d}"
            )
        elif isinstance(v, bool):
            clean[k] = int(v)
        else:
            clean[k] = v

    return clean


def _apply_vertical_attrs(name, attrs):
    """Populate CF "axis" and "positive attributes on Z coordinates.

    :Parameters:

        name: `str`
            Coordinate variable name.

        attrs: `dict`
            Mutable attribute dictionary updated in place.

    """
    if name not in _VERTICAL_NAMES and "isobaric" not in name.lower():
        return

    attrs.setdefault("axis", "Z")
    units_lower = str(attrs.get("units", "")).lower()
    lt_lower = str(attrs.get("level_type", "")).lower()
    name_lower = name.lower()
    if (
        "isobaric" in name_lower
        or "isobaric" in lt_lower
        or lt_lower == "pl"
        or units_lower in _PRESSURE_UNITS
        or "depth" in name_lower
        or "depth" in lt_lower
        or lt_lower == "dp"
    ):
        attrs.setdefault("positive", "down")
    elif (
        "height" in name_lower
        or "altitude" in name_lower
        or "height" in lt_lower
        or "altitude" in lt_lower
    ):
        attrs.setdefault("positive", "up")


def _is_pressure_coord(name, attrs):
    """Check whether a coordinate represents atmospheric pressure.

    :Parameters:

        name: `str`
            Coordinate variable name.

        attrs: `dict`
            Coordinate attributes dictionary.

    :Returns:

        `bool`
            `True` if *name* or *attrs* indicates an isobaric/pressure
            axis.

    """
    std_name = str(attrs.get("standard_name", "")).lower()
    units_lower = str(attrs.get("units", "")).lower()
    lt_lower = str(attrs.get("level_type", "")).lower()
    name_lower = name.lower()
    return (
        std_name == "air_pressure"
        or "isobaric" in name_lower
        or "isobaric" in lt_lower
        or lt_lower == "pl"
        or units_lower in _PRESSURE_UNITS
    )


def _format_cell_methods(vdims, hints, coord_vars):
    """Format a CF cell_methods string from extracted GRIB hints.

    :Parameters:

         vdims: `tuple` of `str`
             Dimension names of the target data variable.

         hints: `tuple` of (`str` or `None`, `str` or `None`), optional
             ``(temporal_cf_method, ensemble_cf_method)`` inferred
             from GRIB keys.

         coord_vars: dict of {str: tuple}
             Polished dimension coordinates used to locate a
             ``T``-axis fallback.

     :Returns:

         `str` or `None`
             Formatted CF ``cell_methods`` attribute string, or
             ``None`` if no methods apply.

    """
    if not hints:
        return None

    temporal_method, ensemble_method = hints
    parts: list[str] = []
    if ensemble_method:
        parts.append(ensemble_method)

    if temporal_method:
        if "time" in vdims:
            t_axis = "time"
        elif "valid_time" in vdims:
            t_axis = "valid_time"
        elif "step" in vdims:
            t_axis = "step"
        else:
            t_axis = next(
                (
                    d
                    for d in vdims
                    if d in coord_vars and coord_vars[d][2].get("axis") == "T"
                ),
                "time",
            )

        parts.append(f"{t_axis}: {temporal_method}")

    if parts:
        return " ".join(parts)

    return None


def _compute_1d_spatial_bounds(name, arr):
    """Compute CF (N, 2) cell bounds for a monotonic 1-d spatial axis.

    :Parameters:

        name: `str`
            Coordinate name (``"latitude"`` bounds are clipped to
            ``[-90, 90]``).

        arr: `numpy.ndarray`
            1-d numeric coordinate array of size at least 2.

    :Returns:

        `numpy.ndarray` or `None`
            ``(N, 2)`` array of lower and upper cell edges, or
            ``None`` if ``arr`` is not 1-d, has fewer than 2 finite
            points, or is non-monotonic.

    """
    if (
        arr.ndim != 1
        or arr.size < 2
        or arr.dtype.kind not in _NUMERIC_DTYPE_KINDS
    ):
        return None

    vals = arr.astype(np.float64)
    if not np.all(np.isfinite(vals)):
        return None

    diffs = np.diff(vals)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        return None

    edges = np.empty(vals.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
    edges[0] = vals[0] - 0.5 * diffs[0]
    edges[-1] = vals[-1] + 0.5 * diffs[-1]

    if name == "latitude":
        edges = np.clip(edges, -90.0, 90.0)

    if arr.dtype.kind == "f":
        out_dtype = arr.dtype
    else:
        out_dtype = np.float64

    return np.column_stack([edges[:-1], edges[1:]]).astype(
        out_dtype, copy=False
    )


def _cf_unit_scale_seconds(units):
    """Return the factor converting a CF time unit to seconds.

    Returns the multiplicative factor converting a CF time unit to
    seconds.

    :Parameters:

        units:
            CF time units string (e.g. ``"hours"`` or ``"hours since
            2024-01-01"``).

    :Returns:

    `float` or `None`
        Number of seconds per unit (``1.0``, ``60.0``, ``3600.0``, or

    """
    if not isinstance(units, str):
        return None

    head = units.strip().split()[0].lower()
    if head in ("s", "sec", "secs", "second", "seconds"):
        return 1.0

    if head in ("min", "mins", "minute", "minutes"):
        return 60.0

    if head in ("h", "hr", "hrs", "hour", "hours"):
        return 3600.0

    if head in ("d", "day", "days"):
        return 86400.0

    return None


def _build_cell_bounds(dimensions, coord_vars, aux_vars, interval_hints):
    """Build CF bounds variables for spatial and temporal axes.

    Mutates coordinate attribute dictionaries in ``coord_vars`` and
    ``aux_vars`` to attach ``"bounds": "<cname>_bnds"`` references
    where bounds are created.

    :Parameters:

        dimensions: `dict`
            Current dataset dimensions.

        coord_vars: `dict`
            Polished 1-d dimension coordinates.

        aux_vars: `dict`
            Polished auxiliary coordinates.

        interval_hints: `tuple` of (`dict`, `dict`), optional
            ``(step_intervals_s, level_intervals)`` extracted by
            `_extract_grib_interval_hints`.

    :Returns:

        (`dict`, `dict`, `set` of `str`)
            ``(new_dims, bnds_vars, bnds_names)`` containing updated
            dimensions (including ``bnds=2`` if used), generated
            boundary variables, and the set of generated boundary
            variable names.

    """
    step_intervals_s, level_intervals = interval_hints or ({}, {})
    new_dims = dict(dimensions)
    bnds_vars = {}
    used_names: set[str] = (
        set(new_dims.keys()) | set(coord_vars.keys()) | set(aux_vars.keys())
    )

    if new_dims.get("bnds") in (None, 2):
        bnds_dim = "bnds"
    else:
        bnds_dim = _unique_name("bnds", used_names)

    def _register_bnds(container, cname, bnds_arr):  # noqa D205
        """Attach a ``*_bnds`` variable and set ``bounds`` on
        ``container[cname]``.

        :Parameters:

            container: `dict`
                Either ``coord_vars`` or ``aux_vars`` holding
                ``cname``.

            cname: `str`
                Coordinate variable name receiving the ``bounds``
                attribute.

            bnds_arr: `numpy.ndarray`
                Boundary array with trailing dimension of size 2.

        """
        cdims, _, cattrs = container[cname]
        if "bounds" in cattrs:
            return

        bname = _unique_name(f"{cname}_bnds", used_names)
        new_dims[bnds_dim] = 2
        cattrs["bounds"] = bname
        bnds_vars[bname] = ((*cdims, bnds_dim), bnds_arr, {})

    # Monotonic 1-d horizontal spatial coordinates
    for cname in list(coord_vars.keys()):
        if cname not in _HORIZONTAL_1D_BOUNDS_NAMES:
            continue

        cdims, arr, _ = coord_vars[cname]
        if cdims != (cname,):
            continue

        bnds_arr = _compute_1d_spatial_bounds(cname, arr)
        if bnds_arr is not None:
            _register_bnds(coord_vars, cname, bnds_arr)

    # Forecast period / time interval bounds from GRIB (startStep,
    # endStep)
    if step_intervals_s:
        if "step" in coord_vars:
            step_container = coord_vars
        elif "step" in aux_vars:
            step_container = aux_vars
        else:
            step_container = None

        duration_by_step_idx: np.ndarray | None = None

        if step_container is not None:
            sdims, sarr, sattrs = step_container["step"]
            s_scale = _cf_unit_scale_seconds(sattrs.get("units"))
            if s_scale is not None and sarr.dtype.kind in _NUMERIC_DTYPE_KINDS:
                s_flat = sarr.ravel()
                starts = np.empty(s_flat.size, dtype=sarr.dtype)
                ends = np.empty(s_flat.size, dtype=sarr.dtype)
                durations = np.zeros(s_flat.size, dtype=np.float64)
                matched_all = s_flat.size > 0
                for idx, val in enumerate(s_flat):
                    e_sec = int(round(float(val) * s_scale))
                    interval = step_intervals_s.get(e_sec)
                    if interval is None:
                        matched_all = False
                        break

                    s_sec, _ = interval
                    if sarr.dtype.kind == "f":
                        starts[idx] = s_sec / s_scale
                    else:
                        starts[idx] = int(round(s_sec / s_scale))

                    ends[idx] = val
                    durations[idx] = float(e_sec - s_sec)

                if matched_all:
                    bnds_arr = np.column_stack([starts, ends]).reshape(
                        (*sarr.shape, 2)
                    )
                    _register_bnds(step_container, "step", bnds_arr)
                    duration_by_step_idx = durations.reshape(sarr.shape)

        # Propagate interval durations to valid_time (or time if no
        # step coord exists)
        for tname in ("valid_time", "time"):
            if tname in coord_vars:
                tcontainer = coord_vars
            elif tname in aux_vars:
                tcontainer = aux_vars
            else:
                tcontainer = None

            if tcontainer is None:
                continue

            tdims, tarr, tattrs = tcontainer[tname]
            t_scale = _cf_unit_scale_seconds(tattrs.get("units"))
            if t_scale is None or tarr.dtype.kind not in _NUMERIC_DTYPE_KINDS:
                continue

            if duration_by_step_idx is not None and step_container is not None:
                sdims = step_container["step"][0]
                if sdims == tdims and duration_by_step_idx.shape == tarr.shape:
                    dur_in_t_units = duration_by_step_idx / t_scale
                    t_bnds = np.stack([tarr - dur_in_t_units, tarr], axis=-1)
                    _register_bnds(tcontainer, tname, t_bnds)
                    break

            elif len(step_intervals_s) == 1:
                s_sec, e_sec = next(iter(step_intervals_s.values()))
                dur_in_t_units = float(e_sec - s_sec) / t_scale
                t_bnds = np.stack([tarr - dur_in_t_units, tarr], axis=-1)
                _register_bnds(tcontainer, tname, t_bnds)
                break

    # Layer-type vertical coordinates (e.g. soil layers with topLevel
    # != bottomLevel)
    if level_intervals:
        for cname in _VERTICAL_NAMES:
            if cname in coord_vars:
                vcontainer = coord_vars
            elif cname in aux_vars:
                vcontainer = aux_vars
            else:
                vcontainer = None

            if vcontainer is None:
                continue

            _, varr, _ = vcontainer[cname]
            if varr.dtype.kind not in _NUMERIC_DTYPE_KINDS:
                continue

            v_flat = varr.ravel()
            tops = np.empty(v_flat.size, dtype=np.float64)
            bots = np.empty(v_flat.size, dtype=np.float64)
            matched_all = v_flat.size > 0
            for idx, val in enumerate(v_flat):
                interval = level_intervals.get(float(val))
                if interval is None:
                    matched_all = False
                    break

                tops[idx], bots[idx] = interval

            if matched_all:
                if varr.dtype.kind == "f":
                    out_dtype = varr.dtype
                else:
                    out_dtype = np.float64

                v_bnds = (
                    np.column_stack([tops, bots])
                    .reshape((*varr.shape, 2))
                    .astype(out_dtype, copy=False)
                )
                _register_bnds(vcontainer, cname, v_bnds)

    return new_dims, bnds_vars, set(bnds_vars.keys())


def _apply_cf_polish(
    dimensions,
    coord_vars,
    data_vars,
    aux_vars,
    global_attrs,
    var_orig_dims=None,
    coord_orig_dims=None,
    raw_cell_methods=None,
    interval_hints=None,
    calendar=CF_CALENDAR,
    decreasing_atmos_pressure=True,
):
    """Apply CF encodings.

    Applies CF coordinate encoding, bounds, cell methods, and
    attribute cleanup.

    :Parameters:

        dimensions: `dict`
            Raw dimension names and sizes from the dataset builder.

        coord_vars: `dict`
            Candidate 1-d dimension coordinate entries ``(dims, vals,
            attrs)``.

        data_vars: `dict`
            Data variable entries ``(dims, data, attrs)``.

        aux_vars: `dict`
            Auxiliary coordinate entries ``(dims, vals, attrs)``.

        global_attrs: `dict`
            Raw global attributes dictionary.

        var_orig_dims: `dict`, optional
            Pre-squeeze dimension sets per data variable for auxiliary
            linking.

        coord_orig_dims: `dict`, optional
            Pre-squeeze dimension sets per coordinate variable for
            auxiliary linking.

        raw_cell_methods: `dict`, optional
            Temporal and ensemble ``cell_methods`` hints per data
            variable.

        interval_hints: `tuple` of (`dict`, `dict`), optional
            Step and vertical layer bounds extracted from GRIB fields.

        calendar: `str`, default "proleptic_gregorian"
            CF calendar used when encoding datetime coordinates.

        decreasing_atmos_pressure: `bool`, default True
            If True, strictly increasing 1-s pressure axes are
            reversed to decreasing order along with matching variable
            axes.

    :Returns:

        `FlatNetCDFSpec`
            Final CF normalised specification.

    """

    def _encode_coord(name, dims, vals, attrs, is_dim_coord):
        """Encode temporal values and populate CF "axis" attributes.

        :Parameters:

            name: `str`
                Coordinate variable name.

            dims: `tuple` of `str`
                Coordinate dimension names.

            vals: array_like
                Raw coordinate values.

            attrs: `dict`
                Raw coordinate attributes.

            is_dim_coord: `bool`
                Whether the coordinate is a 1-d dimension coordinate
                eligible for ``axis`` attribution.

        :Returns:

            tuple of (tuple of `str`, numpy.ndarray, dict)
                ``(dims, encoded_array, sanitized_attrs)``.

        """
        if _is_datetime_values(vals):
            arr, attrs = _encode_cf_datetime(vals, attrs, calendar=calendar)
        elif _is_timedelta_values(vals):
            arr, attrs = _encode_cf_timedelta(vals, attrs)
        else:
            arr = np.asarray(vals)
            attrs = dict(attrs)

        if is_dim_coord and arr.dtype.kind in _NUMERIC_DTYPE_KINDS:
            if name == "latitude":
                attrs.setdefault("axis", "Y")
            elif name == "longitude":
                attrs.setdefault("axis", "X")
            elif _is_time_units(attrs.get("units")):
                attrs.setdefault("axis", "T")
            else:
                _apply_vertical_attrs(name, attrs)

        return dims, arr, _sanitize_nc_attrs(attrs)

    encoded_coords = {
        k: _encode_coord(k, dims, vals, attrs, is_dim_coord=True)
        for k, (dims, vals, attrs) in coord_vars.items()
    }
    polished_coord_vars = {}
    polished_aux_vars = {
        k: _encode_coord(k, dims, vals, attrs, is_dim_coord=False)
        for k, (dims, vals, attrs) in aux_vars.items()
    }
    for k, (dims, arr, attrs) in encoded_coords.items():
        if arr.dtype.kind in _NUMERIC_DTYPE_KINDS:
            polished_coord_vars[k] = (dims, arr, attrs)
        else:
            attrs.pop("axis", None)
            polished_aux_vars[k] = (dims, arr, attrs)

    # Strip typeOfLevel / levtype from data_vars and backfill
    # level_type onto vertical coords
    cleaned_data_vars = {}
    for vname, (vdims, data, attrs) in data_vars.items():
        attrs = _sanitize_nc_attrs(attrs)
        tol = attrs.pop("typeOfLevel", None)
        lt_mars = attrs.pop("levtype", None)
        if tol is None:
            tol = lt_mars

        vdim_set = set(vdims)
        if var_orig_dims:
            v_orig = var_orig_dims.get(vname, vdim_set)
        else:
            v_orig = vdim_set

        if tol is not None:
            for cname in _VERTICAL_NAMES:
                if cname in polished_coord_vars and cname in vdim_set:
                    cattrs = polished_coord_vars[cname][2]
                    cattrs.setdefault("level_type", str(tol))
                    _apply_vertical_attrs(cname, cattrs)
                elif cname in polished_aux_vars:
                    if coord_orig_dims:
                        in_v_orig = coord_orig_dims.get(cname, set()).issubset(
                            v_orig
                        )
                    else:
                        in_v_orig = True

                    if in_v_orig:
                        polished_aux_vars[cname][2].setdefault(
                            "level_type", str(tol)
                        )

        cleaned_data_vars[vname] = (vdims, data, attrs)

    # Order atmospheric pressure dimensions in decreasing values
    # (e.g. 1000 -> 300 hPa)
    if decreasing_atmos_pressure:
        dims_to_flip: set[str] = set()
        for cname, (cdims, arr, cattrs) in list(polished_coord_vars.items()):
            if (
                cdims == (cname,)
                and arr.ndim == 1
                and arr.size >= 2
                and arr.dtype.kind in _NUMERIC_DTYPE_KINDS
                and _is_pressure_coord(cname, cattrs)
                and np.all(np.diff(arr) > 0)
            ):
                polished_coord_vars[cname] = (cdims, arr[::-1], cattrs)
                dims_to_flip.add(cname)

        if dims_to_flip:
            for aname, (adims, aarr, aattrs) in list(
                polished_aux_vars.items()
            ):
                flipped_arr = aarr
                for ax, d in enumerate(adims):
                    if d in dims_to_flip:
                        flipped_arr = np.flip(flipped_arr, axis=ax)

                polished_aux_vars[aname] = (adims, flipped_arr, aattrs)

            for vname, (vdims, vdata, vattrs) in list(
                cleaned_data_vars.items()
            ):
                flipped_data = vdata
                for ax, d in enumerate(vdims):
                    if d in dims_to_flip:
                        if isinstance(flipped_data, LazyTensorArray):
                            flipped_data = flipped_data.with_flipped_axis(ax)
                        else:
                            flipped_data = np.flip(flipped_data, axis=ax)

                cleaned_data_vars[vname] = (vdims, flipped_data, vattrs)

    polished_dimensions, bnds_vars, bnds_names = _build_cell_bounds(
        dimensions, polished_coord_vars, polished_aux_vars, interval_hints
    )
    polished_aux_vars.update(bnds_vars)

    polished_vars = {}
    for vname, (vdims, data, attrs) in cleaned_data_vars.items():
        vdim_set = set(vdims)
        if var_orig_dims:
            v_orig = var_orig_dims.get(vname, vdim_set)
        else:
            v_orig = vdim_set

        linked = [
            aname
            for aname, (adims, _, _) in polished_aux_vars.items()
            if aname not in bnds_names
            and set(adims).issubset(vdim_set)
            and (
                coord_orig_dims.get(aname, set()).issubset(v_orig)
                if coord_orig_dims
                else True
            )
        ]
        if linked:
            existing = attrs.get("coordinates", "").split()
            merged = list(dict.fromkeys([*existing, *linked]))
            attrs["coordinates"] = " ".join(merged)

        if raw_cell_methods:
            cm_hints = raw_cell_methods.get(vname)
        else:
            cm_hints = None

        cm = _format_cell_methods(vdims, cm_hints, polished_coord_vars)
        if cm:
            attrs.setdefault("cell_methods", cm)

        polished_vars[vname] = (vdims, data, attrs)

    global_attrs = _sanitize_nc_attrs(global_attrs)
    if any(
        vname in polished_coord_vars or vname in polished_aux_vars
        for vname in ("level", "levelist")
    ):
        global_attrs.pop("levelist", None)
        global_attrs.pop("level", None)

    if any(
        "level_type" in attrs
        for _, _, attrs in (
            *polished_coord_vars.values(),
            *polished_aux_vars.values(),
        )
    ):
        global_attrs.pop("levtype", None)
        global_attrs.pop("level_type", None)

    global_attrs.setdefault("Conventions", CF_CONVENTIONS)

    return FlatNetCDFSpec(
        dimensions=polished_dimensions,
        coord_vars=polished_coord_vars,
        data_vars=polished_vars,
        aux_vars=polished_aux_vars,
        global_attrs=global_attrs,
    )


def grib_to_flat_netcdf_spec(
    ds,
    profile="auto",
    calendar=CF_CALENDAR,
    decreasing_atmos_pressure=True,
    squeeze=True,
    source_type=None,
    source_kwargs=None,
    profile_kwargs=None,
):
    """Convert GRIB data into a CF compliant `FlatNetCDFSpec`.

    Parses GRIB messages via `earthkit.data`'s ``xr_engine`` builder
    without requiring ``xarray`` or ``dask``, wrapping lazy variable
    arrays in `LazyTensorArray` instances for on-demand slicing and
    applying a CF metadata and coordinate normalization pass.

    :Parameters:

        ds: `str`, os.PathLike, bytes, file-like, FieldList, Source, or sequence of Field
            Input GRIB source. Accepts local paths, ``file://`` URIs,
            ``http(s)://`` or ``ftp://`` URLs, ``fsspec`` URIs or
            ``OpenFile`` objects, raw bytes, open binary streams, an
            existing `earthkit.data.FieldList`/``Source``, or a
            sequence of ``Field`` objects.

        profile: `str` or Profile, default "auto"
             The `earthkit.data` ``xr_engine`` profile name or object
             (e.g. ``'auto'``, ``'grib'``, ``'mars'``, or
             ``'earthkit'``). When ``'auto'`` (the default), selects
             ``'mars'`` if MARS namespace metadata is present on the
             first field and ``'grib'`` otherwise.

        calendar: `str`, default "proleptic_gregorian"
            CF calendar name used when encoding datetime coordinates
            into numeric ``"<unit> since YYYY-01-01"``
            values. Non-default calendars use ``cftime`` for
            conversion.

        decreasing_atmos_pressure: `bool`, default True
            If True, 1-d atmospheric pressure coordinates that are
            strictly increasing (e.g. ``300 -> 1000`` hPa) are
            reversed to decreasing order (surface-to-top, ``1000 ->
            300`` hPa), and all variables sharing that dimension are
            flipped lazily along the corresponding axis.

        squeeze: `bool`, optional
            Whether to drop size 1, non-horizontal dimensions (such as
            ``time``, ``step``, ``level``, or ``number``) and demote
            their coordinates to scalar entries in
            ``aux_vars``. Horizontal grid dimensions (e.g.
            ``latitude``, ``longitude``, ``x``, ``y``, ``values``) are
            never squeezed, even when of size 1. If ``None``
            (default), resolves to False for the ``"earthkit"``
            profile and True otherwise (e.g.  for ``"grib"`` and
            ``"mars"``). Singleton ``level_type`` / ``levtype``
            dimensions are always squeezed and folded onto the
            vertical coordinate as a ``level_type`` attribute.

        source_type: `str`, optional
            Explicit `earthkit.data.from_source` source type (such as
            ``'file'`` or ``'stream'``) to bypass automatic input type
            detection.

        source_kwargs: `dict`, optional
            Additional keyword arguments forwarded to
            `earthkit.data.from_source`.

        profile_kwargs
            Additional keyword arguments forwarded to
            `earthkit.data.xr_engine.profile.Profile` via
            `SingleDatasetBuilder` (for example ``lazy_load=False``,
            ``allow_holes=True``, ``rename_dims=...``, or
            ``add_earthkit_attrs=True``, which defaults to False
            here). Do not pass ``'squeeze'`` here; use the dedicated
            *squeeze* parameter instead.
    
    :Returns:

        `FlatNetCDFSpec`
            Immutable specification containing:

            * ``dimensions``: mapping of dimension names to sizes.
            * ``coord_vars``: 1-d numeric dimension coordinates as
              ``(dims, ndarray, attrs)``.
            * ``data_vars``: data variables as ``(dims, array_like,
              attrs)``, where ``array_like`` is a `LazyTensorArray`
              when ``lazy_load=True`` or a concrete `numpy.ndarray`
              otherwise.
            * ``aux_vars``: scalar, multi-dimensional, string, and
              ``*_bnds`` auxiliary coordinates as ``(dims, ndarray,
              attrs)``.
            * ``global_attrs``: sanitized netCDF global attributes.

    :Notes:

    The CF normalization pass performs the following transformations:

    * Encodes ``datetime64`` / ``datetime`` values as CF numeric time
      offsets (auto-selecting ``days``, ``hours``, or ``seconds``) and
      ``timedelta`` values as integer ``hours`` or ``seconds``.

    * Populates CF ``axis`` (``X``, ``Y``, ``Z``, ``T``) and vertical
      ``positive`` (``"down"`` for pressure/depth, ``"up"`` for
      height) attributes, and folds ``typeOfLevel`` / ``levtype`` onto
      vertical coordinates as ``level_type``.

    * Generates CF ``*_bnds`` boundary variables and ``bounds`` links
      for monotonic 1-d spatial axes, GRIB ``startStep``/``endStep``
      intervals, and multi-layer vertical levels
      (``topLevel``/``bottomLevel``).

    * Infers ``cell_methods`` from GRIB ``stepType``,
      ``typeOfStatisticalProcessing``, and ``derivedForecast``
      metadata, and links applicable auxiliary coordinates via each
      variable's ``coordinates`` attribute.

    :Examples:

    >>> with grib_to_flat_netcdf_spec("tuv_pl.grib", squeeze=True) as spec:
    ...     spec.dimensions
    ...     t_dims, t_lazy, t_attrs = spec.data_vars["t"]
    ...     first_slice = t_lazy[0, 0]

    """
    ds = _as_fieldlist(
        ds, source_type=source_type, source_kwargs=source_kwargs
    )

    # Copy so that the caller's dictionary is not modified
    profile_kwargs = dict(profile_kwargs or {})
    profile_kwargs.setdefault("add_earthkit_attrs", False)
    profile_kwargs["squeeze"] = squeeze

    lock = _get_source_lock(ds)
    with lock:
        builder_cls = _get_dataset_builder_cls()
        builder = builder_cls(
            ds, profile=profile, backend_kwargs=profile_kwargs, lock=lock
        )
        result = builder.build()
        del builder

    return _apply_cf_polish(
        *result,
        calendar=calendar,
        decreasing_atmos_pressure=decreasing_atmos_pressure,
    )
