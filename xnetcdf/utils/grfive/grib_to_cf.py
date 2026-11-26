"""Convert GRIB datasets into CF compliant netCDF specifications.

This module wraps `earthkit.data`'s ``xr_engine`` builder to extract
GRIB metadata, coordinates, and deferred `numpy` array views
(`LazyTensorArray`) into a `FlatNetCDFSpec` without importing
``xarray`` or ``dask``.

"""

from __future__ import annotations

import io
import math
import os
import threading
import uuid
import weakref
from functools import lru_cache
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np

from .grib_constants import (  # noqa: F401
    CF_CALENDAR,
    CF_CONVENTIONS,
    NC_FILL_DOUBLE,
    _GRIB2_DERIVED_FORECAST_TO_CF_METHOD,
    _GRIB2_STAT_PROC_TO_CF_METHOD,
    _GRIB_STEP_TYPE_TO_CF_METHOD,
    _GRIB_STEP_UNITS_TO_SECONDS,
    _LEVEL_TYPE_NAMES,
    _MIDPOINT_BOUNDS_GRID_TYPES,
    _NUMERIC_DTYPE_KINDS,
    _VERTICAL_NAMES,
)
from .grib_grid_mapping import (  # noqa: F401
    _apply_grid_mapping,
    _projected_grid_definition,
)
from .grib_polish import (  # noqa: F401
    _apply_cf_polish,
    _build_cell_bounds,
    _compute_1d_spatial_bounds,
)
from .grib_time import _is_datetime_values, _is_timedelta_values
from .grib_utils import (  # noqa: F401
    FlatNetCDFSpec,
    _default_fill_value,
    _field_meta,
    _grid_type,
    _is_time_units,
    _unique_name,
)


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
        fill_value=None,
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
        self.fill_value = fill_value
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
        n_field = min(self.ndim, len(field_shape))
        n_outer = self.ndim - n_field
        return (1,) * n_outer + field_shape[len(field_shape) - n_field :]

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
            fill_value=self.fill_value,
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
            fill_value=self.fill_value,
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
            if self.fill_value is not None and arr.dtype.kind == "f":
                # Missing GRIB points arrive as NaN; netCDF expects the
                # `_FillValue` instead
                arr = np.where(np.isnan(arr), self.fill_value, arr)

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
            fill_value = _default_fill_value(self.dtype)
            if not self.profile.lazy_load:
                arr = np.asarray(
                    tensor.to_array(
                        array_namespace=self.array_namespace, dtype=self.dtype
                    )
                )
                if fill_value is not None:
                    arr = np.where(np.isnan(arr), fill_value, arr)

                return arr

            return LazyTensorArray(
                tensor,
                var_dims,
                tensor.full_shape,
                self.array_namespace,
                self.dtype,
                name,
                lock=self._lock,
                fill_value=fill_value,
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
                fill_value = _default_fill_value(data.dtype)
                if fill_value is not None:
                    attrs.setdefault("_FillValue", fill_value)

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

    spec = _apply_cf_polish(
        *result,
        calendar=calendar,
        decreasing_atmos_pressure=decreasing_atmos_pressure,
        horizontal_bounds=(
            _grid_type(ds[0]) in _MIDPOINT_BOUNDS_GRID_TYPES
        ),
    )
    return _apply_grid_mapping(spec, ds)
