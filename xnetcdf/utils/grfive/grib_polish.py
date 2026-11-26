"""CF polish pass: attributes, cell methods and cell bounds."""

from __future__ import annotations

import datetime
import numpy as np

from .grib_constants import (CF_CALENDAR, CF_CONVENTIONS, _HORIZONTAL_1D_BOUNDS_NAMES, _LATITUDE_LIKE_NAMES, _LEVEL_TYPE_NAMES, _NUMERIC_DTYPE_KINDS, _PRESSURE_UNITS, _VERTICAL_NAMES)
from .grib_time import (_cf_unit_scale_seconds, _encode_cf_datetime, _encode_cf_timedelta, _is_cftime_datetime, _is_datetime_values, _is_timedelta_values)
from .grib_utils import (FlatNetCDFSpec, _is_time_units, _unique_name)


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
            Coordinate name (``"latitude"`` and ``"grid_latitude"``
            bounds are clipped to ``[-90, 90]``).

        arr: `numpy.ndarray`
            1-d numeric coordinate array of size at least 2.

    :Returns:

        `numpy.ndarray` or `None`
            ``(N, 2)`` array of lower and upper cell edges, or
            ``None`` if ``arr`` is not 1-d, has fewer than 2 finite
            points, is non-monotonic, or is not equally spaced.

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

    # Mid-points are only valid cell edges for equally spaced points
    if not np.allclose(diffs, diffs[0], rtol=1e-3, atol=0):
        return None

    edges = np.empty(vals.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
    edges[0] = vals[0] - 0.5 * diffs[0]
    edges[-1] = vals[-1] + 0.5 * diffs[-1]

    if name in _LATITUDE_LIKE_NAMES:
        edges = np.clip(edges, -90.0, 90.0)

    if arr.dtype.kind == "f":
        out_dtype = arr.dtype
    else:
        out_dtype = np.float64

    return np.column_stack([edges[:-1], edges[1:]]).astype(
        out_dtype, copy=False
    )


def _build_cell_bounds(
    dimensions,
    coord_vars,
    aux_vars,
    interval_hints,
    horizontal_bounds=True,
):
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

        horizontal_bounds: `bool`, default True
            Whether to create bounds for the 1-d horizontal
            coordinates. Should only be True if the mid-points
            between grid points are valid cell edges.

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
        if not horizontal_bounds or cname not in _HORIZONTAL_1D_BOUNDS_NAMES:
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
    horizontal_bounds=True,
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

        horizontal_bounds: `bool`, default True
            Whether to create bounds for 1-d horizontal coordinates,
            as determined by `_MIDPOINT_BOUNDS_GRID_TYPES`.

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
                        if hasattr(flipped_data, "with_flipped_axis"):
                            flipped_data = flipped_data.with_flipped_axis(ax)
                        else:
                            flipped_data = np.flip(flipped_data, axis=ax)

                cleaned_data_vars[vname] = (vdims, flipped_data, vattrs)

    # A scalar level_type that could not be folded onto a vertical
    # coordinate (because there isn't one, as for surface fields)
    # becomes an attribute of the data variables.
    for lt_key in _LEVEL_TYPE_NAMES:
        lt_var = polished_aux_vars.get(lt_key)
        if lt_var is None or lt_var[0] != ():
            continue

        lt_val = str(np.asarray(lt_var[1]).item())
        del polished_aux_vars[lt_key]
        for vname, (vdims, data, attrs) in cleaned_data_vars.items():
            attrs.setdefault("level_type", lt_val)

    polished_dimensions, bnds_vars, bnds_names = _build_cell_bounds(
        dimensions,
        polished_coord_vars,
        polished_aux_vars,
        interval_hints,
        horizontal_bounds=horizontal_bounds,
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
            *polished_vars.values(),
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
