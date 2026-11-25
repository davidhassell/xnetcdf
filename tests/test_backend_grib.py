import datetime
import math
import earthkit.data as ekd
import numpy as np
import pytest

from xnetcdf.utils.grfive.grib_to_cf import LazyTensorArray, _field_meta, grib_to_flat_netcdf_spec

_SPATIAL_DIMS = {"latitude", "longitude", "x", "y", "values"}
_UNIT_SECONDS = {"days": 86400, "hours": 3600, "seconds": 1}

#"/home/david/Downloads/tuv_pl.grib"


def _field_index_for_dim(field, dim, coord_vals, coord_attrs):
    """Find the index in *coord_vals* corresponding to *field* along *dim*."""
    if dim in ("time", "valid_time"):
        unit_str, _, epoch_str = coord_attrs["units"].partition(" since ")
        origin = np.datetime64(epoch_str.strip(), "s")
        scale = _UNIT_SECONDS[unit_str.strip().lower()]
        coord_dt = origin + np.round(coord_vals * scale).astype(
            "timedelta64[s]"
        )

        dt_key = "valid_time" if dim == "valid_time" else "base_time"
        dt = field.datetime()[dt_key]
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)

        matches = np.where(coord_dt == np.datetime64(dt, "s"))[0]
    elif dim == "step":
        step_val = _field_meta(field, "endStep")
        if step_val is None:
            step_val = _field_meta(field, "step")
        scale = 3600.0 if coord_attrs.get("units") == "seconds" else 1.0
        matches = np.where(np.isclose(coord_vals, float(step_val) * scale))[0]
    else:
        val = _field_meta(field, dim)
        if val is None and dim in (
            "level",
            "isobaricInhPa",
            "isobaricInPa",
            "levelist",
        ):
            val = _field_meta(field, "level")
        matches = np.where(np.isclose(coord_vals, float(val)))[0]

    assert len(matches) == 1, (
        f"Expected unique match for {dim} on field {field}, got {matches}"
    )
    return int(matches[0])


@pytest.mark.parametrize("decreasing_atmos_pressure", [True, False])
def test_lazy_tensor_array_t_matches_individual_fields(
    decreasing_atmos_pressure
):
    """Verify LazyTensorArray.__getitem__ matches the individual 't' Fields."""
    ds = ekd.from_source("file", "/home/david/Downloads/tuv_pl.grib").to_fieldlist()
    t_fields = [
        f
        for f in ds
        if any(
            str(_field_meta(f, k) or "").lower() == "t"
            for k in ("param", "shortName", "cfVarName")
        )
    ]
    assert len(t_fields) == 6

    with grib_to_flat_netcdf_spec(
        ds, decreasing_atmos_pressure=decreasing_atmos_pressure
    ) as spec:
        t_dims, t_lazy, _ = spec.data_vars["t"]
        assert isinstance(t_lazy, LazyTensorArray)

        non_spatial_dims = [
            d
            for d in t_dims
            if d not in _SPATIAL_DIMS
            and spec.coord_vars.get(d, ((), None, {}))[2].get("axis")
            not in ("X", "Y")
        ]
        assert math.prod(t_lazy.shape[: len(non_spatial_dims)]) == len(
            t_fields
        )

        expected_full = np.empty(t_lazy.shape, dtype=t_lazy.dtype)
        seen_indices = set()

        for fld in t_fields:
            idx = tuple(
                _field_index_for_dim(
                    fld,
                    dim,
                    spec.coord_vars[dim][1],
                    spec.coord_vars[dim][2],
                )
                for dim in non_spatial_dims
            )
            seen_indices.add(idx)

            expected_field = fld.to_numpy()
            expected_full[idx] = expected_field

            # Single-field integer indexing via
            # LazyTensorArray.__getitem__
            actual_field = t_lazy[idx]
            assert actual_field.shape == expected_field.shape
            np.testing.assert_allclose(actual_field, expected_field)

            # Sub-slice indexing combining integer outer indices and
            # spatial slices
            if expected_field.ndim == 2:
                sub_key = (*idx, slice(1, 4), slice(2, 6))
                np.testing.assert_allclose(
                    t_lazy[sub_key], expected_field[1:4, 2:6]
                )
            elif expected_field.ndim == 1:
                sub_key = (*idx, slice(1, 5))
                np.testing.assert_allclose(
                    t_lazy[sub_key], expected_field[1:5]
                )

        assert len(seen_indices) == len(t_fields)

        # Full-array, __array__, and negative-step reversed slicing
        np.testing.assert_allclose(t_lazy[:], expected_full)
        np.testing.assert_allclose(np.asarray(t_lazy), expected_full)
        np.testing.assert_allclose(t_lazy[::-1], expected_full[::-1])
        np.testing.assert_allclose(t_lazy[..., ::-1], expected_full[..., ::-1])
