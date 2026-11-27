import datetime
import math
import earthkit.data as ekd
import numpy as np
import pytest

from xnetcdf.utils.grfive.grib_to_cf import (
    LazyTensorArray,
    _field_meta,
    _projected_grid_definition,
    grib_to_flat_netcdf_spec,
)

_SPATIAL_DIMS = {"latitude", "longitude", "x", "y", "values"}
_UNIT_SECONDS = {"days": 86400, "hours": 3600, "seconds": 1}


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
    data_dir, decreasing_atmos_pressure
):
    """Verify LazyTensorArray.__getitem__ matches the individual 't' Fields."""
    dataset = data_dir / "tuv_pl.grib"
    ds = ekd.from_source("file", str(dataset)).to_fieldlist()
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


def test_lazy_tensor_array_shared_file_lock_and_threads(data_dir):
    """Arrays from one source share a lock; threaded reads are correct."""
    from concurrent.futures import ThreadPoolExecutor

    dataset = data_dir / "tuv_pl.grib"
    ds = ekd.from_source("file", str(dataset)).to_fieldlist()
    with grib_to_flat_netcdf_spec(ds) as spec:
        arrays = {
            k: v[1]
            for k, v in spec.data_vars.items()
            if isinstance(v[1], LazyTensorArray)
        }
        assert len(arrays) > 1
        assert len({id(a.lock) for a in arrays.values()}) == 1

        # A second spec from the same source reuses the same lock
        with grib_to_flat_netcdf_spec(ds) as spec2:
            other = next(
                v[1]
                for v in spec2.data_vars.values()
                if isinstance(v[1], LazyTensorArray)
            )
            assert other.lock is next(iter(arrays.values())).lock

        expected = {k: np.asarray(a) for k, a in arrays.items()}
        jobs = [
            (k, i) for k, a in arrays.items() for i in range(a.shape[0])
        ] * 8
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda j: arrays[j[0]][j[1]], jobs))

        for (k, i), r in zip(jobs, results):
            np.testing.assert_allclose(r, expected[k][i])


def test_lazy_tensor_array_lock_pickles():
    import pickle

    from xnetcdf.utils.grfive.grib_to_cf import _PicklableLock

    lock = _PicklableLock()
    copy = pickle.loads(pickle.dumps(lock))
    assert copy._lock is lock._lock


def _first_lazy(data_dir):
    dataset = data_dir / "tuv_pl.grib"
    ds = ekd.from_source("file", str(dataset)).to_fieldlist()
    spec = grib_to_flat_netcdf_spec(ds)
    arr = next(
        v[1] for v in spec.data_vars.values() if isinstance(v[1], LazyTensorArray)
    )
    return spec, arr, np.asarray(arr)


def test_lazy_tensor_array_indexing_edge_cases(data_dir):
    spec, a, full = _first_lazy(data_dir)
    with spec:
        n0 = a.shape[0]

        # Ellipsis alongside an array index (used to raise ValueError)
        np.testing.assert_allclose(a[[0, -1], ...], full[[0, -1], ...])
        np.testing.assert_allclose(a[np.array([0, -1]), ..., 1], full[[0, -1], ..., 1])

        # Negative integer arrays, including on flipped axes
        np.testing.assert_allclose(a[np.array([-1, 0])], full[[-1, 0]])

        # Boolean mask
        mask = np.zeros(n0, dtype=bool)
        mask[[0, n0 - 1]] = True
        np.testing.assert_allclose(a[mask], full[mask])

        # Numpy integer scalar
        np.testing.assert_allclose(a[np.int64(-1)], full[-1])

        # Errors
        for bad in (n0, -n0 - 1):
            with pytest.raises(IndexError):
                a[bad]

        with pytest.raises(IndexError):
            a[[n0]]

        with pytest.raises(IndexError):
            a[np.zeros(n0 + 1, dtype=bool)]

        with pytest.raises(IndexError):
            a[None]

        with pytest.raises(IndexError):
            a[..., ...]

        with pytest.raises(IndexError):
            a[(0,) * (a.ndim + 1)]

        with pytest.raises(IndexError):
            a[1.5]


def test_lazy_tensor_array_indexing_flipped_axis(data_dir):
    dataset = data_dir / "tuv_pl.grib"
    ds = ekd.from_source("file", str(dataset)).to_fieldlist()
    with grib_to_flat_netcdf_spec(ds, decreasing_atmos_pressure=True) as s2:
        for name, (dims, a2, _) in s2.data_vars.items():
            if not isinstance(a2, LazyTensorArray) or not a2.flip_axes:
                continue

            full = np.asarray(a2)
            # flip_axes are raw axes, so allow for squeezed axes
            raw = a2.flip_axes[0]
            ax = raw - sum(i < raw for i in a2.squeeze_axes)
            n = a2.shape[ax]
            sel = [slice(None)] * a2.ndim
            sel[ax] = np.array([-1, 0])
            np.testing.assert_allclose(a2[tuple(sel)], full[tuple(sel)])
            sel[ax] = -1
            np.testing.assert_allclose(a2[tuple(sel)], full[tuple(sel)])
            with pytest.raises(IndexError):
                sel[ax] = n
                a2[tuple(sel)]
            return

    pytest.skip("no flipped lazy variable in test data")


def test_lazy_tensor_array_chunks_are_message_aligned(data_dir):
    """Each chunk is exactly one whole GRIB field."""
    dataset = data_dir / "tuv_pl.grib"
    ds = ekd.from_source("file", str(dataset)).to_fieldlist()
    for dap in (True, False):
        with grib_to_flat_netcdf_spec(
            ds, decreasing_atmos_pressure=dap
        ) as spec:
            for dims, a, _ in spec.data_vars.values():
                if not isinstance(a, LazyTensorArray):
                    continue

                assert len(a.chunks) == a.ndim
                assert a.chunks[-2:] == a.shape[-2:]
                assert all(c == 1 for c in a.chunks[:-2])
                assert a.chunks[-2:] == tuple(a.tensor.field_shape)

    # View of a larger array with a squeezed axis keeps alignment
    with grib_to_flat_netcdf_spec(ds) as spec:
        a = next(
            v[1]
            for v in spec.data_vars.values()
            if isinstance(v[1], LazyTensorArray)
        )
        v = a.with_squeezed_axis(0) if a.shape[0] == 1 else a
        assert v.chunks[-2:] == tuple(a.tensor.field_shape)


@pytest.mark.parametrize("squeeze", [True, False])
def test_reduced_gaussian_grid_1d_horizontal(data_dir, squeeze):
    """A reduced Gaussian grid has a single 1-d horizontal axis."""
    dataset = data_dir / "lsm.N128.grib"
    ds = ekd.from_source("file", str(dataset)).to_fieldlist()
    expected = ds[0].to_numpy().reshape(-1)
    npts = expected.size

    with grib_to_flat_netcdf_spec(ds, squeeze=squeeze) as spec:
        dims, lazy, _ = spec.data_vars["lsm"]
        assert isinstance(lazy, LazyTensorArray)
        assert dims[-1] == "values"
        assert lazy.chunks == (1,) * (lazy.ndim - 1) + (npts,)

        if squeeze:
            assert dims == ("values",)
        else:
            # Size-1 outer axes are retained
            assert dims == ("forecast_reference_time", "step", "values")
            assert lazy.shape == (1, 1, npts)

        full = np.asarray(lazy)
        np.testing.assert_allclose(full.reshape(-1), expected)
        np.testing.assert_allclose(lazy[..., 10:20], full[..., 10:20])
        np.testing.assert_allclose(lazy[..., ::-1], full[..., ::-1])
        np.testing.assert_allclose(lazy[..., [-1, 0]], full[..., [-1, 0]])
        assert spec.aux_vars["latitude"][0] == ("values",)
        assert spec.aux_vars["longitude"][0] == ("values",)


def test_ensemble_example_matches_fields(data_dir):
    """Every message of the ensemble example appears once in the data."""
    dataset = data_dir / "example.grib"
    ds = ekd.from_source("file", str(dataset)).to_fieldlist()
    with grib_to_flat_netcdf_spec(ds) as spec:
        total = 0
        for name, (dims, lazy, _) in spec.data_vars.items():
            assert isinstance(lazy, LazyTensorArray)
            assert dims[0] == "member"
            full = np.asarray(lazy)
            assert not np.isnan(full).any()
            total += math.prod(lazy.shape[:-2])

        assert total == len(ds)


def test_rotated_ll_grid_mapping(data_dir):
    """Rotated lat/lon grids get CF rotated coordinates and a mapping."""
    pyproj = pytest.importorskip("pyproj")

    dataset = data_dir / "rotated_wind_20x20.grib"
    ds = ekd.from_source("file", str(dataset)).to_fieldlist()
    with grib_to_flat_netcdf_spec(ds) as spec:
        assert spec.dimensions == {
            "grid_latitude": 9,
            "grid_longitude": 18,
            "bnds": 2,
        }

        rlat = spec.coord_vars["grid_latitude"]
        rlon = spec.coord_vars["grid_longitude"]
        assert rlat[2]["standard_name"] == "grid_latitude"
        assert rlon[2]["standard_name"] == "grid_longitude"
        np.testing.assert_allclose(rlat[1], 80 - 20 * np.arange(9))
        np.testing.assert_allclose(rlon[1], 20 * np.arange(18))

        # Cell bounds are the mid-points between the grid points
        assert rlat[2]["bounds"] == "grid_latitude_bnds"
        assert rlon[2]["bounds"] == "grid_longitude_bnds"
        lat_bnds = spec.aux_vars["grid_latitude_bnds"]
        assert lat_bnds[0] == ("grid_latitude", "bnds")
        np.testing.assert_allclose(
            lat_bnds[1], np.column_stack([90 - 20 * np.arange(9), 70 - 20 * np.arange(9)])
        )
        np.testing.assert_allclose(
            spec.aux_vars["grid_longitude_bnds"][1],
            np.column_stack([20 * np.arange(18) - 10, 20 * np.arange(18) + 10]),
        )
        assert spec.dimensions["bnds"] == 2

        gm_dims, _, gm = spec.aux_vars["rotated_latitude_longitude"]
        assert gm_dims == ()
        assert gm == {
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 20.0,
            "grid_north_pole_longitude": 140.0,
            "north_pole_grid_longitude": 0.0,
        }

        # True lat/lon are not in the GRIB, so are not created
        assert "latitude" not in spec.aux_vars
        assert "longitude" not in spec.aux_vars

        for dims, lazy, attrs in spec.data_vars.values():
            assert dims == ("grid_latitude", "grid_longitude")
            assert attrs["grid_mapping"] == "rotated_latitude_longitude"
            assert "latitude" not in attrs["coordinates"].split()
            assert "longitude" not in attrs["coordinates"].split()
            assert lazy.chunks == lazy.shape

        # The mapping reproduces the GRIB's true lat/lon, apart from
        # the longitude at the geographic poles (which is arbitrary)
        crs = pyproj.CRS.from_cf(gm)
        x, y = np.meshgrid(rlon[1], rlat[1])
        tlon, tlat = pyproj.Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True
        ).transform(x, y)
        np.testing.assert_allclose(tlat, ds[0].geography.latitudes())
        dlon = (tlon - ds[0].geography.longitudes() + 180) % 360 - 180
        assert (np.abs(dlon) > 1e-6).sum() <= 2
        assert np.all(np.abs(ds[0].geography.latitudes()[np.abs(dlon) > 1e-6]) == 90)


@pytest.mark.parametrize(
    "name",
    [
        "lambert_12x10.grib1",
        "lambert_azimuthal_equal_area_12x10.grib2",
        "polar_stereographic_12x10.grib1",
    ],
)
def test_projected_grids_read(data_dir, name):
    """Projected grids give lazy (y, x) data with a grid mapping."""
    dataset = data_dir / name
    ds = ekd.from_source("file", str(dataset)).to_fieldlist()
    with grib_to_flat_netcdf_spec(ds) as spec:
        assert spec.dimensions["y"] == 10
        assert spec.dimensions["x"] == 12
        (dims, lazy, attrs), = spec.data_vars.values()
        assert dims == ("y", "x")
        assert "grid_mapping" in attrs
        expected = ds[0].to_numpy()
        expected = np.where(
            np.isnan(expected), attrs["_FillValue"], expected
        )
        np.testing.assert_allclose(np.asarray(lazy), expected)


def test_surface_level_type_is_a_variable_attribute(data_dir):
    """Without a vertical coordinate, level_type goes on the variables."""
    dataset = data_dir / "rotated_wind_20x20.grib"
    ds = ekd.from_source("file", str(dataset)).to_fieldlist()
    with grib_to_flat_netcdf_spec(ds) as spec:
        assert "level_type" not in spec.aux_vars
        assert "levtype" not in spec.global_attrs
        assert "level_type" not in spec.global_attrs
        for _, _, attrs in spec.data_vars.values():
            assert attrs["level_type"] == "sfc"

    # With a vertical coordinate it stays on that coordinate
    dataset = data_dir / "tuv_pl.grib"
    ds = ekd.from_source("file", str(dataset)).to_fieldlist()
    with grib_to_flat_netcdf_spec(ds) as spec:
        assert "level_type" in spec.coord_vars["level"][2]
        for _, _, attrs in spec.data_vars.values():
            assert "level_type" not in attrs


def test_compute_1d_spatial_bounds():
    from xnetcdf.utils.grfive.grib_to_cf import _compute_1d_spatial_bounds as f

    np.testing.assert_allclose(
        f("longitude", np.array([0.0, 10.0, 20.0])),
        [[-5, 5], [5, 15], [15, 25]],
    )
    # Decreasing
    np.testing.assert_allclose(
        f("x", np.array([20.0, 10.0, 0.0])), [[25, 15], [15, 5], [5, -5]]
    )
    # Latitude-like bounds are clipped to the poles
    for name in ("latitude", "grid_latitude"):
        np.testing.assert_allclose(
            f(name, np.array([-90.0, 0.0, 90.0])),
            [[-90, -45], [-45, 45], [45, 90]],
        )

    # Not monotonic (such as longitudes that cross the wrap), too
    # short, or not numeric, so no bounds
    assert f("longitude", np.array([350.0, 0.0, 10.0])) is None
    assert f("longitude", np.array([0.0])) is None
    assert f("longitude", np.array(["a", "b"])) is None
    assert f("longitude", np.array([0.0, np.nan, 2.0])) is None


def test_horizontal_bounds_only_where_midpoints_are_valid(data_dir):
    from xnetcdf.utils.grfive.grib_to_cf import (
        _build_cell_bounds,
        _compute_1d_spatial_bounds as f,
    )

    # Unequally spaced points (e.g. Gaussian or Mercator latitudes)
    assert f("latitude", np.array([0.0, 1.0, 3.0, 6.0])) is None

    coords = {
        "longitude": (("longitude",), np.array([0.0, 10.0, 20.0]), {}),
    }
    dims = {"longitude": 3}
    _, bnds, _ = _build_cell_bounds(dims, coords, {}, None)
    assert "longitude_bnds" in bnds

    coords = {
        "longitude": (("longitude",), np.array([0.0, 10.0, 20.0]), {}),
    }
    _, bnds, _ = _build_cell_bounds(
        dims, coords, {}, None, horizontal_bounds=False
    )
    assert not bnds
    assert "bounds" not in coords["longitude"][2]

    # Regular lat/lon grids get bounds; reduced grids have no 1-d axes
    for name, expected in (("tuv_pl.grib", True), ("lsm.N128.grib", False)):
        ds = ekd.from_source("file", str(data_dir / name)).to_fieldlist()
        with grib_to_flat_netcdf_spec(ds) as spec:
            assert ("latitude_bnds" in spec.aux_vars) is expected


def test_healpix_grid_mapping(data_dir):
    """HEALPix gets a CF grid mapping and no computed lat/lon."""
    spec = grib_to_flat_netcdf_spec(str(data_dir / "healpix_H8.grib2"))
    assert "latitude" not in spec.aux_vars
    assert "longitude" not in spec.aux_vars
    attrs = spec.aux_vars["healpix"][2]
    assert attrs == {
        "grid_mapping_name": "healpix",
        "indexing_scheme": "ring",
        "refinement_level": 3,
    }
    assert spec.dimensions["healpix_index"] == 12 * 8**2
    idx = spec.coord_vars["healpix_index"]
    assert idx[0] == ("healpix_index",)
    assert idx[2]["standard_name"] == "healpix_index"
    np.testing.assert_array_equal(idx[1], np.arange(768))
    for _, _, a in spec.data_vars.values():
        assert a["grid_mapping"] == "healpix"
        assert "latitude" not in a["coordinates"].split()
    assert not any(n.endswith("bnds") for n in spec.aux_vars)


def test_reduced_rotated_gg_grid_mapping(data_dir):
    """Reduced rotated Gaussian: mapping only, no coords or bounds."""
    spec = grib_to_flat_netcdf_spec(
        str(data_dir / "rotated_N32_subarea.grib")
    )
    attrs = spec.aux_vars["rotated_latitude_longitude"][2]
    assert attrs["grid_north_pole_latitude"] == 22.0
    assert attrs["grid_north_pole_longitude"] == 140.0
    assert "latitude" not in spec.aux_vars
    assert "grid_latitude" not in spec.coord_vars
    assert not any(n.endswith("bnds") for n in spec.aux_vars)
    for _, _, a in spec.data_vars.values():
        assert a["grid_mapping"] == "rotated_latitude_longitude"


@pytest.mark.parametrize(
    "name, gm_name",
    [
        ("lambert_12x10.grib1", "lambert_conformal_conic"),
        (
            "lambert_azimuthal_equal_area_12x10.grib2",
            "lambert_azimuthal_equal_area",
        ),
        ("polar_stereographic_12x10.grib1", "polar_stereographic"),
        ("mercator.grib", "mercator"),
    ],
)
def test_projected_grid_mapping(data_dir, name, gm_name):
    """Projected grids get x/y coordinates with bounds and a mapping."""
    pyproj = pytest.importorskip("pyproj")

    dataset = data_dir / name
    field = ekd.from_source("file", str(dataset)).to_fieldlist()[0]
    spec = grib_to_flat_netcdf_spec(str(dataset))

    assert "latitude" not in spec.aux_vars
    assert "longitude" not in spec.aux_vars
    gm = spec.aux_vars[gm_name][2]
    assert gm["grid_mapping_name"] == gm_name

    y, x = spec.coord_vars["y"], spec.coord_vars["x"]
    assert y[2]["standard_name"] == "projection_y_coordinate"
    assert x[2]["standard_name"] == "projection_x_coordinate"
    assert spec.aux_vars[x[2]["bounds"]][1].shape == (x[1].size, 2)
    assert spec.aux_vars[y[2]["bounds"]][1].shape == (y[1].size, 2)
    assert spec.dimensions["x"] == x[1].size
    assert spec.dimensions["y"] == y[1].size

    for _, _, a in spec.data_vars.values():
        assert a["grid_mapping"] == gm_name
        assert "latitude" not in a.get("coordinates", "").split()

    # The coordinates reproduce earthkit's computed lat/lon
    _, proj = _projected_grid_definition(field)
    crs = pyproj.CRS.from_proj4(proj)
    t = pyproj.Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    X, Y = np.meshgrid(x[1], y[1])
    lon, lat = t.transform(X, Y)
    np.testing.assert_allclose(
        lat, field.geography.latitudes().reshape(lat.shape), atol=1e-6
    )
    dlon = (lon - field.geography.longitudes().reshape(lon.shape) + 180) % 360
    np.testing.assert_allclose(dlon - 180, 0, atol=1e-6)


@pytest.mark.parametrize(
    "name", ["rotated_wind_20x20.grib", "rotated_N32_subarea.grib"]
)
def test_rotated_grid_with_rotation_angle_is_unchanged(
    data_dir, tmp_path, name
):
    """Earthkit ignores angleOfRotation, so no mapping is created."""
    eccodes = pytest.importorskip("eccodes")

    out = tmp_path / "angle.grib"
    with open(data_dir / name, "rb") as f, open(out, "wb") as o:
        h = eccodes.codes_grib_new_from_file(f)
        eccodes.codes_set(h, "angleOfRotation", 30.0)
        eccodes.codes_write(h, o)
        eccodes.codes_release(h)

    spec = grib_to_flat_netcdf_spec(str(out))
    assert "rotated_latitude_longitude" not in spec.aux_vars
    assert "grid_latitude" not in spec.coord_vars
    assert "latitude" in {**spec.aux_vars, **spec.coord_vars}
    for _, _, a in spec.data_vars.values():
        assert "grid_mapping" not in a


def test_grib_cell_methods_use_existing_time_axis():
    """cell_methods never names the forecast reference time."""
    from xnetcdf.utils.grfive.grib_polish import _format_cell_methods

    hints = ("sum", None)
    assert _format_cell_methods(("forecast_reference_time",), hints) is None
    assert _format_cell_methods(("y", "x"), hints, ["step"]) == "step: sum"
    assert _format_cell_methods(("time", "x"), hints, ["step"]) == "time: sum"


def test_grib_missing_points_use_fill_value(data_dir):
    """NaN points are returned as the declared _FillValue."""
    spec = grib_to_flat_netcdf_spec(
        str(data_dir / "polar_stereographic_12x10.grib1")
    )
    _, data, attrs = spec.data_vars["tp"]
    fill = attrs["_FillValue"]
    arr = data[...]
    assert not np.isnan(arr).any()
    assert (arr == fill).sum() == 120
    assert "point" not in attrs.get("cell_methods", "")
    spec.close()


def _write_modified(tmp_path, src, name, edits_list):
    """Write copies of the first message of *src* with edited keys."""
    import eccodes

    out = tmp_path / name
    with open(src, "rb") as f:
        handle = eccodes.codes_grib_new_from_file(f)

    with open(out, "wb") as o:
        for edits in edits_list:
            clone = eccodes.codes_clone(handle)
            for key, value in edits:
                eccodes.codes_set(clone, key, value)

            eccodes.codes_write(clone, o)
            eccodes.codes_release(clone)

    eccodes.codes_release(handle)
    return out


def test_grib_step_bounds_with_zero_length_step(data_dir, tmp_path):
    """Accumulations starting at step 0 still get step bounds."""
    path = _write_modified(
        tmp_path,
        data_dir / "polar_stereographic_12x10.grib1",
        "accum.grib",
        [
            [("stepType", "accum"), ("startStep", 0), ("endStep", e)]
            for e in (0, 6, 12)
        ],
    )
    with grib_to_flat_netcdf_spec(str(path)) as spec:
        assert spec.coord_vars["step"][2]["bounds"] == "step_bnds"
        np.testing.assert_array_equal(
            spec.aux_vars["step_bnds"][1], [[0, 0], [0, 6], [0, 12]]
        )


def test_grib_hybrid_level_has_no_hybrid_standard_name(data_dir, tmp_path):
    """Model level numbers don't get the hybrid sigma standard name."""
    path = _write_modified(
        tmp_path,
        data_dir / "lambert_12x10.grib1",
        "hybrid.grib",
        [[("typeOfLevel", "hybrid"), ("level", n)] for n in (1, 50, 137)],
    )
    with grib_to_flat_netcdf_spec(str(path)) as spec:
        attrs = spec.coord_vars["level"][2]
        assert "standard_name" not in attrs
        assert attrs["positive"] == "down"


def test_grib_dropped_fields_raise(data_dir, tmp_path):
    """A profile that would silently drop fields is an error."""
    path = _write_modified(
        tmp_path,
        data_dir / "lambert_12x10.grib1",
        "soil.grib",
        [
            [
                ("indicatorOfTypeOfLevel", 112),
                ("topLevel", top),
                ("bottomLevel", bottom),
            ]
            for top, bottom in ((0, 7), (7, 28), (28, 100))
        ],
    )
    with pytest.raises(ValueError, match="silently dropped"):
        grib_to_flat_netcdf_spec(str(path))

    with grib_to_flat_netcdf_spec(str(path), profile="grib") as spec:
        assert spec.data_vars["t"][0] == ("depth", "y", "x")


def test_grib_single_height_gets_scalar_coordinate(data_dir):
    """2 metre fields get a scalar height coordinate."""
    with grib_to_flat_netcdf_spec(
        str(data_dir / "lambert_12x10.grib1")
    ) as spec:
        _, _, attrs = spec.data_vars["2t"]
        assert "height" in attrs["coordinates"].split()
        dims, value, hattrs = spec.aux_vars["height"]
        assert dims == ()
        assert value == 2
        assert hattrs["units"] == "m"
        assert hattrs["positive"] == "up"


def test_grib1_soil_depth_units_are_cm(data_dir, tmp_path):
    """GRIB1 soil depths are in cm, not the m that earthkit assumes."""
    path = _write_modified(
        tmp_path,
        data_dir / "lambert_12x10.grib1",
        "soil.grib",
        [
            [
                ("indicatorOfTypeOfLevel", 112),
                ("topLevel", top),
                ("bottomLevel", bottom),
            ]
            for top, bottom in ((0, 7), (7, 28))
        ],
    )
    with grib_to_flat_netcdf_spec(str(path), profile="grib") as spec:
        assert spec.coord_vars["depth"][2]["units"] == "cm"
        bounds = spec.aux_vars[spec.coord_vars["depth"][2]["bounds"]][1]
        np.testing.assert_array_equal(bounds, [[0, 7], [7, 28]])


def test_grib_single_altitude_gets_scalar_coordinate(data_dir, tmp_path):
    """Other single levels with a value get a scalar coordinate too."""
    path = _write_modified(
        tmp_path,
        data_dir / "example.grib",
        "asl.grib",
        [
            [
                ("typeOfFirstFixedSurface", 102),
                ("scaleFactorOfFirstFixedSurface", 0),
                ("scaledValueOfFirstFixedSurface", 500),
            ]
        ],
    )
    with grib_to_flat_netcdf_spec(str(path)) as spec:
        dims, value, attrs = spec.aux_vars["altitude"]
        assert (dims, value) == ((), 500)
        assert attrs["standard_name"] == "altitude"
        assert "altitude" in spec.data_vars["z"][2]["coordinates"].split()


def test_grib_coordinates_named_after_standard_names(data_dir, tmp_path):
    """earthkit's scalar `level` at a height is named `height`."""
    path = _write_modified(
        tmp_path,
        data_dir / "example.grib",
        "h100.grib",
        [
            [
                ("typeOfFirstFixedSurface", 103),
                ("scaleFactorOfFirstFixedSurface", 0),
                ("scaledValueOfFirstFixedSurface", 100),
            ]
        ],
    )
    with grib_to_flat_netcdf_spec(str(path)) as spec:
        assert "level" not in spec.aux_vars
        assert spec.aux_vars["height"][2]["axis"] == "Z"
        assert "height" in spec.data_vars["z"][2]["coordinates"].split()


def test_grib_valid_time_coordinate(data_dir, tmp_path):
    """`time` is reference time plus step, with matching bounds."""
    path = _write_modified(
        tmp_path,
        data_dir / "polar_stereographic_12x10.grib1",
        "accum.grib",
        [
            [("stepType", "accum"), ("startStep", 0), ("endStep", e)]
            for e in (6, 12)
        ],
    )
    with grib_to_flat_netcdf_spec(str(path)) as spec:
        dims, arr, attrs = spec.aux_vars["time"]
        frt = spec.aux_vars["forecast_reference_time"][1]
        assert dims == ("step",)
        np.testing.assert_allclose(arr, frt + np.array([6, 12]))
        assert attrs["standard_name"] == "time"
        assert attrs["bounds"] == "time_bnds"
        np.testing.assert_allclose(
            spec.aux_vars["time_bnds"][1], np.column_stack([arr - [6, 12], arr])
        )
        (vname,) = spec.data_vars
        assert spec.data_vars[vname][2]["cell_methods"] == "time: sum"


def test_grib_variables_with_different_step_intervals(data_dir, tmp_path):
    """Instantaneous and accumulated variables get separate bounds."""
    path = _write_modified(
        tmp_path,
        data_dir / "polar_stereographic_12x10.grib1",
        "mixed.grib",
        [
            [("shortName", name), ("stepType", st), ("startStep", 0), ("endStep", e)]
            for name, st in (("2t", "instant"), ("tp", "accum"))
            for e in (6, 12)
        ],
    )
    with grib_to_flat_netcdf_spec(str(path)) as spec:
        coords = {n: spec.data_vars[n][2]["coordinates"].split() for n in spec.data_vars}
        every = {**spec.coord_vars, **spec.aux_vars}
        assert "bounds" not in every["step"][2]
        assert "bounds" not in every["time"][2]
        # "step" is a dimension; "step_1" is its copy with bounds
        assert "step" in spec.data_vars["tp"][0]
        assert "step_1" in coords["tp"]
        assert "step_1" not in coords["2t"]
        assert "time" in coords["2t"]
        assert "time_1" in coords["tp"]
        assert "time" not in coords["tp"]
        np.testing.assert_array_equal(
            every["step_1_bnds"][1], [[0, 6], [0, 12]]
        )
        assert every["time_1"][2]["bounds"] == "time_1_bnds"
        assert spec.data_vars["tp"][2]["cell_methods"] == "time_1: sum"
        assert "cell_methods" not in spec.data_vars["2t"][2]


def test_grib_conflicting_step_intervals_give_no_bounds(data_dir, tmp_path):
    """One variable ending at one step but starting at different ones."""
    path = _write_modified(
        tmp_path,
        data_dir / "polar_stereographic_12x10.grib1",
        "conflict.grib",
        [
            [
                ("dataDate", date),
                ("stepType", "accum"),
                ("startStep", start),
                ("endStep", 6),
            ]
            for date, start in ((20200101, 0), (20200102, 3))
        ],
    )
    with pytest.warns(UserWarning, match="different start steps"):
        spec = grib_to_flat_netcdf_spec(str(path))

    assert "bounds" not in spec.aux_vars["step"][2]
    assert "time_bnds" not in spec.aux_vars
    spec.close()
