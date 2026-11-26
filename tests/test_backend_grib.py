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
        np.testing.assert_allclose(np.asarray(lazy), ds[0].to_numpy())


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
    assert spec.aux_vars[x[2]["bounds"]][1].shape == (12, 2)
    assert spec.aux_vars[y[2]["bounds"]][1].shape == (10, 2)

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
