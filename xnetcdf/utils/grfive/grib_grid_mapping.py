"""CF grid mappings and horizontal coordinates for GRIB grids."""

from __future__ import annotations

import numpy as np

from .grib_polish import (_build_cell_bounds)
from .grib_utils import (FlatNetCDFSpec, _field_meta, _unique_name)


_ROTATED_LL_GRID_MAPPING_NAME = "rotated_latitude_longitude"


def _rotated_ll_coordinates(field, nj, ni):
    """Return the 1-d rotated-grid coordinates of a ``rotated_ll`` field.

    :Parameters:

        field: `earthkit.data.Field`
            A GRIB field on a ``rotated_ll`` grid.

        nj, ni: `int`
            The numbers of points along the rotated latitude and
            longitude axes.

    :Returns:

        `tuple` of `numpy.ndarray`
            The rotated latitude and longitude values, in degrees,
            in the same order as the field's data.

    """
    lat_first = float(_field_meta(field, "latitudeOfFirstGridPointInDegrees"))
    lat_last = float(_field_meta(field, "latitudeOfLastGridPointInDegrees"))
    lon_first = float(_field_meta(field, "longitudeOfFirstGridPointInDegrees"))
    lon_last = float(_field_meta(field, "longitudeOfLastGridPointInDegrees"))

    # Along-row longitudes may cross the 360 degree wrap
    if _field_meta(field, "iScansNegatively"):
        if lon_last > lon_first:
            lon_last -= 360.0
    elif lon_last < lon_first:
        lon_last += 360.0

    return (
        np.linspace(lat_first, lat_last, nj),
        np.linspace(lon_first, lon_last, ni),
    )


def _rotated_pole_attrs(field):
    """Return CF rotated pole grid mapping attributes for a GRIB field.

    The GRIB south pole of rotation ``(lat_s, lon_s)`` is converted to
    the CF north pole ``(-lat_s, lon_s + 180)``, with
    ``north_pole_grid_longitude`` taken from ``angleOfRotation``.

    """
    south_lat = float(_field_meta(field, "latitudeOfSouthernPoleInDegrees"))
    south_lon = float(_field_meta(field, "longitudeOfSouthernPoleInDegrees"))
    angle = _field_meta(field, "angleOfRotationInDegrees")
    return {
        "grid_mapping_name": _ROTATED_LL_GRID_MAPPING_NAME,
        "grid_north_pole_latitude": -south_lat,
        "grid_north_pole_longitude": (south_lon + 360.0) % 360.0 - 180.0,
        "north_pole_grid_longitude": float(angle or 0.0),
    }


def _replace_true_lat_lon_with_grid_mapping(spec, gm_name, gm_attrs):
    """Replace computed 1-d latitude/longitude with a grid mapping.

    The true latitude and longitude auxiliary coordinates are removed
    (they are computed by `earthkit`, not stored in the GRIB
    messages), and a scalar grid mapping variable is added and
    referenced by every data variable that spans a horizontal
    dimension of the removed coordinates.

    """
    lat_dims = spec.aux_vars.get("latitude", ((),))[0]
    if len(lat_dims) != 1 or lat_dims != spec.aux_vars.get(
        "longitude", ((),)
    )[0]:
        return spec

    gm_name = _unique_name(
        gm_name, {*spec.coord_vars, *spec.aux_vars, *spec.data_vars}
    )
    aux_vars = {
        k: v
        for k, v in spec.aux_vars.items()
        if k not in ("latitude", "longitude")
    }
    aux_vars[gm_name] = ((), np.array(0, dtype="i4"), gm_attrs)

    data_vars = {}
    for k, (d, v, a) in spec.data_vars.items():
        a = dict(a)
        if lat_dims[0] in d:
            a["grid_mapping"] = gm_name

        coords = [
            c
            for c in a.get("coordinates", "").split()
            if c not in ("latitude", "longitude")
        ]
        if coords:
            a["coordinates"] = " ".join(coords)
        else:
            a.pop("coordinates", None)

        data_vars[k] = (d, v, a)

    return FlatNetCDFSpec(
        dimensions=spec.dimensions,
        coord_vars=spec.coord_vars,
        data_vars=data_vars,
        aux_vars=aux_vars,
        global_attrs=spec.global_attrs,
    )


def _grib_earth_shape(field):
    """Return the ``(semi_major, semi_minor)`` axes in metres, or `None`."""
    if _field_meta(field, "edition") == 1:
        if _field_meta(field, "earthIsOblate"):
            return 6378160.0, 6356775.0  # IAU 1965

        return 6367470.0, 6367470.0

    shape = _field_meta(field, "shapeOfTheEarth")
    if shape == 0:
        return 6367470.0, 6367470.0

    if shape in (6, 8):
        r = 6371229.0 if shape == 6 else 6371200.0
        return r, r

    if shape == 1:
        r = _field_meta(field, "radius")
        return (float(r), float(r)) if r else None

    if shape in (2, 3, 7):
        if shape == 2:
            return 6378160.0, 6356775.0

        a = _field_meta(field, "scaledValueOfEarthMajorAxis")
        b = _field_meta(field, "scaledValueOfEarthMinorAxis")
        if not a or not b:
            return None

        # Major/minor axes are in km (shape 3) or m (shape 7)
        f = 1000.0 if shape == 3 else 1.0
        return float(a) * f, float(b) * f

    if shape == 4:
        return 6378137.0, 6356752.314140

    if shape == 5:
        return 6378137.0, 6356752.314245

    return None


# GRIB grid types that are supported as CF projected grids
_PROJECTED_GRID_TYPES = {
    "mercator",
    "lambert",
    "lambert_azimuthal_equal_area",
    "polar_stereographic",
}


def _projected_grid_definition(field):
    """Return the CF grid mapping and PROJ definition of a projected grid.

    :Returns:

        `tuple` or `None`
            The grid mapping attributes and the PROJ string, or
            `None` if the grid is not supported.

    """
    grid_type = _field_meta(field, "gridType")
    earth = _grib_earth_shape(field)
    if earth is None:
        return

    a, b = earth
    axes = {"semi_major_axis": a, "semi_minor_axis": b}
    proj_axes = f"+a={a!r} +b={b!r}"

    def get(key):
        v = _field_meta(field, key)
        return None if v is None else float(v)

    if grid_type == "lambert":
        lad, lov = get("LaDInDegrees"), get("LoVInDegrees")
        lat1, lat2 = get("Latin1InDegrees"), get("Latin2InDegrees")
        if None in (lad, lov, lat1, lat2):
            return

        parallels = [lat1] if lat1 == lat2 else [lat1, lat2]
        attrs = {
            "grid_mapping_name": "lambert_conformal_conic",
            "standard_parallel": parallels,
            "longitude_of_central_meridian": lov,
            "latitude_of_projection_origin": lad,
        }
        proj = (
            f"+proj=lcc +lat_0={lad!r} +lon_0={lov!r} "
            f"+lat_1={lat1!r} +lat_2={lat2!r}"
        )
    elif grid_type == "lambert_azimuthal_equal_area":
        lat0 = get("standardParallelInMicrodegrees")
        lon0 = get("centralLongitudeInMicrodegrees")
        if None in (lat0, lon0):
            return

        lat0, lon0 = lat0 * 1e-6, lon0 * 1e-6
        attrs = {
            "grid_mapping_name": "lambert_azimuthal_equal_area",
            "longitude_of_projection_origin": lon0,
            "latitude_of_projection_origin": lat0,
        }
        proj = f"+proj=laea +lat_0={lat0!r} +lon_0={lon0!r}"
    elif grid_type == "polar_stereographic":
        lad, orient = get("LaDInDegrees"), get("orientationOfTheGridInDegrees")
        if None in (lad, orient):
            return

        south = bool(_field_meta(field, "southPoleOnProjectionPlane"))
        pole = -90.0 if south else 90.0
        attrs = {
            "grid_mapping_name": "polar_stereographic",
            "straight_vertical_longitude_from_pole": orient,
            "latitude_of_projection_origin": pole,
            "standard_parallel": lad,
        }
        proj = f"+proj=stere +lat_0={pole!r} +lat_ts={lad!r} +lon_0={orient!r}"
    elif grid_type == "mercator":
        lad, lon0 = get("LaDInDegrees"), get("orientationOfTheGridInDegrees")
        if None in (lad, lon0):
            return

        attrs = {
            "grid_mapping_name": "mercator",
            "standard_parallel": lad,
            "longitude_of_projection_origin": lon0,
        }
        proj = f"+proj=merc +lat_0=0 +lat_ts={lad!r} +lon_0={lon0!r}"
    else:
        return

    attrs.update(axes)
    return attrs, f"{proj} {proj_axes} +units=m +no_defs"


def _projected_coordinates(field, proj, ny, nx):
    """Return 1-d projection ``(y, x)`` coordinates in metres.

    The coordinates are derived from the first grid point and the grid
    increments, in the same order as the field's data, or `None` if
    `pyproj` is not available.

    """
    try:
        import pyproj
    except ImportError:
        return

    crs = pyproj.CRS.from_proj4(proj)
    transformer = pyproj.Transformer.from_crs(
        crs.geodetic_crs, crs, always_xy=True
    )
    x0, y0 = transformer.transform(
        float(_field_meta(field, "longitudeOfFirstGridPointInDegrees")),
        float(_field_meta(field, "latitudeOfFirstGridPointInDegrees")),
    )
    # Mercator grids use Di/Dj instead of Dx/Dy
    dx = _field_meta(field, "DxInMetres")
    dy = _field_meta(field, "DyInMetres")
    if dx is None or dy is None:
        dx = _field_meta(field, "DiInMetres")
        dy = _field_meta(field, "DjInMetres")

    dx, dy = float(dx), float(dy)
    if _field_meta(field, "iScansNegatively"):
        dx = -dx

    if not _field_meta(field, "jScansPositively"):
        dy = -dy

    return y0 + dy * np.arange(ny), x0 + dx * np.arange(nx)


def _apply_projected_grid_mapping(spec, field):
    """Add projection coordinates and a grid mapping to a projected grid.

    The 2-d true latitude and longitude are removed, since they are
    computed by `earthkit` and are defined by the projection
    coordinates and the grid mapping. Cell bounds are created for the
    projection coordinates, whose points are equally spaced.

    """
    definition = _projected_grid_definition(field)
    if definition is None:
        return spec

    gm_attrs, proj = definition

    # The computed lat/lon are either 2-d auxiliary coordinates, or
    # 1-d coordinate variables (in which case their dimensions, named
    # after them, are renamed to "y" and "x")
    lat_dims = spec.aux_vars.get("latitude", ((),))[0]
    lon_dims = spec.aux_vars.get("longitude", ((),))[0]
    if len(lat_dims) == 2 and lat_dims == lon_dims:
        ydim, xdim = lat_dims
    elif (
        spec.coord_vars.get("latitude", ((),))[0] == ("latitude",)
        and spec.coord_vars.get("longitude", ((),))[0] == ("longitude",)
    ):
        ydim, xdim = "y", "x"
        if {ydim, xdim} & {
            *spec.dimensions,
            *spec.coord_vars,
            *spec.aux_vars,
            *spec.data_vars,
        }:
            return spec
    else:
        return spec

    if (ydim, xdim) == lat_dims:
        ny, nx = spec.dimensions[ydim], spec.dimensions[xdim]
        if "x" in spec.coord_vars or "y" in spec.coord_vars:
            return spec
    else:
        ny, nx = spec.dimensions["latitude"], spec.dimensions["longitude"]

    coords = _projected_coordinates(field, proj, ny, nx)
    if coords is None:
        return spec

    rename = {"latitude": ydim, "longitude": xdim}

    def _rename(dims):
        return tuple(rename.get(d, d) if (ydim, xdim) != lat_dims else d
                     for d in dims)

    y, x = coords
    gm_name = _unique_name(
        gm_attrs["grid_mapping_name"],
        {*spec.coord_vars, *spec.aux_vars, *spec.data_vars},
    )

    # Use the dimensions' own names as the coordinate variable names
    coord_vars = {
        ydim: (
            (ydim,),
            y,
            {
                "standard_name": "projection_y_coordinate",
                "units": "m",
                "axis": "Y",
            },
        ),
        xdim: (
            (xdim,),
            x,
            {
                "standard_name": "projection_x_coordinate",
                "units": "m",
                "axis": "X",
            },
        ),
        **{
            k: (_rename(d), v, a)
            for k, (d, v, a) in spec.coord_vars.items()
            if k not in ("latitude", "longitude")
        },
    }
    aux_vars = {
        k: (_rename(d), v, a)
        for k, (d, v, a) in spec.aux_vars.items()
        if k not in ("latitude", "longitude")
    }
    aux_vars[gm_name] = ((), np.array(0, dtype="i4"), gm_attrs)

    data_vars = {}
    for k, (d, v, a) in spec.data_vars.items():
        d = _rename(d)
        a = dict(a)
        if {ydim, xdim}.issubset(d):
            a["grid_mapping"] = gm_name

        cs = [
            c
            for c in a.get("coordinates", "").split()
            if c not in ("latitude", "longitude")
        ]
        if cs:
            a["coordinates"] = " ".join(cs)
        else:
            a.pop("coordinates", None)

        data_vars[k] = (d, v, a)

    dimensions, bnds_vars, _ = _build_cell_bounds(
        {rename.get(d, d) if (ydim, xdim) != lat_dims else d: n
         for d, n in spec.dimensions.items()},
        coord_vars,
        aux_vars,
        None,
    )
    aux_vars.update(bnds_vars)

    return FlatNetCDFSpec(
        dimensions=dimensions,
        coord_vars=coord_vars,
        data_vars=data_vars,
        aux_vars=aux_vars,
        global_attrs=spec.global_attrs,
    )


def _apply_healpix_grid_mapping(spec, field):
    """Add a CF-1.13 ``healpix`` grid mapping, dropping computed lat/lon."""
    nside = _field_meta(field, "Nside")
    order = _field_meta(field, "orderingConvention")
    if (
        nside is None
        or order not in ("ring", "nested")
        or int(nside) < 1
        or int(nside) & (int(nside) - 1)
    ):
        return spec

    # CF-1.13 describes the resolution by refinement level, with
    # nside = 2 ** refinement_level
    refinement_level = int(nside).bit_length() - 1

    hdim = spec.aux_vars.get("latitude", ((),))[0]
    if len(hdim) != 1 or "healpix_index" in {
        *spec.dimensions,
        *spec.coord_vars,
        *spec.aux_vars,
        *spec.data_vars,
    }:
        return spec

    hdim = hdim[0]
    spec = _replace_true_lat_lon_with_grid_mapping(
        spec,
        "healpix",
        {
            "grid_mapping_name": "healpix",
            "indexing_scheme": str(order),
            "refinement_level": refinement_level,
        },
    )

    # The cell index is implied by the grid definition, and the data
    # are stored in index order
    def _rename(dims):
        return tuple("healpix_index" if d == hdim else d for d in dims)

    dimensions = {
        "healpix_index" if d == hdim else d: n
        for d, n in spec.dimensions.items()
    }
    coord_vars = {
        "healpix_index": (
            ("healpix_index",),
            np.arange(dimensions["healpix_index"], dtype="i8"),
            {"standard_name": "healpix_index", "units": "1"},
        ),
        **{k: (_rename(d), v, a) for k, (d, v, a) in spec.coord_vars.items()},
    }
    return FlatNetCDFSpec(
        dimensions=dimensions,
        coord_vars=coord_vars,
        data_vars={
            k: (_rename(d), v, a) for k, (d, v, a) in spec.data_vars.items()
        },
        aux_vars={
            k: (_rename(d), v, a) for k, (d, v, a) in spec.aux_vars.items()
        },
        global_attrs=spec.global_attrs,
    )


def _apply_reduced_rotated_gg_grid_mapping(spec, field):
    """Add a rotated pole grid mapping for a reduced rotated Gaussian grid.

    The 1-d rotated coordinates are not stored in the GRIB messages,
    so none are created, and the computed true latitude and longitude
    are dropped.

    """
    try:
        attrs = _rotated_pole_attrs(field)
    except (TypeError, ValueError):
        return spec

    return _replace_true_lat_lon_with_grid_mapping(
        spec, _ROTATED_LL_GRID_MAPPING_NAME, attrs
    )


def _apply_grid_mapping(spec, ds):
    """Add CF grid mapping information for supported grids.

    GRIB ``healpix`` and ``reduced_rotated_gg`` grids get a grid
    mapping and lose their computed lat/lon. For GRIB ``rotated_ll``
    grids the
    horizontal dimensions are renamed to ``grid_latitude`` and
    ``grid_longitude``, 1-d rotated coordinate variables are created,
    and a ``rotated_latitude_longitude`` grid mapping variable is
    added and referenced by the data variables' ``grid_mapping``
    attribute. The 2-d true latitude and longitude auxiliary
    coordinates are removed, because they are not stored in the GRIB
    messages (they are computed by `earthkit`) and are fully defined
    by the rotated coordinates and the grid mapping. Specs on other
    grids are returned unchanged.

    The GRIB south pole of rotation ``(lat_s, lon_s)`` is converted to
    the CF north pole ``(-lat_s, lon_s + 180)``, with
    ``north_pole_grid_longitude`` taken from ``angleOfRotation``.

    :Parameters:

        spec: `FlatNetCDFSpec`
            The CF normalised specification.

        ds: `earthkit.data.FieldList`
            The source of *spec*, of which only the first field is
            inspected.

    :Returns:

        `FlatNetCDFSpec`

    """
    field = ds[0]
    grid_type = _field_meta(field, "gridType")
    if grid_type == "healpix":
        return _apply_healpix_grid_mapping(spec, field)

    if grid_type == "reduced_rotated_gg":
        return _apply_reduced_rotated_gg_grid_mapping(spec, field)

    if grid_type in _PROJECTED_GRID_TYPES:
        return _apply_projected_grid_mapping(spec, field)

    if grid_type != "rotated_ll":
        return spec

    lat_dims = spec.aux_vars.get("latitude", ((),))[0]
    lon_dims = spec.aux_vars.get("longitude", ((),))[0]
    if len(lat_dims) != 2 or lat_dims != lon_dims:
        return spec

    ydim, xdim = lat_dims
    rlat, rlon = _rotated_ll_coordinates(
        field, spec.dimensions[ydim], spec.dimensions[xdim]
    )

    gm_attrs = _rotated_pole_attrs(field)
    gm_name = _unique_name(
        _ROTATED_LL_GRID_MAPPING_NAME,
        {*spec.coord_vars, *spec.aux_vars, *spec.data_vars},
    )

    rename = {ydim: "grid_latitude", xdim: "grid_longitude"}
    if any(
        n in spec.dimensions or n in spec.coord_vars
        for n in rename.values()
    ):
        return spec

    def _rename(dims):
        return tuple(rename.get(d, d) for d in dims)

    dimensions = {rename.get(d, d): n for d, n in spec.dimensions.items()}

    coord_vars = {
        "grid_latitude": (
            ("grid_latitude",),
            rlat,
            {
                "standard_name": "grid_latitude",
                "units": "degrees",
                "axis": "Y",
            },
        ),
        "grid_longitude": (
            ("grid_longitude",),
            rlon,
            {
                "standard_name": "grid_longitude",
                "units": "degrees",
                "axis": "X",
            },
        ),
        **spec.coord_vars,
    }

    aux_vars = {
        k: (_rename(d), v, a)
        for k, (d, v, a) in spec.aux_vars.items()
        if k not in ("latitude", "longitude")
    }
    aux_vars[gm_name] = ((), np.array(0, dtype="i4"), gm_attrs)

    data_vars = {}
    for k, (d, v, a) in spec.data_vars.items():
        d = _rename(d)
        a = dict(a)
        if {"grid_latitude", "grid_longitude"}.issubset(d):
            a["grid_mapping"] = gm_name

        coords = [
            c for c in a.get("coordinates", "").split()
            if c not in ("latitude", "longitude")
        ]
        if coords:
            a["coordinates"] = " ".join(coords)
        else:
            a.pop("coordinates", None)

        data_vars[k] = (d, v, a)

    # Cell bounds for the new rotated coordinates, assuming that the
    # grid points are cell centres (as for regular lat/lon grids)
    dimensions, bnds_vars, _ = _build_cell_bounds(
        dimensions, coord_vars, aux_vars, None
    )
    aux_vars.update(bnds_vars)

    return FlatNetCDFSpec(
        dimensions=dimensions,
        coord_vars=coord_vars,
        data_vars=data_vars,
        aux_vars=aux_vars,
        global_attrs=spec.global_attrs,
    )
