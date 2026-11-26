"""Constants and lookup tables for the GRIB to CF conversion."""

from __future__ import annotations

import numpy as np




CF_CONVENTIONS = "CF-1.13"

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

_HORIZONTAL_1D_BOUNDS_NAMES = {
    "latitude",
    "longitude",
    "grid_latitude",
    "grid_longitude",
    "projection_x_coordinate",
    "projection_y_coordinate",
    "x",
    "y",
}


_LATITUDE_LIKE_NAMES = {"latitude", "grid_latitude"}


# GRIB grid types whose 1-d horizontal coordinates are equally spaced
# cell centres, so that the mid-points between them are valid cell
# bounds. This is not the case for, e.g., Gaussian grids (whose cell
# edges are not mid-points) or Mercator (whose latitudes are not
# equally spaced), so those get no horizontal bounds.
_MIDPOINT_BOUNDS_GRID_TYPES = {"regular_ll", "rotated_ll"}

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
