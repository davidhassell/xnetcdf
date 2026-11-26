"""CF time and timedelta encoding helpers for the GRIB to CF conversion."""

from __future__ import annotations

import datetime
import math
import numpy as np

from .grib_constants import (CF_CALENDAR, NC_FILL_DOUBLE, NC_FILL_INT64)


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
