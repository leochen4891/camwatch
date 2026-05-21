"""Daylight predicate for the capture worker's enricher gate.

Sister implementation to camwatch-enricher's `daylight` module. Both
services need to agree on what counts as daylight so the gate (capture
worker skips enricher for non-daylight) matches the enricher's index
filter (defense-in-depth, never matches non-daylight even if called).

Coords default to Livingston, NJ (40.7956, -74.3148). Override via
`enricher.daylight` in config/config.yaml when the camera moves.
"""
from __future__ import annotations

from datetime import date as date_t, datetime, timedelta, timezone
from functools import lru_cache

from astral import LocationInfo
from astral.sun import sun


@lru_cache(maxsize=4096)
def _sun_for(lat: float, lon: float, date_ordinal: int, tz_offset_seconds: int) -> tuple[datetime, datetime]:
    tz = timezone(timedelta(seconds=tz_offset_seconds))
    observer = LocationInfo("loc", "earth", "UTC", lat, lon).observer
    s = sun(observer, date=date_t.fromordinal(date_ordinal), tzinfo=tz)
    return s["sunrise"], s["sunset"]


def is_daylight(
    captured_at_iso: str, lat: float = 40.7956, lon: float = -74.3148,
    buffer_hours: float = 1.0,
) -> bool:
    """True iff `captured_at_iso` falls within (sunrise+buf, sunset-buf).

    `captured_at_iso` is a tz-aware ISO timestamp; sunrise/sunset are
    computed for the capture's own local date and timezone. Defaults
    target Livingston, NJ — override at call site or in config.
    """
    dt = datetime.fromisoformat(captured_at_iso)
    off = dt.utcoffset()
    if off is None:
        raise ValueError("captured_at must be timezone-aware: " + captured_at_iso)
    sunrise, sunset = _sun_for(lat, lon, dt.date().toordinal(), int(off.total_seconds()))
    buf = timedelta(hours=buffer_hours)
    return (sunrise + buf) <= dt <= (sunset - buf)
