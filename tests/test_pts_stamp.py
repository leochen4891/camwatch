"""Tests for the captured_at stamp fix (`_stamp_captured_at`).

Ticket pts-stamp (2026-06-07): captured_at was wall clock at event
processing — grid exit plus an uptime-dependent 0-18 s queue-staleness
term. It is now derived from the grid-entry frame's stream-timeline
timestamp, so the stamp reflects when the vehicle was on the road.

Runs under pytest, or standalone: `python tests/test_pts_stamp.py`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from camwatch.capture_worker import _stamp_captured_at

_WALL = datetime(2026, 6, 7, 12, 0, 0, tzinfo=timezone.utc)


def test_stamp_subtracts_entry_age():
    """Vehicle entered the grid 12.5s ago (1.5s transit + 11s staleness):
    the stamp lands 12.5s in the past, regardless of how stale the
    processing loop is running."""
    stamp, corr = _stamp_captured_at(entry_ts=100.0, now_mono=112.5, now_wall=_WALL)
    assert corr == 12.5
    assert stamp == _WALL - timedelta(seconds=12.5)


def test_stamp_correction_is_uptime_independent():
    """The same transit produces the same correction whether the
    monotonic clock reads minutes or days — only the entry-to-now delta
    matters (the old wall-clock stamp grew ~1s/h of uptime)."""
    _, young = _stamp_captured_at(entry_ts=60.0, now_mono=63.0, now_wall=_WALL)
    _, old = _stamp_captured_at(
        entry_ts=864000.0, now_mono=864003.0, now_wall=_WALL
    )
    assert young == old == 3.0


def test_negative_delta_clamps_to_now():
    """A pathological timeline (entry ts ahead of now) must never stamp
    into the future — clamp to the current wall clock."""
    stamp, corr = _stamp_captured_at(entry_ts=200.0, now_mono=199.0, now_wall=_WALL)
    assert corr == 0.0
    assert stamp == _WALL


def test_stamp_preserves_timezone():
    local_wall = datetime(2026, 6, 7, 8, 0, 0).astimezone()
    stamp, _ = _stamp_captured_at(entry_ts=0.0, now_mono=5.0, now_wall=local_wall)
    assert stamp.tzinfo == local_wall.tzinfo
    assert stamp.isoformat(timespec="seconds").endswith(
        local_wall.isoformat(timespec="seconds")[-6:]
    )


if __name__ == "__main__":
    import sys

    import pytest as _pytest

    sys.exit(_pytest.main([__file__, "-v"]))
