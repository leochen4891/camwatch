"""Tests for the captured_at stamp fix (`_stamp_captured_at`).

Ticket pts-stamp (2026-06-07): captured_at was wall clock at event
processing — grid exit plus an uptime-dependent 0-18 s queue-staleness
term. It is now derived from the grid-EXIT frame's stream-timeline
timestamp (coordinator ruling: exit, not entry, so the standing
consumer anchor captured_at - elapsed_s = grid entry stays valid and
becomes exact — DATA-CONTRACTS rule 7).

Runs under pytest, or standalone: `python tests/test_pts_stamp.py`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from camwatch.capture_worker import _stamp_captured_at

_WALL = datetime(2026, 6, 7, 12, 0, 0, tzinfo=timezone.utc)


def test_stamp_subtracts_processing_staleness():
    """The exit frame is 11s old when the event is processed (stale
    loop): the stamp lands 11s in the past — at road time."""
    stamp, corr = _stamp_captured_at(exit_ts=100.0, now_mono=111.0, now_wall=_WALL)
    assert corr == 11.0
    assert stamp == _WALL - timedelta(seconds=11.0)


def test_stamp_correction_is_uptime_independent():
    """The same staleness produces the same correction whether the
    monotonic clock reads minutes or days — only the exit-to-now delta
    matters (the old wall-clock stamp drifted ~1s/h of uptime)."""
    _, young = _stamp_captured_at(exit_ts=60.0, now_mono=63.0, now_wall=_WALL)
    _, old = _stamp_captured_at(
        exit_ts=864000.0, now_mono=864003.0, now_wall=_WALL
    )
    assert young == old == 3.0


def test_entry_anchor_formula_stays_valid():
    """Consumers anchor grid entry as captured_at - elapsed_s (rule 7).
    With exit stamping that formula is exact: stamp - elapsed lands on
    the entry frame's wall time."""
    entry_ts, exit_ts, now = 100.0, 101.5, 103.0  # 1.5s transit, 1.5s stale
    stamp, _ = _stamp_captured_at(exit_ts, now, _WALL)
    elapsed = exit_ts - entry_ts
    entry_wall = stamp - timedelta(seconds=elapsed)
    # ground truth: entry happened (now - entry_ts) seconds before _WALL
    assert entry_wall == _WALL - timedelta(seconds=now - entry_ts)


def test_negative_delta_clamps_to_now():
    """A pathological timeline (exit ts ahead of now) must never stamp
    into the future — clamp to the current wall clock."""
    stamp, corr = _stamp_captured_at(exit_ts=200.0, now_mono=199.0, now_wall=_WALL)
    assert corr == 0.0
    assert stamp == _WALL


def test_stamp_preserves_timezone():
    local_wall = datetime(2026, 6, 7, 8, 0, 0).astimezone()
    stamp, _ = _stamp_captured_at(exit_ts=0.0, now_mono=5.0, now_wall=local_wall)
    assert stamp.tzinfo == local_wall.tzinfo
    assert stamp.isoformat(timespec="seconds").endswith(
        local_wall.isoformat(timespec="seconds")[-6:]
    )


if __name__ == "__main__":
    import sys

    import pytest as _pytest

    sys.exit(_pytest.main([__file__, "-v"]))
