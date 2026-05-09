"""Grid-entry/exit crossing detector.

Replacement for the pixel-line `CrossingDetector`. A pass is defined as a
track entering the calibrated homography grid and later leaving it (or
being lost while still inside, after `max_track_age_s`). This is the
same rectangle the speed estimator uses, so trigger and speed live in
one coordinate system instead of two.

The output `CrossingEvent` schema is identical to `crossing.py` so callers
(capture_worker, recorder, JSONL writer) don't need to change:
- `t_a` is the chronological first in-grid timestamp (entry)
- `t_b` is the chronological last in-grid timestamp (exit, or last seen)
- `direction` = "N" if Y increased between entry and exit, else "S"

Compared to the 2-line trigger:
- Captures cars in either lane symmetrically (the grid covers the whole
  road by construction, so there is no "near-lane miss" failure mode)
- Uses world-coordinate motion, not pixel x; behavior is uniform along
  the road instead of biased toward the line positions
- One fewer hand-tuned calibration value (the line positions become
  decorative; the recorder still draws them as visual reference markers)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .crossing import CrossingEvent
from .homography import Homography

log = logging.getLogger(__name__)


@dataclass
class _GridState:
    cls_name: str = ""
    bbox: tuple[float, float, float, float] | None = None
    in_grid: bool = False
    entry_ts: float | None = None
    entry_X: float = 0.0
    entry_Y: float = 0.0
    last_in_grid_ts: float | None = None
    last_in_grid_X: float = 0.0
    last_in_grid_Y: float = 0.0
    last_seen: float = 0.0
    fired: bool = False  # event already emitted for this track ID


class GridCrossingDetector:
    """Per-track grid-entry/exit detector.

    Call `update(tracks, ts)` once per frame with the list of tracks
    (objects exposing .track_id, .cls_name, .ground_point, .bbox).
    Yields a `CrossingEvent` when:
      - a track that was inside the grid is now outside (clean exit), OR
      - a track that was inside the grid hasn't been seen for `max_track_age_s`
        (lost while inside; treat last in-grid sample as exit).

    Spurious events from bbox jitter at the boundary are filtered by:
      - `tolerance_m` slack zone on the grid bounds (same one used by the
        speed estimator for in-grid sampling)
      - `min_dy_m` minimum Y displacement between entry and exit
      - `min_elapsed_s` minimum in-grid duration
      - per-direction `dedupe_window_s` to drop split-bbox duplicates
    """

    def __init__(
        self,
        homography: Homography,
        grid_x_min: float,
        grid_x_max: float,
        grid_y_min: float,
        grid_y_max: float,
        tolerance_m: float = 0.5,
        max_track_age_s: float = 5.0,
        min_dy_m: float = 3.0,
        min_elapsed_s: float = 0.2,
        dedupe_window_s: float = 0.5,
    ) -> None:
        self._homog = homography
        self._x_min = grid_x_min - tolerance_m
        self._x_max = grid_x_max + tolerance_m
        self._y_min = grid_y_min - tolerance_m
        self._y_max = grid_y_max + tolerance_m
        self._max_age = float(max_track_age_s)
        self._min_dy = float(min_dy_m)
        self._min_elapsed = float(min_elapsed_s)
        self._dedupe_window = float(dedupe_window_s)
        self._state: dict[int, _GridState] = {}
        self._last_event_t: dict[str, float] = {"N": -1e9, "S": -1e9}

    def _in_grid(self, X: float, Y: float) -> bool:
        return self._x_min <= X <= self._x_max and self._y_min <= Y <= self._y_max

    def update(self, tracks: list[Any], t: float) -> list[CrossingEvent]:
        events: list[CrossingEvent] = []
        seen: set[int] = set()
        for tr in tracks:
            tid = int(tr.track_id)
            seen.add(tid)
            u, v = float(tr.ground_point[0]), float(tr.ground_point[1])
            X, Y = self._homog.project(u, v)
            in_grid = self._in_grid(X, Y)
            st = self._state.get(tid)
            new_track = st is None
            if st is None:
                st = _GridState()
                self._state[tid] = st
            if not st.cls_name:
                st.cls_name = getattr(tr, "cls_name", "")
            st.bbox = tuple(getattr(tr, "bbox", ())) or st.bbox
            st.last_seen = t

            if new_track:
                log.info(
                    "track %d (%s) first seen at u=%.0f v=%.0f X=%.2f Y=%.2f in_grid=%s",
                    tid, st.cls_name, u, v, X, Y, in_grid,
                )

            if in_grid:
                if not st.in_grid:
                    st.in_grid = True
                    st.entry_ts = t
                    st.entry_X = X
                    st.entry_Y = Y
                    log.info(
                        "track %d entered grid at X=%.2f Y=%.2f t=%.3fs",
                        tid, X, Y, t,
                    )
                st.last_in_grid_ts = t
                st.last_in_grid_X = X
                st.last_in_grid_Y = Y
            else:
                if st.in_grid and not st.fired:
                    ev = self._finalize(tid, st, exit_ts=t, exit_X=X, exit_Y=Y)
                    st.in_grid = False
                    if ev is not None and self._accept(ev):
                        events.append(ev)
                        st.fired = True

        # Garbage-collect stale tracks. If still inside the grid when lost,
        # finalize using the last in-grid sample as the exit point.
        stale: list[int] = []
        for tid, st in self._state.items():
            if tid in seen:
                continue
            if (t - st.last_seen) <= self._max_age:
                continue
            stale.append(tid)
            if st.in_grid and not st.fired and st.last_in_grid_ts is not None:
                ev = self._finalize(
                    tid, st,
                    exit_ts=st.last_in_grid_ts,
                    exit_X=st.last_in_grid_X,
                    exit_Y=st.last_in_grid_Y,
                )
                if ev is not None and self._accept(ev):
                    events.append(ev)
                    st.fired = True
                else:
                    log.info(
                        "track %d aged out inside grid (entry Y=%.2f, last Y=%.2f, "
                        "elapsed=%.3fs — below thresholds, dropped)",
                        tid, st.entry_Y, st.last_in_grid_Y,
                        (st.last_in_grid_ts - (st.entry_ts or st.last_in_grid_ts)),
                    )
        for tid in stale:
            self._state.pop(tid, None)
        return events

    def _finalize(
        self,
        tid: int,
        st: _GridState,
        exit_ts: float,
        exit_X: float,
        exit_Y: float,
    ) -> CrossingEvent | None:
        if st.entry_ts is None:
            return None
        elapsed = exit_ts - st.entry_ts
        dy = exit_Y - st.entry_Y
        if elapsed < self._min_elapsed:
            return None
        if abs(dy) < self._min_dy:
            return None
        direction = "N" if dy > 0 else "S"
        log.info(
            "track %d exited grid: entry=(%.2f,%.2f) exit=(%.2f,%.2f) "
            "elapsed=%.3fs dy=%.2fm direction=%s",
            tid, st.entry_X, st.entry_Y, exit_X, exit_Y, elapsed, dy, direction,
        )
        return CrossingEvent(
            track_id=tid,
            cls_name=st.cls_name,
            direction=direction,
            t_a=st.entry_ts,  # chronological first
            t_b=exit_ts,      # chronological last
            elapsed_s=elapsed,
            bbox=tuple(st.bbox) if st.bbox is not None else None,
        )

    def _accept(self, ev: CrossingEvent) -> bool:
        """Apply per-direction dedup window to drop split-bbox duplicates
        that BotSORT occasionally generates for the same vehicle."""
        event_t = max(ev.t_a, ev.t_b)
        gap = event_t - self._last_event_t[ev.direction]
        if gap < self._dedupe_window:
            log.info(
                "track %d %s grid-pass dropped (%.3fs after previous %s pass)",
                ev.track_id, ev.direction, gap, ev.direction,
            )
            return False
        self._last_event_t[ev.direction] = event_t
        return True
