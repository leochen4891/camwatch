"""Two-line crossing state machine.

For each tracked object, holds the most recent (t, x) sample and watches for
the bbox bottom-center's x to cross line A and line B. When both have been
crossed, emits a CrossingEvent with linearly interpolated crossing times.

Used by:
- camwatch.calibrate.cmd_capture (CLI capture window)
- camwatch.capture_worker.CaptureWorker (web UI background thread)
- camwatch.speed.SpeedTracker is a parallel implementation kept for live mode;
  if/when we consolidate further, this module is the source of truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CrossingEvent:
    track_id: int
    cls_name: str
    direction: str           # "N" if A crossed first, else "S"
    t_a: float
    t_b: float
    elapsed_s: float
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class _State:
    last_x: float | None = None
    last_t: float | None = None
    last_seen: float = 0.0
    cls_name: str = ""
    t_cross_a: float | None = None
    t_cross_b: float | None = None


def _interp_cross(
    x_prev: float, x_curr: float, t_prev: float, t_curr: float, line_x: float
) -> float:
    span = x_curr - x_prev
    if span == 0:
        return t_curr
    return t_prev + (line_x - x_prev) / span * (t_curr - t_prev)


class CrossingDetector:
    """Per-track A/B crossing detector.

    Call `update(tracks, t)` once per frame with the list of tracks (objects
    exposing .track_id, .cls_name, .ground_point, .bbox). Yields a
    CrossingEvent for each track that completes both crossings on this frame.
    """

    def __init__(
        self,
        line_a_x: int,
        line_b_x: int,
        max_track_age_s: float = 5.0,
    ) -> None:
        if line_a_x >= line_b_x:
            raise ValueError("line_a_x must be < line_b_x")
        self.line_a = int(line_a_x)
        self.line_b = int(line_b_x)
        self.max_age = float(max_track_age_s)
        self._state: dict[int, _State] = {}

    def update(self, tracks: list[Any], t: float) -> list[CrossingEvent]:
        events: list[CrossingEvent] = []
        seen: set[int] = set()
        for tr in tracks:
            tid = int(tr.track_id)
            seen.add(tid)
            x = float(tr.ground_point[0])
            st = self._state.setdefault(tid, _State())
            new_track = st.last_x is None
            if not st.cls_name:
                st.cls_name = getattr(tr, "cls_name", "")
            had_a = st.t_cross_a is not None
            had_b = st.t_cross_b is not None
            ev = self._step(st, tid, x, t, getattr(tr, "bbox", None))
            if ev is not None:
                events.append(ev)
                self._state.pop(tid, None)
                continue
            # Log first sighting and partial crossings as they happen so the
            # log is enough to diagnose missed cars.
            if new_track:
                log.info("track %d (%s) first seen at x=%.0f", tid, st.cls_name, x)
            if not had_a and st.t_cross_a is not None:
                log.info("track %d crossed line A at x=%.0f t=%.3fs", tid, x, st.t_cross_a)
            if not had_b and st.t_cross_b is not None:
                log.info("track %d crossed line B at x=%.0f t=%.3fs", tid, x, st.t_cross_b)
            st.last_x = x
            st.last_t = t
            st.last_seen = t

        # Garbage-collect stale tracks that never finished crossing.
        stale = [
            tid for tid, st in self._state.items()
            if tid not in seen and (t - st.last_seen) > self.max_age
        ]
        for tid in stale:
            st = self._state.pop(tid)
            if st.t_cross_a is not None or st.t_cross_b is not None:
                log.info(
                    "track %d aged out with partial crossing (a=%s b=%s last_x=%.0f)",
                    tid,
                    "yes" if st.t_cross_a is not None else "no",
                    "yes" if st.t_cross_b is not None else "no",
                    st.last_x or 0,
                )
        return events

    def _step(
        self,
        st: _State,
        tid: int,
        x: float,
        t: float,
        bbox: tuple | None,
    ) -> CrossingEvent | None:
        if st.last_x is None or st.last_t is None:
            return None
        xp, tp = st.last_x, st.last_t

        if (
            st.t_cross_a is None
            and (xp - self.line_a) * (x - self.line_a) <= 0
            and xp != self.line_a
        ):
            st.t_cross_a = _interp_cross(xp, x, tp, t, self.line_a)
        if (
            st.t_cross_b is None
            and (xp - self.line_b) * (x - self.line_b) <= 0
            and xp != self.line_b
        ):
            st.t_cross_b = _interp_cross(xp, x, tp, t, self.line_b)

        if st.t_cross_a is None or st.t_cross_b is None:
            return None

        elapsed = abs(st.t_cross_b - st.t_cross_a)
        if elapsed <= 0:
            log.debug("track %d: zero elapsed at crossing, dropping", tid)
            return None
        direction = "N" if st.t_cross_a < st.t_cross_b else "S"
        return CrossingEvent(
            track_id=tid,
            cls_name=st.cls_name,
            direction=direction,
            t_a=st.t_cross_a,
            t_b=st.t_cross_b,
            elapsed_s=elapsed,
            bbox=tuple(bbox) if bbox is not None else None,
        )
