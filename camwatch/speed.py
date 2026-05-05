"""Two-line crossing speed estimator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .detect import Track

log = logging.getLogger(__name__)

MPS_TO_MPH = 2.2369362920544


@dataclass
class SpeedEvent:
    track_id: int
    cls_name: str
    direction: str  # "N" (left→right, southbound→northbound) or "S"
    speed_mph: float
    t_a: float
    t_b: float
    bbox: tuple[float, float, float, float]


@dataclass
class _TrackState:
    last_x: float | None = None
    last_t: float | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    last_cls: str = ""
    t_cross_a: float | None = None
    t_cross_b: float | None = None
    last_seen: float = field(default=0.0)


def _interp_cross(x_prev: float, x_curr: float, t_prev: float, t_curr: float, line_x: float) -> float:
    span = x_curr - x_prev
    if span == 0:
        return t_curr
    return t_prev + (line_x - x_prev) / span * (t_curr - t_prev)


class SpeedTracker:
    def __init__(
        self,
        line_a_x: int,
        line_b_x: int,
        line_distance_m_north: float,
        line_distance_m_south: float,
        max_track_age_s: float = 5.0,
    ) -> None:
        if line_a_x >= line_b_x:
            raise ValueError("line_a_x must be left of line_b_x")
        self._line_a = line_a_x
        self._line_b = line_b_x
        self._dist_n = line_distance_m_north
        self._dist_s = line_distance_m_south
        self._max_age = max_track_age_s
        self._state: dict[int, _TrackState] = {}

    def update(self, tracks: list[Track], t: float) -> list[SpeedEvent]:
        events: list[SpeedEvent] = []
        seen_ids: set[int] = set()

        for tr in tracks:
            seen_ids.add(tr.track_id)
            st = self._state.setdefault(tr.track_id, _TrackState())
            x = tr.ground_point[0]

            if st.last_x is not None and st.last_t is not None:
                ev = self._check_cross(st, x, t, tr)
                if ev is not None:
                    events.append(ev)
                    del self._state[tr.track_id]
                    continue

            st.last_x = x
            st.last_t = t
            st.last_bbox = tr.bbox
            st.last_cls = tr.cls_name
            st.last_seen = t

        self._gc(t, seen_ids)
        return events

    def _check_cross(
        self, st: _TrackState, x: float, t: float, tr: Track
    ) -> SpeedEvent | None:
        x_prev = st.last_x
        t_prev = st.last_t

        # Line A crossing
        if st.t_cross_a is None and (x_prev - self._line_a) * (x - self._line_a) <= 0 and x_prev != self._line_a:
            st.t_cross_a = _interp_cross(x_prev, x, t_prev, t, self._line_a)
        # Line B crossing
        if st.t_cross_b is None and (x_prev - self._line_b) * (x - self._line_b) <= 0 and x_prev != self._line_b:
            st.t_cross_b = _interp_cross(x_prev, x, t_prev, t, self._line_b)

        if st.t_cross_a is None or st.t_cross_b is None:
            return None

        dt = abs(st.t_cross_b - st.t_cross_a)
        if dt <= 0:
            log.debug("track %d: zero dt at crossing, skipping", tr.track_id)
            return None

        northbound = st.t_cross_a < st.t_cross_b
        direction = "N" if northbound else "S"
        distance_m = self._dist_n if northbound else self._dist_s
        if distance_m <= 0:
            log.warning(
                "track %d: no calibration distance for direction %s; emitting 0 mph",
                tr.track_id, direction,
            )
            speed_mph = 0.0
        else:
            speed_mph = (distance_m / dt) * MPS_TO_MPH

        return SpeedEvent(
            track_id=tr.track_id,
            cls_name=tr.cls_name,
            direction=direction,
            speed_mph=speed_mph,
            t_a=st.t_cross_a,
            t_b=st.t_cross_b,
            bbox=tr.bbox,
        )

    def _gc(self, t_now: float, seen_ids: set[int]) -> None:
        stale = [
            tid for tid, st in self._state.items()
            if tid not in seen_ids and (t_now - st.last_seen) > self._max_age
        ]
        for tid in stale:
            del self._state[tid]
