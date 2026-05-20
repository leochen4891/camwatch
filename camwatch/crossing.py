"""Crossing event dataclass.

The legacy two-line `CrossingDetector` was removed; the active trigger is
`grid_crossing.GridCrossingDetector`, which still emits this dataclass so
downstream code (recorder, DB writer) stays unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossingEvent:
    track_id: int
    cls_name: str
    direction: str           # "N" if Y increased between entry and exit, else "S"
    t_a: float
    t_b: float
    elapsed_s: float
    bbox: tuple[float, float, float, float] | None = None
