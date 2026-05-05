"""Event sink: append JSONL log + save annotated snapshot on alert."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .speed import SpeedEvent

log = logging.getLogger(__name__)


class Sink:
    def __init__(self, events_dir: Path, threshold_mph: float) -> None:
        self._dir = events_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self._dir / "events.jsonl"
        self._threshold = threshold_mph

    def write(self, ev: SpeedEvent, frame: np.ndarray) -> None:
        ts = datetime.now().astimezone()
        alert = ev.speed_mph >= self._threshold
        snapshot_path: str | None = None
        if alert:
            snapshot_path = self._save_snapshot(ev, frame, ts)

        record = {
            "ts": ts.isoformat(timespec="seconds"),
            "track_id": ev.track_id,
            "class": ev.cls_name,
            "direction": ev.direction,
            "speed_mph": round(ev.speed_mph, 1),
            "alert": alert,
            "snapshot": snapshot_path,
        }
        with self._jsonl.open("a") as f:
            f.write(json.dumps(record) + "\n")
        log.info(
            "event: id=%d %s %s %.1f mph%s",
            ev.track_id, ev.cls_name, ev.direction, ev.speed_mph,
            " ALERT" if alert else "",
        )

    def _save_snapshot(self, ev: SpeedEvent, frame: np.ndarray, ts: datetime) -> str:
        x1, y1, x2, y2 = (int(v) for v in ev.bbox)
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"{ev.cls_name} {ev.direction} {ev.speed_mph:.1f} mph"
        cv2.putText(
            annotated, label, (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )

        stamp = ts.strftime("%Y-%m-%dT%H-%M-%S")
        name = f"{stamp}_id{ev.track_id}_{ev.direction}_{int(round(ev.speed_mph))}mph.jpg"
        path = self._dir / name
        cv2.imwrite(str(path), annotated)
        return str(path)
