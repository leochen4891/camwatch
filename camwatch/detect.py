"""YOLO + BotSORT tracker wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)


@dataclass
class Track:
    track_id: int
    cls_idx: int
    cls_name: str
    bbox: tuple[float, float, float, float]
    conf: float
    ground_point: tuple[float, float]


class Detector:
    def __init__(
        self,
        weights: str,
        device: str,
        classes: list[int],
        conf: float = 0.35,
        iou: float = 0.5,
        tracker: str = "botsort.yaml",
    ) -> None:
        self._model = YOLO(weights)
        self._device = device
        self._classes = classes
        self._conf = conf
        self._iou = iou
        self._tracker = tracker
        self._names = self._model.names
        log.info(
            "detector: weights=%s device=%s classes=%s tracker=%s",
            weights, device, classes, tracker,
        )

    def track(self, frame: np.ndarray) -> list[Track]:
        results = self._model.track(
            frame,
            persist=True,
            classes=self._classes,
            conf=self._conf,
            iou=self._iou,
            tracker=self._tracker,
            device=self._device,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        cls = boxes.cls.int().cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        out: list[Track] = []
        for i in range(len(ids)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            ground = ((x1 + x2) / 2.0, y2)
            out.append(
                Track(
                    track_id=int(ids[i]),
                    cls_idx=int(cls[i]),
                    cls_name=self._names.get(int(cls[i]), str(int(cls[i]))),
                    bbox=(x1, y1, x2, y2),
                    conf=float(conf[i]),
                    ground_point=ground,
                )
            )
        return out
