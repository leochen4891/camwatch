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
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._model = YOLO(weights)
        self._device = device
        self._classes = classes
        self._conf = conf
        self._iou = iou
        self._tracker = tracker
        self._roi = roi
        self._names = self._model.names
        log.info(
            "detector: weights=%s device=%s classes=%s tracker=%s roi=%s",
            weights, device, classes, tracker, roi,
        )

    def track(self, frame: np.ndarray) -> list[Track]:
        # Crop to ROI if configured. YOLO sees the smaller window; bbox coords
        # are translated back to full-frame space before returning, so all
        # downstream code (crossing detection, clip overlay) is unaware.
        if self._roi is not None:
            x1, y1, x2, y2 = self._roi
            view = frame[y1:y2, x1:x2]
            ox, oy = x1, y1
        else:
            view = frame
            ox, oy = 0, 0

        results = self._model.track(
            view,
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
            cx1, cy1, cx2, cy2 = (float(v) for v in xyxy[i])
            x1 = cx1 + ox
            y1 = cy1 + oy
            x2 = cx2 + ox
            y2 = cy2 + oy
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
