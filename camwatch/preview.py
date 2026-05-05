"""Live preview frame buffer.

The capture worker pushes one annotated frame per cycle. The FastAPI app
serves the latest as a single JPEG (`GET /preview.jpg`) or as an MJPEG
multipart stream (`GET /preview/stream`).

Frames are downscaled before encoding to keep latency and bandwidth in
check; overlay rendering happens once per push, regardless of how many
viewers are watching (they all consume the same encoded bytes).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


class PreviewBuffer:
    def __init__(self, max_width: int = 960, jpeg_quality: int = 70) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_id = 0
        self._max_width = max_width
        self._quality = jpeg_quality
        # Scene config (set once via configure())
        self._roi: tuple[int, int, int, int] | None = None
        self._line_a_x: int = 0
        self._line_b_x: int = 0

    def configure(
        self,
        roi: tuple[int, int, int, int] | None,
        line_a_x: int,
        line_b_x: int,
    ) -> None:
        self._roi = roi
        self._line_a_x = int(line_a_x)
        self._line_b_x = int(line_b_x)

    def update(self, frame: np.ndarray, tracks: list[Any]) -> None:
        annotated, _ = self._render(frame, tracks)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._frame_id += 1
            self._cond.notify_all()

    def get_latest(self) -> tuple[int, bytes] | None:
        with self._cond:
            if self._jpeg is None:
                return None
            return self._frame_id, self._jpeg

    def wait_for_next(self, since_id: int, timeout: float = 5.0) -> tuple[int, bytes] | None:
        with self._cond:
            self._cond.wait_for(
                lambda: self._frame_id > since_id and self._jpeg is not None,
                timeout=timeout,
            )
            if self._frame_id <= since_id or self._jpeg is None:
                return None
            return self._frame_id, self._jpeg

    def _render(self, frame: np.ndarray, tracks: list[Any]) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if w > self._max_width:
            scale = self._max_width / w
            new_w = self._max_width
            new_h = int(round(h * scale))
            img = cv2.resize(frame, (new_w, new_h))
        else:
            scale = 1.0
            img = frame.copy()

        ih, iw = img.shape[:2]

        # Crossing lines: thin gray, just visible reference markers.
        line_color = (160, 160, 160)
        if self._line_a_x > 0:
            ax = int(round(self._line_a_x * scale))
            cv2.line(img, (ax, 0), (ax, ih), line_color, 1)
        if self._line_b_x > 0:
            bx = int(round(self._line_b_x * scale))
            cv2.line(img, (bx, 0), (bx, ih), line_color, 1)

        # Bboxes + ground points
        for t in tracks:
            tid = getattr(t, "track_id", None)
            bbox = getattr(t, "bbox", None)
            gp = getattr(t, "ground_point", None)
            if bbox is None or gp is None:
                continue
            x1, y1, x2, y2 = (int(round(v * scale)) for v in bbox)
            gx, gy = int(round(gp[0] * scale)), int(round(gp[1] * scale))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if tid is not None:
                cv2.putText(
                    img, f"id={tid}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                )
            cv2.circle(img, (gx, gy), 4, (0, 0, 255), -1)

        return img, scale
