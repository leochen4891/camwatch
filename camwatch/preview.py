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
        # Grid overlay (settable independently of configure())
        self._show_grid: bool = False
        # Pre-computed grid polylines in pre-resize pixel coords; recomputed
        # whenever the homography or grid bounds change. None disables drawing.
        self._grid_polylines: list[np.ndarray] | None = None

    def configure(
        self,
        roi: tuple[int, int, int, int] | None,
        line_a_x: int = 0,  # noqa: ARG002 (kept for caller compat; unused)
        line_b_x: int = 0,  # noqa: ARG002
    ) -> None:
        self._roi = roi

    def set_grid(
        self,
        homography: Any | None,
        grid_x_min: float, grid_x_max: float,
        grid_y_min: float, grid_y_max: float,
    ) -> None:
        """Pre-project the grid corners + inner lines into pixel space so the
        per-frame _render path can draw them with cheap polylines instead of
        re-projecting every frame. Pass homography=None to disable the overlay
        entirely (e.g., if calibration is missing)."""
        if homography is None:
            self._grid_polylines = None
            return
        inv = homography.inv_H

        def to_px(X: float, Y: float) -> tuple[int, int]:
            p = inv @ np.array([X, Y, 1.0])
            return int(round(p[0] / p[2])), int(round(p[1] / p[2]))

        polylines: list[np.ndarray] = []
        # Outer rectangle
        outer = [
            to_px(grid_x_min, grid_y_min),
            to_px(grid_x_max, grid_y_min),
            to_px(grid_x_max, grid_y_max),
            to_px(grid_x_min, grid_y_max),
            to_px(grid_x_min, grid_y_min),
        ]
        polylines.append(np.array(outer, dtype=np.int32))
        # Inner lines every 5 ft (1.524 m) — ladder of road-perpendicular lines
        # along Y from grid_y_min..grid_y_max, plus road-parallel lines along
        # X. Lets the user eyeball perspective and see when a vehicle's red
        # ground-point lands in which cell.
        step = 5.0 * 0.3048
        y = grid_y_min + step
        while y < grid_y_max - 1e-6:
            polylines.append(np.array([to_px(grid_x_min, y), to_px(grid_x_max, y)], dtype=np.int32))
            y += step
        x = grid_x_min + step
        while x < grid_x_max - 1e-6:
            polylines.append(np.array([to_px(x, grid_y_min), to_px(x, grid_y_max)], dtype=np.int32))
            x += step
        self._grid_polylines = polylines

    def set_show_grid(self, show: bool) -> None:
        self._show_grid = bool(show)

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

        # Calibrated measurement grid (yellow outer rectangle, dim inner
        # 5-ft cells). Drawn first so detection bboxes render on top.
        if self._show_grid and self._grid_polylines:
            outer_color = (0, 255, 255)   # yellow
            inner_color = (60, 140, 140)  # dim yellow
            for i, pl in enumerate(self._grid_polylines):
                pts = (pl.astype(np.float64) * scale).astype(np.int32)
                color = outer_color if i == 0 else inner_color
                thickness = 2 if i == 0 else 1
                cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)

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
