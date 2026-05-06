"""Background thumbnail upgrader.

Upgrades a pass's thumbnail from the sub-stream-derived crop to a
higher-resolution crop pulled from a parallel high-res RTSP stream.
The main stream's ffmpeg buffer can deliver frames seconds late, so we
buffer sampled main frames keyed by their PTS-anchored monotonic ts
(`Frame.ts` from `RtspStream`) and look up by the trigger's t_b/t_a —
both timestamps live in the same monotonic domain because each stream's
PTS is anchored to `time.monotonic()` at its own first frame.

Per pass:

  1. Wait until the buffer holds a frame with ts ≥ trigger ts.
  2. Find the buffered main frame whose ts is closest to the trigger
     (within ±1.5s tolerance).
  3. Run YOLO once on that frame.
  4. Pick the detection of the focus track's class with highest IoU
     against the projected sub-stream bbox; fall back to nearest center.
  5. Crop with padding, atomic-rename the JPEG over the existing
     thumbnail.

If anything fails (no matching frame, no detection of the right class,
write error), the existing sub-stream thumbnail stays untouched.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .capture import TimestampedFrameBuffer
from .config import ModelConfig
from .db import Database
from .detect import Detector

log = logging.getLogger(__name__)


@dataclass
class _Job:
    pass_id: int
    thumb_path: str
    focus_cls_name: str
    sub_bbox: tuple[float, float, float, float]
    sub_frame_size: tuple[int, int]  # (w, h) of the sub-stream frame
    target_ts: float                  # sub-stream PTS-anchored monotonic ts at trigger


class ThumbUpgrader:
    def __init__(
        self,
        rtsp_url: str,
        model: ModelConfig,
        db: Database,
        target_w: int = 320,
        queue_size: int = 8,
    ) -> None:
        self._rtsp_url = rtsp_url
        self._model_cfg = model
        self._db = db
        self._target_w = int(target_w)
        self._queue: queue.Queue[_Job] = queue.Queue(maxsize=queue_size)
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._buffer: TimestampedFrameBuffer | None = None
        self._detector: Detector | None = None  # lazy

    def start(self) -> None:
        if self._thread is not None:
            return
        self._buffer = TimestampedFrameBuffer(
            url=self._rtsp_url,
            max_age_s=15.0,
            sample_interval_s=0.25,
            name="thumb-stream",
        )
        self._buffer.start()
        self._thread = threading.Thread(target=self._run, name="thumb-upgrader", daemon=True)
        self._thread.start()
        log.info("thumb upgrader started (pure-PTS lookup)")

    def stop(self) -> None:
        self._stop_evt.set()
        if self._buffer is not None:
            self._buffer.stop()
            self._buffer = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def enqueue(
        self,
        pass_id: int,
        thumb_path: str,
        focus_cls_name: str,
        sub_bbox: tuple[float, float, float, float],
        sub_frame_size: tuple[int, int],
        target_ts: float,
    ) -> None:
        job = _Job(
            pass_id=pass_id,
            thumb_path=thumb_path,
            focus_cls_name=focus_cls_name,
            sub_bbox=sub_bbox,
            sub_frame_size=sub_frame_size,
            target_ts=target_ts,
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                log.warning("thumb upgrader queue full; dropping %s", thumb_path)

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(job)
            except Exception as e:  # noqa: BLE001
                log.warning("thumb upgrade failed for %s: %s", job.thumb_path, e)
                try:
                    self._db.set_thumb_upgrade_status(job.pass_id, "failed")
                except Exception:  # noqa: BLE001
                    pass
        log.info("thumb upgrader stopped")

    def _process(self, job: _Job) -> None:
        assert self._buffer is not None
        thumb_name = Path(job.thumb_path).name
        # The main stream lags the sub stream by ffmpeg buffer depth. Wait
        # for a buffered frame whose ts has reached the trigger before
        # looking up.
        self._wait_until_matching_frame_available(job.target_ts, timeout_s=45.0)
        match = self._buffer.find_frame_at(job.target_ts, tolerance_s=2.5)
        if match is None:
            latest = self._buffer.latest_indexed()
            latest_str = f"{latest[0]:.3f}" if latest else "n/a"
            log.info(
                "thumb upgrade: no main frame matching ts=%.3f (latest=%s) — %s",
                job.target_ts, latest_str, thumb_name,
            )
            self._db.set_thumb_upgrade_status(job.pass_id, "failed")
            return
        matched_ts, frame = match
        h, w = frame.shape[:2]
        log.info(
            "thumb upgrade: matched main frame ts=%.3f for trigger ts=%.3f (Δ=%+.3fs %dx%d) — %s",
            matched_ts, job.target_ts, matched_ts - job.target_ts, w, h, thumb_name,
        )

        if self._detector is None:
            self._detector = Detector(
                weights=self._model_cfg.weights,
                device=self._model_cfg.device,
                classes=self._model_cfg.classes,
                conf=self._model_cfg.conf,
                iou=self._model_cfg.iou,
                roi=None,
            )
        detections = self._detector.detect(frame)
        if not detections:
            log.info("thumb upgrade: 0 detections in matched main frame — %s", thumb_name)
            # Forensic dump so we can see what the matched main frame looked like.
            try:
                dump_path = Path(job.thumb_path).with_name(
                    Path(job.thumb_path).stem + "_main_dbg.jpg"
                )
                cv2.imwrite(str(dump_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            except Exception:  # noqa: BLE001
                pass
            self._db.set_thumb_upgrade_status(job.pass_id, "failed")
            return

        sub_w, sub_h = job.sub_frame_size
        sx = w / max(1, sub_w)
        sy = h / max(1, sub_h)
        proj = (
            job.sub_bbox[0] * sx,
            job.sub_bbox[1] * sy,
            job.sub_bbox[2] * sx,
            job.sub_bbox[3] * sy,
        )

        same_class = [d for d in detections if d.cls_name == job.focus_cls_name]
        candidates = same_class or detections

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
            area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        scored = [(iou(proj, d.bbox), d) for d in candidates]
        scored.sort(key=lambda t: t[0], reverse=True)
        best_iou, best = scored[0]
        if best_iou == 0.0:
            pcx = (proj[0] + proj[2]) / 2.0
            pcy = (proj[1] + proj[3]) / 2.0
            def dist(d):
                cx = (d.bbox[0] + d.bbox[2]) / 2.0
                cy = (d.bbox[1] + d.bbox[3]) / 2.0
                return (cx - pcx) ** 2 + (cy - pcy) ** 2
            best = min(candidates, key=dist)

        thumb = self._crop(frame, best.bbox, target_w=self._target_w)
        big = self._crop(frame, best.bbox, target_w=1280)
        if thumb is None:
            log.info("thumb upgrade: crop too small, skipping %s", thumb_name)
            self._db.set_thumb_upgrade_status(job.pass_id, "failed")
            return
        # Atomic write: encoder is selected from the extension, so the temp
        # path must keep the .jpg suffix. Insert ".tmp" before the suffix
        # rather than appending it.
        out_path = Path(job.thumb_path)
        tmp_path = out_path.with_name(out_path.stem + ".tmp" + out_path.suffix)
        cv2.imwrite(str(tmp_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 82])
        tmp_path.replace(out_path)
        if big is not None:
            big_path = out_path.with_name(out_path.stem + "_big.jpg")
            big_tmp = big_path.with_name(big_path.stem + ".tmp" + big_path.suffix)
            cv2.imwrite(str(big_tmp), big, [cv2.IMWRITE_JPEG_QUALITY, 88])
            big_tmp.replace(big_path)
        log.info(
            "thumb upgraded: %s (iou=%.2f cls=%s)",
            thumb_name, best_iou, best.cls_name,
        )
        self._db.set_thumb_upgrade_status(job.pass_id, "ok")

    def _wait_until_matching_frame_available(
        self, target_ts: float, timeout_s: float
    ) -> bool:
        """Block (with polling) until the buffer holds a frame whose ts is
        at or past `target_ts`, or until the timeout. Returns True if a
        candidate has been indexed within the window."""
        import time as _t
        deadline = _t.monotonic() + timeout_s
        while not self._stop_evt.is_set() and _t.monotonic() < deadline:
            if self._buffer is None:
                return False
            latest = self._buffer.latest_indexed()
            if latest is not None and latest[0] >= target_ts:
                return True
            _t.sleep(0.25)
        return False

    def _crop(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        target_w: int,
    ) -> np.ndarray | None:
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = bbox
        bw = bx2 - bx1
        bh = by2 - by1
        pad_x = max(bw * 0.6, 40)
        pad_y = max(bh * 0.7, 40)
        cx1 = max(0, int(round(bx1 - pad_x)))
        cy1 = max(0, int(round(by1 - pad_y)))
        cx2 = min(w, int(round(bx2 + pad_x)))
        cy2 = min(h, int(round(by2 + pad_y)))
        if cx2 - cx1 < 80 or cy2 - cy1 < 60:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]
        # Don't UPSCALE past the source crop's native size; that just blurs.
        if cw > target_w:
            scale = target_w / cw
            crop = cv2.resize(crop, (target_w, max(1, int(round(ch * scale)))))
        return crop
