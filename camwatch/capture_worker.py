"""Always-on capture thread for the web UI.

Two-stream architecture (when path_sub is configured):
  - Sub stream (typically 640x480 @ 10fps): drives YOLO detection, BotSORT
    tracking, line-crossing math, and the live web preview. Low-res keeps
    the pipeline real-time so cars are seen as they happen.
  - Main stream (typically 2560x1920 @ 20fps): a LatestFrameSource holds
    the most recent high-res frame; the recorder's clip ring is fed from
    this so saved clips and thumbnails preserve full detail.

All calibration (line_a_x, line_b_x, ROI rectangle) is stored in main-stream
pixel coordinates. The worker derives sub-stream coords on startup by
dividing by the constant scale (= main_width / sub_width, almost always 4
for Reolink cameras). Detector + crossing run in sub coords; bboxes are
then scaled UP by the same factor before being pushed to the recorder so
clip overlays land on the correct part of the high-res frame.

If `path_sub` is not set, the worker falls back to single-stream mode and
behaves like the original implementation.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

from .capture import LatestFrameSource, RtspStream
from .config import Config
from .crossing import CrossingDetector
from .db import Database
from .detect import Detector, Track
from .preview import PreviewBuffer
from .recorder import ClipRecorder

log = logging.getLogger(__name__)

# Reolink E1 main is 2560x1920, sub is 640x480 — exactly 4x. Hard-coded
# rather than probed because re-probing every restart costs a stream
# open/close, and the camera's offering doesn't change.
_SUB_TO_MAIN_SCALE = 4.0


def _scale_track_up(t: Track, scale: float) -> Track:
    x1, y1, x2, y2 = t.bbox
    gx, gy = t.ground_point
    return Track(
        track_id=t.track_id,
        cls_idx=t.cls_idx,
        cls_name=t.cls_name,
        bbox=(x1 * scale, y1 * scale, x2 * scale, y2 * scale),
        conf=t.conf,
        ground_point=(gx * scale, gy * scale),
    )


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        cfg: Config,
        db: Database,
        recordings_dir: Path = Path("recordings"),
        preview: PreviewBuffer | None = None,
    ) -> None:
        super().__init__(name="capture-worker", daemon=True)
        self._cfg = cfg
        self._db = db
        self._recordings_dir = Path(recordings_dir)
        self._preview = preview
        self._stop_evt = threading.Event()
        self._sub_stream: RtspStream | None = None
        self._main_source: LatestFrameSource | None = None
        self._error: BaseException | None = None

    def stop(self) -> None:
        self._stop_evt.set()
        if self._sub_stream is not None:
            self._sub_stream.stop()
        if self._main_source is not None:
            self._main_source.stop()

    def run(self) -> None:
        try:
            self._run()
        except BaseException as e:  # noqa: BLE001
            log.exception("capture worker crashed: %s", e)
            self._error = e

    def _run(self) -> None:
        cal = self._cfg.load_calibration()
        if cal is None:
            log.warning(
                "calibration.yaml not found; capture worker idle. "
                "Run `python -m camwatch.calibrate pick-lines` first."
            )
            return
        if cal.line_a_x >= cal.line_b_x:
            log.warning("invalid line positions in calibration.yaml; capture worker idle")
            return

        sub_url = self._cfg.camera.rtsp_url_sub
        main_url = self._cfg.camera.rtsp_url
        dual = sub_url is not None
        if dual:
            scale = _SUB_TO_MAIN_SCALE
            line_a_sub = int(round(cal.line_a_x / scale))
            line_b_sub = int(round(cal.line_b_x / scale))
            roi_sub: tuple[int, int, int, int] | None = None
            if cal.roi is not None:
                rx1, ry1, rx2, ry2 = cal.roi
                roi_sub = (
                    int(round(rx1 / scale)),
                    int(round(ry1 / scale)),
                    int(round(rx2 / scale)),
                    int(round(ry2 / scale)),
                )
            log.info(
                "capture worker (dual-stream) starting: detection on sub %s, "
                "clips from main %s. lines main=(a=%d, b=%d) sub=(a=%d, b=%d). roi=%s",
                sub_url, main_url, cal.line_a_x, cal.line_b_x,
                line_a_sub, line_b_sub, cal.roi,
            )
            det_url = sub_url
            det_line_a = line_a_sub
            det_line_b = line_b_sub
            det_roi = roi_sub
        else:
            scale = 1.0
            det_url = main_url
            det_line_a = cal.line_a_x
            det_line_b = cal.line_b_x
            det_roi = cal.roi
            log.info(
                "capture worker (single-stream) starting: lines a=%d b=%d roi=%s threshold=%.1f mph",
                cal.line_a_x, cal.line_b_x, cal.roi, self._cfg.alert_threshold_mph,
            )

        det = Detector(
            weights=self._cfg.model.weights,
            device=self._cfg.model.device,
            classes=self._cfg.model.classes,
            conf=self._cfg.model.conf,
            iou=self._cfg.model.iou,
            roi=det_roi,
        )
        recorder = ClipRecorder(self._recordings_dir)
        # Crossing detector operates in main coords (since lines are in main
        # coords and the bboxes we feed it have been scaled up).
        crossing = CrossingDetector(
            line_a_x=cal.line_a_x,
            line_b_x=cal.line_b_x,
            max_track_age_s=self._cfg.max_track_age_s,
        )
        if self._preview is not None:
            self._preview.configure(det_roi if dual else cal.roi, det_line_a, det_line_b)

        self._sub_stream = RtspStream(det_url)
        if dual:
            self._main_source = LatestFrameSource(main_url, name="main-stream")
            self._main_source.start()

        last_main_ts = -1.0

        try:
            for fr_det in self._sub_stream.frames():
                if self._stop_evt.is_set():
                    break
                tracks_det = det.track(fr_det.image)

                if dual and self._main_source is not None:
                    main_item = self._main_source.get_latest()
                    if main_item is None:
                        # Main stream not warm yet; preview only, skip recorder/crossing.
                        if self._preview is not None:
                            self._preview.update(fr_det.image, tracks_det)
                        continue
                    main_ts, main_frame = main_item
                    tracks_main = [_scale_track_up(t, scale) for t in tracks_det]
                    if main_ts != last_main_ts:
                        recorder.push(main_frame, main_ts, tracks_main)
                        last_main_ts = main_ts
                else:
                    main_ts = fr_det.ts
                    main_frame = fr_det.image
                    tracks_main = tracks_det
                    recorder.push(main_frame, main_ts, tracks_main)

                if self._preview is not None:
                    self._preview.update(fr_det.image, tracks_det)

                events = crossing.update(tracks_main, fr_det.ts)
                for ev in events:
                    captured_at = datetime.now().astimezone()
                    stamp = captured_at.strftime("%Y%m%dT%H%M%S")
                    clip_name = f"cal_{stamp}_id{ev.track_id}_{ev.direction}.mp4"
                    clip_path = recorder.trigger(
                        name=clip_name,
                        focus_track_id=ev.track_id,
                        line_a_x=cal.line_a_x,
                        line_b_x=cal.line_b_x,
                        t_a=ev.t_a,
                        t_b=ev.t_b,
                    )
                    pid = self._db.insert_pass(
                        captured_at=captured_at.isoformat(timespec="seconds"),
                        track_id=ev.track_id,
                        cls_name=ev.cls_name,
                        direction=ev.direction,
                        elapsed_s=ev.elapsed_s,
                        clip_path=clip_path or None,
                    )
                    log.info(
                        "pass id=%d track=%d %s %s elapsed=%.3fs clip=%s",
                        pid, ev.track_id, ev.cls_name, ev.direction,
                        ev.elapsed_s, clip_name,
                    )
        finally:
            recorder.flush()
            if self._main_source is not None:
                self._main_source.stop()
            log.info("capture worker stopped")
