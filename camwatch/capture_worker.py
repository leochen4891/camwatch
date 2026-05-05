"""Always-on capture thread for the web UI.

Single-stream mode: detection, tracking, crossings, preview, and clip
recording all run from the configured RTSP path. The simplest design that
works.

If the camera is configured to use the sub stream (typically 640x480 at
10fps), YOLO sees the full frame and detection is fast and aligned. The
calibration ROI is applied as a post-detection filter on each track's
ground point rather than as a pre-YOLO crop, because cropping a thin road
belt and letterboxing it to 640x640 squashes cars to a few pixels tall
and produces phantom detections on road texture.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from .capture import RtspStream
from .config import Config
from .crossing import CrossingDetector
from .db import Database
from .detect import Detector, Track
from .preview import PreviewBuffer
from .recorder import ClipRecorder

log = logging.getLogger(__name__)


def _track_in_roi(t: Track, roi: tuple[int, int, int, int] | None) -> bool:
    """Keep tracks whose ground_point falls inside the ROI rectangle."""
    if roi is None:
        return True
    x1, y1, x2, y2 = roi
    gx, gy = t.ground_point
    return x1 <= gx <= x2 and y1 <= gy <= y2


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
        self._stream: RtspStream | None = None
        self._error: BaseException | None = None
        self._recorder: ClipRecorder | None = None

    def update_clip_margin(self, seconds: float) -> None:
        """Apply a new pre/post-roll value to the running recorder, if any.

        Floats are atomically read/written in CPython, so we don't need a lock —
        the recorder thread will see the new value on its next trigger() call.
        """
        s = max(0.0, float(seconds))
        if self._recorder is not None:
            self._recorder._pre_a = s
            self._recorder._post_b = s

    def stop(self) -> None:
        self._stop_evt.set()
        if self._stream is not None:
            self._stream.stop()

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

        log.info(
            "capture worker starting (lines a=%d b=%d, threshold=%.1f mph, roi=%s)",
            cal.line_a_x, cal.line_b_x, self._cfg.alert_threshold_mph, cal.roi,
        )

        # YOLO sees the full frame; ROI is enforced as a post-detection filter
        # on each track's ground point. See module docstring for why.
        det = Detector(
            weights=self._cfg.model.weights,
            device=self._cfg.model.device,
            classes=self._cfg.model.classes,
            conf=self._cfg.model.conf,
            iou=self._cfg.model.iou,
            roi=None,
        )
        recorder = ClipRecorder(
            self._recordings_dir,
            pre_seconds_before_a=self._cfg.clip_margin_s,
            post_seconds_after_b=self._cfg.clip_margin_s,
        )
        self._recorder = recorder
        crossing = CrossingDetector(
            line_a_x=cal.line_a_x,
            line_b_x=cal.line_b_x,
            max_track_age_s=self._cfg.max_track_age_s,
        )
        if self._preview is not None:
            self._preview.configure(cal.roi, cal.line_a_x, cal.line_b_x)

        self._stream = RtspStream(self._cfg.camera.rtsp_url)
        last_purge = time.monotonic()
        purge_interval_s = 3600.0  # check retention once an hour

        try:
            for fr in self._stream.frames():
                if self._stop_evt.is_set():
                    break

                # Periodic retention sweep.
                if time.monotonic() - last_purge > purge_interval_s:
                    last_purge = time.monotonic()
                    days = int(self._cfg.retention_days or 0)
                    if days > 0:
                        n, clips = self._db.purge_older_than(days)
                        for cp in clips:
                            try:
                                Path(cp).unlink(missing_ok=True)
                                Path(cp[:-4] + ".jpg").unlink(missing_ok=True)
                            except Exception as e:  # noqa: BLE001
                                log.debug("retention: %s: %s", cp, e)
                        if n:
                            log.info("retention: purged %d passes older than %d days", n, days)

                all_tracks = det.track(fr.image)
                tracks = [t for t in all_tracks if _track_in_roi(t, cal.roi)]

                recorder.push(fr.image, fr.ts, tracks)
                if self._preview is not None:
                    # Preview gets ALL detections (including outside ROI) so
                    # the user can see what YOLO is doing; the ROI rectangle
                    # is drawn for visual context.
                    self._preview.update(fr.image, all_tracks)

                events = crossing.update(tracks, fr.ts)
                for ev in events:
                    captured_at = datetime.now().astimezone()
                    stamp = captured_at.strftime("%Y%m%dT%H%M%S")
                    clip_name = f"cal_{stamp}_id{ev.track_id}_{ev.direction}.mp4"
                    distance = (
                        cal.line_distance_m_north if ev.direction == "N"
                        else cal.line_distance_m_south
                    )
                    speed_mph: float | None = None
                    if distance > 0 and ev.elapsed_s > 0:
                        speed_mph = (distance / ev.elapsed_s) * 2.2369362920544
                    # Gate VIDEO recording on the configured capture-speed range.
                    # Thumbnails are always written so the list shows a preview
                    # for every pass. If speed is unknown (no calibration), we
                    # err on the side of recording.
                    in_range = (
                        speed_mph is None
                        or (
                            self._cfg.clip_capture_min_mph
                            <= speed_mph
                            <= self._cfg.clip_capture_max_mph
                        )
                    )
                    if not in_range:
                        log.info(
                            "pass at %.1f mph outside capture range [%.1f, %.1f]; thumb only",
                            speed_mph,
                            self._cfg.clip_capture_min_mph,
                            self._cfg.clip_capture_max_mph,
                        )
                    clip_path = recorder.trigger(
                        name=clip_name,
                        focus_track_id=ev.track_id,
                        line_a_x=cal.line_a_x,
                        line_b_x=cal.line_b_x,
                        t_a=ev.t_a,
                        t_b=ev.t_b,
                        speed_mph=speed_mph,
                        record_video=in_range,
                    )
                    # We always set clip_path (the .mp4 base name) so the
                    # thumbnail can be located by stripping ".mp4" → ".jpg".
                    # Whether the .mp4 actually exists on disk is the source
                    # of truth for "has clip" in the UI.
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
            log.info("capture worker stopped")
