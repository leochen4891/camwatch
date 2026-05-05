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
        recorder = ClipRecorder(self._recordings_dir)
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
                    clip_path = recorder.trigger(
                        name=clip_name,
                        focus_track_id=ev.track_id,
                        line_a_x=cal.line_a_x,
                        line_b_x=cal.line_b_x,
                        t_a=ev.t_a,
                        t_b=ev.t_b,
                        speed_mph=speed_mph,
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
            log.info("capture worker stopped")
