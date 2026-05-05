"""Always-on capture thread for the web UI.

Reuses the same RtspStream, Detector, ClipRecorder, and CrossingDetector as
the CLI calibrate workflow, but runs without a deadline and writes each pass
into SQLite (camwatch.db) instead of YAML.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

from .capture import RtspStream
from .config import Config
from .crossing import CrossingDetector
from .db import Database
from .detect import Detector
from .recorder import ClipRecorder

log = logging.getLogger(__name__)


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        cfg: Config,
        db: Database,
        recordings_dir: Path = Path("recordings"),
    ) -> None:
        super().__init__(name="capture-worker", daemon=True)
        self._cfg = cfg
        self._db = db
        self._recordings_dir = Path(recordings_dir)
        self._stop_evt = threading.Event()
        self._stream: RtspStream | None = None
        self._error: BaseException | None = None

    def stop(self) -> None:
        self._stop_evt.set()
        if self._stream is not None:
            self._stream.stop()

    def run(self) -> None:  # threading.Thread entry point
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
        self._stream = RtspStream(self._cfg.camera.rtsp_url)
        det = Detector(
            weights=self._cfg.model.weights,
            device=self._cfg.model.device,
            classes=self._cfg.model.classes,
            conf=self._cfg.model.conf,
            iou=self._cfg.model.iou,
            roi=cal.roi,
        )
        recorder = ClipRecorder(self._recordings_dir)
        crossing = CrossingDetector(
            line_a_x=cal.line_a_x,
            line_b_x=cal.line_b_x,
            max_track_age_s=self._cfg.max_track_age_s,
        )

        try:
            for fr in self._stream.frames():
                if self._stop_evt.is_set():
                    break
                tracks = det.track(fr.image)
                recorder.push(fr.image, fr.ts, tracks)
                events = crossing.update(tracks, fr.ts)
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
            log.info("capture worker stopped")
