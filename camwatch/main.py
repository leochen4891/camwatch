from __future__ import annotations

import logging
import signal

from .capture import RtspStream
from .config import load_config
from .detect import Detector
from .sink import Sink
from .speed import SpeedTracker

log = logging.getLogger("camwatch")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config()
    cal = cfg.load_calibration()
    if cal is None:
        raise SystemExit(
            f"calibration not found at {cfg.calibration_path}. "
            f"Run `python -m camwatch.calibrate pick-lines` first."
        )
    if cal.line_distance_m_north <= 0 and cal.line_distance_m_south <= 0:
        log.warning(
            "calibration has no per-direction distance set; speeds will be reported as 0 mph. "
            "Run `python -m camwatch.calibrate annotate` to populate."
        )

    cap = RtspStream(cfg.camera.rtsp_url)
    det = Detector(
        weights=cfg.model.weights,
        device=cfg.model.device,
        classes=cfg.model.classes,
        conf=cfg.model.conf,
        iou=cfg.model.iou,
        roi=cal.roi,
    )
    spd = SpeedTracker(
        line_a_x=cal.line_a_x,
        line_b_x=cal.line_b_x,
        line_distance_m_north=cal.line_distance_m_north,
        line_distance_m_south=cal.line_distance_m_south,
        max_track_age_s=cfg.max_track_age_s,
    )
    sink = Sink(cfg.events_dir, threshold_mph=cfg.alert_threshold_mph)

    def handle_signal(signum, _frame):
        log.info("got signal %d, stopping", signum)
        cap.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log.info("camwatch starting (alert threshold %.1f mph)", cfg.alert_threshold_mph)
    for fr in cap.frames():
        tracks = det.track(fr.image)
        for ev in spd.update(tracks, fr.ts):
            sink.write(ev, fr.image)
    log.info("camwatch stopped")
