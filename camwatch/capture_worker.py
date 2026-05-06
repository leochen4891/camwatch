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
from .digit_matcher import DigitMatcher
from .preview import PreviewBuffer
from .recorder import ClipRecorder
from .thumb_upgrader import ThumbUpgrader

# OSD pixel rectangle on the Reolink E1 sub stream (640x480) with the OSD
# at the bottom of the frame. Used only for the one-time per-epoch sub-
# stream drift calibration.
_OSD_REGION_SUB = (175, 452, 500, 477)

log = logging.getLogger(__name__)


def _track_in_roi(t: Track, roi: tuple[int, int, int, int] | None) -> bool:
    """Keep tracks whose ground_point falls inside the ROI rectangle."""
    if roi is None:
        return True
    x1, y1, x2, y2 = roi
    gx, gy = t.ground_point
    return x1 <= gx <= x2 and y1 <= gy <= y2


class _StageTimer:
    """Per-stage runtime accumulator for the capture loop.

    Why not cProfile: we already know the structure of the loop (YOLO →
    filter → recorder → preview → crossing), so what we want is a direct
    answer to "which stage's wall-clock time dominates?". cProfile gives
    function-level breakdowns that take post-hoc analysis to interpret;
    stage timing prints the answer in one log line.

    Records `time.perf_counter()` deltas under named buckets and emits a
    p50/p95/mean/max summary every `log_interval_s` seconds. Cleared
    after each emission so each window is independent.
    """

    def __init__(self, log_interval_s: float = 30.0) -> None:
        self._samples: dict[str, list[float]] = {}
        self._interval = log_interval_s
        self._next_log = time.monotonic() + log_interval_s

    def record(self, name: str, dt_s: float) -> None:
        self._samples.setdefault(name, []).append(dt_s)

    def maybe_log(self) -> None:
        now = time.monotonic()
        if now < self._next_log:
            return
        self._next_log = now + self._interval
        if not self._samples:
            return
        rows: list[str] = []
        for name in sorted(self._samples):
            xs = sorted(self._samples[name])
            n = len(xs)
            p50 = xs[n // 2] * 1000.0
            p95 = xs[min(n - 1, int(n * 0.95))] * 1000.0
            mean = (sum(xs) / n) * 1000.0
            mx = xs[-1] * 1000.0
            rows.append(
                f"{name:18s} n={n:4d}  "
                f"p50={p50:6.1f}ms  p95={p95:6.1f}ms  "
                f"mean={mean:6.1f}ms  max={mx:6.1f}ms"
            )
        log.info("PROFILE (last %.0fs):\n  %s", self._interval, "\n  ".join(rows))
        self._samples.clear()


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        cfg: Config,
        db: Database,
        recordings_dir: Path = Path("recordings"),
        preview: PreviewBuffer | None = None,
        profile: bool = False,
    ) -> None:
        super().__init__(name="capture-worker", daemon=True)
        self._cfg = cfg
        self._db = db
        self._recordings_dir = Path(recordings_dir)
        self._preview = preview
        self._profile = bool(profile)
        self._stop_evt = threading.Event()
        self._stream: RtspStream | None = None
        self._error: BaseException | None = None
        self._recorder: ClipRecorder | None = None
        self._upgrader: ThumbUpgrader | None = None
        # Sub-stream drift calibration state. drift_sub = sub_ts(T) - wallclock_unix(T)
        # for any camera-instant T; constant within an RTSP session, but
        # invalidates on reconnect. We learn it by OCR'ing live sub frames
        # until we observe an OSD second-tick (consecutive frames with
        # different OSD seconds), at which point the tick's camera-instant
        # is known to ~half-a-frame-interval precision (~33ms at 15fps).
        self._drift_sub: float | None = None
        self._drift_sub_epoch: int = -1
        # Sliding history of recent (sub_ts, OSD_dt) for tick detection.
        # Cleared once drift_sub is established for the current epoch.
        self._sub_ocr_history: list[tuple[float, datetime]] = []
        try:
            self._sub_matcher: DigitMatcher | None = DigitMatcher("templates/sub")
        except (FileNotFoundError, ValueError) as e:
            log.warning(
                "sub-stream DigitMatcher disabled (%s); drift_sub will not "
                "be calibrated and the upgrader's offset will retain its "
                "datetime.now()-based pipeline-lag bias", e,
            )
            self._sub_matcher = None

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
        if self._upgrader is not None:
            self._upgrader.stop()

    def _maybe_calibrate_sub_drift(self, fr) -> None:
        """OCR the current sub-stream frame's OSD; if we observe an OSD
        second-tick relative to the prior OCR'd frame, derive `drift_sub`
        from the tick midpoint and cache it for this epoch.

        The history list is reset on epoch change (RTSP reconnect) since
        the new session has a fresh PTS anchor."""
        if self._drift_sub_epoch != fr.epoch:
            self._sub_ocr_history = []
            self._drift_sub_epoch = fr.epoch
            self._drift_sub = None
            self._sub_calibration_attempts = 0
            self._sub_calibration_failures = 0
        assert self._sub_matcher is not None  # checked by caller
        self._sub_calibration_attempts = getattr(
            self, "_sub_calibration_attempts", 0
        ) + 1
        dt = self._sub_matcher.read_timestamp(fr.image, _OSD_REGION_SUB)
        if dt is None:
            self._sub_calibration_failures = getattr(
                self, "_sub_calibration_failures", 0
            ) + 1
            # Log rate periodically so we know if OCR is broken on this stream.
            if self._sub_calibration_attempts in (10, 30, 100, 300):
                log.info(
                    "sub-stream OCR: %d/%d failures so far during drift "
                    "calibration on epoch %d",
                    self._sub_calibration_failures,
                    self._sub_calibration_attempts, fr.epoch,
                )
            return
        # Sanity: reject any datetime far from now (template matcher does
        # occasionally pick up junk on heavily occluded crops).
        from datetime import datetime as _dt, timedelta as _td
        if abs(dt - _dt.now()) > _td(seconds=60):
            return
        self._sub_ocr_history.append((fr.ts, dt))
        # Keep memory bounded; we only need the most recent ~30 samples.
        if len(self._sub_ocr_history) > 30:
            self._sub_ocr_history.pop(0)
        if len(self._sub_ocr_history) < 2:
            return
        ts_old, dt_old = self._sub_ocr_history[-2]
        ts_new, dt_new = self._sub_ocr_history[-1]
        delta_seconds = (dt_new - dt_old).total_seconds()
        if delta_seconds != 1.0:
            return
        tick_sub_ts = (ts_old + ts_new) / 2.0
        tick_wallclock_unix = dt_new.timestamp()
        self._drift_sub = tick_sub_ts - tick_wallclock_unix
        self._sub_ocr_history = []  # done; free the memory
        log.info(
            "sub-stream drift_sub=%+.3fs (epoch %d) from tick %s→%s "
            "(ts_old=%.3f ts_new=%.3f gap=%.3fs)",
            self._drift_sub, fr.epoch,
            dt_old.strftime("%H:%M:%S"), dt_new.strftime("%H:%M:%S"),
            ts_old, ts_new, ts_new - ts_old,
        )

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

        # Optional: parallel high-res stream for thumbnail upgrades.
        # Indexing is purely PTS-anchored monotonic now; no OCR region needed.
        thumb_url = self._cfg.camera.rtsp_url_thumb
        if thumb_url:
            self._upgrader = ThumbUpgrader(
                rtsp_url=thumb_url,
                model=self._cfg.model,
                db=self._db,
            )
            self._upgrader.start()

        self._stream = RtspStream(self._cfg.camera.rtsp_url)
        last_purge = time.monotonic()
        purge_interval_s = 3600.0  # check retention once an hour
        prof = _StageTimer() if self._profile else None
        if prof is not None:
            log.info("capture worker: --profile enabled, logging stage timings every 30s")
        last_loop_t: float | None = None

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

                loop_t = time.perf_counter() if prof else 0.0
                if prof and last_loop_t is not None:
                    # Wall-clock between successive frames reaching this point.
                    # Lower-bounded by frame interval; if our work exceeds it,
                    # this gap reflects the real consumer rate.
                    prof.record("interframe_gap", loop_t - last_loop_t)
                last_loop_t = loop_t

                # Sub-stream drift calibration (runs only when drift_sub is
                # missing for the current epoch; once we observe an OSD-tick
                # we cache the result and stop OCR'ing). OCR cost: ~10-20ms
                # per frame via DigitMatcher; only active during the few
                # frames it takes to span an OSD-tick (~1-2s after each
                # session start).
                if self._sub_matcher is not None and (
                    self._drift_sub is None or self._drift_sub_epoch != fr.epoch
                ):
                    self._maybe_calibrate_sub_drift(fr)

                t0 = time.perf_counter() if prof else 0.0
                all_tracks = det.track(fr.image)
                if prof:
                    prof.record("yolo_track", time.perf_counter() - t0)

                t0 = time.perf_counter() if prof else 0.0
                tracks = [t for t in all_tracks if _track_in_roi(t, cal.roi)]
                if prof:
                    prof.record("roi_filter", time.perf_counter() - t0)

                t0 = time.perf_counter() if prof else 0.0
                recorder.push(fr.image, fr.ts, tracks)
                if prof:
                    prof.record("recorder_push", time.perf_counter() - t0)

                if self._preview is not None:
                    # Preview gets ALL detections (including outside ROI) so
                    # the user can see what YOLO is doing; the ROI rectangle
                    # is drawn for visual context.
                    t0 = time.perf_counter() if prof else 0.0
                    self._preview.update(fr.image, all_tracks)
                    if prof:
                        prof.record("preview_update", time.perf_counter() - t0)

                t0 = time.perf_counter() if prof else 0.0
                events = crossing.update(tracks, fr.ts)
                if prof:
                    prof.record("crossing_update", time.perf_counter() - t0)
                if prof:
                    prof.maybe_log()
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
                    # Capture sub-stream context for the optional thumbnail
                    # upgrade. Done before trigger() so the closure below
                    # binds to these specific values.
                    sub_h, sub_w = fr.image.shape[:2]
                    sub_bbox = ev.bbox
                    upgrader = self._upgrader
                    cls_name_for_upgrade = ev.cls_name
                    # Trigger time in two domains for the upgrader:
                    #   target_ts        — sub-stream PTS-anchored monotonic
                    #                      (the time domain Frame.ts lives in)
                    #   target_wallclock — datetime.now() at the trigger; used
                    #                      ONCE per (sub_epoch, main_epoch) to
                    #                      calibrate the cross-stream offset.
                    #   sub_epoch        — bumps on each sub-stream reconnect;
                    #                      the upgrader uses this to detect
                    #                      that the cached offset is stale.
                    target_ts = ev.t_b
                    target_wallclock = captured_at
                    target_sub_epoch = fr.epoch

                    # pass_id isn't known until insert_pass below, but
                    # on_finalize fires later (after the recorder's post-roll
                    # completes), so we plumb the pid in via a holder list
                    # that gets populated immediately after insert_pass.
                    pid_holder: list[int | None] = [None]
                    on_finalize = None
                    if upgrader is not None and sub_bbox is not None:
                        thumb_path_pending = str(self._recordings_dir / (clip_name[:-4] + ".jpg"))

                        def on_finalize(
                            _path=thumb_path_pending,
                            _cls=cls_name_for_upgrade,
                            _bbox=sub_bbox,
                            _size=(sub_w, sub_h),
                            _up=upgrader,
                            _ts=target_ts,
                            _wc=target_wallclock,
                            _ep=target_sub_epoch,
                            _drift_sub=self._drift_sub,
                            _holder=pid_holder,
                        ) -> None:
                            pid = _holder[0]
                            if pid is None:
                                return
                            _up.enqueue(
                                pass_id=pid,
                                thumb_path=_path,
                                focus_cls_name=_cls,
                                sub_bbox=_bbox,
                                sub_frame_size=_size,
                                target_ts=_ts,
                                target_wallclock=_wc,
                                sub_epoch=_ep,
                                drift_sub_override=_drift_sub,
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
                        on_finalize=on_finalize,
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
                    pid_holder[0] = pid
                    log.info(
                        "pass id=%d track=%d %s %s elapsed=%.3fs clip=%s",
                        pid, ev.track_id, ev.cls_name, ev.direction,
                        ev.elapsed_s, clip_name,
                    )
        finally:
            recorder.flush()
            log.info("capture worker stopped")
