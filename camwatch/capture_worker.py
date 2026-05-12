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

import json
import logging
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

_MPH_PER_MPS = 2.2369362920544
_FT_TO_M = 0.3048
# Bounds of the calibrated grid (the rectangle defined by the 4 corner anchor
# points). Speed is only meaningful for trajectory frames whose projected
# (X, Y) falls inside this rectangle — outside it, the homography is
# extrapolating and the v_inst values are unreliable.
#
# `_GRID_TOLERANCE_M` is the slack zone that absorbs bbox-detection jitter
# at the boundary. Without it, a car driving close to the east curb (X ≈ 0)
# sees individual frames flicker in/out of the grid as the bbox center
# wobbles by a few cm, producing visible gaps in the speed line on the
# chart even though the car is plainly on the road. The actual unreliable
# extrapolation tails sit several meters beyond the grid (e.g., Y = -9.87 m
# when the south boundary is -7.62 m), so a 0.5 m slack still excludes
# them while keeping the in-grid portion continuous.
_GRID_X_MIN = -35.0 * _FT_TO_M  # 5 ft west of the actual west curb (-30 ft)
_GRID_X_MAX = 0.0               # east curb (camera-side; tight is fine)
_GRID_Y_MIN = -25.0 * _FT_TO_M  # south
_GRID_Y_MAX = +25.0 * _FT_TO_M  # north
_GRID_TOLERANCE_M = 0.5
# West edge is extended past the physical curb because the bbox bottom-
# center for far-lane cars sometimes projects past the curb in world X
# due to perspective compression at the top of the trapezoid (small v,
# ~206 px). A few pixels of bbox jitter there translate to >1 m in
# world X. Cars physically stay on the asphalt, so this slack zone only
# catches projection artifacts — no real off-road traffic lives there.

# Stationary-track gate. A track whose projected ground point has stayed
# inside a tight box for at least N consecutive samples is treated as
# parked: any crossing event it produces is suppressed (no pass row, no
# clip, no thumbnail). Even a 5 mph driver covers >2 m across 30 sub-stream
# frames (~3 s), well beyond the 0.5 m spread, so legitimate passes never
# trip this gate. Without it, bbox jitter on a parked curb car can
# occasionally satisfy the 2-line crossing condition and produce a phantom
# pass at unrealistically slow speeds.
_STATIONARY_WINDOW_FRAMES = 30
_STATIONARY_SPREAD_M = 0.5

# Centered Y-vs-t regression window for the canonical reported speed.
# Samples whose projected Y is within ±_CENTERED_HALF_WINDOW_M of Y=0 (the
# camera's perpendicular line) are used to fit Y = m·t + b; speed = |m|.
# Anchoring at Y=0 puts the measurement in the part of the grid with the
# smallest homography reprojection error.
#
# Tiered widening: try the primary (±15 ft) window first. When there are
# too few samples in it for a stable fit — fast vehicles, frame drops, or
# tracker splits — expand to ±25 ft (the full grid). Wider samples include
# the noisier grid edges where homography reprojection is largest, so we
# only fall back to the wider window when the primary one isn't enough.
# A single noisy fast pass should be smoothed by more samples; a clean
# slow pass with plenty of in-window samples stays on the tighter fit.
_CENTERED_HALF_WINDOW_M = 15.0 * _FT_TO_M  # primary: ±15 ft = ±4.572 m
_WIDER_HALF_WINDOW_M    = 25.0 * _FT_TO_M  # fallback: ±25 ft = full grid Y-extent
_MIN_PRIMARY_SAMPLES    = 6                # widen when primary window has < this

# Night-mode (IR) gate. The Reolink E1 switches to monochrome IR illumination
# in low light; speed measurements from those frames are unreliable because
# only headlights/taillights are bright enough to detect, the bbox represents
# a motion-blurred light source rather than the wheel-on-road position, and
# the homography assumption (bbox bottom-center ↔ ground point) breaks down.
#
# Detection is content-based: in IR mode every pixel has R = G = B. We sample
# the center region of each frame and average the per-pixel channel deviation
# `|R-G| + |G-B| + |R-B|` over a sliding window of frames. A small threshold
# distinguishes IR (≈0) from color frames (typically ≫ 5) without false
# triggering on, e.g., a uniformly-grey overcast scene that still has some
# faint chroma in the lawn and asphalt.
_NIGHT_DEVIATION_THRESHOLD = 3.0
_NIGHT_WINDOW_FRAMES = 30           # ~2 s at 14 fps; smooths dawn/dusk flicker


class _NightModeDetector:
    """Per-frame grayscale-vs-color check with hysteresis.

    Computes the mean per-pixel channel deviation over a center crop of each
    frame, averaged across a sliding window so a single noisy frame can't
    flip the mode. Logs a single line on each transition so the operator can
    see when the gate kicks in/out.
    """

    def __init__(self, window: int = _NIGHT_WINDOW_FRAMES,
                 threshold: float = _NIGHT_DEVIATION_THRESHOLD) -> None:
        self._window = int(window)
        self._threshold = float(threshold)
        self._deviations: deque[float] = deque(maxlen=self._window)
        self._is_night: bool = False

    def update(self, frame) -> bool:
        h, w = frame.shape[:2]
        crop = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        b = crop[:, :, 0].astype(np.float32)
        g = crop[:, :, 1].astype(np.float32)
        r = crop[:, :, 2].astype(np.float32)
        dev = float((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean())
        self._deviations.append(dev)
        if len(self._deviations) < self._window:
            return self._is_night  # warm-up: hold previous state
        mean_dev = sum(self._deviations) / len(self._deviations)
        new_state = mean_dev < self._threshold
        if new_state != self._is_night:
            log.info(
                "night-mode %s (mean channel deviation = %.2f, threshold = %.2f)",
                "ENGAGED" if new_state else "RELEASED", mean_dev, self._threshold,
            )
            self._is_night = new_state
        return self._is_night


def _in_grid(X: float, Y: float) -> bool:
    return (_GRID_X_MIN - _GRID_TOLERANCE_M <= X <= _GRID_X_MAX + _GRID_TOLERANCE_M
            and _GRID_Y_MIN - _GRID_TOLERANCE_M <= Y <= _GRID_Y_MAX + _GRID_TOLERANCE_M)

from typing import TYPE_CHECKING

from .capture import RtspStream
from .config import Config
from .db import Database
from .detect import Detector, Track
from .digit_matcher import DigitMatcher
from .grid_crossing import GridCrossingDetector
from .homography import Homography
from .preview import PreviewBuffer
from .recorder import ClipRecorder
from .thumb_upgrader import ThumbUpgrader

if TYPE_CHECKING:
    from .metrics import MetricsCollector

# OSD pixel rectangle on the Reolink E1 sub stream (640x480) with the OSD
# at the bottom of the frame. Used only for the one-time per-epoch sub-
# stream drift calibration.
_OSD_REGION_SUB = (175, 452, 500, 477)

log = logging.getLogger(__name__)


def _track_on_road(t: Track, homog: Homography | None) -> bool:
    """Keep tracks whose ground_point projects inside the calibrated grid.

    Replaces the old pixel-rectangle ROI: the grid IS the road by
    construction (its corners are the curbs from calibration), so a
    homography-projected in-grid check is strictly more accurate than a
    hand-tuned axis-aligned pixel rectangle and is symmetric across lanes.
    Falls back to "accept all" if homography is missing — there is no
    sensible road definition without it.
    """
    if homog is None:
        return True
    X, Y = homog.project(t.ground_point[0], t.ground_point[1])
    return _in_grid(X, Y)


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
        metrics: "MetricsCollector | None" = None,
    ) -> None:
        super().__init__(name="capture-worker", daemon=True)
        self._cfg = cfg
        self._db = db
        self._recordings_dir = Path(recordings_dir)
        self._preview = preview
        self._profile = bool(profile)
        self._metrics = metrics
        self._stop_evt = threading.Event()
        self._stream: RtspStream | None = None
        self._error: BaseException | None = None
        self._recorder: ClipRecorder | None = None
        self._upgrader: ThumbUpgrader | None = None
        # Night-mode state, written by the capture loop and read by request
        # handlers (e.g., the live status badge endpoint). Booleans are
        # atomic in CPython, so no lock is needed.
        self._night_mode: bool = False
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
        # Homography-based parallel speed measurement. Loaded from
        # config/homography.yaml; if missing, the parallel speed is silently
        # skipped (the existing 2-line method continues unaffected).
        self._homog: Homography | None = Homography.load(
            Path("config/homography.yaml")
        )
        if self._homog is not None:
            log.info(
                "homography loaded for parallel speed (mean reproj err=%.1fcm, max=%.1fcm)",
                self._homog.mean_reproj_err_m * 100,
                self._homog.max_reproj_err_m * 100,
            )
        # Per-track ground-point trajectory: track_id → deque of
        # (ts, ground_u, ground_v, (bbox_x1, bbox_y1, bbox_x2, bbox_y2)).
        # Bounded per-track memory; whole-dict pruning happens on track GC
        # mirrored to the CrossingDetector's max_age window.
        self._trajectories: dict[
            int,
            deque[tuple[float, float, float, tuple[float, float, float, float]]],
        ] = {}

    def is_night_mode(self) -> bool:
        """Latest night-mode state computed by the capture loop. Read by
        the live-status badge endpoint to flip "live" → "paused" when the
        camera switches to IR illumination and the gate is active."""
        return bool(self._night_mode)

    def _is_stationary(self, track_id: int) -> bool:
        """True iff this track has at least _STATIONARY_WINDOW_FRAMES
        recent samples whose projected (X, Y) ground points all fall
        inside a _STATIONARY_SPREAD_M × _STATIONARY_SPREAD_M box."""
        if self._homog is None:
            return False
        traj = self._trajectories.get(track_id)
        if traj is None or len(traj) < _STATIONARY_WINDOW_FRAMES:
            return False
        recent = list(traj)[-_STATIONARY_WINDOW_FRAMES:]
        Xs: list[float] = []
        Ys: list[float] = []
        for _ts, u, v, _bb in recent:
            X, Y = self._homog.project(u, v)
            Xs.append(X)
            Ys.append(Y)
        return (
            (max(Xs) - min(Xs)) < _STATIONARY_SPREAD_M
            and (max(Ys) - min(Ys)) < _STATIONARY_SPREAD_M
        )

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

    def _save_pass_trajectory_jsonl(
        self,
        *,
        pid: int,
        ev,
        traj: list[tuple[float, float, float, tuple[float, float, float, float]]],
        speed_mph: float | None,
        speed_method: str | None = None,
        half_window_used_m: float = _CENTERED_HALF_WINDOW_M,
    ) -> None:
        """Write per-frame trajectory + per-frame v_inst to events/pass_<pid>.jsonl.

        Format:
          line 0: manifest dict with pass-level metadata
          line 1+: one dict per frame in the track's trajectory, with the
                   real-PTS-anchored timestamp, ground-point pixel, full bbox,
                   homography-projected (X, Y), and the instantaneous speed
                   computed from this frame and the previous one.

        v_inst on the first frame is null (no prior frame to diff against).
        """
        if self._homog is None or not traj:
            return
        out_dir = self._recordings_dir.parent / "events"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pass_{pid}.jsonl"

        # Project each frame's ground-point through H, then derive v_inst from
        # consecutive (X, Y, t) triples. Done here once rather than per-frame
        # in the hot loop.
        projected: list[tuple[float, float, float, tuple, float, float]] = []
        for ts, u, v, bb in traj:
            X, Y = self._homog.project(u, v)
            projected.append((float(ts), float(u), float(v), bb, float(X), float(Y)))

        rows: list[dict] = []
        t0 = projected[0][0]
        for i, (ts, u, v, bb, X, Y) in enumerate(projected):
            in_grid = _in_grid(X, Y)
            v_inst_mph: float | None = None
            # Only compute / publish v_inst when BOTH this frame and the prior
            # frame lie inside the calibrated grid. Outside the grid the
            # homography is extrapolating and the velocity reading would be
            # unreliable, so we suppress it from the chart.
            if i > 0 and in_grid:
                ts_p, _, _, _, X_p, Y_p = projected[i - 1]
                if _in_grid(X_p, Y_p):
                    dt = ts - ts_p
                    if dt > 0:
                        d = ((X - X_p) ** 2 + (Y - Y_p) ** 2) ** 0.5
                        v_inst_mph = (d / dt) * _MPH_PER_MPS
            in_speed_window = abs(Y) <= half_window_used_m
            rows.append({
                "frame": i,
                "ts": ts,
                "t_rel": ts - t0,
                "u": u,
                "v": v,
                "bbox": list(bb),
                "X": X,
                "Y": Y,
                "in_grid": in_grid,
                "in_speed_window": in_speed_window,
                "v_inst_mph": v_inst_mph,
            })

        # Clip start = first crossing minus pre-roll. The recorder writes the
        # clip's first frame at this PTS, so video.currentTime (browser-side)
        # maps directly to (frame_ts - clip_start_ts) in real time.
        clip_pre_roll_s = float(self._cfg.clip_margin_s)
        clip_post_roll_s = float(self._cfg.clip_margin_s)
        clip_start_ts = float(min(ev.t_a, ev.t_b)) - clip_pre_roll_s
        clip_end_ts = float(max(ev.t_a, ev.t_b)) + clip_post_roll_s
        manifest = {
            "type": "manifest",
            "pass_id": int(pid),
            "track_id": int(ev.track_id),
            "cls_name": str(ev.cls_name),
            "direction": str(ev.direction),
            "elapsed_s": float(ev.elapsed_s),
            "t_a_pts": float(ev.t_a),
            "t_b_pts": float(ev.t_b),
            "clip_start_ts": clip_start_ts,
            "clip_end_ts": clip_end_ts,
            "clip_pre_roll_s": clip_pre_roll_s,
            "clip_post_roll_s": clip_post_roll_s,
            # v_homog_mph is kept for compatibility with the existing JS chart
            # legend that reads from manifest. Same canonical speed.
            "v_homog_mph": speed_mph if speed_mph is not None else float("nan"),
            "speed_mph": speed_mph,
            "speed_method": speed_method,  # 'regression' | 'median_fallback' | None
            "speed_window_half_m": float(half_window_used_m),
            "n_frames": len(rows),
            "frame_size_sub": list(self._homog.frame_size_sub),
        }

        with open(out_path, "w") as f:
            f.write(json.dumps(manifest) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")

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
            "capture worker starting (threshold=%.1f mph, lines a=%d b=%d for clip annotation only, "
            "trigger=grid-entry/exit)",
            self._cfg.alert_threshold_mph, cal.line_a_x, cal.line_b_x,
        )

        # YOLO sees the full frame. Tracks are filtered by an in-grid check
        # (homography-projected ground point) before reaching the detector
        # and recorder; the grid covers the whole road by construction.
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
        if self._homog is None:
            log.error(
                "homography missing; grid-based trigger cannot run. "
                "Build it with scripts/build_homography_from_marks.py."
            )
            return
        crossing = GridCrossingDetector(
            homography=self._homog,
            grid_x_min=_GRID_X_MIN, grid_x_max=_GRID_X_MAX,
            grid_y_min=_GRID_Y_MIN, grid_y_max=_GRID_Y_MAX,
            tolerance_m=_GRID_TOLERANCE_M,
            max_track_age_s=self._cfg.max_track_age_s,
        )
        if self._preview is not None:
            self._preview.configure(cal.roi, cal.line_a_x, cal.line_b_x)
            self._preview.set_grid(
                self._homog,
                _GRID_X_MIN, _GRID_X_MAX,
                _GRID_Y_MIN, _GRID_Y_MAX,
            )
            self._preview.set_show_grid(self._cfg.preview_show_grid)

        # Optional: parallel high-res stream for thumbnail upgrades.
        # Indexing is purely PTS-anchored monotonic now; no OCR region needed.
        thumb_url = self._cfg.camera.rtsp_url_thumb
        if thumb_url:
            self._upgrader = ThumbUpgrader(
                rtsp_url=thumb_url,
                model=self._cfg.model,
                db=self._db,
                metrics=self._metrics,
            )
            self._upgrader.start()

        self._stream = RtspStream(
            self._cfg.camera.rtsp_url, log_label="sub", metrics=self._metrics,
        )
        last_purge = time.monotonic()
        purge_interval_s = 3600.0  # check retention once an hour
        prof = _StageTimer() if self._profile else None
        if prof is not None:
            log.info("capture worker: --profile enabled, logging stage timings every 30s")
        last_loop_t: float | None = None
        night_detector = _NightModeDetector()

        try:
            for fr in self._stream.frames():
                if self._stop_evt.is_set():
                    break

                # Periodic retention sweep — two phases:
                #   1. Recordings older than recordings_days: delete clip + thumbs, NULL clip_path
                #   2. Pass rows older than passes_days: hard-delete row + per-pass jsonl
                if time.monotonic() - last_purge > purge_interval_s:
                    last_purge = time.monotonic()
                    events_dir = self._cfg.events_dir

                    days_recordings = int(self._cfg.recordings_days or 0)
                    if days_recordings > 0:
                        rec_items = self._db.purge_recordings_older_than(days_recordings)
                        threshold_mph = float(self._cfg.alert_threshold_mph)
                        archive_dir = Path("recordings_archive")
                        if rec_items:
                            archive_dir.mkdir(parents=True, exist_ok=True)
                        archived = 0
                        deleted = 0
                        for pid, cp, speed in rec_items:
                            base = cp[:-4]
                            thumb_small = base + ".jpg"
                            thumb_big = base + "_big.jpg"
                            if speed is not None and speed >= threshold_mph:
                                # Alarm pass: rescue thumbnails only; delete the .mp4.
                                # Per-pass jsonl follows the standard passes-sweep flow.
                                for src in (thumb_small, thumb_big):
                                    if Path(src).exists():
                                        try:
                                            shutil.move(src, archive_dir / Path(src).name)
                                        except Exception as e:  # noqa: BLE001
                                            log.debug("archive move %s: %s", src, e)
                                try:
                                    Path(cp).unlink(missing_ok=True)
                                except Exception as e:  # noqa: BLE001
                                    log.debug("recordings cleanup (alarm clip): %s: %s", cp, e)
                                archived += 1
                            else:
                                # Non-alarm: delete clip + thumbs (jsonl stays for passes-sweep)
                                for path in (cp, thumb_small, thumb_big):
                                    try:
                                        Path(path).unlink(missing_ok=True)
                                    except Exception as e:  # noqa: BLE001
                                        log.debug("recordings cleanup: %s: %s", path, e)
                                deleted += 1
                        if archived or deleted:
                            log.info(
                                "retention: %d alarm passes archived, %d non-alarm cleaned "
                                "(recordings_days=%d, threshold=%.1f mph)",
                                archived, deleted, days_recordings, threshold_mph,
                            )

                    # Metrics retention: hardcoded 7-day cap. ~120k rows max,
                    # cheap to keep; surfaced through the perf panel only.
                    purged_m = self._db.purge_metrics_older_than(7)
                    if purged_m:
                        log.info("retention: purged %d metric rows older than 7 days", purged_m)

                    days_passes = int(self._cfg.passes_days or 0)
                    if days_passes > 0:
                        n, items = self._db.purge_older_than(days_passes)
                        for pid, cp in items:
                            paths_to_unlink: list[str] = []
                            if cp:
                                paths_to_unlink.append(cp)
                                paths_to_unlink.append(cp[:-4] + ".jpg")
                                paths_to_unlink.append(cp[:-4] + "_big.jpg")
                            paths_to_unlink.append(str(events_dir / f"pass_{pid}.jsonl"))
                            for path in paths_to_unlink:
                                try:
                                    Path(path).unlink(missing_ok=True)
                                except Exception as e:  # noqa: BLE001
                                    log.debug("retention: %s: %s", path, e)
                        if n:
                            log.info("retention: purged %d passes older than %d days", n, days_passes)

                loop_t = time.perf_counter() if prof else 0.0
                if prof and last_loop_t is not None:
                    # Wall-clock between successive frames reaching this point.
                    # Lower-bounded by frame interval; if our work exceeds it,
                    # this gap reflects the real consumer rate.
                    prof.record("interframe_gap", loop_t - last_loop_t)
                last_loop_t = loop_t

                if self._metrics is not None:
                    # Pipeline lag = wallclock now − the frame's PTS-anchored
                    # ts. Both are in the monotonic time domain (PTS is
                    # re-anchored on session start), so the difference is
                    # how far behind realtime this frame is at the consumer.
                    self._metrics.record_lag(time.monotonic() - fr.ts)
                    self._metrics.record_frame("yolo")

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

                # Night-mode gate. The Reolink E1 switches to monochrome IR
                # in low light, where speed measurements are unreliable
                # (headlight-only detections, motion blur, bbox bottom no
                # longer at the wheel-on-road position). When enabled and
                # active, skip YOLO/trigger/recorder for this frame and
                # update the preview with the raw IR feed (no annotations).
                self._night_mode = night_detector.update(fr.image)
                if self._night_mode and self._cfg.pause_at_night:
                    if self._preview is not None:
                        self._preview.update(fr.image, [])
                    last_loop_t = loop_t
                    continue

                # Time YOLO unconditionally so the perf panel always has data;
                # `prof` separately records the same value into the 30s log
                # summary when --profile is set.
                yolo_t0 = time.perf_counter()
                all_tracks = det.track(fr.image)
                yolo_dt = time.perf_counter() - yolo_t0
                if prof:
                    prof.record("yolo_track", yolo_dt)
                if self._metrics is not None:
                    self._metrics.record_stage("yolo", yolo_dt)

                t0 = time.perf_counter() if prof else 0.0
                tracks = [t for t in all_tracks if _track_on_road(t, self._homog)]
                if prof:
                    prof.record("on_road_filter", time.perf_counter() - t0)

                # Per-track trajectory accumulation for homography-based speed
                # AND for the per-pass JSONL trajectory log. Each entry stores
                # ts (real PTS-anchored monotonic), the ground point, and the
                # full bbox — the bbox isn't used for speed, but is captured
                # so future visualization tools can render the box back over
                # the clip frame. Bounded by deque(maxlen=200) per track;
                # whole-track GC is piggybacked on the crossing detector's
                # stale-track logic.
                if self._homog is not None:
                    for tr in tracks:
                        traj = self._trajectories.setdefault(
                            int(tr.track_id), deque(maxlen=200),
                        )
                        gx, gy = tr.ground_point
                        bb = tuple(float(x) for x in tr.bbox)
                        traj.append((fr.ts, float(gx), float(gy), bb))

                # Stationary-track gate at the trajectory layer. When a track
                # has been sitting still for _STATIONARY_WINDOW_FRAMES samples
                # — a car parked inside the grid — clear its trajectory and
                # reset its grid-crossing entry. Without this, a car that
                # parks in the grid and eventually drives away fires a single
                # pass with elapsed_s = parked_duration + motion_duration
                # (e.g., 697 s for a van that parks then leaves 11 minutes
                # later), and the recorder can't pull entry-time frames from
                # its 7 s ring buffer. Resetting here means the next motion
                # is treated as a fresh pass with a recent entry_ts and a
                # trajectory containing only the moving samples.
                for tr in tracks:
                    tid = int(tr.track_id)
                    if self._is_stationary(tid):
                        self._trajectories.pop(tid, None)
                        crossing.reset_in_grid_entry(tid)

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

                # GridCrossingDetector needs to see EVERY YOLO track, not
                # just in-grid ones, so it can observe the in-grid → out-of-
                # grid transition and fire the event immediately. Feeding it
                # the on-road-filtered list would hide the exit, forcing the
                # event to wait for the 5 s age-out path and triggering the
                # recorder several seconds after the car was already gone.
                t0 = time.perf_counter() if prof else 0.0
                events = crossing.update(all_tracks, fr.ts)
                if prof:
                    prof.record("crossing_update", time.perf_counter() - t0)
                if prof:
                    prof.maybe_log()
                for ev in events:
                    # Stationary-track gate. Bbox jitter on a parked curb car
                    # can occasionally satisfy the 2-line crossing condition;
                    # we drop those events here so they never become passes.
                    if self._is_stationary(ev.track_id):
                        log.info(
                            "ignoring crossing event for stationary track id=%d "
                            "(parked or detection-jitter)",
                            ev.track_id,
                        )
                        self._trajectories.pop(ev.track_id, None)
                        continue
                    captured_at = datetime.now().astimezone()
                    stamp = captured_at.strftime("%Y%m%dT%H%M%S")
                    clip_name = f"cal_{stamp}_id{ev.track_id}_{ev.direction}.mp4"

                    # Reported speed: linear regression of Y vs t over samples
                    # whose projected Y is within ±15 ft of Y=0 (the camera's
                    # perpendicular line, where reprojection error is smallest).
                    # The slope of the fit IS the velocity component along the
                    # road; we report its magnitude as the speed of the pass.
                    # We also compute the legacy Method A (median of v_inst over
                    # the full grid) for side-by-side comparison while the
                    # methods are vetted; it's only logged, not stored.
                    # Speed estimation: Y-vs-t regression with a tiered
                    # window. The primary ±15 ft window gives the highest-
                    # accuracy fit (homography error is smallest near Y=0).
                    # When that window has fewer samples than _MIN_PRIMARY_-
                    # SAMPLES — fast vehicles, frame drops, tracker splits,
                    # or trajectories that start mid-grid — widen to ±25 ft
                    # (the full grid) to gather more points. Same math, just
                    # over a larger set. Both paths are "regression"; the
                    # window size is recorded so the UI can flag widened
                    # passes as lower confidence.
                    speed_mph: float | None = None
                    speed_method: str | None = None
                    n_speed_samples = 0
                    speed_r2 = 0.0
                    half_window_used_m = _CENTERED_HALF_WINDOW_M
                    method_a_mph = float("nan")
                    method_a_n = 0
                    if self._homog is not None:
                        traj_for_speed = list(self._trajectories.get(ev.track_id, ()))
                        speed_samples = [(t, u, v) for (t, u, v, _bb) in traj_for_speed]
                        method_a_mph, method_a_n = self._homog.median_speed_in_grid(
                            speed_samples,
                            grid_x_min=_GRID_X_MIN, grid_x_max=_GRID_X_MAX,
                            grid_y_min=_GRID_Y_MIN, grid_y_max=_GRID_Y_MAX,
                            tolerance_m=_GRID_TOLERANCE_M,
                        )

                        # Tier 1: primary ±15 ft window
                        primary_mph, primary_r2, primary_n = self._homog.centered_speed_y_regression(
                            speed_samples,
                            half_window_m=_CENTERED_HALF_WINDOW_M,
                        )
                        if (not (primary_mph != primary_mph)
                                and primary_n >= _MIN_PRIMARY_SAMPLES):
                            speed_mph = float(primary_mph)
                            speed_method = "regression"
                            speed_r2 = primary_r2
                            n_speed_samples = primary_n
                            half_window_used_m = _CENTERED_HALF_WINDOW_M
                        else:
                            # Tier 2: widen to ±25 ft (the full grid)
                            wide_mph, wide_r2, wide_n = self._homog.centered_speed_y_regression(
                                speed_samples,
                                half_window_m=_WIDER_HALF_WINDOW_M,
                            )
                            if not (wide_mph != wide_mph) and wide_n >= 3:
                                speed_mph = float(wide_mph)
                                speed_method = "regression_wide"
                                speed_r2 = wide_r2
                                n_speed_samples = wide_n
                                half_window_used_m = _WIDER_HALF_WINDOW_M
                            # else: speed remains None (true insufficient data)
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
                    upgrader = self._upgrader
                    cls_name_for_upgrade = ev.cls_name

                    # Pick the upgrade target temporally (target_ts) and
                    # spatially (upgrade_bbox). Two strategies, chosen per
                    # pass:
                    #   1. (default) **Midpoint of the in-grid trajectory**.
                    #      The midpoint is most likely to be in an
                    #      unoccluded portion of the road and is far from
                    #      grid-edge bbox-jitter. Both target_ts and the
                    #      bbox come from the *same* frame, so the high-
                    #      res match's crop region is temporally aligned
                    #      with where the car actually is.
                    #   2. **Skip the upgrade entirely** if the trajectory
                    #      was visibly truncated — the track was lost
                    #      well short of any grid Y boundary, almost
                    #      always because of an occluder (a parked car at
                    #      the south curb is the canonical case). In that
                    #      situation the high-res match would land on the
                    #      occluder rather than the moving car. Better to
                    #      keep the sub-stream thumbnail in place.
                    midpoint_ts: float | None = None
                    midpoint_bbox: tuple[float, float, float, float] | None = None
                    truncated = False
                    last_in_grid_Y: float = 0.0
                    if self._homog is not None and traj_for_speed:
                        in_grid_idx: list[int] = []
                        for i, (_ts, u, v, _bb) in enumerate(traj_for_speed):
                            X, Y = self._homog.project(u, v)
                            if _in_grid(X, Y):
                                in_grid_idx.append(i)
                                last_in_grid_Y = Y
                        if in_grid_idx:
                            mid_pos = len(in_grid_idx) // 2
                            mid_i = in_grid_idx[mid_pos]
                            mid_ts, _, _, mid_bb = traj_for_speed[mid_i]
                            midpoint_ts = float(mid_ts)
                            midpoint_bbox = mid_bb
                            # Truncation: did the last in-grid sample reach
                            # a grid Y boundary? Boundary is ±7.62 m; well
                            # short of either side (here, |Y| < 6 m) means
                            # the track ended in mid-grid, almost certainly
                            # because of occlusion.
                            if abs(last_in_grid_Y) < 6.0:
                                truncated = True

                    target_ts = midpoint_ts if midpoint_ts is not None else (ev.t_b - 0.3)
                    upgrade_bbox = midpoint_bbox if midpoint_bbox is not None else ev.bbox
                    target_wallclock = captured_at
                    target_sub_epoch = fr.epoch
                    if truncated:
                        log.info(
                            "track %d: trajectory truncated at Y=%.2f (likely occluded); "
                            "skipping high-res thumb upgrade",
                            ev.track_id, last_in_grid_Y,
                        )

                    # pass_id isn't known until insert_pass below, but
                    # on_finalize fires later (after the recorder's post-roll
                    # completes), so we plumb the pid in via a holder list
                    # that gets populated immediately after insert_pass.
                    pid_holder: list[int | None] = [None]
                    on_finalize = None
                    if upgrader is not None and upgrade_bbox is not None and not truncated:
                        thumb_path_pending = str(self._recordings_dir / (clip_name[:-4] + ".jpg"))

                        def on_finalize(
                            _path=thumb_path_pending,
                            _cls=cls_name_for_upgrade,
                            _bbox=upgrade_bbox,
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
                        speed_mph=speed_mph,
                        speed_method=speed_method,
                    )
                    pid_holder[0] = pid
                    if speed_mph is not None:
                        method_a_str = (
                            f"{method_a_mph:.2f}" if method_a_mph == method_a_mph else "—"
                        )
                        window_ft = int(round(half_window_used_m / _FT_TO_M))
                        wide_tag = " WIDENED" if speed_method == "regression_wide" else ""
                        speed_str = (
                            f"{speed_mph:.2f} mph "
                            f"(regression{wide_tag} on {n_speed_samples} samples, ±{window_ft} ft, "
                            f"r²={speed_r2:.3f}; Method A median={method_a_str} mph over {method_a_n})"
                        )
                    else:
                        speed_str = (
                            f"speed unavailable (insufficient trajectory samples)"
                        )
                    log.info(
                        "pass id=%d track=%d %s %s %s  elapsed=%.3fs  clip=%s",
                        pid, ev.track_id, ev.cls_name, ev.direction,
                        speed_str, ev.elapsed_s, clip_name,
                    )
                    # Persist the full per-frame trajectory + computed v_inst
                    # to events/pass_<pid>.jsonl so the chart and any future
                    # offline analysis has real-PTS-anchored timing. Also
                    # records the canonical speed_mph in the manifest.
                    if self._homog is not None:
                        traj = list(self._trajectories.get(ev.track_id, ()))
                        try:
                            self._save_pass_trajectory_jsonl(
                                pid=pid, ev=ev, traj=traj,
                                speed_mph=speed_mph,
                                speed_method=speed_method,
                                half_window_used_m=half_window_used_m,
                            )
                        except Exception as e:  # noqa: BLE001
                            log.warning(
                                "pass id=%d: failed to write trajectory JSONL: %s",
                                pid, e,
                            )
                        # We've consumed the trajectory for this pass; drop it
                        # so subsequent unrelated re-uses of this track_id (rare,
                        # but possible after BotSORT recycles IDs) start fresh.
                        self._trajectories.pop(ev.track_id, None)
        finally:
            recorder.flush()
            log.info("capture worker stopped")
