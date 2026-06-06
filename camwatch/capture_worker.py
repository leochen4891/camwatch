"""Always-on capture thread for the web UI.

Single-stream mode: detection, tracking, crossings, preview, and clip
recording all run from the configured RTSP path (the camera's main stream,
2048x1536). YOLO sees the full frame and the recorder's thumbnail comes
from whatever frame was current at the trigger moment — no async
high-res upgrade path.

The calibration ROI is applied as a post-detection filter on each track's
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import metrics_push as mp
from .capture import open_frame_source
from .config import Config
from .db import Database
from .detect import Detector, Track
from .grid_crossing import GridCrossingDetector
from .homography import Homography
from .preview import PreviewBuffer
from .recorder import ClipRecorder

_ENRICHER_DEFAULT_URL = "http://127.0.0.1:8765"
_ENRICHER_TIMEOUT_S = 30.0
# The camwatch HTTP base the enricher uses to fetch this pass's images.
# Loopback by design — same host as camwatch:8000, so we bypass Cloudflare
# Access entirely.
_CAMWATCH_LOOPBACK_BASE = "http://127.0.0.1:8000"

# Default coords for the daylight gate. Override in config/config.yaml
# under `enricher.daylight` if the camera ever moves.
_DAYLIGHT_LAT_DEFAULT = 40.7956
_DAYLIGHT_LON_DEFAULT = -74.3148
_DAYLIGHT_BUFFER_HOURS_DEFAULT = 1.0

if TYPE_CHECKING:
    from .capture import RtspStream, StaticFrameStream
    from .metrics import MetricsCollector

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
_GRID_Y_MIN = -40.0 * _FT_TO_M  # south — extended along with the calibration
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
# clip, no thumbnail). Even a 5 mph driver covers >2 m across 30 frames
# (~2 s at the current 15 fps capture rate), well beyond the 0.5 m spread,
# so legitimate passes never trip this gate. Without it, bbox jitter on a
# parked curb car can occasionally satisfy the 2-line crossing condition
# and produce a phantom pass at unrealistically slow speeds.
_STATIONARY_WINDOW_FRAMES = 30
_STATIONARY_SPREAD_M = 0.5

# Reported speed: cumulative-distance / cumulative-time from grid entry.
# At each in-grid sample i (1-indexed): mph_i = arc_length(0..i) / (t_i - t_0).
# The headline is the running average at grid exit. Robust to PTS-burst
# stutter — a brief cluster of frames sharing nearly identical timestamps
# doesn't perturb the totals, only the per-frame v_inst readings.
# Wait for this many samples before computing (1-2 samples are too noisy
# to be useful even as a "current" reading).
_MIN_RUNNING_SAMPLES = 5

# Trustworthiness guards on the headline speed. The running average is only
# meaningful when the trajectory's time base and the focus track are sound;
# two failure modes corrupt it and produce phantom over-speeds:
#   1. Timing compression. A variable-frame-rate source (or a network/decoder
#      burst) can deliver frames with bunched presentation timestamps, so the
#      trajectory's total span — the speed denominator — collapses and the
#      reported speed inflates. A pass whose frames imply a frame rate above
#      what the camera can physically produce is rejected. The 4K Reolink tops
#      out near 25 fps; 35 leaves headroom for a constant-fps stream without
#      admitting the 40-50 fps bursts that VFR produced.
#   2. Spatial jump. When the focus track's box merges with an oncoming vehicle
#      (two cars meeting in frame), its ground point leaps sideways/backward,
#      inflating cumulative arc length far past the straight-line displacement.
#      A clean crossing stays within ~1.03; 1.4 rejects a doubled-back path.
#   3. Partial early burst. A cluster of bunched-PTS frames at track acquisition
#      followed by a normally-timed tail: the per-pass average frame rate stays
#      under the cap (the tail dilutes it), so guard 1 misses it, but the early
#      frames inflate the running average so it is still descending at grid exit
#      instead of converged. A real vehicle can't shed >8% of its speed in one
#      ~0.08 s interval, so a consistently-descending exit is a timing artifact.
#      (The pre-2026-05 2-line speed method was immune to this; the trajectory
#      running-average integrates from acquisition, so it is not.)
# The three checks above flag suspicious *shape*, but on their own they also
# fire on plausible passes (a brief acquisition burst leaves a 25 mph headline
# mildly non-converged; a slow curb-side wobble doubles back). A corrupted
# shape only yields a *wrong* number when the result is also implausibly fast,
# so the speed ceiling gates them: a suspicious pass is rejected only when its
# headline exceeds it. Residential traffic here runs ~10-40 mph; 55 sits above
# any real speeder but below the phantom over-speeds (67-100 mph) the bursts
# produce, and a clean reading above 55 (no suspicious shape) is kept.
# A tripped guard yields speed "unknown" (NULL) rather than a fabricated number.
_MAX_PLAUSIBLE_FPS = 35.0
_MAX_ARC_DISPLACEMENT_RATIO = 1.4
_MAX_EXIT_DESCENT = 0.08
_MAX_PLAUSIBLE_MPH = 55.0
# NOTE (2026-06): the live headline path now uses the cadence time base
# (`_cadence_speed` below), which makes the timing guards (1 and 3) and the
# speed ceiling unnecessary there — only the spatial-jump guard (2) remains
# active, rejecting on shape alone since a merged track corrupts the distance
# itself. The constants above still parameterize `running_avg_speed` as a
# primitive (tests, legacy data tooling).


def _cadence_speed(
    traj,
    rate_fps: "float | None",
    homog,
    *,
    min_samples: int = _MIN_RUNNING_SAMPLES,
    max_arc_displacement_ratio: float = _MAX_ARC_DISPLACEMENT_RATIO,
) -> "tuple[float | None, int]":
    """Headline speed from the cadence time base.

    `traj` is a sequence of (ts, u, v, bbox, seq, epoch) trajectory tuples.
    Per-frame times are reconstructed as t_i = (seq_i - seq_0) / rate_fps —
    the camera's PTS is never consulted (it is untrustworthy; see
    pts_timing_investigation.md). Seq gaps from missed detections or dropped
    frames correctly lengthen the elapsed time.

    Returns (mph, n_samples), with mph None when no trustworthy speed can be
    produced:
      - rate_fps is None (stream warm-up) or non-positive
      - fewer than `min_samples` trajectory points
      - the pass spans an RTSP reconnect (epoch changed; seq spacing is not
        meaningful across sessions)
      - non-increasing seq span
      - the spatial-jump guard fires (arc length far beyond the net
        displacement: a track merge corrupted the distance, so the speed is
        unknown at any magnitude)
    """
    n = len(traj)
    if homog is None or rate_fps is None or rate_fps <= 0 or n < min_samples:
        return None, n
    seq0, epoch0 = traj[0][4], traj[0][5]
    seq_last, epoch_last = traj[-1][4], traj[-1][5]
    if epoch_last != epoch0 or seq_last <= seq0:
        return None, n
    samples = [
        ((seq - seq0) / rate_fps, u, v)
        for (_ts, u, v, _bb, seq, _epoch) in traj
    ]
    final_mph, _per_frame, n_used = homog.running_avg_speed(
        samples, min_samples=min_samples,
        max_plausible_fps=None,          # timing is trustworthy by construction
        max_arc_displacement_ratio=max_arc_displacement_ratio,
        max_exit_descent=None,           # timing is trustworthy by construction
        max_plausible_mph=None,          # reject merged tracks on shape alone
    )
    if final_mph != final_mph:  # NaN: guard rejected or degenerate
        return None, n_used
    return float(final_mph), n_used


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
        self._stream: "RtspStream | StaticFrameStream | None" = None
        self._error: BaseException | None = None
        self._recorder: ClipRecorder | None = None
        # Night-mode state, written by the capture loop and read by request
        # handlers (e.g., the live status badge endpoint). Booleans are
        # atomic in CPython, so no lock is needed.
        self._night_mode: bool = False
        # Homography for projection + speed, from the elected main camera's
        # registry profile (ADR-015 — camera facts live in camwatch-cameras,
        # not in this repo). Election already gated on a calibrated speed
        # capability, so this only fails on a corrupt artifact; the worker
        # then runs without speed, as before.
        self._homog: Homography | None = Homography.from_profile(
            cfg.camera.profile
        )
        if self._homog is not None:
            log.info(
                "homography loaded for %s (mean reproj err=%.1fcm, max=%.1fcm)",
                cfg.camera.main_id,
                self._homog.mean_reproj_err_m * 100,
                self._homog.max_reproj_err_m * 100,
            )
        # Registry-measured cadence (ADR-010: measured, never nominal). The
        # time-base fallback when the live received-frame rate isn't
        # available — stream warm-up and the ~30s after a reconnect — where
        # the previous behavior was "speed unknown".
        self._registry_cadence_fps: float = cfg.camera.profile.cadence_fps()
        # Per-track ground-point trajectory: track_id → deque of
        # (ts, ground_u, ground_v, (bbox_x1, bbox_y1, bbox_x2, bbox_y2),
        # frame_seq, stream_epoch). `frame_seq` is the stream's received-frame
        # sequence number and `stream_epoch` its reconnect epoch — together
        # with the stream's received_fps() they form the cadence time base
        # for speed (the camera's per-frame PTS is untrustworthy; see
        # pts_timing_investigation.md).
        # Bounded per-track memory; whole-dict pruning happens on track GC
        # mirrored to the CrossingDetector's max_age window.
        self._trajectories: dict[
            int,
            deque[
                tuple[
                    float, float, float,
                    tuple[float, float, float, float],
                    int, int,
                ]
            ],
        ] = {}
        # Per-track hysteresis: once a track has been strictly inside the
        # grid at least once, subsequent samples can drift up to
        # _GRID_TOLERANCE_M past the strict bounds and still accumulate.
        # Until that strict-entry happens we ignore samples in the tolerance
        # band (avoids logging a trajectory for a car that only ever rides
        # the slack zone at the curb).
        self._entered_strict: dict[int, bool] = {}
        # Fire-and-forget pool for the local enrichment service. A tiny
        # bounded pool — if the enricher is slow or down, we drop the call
        # for this pass and let the periodic backfill script catch it.
        self._enrich_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="enrich")
        enricher_cfg = getattr(cfg, "enricher", None) or {}
        self._enricher_url = enricher_cfg.get("url", _ENRICHER_DEFAULT_URL) if isinstance(enricher_cfg, dict) else _ENRICHER_DEFAULT_URL
        # Disabled by default while the local enricher is still being trained
        # offline against captured passes. Flip `enricher.enabled: true` in
        # config once the trained model is ready to serve live /enrich calls.
        self._enricher_enabled = bool(
            enricher_cfg.get("enabled", False) if isinstance(enricher_cfg, dict) else False
        )

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
        for _ts, u, v, _bb, _seq, _epoch in recent:
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
        # Drain in-flight enrich calls before exit; any pass already in the
        # queue still gets a chance to label. cancel_futures=True drops
        # not-yet-started ones (they'll be picked up by the backfill script).
        try:
            self._enrich_pool.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass

    def _fire_enrich(self, pass_id: int, direction: str | None, captured_at: str) -> None:
        """POST to the local enrichment service with this pass's image URLs.

        After the camwatch-enricher repo split the enricher no longer touches
        camwatch.db — it returns the decision over HTTP, and we persist it
        here into the `local_*` columns.

        The recorder writes the .jpg only at clip finalization (a few
        seconds after insert_pass), so the first POST sometimes lands
        before the thumbnail exists. The enricher returns 404 in that
        case; we retry on a short backoff until the recorder catches up.

        Skips non-daylight captures entirely — IR/night images would
        force the enricher to label by lighting instead of make/model,
        and pollute the index if the enricher upserts the embedding.
        Those passes go straight to the Opus workflow.
        """
        if not self._enricher_enabled:
            return
        try:
            from .daylight import is_daylight
            if not is_daylight(
                captured_at,
                lat=_DAYLIGHT_LAT_DEFAULT,
                lon=_DAYLIGHT_LON_DEFAULT,
                buffer_hours=_DAYLIGHT_BUFFER_HOURS_DEFAULT,
            ):
                log.info("enrich pass=%d skipped: outside daylight window", pass_id)
                mp.ENRICHMENT.inc(status="skipped_night")
                return
        except Exception as e:  # noqa: BLE001
            log.warning("daylight check failed for pass=%d: %s (proceeding anyway)",
                        pass_id, e)
        url = f"{self._enricher_url.rstrip('/')}/enrich"
        image_urls = {
            "thumb": f"{_CAMWATCH_LOOPBACK_BASE}/passes/{int(pass_id)}/thumb",
            "entry": f"{_CAMWATCH_LOOPBACK_BASE}/passes/{int(pass_id)}/thumb?anchor=entry",
            "exit":  f"{_CAMWATCH_LOOPBACK_BASE}/passes/{int(pass_id)}/thumb?anchor=exit",
        }
        body = {
            "pass_id": int(pass_id),
            "image_urls": image_urls,
            "direction": direction,
        }
        # Retry budget covers the worst-case clip duration (max_clip_s ≈
        # pre+post-roll + crossing window, typically <5s).
        retry_delays_s = (1.0, 1.5, 2.0, 2.5, 3.0)
        try:
            import httpx
            with httpx.Client(timeout=_ENRICHER_TIMEOUT_S) as client:
                attempt = 0
                while True:
                    resp = client.post(url, json=body)
                    if resp.status_code == 404 and attempt < len(retry_delays_s):
                        time.sleep(retry_delays_s[attempt])
                        attempt += 1
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    try:
                        self._db.apply_local_enrichment(pass_id, data)
                    except Exception:  # noqa: BLE001
                        log.exception("failed to persist enrich response for pass=%d", pass_id)
                    log.info(
                        "enrich pass=%d status=%s make=%s model=%s sim=%.3f",
                        pass_id, data.get("status"),
                        data.get("make"), data.get("model"),
                        float(data.get("top_sim") or 0.0),
                    )
                    mp.ENRICHMENT.inc(status="ok")
                    return
        except Exception as e:  # noqa: BLE001
            log.warning("enrich pass=%d failed: %s", pass_id, e)
            mp.ENRICHMENT.inc(status="error")

    def _save_pass_trajectory_jsonl(
        self,
        *,
        pid: int,
        ev,
        traj: list[
            tuple[
                float, float, float,
                tuple[float, float, float, float],
                int, int,
            ]
        ],
        speed_mph: float | None,
        speed_method: str | None = None,
        rate_fps: float | None = None,
        rate_source: str | None = None,
    ) -> None:
        """Write per-frame trajectory to events/pass_<pid>.jsonl.

        Format:
          line 0: manifest dict with pass-level metadata
          line 1+: one dict per frame, with PTS-anchored timestamp (raw,
                   diagnostic — the camera's PTS is untrustworthy), ground
                   pixel, bbox, homography-projected (X, Y), instantaneous
                   speed, and the running average since the first sample
                   (the canonical per-frame speed; final value matches the
                   headline).

        When `rate_fps` is given (cadence path), per-frame times for
        t_rel / v_inst / v_running are reconstructed from the received-frame
        sequence numbers: t_i = (seq_i - seq_0) / rate_fps. The raw PTS `ts`
        is still written per row for diagnostics, and clip_start_ts /
        clip_end_ts stay PTS-anchored — the recorder ring and the browser's
        video.currentTime mapping are indexed by PTS.
        """
        if self._homog is None or not traj:
            return
        out_dir = self._recordings_dir.parent / "events"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pass_{pid}.jsonl"

        projected: list[
            tuple[float, float, float, tuple, float, float, int, int]
        ] = []
        for ts, u, v, bb, seq, epoch in traj:
            X, Y = self._homog.project(u, v)
            projected.append(
                (float(ts), float(u), float(v), bb, float(X), float(Y),
                 int(seq), int(epoch))
            )

        seq0 = projected[0][6]
        epoch_span = projected[-1][7] - projected[0][7]
        use_cadence = rate_fps is not None and rate_fps > 0 and epoch_span == 0

        def t_of(idx: int) -> float:
            """Per-frame time: cadence-reconstructed when available, else
            the raw (unreliable) PTS."""
            if use_cadence:
                return (projected[idx][6] - seq0) / rate_fps
            return projected[idx][0]

        rows: list[dict] = []
        t0 = t_of(0)
        cum_dist = 0.0
        for i, (ts, u, v, bb, X, Y, seq, _epoch) in enumerate(projected):
            in_grid = _in_grid(X, Y)
            t_i = t_of(i)
            v_inst_mph: float | None = None
            v_running_mph: float | None = None
            if i > 0:
                _, _, _, _, X_p, Y_p, _, _ = projected[i - 1]
                dt_step = t_i - t_of(i - 1)
                d_step = ((X - X_p) ** 2 + (Y - Y_p) ** 2) ** 0.5
                cum_dist += d_step
                # v_inst: only when both this and prior frame are in-grid.
                # On the cadence time base a seq gap (missed detection or
                # dropped frame) correctly lengthens dt_step.
                if in_grid and _in_grid(X_p, Y_p) and dt_step > 0:
                    v_inst_mph = (d_step / dt_step) * _MPH_PER_MPS
                # v_running: cumulative distance / cumulative time. Starts
                # at min-samples and is the canonical speed signal; final
                # value matches the headline on the cadence path.
                cum_dt = t_i - t0
                if (i + 1) >= _MIN_RUNNING_SAMPLES and cum_dt > 0:
                    v_running_mph = (cum_dist / cum_dt) * _MPH_PER_MPS
            rows.append({
                "frame": i,
                "ts": ts,
                "t_rel": t_i - t0,
                "u": u,
                "v": v,
                "bbox": list(bb),
                "X": X,
                "Y": Y,
                "in_grid": in_grid,
                "seq": seq,
                "v_inst_mph": v_inst_mph,
                "v_running_mph": v_running_mph,
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
            # legend that reads from manifest. Same canonical speed. Stays None
            # (-> JSON null) when a guard rejected the speed; never NaN, which
            # would emit the bare `NaN` token and break the browser JSON parse
            # (and thus the whole chart) for every rejected pass.
            "v_homog_mph": speed_mph,
            "speed_mph": speed_mph,
            "speed_method": speed_method,  # 'cadence_seq' | 'running_avg' | None
            # Cadence time-base diagnostics (None / 0 on the legacy path).
            # rate_source: 'received' (live rolling rate) | 'registry'
            # (camwatch-cameras measured cadence — warm-up fallback) | None.
            "rate_fps": rate_fps,
            "rate_source": rate_source,
            "seq_first": seq0,
            "seq_last": projected[-1][6],
            "epoch_span": epoch_span,
            "n_frames": len(rows),
            "frame_size": list(self._homog.frame_size),
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
                "Run `python -m camwatch.calibrate pick-roi` first."
            )
            return

        log.info(
            "capture worker starting (threshold=%.1f mph, trigger=grid-entry/exit)",
            self._cfg.alert_threshold_mph,
        )

        # YOLO sees only the ROI crop (Detector translates bboxes back to
        # full-frame coords). A subsequent in-grid check via the homography
        # filters out anything that's inside the ROI but off-road (e.g.,
        # driveway). At 4K this is a ~5x compute reduction vs. full-frame.
        det = Detector(
            weights=self._cfg.model.weights,
            device=self._cfg.model.device,
            classes=self._cfg.model.classes,
            conf=self._cfg.model.conf,
            iou=self._cfg.model.iou,
            roi=cal.roi,
            conf_per_class=self._cfg.model.conf_per_class,
        )
        recorder = ClipRecorder(
            self._recordings_dir,
            pre_seconds_before_a=self._cfg.clip_margin_s,
            post_seconds_after_b=self._cfg.clip_margin_s,
            homography=self._homog,
            grid_x_min=_GRID_X_MIN, grid_x_max=_GRID_X_MAX,
            grid_y_min=_GRID_Y_MIN, grid_y_max=_GRID_Y_MAX,
            grid_tolerance_m=_GRID_TOLERANCE_M,
            min_running_samples=_MIN_RUNNING_SAMPLES,
        )
        self._recorder = recorder
        if self._homog is None:
            log.error(
                "homography missing; grid-based trigger cannot run. "
                "Calibrate the camera in the camwatch-cameras registry "
                "(its self-contained tooling) and bump the dependency."
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
            self._preview.configure(cal.roi)
            self._preview.set_grid(
                self._homog,
                _GRID_X_MIN, _GRID_X_MAX,
                _GRID_Y_MIN, _GRID_Y_MAX,
            )
            self._preview.set_show_grid(self._cfg.preview_show_grid)

        self._stream = open_frame_source(self._cfg, metrics=self._metrics)
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

                # Periodic retention sweep — three phases:
                #   1. clips_days: delete .mp4 only (thumbnail stays, DB row stays)
                #   2. thumbs_days: delete .jpg (archive alarm thumbs), NULL clip_path
                #   3. passes_days: hard-delete row + per-pass jsonl
                if time.monotonic() - last_purge > purge_interval_s:
                    last_purge = time.monotonic()
                    events_dir = self._cfg.events_dir
                    threshold_mph = float(self._cfg.alert_threshold_mph)
                    archive_dir = Path("recordings_archive")

                    days_clips = int(self._cfg.clips_days or 0)
                    if days_clips > 0:
                        clip_items = self._db.passes_with_clip_older_than(days_clips)
                        if clip_items:
                            archive_dir.mkdir(parents=True, exist_ok=True)
                        archived_clips = 0
                        removed_clips = 0
                        for _pid, cp, speed in clip_items:
                            p = Path(cp)
                            if not p.exists():
                                continue
                            try:
                                if speed is not None and speed >= threshold_mph:
                                    shutil.move(str(p), archive_dir / p.name)
                                    archived_clips += 1
                                else:
                                    p.unlink(missing_ok=True)
                                    removed_clips += 1
                            except Exception as e:  # noqa: BLE001
                                log.debug("clip cleanup: %s: %s", cp, e)
                        if archived_clips or removed_clips:
                            log.info(
                                "retention: %d alarm clips archived, %d non-alarm deleted "
                                "(clips_days=%d, threshold=%.1f mph)",
                                archived_clips, removed_clips, days_clips, threshold_mph,
                            )

                    days_thumbs = int(self._cfg.thumbs_days or 0)
                    if days_thumbs > 0:
                        thumb_items = self._db.purge_thumbs_older_than(days_thumbs)
                        if thumb_items:
                            archive_dir.mkdir(parents=True, exist_ok=True)
                        archived = 0
                        deleted = 0
                        for _pid, cp, speed in thumb_items:
                            base = cp[:-4] if cp.endswith(".mp4") else cp
                            thumb = base + ".jpg"
                            # Belt-and-braces: archive (alarm) or nuke (non-alarm)
                            # the .mp4 if it survived phase 1.
                            mp4 = Path(cp)
                            if mp4.exists():
                                try:
                                    if speed is not None and speed >= threshold_mph:
                                        shutil.move(str(mp4), archive_dir / mp4.name)
                                    else:
                                        mp4.unlink(missing_ok=True)
                                except Exception as e:  # noqa: BLE001
                                    log.debug("late clip cleanup: %s: %s", cp, e)
                            if speed is not None and speed >= threshold_mph:
                                if Path(thumb).exists():
                                    try:
                                        shutil.move(thumb, archive_dir / Path(thumb).name)
                                        archived += 1
                                    except Exception as e:  # noqa: BLE001
                                        log.debug("archive move %s: %s", thumb, e)
                            else:
                                try:
                                    Path(thumb).unlink(missing_ok=True)
                                    deleted += 1
                                except Exception as e:  # noqa: BLE001
                                    log.debug("thumb cleanup: %s: %s", thumb, e)
                            # Entry/exit spot-check images: always delete, never archive.
                            for side in (".entry.jpg", ".exit.jpg"):
                                try:
                                    Path(base + side).unlink(missing_ok=True)
                                except Exception as e:  # noqa: BLE001
                                    log.debug("anchor cleanup: %s%s: %s", base, side, e)
                        if archived or deleted:
                            log.info(
                                "retention: %d alarm thumbs archived, %d non-alarm deleted "
                                "(thumbs_days=%d, threshold=%.1f mph)",
                                archived, deleted, days_thumbs, threshold_mph,
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
                                base = cp[:-4] if cp.endswith(".mp4") else cp
                                paths_to_unlink.append(cp)
                                paths_to_unlink.append(base + ".jpg")
                                paths_to_unlink.append(base + ".entry.jpg")
                                paths_to_unlink.append(base + ".exit.jpg")
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

                # Night-mode gate. The Reolink E1 switches to monochrome IR
                # in low light, where speed measurements are unreliable
                # (headlight-only detections, motion blur, bbox bottom no
                # longer at the wheel-on-road position). When enabled and
                # active, skip YOLO/trigger/recorder for this frame and
                # update the preview with the raw IR feed (no annotations).
                self._night_mode = night_detector.update(fr.image)
                paused = self._night_mode and self._cfg.pause_at_night
                mp.CAPTURE_PAUSED.set(1.0 if paused else 0.0)
                if paused:
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
                # AND for the per-pass JSONL trajectory log. Hysteresis-gated:
                # don't accumulate until the bbox bottom-center has been
                # strictly inside the calibrated rectangle at least once.
                # After that, samples in the _GRID_TOLERANCE_M slack zone
                # continue to accumulate so bbox jitter at the boundary
                # doesn't drop samples from the speed fit. Bounded by
                # deque(maxlen=200) per track; whole-track GC is piggybacked
                # on the crossing detector's stale-track logic.
                if self._homog is not None:
                    for tr in tracks:
                        tid = int(tr.track_id)
                        gx, gy = tr.ground_point
                        X, Y = self._homog.project(float(gx), float(gy))
                        strict_in = (
                            _GRID_X_MIN <= X <= _GRID_X_MAX
                            and _GRID_Y_MIN <= Y <= _GRID_Y_MAX
                        )
                        if not self._entered_strict.get(tid):
                            if not strict_in:
                                continue
                            self._entered_strict[tid] = True
                        traj = self._trajectories.setdefault(tid, deque(maxlen=200))
                        bb = tuple(float(x) for x in tr.bbox)
                        traj.append(
                            (fr.ts, float(gx), float(gy), bb, fr.seq, fr.epoch)
                        )

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
                        self._entered_strict.pop(tid, None)
                        crossing.reset_in_grid_entry(tid)

                t0 = time.perf_counter() if prof else 0.0
                recorder.push(fr.image, fr.ts, tracks, seq=fr.seq)
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

                    # Reported speed: cadence time base — total projected
                    # distance over (received-frame count / live received
                    # rate). The camera's per-frame PTS is untrustworthy
                    # (see pts_timing_investigation.md); the received-frame
                    # cadence against the local clock is. Seq gaps from
                    # missed detections or dropped frames correctly add
                    # elapsed time. Validated at 3-6% error against
                    # camera-native ground truth.
                    speed_mph: float | None = None
                    speed_method: str | None = None
                    n_speed_samples = 0
                    rate_fps = (
                        self._stream.received_fps()
                        if self._stream is not None else None
                    )
                    rate_source = "received"
                    if rate_fps is None:
                        # Stream warm-up / post-reconnect: the rolling window
                        # has <30s of coverage, so no live rate yet. Fall back
                        # to the registry's measured cadence (ADR-015) instead
                        # of reporting the speed unknown — same camera, same
                        # quantity, measured offline from clean FTP clips.
                        rate_fps = self._registry_cadence_fps
                        rate_source = "registry"
                    traj_for_speed = list(self._trajectories.get(ev.track_id, ()))
                    if self._homog is not None:
                        speed_mph, n_speed_samples = _cadence_speed(
                            traj_for_speed, rate_fps, self._homog,
                        )
                        if speed_mph is not None:
                            speed_method = "cadence_seq"
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
                    # Optional inset for anchor image capture points: shift the
                    # south anchor inward from Y_MIN and the north anchor inward
                    # from Y_MAX, leaving the speed-measurement grid untouched.
                    # Picks the trajectory timestamp whose projected Y is closest
                    # to the target; falls back to the crossing ts when the
                    # trajectory doesn't reach the target or homography is off.
                    entry_anchor_ts: float | None = None
                    exit_anchor_ts: float | None = None
                    south_inset_ft = self._cfg.recorder_south_anchor_inset_ft
                    north_inset_ft = self._cfg.recorder_north_anchor_inset_ft
                    if (south_inset_ft or north_inset_ft) and self._homog is not None:
                        traj = self._trajectories.get(ev.track_id)
                        if traj:
                            samples_xy = [
                                (ts, self._homog.project(u, v)[1])
                                for ts, u, v, _bb, _seq, _epoch in traj
                            ]
                            y_min = min(y for _, y in samples_xy)
                            y_max = max(y for _, y in samples_xy)
                            south_ts: float | None = None
                            north_ts: float | None = None
                            if south_inset_ft > 0:
                                south_target = _GRID_Y_MIN + south_inset_ft * _FT_TO_M
                                if y_min <= south_target:
                                    south_ts = min(samples_xy, key=lambda s: abs(s[1] - south_target))[0]
                            if north_inset_ft > 0:
                                north_target = _GRID_Y_MAX - north_inset_ft * _FT_TO_M
                                if y_max >= north_target:
                                    north_ts = min(samples_xy, key=lambda s: abs(s[1] - north_target))[0]
                            if ev.direction == "N":
                                # N-bound: t_a is south, t_b is north
                                entry_anchor_ts = south_ts
                                exit_anchor_ts = north_ts
                            elif ev.direction == "S":
                                # S-bound: t_a is north, t_b is south
                                entry_anchor_ts = north_ts
                                exit_anchor_ts = south_ts

                    clip_path = recorder.trigger(
                        name=clip_name,
                        focus_track_id=ev.track_id,
                        t_a=ev.t_a,
                        t_b=ev.t_b,
                        speed_mph=speed_mph,
                        record_video=in_range,
                        entry_anchor_ts=entry_anchor_ts,
                        exit_anchor_ts=exit_anchor_ts,
                        rate_fps=rate_fps,
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
                        # Camera provenance (ADR-013): the elected main camera
                        # produced this pass. Stored per row so passes captured
                        # before a main-camera switch keep the right camera.
                        camera=self._cfg.camera.main_id,
                    )
                    # Observability: pass counters + speed distribution. The
                    # dashboard derives passes/hour and p95 speed from these.
                    mp.PASSES.inc(
                        direction=str(ev.direction), method=speed_method or "none",
                    )
                    if speed_mph is not None:
                        mp.PASS_SPEED.observe(speed_mph, direction=str(ev.direction))
                        if speed_mph >= self._cfg.alert_threshold_mph:
                            mp.ALARMS.inc(direction=str(ev.direction))
                    # Hand off to the local enrichment service (non-blocking).
                    # High-confidence local matches will UPDATE the row before
                    # the user ever sees it; low-confidence ones leave the
                    # vehicle_* fields NULL so the existing Opus workflow drains
                    # them in the next batch.
                    if clip_path:
                        self._enrich_pool.submit(
                            self._fire_enrich, pid, ev.direction,
                            captured_at.isoformat(timespec="seconds"),
                        )
                    if speed_mph is not None:
                        speed_str = (
                            f"{speed_mph:.2f} mph "
                            f"(cadence_seq over {n_speed_samples} samples"
                            f" @ {rate_fps:.2f} fps [{rate_source}])"
                        )
                    else:
                        speed_str = (
                            "speed unavailable (too few samples, pass spans "
                            "a reconnect, or trajectory rejected as "
                            "spatially jumped)"
                        )
                    log.info(
                        "pass id=%d track=%d %s %s %s  elapsed=%.3fs  clip=%s",
                        pid, ev.track_id, ev.cls_name, ev.direction,
                        speed_str, ev.elapsed_s, clip_name,
                    )
                    # Persist the full per-frame trajectory + computed v_inst
                    # to events/pass_<pid>.jsonl so the chart and any future
                    # offline analysis has cadence-anchored timing (raw PTS is
                    # also kept per row, for diagnostics). Also records the
                    # canonical speed_mph in the manifest.
                    if self._homog is not None:
                        try:
                            self._save_pass_trajectory_jsonl(
                                pid=pid, ev=ev, traj=traj_for_speed,
                                speed_mph=speed_mph,
                                speed_method=speed_method,
                                rate_fps=rate_fps,
                                rate_source=rate_source,
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
                        self._entered_strict.pop(ev.track_id, None)
        finally:
            recorder.flush()
            log.info("capture worker stopped")
