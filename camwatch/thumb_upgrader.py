"""Background thumbnail upgrader.

Upgrades a pass's thumbnail from the sub-stream-derived crop to a
higher-resolution crop pulled from a parallel high-res RTSP stream.
The main stream's ffmpeg buffer can deliver frames seconds late, so we
buffer sampled main frames keyed by their PTS-anchored monotonic ts
(`Frame.ts` from `RtspStream`) and look up by the trigger's t_b/t_a.

Cross-stream offset: sub_ts and main_ts are each anchored to
`time.monotonic()` at their stream's first frame. ffmpeg's initial RTSP
buffer depth differs between streams, so the two ts spaces are offset
by a constant (~+1-2s on Reolink E1, with main running ahead of sub).
We learn the offset once via OCR on a recent main frame plus the first
trigger's (sub_ts, wallclock) pair, cache it for the session, and add
it to every subsequent trigger's target ts before lookup.

Per pass:

  1. Compute target_main_ts = trigger.sub_ts + cross_stream_offset.
  2. Wait until the buffer holds a frame with ts ≥ target_main_ts.
  3. Find the buffered main frame whose ts is closest within ±1.5s.
  4. Run YOLO once on that frame.
  5. Pick the detection of the focus track's class with highest IoU
     against the projected sub-stream bbox; fall back to nearest center.
  6. Crop with padding, atomic-rename the JPEG over the existing
     thumbnail.

If anything fails (no matching frame, no detection of the right class,
write error), the existing sub-stream thumbnail stays untouched.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .capture import TimestampedFrameBuffer
from .config import ModelConfig
from .db import Database
from .detect import Detector
from .digit_matcher import DigitMatcher

log = logging.getLogger(__name__)

# Pixel rectangle of the OSD timestamp on the Reolink E1 main stream
# (2560x1920) with the OSD at bottom-center. Used only for the one-time
# cross-stream offset calibration.
# Main resolution is 2048x1536 (4:3). OSD timestamp spans roughly
# x=657..1360 in the bottom band (auto-detected from bright-pixel runs).
# If you change the camera's main resolution again, re-detect by
# inspecting a bottom-band crop.
_OSD_REGION_MAIN = (650, 1469, 1370, 1517)


@dataclass
class _Job:
    pass_id: int
    thumb_path: str
    focus_cls_name: str
    sub_bbox: tuple[float, float, float, float]
    sub_frame_size: tuple[int, int]  # (w, h) of the sub-stream frame
    target_ts: float                  # sub-stream PTS-anchored monotonic ts at trigger
    target_wallclock_unix: float      # unix seconds at trigger (fallback when drift_sub_override is None)
    sub_epoch: int                    # sub-stream session epoch when ts was captured
    # If set, overrides the (target_ts - target_wallclock_unix) computation
    # of drift_sub. Carries the value capture_worker derived from a sub-
    # stream OSD-tick — bias-free, unlike datetime.now() which includes the
    # full sub-stream pipeline lag.
    drift_sub_override: float | None = None


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
        # cross_stream_offset = main_ts - sub_ts for the same camera-instant.
        # Computed lazily on the first job by OCR'ing a recent main frame.
        # The cached value is only valid for the (sub_epoch, main_epoch) pair
        # under which it was calibrated; either stream reconnecting bumps an
        # epoch and forces a recalibration on the next lookup.
        self._cross_stream_offset: float | None = None
        self._cached_sub_epoch: int = -1
        self._cached_main_epoch: int = -1
        # Template-based OCR for the calibration step. Loaded once. The
        # templates were collected from the main stream's OSD, so this
        # matcher is specific to that stream's font/resolution.
        try:
            self._main_matcher: DigitMatcher | None = DigitMatcher("templates/main")
        except (FileNotFoundError, ValueError) as e:
            log.warning(
                "DigitMatcher disabled (templates/main missing or invalid: %s); "
                "calibration will silently fall back to no-OCR and offset will "
                "stay None forever — collect templates first",
                e,
            )
            self._main_matcher = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._buffer = TimestampedFrameBuffer(
            url=self._rtsp_url,
            max_age_s=30.0,
            # Tight sub-sampling so the buffer holds ~10 main frames per
            # second of monotonic time. ffmpeg delivers frames in bursts;
            # a coarser interval can leave 1+ second gaps in main_ts space
            # exactly where a trigger lookup might land. 0.1s gives ~10
            # samples per second with negligible CPU/memory cost.
            sample_interval_s=0.1,
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
        target_wallclock: datetime,
        sub_epoch: int,
        drift_sub_override: float | None = None,
    ) -> None:
        target_wallclock_unix = target_wallclock.timestamp()
        job = _Job(
            pass_id=pass_id,
            thumb_path=thumb_path,
            focus_cls_name=focus_cls_name,
            sub_bbox=sub_bbox,
            sub_frame_size=sub_frame_size,
            target_ts=target_ts,
            target_wallclock_unix=target_wallclock_unix,
            sub_epoch=sub_epoch,
            drift_sub_override=drift_sub_override,
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

        # The cached cross_stream_offset is only valid for the (sub_epoch,
        # main_epoch) pair under which it was computed. If either stream
        # reconnected since calibration, the ts spaces are reset and the
        # offset is meaningless — recalibrate.
        main_epoch = self._buffer.current_epoch()
        if (
            self._cross_stream_offset is None
            or job.sub_epoch != self._cached_sub_epoch
            or main_epoch != self._cached_main_epoch
        ):
            if self._cross_stream_offset is not None:
                log.info(
                    "thumb upgrade: epoch changed (sub %d→%d, main %d→%d); "
                    "recalibrating offset",
                    self._cached_sub_epoch, job.sub_epoch,
                    self._cached_main_epoch, main_epoch,
                )
            offset = self._calibrate_offset(job)
            if offset is None:
                log.info(
                    "thumb upgrade: cross-stream offset calibration failed — "
                    "%s; will retry on next pass",
                    thumb_name,
                )
                self._db.set_thumb_upgrade_status(job.pass_id, "failed")
                return
            self._cross_stream_offset = offset
            self._cached_sub_epoch = job.sub_epoch
            self._cached_main_epoch = main_epoch

        # Map sub-anchored target ts into main-anchored ts space.
        target_main_ts = job.target_ts + self._cross_stream_offset

        # Main stream lags sub by ffmpeg buffer depth. Wait until the buffer
        # has reached target_main_ts before looking up.
        self._wait_until_matching_frame_available(target_main_ts, timeout_s=45.0)
        # Tolerance allows for some sub-sampling jitter and any residual
        # offset error. With cross_stream_offset now derived from OSD-tick
        # midpoints on both streams, the residual should be <100ms; the
        # 1.5s window mostly absorbs main-stream burst-delivery gaps in
        # the sampled buffer.
        match = self._buffer.find_frame_at(target_main_ts, tolerance_s=1.5)
        if match is None:
            latest = self._buffer.latest_indexed()
            latest_str = f"{latest[0]:.3f}" if latest else "n/a"
            # Diagnostic: report the closest main_ts (without tolerance)
            # so we can tell whether we missed by a hair or by a mile.
            with self._buffer._lock:  # noqa: SLF001
                current_epoch = self._buffer._latest_epoch  # noqa: SLF001
                same_epoch_ts = [
                    ts for ts, ep, _ in self._buffer._frames  # noqa: SLF001
                    if ep == current_epoch
                ]
            closest = min(
                same_epoch_ts, key=lambda t: abs(t - target_main_ts), default=None,
            )
            closest_str = (
                f"{closest:.3f} (Δ={closest - target_main_ts:+.3f}s)"
                if closest is not None else "n/a"
            )
            log.info(
                "thumb upgrade: no main frame matching ts=%.3f within ±1.5s "
                "(latest=%s closest=%s offset=%+.3fs) — %s",
                target_main_ts, latest_str, closest_str,
                self._cross_stream_offset, thumb_name,
            )
            self._db.set_thumb_upgrade_status(job.pass_id, "failed")
            return
        matched_ts, frame = match
        h, w = frame.shape[:2]
        log.info(
            "thumb upgrade: matched main ts=%.3f for trigger sub_ts=%.3f "
            "(target_main=%.3f offset=%+.3f Δ=%+.3fs %dx%d) — %s",
            matched_ts, job.target_ts, target_main_ts, self._cross_stream_offset,
            matched_ts - target_main_ts, w, h, thumb_name,
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

    def _compute_drift_main_via_tick(self) -> float | None:
        """Find the camera-instant of an OSD second-tick on the main stream
        and use it to derive `drift_main = main_ts(tick) - tick_unix`.

        OCR'ing a single main frame gives only 1-second-resolution wallclock
        (the OSD second the frame was captured in), so the derived drift has
        ±0.5s error from a midpoint approximation. By scanning multiple
        consecutive frames and finding two adjacent frames whose OSDs differ
        by 1 second, we know the true OSD-tick (the camera-instant when the
        next second begins) sits between their main_ts values. Using their
        midpoint cuts the drift error to ~half a frame interval (~33ms at
        15fps), which is plenty to land the matched frame on the right
        camera-instant.

        Returns None if not enough current-epoch frames OCR'd to clear the
        wallclock sanity check, or if no OSD-tick was observed in the
        scanned window. Caller should retry on the next job."""
        if self._buffer is None or self._main_matcher is None:
            return None
        with self._buffer._lock:  # noqa: SLF001
            current_epoch = self._buffer._latest_epoch  # noqa: SLF001
            candidates = [
                (ts, frame) for ts, epoch, frame in self._buffer._frames  # noqa: SLF001
                if epoch == current_epoch
            ]
        if len(candidates) < 2:
            return None
        # OCR each candidate (oldest → newest so consecutive pairs are
        # naturally adjacent); apply wallclock sanity check on every read.
        from datetime import datetime as _dt, timedelta as _td
        now_local = _dt.now()
        observed: list[tuple[float, "_dt"]] = []  # (main_ts, OSD_dt)
        for main_ts, main_frame in candidates:
            dt = self._main_matcher.read_timestamp(main_frame, _OSD_REGION_MAIN)
            if dt is None:
                continue
            if abs(dt - now_local) > _td(seconds=60):
                continue
            observed.append((main_ts, dt))
        if len(observed) < 2:
            log.info(
                "thumb upgrade: only %d OCR'd main frames, need >=2 for tick "
                "calibration; will retry on next job", len(observed),
            )
            return None
        # Sort by main_ts ascending (should already be, but defensive) and
        # find the FIRST adjacent pair with a 1-second OSD jump.
        observed.sort(key=lambda r: r[0])
        for (ts_old, dt_old), (ts_new, dt_new) in zip(observed, observed[1:]):
            delta_seconds = (dt_new - dt_old).total_seconds()
            if delta_seconds == 1.0:
                # Tick: dt_new began at camera-instant (ts_old + ts_new) / 2
                # plus or minus half a frame interval.
                tick_main_ts = (ts_old + ts_new) / 2.0
                tick_wallclock_unix = dt_new.timestamp()
                drift_main = tick_main_ts - tick_wallclock_unix
                log.info(
                    "thumb upgrade: drift_main=%+.3fs from tick %s→%s "
                    "(ts_old=%.3f ts_new=%.3f gap=%.3fs)",
                    drift_main, dt_old.strftime("%H:%M:%S"),
                    dt_new.strftime("%H:%M:%S"),
                    ts_old, ts_new, ts_new - ts_old,
                )
                return drift_main
        # No tick observed across the buffered window. This happens when the
        # buffer holds <1s of OCR-readable frames; caller will retry next
        # time and the buffer will hold more by then.
        log.info(
            "thumb upgrade: no OSD-tick found across %d OCR'd main frames "
            "(spans %.2fs); will retry on next job",
            len(observed), observed[-1][0] - observed[0][0],
        )
        return None

    def _calibrate_offset(self, job: _Job) -> float | None:
        """Compute cross_stream_offset = drift_main - drift_sub.

        drift_main is derived from an OSD second-tick observed in the main
        buffer (precise camera-instant ±33ms).

        drift_sub: prefer the bias-free value capture_worker derived from
        a sub-stream OSD-tick (passed in `job.drift_sub_override`). Fall
        back to (target_ts - target_wallclock_unix) only if the sub-stream
        calibration hasn't completed yet — that fallback carries the full
        sub-stream pipeline-lag bias (~2s) and will produce off-by-2s
        matches, but it's better than no calibration at all and the next
        trigger after sub calibration completes will recompute correctly.
        """
        if self._main_matcher is None:
            log.warning(
                "thumb upgrade: no DigitMatcher loaded; cannot calibrate offset"
            )
            return None
        drift_main = self._compute_drift_main_via_tick()
        if drift_main is None:
            return None
        if job.drift_sub_override is not None:
            drift_sub = job.drift_sub_override
            drift_sub_source = "sub-tick"
        else:
            drift_sub = job.target_ts - job.target_wallclock_unix
            drift_sub_source = "datetime.now() (BIASED)"
        offset = drift_main - drift_sub
        log.info(
            "thumb upgrade: calibrated cross_stream_offset=%+.3fs "
            "(drift_main=%.3f drift_sub=%.3f via %s)",
            offset, drift_main, drift_sub, drift_sub_source,
        )
        return offset

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
