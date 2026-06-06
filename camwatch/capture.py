"""RTSP frame source with hardware-accelerated decode and reconnect.

The Reolink E1 camera produces main-stream frames at ~15-20fps; the YOLO+
tracker consumer on the 3060 Ti runs faster than the source on average
(yolo_track p50 ≈ 35ms vs frame interval ≈ 67ms at 15fps). Goal: process
every decoded frame rather than dropping any, so clips have full motion
detail and the speed regression has every available sample. Three layers
keep this working:

1. Hardware decode via NVDEC (libav's `cuda` hwaccel). H.264 NAL units come
   off the RTSP socket and are handed straight to the GPU's dedicated decode
   block, which produces frames at near-zero CPU cost. PyAV does the
   YUV→BGR24 conversion via libswscale; that plus the surface download is a
   few percent of one core, vs ~40% for full software decode.
   `allow_software_fallback=True` on every config means a missing hwaccel
   silently degrades to software instead of crashing.
2. Low-latency demux flags via av.open() options — `rtsp_transport=tcp`
   for reliability + `fflags=nobuffer`, `flags=low_delay` to surface
   frames as soon as they arrive instead of holding a multi-frame jitter
   buffer.
3. A background reader thread continuously demuxes packets and decodes
   them into a bounded FIFO (`deque(maxlen=queue_size)`). With the
   default 150 (≈10s at 15fps, ~1.4 GB peak RAM at 2048×1536 BGR), a
   typical Reolink burst-and-gap cycle is absorbed without dropping
   anything — consumer drains the burst during the following gap. The
   deque only evicts oldest-first when its capacity is exceeded, which
   only happens during sustained backlog (GPU contention, system load
   spike) — a real problem the warning log surfaces explicitly.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import av
import av.codec.hwaccel
import av.error
import cv2
import numpy as np

if TYPE_CHECKING:
    from .config import Config
    from .metrics import MetricsCollector

log = logging.getLogger(__name__)

# Live-capture demux/decode options. Same intent as the prior
# OPENCV_FFMPEG_CAPTURE_OPTIONS string ("rtsp_transport;tcp|fflags;nobuffer|flags;low_delay").
_LIVE_OPTIONS: dict[str, str] = {
    "rtsp_transport": "tcp",
    "fflags": "nobuffer",
    "flags": "low_delay",
}


def _make_hwaccel() -> "tuple[av.codec.hwaccel.HWAccel | None, str]":
    """Pick the platform's hardware decoder. Returns (HWAccel-or-None, name)
    where name is just a label for logging — PyAV's HWAccel doesn't expose
    its device_type as an attribute, so we keep it alongside.

    Falls back to software automatically if the requested backend isn't
    available in the linked libav build (`allow_software_fallback=True`).
    """
    if sys.platform == "darwin":
        return av.codec.hwaccel.HWAccel(
            device_type="videotoolbox",
            allow_software_fallback=True,
        ), "videotoolbox"
    if sys.platform.startswith("linux"):
        return av.codec.hwaccel.HWAccel(
            device_type="cuda",
            allow_software_fallback=True,
        ), "cuda"
    return None, "software"


_HWACCEL, _HWACCEL_NAME = _make_hwaccel()


@dataclass
class Frame:
    image: np.ndarray
    ts: float
    seq: int
    epoch: int = 0  # Bumps on each RTSP reconnect; ts space resets per epoch.


class RtspStream:
    def __init__(
        self,
        url: str,
        reconnect_delay_s: float = 2.0,
        max_consecutive_read_failures: int = 30,
        log_label: str = "stream",
        metrics: "MetricsCollector | None" = None,
        # 150 frames ≈ 10s at 15fps, ~1.4 GB peak RAM at 2048×1536 BGR.
        # Sized to absorb Reolink's burst-and-gap delivery pattern without
        # frame loss given the 3060 Ti's headroom; if the queue ever fills
        # up it means processing is genuinely behind, not just bursty —
        # the warning log surfaces that.
        queue_size: int = 150,
        # Backlog-depth warning fires when drained depth stays above
        # `warn_threshold * queue_size` for `warn_sustain_s` consecutive
        # consumer cycles. Defaults: half the queue, sustained 5s. A
        # backlog of 75 frames means we're 5s behind real-time — worth
        # investigating before we lose anything.
        warn_threshold: float = 0.5,
        warn_sustain_s: float = 5.0,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        self._url = url
        self._reconnect_delay_s = reconnect_delay_s
        self._max_failures = max_consecutive_read_failures
        self._stop = False
        # Lightweight diagnostic. Counts every successfully decoded frame at
        # the earliest possible point — the inner reader thread, before any
        # of our consumer-side filtering / sampling logic. If this counter
        # ticks at the camera's nominal fps, no frames are being lost
        # between the camera and our decoder; any apparent loss downstream
        # is then in our wrapper code, not the network or codec.
        self._frames_read = 0
        # Monotonic arrival stamp of every decoded frame, for the rolling
        # received-frame rate (`received_fps`). The camera's RTP timestamps
        # are not trustworthy (see pts_timing_investigation.md), so the
        # cadence-based speed path times passes against this wall-clock rate
        # instead. Sized to cover >60s at any plausible frame rate.
        self._arrivals: "deque[float]" = deque(maxlen=2048)
        self._arrivals_lock = threading.Lock()
        self._log_label = log_label
        self._metrics = metrics
        self._queue_size = int(queue_size)
        self._warn_threshold_depth = max(1, int(queue_size * float(warn_threshold)))
        self._warn_sustain_s = float(warn_sustain_s)
        # session_epoch increments on each reconnect (each new container).
        # Frames yielded from the same RTSP session share an epoch, and that
        # epoch is the validity scope of their ts space — PTS is re-anchored
        # to monotonic at the first frame of every session, so anything that
        # caches a derived value must invalidate when the epoch advances.
        self._session_epoch = 0
        self._container: "av.container.InputContainer | None" = None

    @property
    def session_epoch(self) -> int:
        return self._session_epoch

    def received_fps(self, window_s: float = 60.0) -> "float | None":
        """Rolling received-frame rate over the trailing `window_s` seconds,
        measured against the local monotonic clock (independent of the
        camera's unreliable RTP timestamps).

        Returns None during warm-up: fewer than 2 frames in the window, or
        less than ~30s of coverage. The coverage floor matters: right after a
        (re)connect the camera flushes a catch-up burst of buffered frames,
        which inflates a young window by ~5-10% (observed: 14.5 vs the steady
        13.7 twenty seconds after a restart). 30s dilutes that transient to
        a few percent; by 60s it has aged out entirely.
        """
        now = time.monotonic()
        with self._arrivals_lock:
            stamps = [t for t in self._arrivals if t >= now - window_s]
        if len(stamps) < 2:
            return None
        span = stamps[-1] - stamps[0]
        if span < 30.0:
            return None
        return (len(stamps) - 1) / span

    def stop(self) -> None:
        self._stop = True

    def frames(self) -> Iterator[Frame]:
        seq = 0
        while not self._stop:
            try:
                self._open()
            except av.error.FFmpegError as e:
                # Most commonly ConnectionRefusedError when the camera is
                # rebooting (Reolink runs a nightly maintenance reboot
                # ~02:00). Without this, the exception propagates up
                # through capture_worker.run() and kills the worker for
                # the rest of the day. Sleep and retry the open instead.
                log.warning(
                    "stream %s: open failed, retrying in %.1fs (%s)",
                    self._log_label, self._reconnect_delay_s, e,
                )
                if self._metrics is not None:
                    self._metrics.record_reconnect("open_failed")
                time.sleep(self._reconnect_delay_s)
                continue
            container = self._container
            assert container is not None
            vstream = container.streams.video[0]
            self._session_epoch += 1
            current_epoch = self._session_epoch

            # Background reader writes into a bounded FIFO deque. Under
            # normal load it sits near-empty (consumer outpaces source).
            # During Reolink's burst phases it grows by a few frames and
            # drains during the following gap. Only sustained overflow
            # (depth at maxlen) drops oldest frames — and `_check_backlog`
            # below warns well before we reach that point.
            latest_buf: "deque[tuple[np.ndarray, float]]" = deque(maxlen=self._queue_size)
            buf_lock = threading.Lock()
            new_frame_evt = threading.Event()
            reader_stop = threading.Event()
            reader_failures = [0]

            # Use the stream's PTS (presentation timestamp from RTP) as the
            # frame timestamp, not time.monotonic() at decode time. ffmpeg
            # buffers frames and delivers them in bursts, so monotonic at
            # decode return doesn't reflect when the camera captured the
            # frame. PTS does, and it's rock-solid in our diagnostic
            # (camera-side std=0). Crossing-time interpolation depends on
            # accurate inter-frame intervals; this is the load-bearing fix.
            def reader() -> None:
                pts_offset: float | None = None
                last_log_t = time.monotonic()
                last_log_count = self._frames_read
                last_log_pts: float | None = None
                saw_keyframe = False
                # Wallclock arrival time of the previous decoded frame, used
                # to surface bursty delivery to the metrics collector. Stays
                # None until the second frame so the first frame's "gap"
                # (which would just be the keyframe-wait warmup) doesn't
                # pollute the bucket's max.
                last_arrival_mono: float | None = None
                try:
                    for packet in container.demux(vstream):
                        if reader_stop.is_set() or self._stop:
                            return
                        # RTSP streams start mid-GOP; VideoToolbox refuses
                        # P-frames before the first IDR. Software decoders
                        # silently swallow them; HW does not.
                        if not saw_keyframe:
                            if not packet.is_keyframe:
                                continue
                            saw_keyframe = True
                        try:
                            decoded = packet.decode()
                        except av.error.FFmpegError as e:
                            reader_failures[0] += 1
                            if reader_failures[0] >= self._max_failures:
                                log.warning(
                                    "stream %s: %d consecutive decode failures, "
                                    "reconnecting (%s)",
                                    self._log_label, reader_failures[0], e,
                                )
                                if self._metrics is not None:
                                    self._metrics.record_reconnect("decode_failures")
                                return
                            continue
                        for frame in decoded:
                            reader_failures[0] = 0
                            self._frames_read += 1
                            # Arrival stamp for received_fps(). Recorded for
                            # every decoded frame regardless of whether the
                            # metrics collector is attached — the cadence
                            # speed path depends on this rate.
                            arrival_mono = time.monotonic()
                            with self._arrivals_lock:
                                self._arrivals.append(arrival_mono)
                            if self._metrics is not None:
                                self._metrics.record_frame(self._log_label)
                                if last_arrival_mono is not None:
                                    self._metrics.record_frame_gap(
                                        self._log_label,
                                        arrival_mono - last_arrival_mono,
                                    )
                                last_arrival_mono = arrival_mono
                            pts_s = float(frame.time) if frame.pts is not None else None
                            if pts_s is not None:
                                # Re-anchor PTS (which is stream-relative) to
                                # the monotonic clock once at startup so
                                # downstream code that mixes our `ts` with
                                # monotonic deadlines stays internally
                                # consistent.
                                if pts_offset is None:
                                    pts_offset = time.monotonic() - pts_s
                                ts = pts_s + pts_offset
                            else:
                                ts = time.monotonic()
                            # Periodic raw-rate diagnostic. fps_real = frames
                            # decoded per wallclock second; fps_pts = how
                            # fast camera-time advances per wallclock second.
                            # If fps_real ≪ fps_pts we're getting bursts
                            # after gaps (frames truly missed somewhere
                            # upstream of our decoder). If fps_real ≈
                            # fps_pts ≈ camera nominal fps, no loss.
                            now = time.monotonic()
                            if now - last_log_t >= 10.0:
                                dt_wall = now - last_log_t
                                dn = self._frames_read - last_log_count
                                fps_real = dn / dt_wall if dt_wall else 0.0
                                if last_log_pts is not None and pts_s is not None:
                                    fps_pts = (pts_s - last_log_pts) / dt_wall
                                    log.info(
                                        "stream %s: raw read fps=%.2f, pts advance=%.2fx "
                                        "(total frames=%d in this session)",
                                        self._log_label, fps_real, fps_pts,
                                        self._frames_read,
                                    )
                                else:
                                    log.info(
                                        "stream %s: raw read fps=%.2f (initial)",
                                        self._log_label, fps_real,
                                    )
                                last_log_t = now
                                last_log_count = self._frames_read
                                if pts_s is not None:
                                    last_log_pts = pts_s
                            # HW surface → BGR24 numpy via libswscale.
                            # Implicit GPU→CPU download + colorspace convert.
                            image = frame.to_ndarray(format="bgr24")
                            with buf_lock:
                                latest_buf.append((image, ts))
                            new_frame_evt.set()
                except av.error.FFmpegError as e:
                    log.warning(
                        "stream %s: demux/decode error, reconnecting (%s)",
                        self._log_label, e,
                    )
                    if self._metrics is not None:
                        self._metrics.record_reconnect("demux_error")

            reader_thread = threading.Thread(target=reader, name="rtsp-reader", daemon=True)
            reader_thread.start()

            # Sustained-backlog detector. `over_since` holds the monotonic
            # time at which the drained depth first crossed
            # `_warn_threshold_depth` and has stayed there continuously.
            # The warning fires once each time the sustain window is met,
            # then resets — so a chronically-overloaded box keeps emitting
            # at the cadence the warning condition itself defines.
            over_since: float | None = None
            try:
                while not self._stop:
                    # Drain everything the reader has queued. items is a
                    # FIFO snapshot — under typical load len(items) is 1
                    # or 2; during a Reolink burst it can be 3-5; only
                    # under genuine backlog does it climb higher. Yielding
                    # inside the lock is wrong (would block the reader for
                    # the duration of the consumer's work), so we copy out
                    # then yield.
                    with buf_lock:
                        items = list(latest_buf)
                        latest_buf.clear()
                    depth = len(items)
                    if self._metrics is not None:
                        self._metrics.record_queue_depth(self._log_label, depth)
                    now = time.monotonic()
                    if depth >= self._warn_threshold_depth:
                        if over_since is None:
                            over_since = now
                        elif now - over_since >= self._warn_sustain_s:
                            log.warning(
                                "stream %s: backlog depth=%d (>= %d) sustained for "
                                "%.1fs — consumer is falling behind real-time; "
                                "investigate GPU contention or system load",
                                self._log_label, depth,
                                self._warn_threshold_depth, now - over_since,
                            )
                            over_since = now  # rate-limit: re-warn after each sustain window
                    else:
                        over_since = None
                    if not items:
                        if not new_frame_evt.wait(timeout=2.0):
                            if not reader_thread.is_alive():
                                break
                            continue
                        new_frame_evt.clear()
                        continue
                    new_frame_evt.clear()
                    for image, ts in items:
                        seq += 1
                        yield Frame(image=image, ts=ts, seq=seq, epoch=current_epoch)
                    if not reader_thread.is_alive():
                        break
            finally:
                reader_stop.set()
                reader_thread.join(timeout=2.0)
                self._close()
            if not self._stop:
                time.sleep(self._reconnect_delay_s)

    def _open(self) -> None:
        log.info(
            "stream: opening %s (hwaccel=%s)",
            _redact_url(self._url),
            _HWACCEL_NAME,
        )
        # timeout=(open_s, read_s). Bounded read timeout means the demux
        # loop will surface an error within ~5s of the camera disconnecting,
        # so reconnect kicks in quickly.
        container = av.open(
            self._url,
            options=dict(_LIVE_OPTIONS),
            hwaccel=_HWACCEL,
            timeout=(10.0, 5.0),
        )
        self._container = container

    def _close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None


def _redact_url(url: str) -> str:
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":", 1)[0] if ":" in creds else creds
        return f"{scheme}://{user}:***@{host}"
    return url


class StaticFrameStream:
    """Dev/test source that loops a single JPEG at a fixed rate.

    Same shape as `RtspStream` (frames() iterator, stop(), session_epoch),
    so the capture worker can swap one for the other without caring which.
    No reader thread, no reconnect — just a sleep loop emitting copies of
    the loaded image with monotonic `ts` values that advance at `fps`.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float = 20.0,
        log_label: str = "static",
        metrics: "MetricsCollector | None" = None,
    ) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"static frame source not found: {self._path}")
        img = cv2.imread(str(self._path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2 failed to decode static frame: {self._path}")
        self._image = img
        self._interval = 1.0 / float(fps)
        self._log_label = log_label
        self._metrics = metrics
        self._stop = False
        self._session_epoch = 1
        # Same rolling arrival record as RtspStream so dev mode exercises
        # the real cadence-speed path (converges to the configured fps).
        self._arrivals: "deque[float]" = deque(maxlen=2048)
        self._arrivals_lock = threading.Lock()

    @property
    def session_epoch(self) -> int:
        return self._session_epoch

    received_fps = RtspStream.received_fps

    def stop(self) -> None:
        self._stop = True

    def frames(self) -> Iterator[Frame]:
        log.info(
            "static stream: looping %s (%dx%d) at %.1f fps",
            self._path, self._image.shape[1], self._image.shape[0],
            1.0 / self._interval,
        )
        seq = 0
        next_t = time.monotonic()
        last_arrival_mono: float | None = None
        while not self._stop:
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += self._interval
            seq += 1
            ts = time.monotonic()
            with self._arrivals_lock:
                self._arrivals.append(ts)
            if self._metrics is not None:
                self._metrics.record_frame(self._log_label)
                if last_arrival_mono is not None:
                    self._metrics.record_frame_gap(
                        self._log_label, ts - last_arrival_mono,
                    )
                last_arrival_mono = ts
            # copy() so a downstream consumer that mutates (e.g. annotation
            # overlay) doesn't corrupt the source frame for the next iteration.
            yield Frame(image=self._image.copy(), ts=ts, seq=seq, epoch=self._session_epoch)


def open_frame_source(
    cfg: "Config",
    metrics: "MetricsCollector | None" = None,
) -> "RtspStream | StaticFrameStream":
    """Pick the live RTSP stream or a static-frame replacement based on cfg."""
    if cfg.camera.static_frame_path:
        return StaticFrameStream(
            cfg.camera.static_frame_path,
            log_label="main",
            metrics=metrics,
        )
    return RtspStream(
        cfg.camera.rtsp_url,
        log_label="main",
        metrics=metrics,
    )
