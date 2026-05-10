"""RTSP frame source with hardware-accelerated decode and reconnect.

The Reolink E1 camera produces frames at ~12fps (sub) / ~20fps (main); the
YOLO+tracker consumer runs at ~6-10fps. Without intervention, the network
+ codec pipeline buffers fill up and we end up processing frames many
seconds old. Three layers of mitigation:

1. Hardware decode via Apple's VideoToolbox media engine. H.264 NAL units
   come off the RTSP socket and are handed straight to the SoC's dedicated
   decode block, which produces NV12 frames at near-zero CPU cost. PyAV
   does the NV12→BGR24 conversion via libswscale; that plus the surface
   download is a few percent of one core, vs ~40% for full software decode.
2. Low-latency demux flags via av.open() options — `rtsp_transport=tcp`
   for reliability + `fflags=nobuffer`, `flags=low_delay` to surface
   frames as soon as they arrive instead of holding a multi-frame jitter
   buffer. The thumb upgrader's main-stream buffer overrides these via
   RtspStream(ffmpeg_options=...) when it wants ffmpeg to retain a
   decoded-frame backlog.
3. A background reader thread continuously demuxes packets and decodes
   them into a deque(maxlen=queue_size). With queue_size=1 (the live
   default) older frames are silently overwritten — the consumer always
   sees the latest. With queue_size>1 it's a bounded FIFO that drops
   the oldest on overflow.

Net: end-to-end latency stays close to the camera's frame interval,
detection always sees fresh state, and the CPU headroom freed by hardware
decode goes to YOLO+tracking.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import av
import av.codec.hwaccel
import av.error
import numpy as np

log = logging.getLogger(__name__)

# Live-capture demux/decode options. Same intent as the prior
# OPENCV_FFMPEG_CAPTURE_OPTIONS string ("rtsp_transport;tcp|fflags;nobuffer|flags;low_delay").
_LIVE_OPTIONS: dict[str, str] = {
    "rtsp_transport": "tcp",
    "fflags": "nobuffer",
    "flags": "low_delay",
}

# Single shared HWAccel config — there's no per-stream state, just the
# device-type request, so multiple containers can use the same instance.
_VT_HWACCEL = av.codec.hwaccel.HWAccel(
    device_type="videotoolbox",
    allow_software_fallback=True,
)


def _parse_ffmpeg_options(s: str | None) -> dict[str, str]:
    """Parse an OPENCV_FFMPEG_CAPTURE_OPTIONS-style string ("k1;v1|k2;v2")
    into a dict for av.open(options=...). None returns the live default
    (low-latency)."""
    if s is None:
        return dict(_LIVE_OPTIONS)
    out: dict[str, str] = {}
    for pair in s.split("|"):
        if not pair or ";" not in pair:
            continue
        k, v = pair.split(";", 1)
        out[k] = v
    return out


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
        ffmpeg_options: str | None = None,
        bufsize: int | None = 1,
        log_label: str = "stream",
        queue_size: int = 1,
    ) -> None:
        self._url = url
        self._reconnect_delay_s = reconnect_delay_s
        self._max_failures = max_consecutive_read_failures
        self._stop = False
        # queue_size controls the reader→consumer hand-off:
        #   1  → single-slot semantics (always keep latest, drop older).
        #        Right for live capture where freshness > completeness.
        #   N  → bounded FIFO (deque(maxlen=N), oldest evicted on overflow).
        #        Right for non-live consumers like the thumb upgrader, where
        #        we want to retain every frame ffmpeg gives us during a
        #        backlog burst — single-slot would silently drop ~120 of 140
        #        burst frames since the consumer can't iterate fast enough.
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        self._queue_size = queue_size
        # Lightweight diagnostic. Counts every successfully decoded frame at
        # the earliest possible point — the inner reader thread, before any
        # of our consumer-side filtering / sampling / single-slot logic. If
        # this counter ticks at the camera's nominal fps, no frames are
        # being lost between the camera and our decoder; any apparent loss
        # downstream is then in our wrapper code, not the network or codec.
        # Periodically logged with `log_label` so sub and main can be
        # compared side-by-side.
        self._frames_read = 0
        self._log_label = log_label
        # ffmpeg_options is a string in OPENCV_FFMPEG_CAPTURE_OPTIONS format
        # (e.g. "rtsp_transport;tcp|fflags;nobuffer"). None uses the
        # live-capture default (low-latency). The thumb upgrader passes
        # "rtsp_transport;tcp" alone — without nobuffer/low_delay — so
        # ffmpeg retains a decoded-frame backlog rather than discarding all
        # but the latest after a keyframe wait.
        self._ffmpeg_options = ffmpeg_options
        # bufsize was OpenCV's CAP_PROP_BUFFERSIZE knob. PyAV exposes no
        # equivalent; the parameter is retained for API compatibility but
        # has no effect under the libav-based pipeline.
        self._bufsize = bufsize
        # session_epoch increments on each reconnect (each new container).
        # Frames yielded from the same RTSP session share an epoch, and that
        # epoch is the validity scope of their ts space — PTS is re-anchored
        # to monotonic at the first frame of every session, so anything that
        # caches a derived value (e.g. a cross-stream offset) must invalidate
        # when the epoch advances.
        self._session_epoch = 0
        self._container: "av.container.InputContainer | None" = None

    @property
    def session_epoch(self) -> int:
        return self._session_epoch

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
                time.sleep(self._reconnect_delay_s)
                continue
            container = self._container
            assert container is not None
            vstream = container.streams.video[0]
            self._session_epoch += 1
            current_epoch = self._session_epoch

            # Background reader writes into a deque(maxlen=queue_size). With
            # maxlen=1 this is identical to a single-slot keep-latest buffer;
            # with maxlen>1 it's a bounded FIFO that evicts the oldest frame
            # on overflow (the deque's automatic policy). Consumer drains
            # every queued frame on each cycle.
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
                                return
                            continue
                        for frame in decoded:
                            reader_failures[0] = 0
                            self._frames_read += 1
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
                            # NV12 (HW surface) → BGR24 numpy via libswscale.
                            # Implicit IOSurface download + colorspace convert.
                            image = frame.to_ndarray(format="bgr24")
                            with buf_lock:
                                latest_buf.append((image, ts))
                            new_frame_evt.set()
                except av.error.FFmpegError as e:
                    log.warning(
                        "stream %s: demux/decode error, reconnecting (%s)",
                        self._log_label, e,
                    )

            reader_thread = threading.Thread(target=reader, name="rtsp-reader", daemon=True)
            reader_thread.start()

            try:
                while not self._stop:
                    # Drain any frames the reader has queued. With queue_size=1
                    # this returns 0 or 1 items (single-slot semantics); with
                    # queue_size>1 it returns up to queue_size items in
                    # arrival order. Yielding inside the lock is wrong (would
                    # block the reader for the duration of the consumer's
                    # work), so we copy out then yield.
                    with buf_lock:
                        items = list(latest_buf)
                        latest_buf.clear()
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
            "stream: opening %s (ffmpeg_opts=%s, hwaccel=videotoolbox)",
            _redact_url(self._url),
            "default" if self._ffmpeg_options is None else "custom",
        )
        opts = _parse_ffmpeg_options(self._ffmpeg_options)
        # timeout=(open_s, read_s). Bounded read timeout means the demux
        # loop will surface an error within ~5s of the camera disconnecting,
        # so reconnect kicks in quickly.
        container = av.open(
            self._url,
            options=opts,
            hwaccel=_VT_HWACCEL,
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


class TimestampedFrameBuffer:
    """Parallel high-res RTSP reader indexed by camera-side PTS.

    The main stream's ffmpeg buffer can deliver frames many seconds late
    relative to the live event we want to correlate with. We use the
    frame's PTS anchored to `time.monotonic()` at the first frame so each
    buffered frame's `ts` reflects its true capture time. Callers look up
    by ts directly — same time domain as `Frame.ts` from RtspStream, so a
    trigger event's t_a/t_b can be passed straight in.

    The PTS counter is stream-relative (resets on each RTSP session
    open), so this stream's anchor is independent of the sub-stream's.
    Pure-PTS matching across the two sessions assumes the camera stamps
    PTS from a shared internal video clock, so two near-simultaneous
    session opens produce offsets that cancel — empirically correct for
    Reolink E1; if it ever drifts, swap to OCR-driven sync.

    Frames are sub-sampled by interval (`sample_interval_s`, default
    0.25s); the buffer evicts entries older than `max_age_s`.
    """

    def __init__(
        self,
        url: str,
        max_age_s: float = 30.0,
        sample_interval_s: float = 0.25,
        name: str = "ts-stream",
    ) -> None:
        # Open the main stream WITHOUT live-capture's nobuffer/low_delay
        # flags. Empirically (scripts/probe_main_gaps.py), ffmpeg with
        # nobuffer flags handles main-stream H.264 by waiting for an IDR
        # then batch-decoding the queued P/B frames and surfacing only the
        # latest — leaving multi-second gaps in the fr.ts values our buffer
        # actually receives. Removing the flags has ffmpeg keep the
        # decoded-frame backlog so the demuxer drains it one packet at a
        # time. The upgrader is async w.r.t. the live trigger, so the
        # extra latency this adds is acceptable.
        self._stream = RtspStream(
            url,
            ffmpeg_options="rtsp_transport;tcp",
            bufsize=None,
            log_label="main",
            # Generous queue to absorb worst-case ffmpeg post-keyframe
            # bursts. Observed: ~35s of camera-time delivered in ~5
            # wallclock seconds → 140 frames at 4fps. 200 leaves
            # headroom in case a longer stall happens. Memory peak per
            # frame at 2048x1536 is ~9.4MB, but the consumer drains
            # ~1000× faster than the reader fills, so peak occupancy
            # stays in the single digits except during a burst.
            queue_size=200,
        )
        self._max_age = float(max_age_s)
        self._sample_interval = float(sample_interval_s)
        self._name = name
        self._lock = threading.Lock()
        # Sorted-by-insertion list of (ts, epoch, frame). Sample interval is
        # uniform-ish so insertion order matches ts order; we don't bisect.
        # Epoch is stored per-frame because RTSP reconnect resets the ts
        # anchor, so a cached cross-stream offset is only valid against
        # frames from the same epoch.
        self._frames: list[tuple[float, int, np.ndarray]] = []
        self._latest_ts: float = 0.0
        self._latest_epoch: int = 0
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._last_sample_t: float = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        self._stream.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def find_frame_at(
        self, target_ts: float, tolerance_s: float = 1.5
    ) -> "tuple[float, np.ndarray] | None":
        """Return the (ts, frame) closest to `target_ts` within tolerance.

        Only frames from the current (latest) epoch are considered — older
        epochs hold ts values from a stale anchor and are not comparable.
        """
        with self._lock:
            current_epoch = self._latest_epoch
            best: tuple[float, float, np.ndarray] | None = None
            for ts, epoch, frame in self._frames:
                if epoch != current_epoch:
                    continue
                delta = abs(ts - target_ts)
                if delta > tolerance_s:
                    continue
                if best is None or delta < best[0]:
                    best = (delta, ts, frame)
        if best is None:
            return None
        return (best[1], best[2])

    def latest_indexed(self) -> "tuple[float, np.ndarray] | None":
        """Most recent buffered frame (used for diagnostics)."""
        with self._lock:
            if not self._frames:
                return None
            ts, _epoch, frame = self._frames[-1]
            return (ts, frame)

    def current_epoch(self) -> int:
        """Latest RTSP session epoch the buffer has seen, or 0 if empty."""
        with self._lock:
            return self._latest_epoch

    def _run(self) -> None:
        last_status_log = 0.0
        n_sampled = 0
        last_sampled_ts: float = -1e18  # so the first frame is always kept
        for fr in self._stream.frames():
            if self._stop_evt.is_set():
                return
            now = time.monotonic()
            # Sub-sample by *camera-time* (fr.ts) rather than monotonic.
            # ffmpeg/RTSP buffer pumping delivers frames in bursts; sampling
            # by monotonic skips most frames within a burst, leaving
            # multi-second gaps in main_ts space exactly where lookups can
            # land. Sampling by fr.ts guarantees ≤ sample_interval_s gaps
            # in the buffer's ts coverage no matter how bursty delivery is.
            if fr.ts - last_sampled_ts < self._sample_interval:
                continue
            last_sampled_ts = fr.ts
            n_sampled += 1
            with self._lock:
                if fr.epoch != self._latest_epoch and self._latest_epoch != 0:
                    log.info(
                        "ts buffer: stream reconnected (epoch %d → %d); ts space reset",
                        self._latest_epoch, fr.epoch,
                    )
                self._frames.append((fr.ts, fr.epoch, fr.image))
                self._latest_ts = fr.ts
                self._latest_epoch = fr.epoch
                self._evict_locked(now)
            if now - last_status_log >= 10.0:
                last_status_log = now
                lag = now - fr.ts
                log.info(
                    "ts buffer: latest_ts=%.3f lag=%+.2fs n=%d sampled=%d epoch=%d",
                    fr.ts, lag, len(self._frames), n_sampled, fr.epoch,
                )
                n_sampled = 0

    def _evict_locked(self, now: float) -> None:
        # Anchor eviction to the buffer's own latest ts rather than to
        # monotonic-now. ffmpeg/RTSP delivers frames with a variable
        # pipeline lag (anywhere from 1s to 15s observed); using
        # monotonic-now as the cutoff causes the buffer's ts coverage to
        # shrink dramatically during high-lag periods (because latest_ts
        # is far behind monotonic-now, but eviction proceeds anyway,
        # cutting into the older end of useful coverage). Anchoring to
        # latest_ts gives stable "last N seconds of camera-time" coverage
        # regardless of lag oscillation.
        if not self._frames:
            return
        latest_ts = self._frames[-1][0]
        cutoff = latest_ts - self._max_age
        i = 0
        for ts, _epoch, _frame in self._frames:
            if ts >= cutoff:
                break
            i += 1
        if i > 0:
            del self._frames[:i]
