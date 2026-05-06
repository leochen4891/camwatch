"""RTSP frame source with reconnect.

Camera produces frames at ~20fps; the YOLO+tracker consumer runs at
~6-10fps. Without intervention, ffmpeg's RTSP buffer fills up and we
end up processing frames many seconds old. Two layers of mitigation:

1. Pass low-latency flags to ffmpeg via the OPENCV_FFMPEG_CAPTURE_OPTIONS
   env var (`fflags=nobuffer`, `flags=low_delay`).
2. A background reader thread continuously calls cap.read() and stores
   only the latest frame in a single-slot buffer. The consumer (the
   `frames()` iterator) always sees the most recent frame; everything
   else is silently dropped.

Net: end-to-end latency stays close to the camera's frame interval, and
detection always sees fresh state instead of a 15-second-stale backlog.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

# Set ffmpeg options before the first cv2.VideoCapture call. TCP transport
# avoids UDP packet loss; nobuffer + low_delay tell ffmpeg to surface frames
# as soon as they arrive instead of holding a multi-frame jitter buffer.
# This is the LIVE-CAPTURE default. Streams that don't need low latency
# (like the thumb-upgrader's main-stream buffer) override at open time via
# RtspStream(ffmpeg_options=...) — see _open_with_ffmpeg_options below.
_LIVE_FFMPEG_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _LIVE_FFMPEG_OPTIONS)

import cv2  # noqa: E402  (must come after the env var)
import numpy as np  # noqa: E402

log = logging.getLogger(__name__)

# Lock around env-var-swap during cv2.VideoCapture() open. The FFmpeg
# backend reads OPENCV_FFMPEG_CAPTURE_OPTIONS at open time, so we change
# the env var, open, then restore. The lock keeps two threads opening
# streams with different options from clobbering each other.
_OPEN_LOCK = threading.Lock()


def _open_with_ffmpeg_options(url: str, options: str | None) -> "cv2.VideoCapture":
    """Open a VideoCapture with stream-specific ffmpeg options. If options
    is None, uses whatever's already in OPENCV_FFMPEG_CAPTURE_OPTIONS
    (i.e., the live-capture default)."""
    if options is None:
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    with _OPEN_LOCK:
        prior = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        finally:
            if prior is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prior
    return cap


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
        self._cap: cv2.VideoCapture | None = None
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
        # Lightweight diagnostic. Counts every successful cap.read() at the
        # earliest possible point — the inner reader thread, before any of
        # our consumer-side filtering / sampling / single-slot logic. If
        # this counter ticks at the camera's nominal fps, no frames are
        # being lost between the camera and our cap.read() return; any
        # apparent loss downstream is then caused by our wrapper code, not
        # ffmpeg or the network. Periodically logged with `log_label` so
        # sub and main can be compared side-by-side.
        self._frames_read = 0
        self._log_label = log_label
        # ffmpeg_options is a string in OPENCV_FFMPEG_CAPTURE_OPTIONS format
        # (e.g. "rtsp_transport;tcp|fflags;nobuffer"). None uses the
        # process-wide default (low-latency). Pass a different string to
        # opt out of low-latency mode for offline-ish consumers — the
        # thumb upgrader does this so ffmpeg keeps decoded-frame backlogs
        # rather than discarding all but the latest after a keyframe wait.
        self._ffmpeg_options = ffmpeg_options
        # bufsize → cv2.CAP_PROP_BUFFERSIZE. None skips the .set() call,
        # leaving ffmpeg's default queue depth.
        self._bufsize = bufsize
        # session_epoch increments on each reconnect (each new VideoCapture).
        # Frames yielded from the same RTSP session share an epoch, and that
        # epoch is the validity scope of their ts space — PTS is re-anchored
        # to monotonic at the first frame of every session, so anything that
        # caches a derived value (e.g. a cross-stream offset) must invalidate
        # when the epoch advances.
        self._session_epoch = 0

    @property
    def session_epoch(self) -> int:
        return self._session_epoch

    def stop(self) -> None:
        self._stop = True

    def frames(self) -> Iterator[Frame]:
        seq = 0
        while not self._stop:
            self._open()
            cap = self._cap
            assert cap is not None
            self._session_epoch += 1
            current_epoch = self._session_epoch

            # Background reader writes into a deque(maxlen=queue_size).
            # With maxlen=1 this is identical to a single-slot keep-latest
            # buffer; with maxlen>1 it's a bounded FIFO that evicts the
            # oldest frame on overflow (the same automatic policy the deque
            # gives us). Consumer drains every queued frame on each cycle.
            latest_buf: "deque[tuple[np.ndarray, float]]" = deque(maxlen=self._queue_size)
            buf_lock = threading.Lock()
            new_frame_evt = threading.Event()
            reader_stop = threading.Event()
            reader_failures = [0]

            # Use the stream's PTS (presentation timestamp from RTP) as the
            # frame timestamp, not time.monotonic() at read time. ffmpeg
            # buffers frames and delivers them in bursts, so monotonic at
            # cap.read() return doesn't reflect when the camera captured the
            # frame. PTS does, and it's rock-solid in our diagnostic
            # (camera-side std=0). Crossing-time interpolation depends on
            # accurate inter-frame intervals; this is the load-bearing fix.
            def reader() -> None:
                pts_offset: float | None = None
                last_log_t = time.monotonic()
                last_log_count = self._frames_read
                last_log_pts: float | None = None
                while not reader_stop.is_set() and not self._stop:
                    ok, image = cap.read()
                    if not ok:
                        reader_failures[0] += 1
                        if reader_failures[0] >= self._max_failures:
                            log.warning(
                                "stream %s: %d consecutive read failures, reconnecting",
                                self._log_label, reader_failures[0],
                            )
                            return
                        time.sleep(0.01)
                        continue
                    reader_failures[0] = 0
                    self._frames_read += 1
                    pts_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                    if pts_s > 0:
                        # Re-anchor PTS (which is stream-relative) to the
                        # monotonic clock once at startup so downstream code
                        # that mixes our `ts` with monotonic deadlines stays
                        # internally consistent.
                        if pts_offset is None:
                            pts_offset = time.monotonic() - pts_s
                        ts = pts_s + pts_offset
                    else:
                        ts = time.monotonic()
                    # Periodic raw-rate diagnostic. fps_real = frames pulled
                    # per wallclock second from cap.read(); fps_pts = how
                    # fast camera-time advances per wallclock second. If
                    # fps_real ≪ fps_pts we're getting bursts after gaps
                    # (frames truly missed somewhere upstream of our read).
                    # If fps_real ≈ fps_pts ≈ camera nominal fps, no loss.
                    now = time.monotonic()
                    if now - last_log_t >= 10.0:
                        dt_wall = now - last_log_t
                        dn = self._frames_read - last_log_count
                        fps_real = dn / dt_wall
                        if last_log_pts is not None and pts_s > 0:
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
                        last_log_pts = pts_s if pts_s > 0 else last_log_pts
                    with buf_lock:
                        latest_buf.append((image, ts))
                    new_frame_evt.set()

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
            "stream: opening %s (ffmpeg_opts=%s, bufsize=%s)",
            _redact_url(self._url),
            "default" if self._ffmpeg_options is None else "custom",
            self._bufsize,
        )
        cap = _open_with_ffmpeg_options(self._url, self._ffmpeg_options)
        if not cap.isOpened():
            raise RuntimeError("cv2.VideoCapture failed to open stream")
        if self._bufsize is not None:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self._bufsize)
        self._cap = cap

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


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
    frame's PTS (`cv2.CAP_PROP_POS_MSEC`) anchored to `time.monotonic()`
    at the first frame so each buffered frame's `ts` reflects its true
    capture time. Callers look up by ts directly — same time domain as
    `Frame.ts` from RtspStream, so a trigger event's t_a/t_b can be
    passed straight in.

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
        # decoded-frame backlog so cap.read() drains it one frame at a
        # time. The upgrader is async w.r.t. the live trigger, so the
        # extra latency this adds is acceptable.
        # Also drop CAP_PROP_BUFFERSIZE=1 so ffmpeg's default frame queue
        # depth (which absorbs short bursts) is in effect.
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
