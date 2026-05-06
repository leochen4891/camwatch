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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

# Set ffmpeg options before the first cv2.VideoCapture call. TCP transport
# avoids UDP packet loss; nobuffer + low_delay tell ffmpeg to surface frames
# as soon as they arrive instead of holding a multi-frame jitter buffer.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

import cv2  # noqa: E402  (must come after the env var)
import numpy as np  # noqa: E402

log = logging.getLogger(__name__)


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
    ) -> None:
        self._url = url
        self._reconnect_delay_s = reconnect_delay_s
        self._max_failures = max_consecutive_read_failures
        self._cap: cv2.VideoCapture | None = None
        self._stop = False
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

            # Background reader fills a single-slot buffer with the latest frame.
            latest: list[tuple[np.ndarray, float] | None] = [None]
            latest_lock = threading.Lock()
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
                while not reader_stop.is_set() and not self._stop:
                    ok, image = cap.read()
                    if not ok:
                        reader_failures[0] += 1
                        if reader_failures[0] >= self._max_failures:
                            log.warning(
                                "stream: %d consecutive read failures, reconnecting",
                                reader_failures[0],
                            )
                            return
                        time.sleep(0.01)
                        continue
                    reader_failures[0] = 0
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
                    with latest_lock:
                        latest[0] = (image, ts)
                    new_frame_evt.set()

            reader_thread = threading.Thread(target=reader, name="rtsp-reader", daemon=True)
            reader_thread.start()

            try:
                while not self._stop:
                    if not new_frame_evt.wait(timeout=2.0):
                        if not reader_thread.is_alive():
                            break
                        continue
                    new_frame_evt.clear()
                    with latest_lock:
                        item = latest[0]
                        latest[0] = None
                    if item is None:
                        continue
                    image, ts = item
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
        log.info("stream: opening %s", _redact_url(self._url))
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("cv2.VideoCapture failed to open stream")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
        max_age_s: float = 15.0,
        sample_interval_s: float = 0.25,
        name: str = "ts-stream",
    ) -> None:
        self._stream = RtspStream(url)
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
        for fr in self._stream.frames():
            if self._stop_evt.is_set():
                return
            now = time.monotonic()
            # Sub-sample by wall-clock interval so the buffer stays bounded
            # but still has multiple frames per second of camera time.
            if now - self._last_sample_t < self._sample_interval:
                continue
            self._last_sample_t = now
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
        # ts is monotonic-anchored, so age = now - ts. Drop the prefix that's
        # too old — list is in insertion (== ts) order. Frames from a previous
        # epoch are also dropped here as they age out: their ts isn't in the
        # current epoch's space, but the prefix sweep doesn't care because
        # those frames are necessarily older (by wall-clock) than current ones.
        cutoff = now - self._max_age
        i = 0
        for ts, _epoch, _frame in self._frames:
            if ts >= cutoff:
                break
            i += 1
        if i > 0:
            del self._frames[:i]
