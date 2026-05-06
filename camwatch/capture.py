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

    def stop(self) -> None:
        self._stop = True

    def frames(self) -> Iterator[Frame]:
        seq = 0
        while not self._stop:
            self._open()
            cap = self._cap
            assert cap is not None

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
                    yield Frame(image=image, ts=ts, seq=seq)
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
    """Parallel high-res RTSP reader keyed by the camera's burned-in OSD time.

    The main stream's ffmpeg buffer is deep enough that the "latest" frame
    we read can be many seconds behind the live event we're trying to
    correlate with. Rather than fight that buffering, we lean on the fact
    that the camera burns a 1-second-resolution wall clock into every
    frame: we OCR each sampled frame's OSD, store it under that datetime
    key, and let callers ask "give me the frame whose content was captured
    at time T."

    Frames are sub-sampled (`sample_interval_s`, default 1s) so OCR work
    stays bounded; the buffer evicts entries older than `max_age_s`.
    """

    def __init__(
        self,
        url: str,
        ocr_region: tuple[int, int, int, int],
        ocr_fn: "Callable[[np.ndarray, tuple[int,int,int,int]], object] | None" = None,
        max_age_s: float = 15.0,
        sample_interval_s: float = 1.0,
        name: str = "ts-stream",
    ) -> None:
        self._stream = RtspStream(url)
        self._region = ocr_region
        self._ocr = ocr_fn
        self._max_age = float(max_age_s)
        self._sample_interval = float(sample_interval_s)
        self._name = name
        self._lock = threading.Lock()
        # Indexed by datetime (second-resolution) -> (monotonic_ts, frame).
        self._frames: "dict[Any, tuple[float, np.ndarray]]" = {}
        self._latest_dt: Any = None
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
        self, target_dt: Any, tolerance_s: float = 1.5
    ) -> "tuple[Any, np.ndarray] | None":
        """Return the (datetime, frame) closest to `target_dt` within tolerance."""
        with self._lock:
            best: tuple[float, Any, np.ndarray] | None = None
            for dt, (_mono, frame) in self._frames.items():
                # both naive datetimes; compute |delta|
                if hasattr(target_dt, "tzinfo") and target_dt.tzinfo is not None:
                    cmp = target_dt.replace(tzinfo=None)
                else:
                    cmp = target_dt
                delta = abs((dt - cmp).total_seconds())
                if delta > tolerance_s:
                    continue
                if best is None or delta < best[0]:
                    best = (delta, dt, frame)
        if best is None:
            return None
        return (best[1], best[2])

    def latest_indexed(self) -> "tuple[Any, np.ndarray] | None":
        """Most recent frame for which OCR succeeded (used for diagnostics)."""
        with self._lock:
            if self._latest_dt is None or self._latest_dt not in self._frames:
                return None
            _mono, frame = self._frames[self._latest_dt]
            return (self._latest_dt, frame)

    def _run(self) -> None:
        last_status_log = 0.0
        ocr_attempts = 0
        ocr_failures = 0
        for fr in self._stream.frames():
            if self._stop_evt.is_set():
                return
            now = time.monotonic()
            # Sub-sample: at most one indexed frame per sample_interval.
            if now - self._last_sample_t < self._sample_interval:
                continue
            if self._ocr is None:
                continue
            self._last_sample_t = now
            ocr_attempts += 1
            dt = self._ocr(fr.image, self._region)
            if dt is None:
                ocr_failures += 1
                # Periodically dump a failed crop for forensic inspection.
                if ocr_failures <= 3 or ocr_failures % 50 == 0:
                    import cv2 as _cv2
                    x1, y1, x2, y2 = self._region
                    _cv2.imwrite(
                        f"/tmp/ts_fail_{ocr_failures:03d}.png",
                        fr.image[y1:y2, x1:x2],
                    )
                continue
            # Sanity: the camera clock should be sync'd with ours, and
            # the main-stream RTSP buffer never drifts more than ~25s. Reject
            # OCR readings outside (wall_clock - 60s, wall_clock + 5s).
            # This catches Tesseract hallucinations like 36→55 in low light.
            from datetime import datetime as _dt, timedelta as _td
            now_naive = _dt.now().replace(microsecond=0)
            if dt > now_naive + _td(seconds=5) or now_naive - dt > _td(seconds=60):
                ocr_failures += 1
                if ocr_failures % 25 == 0:
                    log.info(
                        "ts buffer: rejected OCR (wall-clock mismatch) osd=%s now=%s",
                        dt.strftime("%H:%M:%S"), now_naive.strftime("%H:%M:%S"),
                    )
                continue
            with self._lock:
                self._frames[dt] = (now, fr.image)
                self._latest_dt = dt
                self._evict_locked(now)
            # Periodic diagnostic so we can see how far behind real time the
            # main stream is running. Lag = (wall clock now) - (frame's OSD).
            if now - last_status_log >= 10.0:
                last_status_log = now
                from datetime import datetime as _dt
                lag = (_dt.now().replace(microsecond=0) - dt).total_seconds()
                log.info(
                    "ts buffer: latest osd=%s lag=%+.0fs n=%d ocr_ok=%d/%d",
                    dt.strftime("%H:%M:%S"), lag, len(self._frames),
                    ocr_attempts - ocr_failures, ocr_attempts,
                )
                ocr_attempts = 0
                ocr_failures = 0

    def _evict_locked(self, now: float) -> None:
        cutoff = now - self._max_age
        stale = [k for k, (mono, _) in self._frames.items() if mono < cutoff]
        for k in stale:
            self._frames.pop(k, None)
