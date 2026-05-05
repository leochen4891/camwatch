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
from collections.abc import Iterator
from dataclasses import dataclass

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

            def reader() -> None:
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
                    with latest_lock:
                        latest[0] = (image, time.monotonic())
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
