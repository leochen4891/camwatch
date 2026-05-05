"""RTSP frame source with reconnect."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

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
            failures = 0
            while not self._stop:
                ok, image = self._cap.read()
                if not ok:
                    failures += 1
                    if failures >= self._max_failures:
                        log.warning(
                            "stream: %d consecutive read failures, reconnecting",
                            failures,
                        )
                        break
                    continue
                failures = 0
                seq += 1
                yield Frame(image=image, ts=time.monotonic(), seq=seq)
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
