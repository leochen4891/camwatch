"""Rolling video clip recorder.

Maintains a ring buffer of the last `pre_seconds` of frames. When `trigger()`
is called, it opens an mp4 writer, dumps the ring buffer to disk, and keeps
appending live frames for `post_seconds` after.

Frames are downscaled before storage to keep memory and file size in check.
The 5MP main stream is overkill for "is this my car?" review.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class _Active:
    writer: cv2.VideoWriter
    remaining_frames: int
    path: str


class ClipRecorder:
    def __init__(
        self,
        recordings_dir: Path,
        fps: int = 10,
        pre_seconds: float = 2.0,
        post_seconds: float = 1.5,
        max_width: int = 1280,
    ) -> None:
        self._dir = Path(recordings_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._pre_n = max(1, int(pre_seconds * fps))
        self._post_n = max(1, int(post_seconds * fps))
        self._max_width = max_width
        self._size: tuple[int, int] | None = None
        self._ring: collections.deque[np.ndarray] = collections.deque(maxlen=self._pre_n)
        self._active: list[_Active] = []

    def push(self, frame: np.ndarray) -> None:
        small = self._scale(frame)
        self._ring.append(small)

        survivors: list[_Active] = []
        for entry in self._active:
            entry.writer.write(small)
            entry.remaining_frames -= 1
            if entry.remaining_frames > 0:
                survivors.append(entry)
            else:
                entry.writer.release()
                log.debug("clip closed: %s", entry.path)
        self._active = survivors

    def trigger(self, name: str) -> str:
        if self._size is None:
            raise RuntimeError("trigger() called before any frames were pushed")
        path = self._dir / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self._fps, self._size)
        if not writer.isOpened():
            log.warning("clip writer failed to open at %s", path)
            return ""
        for f in self._ring:
            writer.write(f)
        self._active.append(_Active(writer=writer, remaining_frames=self._post_n, path=str(path)))
        return str(path)

    def flush(self) -> None:
        for entry in self._active:
            entry.writer.release()
        self._active.clear()

    def _scale(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self._max_width:
            self._size = (w, h)
            return frame
        scale = self._max_width / w
        new_w = self._max_width
        new_h = int(round(h * scale))
        if new_h % 2 == 1:
            new_h -= 1  # mp4v wants even dimensions
        out = cv2.resize(frame, (new_w, new_h))
        self._size = (new_w, new_h)
        return out
