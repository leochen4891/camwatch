"""Rolling video clip recorder with detection/line overlay.

For each pushed frame we keep the raw image + the detection list. When a
crossing fires `trigger()`, we snapshot the ring buffer (pre-roll) and
keep accumulating frames for `post_seconds` more. Once the post-roll
quota is met, the clip is rendered with overlays and written to mp4.

Overlay draws:
  - both vertical lines, dim before crossed and bright once the trigger
    track's `ground_point` has passed them (timestamp shown next to each)
  - every detection's bbox in gray, plus the trigger track's bbox in red
  - a small dot at each car's `ground_point` (the bbox bottom-center the
    speed math anchors to), large red for the trigger track
  - a header strip with t (relative to t_a) and the total span (t_b - t_a)

Frames are downscaled before storage to keep memory + file size sane.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class _FrameRec:
    image: np.ndarray
    ts: float
    detections: list[Any]  # objects exposing .track_id, .bbox, .ground_point


@dataclass
class _ActiveClip:
    path: str
    frames: list[_FrameRec]
    target_count: int
    focus_track_id: int
    line_a_x: int  # in scaled coords
    line_b_x: int  # in scaled coords
    t_a: float
    t_b: float


_GRAY = (180, 180, 180)
_RED = (0, 0, 255)
_GREEN_BRIGHT = (0, 220, 0)
_BLUE_BRIGHT = (220, 120, 0)
_DIM = (90, 90, 90)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


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
        self._scale: float = 1.0
        self._ring: collections.deque[_FrameRec] = collections.deque(maxlen=self._pre_n)
        self._active: list[_ActiveClip] = []

    def push(self, frame: np.ndarray, ts: float, detections: list[Any]) -> None:
        small = self._scale_frame(frame)
        rec = _FrameRec(image=small, ts=ts, detections=list(detections))
        self._ring.append(rec)

        completed: list[_ActiveClip] = []
        for clip in self._active:
            clip.frames.append(rec)
            if len(clip.frames) >= clip.target_count:
                completed.append(clip)
        for clip in completed:
            self._finalize(clip)
            self._active.remove(clip)

    def trigger(
        self,
        name: str,
        focus_track_id: int,
        line_a_x: int,
        line_b_x: int,
        t_a: float,
        t_b: float,
    ) -> str:
        if self._size is None:
            raise RuntimeError("trigger() called before any frames were pushed")
        path = str(self._dir / name)
        clip = _ActiveClip(
            path=path,
            frames=list(self._ring),  # snapshot of pre-roll
            target_count=len(self._ring) + self._post_n,
            focus_track_id=focus_track_id,
            line_a_x=int(round(line_a_x * self._scale)),
            line_b_x=int(round(line_b_x * self._scale)),
            t_a=t_a,
            t_b=t_b,
        )
        self._active.append(clip)
        return path

    def flush(self) -> None:
        for clip in self._active:
            self._finalize(clip)
        self._active.clear()

    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self._max_width:
            self._size = (w, h)
            self._scale = 1.0
            return frame
        scale = self._max_width / w
        new_w = self._max_width
        new_h = int(round(h * scale))
        if new_h % 2 == 1:
            new_h -= 1  # mp4v wants even dimensions
        self._size = (new_w, new_h)
        self._scale = scale
        return cv2.resize(frame, (new_w, new_h))

    def _finalize(self, clip: _ActiveClip) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(clip.path, fourcc, self._fps, self._size)
        if not writer.isOpened():
            log.warning("clip writer failed to open at %s", clip.path)
            return
        rendered: list[np.ndarray] = []
        for rec in clip.frames:
            img = self._render(rec, clip)
            writer.write(img)
            rendered.append(img)
        writer.release()
        self._write_thumbnail(clip, rendered)
        log.debug("clip closed: %s (%d frames)", clip.path, len(clip.frames))

    def _write_thumbnail(self, clip: _ActiveClip, rendered: list[np.ndarray]) -> None:
        """Save a 320px-wide JPEG of the frame closest to the midpoint between
        the two crossings. Uses the already-rendered overlay frames so it
        matches the mp4 visually."""
        if not rendered:
            return
        # Find the frame whose ts is closest to (t_a + t_b) / 2.
        midpoint = (clip.t_a + clip.t_b) / 2.0
        best_idx = 0
        best_diff = float("inf")
        for i, rec in enumerate(clip.frames):
            d = abs(rec.ts - midpoint)
            if d < best_diff:
                best_diff = d
                best_idx = i
        frame = rendered[best_idx]
        h, w = frame.shape[:2]
        target_w = 320
        if w > target_w:
            scale = target_w / w
            thumb = cv2.resize(frame, (target_w, int(round(h * scale))))
        else:
            thumb = frame
        thumb_path = clip.path.replace(".mp4", ".jpg") if clip.path.endswith(".mp4") else clip.path + ".jpg"
        cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])

    def _render(self, rec: _FrameRec, clip: _ActiveClip) -> np.ndarray:
        img = rec.image.copy()
        h, w = img.shape[:2]
        s = self._scale

        # Lines: dim before this frame's ts crossed them, bright after.
        a_crossed = rec.ts >= clip.t_a
        b_crossed = rec.ts >= clip.t_b
        line_a_color = _GREEN_BRIGHT if a_crossed else _DIM
        line_b_color = _BLUE_BRIGHT if b_crossed else _DIM
        cv2.line(img, (clip.line_a_x, 0), (clip.line_a_x, h), line_a_color, 2)
        cv2.line(img, (clip.line_b_x, 0), (clip.line_b_x, h), line_b_color, 2)
        # Crossing timestamps next to each line (relative to t_a).
        a_label = f"A  +{(clip.t_a - clip.t_a):.3f}s"
        b_label = f"B  +{(clip.t_b - clip.t_a):.3f}s"
        _stamp(img, a_label, (clip.line_a_x + 6, 22), line_a_color)
        _stamp(img, b_label, (clip.line_b_x + 6, 22), line_b_color)

        # Bboxes + ground points
        for d in rec.detections:
            tid = getattr(d, "track_id", None)
            bbox = getattr(d, "bbox", None)
            gp = getattr(d, "ground_point", None)
            if bbox is None or gp is None:
                continue
            x1, y1, x2, y2 = (int(round(v * s)) for v in bbox)
            gx, gy = int(round(gp[0] * s)), int(round(gp[1] * s))
            is_focus = tid == clip.focus_track_id
            if is_focus:
                cv2.rectangle(img, (x1, y1), (x2, y2), _RED, 2)
                cv2.circle(img, (gx, gy), 6, _RED, -1)
                cv2.circle(img, (gx, gy), 6, _WHITE, 1)
                _stamp(img, f"id={tid}", (x1, max(15, y1 - 6)), _RED, scale=0.6, thickness=2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), _GRAY, 1)
                cv2.circle(img, (gx, gy), 3, _GRAY, -1)
                if tid is not None:
                    _stamp(img, f"id={tid}", (x1, max(12, y1 - 4)), _GRAY, scale=0.4, thickness=1)

        # Header (top): focus id + total span
        span = clip.t_b - clip.t_a
        _stamp(
            img,
            f"focus id={clip.focus_track_id}   span (B-A) = {span:.3f}s",
            (10, 22),
            _WHITE,
            scale=0.6,
            thickness=2,
            bg=True,
        )

        # Footer (bottom): time relative to t_a
        rel_t = rec.ts - clip.t_a
        _stamp(
            img,
            f"t = {rel_t:+.3f}s  (relative to line A)",
            (10, h - 12),
            _WHITE,
            scale=0.6,
            thickness=2,
            bg=True,
        )
        return img


def _stamp(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
    bg: bool = False,
) -> None:
    if bg:
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), _BLACK, -1)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)
