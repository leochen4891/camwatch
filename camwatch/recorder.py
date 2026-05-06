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
from collections.abc import Callable
from dataclasses import dataclass, field
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
    target_end_ts: float  # finalize once a pushed frame has ts >= this
    focus_track_id: int
    line_a_x: int  # in scaled coords
    line_b_x: int  # in scaled coords
    t_a: float
    t_b: float
    speed_mph: float | None = None
    record_video: bool = True  # if False, only the thumbnail JPEG is written
    on_finalize: Callable[[], None] | None = None  # invoked after _write_thumbnail


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
        pre_seconds_before_a: float = 1.0,
        post_seconds_after_b: float = 1.0,
        max_clip_seconds: float = 5.0,
        ring_seconds: float = 7.0,
        max_width: int = 2560,
    ) -> None:
        self._dir = Path(recordings_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._pre_a = float(pre_seconds_before_a)
        self._post_b = float(post_seconds_after_b)
        self._max_clip = float(max_clip_seconds)
        self._max_width = max_width
        self._size: tuple[int, int] | None = None
        self._scale: float = 1.0
        # Ring buffer must hold enough frames that, at trigger time, we can
        # still find a frame at (t_a - pre_seconds_before_a). Worst case is a
        # slow car (long elapsed) plus the pre-roll window. Default 7s gives
        # comfortable headroom over a 5s clip cap with 5s max track age.
        self._pre_n = max(1, int(ring_seconds * fps))
        self._ring: collections.deque[_FrameRec] = collections.deque(maxlen=self._pre_n)
        self._active: list[_ActiveClip] = []

    def push(self, frame: np.ndarray, ts: float, detections: list[Any]) -> None:
        small = self._scale_frame(frame)
        rec = _FrameRec(image=small, ts=ts, detections=list(detections))
        self._ring.append(rec)

        completed: list[_ActiveClip] = []
        for clip in self._active:
            clip.frames.append(rec)
            if rec.ts >= clip.target_end_ts:
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
        speed_mph: float | None = None,
        record_video: bool = True,
        on_finalize: Callable[[], None] | None = None,
    ) -> str:
        if self._size is None:
            raise RuntimeError("trigger() called before any frames were pushed")

        # Default clip range: pre_seconds before the FIRST line crossing through
        # post_seconds after the LAST. For northbound t_a < t_b but for southbound
        # the order flips, so use min/max rather than assuming line A came first.
        first_t = min(t_a, t_b)
        last_t = max(t_a, t_b)
        desired_start = first_t - self._pre_a
        desired_end = last_t + self._post_b
        # If that exceeds the cap, center the crossing window in a max-length clip.
        if desired_end - desired_start > self._max_clip:
            midpoint = (first_t + last_t) / 2.0
            desired_start = midpoint - self._max_clip / 2.0
            desired_end = midpoint + self._max_clip / 2.0

        path = str(self._dir / name)
        pre_frames = [r for r in self._ring if r.ts >= desired_start]
        clip = _ActiveClip(
            path=path,
            frames=pre_frames,
            target_end_ts=desired_end,
            focus_track_id=focus_track_id,
            line_a_x=int(round(line_a_x * self._scale)),
            line_b_x=int(round(line_b_x * self._scale)),
            t_a=t_a,
            t_b=t_b,
            speed_mph=speed_mph,
            record_video=record_video,
            on_finalize=on_finalize,
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
        # Always write the thumbnail; skip the .mp4 when the caller said so
        # (e.g., a pass outside the configured speed-capture range).
        if clip.record_video:
            # Try H.264 (avc1) first; modern browsers play it natively. Fall back to
            # MPEG-4 Part 2 (mp4v) if the OpenCV build doesn't ship libx264 — Chrome
            # won't play those, but at least the file is on disk for ffplay/VLC.
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(clip.path, fourcc, self._fps, self._size)
            if not writer.isOpened():
                log.warning("avc1 fourcc not available, falling back to mp4v at %s", clip.path)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(clip.path, fourcc, self._fps, self._size)
            if not writer.isOpened():
                log.warning("clip writer failed to open at %s", clip.path)
            else:
                for rec in clip.frames:
                    writer.write(self._render(rec, clip))
                writer.release()
        self._write_thumbnail(clip)
        log.debug(
            "clip closed: %s (%d frames, video=%s)",
            clip.path, len(clip.frames), clip.record_video,
        )
        if clip.on_finalize is not None:
            try:
                clip.on_finalize()
            except Exception as e:  # noqa: BLE001
                log.warning("clip on_finalize callback raised: %s", e)

    def _write_thumbnail(self, clip: _ActiveClip) -> None:
        """Save a clean (no overlay) JPEG cropped tight to the focus car.

        The crop uses the focus track's bbox at midcrossing time, but the
        IMAGE used is the raw frame from the ring buffer rather than the
        overlay-rendered frame written into the mp4. So thumbnails have no
        bbox, no labels, no lines drawn on them.
        """
        if not clip.frames:
            return
        midpoint = (clip.t_a + clip.t_b) / 2.0

        # Find the rec closest to midpoint where the focus track is detected.
        best_with_focus: tuple[float, int, tuple[float, float, float, float]] | None = None
        best_any: tuple[float, int] = (float("inf"), 0)
        for i, rec in enumerate(clip.frames):
            diff = abs(rec.ts - midpoint)
            if diff < best_any[0]:
                best_any = (diff, i)
            for d in rec.detections:
                if getattr(d, "track_id", None) == clip.focus_track_id:
                    bbox = getattr(d, "bbox", None)
                    if bbox is not None:
                        if best_with_focus is None or diff < best_with_focus[0]:
                            best_with_focus = (diff, i, bbox)
                    break

        target_w = 320
        if best_with_focus is not None:
            _, idx, bbox = best_with_focus
            raw = clip.frames[idx].image
            fh, fw = raw.shape[:2]
            s = self._scale
            bx1, by1, bx2, by2 = (v * s for v in bbox)
            bw = bx2 - bx1
            bh = by2 - by1
            pad_x = max(bw * 0.6, 40)
            pad_y = max(bh * 0.7, 40)
            cx1 = max(0, int(round(bx1 - pad_x)))
            cy1 = max(0, int(round(by1 - pad_y)))
            cx2 = min(fw, int(round(bx2 + pad_x)))
            cy2 = min(fh, int(round(by2 + pad_y)))
            if cx2 - cx1 < 80 or cy2 - cy1 < 60:
                thumb = self._fallback_thumb(clip.frames[best_any[1]].image, target_w)
            else:
                thumb = raw[cy1:cy2, cx1:cx2]
                th, tw = thumb.shape[:2]
                if tw != target_w:
                    scale = target_w / tw
                    thumb = cv2.resize(thumb, (target_w, max(1, int(round(th * scale)))))
        else:
            thumb = self._fallback_thumb(clip.frames[best_any[1]].image, target_w)

        thumb_path = clip.path[:-4] + ".jpg" if clip.path.endswith(".mp4") else clip.path + ".jpg"
        cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 82])

    @staticmethod
    def _fallback_thumb(frame: np.ndarray, target_w: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= target_w:
            return frame
        scale = target_w / w
        return cv2.resize(frame, (target_w, int(round(h * scale))))

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
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), _GRAY, 1)
                cv2.circle(img, (gx, gy), 3, _GRAY, -1)

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

        # Footer (bottom): time relative to t_a. Kept short so it doesn't
        # overlap the camera's burned-in OSD timestamp at bottom-center, which
        # the verifier tool needs to read.
        rel_t = rec.ts - clip.t_a
        _stamp(
            img,
            f"t={rel_t:+.3f}s",
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
