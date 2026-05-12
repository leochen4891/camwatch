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
        ring_seconds: float = 12.0,
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
        # Ring buffer is sized in REAL TIME (PTS seconds), not in frame count.
        # The previous frame-count sizing (int(ring_seconds * fps)) used the
        # playback fps, not the actual capture fps, so a 7 s × 10 fps = 70-frame
        # ring only held ~4.7 s at the actual 15 fps capture rate. That was
        # smaller than the worst-case trigger latency: when GridCrossingDetector
        # ages a track out, the trigger fires up to max_track_age_s after the
        # car left the grid, so by then the in-grid frames had been evicted
        # and the resulting clip was a fraction of a second of stale frames.
        # ring_seconds default 12 s comfortably covers max_clip cap (5 s) +
        # max_track_age_s (5 s) + pre_a (~0.5 s) + slack.
        self._ring_seconds = float(ring_seconds)
        self._ring: collections.deque[_FrameRec] = collections.deque()
        self._active: list[_ActiveClip] = []

    def push(self, frame: np.ndarray, ts: float, detections: list[Any]) -> None:
        small = self._scale_frame(frame)
        rec = _FrameRec(image=small, ts=ts, detections=list(detections))
        self._ring.append(rec)
        # Time-based eviction: drop frames older than ring_seconds before the
        # newest pushed frame. This is robust to varying capture fps (a frame-
        # count-based deque was the bug behind the truncated-clip regression).
        cutoff = ts - self._ring_seconds
        while self._ring and self._ring[0].ts < cutoff:
            self._ring.popleft()

        completed: list[_ActiveClip] = []
        for clip in self._active:
            # Drop any frame past target_end_ts so a late-firing trigger
            # (e.g., grid-detector age-out, which fires up to max_track_age_s
            # after the car has actually left) doesn't pad the clip with
            # post-exit dead air.
            if rec.ts <= clip.target_end_ts:
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
        # Clamp pre_frames to BOTH ends of the desired window. A late-firing
        # trigger (e.g., grid-detector age-out 5 s after the actual exit)
        # would otherwise sweep up the entire ring buffer's worth of post-
        # exit dead air and bake it into the clip.
        pre_frames = [r for r in self._ring if desired_start <= r.ts <= desired_end]
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
        """Save two JPEGs cropped tight to the focus car: a 320 px-wide
        thumb for list view and an 800 px-wide thumb for the detail view.

        Source frames come from the main-stream ring buffer (2048x1536),
        so both crops are derived from a single high-res image — no async
        upgrade needed. The crop is taken from the raw frame, not the
        overlay-rendered frame written into the .mp4, so thumbnails have
        no bbox/label/line overlays.
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

        crop: np.ndarray | None = None
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
            if cx2 - cx1 >= 80 and cy2 - cy1 >= 60:
                crop = raw[cy1:cy2, cx1:cx2]
        if crop is None:
            crop = clip.frames[best_any[1]].image

        base = clip.path[:-4] if clip.path.endswith(".mp4") else clip.path
        thumb_small = self._resize_to_width(crop, 320)
        thumb_big = self._resize_to_width(crop, 800)
        cv2.imwrite(base + ".jpg", thumb_small, [cv2.IMWRITE_JPEG_QUALITY, 82])
        cv2.imwrite(base + "_big.jpg", thumb_big, [cv2.IMWRITE_JPEG_QUALITY, 85])

    @staticmethod
    def _resize_to_width(frame: np.ndarray, target_w: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if w == target_w:
            return frame
        scale = target_w / w
        return cv2.resize(frame, (target_w, max(1, int(round(h * scale)))))

    def _render(self, rec: _FrameRec, clip: _ActiveClip) -> np.ndarray:
        img = rec.image.copy()
        h, w = img.shape[:2]
        s = self._scale

        # Line A / Line B vertical markers were drawn here historically for
        # the 2-line speed-measurement debug view. The grid overlay rendered
        # by the web player now provides equivalent (and more informative)
        # spatial reference, so the burned-in lines are no longer drawn on
        # the clip itself.

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

        # Single header line: per-frame PTS timestamp. The camera's burned-
        # in wallclock OSD already lives at the bottom of the frame; the PTS
        # gives us a stream-anchored monotonic clock that drift-calibration
        # and offline analysis can key off of.
        _stamp(
            img,
            f"PTS time = {rec.ts:.3f}s",
            (10, 22),
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
