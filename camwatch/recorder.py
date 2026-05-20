"""Rolling video clip recorder with detection overlay.

For each pushed frame we keep the raw image + the detection list. When a
crossing fires `trigger()`, we snapshot the ring buffer (pre-roll) and
keep accumulating frames for `post_seconds` more. Once the post-roll
quota is met, the clip is rendered with overlays and written to mp4.

Overlay draws:
  - every detection's bbox in gray, plus the trigger track's bbox in red
  - a small dot at each car's `ground_point` (the bbox bottom-center the
    speed math anchors to), large red for the trigger track
  - the 5 ft homography grid (yellow) burned into the frame
  - a header strip with t (relative to t_a) and the total span (t_b - t_a)

Frames are downscaled before storage to keep memory + file size sane.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
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
    t_a: float
    t_b: float
    speed_mph: float | None = None
    record_video: bool = True  # if False, only the thumbnail JPEG is written
    on_finalize: Callable[[], None] | None = None  # invoked after _write_thumbnail
    # Optional overrides for the entry/exit anchor image timestamps. When set,
    # the anchor picker targets these instead of t_a/t_b. Used to inset the
    # anchor capture point away from grid edges (e.g., to avoid a tree right
    # at the south crossing). t_a/t_b are still the speed/midpoint references.
    entry_anchor_ts: float | None = None
    exit_anchor_ts: float | None = None


_GRAY = (180, 180, 180)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GRID_YELLOW = (0, 255, 255)   # bright yellow grid lines (BGR); matches preview
_TRAIL_RED = (0, 0, 255)       # focus track's bottom-center trail (same as bbox)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

_FT_TO_M = 0.3048
_MPH_PER_MPS = 2.2369362920544


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
        # Burn-in overlay: pass the homography + grid bounds and each clip
        # frame will get the measurement grid (thin yellow), the focus
        # track's ground-point trail, and a running-stats label above the
        # focus bbox. Leave homography=None to disable the burn-in.
        homography: Any | None = None,
        grid_x_min: float = 0.0,
        grid_x_max: float = 0.0,
        grid_y_min: float = 0.0,
        grid_y_max: float = 0.0,
        grid_tolerance_m: float = 0.5,
        min_running_samples: int = 5,
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
        self._homog = homography
        self._grid_x_min = float(grid_x_min)
        self._grid_x_max = float(grid_x_max)
        self._grid_y_min = float(grid_y_min)
        self._grid_y_max = float(grid_y_max)
        self._grid_tol = float(grid_tolerance_m)
        self._min_running = int(min_running_samples)
        # Precomputed yellow grid polylines, in scaled clip pixel coords.
        # Built lazily on the first push() (we need to know _scale first).
        self._grid_polylines: list[np.ndarray] = []
        self._grid_built: bool = False
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
        if not self._grid_built and self._homog is not None and self._size is not None:
            self._build_grid_polylines()
            self._grid_built = True
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
        t_a: float,
        t_b: float,
        speed_mph: float | None = None,
        record_video: bool = True,
        on_finalize: Callable[[], None] | None = None,
        entry_anchor_ts: float | None = None,
        exit_anchor_ts: float | None = None,
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
            t_a=t_a,
            t_b=t_b,
            speed_mph=speed_mph,
            record_video=record_video,
            on_finalize=on_finalize,
            entry_anchor_ts=entry_anchor_ts,
            exit_anchor_ts=exit_anchor_ts,
        )
        self._active.append(clip)
        return path

    def flush(self) -> None:
        for clip in self._active:
            self._finalize(clip)
        self._active.clear()

    def _build_grid_polylines(self) -> None:
        """Sample the homography's world rectangle into pixel-space polylines
        (5 ft grid + outer bound), scaled to the clip frame size. Calls into
        Homography.world_polyline() so the grid follows lens distortion."""
        homog = self._homog
        if homog is None:
            return
        s = self._scale
        x_min_ft = int(round(self._grid_x_min / _FT_TO_M))
        x_max_ft = int(round(self._grid_x_max / _FT_TO_M))
        y_min_ft = int(round(self._grid_y_min / _FT_TO_M))
        y_max_ft = int(round(self._grid_y_max / _FT_TO_M))
        polylines: list[np.ndarray] = []

        def scaled_polyline(X1: float, Y1: float, X2: float, Y2: float) -> np.ndarray:
            pl = homog.world_polyline(X1, Y1, X2, Y2)
            if len(pl) == 0:
                return pl
            return (pl.astype(np.float64) * s).astype(np.int32)

        # 5 ft sub-grid (along X and Y)
        for x_ft in range(x_min_ft, x_max_ft + 1, 5):
            X = x_ft * _FT_TO_M
            polylines.append(scaled_polyline(X, self._grid_y_min, X, self._grid_y_max))
        for y_ft in range(y_min_ft, y_max_ft + 1, 5):
            Y = y_ft * _FT_TO_M
            polylines.append(scaled_polyline(self._grid_x_min, Y, self._grid_x_max, Y))
        self._grid_polylines = [pl for pl in polylines if len(pl) >= 2]

    def _compute_focus_state(self, clip: _ActiveClip) -> list[dict]:
        """Walk clip.frames in order, accumulating distance + time for the
        focus track. Hysteresis on grid membership: a sample only starts
        counting once the focus track's bbox bottom-center has been
        strictly inside the grid at least once; after that, samples in
        the _grid_tol slack zone past the strict bounds continue to
        accumulate. The same in_grid flag drives the rendered trail.
        Returns one dict per frame; empty dict if the focus track isn't
        visible / projectable in that frame.
        """
        n = len(clip.frames)
        states: list[dict] = [{} for _ in range(n)]
        if self._homog is None:
            return states
        xmin = self._grid_x_min - self._grid_tol
        xmax = self._grid_x_max + self._grid_tol
        ymin = self._grid_y_min - self._grid_tol
        ymax = self._grid_y_max + self._grid_tol
        cum_dist = 0.0
        first_in_grid_ts: float | None = None
        prev_xy: tuple[float, float] | None = None
        in_grid_count = 0
        entered_strict = False
        s = self._scale
        for i, rec in enumerate(clip.frames):
            focus = None
            for d in rec.detections:
                if getattr(d, "track_id", None) == clip.focus_track_id:
                    focus = d
                    break
            if focus is None or getattr(focus, "ground_point", None) is None:
                continue
            gx_full, gy_full = float(focus.ground_point[0]), float(focus.ground_point[1])
            X, Y = self._homog.project(gx_full, gy_full)
            strict_in = (
                self._grid_x_min <= X <= self._grid_x_max
                and self._grid_y_min <= Y <= self._grid_y_max
            )
            tol_in = (xmin <= X <= xmax and ymin <= Y <= ymax)
            # Hysteresis: strict for entry, tolerance for exit.
            if not entered_strict:
                in_grid = strict_in
                if strict_in:
                    entered_strict = True
            else:
                in_grid = tol_in
            ground_px = (int(round(gx_full * s)), int(round(gy_full * s)))
            cum_dt = (rec.ts - first_in_grid_ts) if first_in_grid_ts is not None else 0.0
            running_mph: float | None = None
            if in_grid:
                if first_in_grid_ts is None:
                    first_in_grid_ts = rec.ts
                    cum_dt = 0.0
                elif prev_xy is not None:
                    dx = X - prev_xy[0]
                    dy = Y - prev_xy[1]
                    cum_dist += (dx * dx + dy * dy) ** 0.5
                prev_xy = (X, Y)
                in_grid_count += 1
                cum_dt = rec.ts - first_in_grid_ts
                if in_grid_count >= self._min_running and cum_dt > 0:
                    running_mph = (cum_dist / cum_dt) * _MPH_PER_MPS
            states[i] = {
                "ground_px": ground_px,
                "X": X,
                "Y": Y,
                "in_grid": in_grid,
                "cum_dist_m": cum_dist,
                "cum_dt_s": cum_dt,
                "running_mph": running_mph,
            }
        return states

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
            states = self._compute_focus_state(clip)
            try:
                self._write_clip_h264(clip, states)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "H.264 encode failed for %s (%s); falling back to mp4v",
                    clip.path, e,
                )
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(clip.path, fourcc, self._fps, self._size)
                if not writer.isOpened():
                    log.warning("clip writer failed to open at %s", clip.path)
                else:
                    for i, rec in enumerate(clip.frames):
                        writer.write(self._render(rec, clip, states, i))
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

    def _write_clip_h264(self, clip: _ActiveClip, states: list[dict]) -> None:
        """Encode the clip's frames as H.264/AVC in an MP4 container.

        Uses libx264 (CPU) at preset=veryfast, crf=23 — fast enough that
        a few-second clip finishes encoding in well under a second on this
        hardware, and the output is browser-playable (which mp4v isn't).
        Raises on any failure so the caller can fall back.

        Sets stream.time_base = 1/fps and frame.pts = i explicitly. Letting
        the encoder auto-assign PTS leaves them all at 0 (libx264 default
        time_base is 1/10240, not 1/fps) — the resulting file's duration
        metadata is 0.1s regardless of frame count, which makes browsers
        stall on playback.
        """
        if not clip.frames or self._size is None:
            return
        # Encode fps = (N-1) / (last_ts - first_ts) so playback matches real
        # capture time regardless of the configured camera rate.
        if len(clip.frames) >= 2:
            span = clip.frames[-1].ts - clip.frames[0].ts
            fps = int(round((len(clip.frames) - 1) / span)) if span > 0 else int(round(self._fps))
            fps = max(1, fps)
        else:
            fps = int(round(self._fps))
        w, h = self._size
        tb = Fraction(1, fps)
        # `movflags=+faststart` rewrites the file at close so the moov atom
        # lives at the front — needed for the browser to seek before the
        # full file is downloaded.
        container = av.open(clip.path, mode="w", options={"movflags": "+faststart"})
        try:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            # All-intra: every frame is a keyframe (g=1, bf=0). The clip is
            # only a few seconds long, so the size penalty is small and the
            # scrubber becomes instantly responsive — every frame is its own
            # seek point.
            stream.options = {
                "preset": "veryfast",
                "crf": "23",
                "g": "1",
                "bf": "0",
            }
            for i, rec in enumerate(clip.frames):
                img = self._render(rec, clip, states, i)
                frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                frame.pts = i
                frame.time_base = tb
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

    def _write_thumbnail(self, clip: _ActiveClip) -> None:
        """Save one JPEG cropped tight to the focus car, sized for the
        detail view (800 px wide). The list view's small thumb element is
        130 px wide and downscales this in the browser — extra bandwidth
        is on the order of ~100 KB per row, acceptable for a single-user
        UI. The crop is taken from the raw frame, not the overlay-rendered
        frame written into the .mp4, so the thumb has no bbox/label/line
        overlays.

        Candidate scoring: each frame containing the focus track gets a
        score combining proximity to the clip midpoint and the maximum
        overlap of any other detected vehicle with the focus car's padded
        crop region. The lowest-scoring frame wins. This avoids picking
        a midpoint frame where an opposite-direction car has driven
        through the crop region (the street is two-lane so the only way
        the focus can be "blocked" is by oncoming traffic).
        """
        if not clip.frames:
            return
        midpoint = (clip.t_a + clip.t_b) / 2.0
        # Trade-off weight: a 50% occluder pushes the picker ~250 ms away
        # from the midpoint, enough to skip a single overlapping frame
        # without abandoning midpoint preference on clean passes.
        occlusion_weight = 0.5

        s = self._scale
        best: tuple[float, int, tuple[float, float, float, float]] | None = None
        best_any: tuple[float, int] = (float("inf"), 0)
        for i, rec in enumerate(clip.frames):
            diff = abs(rec.ts - midpoint)
            if diff < best_any[0]:
                best_any = (diff, i)
            focus_bbox = None
            other_bboxes: list[tuple[float, float, float, float]] = []
            for d in rec.detections:
                bbox = getattr(d, "bbox", None)
                if bbox is None:
                    continue
                if getattr(d, "track_id", None) == clip.focus_track_id:
                    focus_bbox = bbox
                else:
                    other_bboxes.append(bbox)
            if focus_bbox is None:
                continue
            # Padded crop region (matches the crop applied below).
            bx1, by1, bx2, by2 = (v * s for v in focus_bbox)
            bw = bx2 - bx1
            bh = by2 - by1
            pad_x = max(bw * 0.6, 40)
            pad_y = max(bh * 0.7, 40)
            cx1, cy1, cx2, cy2 = bx1 - pad_x, by1 - pad_y, bx2 + pad_x, by2 + pad_y
            crop_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))
            max_overlap = 0.0
            for ob in other_bboxes:
                ox1, oy1, ox2, oy2 = (v * s for v in ob)
                ix1 = max(cx1, ox1)
                iy1 = max(cy1, oy1)
                ix2 = min(cx2, ox2)
                iy2 = min(cy2, oy2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                if iw == 0 or ih == 0:
                    continue
                overlap = (iw * ih) / crop_area
                if overlap > max_overlap:
                    max_overlap = overlap
            score = diff + occlusion_weight * max_overlap
            if best is None or score < best[0]:
                best = (score, i, focus_bbox)

        crop: np.ndarray | None = None
        if best is not None:
            _, idx, bbox = best
            raw = clip.frames[idx].image
            fh, fw = raw.shape[:2]
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
        thumb = self._resize_to_width(crop, 800)
        cv2.imwrite(base + ".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Entry + exit anchor images: closest focus-visible frame at-or-after
        # t_a and at-or-before t_b respectively. Useful for spot-checking that
        # the trigger fired at the right scene moment without playing the clip.
        # entry/exit_anchor_ts overrides shift the anchor capture point inward
        # from the grid edge (e.g., to dodge a tree right at the south crossing);
        # when unset, fall back to the crossing timestamps as before.
        entry_ts = clip.entry_anchor_ts if clip.entry_anchor_ts is not None else clip.t_a
        exit_ts = clip.exit_anchor_ts if clip.exit_anchor_ts is not None else clip.t_b
        entry_crop = self._pick_anchor_crop(clip, entry_ts, prefer="after")
        if entry_crop is not None:
            cv2.imwrite(
                base + ".entry.jpg",
                self._resize_to_width(entry_crop, 800),
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
        exit_crop = self._pick_anchor_crop(clip, exit_ts, prefer="before")
        if exit_crop is not None:
            cv2.imwrite(
                base + ".exit.jpg",
                self._resize_to_width(exit_crop, 800),
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )

    def _pick_anchor_crop(
        self,
        clip: _ActiveClip,
        target_ts: float,
        prefer: str,        # "after" = first focus-visible frame at-or-after target_ts;
                            # "before" = last focus-visible frame at-or-before target_ts
    ) -> np.ndarray | None:
        """Return a tight focus-bbox crop from the frame nearest target_ts
        in the requested direction. None if the focus track is not visible
        in any clip frame, or if the resulting crop is degenerately small."""
        s = self._scale
        best: tuple[float, int, tuple[float, float, float, float]] | None = None
        # Frames before target_ts get a 10x score penalty when prefer=="after"
        # (and vice versa) — they still win if nothing on the preferred side
        # contains the focus track, but lose to any same-side candidate.
        side_penalty = 10.0
        for i, rec in enumerate(clip.frames):
            focus_bbox = None
            for d in rec.detections:
                if getattr(d, "track_id", None) == clip.focus_track_id:
                    focus_bbox = getattr(d, "bbox", None)
                    break
            if focus_bbox is None:
                continue
            dt = rec.ts - target_ts
            if prefer == "after":
                score = dt if dt >= 0 else -dt * side_penalty
            else:  # "before"
                score = -dt if dt <= 0 else dt * side_penalty
            if best is None or score < best[0]:
                best = (score, i, focus_bbox)
        if best is None:
            return None
        _, idx, bbox = best
        raw = clip.frames[idx].image
        fh, fw = raw.shape[:2]
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
            return None
        return raw[cy1:cy2, cx1:cx2]

    @staticmethod
    def _resize_to_width(frame: np.ndarray, target_w: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if w == target_w:
            return frame
        scale = target_w / w
        return cv2.resize(frame, (target_w, max(1, int(round(h * scale)))))

    def _render(self, rec: _FrameRec, clip: _ActiveClip,
                states: list[dict], i: int) -> np.ndarray:
        img = rec.image.copy()
        h, w = img.shape[:2]
        s = self._scale

        # Thin yellow measurement grid (precomputed in scaled coords).
        for pl in self._grid_polylines:
            cv2.polylines(img, [pl], False, _GRID_YELLOW, 1, cv2.LINE_AA)

        # Trail of the focus track's past in-grid ground points up to this
        # frame. Uses the same hysteresis-aware in_grid flag as the running
        # average, so the trail and the speed accumulator stay in sync.
        trail_pts: list[tuple[int, int]] = []
        for j in range(i + 1):
            st = states[j]
            if st.get("in_grid") and "ground_px" in st:
                trail_pts.append(st["ground_px"])
        if len(trail_pts) >= 2:
            arr = np.array(trail_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [arr], False, _TRAIL_RED, 2, cv2.LINE_AA)
            for p in trail_pts:
                cv2.circle(img, p, 3, _TRAIL_RED, -1, cv2.LINE_AA)

        # Bboxes + ground points
        focus_bbox_top: tuple[int, int] | None = None
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
                focus_bbox_top = (x1, y1)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), _GRAY, 1)
                cv2.circle(img, (gx, gy), 3, _GRAY, -1)

        # Running stats label above the focus bbox. d/t go on one line and
        # V wraps to a second line below so the label fits within the frame
        # even on the right edge.
        st = states[i]
        if focus_bbox_top is not None and st.get("ground_px") is not None:
            cum_d = st.get("cum_dist_m", 0.0)
            cum_t = st.get("cum_dt_s", 0.0)
            mph = st.get("running_mph")
            scale, thickness = 1.2, 4
            (_, th_ref), _ = cv2.getTextSize("Hg", _FONT, scale, thickness)
            line_dy = 3 * th_ref  # vertical gap between stacked baselines
            if mph is not None:
                line1 = f"d={cum_d:.2f} m  t={cum_t:.2f} s"
                line2 = f"V={mph:.1f} mph"
            elif st.get("in_grid"):
                line1 = f"d={cum_d:.2f} m  t={cum_t:.2f} s"
                line2 = "V=…"
            else:
                line1 = "(outside grid)"
                line2 = None
            lx = max(30, focus_bbox_top[0])
            # Bottom-line baseline: leave a one-font-height gap between the
            # bg rect and the bbox top (rect extends th_ref below baseline,
            # so subtract 2*th_ref to get a clean th_ref gap above the box).
            # Floor keeps the top of the stack on-frame.
            ly_floor = 2 * th_ref + (line_dy if line2 is not None else 0) + 4
            ly = max(ly_floor, focus_bbox_top[1] - 2 * th_ref)
            if line2 is not None:
                _stamp(img, line1, (lx, ly - line_dy), _WHITE, scale=scale, thickness=thickness, bg=True)
                _stamp(img, line2, (lx, ly), _WHITE, scale=scale, thickness=thickness, bg=True)
            else:
                _stamp(img, line1, (lx, ly), _WHITE, scale=scale, thickness=thickness, bg=True)

        # Single header line: per-frame PTS timestamp. The camera's burned-
        # in wallclock OSD already lives at the bottom of the frame; the PTS
        # gives us a stream-anchored monotonic clock that drift-calibration
        # and offline analysis can key off of.
        _stamp(
            img,
            f"PTS time = {rec.ts:.3f}s",
            (30, 60),
            _WHITE,
            scale=1.2,
            thickness=4,
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
        m = th  # background padding equals the rendered font height
        cv2.rectangle(img, (x - m, y - th - m), (x + tw + m, y + m), _BLACK, -1)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)
