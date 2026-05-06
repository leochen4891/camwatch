"""Verify a recorded pass's speed using OSD ticks in its clip.

Why we don't OCR the OSD: at 640×480 sub-stream resolution the burned-in
digits are ~10 px tall, anti-aliased on a busy outdoor scene. Tesseract
can't read them reliably even with upscaling. Browsers can interpolate
them legibly for human eyes; cv2 sees the raw pixels.

What we do instead — pixel-difference tick detection:

  Within a single OSD-second the digit pixels don't change at all (the
  camera draws the same string). The only frame-to-frame variation in the
  seconds region is background noise (trees, etc.). At a tick the digit
  pixels change dramatically (e.g., "24" → "25" flips many pixels between
  bright and dark). We count *bright-pixel flips* between adjacent frames
  rather than mean-abs-diff: tree noise rarely crosses the bright
  threshold so it doesn't contribute, while a digit tick flips dozens of
  pixels' bright-state at once. Spikes in the flip count → ticks →
  frames per second → camera's true fps.

We need to know the pixel rectangle holding the seconds digits. It varies
by camera firmware and OSD-position setting (Reolink lets you place the
OSD at top or bottom). Pass `--seconds-region X1,Y1,X2,Y2` on the CLI.
The defaults match the user's two known positions on a 640×480 sub stream.

Usage:
    uv run python scripts/verify_speed.py <pass_id>                     # auto-pick
    uv run python scripts/verify_speed.py <pass_id> --seconds-region 540,5,600,25
    uv run python scripts/verify_speed.py <pass_id> --dump-frames
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from camwatch.config import load_config  # noqa: E402
from camwatch.db import Database  # noqa: E402

MPS_TO_MPH = 2.2369362920544

# Seconds-only sub-rectangle of the OSD on a 640×480 sub-stream clip.
# The OSD position changed mid-day (top → bottom). We pick by clip mtime.
SECONDS_REGION_TOP = (540, 0, 600, 26)
SECONDS_REGION_BOTTOM = (300, 460, 360, 478)


def auto_pick_region(clip_path: Path) -> tuple[int, int, int, int]:
    """Heuristic: looking at the first frame, pick whichever default region
    has more bright (>200) pixels — the OSD is white text and should win
    against either dark trees (top OSD) or green lawn (bottom OSD)."""
    cap = cv2.VideoCapture(str(clip_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return SECONDS_REGION_BOTTOM
    h, w = frame.shape[:2]

    def brightness(region):
        x1, y1, x2, y2 = region
        x2 = min(x2, w)
        y2 = min(y2, h)
        if x2 <= x1 or y2 <= y1:
            return 0
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return int((gray > 200).sum())

    if brightness(SECONDS_REGION_TOP) > brightness(SECONDS_REGION_BOTTOM):
        return SECONDS_REGION_TOP
    return SECONDS_REGION_BOTTOM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pass_id", type=int)
    parser.add_argument(
        "--seconds-region",
        default=None,
        help="OSD seconds-only sub-rect 'x1,y1,x2,y2' (auto-picked by default)",
    )
    parser.add_argument(
        "--tick-threshold",
        type=int,
        default=15,
        help="count of bright-pixel flips above which we declare a tick",
    )
    parser.add_argument(
        "--bright",
        type=int,
        default=200,
        help="grey value above which a pixel is considered 'bright' (digit ink)",
    )
    parser.add_argument("--dump-frames", action="store_true",
                        help="Save first/last clip frames + the seconds-region "
                             "crop for visual inspection")
    args = parser.parse_args()

    cfg = load_config()
    cal = cfg.load_calibration()
    if cal is None:
        raise SystemExit("calibration.yaml missing")
    db = Database()
    p = db.get_pass(args.pass_id)
    if p is None:
        raise SystemExit(f"pass {args.pass_id} not found")
    if not p.clip_path:
        raise SystemExit(f"pass {args.pass_id} has no clip")
    clip_path = Path(p.clip_path)
    if not clip_path.exists():
        raise SystemExit(f"clip not on disk: {clip_path}")

    if args.seconds_region:
        parts = [int(v) for v in args.seconds_region.split(",")]
        if len(parts) != 4:
            raise SystemExit("--seconds-region must be x1,y1,x2,y2")
        region = tuple(parts)
    else:
        region = auto_pick_region(clip_path)

    distance_m = (
        cal.line_distance_m_north if p.direction == "N"
        else cal.line_distance_m_south
    )
    stored_speed = (distance_m / p.elapsed_s) * MPS_TO_MPH if p.elapsed_s > 0 else 0.0

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open clip: {clip_path}")
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_meta = cap.get(cv2.CAP_PROP_FPS)

    print(f"clip: {clip_path}")
    print(f"  meta: {n_meta} frames @ {fps_meta:.1f} fps")
    print(f"  pass: id={p.id}  dir={p.direction}  "
          f"elapsed_s={p.elapsed_s:.3f}s  speed_stored={stored_speed:.1f} mph")
    print(f"  seconds region: {region}\n")

    diffs: list[int] = []
    crops: list[np.ndarray] = []
    bright_masks: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x1, y1, x2, y2 = region
        x2 = min(x2, frame.shape[1])
        y2 = min(y2, frame.shape[0])
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        crops.append(crop)
        bright = (crop > args.bright)
        bright_masks.append(bright)
        if len(bright_masks) >= 2:
            flips = int(np.bitwise_xor(bright_masks[-1], bright_masks[-2]).sum())
            diffs.append(flips)
    cap.release()

    if not crops:
        raise SystemExit("clip read 0 frames")

    if args.dump_frames:
        cv2.imwrite("/tmp/verify_first_frame.png", cv2.cvtColor(crops[0], cv2.COLOR_GRAY2BGR))
        cv2.imwrite("/tmp/verify_last_frame.png", cv2.cvtColor(crops[-1], cv2.COLOR_GRAY2BGR))
        # Stack all crops horizontally so the user can scrub the timestamp
        stacked = np.hstack(crops)
        cv2.imwrite("/tmp/verify_seconds_strip.png", stacked)
        print("dumped: /tmp/verify_seconds_strip.png  (horizontal strip of "
              "every frame's seconds region; eyeball it for tick boundaries)\n")

    # Print every frame's flip count and tag suspected ticks.
    ticks: list[int] = []
    print("frame -> bright-flips (* = suspected tick)")
    for i, d in enumerate(diffs):
        is_tick = d >= args.tick_threshold
        if is_tick:
            ticks.append(i + 1)  # diff[i] compares frame i+1 to frame i
        marker = " *" if is_tick else "  "
        print(f"  {i+1:3d}  {d:5d}{marker}")

    if not ticks:
        print(f"\nno ticks detected (threshold={args.tick_threshold}). "
              "Either the clip is shorter than 1 OSD-second OR the seconds-"
              "region is wrong. Try --dump-frames to eyeball the strip.")
        return 1

    # Frames per OSD-second: gaps between ticks (drop the first and last
    # because they're truncated by clip boundaries).
    inner_gaps = [ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)]
    print(f"\nticks at frames: {ticks}")
    if not inner_gaps:
        print("only one tick observed — clip too short to estimate fps")
        return 1

    observed_fps = statistics.fmean(inner_gaps)
    print(f"frames per OSD-second (inner): {inner_gaps}")
    print(f"observed camera fps: mean={observed_fps:.2f}  "
          f"(meta_fps={fps_meta:.1f})")

    if observed_fps > 0 and stored_speed > 0:
        ratio = observed_fps / fps_meta
        print(f"\nfps ratio (observed / meta): {ratio:.2f}")
        if ratio < 0.7:
            corrected_elapsed = p.elapsed_s * (fps_meta / observed_fps)
            corrected_speed = (distance_m / corrected_elapsed) * MPS_TO_MPH
            print(f"⚠ stored elapsed is likely UNDER-stated (ffmpeg burst "
                  f"compressed monotonic intervals).")
            print(f"  rough corrected speed if elapsed scales by meta/observed: "
                  f"{corrected_speed:.1f} mph (was {stored_speed:.1f} mph)")
        elif ratio > 1.3:
            print("⚠ observed > meta; double-check the seconds-region is on the OSD.")
        else:
            print("✓ frame rate looks consistent with stored speed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
