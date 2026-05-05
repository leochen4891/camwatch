"""Regenerate thumbnail JPEGs from existing clip mp4s.

Usage:
  uv run python scripts/regen_thumbs.py            # regenerate all
  uv run python scripts/regen_thumbs.py --ids 4 5  # specific pass ids
  uv run python scripts/regen_thumbs.py --dry-run

For each pass with a clip on disk, this script seeks to the middle of the
clip, extracts one frame, crops to the line span + ROI band, and writes
the result over the existing thumbnail JPEG.

CAVEAT: clips recorded before the "clean thumbnails" change have the
debug overlay (bbox, labels, lines) baked into the video, so regenerated
thumbs will still show those overlays. The crop is centered on the road
area between the lines, so identifying the car still works, just not as
cleanly as a freshly-captured pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camwatch.config import load_config
from camwatch.db import Database


def regen_one(
    clip_path: Path,
    line_a_x: int,
    line_b_x: int,
    roi: tuple[int, int, int, int] | None,
    target_w: int = 320,
) -> bool:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"  could not open {clip_path}")
        return False
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = max(0, n_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"  failed to read mid-frame from {clip_path}")
        return False

    h, w = frame.shape[:2]
    # Crop to line span + ROI Y range with some padding.
    span = line_b_x - line_a_x
    pad_x = int(span * 0.25)
    cx1 = max(0, line_a_x - pad_x)
    cx2 = min(w, line_b_x + pad_x)
    if roi is not None:
        cy1, cy2 = max(0, roi[1]), min(h, roi[3])
    else:
        cy1, cy2 = 0, h
    cropped = frame[cy1:cy2, cx1:cx2]
    th, tw = cropped.shape[:2]
    if tw > target_w:
        scale = target_w / tw
        cropped = cv2.resize(cropped, (target_w, max(1, int(round(th * scale)))))

    thumb_path = clip_path.with_suffix(".jpg")
    cv2.imwrite(str(thumb_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=int, nargs="*", help="specific pass ids; default = all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    cal = cfg.load_calibration()
    if cal is None:
        print("ERROR: calibration.yaml missing; can't determine line span")
        return 1

    db = Database()
    passes = db.list_passes(limit=10000)
    if args.ids:
        wanted = set(args.ids)
        passes = [p for p in passes if p.id in wanted]

    n_ok = n_skip = n_fail = 0
    for p in passes:
        if not p.clip_path:
            n_skip += 1
            continue
        clip = Path(p.clip_path)
        if not clip.exists():
            print(f"id={p.id}: clip missing at {clip}")
            n_skip += 1
            continue
        if args.dry_run:
            print(f"id={p.id}: would regen {clip.with_suffix('.jpg')}")
            n_ok += 1
            continue
        if regen_one(clip, cal.line_a_x, cal.line_b_x, cal.roi):
            print(f"id={p.id}: ok ({clip.name})")
            n_ok += 1
        else:
            n_fail += 1

    print(f"\ndone: ok={n_ok} skipped={n_skip} failed={n_fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
