"""Render the geometric visual grid (yellow rectangle + blue 5ft + cyan 1ft)
on the cached calibration frame, using the current `config/marked_points.yaml`.

This is the same grid drawn live in mark_rectangle.py — purely from click
geometry, no homography. Each blue line passes through its own clicked dot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from mark_rectangle import draw_visual_grid  # noqa: E402

MARKS_YAML = REPO / "config" / "marked_points.yaml"
FRAME_JPG = REPO / "events" / "calibration_main_frame.jpg"
OUT_JPG = REPO / "config" / "visual_grid_verify.jpg"


def main() -> None:
    img = cv2.imread(str(FRAME_JPG))
    if img is None:
        sys.exit(f"could not read {FRAME_JPG}")
    data = yaml.safe_load(MARKS_YAML.read_text())["marked_points"]
    pts_by_idx = {int(p["idx"]): tuple(p["pixel"]) for p in data["points"]}
    if set(pts_by_idx.keys()) != set(range(1, 14)):
        sys.exit(f"expected indices 1..13, got {sorted(pts_by_idx.keys())}")

    # Build the click_main list in the order draw_visual_grid expects:
    #   index 0 = NE (point 1), 1 = SE (point 11), 2 = NW (point 12),
    #   3 = SW (point 13), 4..12 = inner east-curb (points 2..10)
    order = [1, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    clicks_main = [pts_by_idx[i] for i in order]

    overlay = img.copy()
    draw_visual_grid(overlay, clicks_main)

    # Add red dot + label for each marked point
    for idx, (u, v) in pts_by_idx.items():
        cv2.circle(overlay, (u, v), 8, (0, 0, 255), -1)
        cv2.circle(overlay, (u, v), 9, (0, 0, 0), 1)
        is_west = idx in (12, 13)
        lp = (u + 10, v - 10) if is_west else (u + 10, v + 22)
        cv2.putText(
            overlay, str(idx), lp,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )

    cv2.imwrite(str(OUT_JPG), overlay)
    print(f"wrote {OUT_JPG}")


if __name__ == "__main__":
    main()
