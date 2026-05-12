"""Build the homography matrix from 13 hand-clicked points on the asphalt.

Layout (per the user's marking session):
  Points 1..11   East curb, north→south, 5 ft apart.
                 Point 6 is at the camera-perpendicular line.
                 → world Y from +25 ft (point 1) to -25 ft (point 11), X = 0.
  Point 12       West curb, NW corner — perpendicular across from point 1.
                 → world (X = -road_width, Y = +25 ft).
  Point 13       West curb, SW corner — perpendicular across from point 11.
                 → world (X = -road_width, Y = -25 ft).

Coordinate system:
  Origin = point 6 (east curb, camera's perpendicular).
  +X     = east (toward camera).  Road is at X ≤ 0.
  +Y     = along the road, "north-ish" (toward point 1).

Speed = |dY/dt| in this frame (the road runs along Y), independent of the
assumed road width — the 5 ft east-curb spacing alone fixes the Y-axis
scale.

Pixel coords from the marking session are in main-stream space (2048x1536),
which is also what the live capture worker now sees. H is fit directly
against main-stream pixels → meters.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
MARKED = REPO / "config" / "marked_points.yaml"
OUT_HOMOG = REPO / "config" / "homography.yaml"
BACKUP_HOMOG = REPO / "config" / "homography.gps_v1.yaml"

FT_TO_M = 0.3048
SPACING_FT = 5.0
FRAME_W, FRAME_H = 2048, 1536  # main-stream resolution H is fit against
ROAD_WIDTH_FT = 30.0  # ~matches GPS-derived 30.1 ft; only affects X-axis, not speed


def world_for(idx: int) -> tuple[float, float]:
    """Return (X, Y) in meters for marked point `idx`."""
    if 1 <= idx <= 11:
        # idx 1 → +25 ft, idx 6 → 0, idx 11 → -25 ft
        y_ft = (6 - idx) * SPACING_FT
        return (0.0, y_ft * FT_TO_M)
    if idx == 12:
        return (-ROAD_WIDTH_FT * FT_TO_M, +5 * SPACING_FT * FT_TO_M)
    if idx == 13:
        return (-ROAD_WIDTH_FT * FT_TO_M, -5 * SPACING_FT * FT_TO_M)
    raise ValueError(f"unexpected idx {idx}")


def main() -> None:
    data = yaml.safe_load(MARKED.read_text())["marked_points"]
    pts = data["points"]
    if len(pts) != 13:
        raise SystemExit(f"expected 13 marked points, got {len(pts)}")
    main_pixels: dict[int, tuple[float, float]] = {
        int(p["idx"]): (float(p["pixel"][0]), float(p["pixel"][1])) for p in pts
    }
    if set(main_pixels.keys()) != set(range(1, 14)):
        raise SystemExit(f"expected indices 1..13, got {sorted(main_pixels.keys())}")

    # Build the 13-anchor calibration set against main-stream pixels:
    #   - 4 corners (raw clicks): NE=1, SE=11, NW=12, SW=13
    #   - 9 inner east-curb anchors at raw click positions.
    NE = main_pixels[1]
    SE = main_pixels[11]
    NW = main_pixels[12]
    SW = main_pixels[13]
    inner_dots = [main_pixels[i] for i in range(2, 11)]  # points 2..10

    def frac_along(p: tuple[float, float]) -> float:
        vx = SE[0] - NE[0]
        vy = SE[1] - NE[1]
        L2 = vx * vx + vy * vy
        if L2 == 0:
            return 0.0
        return ((p[0] - NE[0]) * vx + (p[1] - NE[1]) * vy) / L2

    inner_y_ft = [+20, +15, +10, +5, 0, -5, -10, -15, -20]  # for points 2..10

    src_anchors: list[tuple[float, float]] = []
    dst_anchors: list[tuple[float, float]] = []
    anchor_labels: list[str] = []

    # 4 corners (raw clicks)
    for label, idx, world_xy in [
        ("NE (point 1)", 1, world_for(1)),
        ("SE (point 11)", 11, world_for(11)),
        ("NW (point 12)", 12, world_for(12)),
        ("SW (point 13)", 13, world_for(13)),
    ]:
        src_anchors.append(main_pixels[idx])
        dst_anchors.append(world_xy)
        anchor_labels.append(label)

    # 9 inner east-curb anchors at the user's RAW click positions. No
    # smoothing kernel — earlier we used a 0.25/0.5/0.25 kernel with NE/SE
    # corners as boundary "neighbors", but that pulled the anchors away
    # from the actual asphalt white marks the user painted, which made
    # rendered grid lines miss the dots. Trusting raw clicks puts each
    # grid line through the dot it represents.
    for idx, y_ft in zip(range(2, 11), inner_y_ft):
        src_anchors.append(main_pixels[idx])
        dst_anchors.append((0.0, y_ft * FT_TO_M))
        anchor_labels.append(f"east-curb @ Y={y_ft:+}ft (raw pt{idx})")

    src = np.array(src_anchors, dtype=np.float32)
    dst = np.array(dst_anchors, dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise SystemExit("findHomography returned None — check input geometry")

    # Reprojection diagnostics — show per-anchor residual
    print(f"{'#':>3}  {'anchor':<38}  {'main px':>16}  {'projected (m)':>20}  {'target (m)':>20}  {'err (m)':>8}")
    print("-" * 120)
    errors_m: list[float] = []
    for i in range(len(src_anchors)):
        u, v = src_anchors[i]
        tx, ty = dst_anchors[i]
        p = H @ np.array([u, v, 1.0])
        Xm, Ym = p[0] / p[2], p[1] / p[2]
        err = math.hypot(Xm - tx, Ym - ty)
        errors_m.append(err)
        print(
            f"  {i+1:>3}  {anchor_labels[i]:<38}  ({u:6.1f},{v:6.1f})  "
            f"({Xm:>+7.3f},{Ym:>+7.3f})  ({tx:>+7.3f},{ty:>+7.3f})  {err:>8.4f}"
        )

    err_mean = float(np.mean(errors_m))
    err_max = float(max(errors_m))
    print(f"\nReprojection over {len(errors_m)} anchors: mean={err_mean*100:.2f} cm, max={err_max*100:.2f} cm")

    if OUT_HOMOG.exists():
        shutil.copy(OUT_HOMOG, BACKUP_HOMOG)
        print(f"Backed up old homography → {BACKUP_HOMOG}")

    payload = {
        "homography": {
            "H": H.tolist(),
            "frame_size": [FRAME_W, FRAME_H],
            "origin": "point 6 — east curb, camera's perpendicular",
            "axes": "+X = east (toward camera); +Y = along road toward point 1 (north-ish)",
            "road_width_ft": ROAD_WIDTH_FT,
            "spacing_ft": SPACING_FT,
            "method": "13-anchor (4 corners + 9 raw east-curb clicks) least-squares fit, main-stream pixels",
            "max_reprojection_error_m": err_max,
            "mean_reprojection_error_m": err_mean,
            "pixel_pts": [
                {"idx": i, "u": float(main_pixels[i][0]), "v": float(main_pixels[i][1])}
                for i in range(1, 14)
            ],
            "meter_pts": [
                {"idx": i, "X": float(world_for(i)[0]), "Y": float(world_for(i)[1])}
                for i in range(1, 14)
            ],
        }
    }
    OUT_HOMOG.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote {OUT_HOMOG}")


if __name__ == "__main__":
    main()
