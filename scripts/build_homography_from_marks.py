"""Build the homography matrix from 17 hand-clicked points on the asphalt.

Layout (per the user's marking session, CX410W era):
  Points 1..14   East curb, north→south, 5 ft apart.
                 Point 6 is at the camera-perpendicular line (origin).
                 → world Y from +25 ft (point 1) to -40 ft (point 14), X = 0.
                 Points 12..14 are the south-extension dots added after the
                 camera upgrade; points 1..11 are the original east-curb anchors.
  Point 15       West curb, new SW corner — perpendicular across from point 14.
                 → world (X = -road_width, Y = -40 ft).
  Point 16       West curb, old SW corner — perpendicular across from point 11.
                 → world (X = -road_width, Y = -25 ft).
  Point 17       West curb, old NW corner — perpendicular across from point 1.
                 → world (X = -road_width, Y = +25 ft).

Coordinate system:
  Origin = point 6 (east curb, camera's perpendicular).
  +X     = east (toward camera).  Road is at X ≤ 0.
  +Y     = along the road, "north-ish" (toward point 1).

Speed = |dY/dt| in this frame (the road runs along Y), independent of the
assumed road width — the 5 ft east-curb spacing alone fixes the Y-axis
scale.

Pixel coords from the marking session are in main-stream space (read from
the marked_points.yaml the clicker wrote), which is also what the live
capture worker sees. H is fit directly against main-stream pixels → meters.
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
ROAD_WIDTH_FT = 30.0  # ~matches GPS-derived 30.1 ft; only affects X-axis, not speed


def world_for(idx: int) -> tuple[float, float]:
    """Return (X, Y) in meters for marked point `idx`."""
    if 1 <= idx <= 14:
        # idx 1 → +25 ft, idx 6 → 0, idx 11 → -25 ft, idx 14 → -40 ft
        y_ft = (6 - idx) * SPACING_FT
        return (0.0, y_ft * FT_TO_M)
    if idx == 15:
        return (-ROAD_WIDTH_FT * FT_TO_M, -8 * SPACING_FT * FT_TO_M)  # Y = -40 ft
    if idx == 16:
        return (-ROAD_WIDTH_FT * FT_TO_M, -5 * SPACING_FT * FT_TO_M)  # Y = -25 ft
    if idx == 17:
        return (-ROAD_WIDTH_FT * FT_TO_M, +5 * SPACING_FT * FT_TO_M)  # Y = +25 ft
    raise ValueError(f"unexpected idx {idx}")


def main() -> None:
    data = yaml.safe_load(MARKED.read_text())["marked_points"]
    pts = data["points"]
    if len(pts) != 17:
        raise SystemExit(f"expected 17 marked points, got {len(pts)}")
    frame_w, frame_h = [int(v) for v in data["frame_size"]]
    print(f"Fitting against main-stream frame_size={frame_w}x{frame_h}")
    main_pixels: dict[int, tuple[float, float]] = {
        int(p["idx"]): (float(p["pixel"][0]), float(p["pixel"][1])) for p in pts
    }
    if set(main_pixels.keys()) != set(range(1, 18)):
        raise SystemExit(f"expected indices 1..17, got {sorted(main_pixels.keys())}")

    # All 17 raw clicks → world coords. findHomography does least-squares,
    # so we just feed every anchor in.
    src_anchors: list[tuple[float, float]] = []
    dst_anchors: list[tuple[float, float]] = []
    anchor_labels: list[str] = []
    for idx in range(1, 18):
        src_anchors.append(main_pixels[idx])
        world_xy = world_for(idx)
        dst_anchors.append(world_xy)
        if idx == 1:
            tag = "east-curb NE (pt1, Y=+25ft)"
        elif idx == 14:
            tag = "east-curb new-SE (pt14, Y=-40ft)"
        elif idx == 15:
            tag = "west-curb new-SW (pt15, Y=-40ft)"
        elif idx == 16:
            tag = "west-curb old-SW (pt16, Y=-25ft)"
        elif idx == 17:
            tag = "west-curb old-NW (pt17, Y=+25ft)"
        else:
            tag = f"east-curb @ Y={world_xy[1]/FT_TO_M:+.0f}ft (raw pt{idx})"
        anchor_labels.append(tag)

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
            "frame_size": [frame_w, frame_h],
            "origin": "point 6 — east curb, camera's perpendicular",
            "axes": "+X = east (toward camera); +Y = along road toward point 1 (north-ish)",
            "road_width_ft": ROAD_WIDTH_FT,
            "spacing_ft": SPACING_FT,
            "method": "17-anchor (14 east-curb + 3 west-curb raw clicks) least-squares fit, main-stream pixels",
            "max_reprojection_error_m": err_max,
            "mean_reprojection_error_m": err_mean,
            "pixel_pts": [
                {"idx": i, "u": float(main_pixels[i][0]), "v": float(main_pixels[i][1])}
                for i in range(1, 18)
            ],
            "meter_pts": [
                {"idx": i, "X": float(world_for(i)[0]), "Y": float(world_for(i)[1])}
                for i in range(1, 18)
            ],
        }
    }
    OUT_HOMOG.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote {OUT_HOMOG}")


if __name__ == "__main__":
    main()
