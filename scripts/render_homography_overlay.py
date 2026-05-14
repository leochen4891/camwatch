"""Render the current homography's grid onto the calibration frame so the
operator can eyeball whether the projection lines up with the road.

Reads config/homography.yaml (with optional K + D for lens distortion).
The world → distorted-pixel mapping handles distortion when K + D are
present, so the rendered grid bends with the lens.

Draws:
  - red outer rectangle around the calibrated road area
  - blue 5 ft grid (lines along X and Y)
  - black 1 ft sub-grid (thin)
  - red anchor dots with their indices

The calibrated road area is the axis-aligned bbox of the marked points'
world coordinates (i.e., X=0..-30 ft, Y=-40..+25 ft for the 17-point CX410W
layout).

Usage:
    uv run python scripts/render_homography_overlay.py
    uv run python scripts/render_homography_overlay.py --out /tmp/out.jpg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FRAME = REPO / "events" / "calibration_main_frame.jpg"
DEFAULT_HOMOG = REPO / "config" / "homography.yaml"
DEFAULT_OUT = Path("/tmp/homography_overlay.jpg")

FT_TO_M = 0.3048

# BGR colors
RED = (0, 0, 220)
BLUE = (255, 80, 0)
BLACK = (0, 0, 0)
DOT = (60, 60, 240)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    ap.add_argument("--homog", type=Path, default=DEFAULT_HOMOG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    img = cv2.imread(str(args.frame))
    if img is None:
        raise SystemExit(f"could not read frame: {args.frame}")
    print(f"frame: {args.frame} ({img.shape[1]}x{img.shape[0]})")

    data = yaml.safe_load(args.homog.read_text())["homography"]
    H = np.array(data["H"], dtype=np.float64)
    Hinv = np.linalg.inv(H)
    K = np.array(data["K"], dtype=np.float64) if "K" in data else None
    D = np.array(data["D"], dtype=np.float64).reshape(-1) if "D" in data else None
    print(f"H: frame_size={data.get('frame_size')}, "
          f"mean_err={data['mean_reprojection_error_m']*100:.1f} cm, "
          f"max_err={data['max_reprojection_error_m']*100:.1f} cm, "
          f"K/D={'yes' if K is not None else 'no'}")

    def m2px(X: float, Y: float) -> tuple[int, int] | None:
        """World (X, Y) → distorted pixel (u, v). Applies lens distortion if K + D are loaded."""
        p = Hinv @ np.array([X, Y, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        u_undist = p[0] / p[2]
        v_undist = p[1] / p[2]
        if not (math.isfinite(u_undist) and math.isfinite(v_undist)):
            return None
        if K is None or D is None:
            return int(round(u_undist)), int(round(v_undist))
        x_cam = (u_undist - K[0, 2]) / K[0, 0]
        y_cam = (v_undist - K[1, 2]) / K[1, 1]
        pts3d = np.array([[[x_cam, y_cam, 1.0]]], dtype=np.float64)
        out_pts, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K, D)
        u, v = float(out_pts[0, 0, 0]), float(out_pts[0, 0, 1])
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return int(round(u)), int(round(v))

    def polyline_world(X1: float, Y1: float, X2: float, Y2: float, n: int = 64) -> list[tuple[int, int]]:
        ts = np.linspace(0.0, 1.0, n)
        pts: list[tuple[int, int]] = []
        for t in ts:
            X = X1 + (X2 - X1) * t
            Y = Y1 + (Y2 - Y1) * t
            p = m2px(float(X), float(Y))
            if p is not None:
                pts.append(p)
        return pts

    # World-space extent: axis-aligned bbox of all anchor points, rounded to ft.
    Xs_m = [p["X"] for p in data["meter_pts"]]
    Ys_m = [p["Y"] for p in data["meter_pts"]]
    x_min_ft = int(round(min(Xs_m) / FT_TO_M))   # -30
    x_max_ft = int(round(max(Xs_m) / FT_TO_M))   # 0
    y_min_ft = int(round(min(Ys_m) / FT_TO_M))   # -40
    y_max_ft = int(round(max(Ys_m) / FT_TO_M))   # +25
    print(f"world bbox (ft): X={x_min_ft}..{x_max_ft}, Y={y_min_ft}..{y_max_ft}")

    out = img.copy()

    def draw_polyline(pts: list[tuple[int, int]], color: tuple[int, int, int], thickness: int) -> None:
        if len(pts) < 2:
            return
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [arr], False, color, thickness, cv2.LINE_AA)

    # 1 ft sub-grid (thin black). Drawn first so 5 ft + outer overlay on top.
    for x_ft in range(x_min_ft, x_max_ft + 1):
        X = x_ft * FT_TO_M
        draw_polyline(polyline_world(X, y_min_ft * FT_TO_M, X, y_max_ft * FT_TO_M), BLACK, 1)
    for y_ft in range(y_min_ft, y_max_ft + 1):
        Y = y_ft * FT_TO_M
        draw_polyline(polyline_world(x_min_ft * FT_TO_M, Y, x_max_ft * FT_TO_M, Y), BLACK, 1)

    # 5 ft grid (blue, thicker)
    for x_ft in range(x_min_ft, x_max_ft + 1, 5):
        X = x_ft * FT_TO_M
        draw_polyline(polyline_world(X, y_min_ft * FT_TO_M, X, y_max_ft * FT_TO_M), BLUE, 2)
    for y_ft in range(y_min_ft, y_max_ft + 1, 5):
        Y = y_ft * FT_TO_M
        draw_polyline(polyline_world(x_min_ft * FT_TO_M, Y, x_max_ft * FT_TO_M, Y), BLUE, 2)

    # Red outer rectangle (4 edges drawn as polylines so they bend with the lens)
    edges = [
        (x_min_ft, y_min_ft, x_max_ft, y_min_ft),
        (x_max_ft, y_min_ft, x_max_ft, y_max_ft),
        (x_max_ft, y_max_ft, x_min_ft, y_max_ft),
        (x_min_ft, y_max_ft, x_min_ft, y_min_ft),
    ]
    for x1, y1, x2, y2 in edges:
        draw_polyline(
            polyline_world(x1 * FT_TO_M, y1 * FT_TO_M, x2 * FT_TO_M, y2 * FT_TO_M),
            RED, 3,
        )

    # Anchor dots with index labels
    for pt in data["pixel_pts"]:
        idx = int(pt["idx"])
        u = int(round(float(pt["u"])))
        v = int(round(float(pt["v"])))
        cv2.circle(out, (u, v), 8, DOT, -1)
        cv2.circle(out, (u, v), 8, (255, 255, 255), 1)
        cv2.putText(out, str(idx), (u + 12, v + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend
    cv2.rectangle(out, (10, 10), (520, 160), (40, 40, 40), -1)
    cv2.rectangle(out, (10, 10), (520, 160), (255, 255, 255), 1)
    cv2.putText(out, f"red    = outer rectangle ({x_max_ft - x_min_ft}x{y_max_ft - y_min_ft} ft)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
    cv2.putText(out, "blue   = 5 ft grid", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2, cv2.LINE_AA)
    cv2.putText(out, "black  = 1 ft sub-grid", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2, cv2.LINE_AA)
    cv2.putText(out, f"err: mean={data['mean_reprojection_error_m']*100:.1f}cm  max={data['max_reprojection_error_m']*100:.1f}cm",
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.imwrite(str(args.out), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
