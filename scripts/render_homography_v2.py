"""Render the v2 homography with grid + the 13 hand-marked points overlaid.

Reads:
  config/homography.yaml          (v2 schema: pixel_pts_main and pixel_pts_sub
                                    are lists of {idx, u, v} dicts)
  events/calibration_main_frame.jpg
  config/calibration.yaml         (for line A / line B / ROI overlays)

Writes:
  config/homography_v2_verify.jpg

Overlays:
  - 1 m minor grid (green)
  - 5 ft major grid lines parallel to road (cyan) labelled by Y in feet
  - Yellow axes through point 6 (Y=0 perpendicular to road, X=0 along road)
  - Red dots + indices at each of the 13 marked points (in main-stream pixels)
  - Existing 2-line system: line A and line B drawn as vertical orange/blue
    lines (in sub-stream coords, projected to main-stream coords by ×3.2)
  - Reprojection error per point as small numeric annotations
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
HOMOG_YAML = REPO / "config" / "homography.yaml"
CALIB_YAML = REPO / "config" / "calibration.yaml"
FRAME_JPG = REPO / "events" / "calibration_main_frame.jpg"
OUT_JPG = REPO / "config" / "homography_v2_verify.jpg"

FT_TO_M = 0.3048


def m2px_main(Hinv: np.ndarray, scale: float, X: float, Y: float) -> tuple[int, int] | None:
    """Project meter (X, Y) → main-stream pixel via H^-1 (sub) then ×scale."""
    p = Hinv @ np.array([X, Y, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    u = p[0] / p[2] * scale
    v = p[1] / p[2] * scale
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    if abs(u) > 1e5 or abs(v) > 1e5:
        return None
    return int(round(u)), int(round(v))


def main() -> None:
    homog = yaml.safe_load(HOMOG_YAML.read_text())["homography"]
    H = np.array(homog["H"], dtype=np.float64)
    Hinv = np.linalg.inv(H)
    scale = float(homog["main_to_sub_scale"])  # 3.2

    img = cv2.imread(str(FRAME_JPG))
    if img is None:
        raise SystemExit(f"could not read {FRAME_JPG}")
    H_img, W_img = img.shape[:2]

    overlay = img.copy()

    # Range: a couple meters beyond the meter-pts span, but only along the
    # road area (don't extrapolate way off into the lawns)
    Xs = [p["X"] for p in homog["meter_pts"]]
    Ys = [p["Y"] for p in homog["meter_pts"]]
    x_min = math.floor(min(Xs)) - 1
    x_max = math.ceil(max(Xs)) + 1
    y_min = math.floor(min(Ys)) - 1
    y_max = math.ceil(max(Ys)) + 1

    major = (255, 200, 0)
    axis = (0, 220, 255)
    point_color = (0, 0, 255)
    line_a_color = (0, 165, 255)
    line_b_color = (255, 100, 0)

    # 5ft major lines (Y axis): 5 ft = 1.524 m increments
    for n in range(-5, 6):
        Y = n * 5 * FT_TO_M
        a = m2px_main(Hinv, scale, float(x_min), Y)
        b = m2px_main(Hinv, scale, float(x_max), Y)
        if a and b:
            cv2.line(overlay, a, b, major, 1, cv2.LINE_AA)
        lbl = m2px_main(Hinv, scale, 0.0, Y)
        if lbl is not None:
            cv2.putText(
                overlay, f"Y={n*5}ft", (lbl[0] + 10, lbl[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, major, 1, cv2.LINE_AA,
            )

    # Yellow axes through point 6 (Y=0 perpendicular to road, X=0 along road)
    a = m2px_main(Hinv, scale, float(x_min), 0.0)
    b = m2px_main(Hinv, scale, float(x_max), 0.0)
    if a and b:
        cv2.line(overlay, a, b, axis, 1, cv2.LINE_AA)
    a = m2px_main(Hinv, scale, 0.0, float(y_min))
    b = m2px_main(Hinv, scale, 0.0, float(y_max))
    if a and b:
        cv2.line(overlay, a, b, axis, 1, cv2.LINE_AA)

    # 13 marked points (in main-stream coords). Draw small ring + small label
    # so the actual dot/mark on the asphalt under each is still visible.
    for p in homog["pixel_pts_main"]:
        idx = p["idx"]
        u, v = p["u"], p["v"]
        cv2.circle(overlay, (u, v), 6, point_color, 1, cv2.LINE_AA)
        cv2.putText(
            overlay, str(idx), (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, point_color, 1, cv2.LINE_AA,
        )

    # Existing 2-line system, projected to main-stream coords
    cal = yaml.safe_load(CALIB_YAML.read_text())
    line_a_x_sub = float(cal["line_a_x"])
    line_b_x_sub = float(cal["line_b_x"])
    line_a_x_main = int(round(line_a_x_sub * scale))
    line_b_x_main = int(round(line_b_x_sub * scale))
    cv2.line(
        overlay, (line_a_x_main, 0), (line_a_x_main, H_img),
        line_a_color, 1, cv2.LINE_AA,
    )
    cv2.line(
        overlay, (line_b_x_main, 0), (line_b_x_main, H_img),
        line_b_color, 1, cv2.LINE_AA,
    )
    cv2.putText(
        overlay, "Line A", (line_a_x_main + 4, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_a_color, 1, cv2.LINE_AA,
    )
    cv2.putText(
        overlay, "Line B", (line_b_x_main + 4, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_b_color, 1, cv2.LINE_AA,
    )

    # Header
    cv2.rectangle(overlay, (0, 0), (W_img, 36), (0, 0, 0), -1)
    err_max_cm = float(homog["max_reprojection_error_m"]) * 100
    err_mean_cm = float(homog["mean_reprojection_error_m"]) * 100
    cv2.putText(
        overlay,
        f"v2 homography  |  13 marked points  |  mean reproj err={err_mean_cm:.1f} cm, max={err_max_cm:.1f} cm  |  "
        f"cyan=5ft Y lines  yellow=axes thru pt6  red=marked  orange=lineA  blue=lineB",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )

    cv2.imwrite(str(OUT_JPG), overlay)
    print(f"wrote {OUT_JPG}")


if __name__ == "__main__":
    main()
