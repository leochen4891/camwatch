"""Re-render the homography verification image with a richer grid.

Reads:
  config/homography.yaml
  events/calibration_frame.jpg

Writes:
  config/homography_grid.jpg

Differences from calibrate_homography.py's built-in render:
  - 1m minor grid (dim green) + 5m major grid (bright cyan) so depth is
    legible without counting cells
  - Yellow X-axis line (Y=0, perpendicular to road, through camera's NS)
  - Yellow Y-axis line (X=0, parallel to road, through camera's EW)
  - Magenta segment 3↔4 labelled with both the GPS-derived meter
    distance and the user's 21ft physical measurement, for direct
    visual cross-check
  - Each calibration point labelled with its meter coordinates
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
HOMOG_YAML = REPO / "config" / "homography.yaml"
FRAME_JPG = REPO / "events" / "calibration_frame.jpg"
OUT_JPG = REPO / "config" / "homography_grid.jpg"

LABELS = ["1", "2", "3", "4"]
PHYSICAL_DRIVEWAY_FT = 21.0


def m2px(Hinv: np.ndarray, X: float, Y: float) -> tuple[int, int] | None:
    p = Hinv @ np.array([X, Y, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    u, v = p[0] / p[2], p[1] / p[2]
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    if abs(u) > 1e5 or abs(v) > 1e5:
        return None
    return int(round(u)), int(round(v))


def main() -> None:
    data = yaml.safe_load(HOMOG_YAML.read_text())["homography"]
    H = np.array(data["H"], dtype=np.float64)
    Hinv = np.linalg.inv(H)
    pixel_pts = data["pixel_pts"]
    meter_pts = data["meter_pts"]

    img = cv2.imread(str(FRAME_JPG))
    if img is None:
        raise SystemExit(f"could not read {FRAME_JPG}")

    # Grid extent: a couple meters beyond the calibration quadrilateral.
    xs = [m[0] for m in meter_pts]
    ys = [m[1] for m in meter_pts]
    x_min = math.floor(min(xs)) - 2
    x_max = math.ceil(max(xs)) + 2
    y_min = math.floor(min(ys)) - 2
    y_max = math.ceil(max(ys)) + 2

    minor = (60, 180, 60)        # dim green
    major = (255, 200, 0)        # bright cyan/orange
    axis = (0, 220, 255)         # yellow
    segment = (255, 0, 255)      # magenta
    point_color = (0, 0, 255)    # red

    overlay = img.copy()

    # 1m minor grid
    for X in range(x_min, x_max + 1):
        a = m2px(Hinv, float(X), float(y_min))
        b = m2px(Hinv, float(X), float(y_max))
        if a and b:
            cv2.line(overlay, a, b, minor, 1, cv2.LINE_AA)
    for Y in range(y_min, y_max + 1):
        a = m2px(Hinv, float(x_min), float(Y))
        b = m2px(Hinv, float(x_max), float(Y))
        if a and b:
            cv2.line(overlay, a, b, minor, 1, cv2.LINE_AA)

    # 5m major grid + labels
    for X in range(x_min - (x_min % 5), x_max + 1, 5):
        a = m2px(Hinv, float(X), float(y_min))
        b = m2px(Hinv, float(X), float(y_max))
        if a and b:
            cv2.line(overlay, a, b, major, 1, cv2.LINE_AA)
        # Label at the row Y = y_max (north end) so labels stack at top of grid
        lbl = m2px(Hinv, float(X), float(y_max))
        if lbl is not None and 0 < lbl[0] < img.shape[1] and 0 < lbl[1] < img.shape[0]:
            cv2.putText(
                overlay, f"{X}m", (lbl[0] + 2, lbl[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, major, 1, cv2.LINE_AA,
            )
    for Y in range(y_min - (y_min % 5), y_max + 1, 5):
        a = m2px(Hinv, float(x_min), float(Y))
        b = m2px(Hinv, float(x_max), float(Y))
        if a and b:
            cv2.line(overlay, a, b, major, 1, cv2.LINE_AA)
        # Label on the east end
        lbl = m2px(Hinv, float(x_max), float(Y))
        if lbl is not None and 0 < lbl[0] < img.shape[1] and 0 < lbl[1] < img.shape[0]:
            cv2.putText(
                overlay, f"{Y}m", (lbl[0] + 2, lbl[1] + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, major, 1, cv2.LINE_AA,
            )

    # Yellow axes (X=0 and Y=0 through camera position)
    a = m2px(Hinv, float(x_min), 0.0)
    b = m2px(Hinv, float(x_max), 0.0)
    if a and b:
        cv2.line(overlay, a, b, axis, 2, cv2.LINE_AA)
    a = m2px(Hinv, 0.0, float(y_min))
    b = m2px(Hinv, 0.0, float(y_max))
    if a and b:
        cv2.line(overlay, a, b, axis, 2, cv2.LINE_AA)

    # 3↔4 driveway-width segment (the user's 21ft cross-check)
    p3 = pixel_pts[2]
    p4 = pixel_pts[3]
    cv2.line(overlay, tuple(p3), tuple(p4), segment, 2, cv2.LINE_AA)
    mx, my = (p3[0] + p4[0]) // 2, (p3[1] + p4[1]) // 2
    gps_dist_m = math.hypot(meter_pts[2][0] - meter_pts[3][0], meter_pts[2][1] - meter_pts[3][1])
    gps_dist_ft = gps_dist_m / 0.3048
    label = f"3-4: GPS={gps_dist_m:.2f}m ({gps_dist_ft:.1f}ft)  actual={PHYSICAL_DRIVEWAY_FT:.0f}ft"
    cv2.putText(
        overlay, label, (mx - 180, my - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, segment, 1, cv2.LINE_AA,
    )

    # Calibration points with meter labels
    for (u, v), (X, Y), lbl in zip(pixel_pts, meter_pts, LABELS):
        cv2.circle(overlay, (u, v), 7, point_color, -1)
        text = f"{lbl}: ({X:.1f}, {Y:.1f})"
        cv2.putText(
            overlay, text, (u + 9, v - 9),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, point_color, 1, cv2.LINE_AA,
        )

    # Header
    h, w = img.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, 18), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        "1m grid (green)  5m major (cyan)  axes Y=0/X=0 (yellow)  "
        "3-4 segment (magenta)",
        (4, 13),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
    )

    cv2.imwrite(str(OUT_JPG), overlay)
    print(f"wrote {OUT_JPG}")


if __name__ == "__main__":
    main()
