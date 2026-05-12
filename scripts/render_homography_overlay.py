"""Render the current homography's grid + anchor markers onto the
calibration frame so the operator can eyeball whether H lines up with
the road stripes.

Reads config/homography.yaml (main-stream H), draws:
  - 11 cyan 5 ft horizontal lines (Y = -25, -20, ..., +25 ft)
  - 2 yellow axes (X = 0 east curb, Y = 0 perpendicular)
  - 13 numbered red dots at the original click positions

Writes to /tmp/homography_overlay.jpg by default.

Usage:
    uv run python scripts/render_homography_overlay.py
    uv run python scripts/render_homography_overlay.py --frame calibration_main_frame.jpg --out /tmp/out.jpg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FRAME = REPO / "calibration_main_frame.jpg"
DEFAULT_HOMOG = REPO / "config" / "homography.yaml"
DEFAULT_OUT = Path("/tmp/homography_overlay.jpg")

FT_TO_M = 0.3048
MAJOR = (255, 200, 0)   # cyan-ish
AXIS = (0, 220, 255)    # yellow
DOT = (60, 60, 240)     # red


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
    print(f"H: frame_size={data.get('frame_size')}, "
          f"mean_err={data['mean_reprojection_error_m']*100:.1f} cm, "
          f"max_err={data['max_reprojection_error_m']*100:.1f} cm")

    def m2px(X: float, Y: float) -> tuple[int, int] | None:
        p = Hinv @ np.array([X, Y, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        u = p[0] / p[2]
        v = p[1] / p[2]
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return int(round(u)), int(round(v))

    Xs = [p["X"] for p in data["meter_pts"]]
    Ys = [p["Y"] for p in data["meter_pts"]]
    x_min = math.floor(min(Xs)) - 1
    x_max = math.ceil(max(Xs)) + 1
    y_min = math.floor(min(Ys)) - 1
    y_max = math.ceil(max(Ys)) + 1

    out = img.copy()

    # 11 cyan 5 ft horizontal Y-lines spanning the road width
    for n in range(-5, 6):
        Y = n * 5 * FT_TO_M
        a = m2px(float(x_min), Y)
        b = m2px(float(x_max), Y)
        if a and b:
            cv2.line(out, a, b, MAJOR, 2, cv2.LINE_AA)

    # Yellow axes through point 6 (X=0, Y=0)
    a = m2px(float(x_min), 0.0)
    b = m2px(float(x_max), 0.0)
    if a and b:
        cv2.line(out, a, b, AXIS, 2, cv2.LINE_AA)
    a = m2px(0.0, float(y_min))
    b = m2px(0.0, float(y_max))
    if a and b:
        cv2.line(out, a, b, AXIS, 2, cv2.LINE_AA)

    # 13 red dots at the original click positions, with their numbers
    for pt in data["pixel_pts"]:
        idx = int(pt["idx"])
        u = int(round(float(pt["u"])))
        v = int(round(float(pt["v"])))
        cv2.circle(out, (u, v), 8, DOT, -1)
        cv2.circle(out, (u, v), 8, (255, 255, 255), 1)
        cv2.putText(out, str(idx), (u + 12, v + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend at the top-left
    cv2.rectangle(out, (10, 10), (430, 130), (40, 40, 40), -1)
    cv2.rectangle(out, (10, 10), (430, 130), (255, 255, 255), 1)
    cv2.putText(out, "cyan = 5 ft Y-lines", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, MAJOR, 2, cv2.LINE_AA)
    cv2.putText(out, "yellow = X=0 / Y=0 axes", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, AXIS, 2, cv2.LINE_AA)
    cv2.putText(out, "red = original click anchors 1..13", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DOT, 2, cv2.LINE_AA)

    cv2.imwrite(str(args.out), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
