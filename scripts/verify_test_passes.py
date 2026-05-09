"""Recompute homography speed for the two 30 mph test passes.

Bypasses clip-timing issue by:
  1. Replaying each test clip through YOLO to get bbox ground-points per frame.
  2. Interpolating to find ground_y at the moment the bbox crosses line A (x=135)
     and line B (x=471).
  3. Projecting both crossing points through the *new* homography to get their
     (X, Y) in meters.
  4. Computing the actual road distance between the crossings.
  5. Speed = distance / elapsed_s, where elapsed_s comes from the live 2-line
     measurement (PTS-anchored, accurate).

Compare against the user's verified 30 mph drive.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from camwatch.detect import Detector  # noqa: E402

MPH_PER_MPS = 2.2369362920544


def load_homog() -> np.ndarray:
    data = yaml.safe_load((REPO / "config/homography.yaml").read_text())["homography"]
    return np.array(data["H"], dtype=np.float64)


def project(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


def trace_clip(clip: Path, det: Detector, roi: tuple[int, int, int, int]) -> list[tuple[int, float, float]]:
    """Replay clip; return list of (frame_idx, ground_x_pix, ground_y_pix)
    for the largest in-ROI vehicle bbox in each frame."""
    cap = cv2.VideoCapture(str(clip))
    samples = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        dets = det.detect(fr)
        best = None
        best_a = 0.0
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            cx = (x1 + x2) / 2.0
            gy = y2
            if not (roi[0] <= cx <= roi[2] and roi[1] <= gy <= roi[3]):
                continue
            area = (x2 - x1) * (y2 - y1)
            if area > best_a:
                best = (cx, gy)
                best_a = area
        if best is not None:
            samples.append((i, float(best[0]), float(best[1])))
        i += 1
    cap.release()
    return samples


def interp_at_x(samples: list[tuple[int, float, float]], target_x: float) -> tuple[float, float] | None:
    """Find the frame where ground_x crosses target_x, return (frame_idx, ground_y_pix)
    via linear interpolation between the two consecutive samples that straddle it."""
    for i in range(len(samples) - 1):
        x0 = samples[i][1]
        x1 = samples[i + 1][1]
        if (x0 - target_x) * (x1 - target_x) <= 0 and x0 != x1:
            t = (target_x - x0) / (x1 - x0)
            f = samples[i][0] + t * (samples[i + 1][0] - samples[i][0])
            y = samples[i][2] + t * (samples[i + 1][2] - samples[i][2])
            return float(f), float(y)
    return None


def main() -> None:
    H = load_homog()
    cal = yaml.safe_load((REPO / "config/calibration.yaml").read_text())
    line_a_x = float(cal["line_a_x"])
    line_b_x = float(cal["line_b_x"])
    roi = (
        int(cal["roi_x1"]), int(cal["roi_y1"]),
        int(cal["roi_x2"]), int(cal["roi_y2"]),
    )

    det = Detector(
        weights="yolo11n.pt", device="mps",
        classes=[2, 3, 5, 7], conf=0.35, iou=0.5, roi=None,
    )

    # The two 30 mph test passes: clip path, direction, elapsed_s from the live log
    tests = [
        ("recordings/cal_20260508T132934_id57_S.mp4", "S", 1.086, 30.0),
        ("recordings/cal_20260508T133032_id60_N.mp4", "N", 0.928, 30.0),
    ]

    print(f"{'pass':>20}  {'dir':>3}  {'elapsed':>8}  {'(Xa,Ya)':>16}  {'(Xb,Yb)':>16}  "
          f"{'dist m':>7}  {'homog mph':>10}  {'truth':>6}  {'err %':>7}")
    print("-" * 110)
    for clip_rel, direction, elapsed, truth in tests:
        clip = REPO / clip_rel
        samples = trace_clip(clip, det, roi)
        if not samples:
            print(f"  {clip.name:>20}  {direction}  no detections")
            continue

        ca = interp_at_x(samples, line_a_x)
        cb = interp_at_x(samples, line_b_x)
        if ca is None or cb is None:
            print(f"  {clip.name:>20}  {direction}  could not interpolate both line crossings ({len(samples)} samples)")
            continue

        f_a, y_a = ca
        f_b, y_b = cb
        Xa, Ya = project(H, line_a_x, y_a)
        Xb, Yb = project(H, line_b_x, y_b)
        dist_m = math.hypot(Xa - Xb, Ya - Yb)
        speed_mph = (dist_m / elapsed) * MPH_PER_MPS
        err_pct = (speed_mph - truth) / truth * 100

        print(
            f"  {clip.name:>20}  {direction}  {elapsed:>8.3f}  "
            f"({Xa:>+5.2f},{Ya:>+5.2f})  ({Xb:>+5.2f},{Yb:>+5.2f})  "
            f"{dist_m:>7.3f}  {speed_mph:>10.2f}  {truth:>6.1f}  {err_pct:>+6.1f}%"
        )


if __name__ == "__main__":
    main()
