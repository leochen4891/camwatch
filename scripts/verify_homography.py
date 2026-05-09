"""Verify the homography by comparing speed estimates against the known-mph
calibration passes in config/calibration.yaml.

For each reference pass, two speed estimates are computed and compared
against the user-verified `known_mph`:

  1. 2-line method (existing)
       speed = line_distance_m / elapsed_s
     where line_distance_m is from calibration.yaml (per-direction)
     and elapsed_s is the (t_b - t_a) recorded with the pass.

  2. Homography method (new)
       For each clip frame:
         - run YOLO predict (no tracker, one-shot detection)
         - pick the largest vehicle bbox whose ground-point lies in the ROI
         - project the ground-point through H to get (X, Y) in meters
       Linear-regress Y(t) (the road is N-S aligned, so Y-velocity is speed)
       and convert |slope| to mph.

Both methods are then compared against `known_mph` to see which is closer.

Run:
  uv run python scripts/verify_homography.py
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


def project(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def speed_via_homography(
    clip: Path,
    H: np.ndarray,
    roi: tuple[int, int, int, int],
    detector: Detector,
) -> tuple[float, int, list[tuple[float, float, float]]]:
    """Returns (mph, n_samples, samples=[(t_s, X_m, Y_m), ...])."""
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    samples: list[tuple[float, float, float]] = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        dets = detector.detect(fr)
        best: tuple[float, float] | None = None
        best_area = 0.0
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            cx = (x1 + x2) / 2.0
            gy = y2
            if not (roi[0] <= cx <= roi[2] and roi[1] <= gy <= roi[3]):
                continue
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best = (cx, gy)
                best_area = area
        if best is not None:
            X, Y = project(H, best[0], best[1])
            samples.append((i / fps, X, Y))
        i += 1
    cap.release()

    if len(samples) < 3:
        return float("nan"), len(samples), samples

    ts = np.array([s[0] for s in samples])
    ys = np.array([s[2] for s in samples])
    A = np.vstack([ts, np.ones_like(ts)]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    return abs(slope) * MPH_PER_MPS, len(samples), samples


def main() -> None:
    homog = yaml.safe_load((REPO / "config/homography.yaml").read_text())["homography"]
    H = np.array(homog["H"], dtype=np.float64)

    cal = yaml.safe_load((REPO / "config/calibration.yaml").read_text())
    dist_n = float(cal["line_distance_m_north"])
    dist_s = float(cal["line_distance_m_south"])
    roi = (
        int(cal["roi_x1"]), int(cal["roi_y1"]),
        int(cal["roi_x2"]), int(cal["roi_y2"]),
    )
    refs = cal.get("calibration_points") or []
    print(f"Loaded {len(refs)} reference passes; H from config/homography.yaml")
    print(f"2-line distances: N={dist_n:.3f}m  S={dist_s:.3f}m")
    print(f"ROI: {roi}\n")

    detector = Detector(
        weights="yolo11n.pt",
        device="mps",
        classes=[2, 3, 5, 7],
        conf=0.35,
        iou=0.5,
        roi=None,
    )

    fmt_h = (
        f"{'track':>5}  {'dir':>3}  {'known':>6}  "
        f"{'2line':>7}  {'homog':>7}  {'2Lerr':>7}  {'HGerr':>7}  "
        f"{'n':>3}  clip"
    )
    print(fmt_h)
    print("-" * len(fmt_h))

    err2_abs: list[float] = []
    errh_abs: list[float] = []
    for p in refs:
        clip = REPO / p["clip_path"]
        if not clip.exists():
            print(f"  (skip; clip missing: {clip.name})")
            continue
        direction = p["direction"]
        elapsed = float(p["elapsed_s"])
        known = float(p["known_mph"])
        line_dist = dist_n if direction == "N" else dist_s
        v_2line = (line_dist / elapsed) * MPH_PER_MPS

        v_homog, n_samples, _ = speed_via_homography(clip, H, roi, detector)

        if math.isnan(v_homog):
            print(
                f"  {p['track_id']:>5}  {direction:>3}  {known:>6.1f}  "
                f"{v_2line:>7.2f}  {'-':>7}  {v_2line-known:>+7.2f}  {'-':>7}  "
                f"{n_samples:>3}  {clip.name}  (too few homog samples)"
            )
            err2_abs.append(abs(v_2line - known))
            continue

        e2 = v_2line - known
        eh = v_homog - known
        err2_abs.append(abs(e2))
        errh_abs.append(abs(eh))
        print(
            f"  {p['track_id']:>5}  {direction:>3}  {known:>6.1f}  "
            f"{v_2line:>7.2f}  {v_homog:>7.2f}  {e2:>+7.2f}  {eh:>+7.2f}  "
            f"{n_samples:>3}  {clip.name}"
        )

    if err2_abs and errh_abs:
        print()
        print(f"  2-line  mean |err| = {np.mean(err2_abs):.2f} mph,  "
              f"max |err| = {max(err2_abs):.2f} mph")
        print(f"  homog   mean |err| = {np.mean(errh_abs):.2f} mph,  "
              f"max |err| = {max(errh_abs):.2f} mph")


if __name__ == "__main__":
    main()
