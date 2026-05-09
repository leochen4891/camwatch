"""Compute Method A (per-frame instantaneous speed) and Method C (regression)
for the two 30 mph ground-truth passes, printing all numbers.

Re-times clip frames to camera-time using the live-measured elapsed_s
between line A and line B crossings as the anchor:

    camera_time(frame_i) = (frame_i - f_a) / (f_b - f_a) * elapsed_s

This sidesteps the clip's broken 10 fps fake PTS; the per-frame Δt's are
in real seconds.
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

TESTS = [
    ("recordings/cal_20260508T132934_id57_S.mp4", "S", 1.086, 30.0),
    ("recordings/cal_20260508T133032_id60_N.mp4", "N", 0.928, 30.0),
]


def load_H() -> np.ndarray:
    return np.array(
        yaml.safe_load((REPO / "config/homography.yaml").read_text())["homography"]["H"],
        dtype=np.float64,
    )


def project(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def trace(clip: Path, det: Detector, roi) -> list[tuple[int, float, float]]:
    cap = cv2.VideoCapture(str(clip))
    out: list[tuple[int, float, float]] = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        dets = det.detect(fr)
        best, best_a = None, 0.0
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            cx = (x1 + x2) / 2.0
            gy = y2
            if not (roi[0] <= cx <= roi[2] and roi[1] <= gy <= roi[3]):
                continue
            a = (x2 - x1) * (y2 - y1)
            if a > best_a:
                best, best_a = (cx, gy), a
        if best is not None:
            out.append((i, float(best[0]), float(best[1])))
        i += 1
    cap.release()
    return out


def interp_frame_at_x(samples, target_x: float) -> float | None:
    for i in range(len(samples) - 1):
        f0, x0, _ = samples[i]
        f1, x1, _ = samples[i + 1]
        if (x0 - target_x) * (x1 - target_x) <= 0 and x0 != x1:
            t = (target_x - x0) / (x1 - x0)
            return f0 + t * (f1 - f0)
    return None


def analyze(clip: Path, direction: str, elapsed_s: float, truth: float, H, det, roi, line_a_x, line_b_x) -> None:
    print(f"\n========================================")
    print(f"Pass: {clip.name}  direction={direction}  elapsed_s={elapsed_s}  truth={truth} mph")
    print(f"========================================")

    samples = trace(clip, det, roi)
    print(f"YOLO detected vehicle in {len(samples)} clip frames")

    f_a = interp_frame_at_x(samples, line_a_x)
    f_b = interp_frame_at_x(samples, line_b_x)
    if f_a is None or f_b is None:
        print("could not locate line A/B crossings")
        return
    print(f"line A crossing at clip frame {f_a:.2f}, line B at {f_b:.2f}, span={f_b-f_a:.2f} frames")

    # Re-time using |Δframe| so t always advances forward with clip frame index.
    # Camera-time per clip-frame = elapsed_s / |f_b - f_a|.
    span_frames = abs(f_b - f_a)
    if span_frames == 0:
        print("zero span")
        return
    sec_per_frame = elapsed_s / span_frames
    f_first = min(f_a, f_b)  # the first line crossing in clip-frame order
    print(f"camera-time per clip-frame = {sec_per_frame*1000:.1f} ms  "
          f"(implied capture rate ≈ {1/sec_per_frame:.1f} fps)")

    # Project + re-time every detection. t = 0 at the first line crossing.
    rows = []  # (clip_frame, t_camera, ground_x_pix, ground_y_pix, X_m, Y_m)
    for f, gx, gy in samples:
        t = (f - f_first) * sec_per_frame
        Xm, Ym = project(H, gx, gy)
        rows.append((f, t, gx, gy, Xm, Ym))

    # --- METHOD A: per-frame instantaneous speed ---
    print(f"\n--- Method A: per-frame instantaneous speed ---")
    print(f"{'clip_f':>6}  {'t (s)':>7}  {'ground (u,v)':>14}  {'(X,Y) m':>16}  "
          f"{'ΔX':>6}  {'ΔY':>6}  {'Δt':>6}  {'v_inst':>7}")
    a_speeds_mph = []
    for i in range(len(rows)):
        f, t, gx, gy, X, Y = rows[i]
        if i == 0:
            print(f"  {f:>4}    {t:>+6.3f}  ({gx:>5.1f},{gy:>5.1f})  ({X:>+6.2f},{Y:>+6.2f})  "
                  f"  ---     ---     ---     ---")
            continue
        f0, t0, gx0, gy0, X0, Y0 = rows[i - 1]
        dX, dY, dt = X - X0, Y - Y0, t - t0
        if dt <= 0:
            print(f"  {f:>4}    {t:>+6.3f}  ({gx:>5.1f},{gy:>5.1f})  ({X:>+6.2f},{Y:>+6.2f})  "
                  f"{dX:>+6.2f}  {dY:>+6.2f}  {dt:>+6.3f}    n/a")
            continue
        v_mps = math.hypot(dX, dY) / dt
        v_mph = v_mps * MPH_PER_MPS
        a_speeds_mph.append(v_mph)
        print(f"  {f:>4}    {t:>+6.3f}  ({gx:>5.1f},{gy:>5.1f})  ({X:>+6.2f},{Y:>+6.2f})  "
              f"{dX:>+6.2f}  {dY:>+6.2f}  {dt:>+6.3f}  {v_mph:>6.2f}")

    if a_speeds_mph:
        arr = np.array(a_speeds_mph)
        print(f"\n  Method A summary:")
        print(f"    n = {len(arr)} per-frame samples")
        print(f"    mean   = {arr.mean():.2f} mph")
        print(f"    median = {float(np.median(arr)):.2f} mph")
        print(f"    stdev  = {arr.std():.2f} mph")
        print(f"    min    = {arr.min():.2f} mph")
        print(f"    max    = {arr.max():.2f} mph")
        # Trimmed mean (drop top/bottom 10%) — robust to outliers from bbox jitter
        sorted_arr = np.sort(arr)
        k = max(1, len(sorted_arr) // 10)
        trimmed = sorted_arr[k:-k] if len(sorted_arr) > 2 * k else sorted_arr
        print(f"    trimmed mean (drop {k} top + {k} bottom) = {trimmed.mean():.2f} mph")
        print(f"    truth  = {truth:.2f} mph  (mean err = {(arr.mean() - truth):+.2f} mph, "
              f"{(arr.mean() - truth)/truth*100:+.2f}%)")

    # --- METHOD C: linear regression on full trajectory ---
    print(f"\n--- Method C: linear regression on (t, X) and (t, Y) ---")
    ts = np.array([r[1] for r in rows])
    Xs = np.array([r[4] for r in rows])
    Ys = np.array([r[5] for r in rows])
    A = np.vstack([ts, np.ones_like(ts)]).T
    sx, ix = np.linalg.lstsq(A, Xs, rcond=None)[0]
    sy, iy = np.linalg.lstsq(A, Ys, rcond=None)[0]
    v_mps = math.hypot(sx, sy)
    v_mph = v_mps * MPH_PER_MPS
    # R² on Y axis (dominant motion)
    Y_pred = sy * ts + iy
    ss_res = float(np.sum((Ys - Y_pred) ** 2))
    ss_tot = float(np.sum((Ys - Ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    print(f"  slope_X = {sx:+.4f} m/s")
    print(f"  slope_Y = {sy:+.4f} m/s")
    print(f"  speed   = {v_mps:.4f} m/s  =  {v_mph:.2f} mph")
    print(f"  R² (Y)  = {r2:.4f}")
    print(f"  truth   = {truth:.2f} mph  (err = {(v_mph - truth):+.2f} mph, "
          f"{(v_mph - truth)/truth*100:+.2f}%)")


def main() -> None:
    cal = yaml.safe_load((REPO / "config/calibration.yaml").read_text())
    line_a_x = float(cal["line_a_x"])
    line_b_x = float(cal["line_b_x"])
    roi = (
        int(cal["roi_x1"]), int(cal["roi_y1"]),
        int(cal["roi_x2"]), int(cal["roi_y2"]),
    )
    H = load_H()
    det = Detector(weights="yolo11n.pt", device="mps", classes=[2,3,5,7], conf=0.35, iou=0.5, roi=None)
    for clip_rel, direction, elapsed, truth in TESTS:
        analyze(REPO / clip_rel, direction, elapsed, truth, H, det, roi, line_a_x, line_b_x)


if __name__ == "__main__":
    main()
