"""Tune the homography's calibration X-coords to match ground-truth 30 mph drives.

Inputs we trust:
  - Pixel positions of all 13 marked points (trustworthy clicks)
  - Y coordinates of 1-11 spaced 5 ft apart (tape measure verified)
  - Y coordinates of 12, 13 at +25 / -25 ft (from being perpendicular to 1, 11)
  - Two test clips with verified 30 mph speedometer readings

Inputs we don't trust:
  - X = 0 for points 1-11: user may have clicked on top of curb, not at
    curb-road interface
  - X = -road_width for points 12, 13: user can't see bottom of west curb
    from camera position; may have clicked on top of curb or on grass

Free parameters:
  X_east  — actual X (in meters) of points 1-11 in the world frame
  X_west  — actual X of points 12, 13

Procedure:
  1. Replay both test clips through YOLO once; cache pixel trajectories.
  2. Grid-search (X_east, X_west). For each candidate:
        - Build H from (pixel_pts, world_pts(X_east, X_west))
        - Project line crossings through H, compute distance, divide by
          known elapsed_s, get speed
        - Total loss = (speed_S - 30)^2 + (speed_N - 30)^2
  3. Report best (X_east, X_west) and update config/homography.yaml.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from camwatch.detect import Detector  # noqa: E402

MPH_PER_MPS = 2.2369362920544
FT_TO_M = 0.3048
MAIN_TO_SUB = 3.2

TESTS = [
    ("recordings/cal_20260508T132934_id57_S.mp4", "S", 1.086, 30.0),
    ("recordings/cal_20260508T133032_id60_N.mp4", "N", 0.928, 30.0),
]


def load_marked_main_pixels() -> dict[int, tuple[int, int]]:
    data = yaml.safe_load((REPO / "config/marked_points.yaml").read_text())["marked_points"]
    return {int(p["idx"]): tuple(p["pixel"]) for p in data["points"]}


def world_for(idx: int, X_east: float, X_west: float) -> tuple[float, float]:
    if 1 <= idx <= 11:
        Y_ft = (6 - idx) * 5
        return (X_east, Y_ft * FT_TO_M)
    if idx == 12:
        return (X_west, 25 * FT_TO_M)
    if idx == 13:
        return (X_west, -25 * FT_TO_M)
    raise ValueError(idx)


def build_H(main_pixels: dict[int, tuple[int, int]], X_east: float, X_west: float) -> np.ndarray | None:
    src = []
    dst = []
    for idx in range(1, 14):
        u_main, v_main = main_pixels[idx]
        src.append([u_main / MAIN_TO_SUB, v_main / MAIN_TO_SUB])
        dst.append(list(world_for(idx, X_east, X_west)))
    H, _ = cv2.findHomography(
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32),
        method=0,
    )
    return H


def trace_clip(clip: Path, det: Detector, roi: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    """Return list of (ground_x_pix, ground_y_pix) per frame (only frames with detections)."""
    cap = cv2.VideoCapture(str(clip))
    samples: list[tuple[float, float]] = []
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
            samples.append(best)
    cap.release()
    return samples


def interp_y_at_x(traj: list[tuple[float, float]], target_x: float) -> float | None:
    for i in range(len(traj) - 1):
        x0, y0 = traj[i]
        x1, y1 = traj[i + 1]
        if (x0 - target_x) * (x1 - target_x) <= 0 and x0 != x1:
            t = (target_x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


def speed_from_traj(
    H: np.ndarray, traj: list[tuple[float, float]],
    line_a_x: float, line_b_x: float, elapsed_s: float,
) -> tuple[float, tuple[float, float], tuple[float, float]] | None:
    y_a = interp_y_at_x(traj, line_a_x)
    y_b = interp_y_at_x(traj, line_b_x)
    if y_a is None or y_b is None:
        return None
    pa = H @ np.array([line_a_x, y_a, 1.0])
    pb = H @ np.array([line_b_x, y_b, 1.0])
    Xa, Ya = pa[0] / pa[2], pa[1] / pa[2]
    Xb, Yb = pb[0] / pb[2], pb[1] / pb[2]
    dist = math.hypot(Xa - Xb, Ya - Yb)
    return dist / elapsed_s * MPH_PER_MPS, (Xa, Ya), (Xb, Yb)


def main() -> None:
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

    print("Replaying test clips through YOLO …")
    trajs = []
    for clip_rel, direction, elapsed, truth in TESTS:
        traj = trace_clip(REPO / clip_rel, det, roi)
        print(f"  {direction}: {len(traj)} frames with vehicle bbox")
        trajs.append((traj, direction, elapsed, truth))

    main_pixels = load_marked_main_pixels()

    # Grid search. Plausible ranges:
    #   X_east ∈ [-0.5, +0.2] m  — points 1-11 are clicked at curb-edge,
    #     could be slightly inside or outside actual road-plane edge
    #   X_west ∈ [-11.0, -8.5] m  — points 12, 13 might be on top of curb
    #     or on grass beyond, so true west-curb-bottom is more negative
    print("\nGrid-searching (X_east, X_west) …")
    best = None
    coarse_e = np.linspace(-0.5, 0.2, 36)
    coarse_w = np.linspace(-11.0, -8.5, 51)
    for X_east in coarse_e:
        for X_west in coarse_w:
            H = build_H(main_pixels, float(X_east), float(X_west))
            if H is None:
                continue
            losses = []
            speeds = []
            ok = True
            for traj, direction, elapsed, truth in trajs:
                r = speed_from_traj(H, traj, line_a_x, line_b_x, elapsed)
                if r is None:
                    ok = False
                    break
                speed, _, _ = r
                speeds.append(speed)
                losses.append((speed - truth) ** 2)
            if not ok:
                continue
            total = sum(losses)
            if best is None or total < best[0]:
                best = (total, float(X_east), float(X_west), tuple(speeds))

    if best is None:
        sys.exit("no valid configuration found in grid")

    loss, X_east, X_west, speeds = best
    print(f"\nBest fit: X_east={X_east:+.3f} m  X_west={X_west:+.3f} m  total loss={loss:.4f}")
    for (traj, direction, elapsed, truth), speed in zip(trajs, speeds):
        err_pct = (speed - truth) / truth * 100
        print(f"  {direction}-bound:  homog={speed:.2f} mph  truth={truth:.1f}  err={err_pct:+.2f}%")

    # Now refine around best with a finer grid
    print("\nRefining …")
    fine_e = np.linspace(X_east - 0.1, X_east + 0.1, 21)
    fine_w = np.linspace(X_west - 0.2, X_west + 0.2, 21)
    refined = None
    for xe in fine_e:
        for xw in fine_w:
            H = build_H(main_pixels, float(xe), float(xw))
            if H is None:
                continue
            losses = []
            speeds = []
            ok = True
            for traj, direction, elapsed, truth in trajs:
                r = speed_from_traj(H, traj, line_a_x, line_b_x, elapsed)
                if r is None:
                    ok = False
                    break
                speed, _, _ = r
                speeds.append(speed)
                losses.append((speed - truth) ** 2)
            if not ok:
                continue
            total = sum(losses)
            if refined is None or total < refined[0]:
                refined = (total, float(xe), float(xw), tuple(speeds))

    loss, X_east, X_west, speeds = refined
    print(f"\nRefined: X_east={X_east:+.3f} m  X_west={X_west:+.3f} m  total loss={loss:.4f}")
    for (traj, direction, elapsed, truth), speed in zip(trajs, speeds):
        err_pct = (speed - truth) / truth * 100
        print(f"  {direction}-bound:  homog={speed:.2f} mph  truth={truth:.1f}  err={err_pct:+.2f}%")

    # Write tuned homography
    H_final = build_H(main_pixels, X_east, X_west)
    backup = REPO / "config/homography.before_truth_tuning.yaml"
    if (REPO / "config/homography.yaml").exists():
        shutil.copy(REPO / "config/homography.yaml", backup)
        print(f"\nBacked up old homography to {backup}")

    err_per_point = []
    for idx in range(1, 14):
        u_main, v_main = main_pixels[idx]
        u_sub = u_main / MAIN_TO_SUB
        v_sub = v_main / MAIN_TO_SUB
        target = world_for(idx, X_east, X_west)
        p = H_final @ np.array([u_sub, v_sub, 1.0])
        Xm, Ym = p[0] / p[2], p[1] / p[2]
        err = math.hypot(Xm - target[0], Ym - target[1])
        err_per_point.append(err)

    payload = {
        "homography": {
            "H": H_final.tolist(),
            "frame_size_sub": [640, 480],
            "frame_size_main": [2048, 1536],
            "main_to_sub_scale": MAIN_TO_SUB,
            "origin": "point 6 — east curb, camera's perpendicular",
            "axes": "+X = east, +Y = along road toward point 1",
            "X_east_tuned_m": X_east,
            "X_west_tuned_m": X_west,
            "road_width_implied_ft": (X_east - X_west) / FT_TO_M,
            "spacing_ft": 5.0,
            "method": "cv2.findHomography least-squares + ground-truth-tuned X anchors",
            "max_reprojection_error_m": float(max(err_per_point)),
            "mean_reprojection_error_m": float(np.mean(err_per_point)),
            "tuned_against": [
                {"clip": c, "dir": d, "elapsed_s": e, "truth_mph": t,
                 "homog_mph": float(s)}
                for (c, d, e, t), s in zip(TESTS, speeds)
            ],
            "pixel_pts_sub": [
                {"idx": i, "u": float(main_pixels[i][0] / MAIN_TO_SUB),
                 "v": float(main_pixels[i][1] / MAIN_TO_SUB)}
                for i in range(1, 14)
            ],
            "pixel_pts_main": [
                {"idx": i, "u": int(main_pixels[i][0]), "v": int(main_pixels[i][1])}
                for i in range(1, 14)
            ],
            "meter_pts": [
                {"idx": i, "X": float(world_for(i, X_east, X_west)[0]),
                 "Y": float(world_for(i, X_east, X_west)[1])}
                for i in range(1, 14)
            ],
        }
    }
    (REPO / "config/homography.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote tuned homography to {REPO / 'config/homography.yaml'}")


if __name__ == "__main__":
    main()
