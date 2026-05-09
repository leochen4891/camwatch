"""Tune homography using BOTH the asphalt + curb click sets, with the
asphalt's X offset and the west-corner X position as free parameters,
optimized against the two 30 mph ground-truth passes.

Combined anchors (26 east-side / 2 west-side):
  - 11 new curb points (X = 0, tape-measured Y at 5ft spacings)
  - 11 old asphalt points (X = X_asphalt unknown, same Y values)
  - 2 west corners (X = X_west unknown, Y = ±25 ft)

Search:
  X_asphalt ∈ [-1.5, -0.3] m  (user said 2-4 ft into asphalt = -0.6 to -1.2 m)
  X_west    ∈ [-11.0, -8.0] m

Loss = (S_pass_speed - 30)² + (N_pass_speed - 30)²
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


def load_marks(path: Path) -> dict[int, tuple[int, int]]:
    data = yaml.safe_load(path.read_text())["marked_points"]
    return {int(p["idx"]): tuple(p["pixel"]) for p in data["points"]}


def y_for(idx: int) -> float:
    """Y in meters for points 1-11, 12, 13."""
    if 1 <= idx <= 11:
        return (6 - idx) * 5 * FT_TO_M
    if idx == 12:
        return 25 * FT_TO_M
    if idx == 13:
        return -25 * FT_TO_M
    raise ValueError(idx)


def build_H(
    curb: dict[int, tuple[int, int]],
    asphalt: dict[int, tuple[int, int]],
    X_asphalt: float, X_west: float,
) -> np.ndarray:
    src, dst = [], []
    # New curb points 1-11 at X=0
    for i in range(1, 12):
        u, v = curb[i]
        src.append([u / MAIN_TO_SUB, v / MAIN_TO_SUB])
        dst.append([0.0, y_for(i)])
    # New west corners 12, 13 at X=X_west
    for i in (12, 13):
        u, v = curb[i]
        src.append([u / MAIN_TO_SUB, v / MAIN_TO_SUB])
        dst.append([X_west, y_for(i)])
    # Old asphalt points 1-11 at X=X_asphalt
    for i in range(1, 12):
        u, v = asphalt[i]
        src.append([u / MAIN_TO_SUB, v / MAIN_TO_SUB])
        dst.append([X_asphalt, y_for(i)])
    # NOTE: we don't reuse the asphalt 12, 13 since they're redundant with curb 12,13
    H, _ = cv2.findHomography(
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32),
        method=0,
    )
    return H


def trace_clip(clip: Path, det: Detector, roi: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    cap = cv2.VideoCapture(str(clip))
    samples: list[tuple[float, float]] = []
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
            area = (x2 - x1) * (y2 - y1)
            if area > best_a:
                best, best_a = (cx, gy), area
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


def speed_via(H: np.ndarray, traj, line_a_x, line_b_x, elapsed_s) -> float | None:
    y_a = interp_y_at_x(traj, line_a_x)
    y_b = interp_y_at_x(traj, line_b_x)
    if y_a is None or y_b is None:
        return None
    pa = H @ np.array([line_a_x, y_a, 1.0])
    pb = H @ np.array([line_b_x, y_b, 1.0])
    Xa, Ya = pa[0] / pa[2], pa[1] / pa[2]
    Xb, Yb = pb[0] / pb[2], pb[1] / pb[2]
    return math.hypot(Xa - Xb, Ya - Yb) / elapsed_s * MPH_PER_MPS


def main() -> None:
    cal = yaml.safe_load((REPO / "config/calibration.yaml").read_text())
    line_a_x = float(cal["line_a_x"])
    line_b_x = float(cal["line_b_x"])
    roi = (
        int(cal["roi_x1"]), int(cal["roi_y1"]),
        int(cal["roi_x2"]), int(cal["roi_y2"]),
    )

    curb = load_marks(REPO / "config/marked_points.yaml")
    asphalt = load_marks(REPO / "config/marked_points_asphalt.yaml")
    print(f"Curb clicks: {len(curb)}, asphalt clicks: {len(asphalt)}")

    det = Detector(weights="yolo11n.pt", device="mps", classes=[2,3,5,7], conf=0.35, iou=0.5, roi=None)

    print("Replaying test clips through YOLO …")
    trajs = []
    for clip_rel, direction, elapsed, truth in TESTS:
        traj = trace_clip(REPO / clip_rel, det, roi)
        trajs.append((traj, direction, elapsed, truth))
        print(f"  {direction}: {len(traj)} frames")

    print("\nGrid-searching (X_asphalt, X_west) …")
    best = None
    coarse_a = np.linspace(-1.5, -0.3, 25)
    coarse_w = np.linspace(-11.0, -8.0, 31)
    for Xa in coarse_a:
        for Xw in coarse_w:
            H = build_H(curb, asphalt, float(Xa), float(Xw))
            if H is None:
                continue
            speeds, ok = [], True
            for traj, _, elapsed, _ in trajs:
                s = speed_via(H, traj, line_a_x, line_b_x, elapsed)
                if s is None:
                    ok = False; break
                speeds.append(s)
            if not ok:
                continue
            loss = sum((s - 30) ** 2 for s in speeds)
            if best is None or loss < best[0]:
                best = (loss, float(Xa), float(Xw), tuple(speeds))

    loss, Xa, Xw, speeds = best
    print(f"\nCoarse: X_asphalt={Xa:+.3f} m  X_west={Xw:+.3f} m  loss={loss:.3f}")
    for (traj, d, e, t), s in zip(trajs, speeds):
        print(f"  {d}-bound: homog={s:.2f}  truth={t}  err={(s-t)/t*100:+.2f}%")

    # Refine
    print("\nRefining …")
    fine_a = np.linspace(Xa - 0.1, Xa + 0.1, 21)
    fine_w = np.linspace(Xw - 0.2, Xw + 0.2, 21)
    refined = None
    for xa in fine_a:
        for xw in fine_w:
            H = build_H(curb, asphalt, float(xa), float(xw))
            if H is None: continue
            speeds, ok = [], True
            for traj, _, elapsed, _ in trajs:
                s = speed_via(H, traj, line_a_x, line_b_x, elapsed)
                if s is None: ok = False; break
                speeds.append(s)
            if not ok: continue
            loss = sum((s - 30) ** 2 for s in speeds)
            if refined is None or loss < refined[0]:
                refined = (loss, float(xa), float(xw), tuple(speeds))

    loss, Xa, Xw, speeds = refined
    print(f"\nRefined: X_asphalt={Xa:+.3f} m ({Xa/FT_TO_M:.2f} ft)  "
          f"X_west={Xw:+.3f} m ({Xw/FT_TO_M:.2f} ft)  loss={loss:.4f}")
    for (traj, d, e, t), s in zip(trajs, speeds):
        print(f"  {d}-bound: homog={s:.3f}  truth={t}  err={(s-t)/t*100:+.3f}%")

    # Save
    H_final = build_H(curb, asphalt, Xa, Xw)
    backup = REPO / "config/homography.before_combined_tuning.yaml"
    if (REPO / "config/homography.yaml").exists():
        shutil.copy(REPO / "config/homography.yaml", backup)
        print(f"\nBacked up to {backup}")

    payload = {
        "homography": {
            "H": H_final.tolist(),
            "frame_size_sub": [640, 480],
            "frame_size_main": [2048, 1536],
            "main_to_sub_scale": MAIN_TO_SUB,
            "method": "combined curb+asphalt anchors, X_asphalt and X_west tuned to 30 mph ground truth",
            "X_asphalt_tuned_m": Xa,
            "X_west_tuned_m": Xw,
            "X_curb": 0.0,
            "spacing_ft": 5.0,
            "tuned_against": [
                {"clip": c, "dir": d, "elapsed_s": e, "truth_mph": t,
                 "homog_mph": float(s)}
                for (c, d, e, t), s in zip(TESTS, speeds)
            ],
            # Reproject all anchor points for diagnostics
            "anchor_count": 24,
        }
    }
    (REPO / "config/homography.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote {REPO / 'config/homography.yaml'}")


if __name__ == "__main__":
    main()
