"""Try several anchor-subset variants of the homography fit and re-project
recent test passes through each, to see which (if any) reduces the
lane-dependent bias.

Subsets tested:
  A) 13 anchors as-is (current: 4 corners + 9 raw east-curb)
  B) 4 corners only (NE, SE, NW, SW)
  C) East side only + NW alone (12 anchors): drops SW
  D) East side only + SW alone (12 anchors): drops NW

Test passes (re-projected with each H):
  1502, 1503, 1505 — all driven at 25 mph in W lane (X ≈ −7 m)
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
MARKED = REPO / "config" / "marked_points.yaml"

FT_TO_M = 0.3048
ROAD_WIDTH_FT = 30.0
MAIN_TO_SUB = 3.2
MPH_PER_MPS = 2.2369362920544


def world_for(idx: int) -> tuple[float, float]:
    if 1 <= idx <= 11:
        return (0.0, (6 - idx) * 5 * FT_TO_M)
    if idx == 12:
        return (-ROAD_WIDTH_FT * FT_TO_M, +25 * FT_TO_M)
    if idx == 13:
        return (-ROAD_WIDTH_FT * FT_TO_M, -25 * FT_TO_M)
    raise ValueError(idx)


def load_marked_subpix() -> dict[int, tuple[float, float]]:
    data = yaml.safe_load(MARKED.read_text())["marked_points"]
    return {
        int(p["idx"]): (p["pixel"][0] / MAIN_TO_SUB, p["pixel"][1] / MAIN_TO_SUB)
        for p in data["points"]
    }


def fit_subset(
    subpix: dict[int, tuple[float, float]],
    idx_list: list[int],
    west_x_override: float | None = None,
) -> np.ndarray | None:
    """Fit H with the given anchor indices. If west_x_override is set, NW (12)
    and SW (13) get their world X shifted to that value (e.g., to compensate
    for west-curb-top click elevation by treating them as further west)."""
    src_pts, dst_pts = [], []
    for i in idx_list:
        src_pts.append(list(subpix[i]))
        wx, wy = world_for(i)
        if west_x_override is not None and i in (12, 13):
            wx = west_x_override
        dst_pts.append([wx, wy])
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=0)
    return H


def reproject(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def reanalyze_pass(H: np.ndarray, pass_id: int) -> dict:
    path = REPO / "events" / f"pass_{pass_id}.jsonl"
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    samples = rows[1:]

    GRID_X_MIN = -ROAD_WIDTH_FT * FT_TO_M - 0.5
    GRID_X_MAX = 0.0 + 0.5
    GRID_Y_MIN = -25 * FT_TO_M - 0.5
    GRID_Y_MAX = +25 * FT_TO_M + 0.5

    proj = []
    for s in samples:
        X, Y = reproject(H, s["u"], s["v"])
        in_grid = (GRID_X_MIN <= X <= GRID_X_MAX and GRID_Y_MIN <= Y <= GRID_Y_MAX)
        proj.append({"ts": s["ts"], "X": X, "Y": Y, "in_grid": in_grid, "u": s["u"]})

    v_inst_mph = []
    Xs_in = []
    for i in range(1, len(proj)):
        if not (proj[i]["in_grid"] and proj[i-1]["in_grid"]):
            continue
        dt = proj[i]["ts"] - proj[i-1]["ts"]
        if dt <= 0:
            continue
        d = math.hypot(proj[i]["X"] - proj[i-1]["X"], proj[i]["Y"] - proj[i-1]["Y"])
        v_inst_mph.append((d / dt) * MPH_PER_MPS)
        Xs_in.append(proj[i]["X"])
    return {
        "n": len(v_inst_mph),
        "mean": statistics.mean(v_inst_mph) if v_inst_mph else None,
        "median": statistics.median(v_inst_mph) if v_inst_mph else None,
        "stdev": statistics.stdev(v_inst_mph) if len(v_inst_mph) > 1 else None,
        "X_mean": (sum(Xs_in) / len(Xs_in)) if Xs_in else None,
    }


def main() -> None:
    subpix = load_marked_subpix()

    SUBSETS = [
        ("A: 13 anchors, west X = -9.144",  list(range(1, 14)), None),
        ("E: 13 anchors, west X = -9.5",    list(range(1, 14)), -9.5),
        ("F: 13 anchors, west X = -10.0",   list(range(1, 14)), -10.0),
        ("G: 13 anchors, west X = -10.5",   list(range(1, 14)), -10.5),
        ("H: 13 anchors, west X = -11.0",   list(range(1, 14)), -11.0),
    ]

    PASSES = [
        ("1502", "N", 25.0),
        ("1503", "S", 25.0),
        ("1505", "N", 25.0),
    ]

    print(f"{'subset':<32}  {'pass':<5}  {'dir':>3}  {'truth':>6}  "
          f"{'X_mean':>7}  {'A_mean':>7}  {'A_median':>9}  {'stdev':>5}  {'n':>3}")
    print("-" * 100)
    for label, idx_list, west_override in SUBSETS:
        H = fit_subset(subpix, idx_list, west_override)
        if H is None:
            print(f"{label}: FIT FAILED")
            continue
        for pid, dirn, truth in PASSES:
            r = reanalyze_pass(H, int(pid))
            if r["median"] is None:
                continue
            print(f"{label:<32}  {pid:<5}  {dirn:>3}  {truth:>6.1f}  "
                  f"{r['X_mean']:>+7.2f}  {r['mean']:>7.2f}  {r['median']:>9.2f}  "
                  f"{r['stdev']:>5.2f}  {r['n']:>3}")
        print()


if __name__ == "__main__":
    main()
