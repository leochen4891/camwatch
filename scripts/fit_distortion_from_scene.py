"""Joint fit of (K, D, H) from scene constraints (the 17 marked dots).

Why: the CX410W's wide lens has barrel distortion a plain homography can't
absorb. A homography fit alone landed mean=63 cm / max=125 cm residuals.

This script uses the known scene geometry — 14 east-curb dots colinear at
X=0, equally spaced 5 ft in Y; 3 west-curb dots colinear at X=-30 ft — to
jointly fit:
  fx (= fy):     focal length in pixels, init'd from spec HFOV=89°
  D = (k1..k3, p1, p2):  standard 5-coefficient distortion model
  H (3×3):       maps undistorted pixels → road meters

cx, cy are fixed at image center (well within a degree of true principal
point for any commodity camera; not worth fitting from this little data).

That's 14 free parameters vs 17 anchors × 2 coords = 34 equations.
Solved with scipy.optimize.least_squares (Levenberg-Marquardt).

At runtime: a single point (u, v) → (X, Y) takes one cv2.undistortPoints
call plus the 3×3 matmul. No full-frame undistortion needed.

Usage:
    uv run python scripts/fit_distortion_from_scene.py
    uv run python scripts/fit_distortion_from_scene.py --hfov-deg 89
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parent.parent
MARKED = REPO / "config" / "marked_points.yaml"
OUT_HOMOG = REPO / "config" / "homography.yaml"
BACKUP_HOMOG = REPO / "config" / "homography.scene_v0.yaml"

FT_TO_M = 0.3048
SPACING_FT = 5.0
ROAD_WIDTH_FT = 30.0


def world_for(idx: int) -> tuple[float, float]:
    """World (X, Y) in meters for marked-point idx (1..17). Layout doc:
    see scripts/build_homography_from_marks.py."""
    if 1 <= idx <= 14:
        y_ft = (6 - idx) * SPACING_FT  # idx 6 → 0, idx 1 → +25, idx 14 → -40
        return (0.0, y_ft * FT_TO_M)
    if idx == 15:
        return (-ROAD_WIDTH_FT * FT_TO_M, -8 * SPACING_FT * FT_TO_M)  # Y = -40 ft
    if idx == 16:
        return (-ROAD_WIDTH_FT * FT_TO_M, -5 * SPACING_FT * FT_TO_M)  # Y = -25 ft
    if idx == 17:
        return (-ROAD_WIDTH_FT * FT_TO_M, +5 * SPACING_FT * FT_TO_M)  # Y = +25 ft
    raise ValueError(f"unexpected idx {idx}")


def build_K(fx: float, cx: float, cy: float) -> np.ndarray:
    """Pinhole intrinsics with fy = fx (square pixels)."""
    return np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def project_anchors(params: np.ndarray, src: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Apply (K from fx, D, H from params) to src=Nx2 distorted pixels → Nx2 world meters."""
    fx = params[0]
    D = params[1:6].astype(np.float64).reshape(5)
    h_flat = params[6:14]
    H = np.array(
        [
            [h_flat[0], h_flat[1], h_flat[2]],
            [h_flat[3], h_flat[4], h_flat[5]],
            [h_flat[6], h_flat[7], 1.0],
        ],
        dtype=np.float64,
    )
    K = build_K(fx, cx, cy)
    # cv2.undistortPoints wants Nx1x2; P=K returns coords back in pixel space.
    pts = src.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)
    n = undist.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    homog = np.hstack([undist, ones])  # N×3
    proj = homog @ H.T  # N×3
    out = proj[:, :2] / proj[:, 2:3]
    return out


def fit_from_anchors(
    main_pixels: dict[int, tuple[float, float]],
    frame_w: int,
    frame_h: int,
    hfov_deg: float = 89.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Joint LM fit of (fx, D, H) from 17 (u,v) anchors at known world positions.

    Returns: (H_fit, K_fit, D_fit, errors_m, mean_err_m, max_err_m).

    Used both by the standalone script (with --hfov-deg flag) and by the
    interactive calibrator inspect_homography.py after a drag releases.
    """
    if set(main_pixels.keys()) != set(range(1, 18)):
        raise ValueError(f"expected indices 1..17, got {sorted(main_pixels.keys())}")
    cx, cy = frame_w / 2.0, frame_h / 2.0
    src = np.array([main_pixels[i] for i in range(1, 18)], dtype=np.float64)
    dst = np.array([world_for(i) for i in range(1, 18)], dtype=np.float64)

    fx0 = (frame_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    H_init, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), method=0)
    h_init = H_init.flatten()
    h_init = h_init / h_init[8]
    x0 = np.concatenate([[fx0], np.zeros(5), h_init[:8]])

    def residuals(params: np.ndarray) -> np.ndarray:
        pred = project_anchors(params, src, cx, cy)
        return (pred - dst).ravel()

    result = least_squares(
        residuals, x0, method="lm",
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
        max_nfev=10000,
    )

    fx_fit = float(result.x[0])
    D_fit = np.array(result.x[1:6], dtype=np.float64)
    h_fit = np.append(result.x[6:14], 1.0).reshape(3, 3).astype(np.float64)
    K_fit = build_K(fx_fit, cx, cy)

    pred = project_anchors(result.x, src, cx, cy)
    errors_m = np.linalg.norm(pred - dst, axis=1)
    err_mean = float(errors_m.mean())
    err_max = float(errors_m.max())

    if verbose:
        print(f"\nConverged: {result.success}  status={result.status}  nfev={result.nfev}")
        print(f"Fitted fx = {fx_fit:.2f} px (init {fx0:.1f} → moved {fx_fit - fx0:+.1f})")
        print(f"Fitted D  = [k1={D_fit[0]:+.4f}, k2={D_fit[1]:+.4f}, p1={D_fit[2]:+.5f}, p2={D_fit[3]:+.5f}, k3={D_fit[4]:+.4f}]")
        print(f"\n{'#':>3}  {'click (u,v)':>16}  {'world target (m)':>22}  {'projected (m)':>22}  {'err (cm)':>8}")
        print("-" * 90)
        for i in range(17):
            u, v = src[i]
            tx, ty = dst[i]
            px, py = pred[i]
            print(f"  {i+1:>3}  ({u:7.1f},{v:7.1f})  ({tx:>+7.3f},{ty:>+7.3f})  ({px:>+7.3f},{py:>+7.3f})  {errors_m[i]*100:>8.2f}")
        print(f"\nReprojection over 17 anchors: mean={err_mean*100:.2f} cm, max={err_max*100:.2f} cm")

    return h_fit, K_fit, D_fit, errors_m, err_mean, err_max


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hfov-deg", type=float, default=89.0,
                    help="Camera horizontal FOV in degrees (CX410W spec = 89°). Sets initial fx.")
    args = ap.parse_args()

    data = yaml.safe_load(MARKED.read_text())["marked_points"]
    pts = data["points"]
    if len(pts) != 17:
        raise SystemExit(f"expected 17 marked points, got {len(pts)}")
    frame_w, frame_h = [int(v) for v in data["frame_size"]]
    print(f"Frame: {frame_w}x{frame_h}")

    main_pixels = {int(p["idx"]): (float(p["pixel"][0]), float(p["pixel"][1])) for p in pts}
    print(f"Fitting (Levenberg-Marquardt) from {len(main_pixels)} anchors …")
    h_fit, K_fit, D_fit, errors_m, err_mean, err_max = fit_from_anchors(
        main_pixels, frame_w, frame_h, hfov_deg=args.hfov_deg, verbose=True,
    )

    # Preserve manually-curated fields (east-curb offsets, etc.) across re-fits.
    preserve: dict = {}
    if OUT_HOMOG.exists():
        existing = yaml.safe_load(OUT_HOMOG.read_text()).get("homography", {})
        for key in ("east_curb_offset_in_at_y_ft", "east_curb_offset_note"):
            if key in existing:
                preserve[key] = existing[key]

    # Backup and write
    if OUT_HOMOG.exists():
        shutil.copy(OUT_HOMOG, BACKUP_HOMOG)
        print(f"Backed up old homography → {BACKUP_HOMOG}")

    payload = {
        "homography": {
            "H": h_fit.tolist(),
            "K": K_fit.tolist(),
            "D": D_fit.tolist(),
            "frame_size": [frame_w, frame_h],
            "origin": "point 6 — east curb, camera's perpendicular",
            "axes": "+X = east (toward camera); +Y = along road toward point 1 (north-ish)",
            "road_width_ft": ROAD_WIDTH_FT,
            "spacing_ft": SPACING_FT,
            "method": "scene-fit (jointly optimized fx + 5-coeff D + H against 17 plumb-line anchors)",
            "hfov_init_deg": args.hfov_deg,
            "max_reprojection_error_m": err_max,
            "mean_reprojection_error_m": err_mean,
            "pixel_pts": [
                {"idx": i, "u": float(main_pixels[i][0]), "v": float(main_pixels[i][1])}
                for i in range(1, 18)
            ],
            "meter_pts": [
                {"idx": i, "X": float(world_for(i)[0]), "Y": float(world_for(i)[1])}
                for i in range(1, 18)
            ],
            **preserve,
        }
    }
    OUT_HOMOG.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote {OUT_HOMOG}")


if __name__ == "__main__":
    main()
