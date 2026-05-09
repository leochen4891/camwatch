"""Homography-based pixel→ground-plane projection and trajectory speed.

Loads the 3×3 matrix from `config/homography.yaml` (built by
`scripts/build_homography_from_marks.py`). The matrix maps sub-stream
pixel coordinates (640×480) to road-plane meters in a coordinate
system whose origin is at "point 6" (the east curb directly across
from the camera) with +Y running along the road toward "point 1"
(north-ish).

Speed = |slope of Y(t)| from a linear regression over the projected
trajectory of a single track. The 5 ft east-curb spacing (11 hand-
clicked points) anchors the Y-axis scale, so speed is not sensitive
to the assumed road width.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

log = logging.getLogger(__name__)

MPH_PER_MPS = 2.2369362920544


@dataclass
class Homography:
    H: np.ndarray                # 3×3, sub-stream pixel → meters
    inv_H: np.ndarray            # for visualization / inverse projection
    frame_size_sub: tuple[int, int]
    mean_reproj_err_m: float
    max_reproj_err_m: float

    @classmethod
    def load(cls, path: Path | str) -> "Homography | None":
        try:
            data = yaml.safe_load(Path(path).read_text())["homography"]
        except FileNotFoundError:
            log.warning("homography file not found at %s; speed-by-homography disabled", path)
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("failed to load homography from %s: %s; speed-by-homography disabled", path, e)
            return None
        H = np.array(data["H"], dtype=np.float64)
        return cls(
            H=H,
            inv_H=np.linalg.inv(H),
            frame_size_sub=tuple(data.get("frame_size_sub", (640, 480))),
            mean_reproj_err_m=float(data.get("mean_reprojection_error_m", 0.0)),
            max_reproj_err_m=float(data.get("max_reprojection_error_m", 0.0)),
        )

    def project(self, u: float, v: float) -> tuple[float, float]:
        """Sub-stream pixel (u, v) → road-plane meters (X, Y)."""
        p = self.H @ np.array([float(u), float(v), 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def speed_from_trajectory(
        self,
        samples: Sequence[tuple[float, float, float]],
    ) -> tuple[float, float, int]:
        """Fit speed from a list of (t_seconds, ground_u_pixel, ground_v_pixel).

        Returns (mph, r_squared, n). Both X(t) and Y(t) are linear-regressed;
        speed is the magnitude of the velocity vector, so it works whether the
        car is going purely along Y, or has some lane-change component in X.
        """
        n = len(samples)
        if n < 3:
            return float("nan"), 0.0, n
        ts = np.array([s[0] for s in samples], dtype=np.float64)
        Xs = np.empty(n, dtype=np.float64)
        Ys = np.empty(n, dtype=np.float64)
        for i, (_, u, v) in enumerate(samples):
            X, Y = self.project(u, v)
            Xs[i] = X
            Ys[i] = Y
        A = np.vstack([ts, np.ones_like(ts)]).T
        slope_x, _ = np.linalg.lstsq(A, Xs, rcond=None)[0]
        slope_y, _ = np.linalg.lstsq(A, Ys, rcond=None)[0]
        # R² on the dominant (Y) axis since the road is along Y.
        Y_pred = slope_y * ts + (Ys.mean() - slope_y * ts.mean())
        ss_res = float(np.sum((Ys - Y_pred) ** 2))
        ss_tot = float(np.sum((Ys - Ys.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        v_mps = float(np.hypot(slope_x, slope_y))
        return v_mps * MPH_PER_MPS, r2, n

    def centered_speed_y_regression(
        self,
        samples: Sequence[tuple[float, float, float]],
        half_window_m: float,
    ) -> tuple[float, float, int]:
        """Speed estimate via linear regression of Y vs t over samples whose
        projected Y is within ±half_window_m of Y=0 (the camera's
        perpendicular line, where homography reprojection error is smallest).

        Returns (mph, r_squared, n_samples). The fit is Y = m·t + b; speed is
        |m| in m/s converted to mph. Equivalent to "average speed across the
        central 2·half_window_m of the road," anchored at the most accurate
        portion of the calibrated grid.

        If fewer than 3 samples fall inside the window the fit is unreliable
        and we return (nan, 0.0, n).
        """
        if len(samples) < 3:
            return float("nan"), 0.0, 0
        ts_filt: list[float] = []
        Ys_filt: list[float] = []
        for ts, u, v in samples:
            _X, Y = self.project(u, v)
            if -half_window_m <= Y <= half_window_m:
                ts_filt.append(float(ts))
                Ys_filt.append(float(Y))
        n = len(ts_filt)
        if n < 3:
            return float("nan"), 0.0, n
        ts_arr = np.array(ts_filt, dtype=np.float64)
        Ys_arr = np.array(Ys_filt, dtype=np.float64)
        A = np.vstack([ts_arr, np.ones_like(ts_arr)]).T
        slope_y, intercept = np.linalg.lstsq(A, Ys_arr, rcond=None)[0]
        Y_pred = slope_y * ts_arr + intercept
        ss_res = float(np.sum((Ys_arr - Y_pred) ** 2))
        ss_tot = float(np.sum((Ys_arr - Ys_arr.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        return abs(float(slope_y)) * MPH_PER_MPS, r2, n

    def median_speed_in_grid(
        self,
        samples: Sequence[tuple[float, float, float]],
        grid_x_min: float, grid_x_max: float,
        grid_y_min: float, grid_y_max: float,
        tolerance_m: float = 0.5,
    ) -> tuple[float, int]:
        """Method-A speed estimator: project each sample, compute per-frame
        v_inst between consecutive samples that are BOTH inside the grid
        (with tolerance), then take the median.

        Robust to per-frame bbox jitter and to homography extrapolation at
        the trajectory edges. Returns (median_mph, n_samples_used). If fewer
        than 2 in-grid pairs exist, returns (nan, 0).
        """
        if len(samples) < 2:
            return float("nan"), 0
        xmin = grid_x_min - tolerance_m
        xmax = grid_x_max + tolerance_m
        ymin = grid_y_min - tolerance_m
        ymax = grid_y_max + tolerance_m
        projected: list[tuple[float, float, float, bool]] = []
        for ts, u, v in samples:
            X, Y = self.project(u, v)
            in_grid = (xmin <= X <= xmax and ymin <= Y <= ymax)
            projected.append((ts, X, Y, in_grid))
        v_inst: list[float] = []
        for i in range(1, len(projected)):
            ts_i, X_i, Y_i, ig_i = projected[i]
            ts_p, X_p, Y_p, ig_p = projected[i - 1]
            if not (ig_i and ig_p):
                continue
            dt = ts_i - ts_p
            if dt <= 0:
                continue
            d = ((X_i - X_p) ** 2 + (Y_i - Y_p) ** 2) ** 0.5
            v_inst.append((d / dt) * MPH_PER_MPS)
        if len(v_inst) < 2:
            return float("nan"), len(v_inst)
        v_inst.sort()
        n = len(v_inst)
        median = (v_inst[n // 2] if n % 2 else (v_inst[n // 2 - 1] + v_inst[n // 2]) / 2.0)
        return median, n
