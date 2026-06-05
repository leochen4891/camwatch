"""Homography-based pixel→ground-plane projection and trajectory speed.

Loads the 3×3 matrix from `config/homography.yaml` (built by
`scripts/build_homography_from_marks.py`). The matrix maps main-stream
pixel coordinates (resolution depends on the camera; recorded in the
yaml's `frame_size`) to road-plane meters in a coordinate system whose
origin is at "point 6" (the east curb directly across
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

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)

MPH_PER_MPS = 2.2369362920544


@dataclass
class Homography:
    H: np.ndarray                # 3×3, undistorted-pixel → meters
    inv_H: np.ndarray            # for visualization / inverse projection
    frame_size: tuple[int, int]
    mean_reproj_err_m: float
    max_reproj_err_m: float
    # Optional intrinsics + distortion. When both are present, project() runs
    # cv2.undistortPoints before applying H. Built by
    # scripts/fit_distortion_from_scene.py for the CX410W; older configs
    # without K/D fall back to plain pinhole (no undistort).
    K: np.ndarray | None = None
    D: np.ndarray | None = None

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
        K = np.array(data["K"], dtype=np.float64) if "K" in data else None
        D = np.array(data["D"], dtype=np.float64).reshape(-1) if "D" in data else None
        return cls(
            H=H,
            inv_H=np.linalg.inv(H),
            frame_size=tuple(data.get("frame_size", (2048, 1536))),
            mean_reproj_err_m=float(data.get("mean_reprojection_error_m", 0.0)),
            max_reproj_err_m=float(data.get("max_reprojection_error_m", 0.0)),
            K=K,
            D=D,
        )

    def project(self, u: float, v: float) -> tuple[float, float]:
        """Main-stream pixel (u, v) → road-plane meters (X, Y).

        If K + D are loaded, the input pixel is run through
        cv2.undistortPoints first so H operates on undistorted coords.
        """
        if self.K is not None and self.D is not None:
            pt = np.array([[[float(u), float(v)]]], dtype=np.float64)
            undist = cv2.undistortPoints(pt, self.K, self.D, P=self.K).reshape(2)
            uu, vv = float(undist[0]), float(undist[1])
        else:
            uu, vv = float(u), float(v)
        p = self.H @ np.array([uu, vv, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def world_to_pixel(self, X: float, Y: float) -> tuple[float, float]:
        """Inverse of project(): road meters (X, Y) → distorted pixel (u, v).

        If K + D are loaded, applies cv2.projectPoints to re-distort after
        inv_H so the output lands where the world point actually appears in
        the lens-distorted image. Used for drawing world-aligned overlays.
        """
        p = self.inv_H @ np.array([float(X), float(Y), 1.0])
        if abs(p[2]) < 1e-9:
            return float("nan"), float("nan")
        u_undist = p[0] / p[2]
        v_undist = p[1] / p[2]
        if self.K is None or self.D is None:
            return float(u_undist), float(v_undist)
        x_cam = (u_undist - self.K[0, 2]) / self.K[0, 0]
        y_cam = (v_undist - self.K[1, 2]) / self.K[1, 1]
        pts3d = np.array([[[x_cam, y_cam, 1.0]]], dtype=np.float64)
        out, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), self.K, self.D)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def world_polyline(self, X1: float, Y1: float, X2: float, Y2: float, n: int = 32) -> np.ndarray:
        """Sample a straight world-space segment into a dense pixel-space polyline
        that follows the lens curvature. Returns Nx2 int32 (u, v) array."""
        ts = np.linspace(0.0, 1.0, n)
        pts: list[tuple[int, int]] = []
        for t in ts:
            X = X1 + (X2 - X1) * t
            Y = Y1 + (Y2 - Y1) * t
            u, v = self.world_to_pixel(X, Y)
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            pts.append((int(round(u)), int(round(v))))
        return np.array(pts, dtype=np.int32) if pts else np.zeros((0, 2), dtype=np.int32)

    def running_avg_speed(
        self,
        samples: Sequence[tuple[float, float, float]],
        min_samples: int = 5,
        max_plausible_fps: float | None = None,
        max_arc_displacement_ratio: float | None = None,
        max_exit_descent: float | None = None,
        max_plausible_mph: float | None = None,
    ) -> tuple[float, list[float], int]:
        """Cumulative-distance / cumulative-time speed from the first sample.

        For each frame i ≥ min_samples-1, returns the running average speed
        from sample 0 to sample i: `mph_i = (cum_arc_length / (t_i - t_0))`.
        The final value is the headline speed.

        Robust to PTS-burst stutter: a brief cluster of frames sharing nearly
        identical timestamps doesn't perturb the totals — once timestamps
        recover, `cum_dist / cum_dt` returns to the true speed. (Contrast with
        a per-frame v_inst chart, where the same burst produces 100-600 mph
        spikes.)

        That robustness only holds when the time base and the focus track are
        sound. Two failure modes corrupt the headline itself, and the optional
        guards below reject them (returning NaN — "speed unknown") rather than
        emitting a fabricated over-speed:

        * `max_plausible_fps` — if the trajectory's frames imply a frame rate
          above this, the presentation timestamps are bunched (variable-frame-
          rate source, or a decode burst), so the whole `t_last - t_0` span is
          compressed and `cum_dist / span` inflates. Not a brief mid-trajectory
          cluster — the entire denominator is wrong, which the running average
          cannot recover from.
        * `max_arc_displacement_ratio` — if cumulative arc length exceeds the
          straight-line displacement by more than this factor, the track
          doubled back (e.g. the focus box merged with an oncoming vehicle),
          inflating distance. A clean crossing stays within ~1.03.
        * `max_exit_descent` — catches a *partial* early burst that the global
          fps guard misses: a cluster of bunched-PTS frames at track
          acquisition followed by a normally-timed tail. The early frames
          contribute distance with almost no time, so the running average
          starts high and is still descending at grid exit — it never
          converged. A real vehicle can't lose this fraction of its speed in
          the final inter-frame interval, so a still-descending exit means the
          headline is a timing artifact. Rejected only when the tail is
          consistently descending (guards against a single noisy final frame).

        `max_plausible_mph` gates all three checks: a suspicious-shape pass is
        rejected only when its headline also exceeds this ceiling. Most
        acquisition bursts and track wobbles leave a *plausible* speed (the
        residential traffic here runs well under it) and are kept; only a
        reading that is both suspicious and implausibly fast — a phantom
        over-speed — is discarded. A clean high reading (a real speeder, no
        suspicious shape) is never rejected. When None, the checks reject on
        suspicious shape alone.

        Returns:
            (final_mph, per_frame_running, n_samples).
            `final_mph` is NaN if fewer than `min_samples` samples, the
            cumulative dt never becomes positive, or a guard rejects the
            trajectory. `per_frame_running[i]` is NaN until enough samples have
            accumulated, and is left intact even when a guard rejects the
            headline (the diagnostic chart still renders the raw series).
        """
        n = len(samples)
        per_frame: list[float] = [float("nan")] * n
        if n < max(2, min_samples):
            return float("nan"), per_frame, n
        projected: list[tuple[float, float, float]] = []
        for ts, u, v in samples:
            X, Y = self.project(u, v)
            projected.append((float(ts), float(X), float(Y)))
        t0 = projected[0][0]
        cum_dist = 0.0
        last_valid = float("nan")
        for i in range(1, n):
            ti, Xi, Yi = projected[i]
            _tp, Xp, Yp = projected[i - 1]
            cum_dist += ((Xi - Xp) ** 2 + (Yi - Yp) ** 2) ** 0.5
            cum_dt = ti - t0
            if (i + 1) >= min_samples and cum_dt > 0:
                mph = (cum_dist / cum_dt) * MPH_PER_MPS
                per_frame[i] = mph
                last_valid = mph

        # Trustworthiness guards on the headline (per_frame is left intact).
        span = projected[-1][0] - t0
        if span <= 0:
            return float("nan"), per_frame, n

        # The three checks below detect *suspicious shape* (corrupted timing or
        # a track jump). On their own they also fire on plenty of plausible
        # passes — e.g. a brief acquisition burst leaves a 25 mph headline
        # mildly non-converged. A high frame rate or a doubled-back path only
        # produces a *wrong* speed when the result is also implausibly high, so
        # `max_plausible_mph` gates them: a suspicious pass is rejected only
        # when its headline exceeds the ceiling. A clean high reading (real
        # speeder) is kept; a plausible reading is kept even if its shape is
        # suspicious. When `max_plausible_mph` is None the checks reject on
        # suspicion alone (legacy behavior).
        suspicious = False
        if max_plausible_fps is not None and (n - 1) / span > max_plausible_fps:
            suspicious = True
        if max_arc_displacement_ratio is not None:
            net = (
                (projected[-1][1] - projected[0][1]) ** 2
                + (projected[-1][2] - projected[0][2]) ** 2
            ) ** 0.5
            if net > 0 and cum_dist / net > max_arc_displacement_ratio:
                suspicious = True
        if max_exit_descent is not None:
            vr = [v for v in per_frame if v == v]  # drop NaNs
            # Need a settled tail to distinguish "still paying off an early
            # burst" (high → descending) from a clean pass converging up from
            # below; require a consistently descending final pair.
            if (
                len(vr) >= 5
                and vr[-1] > 0
                and vr[-3] > vr[-2] > vr[-1]
                and (vr[-2] - vr[-1]) / vr[-1] > max_exit_descent
            ):
                suspicious = True
        if suspicious and (
            max_plausible_mph is None or last_valid > max_plausible_mph
        ):
            return float("nan"), per_frame, n
        return last_valid, per_frame, n

