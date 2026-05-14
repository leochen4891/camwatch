"""Interactive homography calibration + visualization.

Combined verifier and refit tool. Loads an existing calibration; drag any
visible anchor dot onto its true position in the image and the tool refits
the homography automatically on drag release.

Workflow when adjusting after the camera moves or new dots are painted:
  1. uv run python scripts/inspect_homography.py
  2. (optional) press 'r' to fetch a fresh frame from RTSP
  3. Toggle which anchors are draggable (rect corners vs all 17 dots)
  4. Drag anchors that are off, release, watch the grid refit
  5. Press 's' to save (marked_points.yaml + refit homography.yaml)

For a fresh first-time calibration with no existing homography, still start
with scripts/mark_points.py to click the initial 17 dots, then run this.

Keys:
  1     toggle red outer rectangle. When on: corner anchors (1, 14, 15, 17)
        are draggable even if the dots layer is off.
  2     toggle blue 5 ft grid
  3     toggle black 1 ft sub-grid
  4     toggle yellow X/Y axes (through origin)
  5     toggle anchor dots (when on: all 17 dots are draggable)
  c     recompute (refit) the homography from current anchor positions.
        Run this after dragging dots — the grid stays stale until you
        press 'c'.
  r     refresh: fetch a fresh frame from the RTSP main stream
        (overwrites events/calibration_main_frame.jpg). Anchor positions
        are kept as-is — drag them onto the new frame's painted dots.
  +/-   thicker / thinner lines
  s     save: marked_points.yaml + refit homography.yaml
  w     save current view as a PNG to /tmp/homography_inspect.jpg
  q/Esc quit

Drag any anchor dot to reposition it. The drawn grid (red rect, 5/1 ft,
axes) stays stale until you press 'c' to refit and recompute.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from fit_distortion_from_scene import fit_from_anchors
from mark_points import grab_mainstream_frame, build_main_url

DEFAULT_FRAME = REPO / "events" / "calibration_main_frame.jpg"
DEFAULT_HOMOG = REPO / "config" / "homography.yaml"
DEFAULT_MARKED = REPO / "config" / "marked_points.yaml"
SAVE_PATH = Path("/tmp/homography_inspect.jpg")

FT_TO_M = 0.3048

# BGR
RED = (0, 0, 220)
BLUE = (255, 80, 0)
BLACK = (0, 0, 0)
YELLOW = (0, 220, 255)
DOT = (60, 60, 240)
DOT_DRAG = (0, 255, 0)
DOT_CORNER = (0, 160, 255)


def project_world_to_pixel(Hinv, K, D, X, Y) -> tuple[float, float] | None:
    p = Hinv @ np.array([X, Y, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    uu = p[0] / p[2]
    vv = p[1] / p[2]
    if not (math.isfinite(uu) and math.isfinite(vv)):
        return None
    if K is None or D is None:
        return float(uu), float(vv)
    xc = (uu - K[0, 2]) / K[0, 0]
    yc = (vv - K[1, 2]) / K[1, 1]
    pts3d = np.array([[[xc, yc, 1.0]]], dtype=np.float64)
    out, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K, D)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def polyline_for_segment(Hinv, K, D, X1, Y1, X2, Y2, n=64) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, n)
    pts: list[tuple[int, int]] = []
    for t in ts:
        X = X1 + (X2 - X1) * t
        Y = Y1 + (Y2 - Y1) * t
        p = project_world_to_pixel(Hinv, K, D, float(X), float(Y))
        if p is None:
            continue
        u, v = p
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        pts.append((int(round(u)), int(round(v))))
    return np.array(pts, dtype=np.int32) if pts else np.zeros((0, 2), dtype=np.int32)


CORNER_INDICES = (1, 14, 15, 17)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    ap.add_argument("--homog", type=Path, default=DEFAULT_HOMOG)
    ap.add_argument("--marked", type=Path, default=DEFAULT_MARKED)
    ap.add_argument("--max-width", type=int, default=1800,
                    help="Display scale: window resized so width<=this px.")
    args = ap.parse_args()

    img_holder: list[np.ndarray] = [cv2.imread(str(args.frame))]
    if img_holder[0] is None:
        raise SystemExit(f"could not read {args.frame}")
    H_img, W_img = img_holder[0].shape[:2]

    marked = yaml.safe_load(args.marked.read_text())["marked_points"]
    frame_w, frame_h = [int(v) for v in marked["frame_size"]]
    anchors: dict[int, list[float]] = {
        int(p["idx"]): [float(p["pixel"][0]), float(p["pixel"][1])]
        for p in marked["points"]
    }
    if set(anchors.keys()) != set(range(1, 18)):
        raise SystemExit(f"marked_points.yaml must have indices 1..17, got {sorted(anchors.keys())}")

    data = yaml.safe_load(args.homog.read_text())["homography"]
    state: dict = {
        "H": np.array(data["H"], dtype=np.float64),
        "K": np.array(data["K"], dtype=np.float64) if "K" in data else None,
        "D": np.array(data["D"], dtype=np.float64).reshape(-1) if "D" in data else None,
        "mean_err_m": float(data.get("mean_reprojection_error_m", 0.0)),
        "max_err_m": float(data.get("max_reprojection_error_m", 0.0)),
    }
    state["Hinv"] = np.linalg.inv(state["H"])

    x_min_ft, x_max_ft = -30, 0
    y_min_ft, y_max_ft = -40, 25

    polylines: dict[str, list[np.ndarray]] = {"1ft": [], "5ft": [], "outer": [], "axes": []}

    def recompute_grid_polylines() -> None:
        Hinv = state["Hinv"]
        K, D = state["K"], state["D"]
        l1: list[np.ndarray] = []
        for x_ft in range(x_min_ft, x_max_ft + 1):
            X = x_ft * FT_TO_M
            l1.append(polyline_for_segment(Hinv, K, D, X, y_min_ft * FT_TO_M, X, y_max_ft * FT_TO_M))
        for y_ft in range(y_min_ft, y_max_ft + 1):
            Y = y_ft * FT_TO_M
            l1.append(polyline_for_segment(Hinv, K, D, x_min_ft * FT_TO_M, Y, x_max_ft * FT_TO_M, Y))
        polylines["1ft"] = l1

        l5: list[np.ndarray] = []
        for x_ft in range(x_min_ft, x_max_ft + 1, 5):
            X = x_ft * FT_TO_M
            l5.append(polyline_for_segment(Hinv, K, D, X, y_min_ft * FT_TO_M, X, y_max_ft * FT_TO_M))
        for y_ft in range(y_min_ft, y_max_ft + 1, 5):
            Y = y_ft * FT_TO_M
            l5.append(polyline_for_segment(Hinv, K, D, x_min_ft * FT_TO_M, Y, x_max_ft * FT_TO_M, Y))
        polylines["5ft"] = l5

        outer: list[np.ndarray] = []
        for x1, y1, x2, y2 in [
            (x_min_ft, y_min_ft, x_max_ft, y_min_ft),
            (x_max_ft, y_min_ft, x_max_ft, y_max_ft),
            (x_max_ft, y_max_ft, x_min_ft, y_max_ft),
            (x_min_ft, y_max_ft, x_min_ft, y_min_ft),
        ]:
            outer.append(polyline_for_segment(
                Hinv, K, D, x1 * FT_TO_M, y1 * FT_TO_M, x2 * FT_TO_M, y2 * FT_TO_M))
        polylines["outer"] = outer

        ax: list[np.ndarray] = []
        ax.append(polyline_for_segment(Hinv, K, D, x_min_ft * FT_TO_M, 0.0, x_max_ft * FT_TO_M, 0.0))
        ax.append(polyline_for_segment(Hinv, K, D, 0.0, y_min_ft * FT_TO_M, 0.0, y_max_ft * FT_TO_M))
        polylines["axes"] = ax

    def refit() -> bool:
        pixels_dict = {i: (float(anchors[i][0]), float(anchors[i][1])) for i in range(1, 18)}
        try:
            h, k, d, _errs, mean, mx = fit_from_anchors(pixels_dict, frame_w, frame_h)
        except Exception as e:  # noqa: BLE001
            print(f"refit failed: {e}")
            return False
        state["H"] = h
        state["Hinv"] = np.linalg.inv(h)
        state["K"] = k
        state["D"] = d
        state["mean_err_m"] = mean
        state["max_err_m"] = mx
        recompute_grid_polylines()
        print(f"refit: mean={mean*100:.2f} cm, max={mx*100:.2f} cm")
        return True

    print(f"frame: {args.frame} ({W_img}x{H_img})")
    print(f"loaded H/K/D: mean={state['mean_err_m']*100:.1f} cm, max={state['max_err_m']*100:.1f} cm")
    print("precomputing polylines …")
    recompute_grid_polylines()
    print("ready.")

    show = {"rect": True, "g5": True, "g1": False, "axes": False, "dots": True}
    thick_off = 0
    drag: list[int | None] = [None]  # anchor idx being dragged, or None
    dirty = [False]  # True if anchors moved since the last refit

    win_scale = min(1.0, args.max_width / W_img)
    win_w = int(W_img * win_scale)
    win_h = int(H_img * win_scale)
    cv2.namedWindow("inspect homography", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("inspect homography", win_w, win_h)

    def draggable_anchor_indices() -> set[int]:
        s: set[int] = set()
        if show["rect"]:
            s.update(CORNER_INDICES)
        if show["dots"]:
            s.update(range(1, 18))
        return s

    def on_mouse(event: int, x: int, y: int, flags: int, _: object) -> None:
        u_full = x / win_scale
        v_full = y / win_scale
        grab_thresh_full = (28.0 / win_scale) ** 2

        if event == cv2.EVENT_LBUTTONDOWN:
            best_idx = None
            best_d = grab_thresh_full
            for idx in draggable_anchor_indices():
                u, v = anchors[idx]
                d = (u - u_full) ** 2 + (v - v_full) ** 2
                if d < best_d:
                    best_d = d
                    best_idx = idx
            drag[0] = best_idx
        elif event == cv2.EVENT_MOUSEMOVE and drag[0] is not None:
            anchors[drag[0]][0] = u_full
            anchors[drag[0]][1] = v_full
            dirty[0] = True
        elif event == cv2.EVENT_LBUTTONUP:
            drag[0] = None

    cv2.setMouseCallback("inspect homography", on_mouse)

    def render(refitting: bool = False, fetching: bool = False) -> np.ndarray:
        out = img_holder[0].copy()
        t1 = max(1, 1 + thick_off)
        t5 = max(1, 1 + thick_off)
        tr = max(1, 2 + thick_off)
        ta = max(1, 1 + thick_off)
        if show["g1"]:
            for pl in polylines["1ft"]:
                if len(pl) >= 2:
                    cv2.polylines(out, [pl], False, BLACK, t1, cv2.LINE_AA)
        if show["g5"]:
            for pl in polylines["5ft"]:
                if len(pl) >= 2:
                    cv2.polylines(out, [pl], False, BLUE, t5, cv2.LINE_AA)
        if show["rect"]:
            for pl in polylines["outer"]:
                if len(pl) >= 2:
                    cv2.polylines(out, [pl], False, RED, tr, cv2.LINE_AA)
        if show["axes"]:
            for pl in polylines["axes"]:
                if len(pl) >= 2:
                    cv2.polylines(out, [pl], False, YELLOW, ta, cv2.LINE_AA)

        draggable = draggable_anchor_indices()
        for idx in sorted(draggable):
            u, v = int(round(anchors[idx][0])), int(round(anchors[idx][1]))
            if drag[0] == idx:
                color = DOT_DRAG
            elif idx in CORNER_INDICES:
                color = DOT_CORNER
            else:
                color = DOT
            cv2.circle(out, (u, v), 7, color, -1, cv2.LINE_AA)
            cv2.circle(out, (u, v), 8, (255, 255, 255), 1, cv2.LINE_AA)
            # Labels outside the grid: east-curb dots (idx 1-14, near the
            # image bottom) get labels below; west-curb dots (15-17) above.
            if 1 <= idx <= 14:
                lx, ly = u - 6, v + 24
            else:
                lx, ly = u - 6, v - 14
            cv2.putText(out, str(idx), (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, str(idx), (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.rectangle(out, (0, 0), (W_img, 30), (0, 0, 0), -1)
        dirty_tag = "  [DIRTY: press c to refit]" if dirty[0] else ""
        status = (
            f"[1]rect [2]5ft [3]1ft [4]axes [5]dots  "
            f"[c]refit [r]refresh [s]save [w]png [q]quit   "
            f"err: mean={state['mean_err_m']*100:.1f}cm max={state['max_err_m']*100:.1f}cm"
            f"{dirty_tag}"
        )
        if refitting:
            status = "REFITTING …  " + status
        elif fetching:
            status = "FETCHING FRAME …  " + status
        cv2.putText(out, status, (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    while True:
        full = render()
        disp = cv2.resize(full, (win_w, win_h)) if win_scale != 1.0 else full
        cv2.imshow("inspect homography", disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("1"):
            show["rect"] = not show["rect"]
        elif k == ord("2"):
            show["g5"] = not show["g5"]
        elif k == ord("3"):
            show["g1"] = not show["g1"]
        elif k == ord("4"):
            show["axes"] = not show["axes"]
        elif k == ord("5"):
            show["dots"] = not show["dots"]
        elif k == ord("c"):
            full = render(refitting=True)
            disp = cv2.resize(full, (win_w, win_h)) if win_scale != 1.0 else full
            cv2.imshow("inspect homography", disp)
            cv2.waitKey(1)
            if refit():
                dirty[0] = False
        elif k == ord("s"):
            marked_payload = {
                "marked_points": {
                    "source_image": marked.get("source_image", "events/calibration_main_frame.jpg"),
                    "frame_size": [int(frame_w), int(frame_h)],
                    "points": [
                        {"idx": i, "pixel": [int(round(anchors[i][0])), int(round(anchors[i][1]))]}
                        for i in range(1, 18)
                    ],
                }
            }
            args.marked.write_text(yaml.safe_dump(marked_payload, sort_keys=False))
            existing = yaml.safe_load(args.homog.read_text())["homography"]
            existing["H"] = state["H"].tolist()
            if state["K"] is not None:
                existing["K"] = state["K"].tolist()
            if state["D"] is not None:
                existing["D"] = state["D"].tolist()
            existing["mean_reprojection_error_m"] = state["mean_err_m"]
            existing["max_reprojection_error_m"] = state["max_err_m"]
            existing["pixel_pts"] = [
                {"idx": i, "u": float(anchors[i][0]), "v": float(anchors[i][1])}
                for i in range(1, 18)
            ]
            args.homog.write_text(yaml.safe_dump({"homography": existing}, sort_keys=False))
            print(f"saved: {args.marked.name} + {args.homog.name} "
                  f"(mean={state['mean_err_m']*100:.2f}cm max={state['max_err_m']*100:.2f}cm)")
        elif k == ord("w"):
            cv2.imwrite(str(SAVE_PATH), full, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"saved view → {SAVE_PATH}")
        elif k == ord("r"):
            full = render(fetching=True)
            disp = cv2.resize(full, (win_w, win_h)) if win_scale != 1.0 else full
            cv2.imshow("inspect homography", disp)
            cv2.waitKey(1)
            try:
                url = build_main_url()
                new_img = grab_mainstream_frame(url)
            except Exception as e:  # noqa: BLE001
                print(f"frame grab failed: {e}")
                continue
            new_h, new_w = new_img.shape[:2]
            if (new_w, new_h) != (W_img, H_img):
                print(f"warning: new frame is {new_w}x{new_h}, expected "
                      f"{W_img}x{H_img}. Skipping update.")
                continue
            cv2.imwrite(str(args.frame), new_img)
            img_holder[0] = new_img
            print(f"fetched fresh frame → {args.frame}")
        elif k in (ord("+"), ord("=")):
            thick_off += 1
        elif k in (ord("-"), ord("_")):
            thick_off -= 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
