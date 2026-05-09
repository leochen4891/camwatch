"""Calibration click tool: 4 rectangle corners + 9 east-curb inner points
with live homography refit and toggleable grid overlay.

Click sequence (13 total):
   1: NE corner  →  point 1   (east curb, north end, Y = +25 ft)
   2: SE corner  →  point 11  (east curb, south end, Y = -25 ft)
   3: NW corner  →  point 12  (west curb, north end, Y = +25 ft, X = -30 ft)
   4: SW corner  →  point 13  (west curb, south end, Y = -25 ft, X = -30 ft)
   5: inner #1   →  point 2   (east curb, Y = +20 ft)
   6: inner #2   →  point 3   (east curb, Y = +15 ft)
   7: inner #3   →  point 4   (east curb, Y = +10 ft)
   8: inner #4   →  point 5   (east curb, Y =  +5 ft)
   9: inner #5   →  point 6   (east curb, Y =   0 ft)
  10: inner #6   →  point 7   (east curb, Y =  -5 ft)
  11: inner #7   →  point 8   (east curb, Y = -10 ft)
  12: inner #8   →  point 9   (east curb, Y = -15 ft)
  13: inner #9   →  point 10  (east curb, Y = -20 ft)

Result is saved with the conventional 1-13 indexing so
build_homography_from_marks.py works without modification.

Live behavior:
  - After click #4, the homography is fit from the 4 corners (exactly determined).
  - After each subsequent click, H is refit using all clicks so far (over-determined
    least-squares).
  - Press `g` at any time to toggle the cyan 5 ft Y-grid + yellow axes
    overlay, computed from the current H. Lets you visually verify the next
    click before placing it.

Controls:
  left-click   add the next point in the prescribed sequence
  u            undo last click (refits H)
  r            reset all clicks
  g            toggle grid overlay
  s / Enter    save and exit (only when all 13 clicked)
  q / Esc      abort

Output:
  config/marked_points.yaml         13 points indexed 1-13
  config/marked_points_overlay.jpg  frame with all points drawn
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
OUT_YAML = REPO / "config" / "marked_points.yaml"
OUT_OVERLAY = REPO / "config" / "marked_points_overlay.jpg"
SNAPSHOT = REPO / "events" / "calibration_main_frame.jpg"

FT_TO_M = 0.3048
ROAD_WIDTH_FT = 30.0
ROAD_WIDTH_M = ROAD_WIDTH_FT * FT_TO_M
MAIN_TO_SUB = 3.2

# (label, conventional point index, world X in m, world Y in m). Click order
# matches the index of this list.
CLICK_SEQUENCE: list[tuple[str, int, float, float]] = [
    ("NE corner",         1,  0.0,            +25 * FT_TO_M),
    ("SE corner",         11, 0.0,            -25 * FT_TO_M),
    ("NW corner",         12, -ROAD_WIDTH_M,  +25 * FT_TO_M),
    ("SW corner",         13, -ROAD_WIDTH_M,  -25 * FT_TO_M),
    ("inner #1 (north)",  2,  0.0,            +20 * FT_TO_M),
    ("inner #2",          3,  0.0,            +15 * FT_TO_M),
    ("inner #3",          4,  0.0,            +10 * FT_TO_M),
    ("inner #4",          5,  0.0,             +5 * FT_TO_M),
    ("inner #5 (mid pt6)", 6, 0.0,              0.0),
    ("inner #6",          7,  0.0,             -5 * FT_TO_M),
    ("inner #7",          8,  0.0,            -10 * FT_TO_M),
    ("inner #8",          9,  0.0,            -15 * FT_TO_M),
    ("inner #9 (south)",  10, 0.0,            -20 * FT_TO_M),
]


def grab_mainstream_frame(url: str) -> np.ndarray:
    import av
    import av.codec.hwaccel as hw

    print("Opening main stream …")
    hwa = hw.HWAccel(device_type="videotoolbox", allow_software_fallback=True)
    container = av.open(url, options={"rtsp_transport": "tcp"}, hwaccel=hwa, timeout=(15.0, 15.0))
    try:
        vstream = container.streams.video[0]
        saw = False
        for packet in container.demux(vstream):
            if not saw:
                if not packet.is_keyframe:
                    continue
                saw = True
            for frame in packet.decode():
                return frame.to_ndarray(format="bgr24")
    finally:
        container.close()
    raise RuntimeError("no frame")


def build_main_url() -> str:
    load_dotenv(REPO / ".env")
    user = quote(os.environ["REOLINK_USER"], safe="")
    pw = quote(os.environ["REOLINK_PASS"], safe="")
    cfg = yaml.safe_load((REPO / "config/config.yaml").read_text())
    cam = cfg["camera"]
    path_main = cam.get("path_thumb") or cam["path"].replace("_sub", "_main")
    return f"rtsp://{user}:{pw}@{cam['host']}:{cam['port']}{path_main}"


def fit_homography(clicks_main: list[tuple[int, int]]) -> np.ndarray | None:
    """Fit H from the first len(clicks_main) entries of CLICK_SEQUENCE."""
    n = len(clicks_main)
    if n < 4:
        return None
    src = np.array(
        [[u / MAIN_TO_SUB, v / MAIN_TO_SUB] for (u, v) in clicks_main],
        dtype=np.float32,
    )
    dst = np.array(
        [[CLICK_SEQUENCE[i][2], CLICK_SEQUENCE[i][3]] for i in range(n)],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src, dst, method=0)
    return H


def _frac_along(NE: tuple[int, int], SE: tuple[int, int], p: tuple[int, int]) -> float:
    """Fraction of p's projection along NE→SE (0 at NE, 1 at SE)."""
    vx = SE[0] - NE[0]
    vy = SE[1] - NE[1]
    L2 = vx * vx + vy * vy
    if L2 == 0:
        return 0.0
    return ((p[0] - NE[0]) * vx + (p[1] - NE[1]) * vy) / L2


def draw_visual_grid(img: np.ndarray, clicks_main: list[tuple[int, int]]) -> None:
    """Draw the visual calibration grid in pure pixel-space:
       - yellow rectangle from the 4 corners (NE, SE, NW, SW)
       - blue 5ft cross-section lines, one per inner east-curb click,
         each spanning from the click position (east) to the proportionally
         matching point on the NW→SW line (west)
       - cyan 1ft sub-grid: 4 evenly-spaced sub-lines between every pair
         of consecutive horizontal lines (north-edge, blue lines, south-edge)
    Requires at least 4 clicks (the corners). Inner east-curb lines are
    drawn only when those points have been placed.
    """
    if len(clicks_main) < 4:
        return

    NE = clicks_main[0]
    SE = clicks_main[1]
    NW = clicks_main[2]
    SW = clicks_main[3]

    yellow = (0, 220, 255)
    blue = (255, 50, 0)
    cyan = (255, 200, 0)

    # Yellow rectangle: 4 sides
    cv2.line(img, NE, SE, yellow, 1, cv2.LINE_AA)  # east curb
    cv2.line(img, NW, SW, yellow, 1, cv2.LINE_AA)  # west curb
    cv2.line(img, NE, NW, yellow, 1, cv2.LINE_AA)  # north end
    cv2.line(img, SE, SW, yellow, 1, cv2.LINE_AA)  # south end

    # 9 inner east-curb clicks live at clicks_main[4..12]. Each click defines
    # a fraction along NE→SE.
    #
    # When all 9 are placed, apply a 3-tap smoothing kernel (0.25/0.5/0.25)
    # so each blue line's east-curb position is the weighted average of its
    # own dot and the two immediate neighbors (NE corner serves as the left
    # neighbor of point 2; SE corner serves as the right neighbor of point
    # 10). This absorbs per-click noise. The bias from using the corners as
    # neighbors is the cost of smoothing — small in practice, and the user's
    # downstream homography fit also includes the corner anchors directly.
    inner = clicks_main[4:13]
    raw_fracs = [_frac_along(NE, SE, p) for p in inner]

    if len(inner) == 9:
        smoothed_fracs: list[float] = []
        for i in range(9):
            f_self = raw_fracs[i]
            f_left = raw_fracs[i - 1] if i > 0 else 0.0   # NE corner
            f_right = raw_fracs[i + 1] if i < 8 else 1.0  # SE corner
            smoothed_fracs.append(0.25 * f_left + 0.5 * f_self + 0.25 * f_right)
    else:
        smoothed_fracs = raw_fracs

    blue_lines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for f in smoothed_fracs:
        east_pt = (
            int(round(NE[0] + f * (SE[0] - NE[0]))),
            int(round(NE[1] + f * (SE[1] - NE[1]))),
        )
        west_pt = (
            int(round(NW[0] + f * (SW[0] - NW[0]))),
            int(round(NW[1] + f * (SW[1] - NW[1]))),
        )
        cv2.line(img, east_pt, west_pt, blue, 1, cv2.LINE_AA)
        blue_lines.append((east_pt, west_pt))

    # All horizontal lines, ordered north→south by their fraction along NE→SE.
    h_with_frac: list[tuple[float, tuple[tuple[int, int], tuple[int, int]]]] = [
        (0.0, (NE, NW))
    ]
    for f, line in zip(smoothed_fracs, blue_lines):
        h_with_frac.append((f, line))
    h_with_frac.append((1.0, (SE, SW)))
    h_with_frac.sort(key=lambda x: x[0])
    horizontals = [line for _f, line in h_with_frac]

    # Cyan 1ft sub-grid: 4 lines per gap (between consecutive horizontal lines).
    for i in range(len(horizontals) - 1):
        e_top, w_top = horizontals[i]
        e_bot, w_bot = horizontals[i + 1]
        for k in range(1, 5):
            f = k / 5.0
            sub_e = (
                int(round(e_top[0] + f * (e_bot[0] - e_top[0]))),
                int(round(e_top[1] + f * (e_bot[1] - e_top[1]))),
            )
            sub_w = (
                int(round(w_top[0] + f * (w_bot[0] - w_top[0]))),
                int(round(w_top[1] + f * (w_bot[1] - w_top[1]))),
            )
            cv2.line(img, sub_e, sub_w, cyan, 1, cv2.LINE_AA)


def _project_onto_line(
    p1: tuple[int, int], p2: tuple[int, int], p: tuple[int, int],
) -> tuple[int, int]:
    """Project p onto the line p1→p2 in image space, clamped to the segment.
    Returns (round_x, round_y).
    """
    x1, y1 = p1
    x2, y2 = p2
    vx, vy = x2 - x1, y2 - y1
    L2 = vx * vx + vy * vy
    if L2 == 0:
        return p1
    t = ((p[0] - x1) * vx + (p[1] - y1) * vy) / L2
    t = max(0.0, min(1.0, t))
    return int(round(x1 + t * vx)), int(round(y1 + t * vy))


def _maybe_snap(
    ox: int, oy: int, click_idx: int, clicks_main: list[tuple[int, int]],
) -> tuple[int, int]:
    """For inner east-curb points (CLICK_SEQUENCE indices 4..12 = points 2-10),
    project the proposed (ox, oy) onto the NE→SE segment. NE = clicks_main[0],
    SE = clicks_main[1]. If those aren't placed yet, no snap.
    """
    if 4 <= click_idx <= 12 and len(clicks_main) >= 2:
        return _project_onto_line(clicks_main[0], clicks_main[1], (ox, oy))
    return (ox, oy)


def click_loop(img: np.ndarray) -> list[tuple[int, int]]:
    H_img, W_img = img.shape[:2]
    clicks_main: list[tuple[int, int]] = []
    H_current: np.ndarray | None = None
    show_grid = True
    dragging_idx: int | None = None
    HIT_RADIUS = 12  # pixels in original image space — generous enough to grab on small displays

    max_w = 1600
    scale = min(1.0, max_w / W_img)
    win_w = int(W_img * scale)
    win_h = int(H_img * scale)

    def find_hit(x_orig: int, y_orig: int) -> int | None:
        """Return the index of the closest point within HIT_RADIUS, else None."""
        best = None
        best_d2 = HIT_RADIUS * HIT_RADIUS
        for i, (u, v) in enumerate(clicks_main):
            d2 = (u - x_orig) ** 2 + (v - y_orig) ** 2
            if d2 <= best_d2:
                best = i
                best_d2 = d2
        return best

    def render() -> np.ndarray:
        out = img.copy()
        if show_grid:
            draw_visual_grid(out, clicks_main)
        for i, (u, v) in enumerate(clicks_main):
            _label, idx, _, _ = CLICK_SEQUENCE[i]
            color = (0, 255, 255) if i == dragging_idx else (0, 0, 255)
            cv2.circle(out, (u, v), 8, color, -1)
            cv2.circle(out, (u, v), 9, (0, 0, 0), 1)
            # East-curb dots (points 1, 2-10, 11 = CLICK_SEQUENCE indices 0, 4..12, 1)
            # have labels placed BELOW the dot so the dot itself sits cleanly
            # on the curb edge in the image (labels above would visually pull
            # the eye northward of the dot).
            is_west = i in (2, 3)  # NW or SW
            label_pos = (u + 10, v - 10) if is_west else (u + 10, v + 22)
            cv2.putText(
                out, str(idx), label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )
        cv2.rectangle(out, (0, 0), (W_img, 38), (0, 0, 0), -1)
        if len(clicks_main) < len(CLICK_SEQUENCE):
            label, idx, X, Y = CLICK_SEQUENCE[len(clicks_main)]
            grid_state = "ON" if show_grid else "OFF"
            status = (
                f"Click {len(clicks_main)+1}/13: {label}  →  point {idx}  "
                f"(Y = {Y/FT_TO_M:+.0f} ft)   "
                f"drag any placed point to refine   "
                f"grid:{grid_state} (g)  u=undo  r=reset  q=abort"
            )
        else:
            status = (
                f"All 13 clicked. Drag to refine, s/Enter to save, "
                f"u=undo  r=reset  g=toggle grid"
            )
        cv2.putText(
            out, status, (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return out

    def on_mouse(event: int, x: int, y: int, flags: int, _: object) -> None:
        nonlocal H_current, dragging_idx
        ox = int(round(x / scale))
        oy = int(round(y / scale))

        if event == cv2.EVENT_LBUTTONDOWN:
            hit = find_hit(ox, oy)
            if hit is not None:
                dragging_idx = hit
                idx = CLICK_SEQUENCE[hit][1]
                print(f"  grab: dragging point {idx}")
                return
            # Not on an existing point → place next in sequence (if any left)
            if len(clicks_main) >= len(CLICK_SEQUENCE):
                return
            new_idx = len(clicks_main)
            sx, sy = _maybe_snap(ox, oy, new_idx, clicks_main)
            clicks_main.append((sx, sy))
            label, idx, X, Y = CLICK_SEQUENCE[new_idx]
            extra = " [snapped to NE→SE]" if (sx, sy) != (ox, oy) else ""
            print(
                f"  click {new_idx + 1}: {label} → point {idx}, pixel ({sx}, {sy}){extra}"
            )
            H_new = fit_homography(clicks_main)
            if H_new is not None:
                H_current = H_new

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_idx is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
                sx, sy = _maybe_snap(ox, oy, dragging_idx, clicks_main)
                clicks_main[dragging_idx] = (sx, sy)
                H_new = fit_homography(clicks_main)
                if H_new is not None:
                    H_current = H_new

        elif event == cv2.EVENT_LBUTTONUP:
            if dragging_idx is not None:
                final = clicks_main[dragging_idx]
                idx = CLICK_SEQUENCE[dragging_idx][1]
                print(f"  drop: point {idx} → ({final[0]}, {final[1]})")
                dragging_idx = None

    cv2.namedWindow("rectangle calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("rectangle calibration", win_w, win_h)
    cv2.setMouseCallback("rectangle calibration", on_mouse)

    print(f"\nFrame size: {W_img}x{H_img}  display scale: {scale:.2f}")
    print("Click sequence:")
    for i, (label, idx, X, Y) in enumerate(CLICK_SEQUENCE):
        print(f"  {i+1:>2}: {label:<20} → point {idx:<2}  Y={Y/FT_TO_M:+.0f}ft")
    print()

    while True:
        disp = render()
        if scale != 1.0:
            disp = cv2.resize(disp, (win_w, win_h))
        cv2.imshow("rectangle calibration", disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            cv2.destroyAllWindows()
            sys.exit("aborted (no file written)")
        if k == ord("r"):
            clicks_main.clear()
            H_current = None
            print("  (reset)")
        if k == ord("u"):
            if clicks_main:
                p = clicks_main.pop()
                H_current = fit_homography(clicks_main)
                print(f"  (undo: removed click {len(clicks_main)+1}: {p})")
        if k == ord("g"):
            show_grid = not show_grid
            print(f"  grid overlay: {'ON' if show_grid else 'OFF'}")
        if k == ord("s") or k in (13, 10):
            if len(clicks_main) < len(CLICK_SEQUENCE):
                print(
                    f"  need {len(CLICK_SEQUENCE) - len(clicks_main)} more clicks before saving"
                )
                continue
            break

    cv2.destroyAllWindows()
    return clicks_main


def save(img: np.ndarray, clicks_main: list[tuple[int, int]]) -> None:
    H_img, W_img = img.shape[:2]
    points = []
    for i, (u, v) in enumerate(clicks_main):
        _label, idx, _, _ = CLICK_SEQUENCE[i]
        points.append({"idx": idx, "pixel": [int(u), int(v)]})
    points.sort(key=lambda p: p["idx"])

    payload = {
        "marked_points": {
            "source_image": str(SNAPSHOT.relative_to(REPO)),
            "frame_size": [int(W_img), int(H_img)],
            "points": points,
        }
    }
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUT_YAML.write_text(yaml.safe_dump(payload, sort_keys=False))

    overlay = img.copy()
    for i, (u, v) in enumerate(clicks_main):
        _label, idx, _, _ = CLICK_SEQUENCE[i]
        cv2.circle(overlay, (u, v), 12, (0, 255, 0), -1)
        cv2.circle(overlay, (u, v), 13, (0, 0, 0), 2)
        cv2.putText(
            overlay, str(idx), (u + 14, v - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, str(idx), (u + 14, v - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA,
        )
    cv2.imwrite(str(OUT_OVERLAY), overlay)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", help="Use a saved JPEG instead of grabbing fresh")
    ap.add_argument(
        "--use-cached", action="store_true",
        help="Use events/calibration_main_frame.jpg (no RTSP grab)",
    )
    args = ap.parse_args()

    if args.frame:
        img = cv2.imread(args.frame)
        if img is None:
            sys.exit(f"failed to read {args.frame}")
        print(f"Loaded frame: {args.frame} ({img.shape[1]}x{img.shape[0]})")
    elif args.use_cached:
        if not SNAPSHOT.exists():
            sys.exit(f"--use-cached: {SNAPSHOT} doesn't exist")
        img = cv2.imread(str(SNAPSHOT))
        print(f"Using cached frame: {SNAPSHOT} ({img.shape[1]}x{img.shape[0]})")
    else:
        url = build_main_url()
        img = grab_mainstream_frame(url)
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SNAPSHOT), img)
        print(f"Saved snapshot to {SNAPSHOT} ({img.shape[1]}x{img.shape[0]})")

    clicks_main = click_loop(img)
    save(img, clicks_main)
    print(f"\nWrote {OUT_YAML}")
    print(f"Wrote {OUT_OVERLAY}")


if __name__ == "__main__":
    main()
