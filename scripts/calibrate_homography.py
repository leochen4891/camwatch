"""Interactive 4-point homography calibration.

Grabs a fresh frame from the configured sub-stream, opens it in an OpenCV
window, you left-click points 1, 2, 3, 4 in order at the locations on the
road plane corresponding to the GPS-derived meter coordinates below.

Output:
  config/homography.yaml         the homography matrix + metadata
  config/homography_verify.jpg   the calibration frame with a 1-meter
                                 grid projected back through H. If
                                 calibration is good, the grid should
                                 lie flush along the road plane.

Run from the repo root:
  uv run python scripts/calibrate_homography.py
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

# Real-world meter coordinates of the four labelled points.
# Origin = camera. +X = east, +Y = north. Derived from GPS.
METER_POINTS: dict[str, tuple[float, float]] = {
    "1": (-18.873, 5.651),
    "2": (-18.816, -7.063),
    "3": (-27.970, 4.407),
    "4": (-28.083, -1.808),
}
ORIGIN_GPS = (40.8016348365999, -74.30519293037646)
LABELS = ["1", "2", "3", "4"]


def grab_substream_frame(url: str) -> np.ndarray:
    """Open the RTSP sub-stream, decode the first keyframe, return BGR ndarray."""
    import av
    import av.codec.hwaccel as hw

    hwa = hw.HWAccel(device_type="videotoolbox", allow_software_fallback=True)
    opts = {"rtsp_transport": "tcp", "fflags": "nobuffer", "flags": "low_delay"}
    container = av.open(url, options=opts, hwaccel=hwa, timeout=(10.0, 5.0))
    try:
        vstream = container.streams.video[0]
        saw_key = False
        for packet in container.demux(vstream):
            if not saw_key:
                if not packet.is_keyframe:
                    continue
                saw_key = True
            for frame in packet.decode():
                return frame.to_ndarray(format="bgr24")
    finally:
        container.close()
    raise RuntimeError("no frame decoded")


def build_rtsp_url() -> str:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    user = quote(os.environ["REOLINK_USER"], safe="")
    pw = quote(os.environ["REOLINK_PASS"], safe="")
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cam = cfg["camera"]
    return f"rtsp://{user}:{pw}@{cam['host']}:{cam['port']}{cam['path']}"


def click_four_points(img: np.ndarray) -> list[tuple[int, int]]:
    """Open img in a window; user left-clicks 4 points in order, 'r' to reset,
    'q'/Esc to abort, Enter to commit."""
    h, w = img.shape[:2]
    clicks: list[tuple[int, int]] = []
    overlay = img.copy()

    def redraw_overlay() -> None:
        nonlocal overlay
        overlay = img.copy()
        for (x, y), lbl in zip(clicks, LABELS):
            cv2.circle(overlay, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                overlay, lbl, (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
            )

    def on_mouse(event: int, x: int, y: int, flags: int, _: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            print(
                f"  point {LABELS[len(clicks)-1]}: pixel=({x}, {y})  "
                f"meter={METER_POINTS[LABELS[len(clicks)-1]]}"
            )
            redraw_overlay()

    cv2.namedWindow("homography", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("homography", w * 2, h * 2)  # 2x for 640x480 → 1280x960
    cv2.setMouseCallback("homography", on_mouse)
    print(f"\nFrame: {w}x{h}. Click points in order: 1, 2, 3, 4.")
    print("  r = reset, q/Esc = abort, Enter = commit when 4 points clicked.\n")

    while True:
        disp = overlay.copy()
        next_lbl = LABELS[len(clicks)] if len(clicks) < 4 else "(press Enter)"
        hint = f"Next: {next_lbl}  |  r=reset  q=quit  Enter=commit"
        cv2.rectangle(disp, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(
            disp, hint, (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.imshow("homography", disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            cv2.destroyAllWindows()
            sys.exit("aborted")
        if k == ord("r"):
            clicks.clear()
            redraw_overlay()
            print("  (reset)")
        if k in (13, 10) and len(clicks) == 4:
            break

    cv2.destroyAllWindows()
    return clicks


def render_verification(
    img: np.ndarray,
    H: np.ndarray,
    clicks: list[tuple[int, int]],
    out_path: Path,
) -> None:
    """Project a 1-meter grid from world space back through H^-1 onto the image."""
    Hinv = np.linalg.inv(H)
    verify = img.copy()

    def m2px(X: float, Y: float) -> tuple[int, int] | None:
        p = Hinv @ np.array([X, Y, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        u, v = p[0] / p[2], p[1] / p[2]
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return int(round(u)), int(round(v))

    # Range a few meters beyond the calibration quadrilateral.
    xs = [m[0] for m in METER_POINTS.values()]
    ys = [m[1] for m in METER_POINTS.values()]
    x_min, x_max = math.floor(min(xs)) - 2, math.ceil(max(xs)) + 2
    y_min, y_max = math.floor(min(ys)) - 2, math.ceil(max(ys)) + 2

    grid_color = (0, 200, 0)
    for X in range(x_min, x_max + 1):
        a = m2px(float(X), float(y_min))
        b = m2px(float(X), float(y_max))
        if a is not None and b is not None:
            cv2.line(verify, a, b, grid_color, 1, cv2.LINE_AA)
    for Y in range(y_min, y_max + 1):
        a = m2px(float(x_min), float(Y))
        b = m2px(float(x_max), float(Y))
        if a is not None and b is not None:
            cv2.line(verify, a, b, grid_color, 1, cv2.LINE_AA)

    # Highlight the X-axis (Y=0) and Y-axis (X=0) more boldly.
    a = m2px(float(x_min), 0.0)
    b = m2px(float(x_max), 0.0)
    if a and b:
        cv2.line(verify, a, b, (0, 255, 255), 2, cv2.LINE_AA)
    a = m2px(0.0, float(y_min))
    b = m2px(0.0, float(y_max))
    if a and b:
        cv2.line(verify, a, b, (0, 255, 255), 2, cv2.LINE_AA)

    # Re-mark the clicked points.
    for (u, v), lbl in zip(clicks, LABELS):
        cv2.circle(verify, (u, v), 7, (0, 0, 255), -1)
        cv2.putText(
            verify, lbl, (u + 9, v - 9),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), verify)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--frame",
        help="Use a saved JPEG instead of grabbing fresh. "
             "Default: grab from RTSP sub-stream.",
    )
    ap.add_argument(
        "--out", default="config/homography.yaml",
        help="YAML output path (default: config/homography.yaml)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.frame:
        frame_path = Path(args.frame)
        img = cv2.imread(str(frame_path))
        if img is None:
            sys.exit(f"failed to read {frame_path}")
        print(f"Loaded saved frame: {frame_path} ({img.shape[1]}x{img.shape[0]})")
    else:
        url = build_rtsp_url()
        print("Grabbing fresh frame from sub-stream …")
        img = grab_substream_frame(url)
        snapshot = repo_root / "events" / "calibration_frame.jpg"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snapshot), img)
        print(f"Saved snapshot to {snapshot} ({img.shape[1]}x{img.shape[0]})")

    clicks = click_four_points(img)

    pixel_pts = np.array(clicks, dtype=np.float32)
    meter_pts = np.array(
        [METER_POINTS[lbl] for lbl in LABELS], dtype=np.float32
    )
    H, _ = cv2.findHomography(pixel_pts, meter_pts)
    if H is None:
        sys.exit("findHomography failed (collinear points?)")

    print("\nReprojection check (clicked pixel -> meters via H):")
    max_err = 0.0
    for (u, v), lbl, target in zip(clicks, LABELS, meter_pts):
        p = H @ np.array([float(u), float(v), 1.0])
        Xm, Ym = p[0] / p[2], p[1] / p[2]
        err = math.hypot(Xm - target[0], Ym - target[1])
        max_err = max(max_err, err)
        print(
            f"  {lbl}: ({u:4d},{v:4d}) -> ({Xm:7.2f}, {Ym:7.2f})  "
            f"target=({target[0]:7.2f}, {target[1]:7.2f})  err={err:.3f} m"
        )
    print(f"  max reprojection error: {max_err:.3f} m")

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "homography": {
            "H": H.tolist(),
            "pixel_pts": [[int(u), int(v)] for u, v in clicks],
            "meter_pts": [[float(x), float(y)] for x, y in meter_pts.tolist()],
            "origin_gps": [float(ORIGIN_GPS[0]), float(ORIGIN_GPS[1])],
            "frame_size": [int(img.shape[1]), int(img.shape[0])],
            "max_reprojection_error_m": float(max_err),
        },
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"\nWrote {out_path}")

    verify_path = out_path.parent / "homography_verify.jpg"
    render_verification(img, H, clicks, verify_path)
    print(f"Wrote {verify_path}")
    print(
        "\nOpen the verify image. The green 1-meter grid should align with "
        "the road plane (parallel/perpendicular to lane stripes; consistent "
        "spacing along the road). The yellow lines are the Y-axis (camera's "
        "north-south through origin) and X-axis (camera's east-west through "
        "origin)."
    )


if __name__ == "__main__":
    main()
