"""Click points on a full-resolution main-stream frame.

Grabs a fresh frame from the MAIN stream (whatever its native resolution
is — 2048x1536 on the old E1, 2560x1440 on the CX410) and opens it for
clicking. Each click adds a numbered point. The numbering is stable
(point 1 stays point 1 even if you undo later points), so you can
describe what each point is afterwards (e.g. "1 is the manhole cover,
2-7 are 5 ft apart along the white line, 8 is the corner of the
driveway").

Controls:
  left-click    add a numbered point
  u             undo last point (decrement counter)
  r             reset everything
  s             save in-place and keep clicking
  f             refresh image from main stream (preserves existing clicks)
  Enter         save and exit
  q / Esc       abort without saving

Output:
  config/marked_points.yaml          numbered points in main-stream pixel coords
  config/marked_points_overlay.jpg   the frame with all points labelled
  events/calibration_main_frame.jpg  the captured frame (cached)
"""

from __future__ import annotations

import argparse
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
HOMOG_YAML = REPO / "config" / "homography.yaml"

FT_TO_M = 0.3048


def overlay_grid_from_homography(img: np.ndarray) -> np.ndarray:
    """Draw the current homography's 5ft major grid + axes onto img (returned
    as a copy). H now maps main-stream pixels directly, so no scale factor."""
    import math
    if not HOMOG_YAML.exists():
        print(f"WARNING: {HOMOG_YAML} not found — no grid overlay")
        return img.copy()
    data = yaml.safe_load(HOMOG_YAML.read_text())["homography"]
    H = np.array(data["H"], dtype=np.float64)
    Hinv = np.linalg.inv(H)

    out = img.copy()

    def m2px(X: float, Y: float) -> tuple[int, int] | None:
        p = Hinv @ np.array([X, Y, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        u = p[0] / p[2]
        v = p[1] / p[2]
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return int(round(u)), int(round(v))

    Xs = [p["X"] for p in data["meter_pts"]]
    Ys = [p["Y"] for p in data["meter_pts"]]
    x_min = math.floor(min(Xs)) - 1
    x_max = math.ceil(max(Xs)) + 1
    y_min = math.floor(min(Ys)) - 1
    y_max = math.ceil(max(Ys)) + 1

    major = (255, 200, 0)   # cyan-ish
    axis = (0, 220, 255)    # yellow

    # 11 cyan 5ft horizontal Y-lines spanning the road
    for n in range(-5, 6):
        Y = n * 5 * FT_TO_M
        a = m2px(float(x_min), Y)
        b = m2px(float(x_max), Y)
        if a and b:
            cv2.line(out, a, b, major, 1, cv2.LINE_AA)

    # Yellow axes through point 6 (X=0, Y=0)
    a = m2px(float(x_min), 0.0)
    b = m2px(float(x_max), 0.0)
    if a and b:
        cv2.line(out, a, b, axis, 1, cv2.LINE_AA)
    a = m2px(0.0, float(y_min))
    b = m2px(0.0, float(y_max))
    if a and b:
        cv2.line(out, a, b, axis, 1, cv2.LINE_AA)

    return out


def grab_mainstream_frame(url: str) -> np.ndarray:
    """Open RTSP main stream, decode the first keyframe, return BGR."""
    import sys
    import av
    import av.codec.hwaccel as hw

    print("Opening main stream … (waiting for keyframe — Reolink main GOPs "
          "can be 5–10 s, so this may take a moment)")
    # Platform-conditional hwaccel: matches camwatch/capture.py.
    if sys.platform == "darwin":
        hwa = hw.HWAccel(device_type="videotoolbox", allow_software_fallback=True)
    elif sys.platform.startswith("linux"):
        hwa = hw.HWAccel(device_type="cuda", allow_software_fallback=True)
    else:
        hwa = None
    opts = {"rtsp_transport": "tcp"}
    container = av.open(url, options=opts, hwaccel=hwa, timeout=(15.0, 15.0))
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
    raise RuntimeError("no frame decoded from main stream")


def build_main_url() -> str:
    load_dotenv(REPO / ".env")
    user = quote(os.environ["REOLINK_USER"], safe="")
    pw = quote(os.environ["REOLINK_PASS"], safe="")
    cfg = yaml.safe_load((REPO / "config" / "config.yaml").read_text())
    cam = cfg["camera"]
    return f"rtsp://{user}:{pw}@{cam['host']}:{cam['port']}{cam['path']}"


def click_loop(img: np.ndarray, refresh=None) -> list[tuple[int, int]]:
    """Open img, collect numbered points. Returns the click list.

    If `refresh` is callable, the `f` key swaps the displayed image for the
    one returned by `refresh()` (preserving existing clicks). Used when a
    transient occlusion (e.g. a parked car) covered a click target.
    """
    img_box = [img]
    H, W = img.shape[:2]
    points: list[tuple[int, int]] = []

    # Auto-fit window to a max width
    max_w = 1600
    scale = min(1.0, max_w / W)
    win_w = int(W * scale)
    win_h = int(H * scale)

    def render() -> np.ndarray:
        out = img_box[0].copy()
        for i, (x, y) in enumerate(points, 1):
            cv2.circle(out, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(out, (x, y), 9, (0, 0, 0), 1)
            cv2.putText(
                out, str(i), (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                out, str(i), (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA,
            )
        cv2.rectangle(out, (0, 0), (W, 36), (0, 0, 0), -1)
        keys = "u=undo  r=reset  s=save"
        if refresh is not None:
            keys += "  f=refresh"
        keys += "  Enter=save+exit  q=abort"
        status = f"Points: {len(points)}    Next index: {len(points)+1}    {keys}"
        cv2.putText(
            out, status, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return out

    def on_mouse(event: int, x: int, y: int, flags: int, _: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox = int(round(x / scale))
        oy = int(round(y / scale))
        points.append((ox, oy))
        print(f"  point {len(points)}: ({ox}, {oy})")

    cv2.namedWindow("mark points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mark points", win_w, win_h)
    cv2.setMouseCallback("mark points", on_mouse)

    print(f"\nFrame size: {W}x{H}  display scale: {scale:.2f}")
    print("Click points; describe what each one is afterwards.\n")

    while True:
        disp = render()
        if scale != 1.0:
            disp = cv2.resize(disp, (win_w, win_h))
        cv2.imshow("mark points", disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            cv2.destroyAllWindows()
            sys.exit("aborted (no file written)")
        if k == ord("r"):
            points.clear()
            print("  (reset)")
        if k == ord("u"):
            if points:
                p = points.pop()
                print(f"  (undo: removed point {len(points)+1}: {p})")
        if k == ord("s"):
            save(img_box[0], points)
            print(f"  (saved {len(points)} points; keep clicking or press Enter to exit)")
        if k == ord("f") and refresh is not None:
            print("  (refreshing from main stream — waiting for keyframe …)")
            try:
                new_img = refresh()
            except Exception as e:  # noqa: BLE001
                print(f"  (refresh failed: {e}; keeping current frame)")
            else:
                if new_img.shape[:2] != (H, W):
                    print(
                        f"  (refresh got {new_img.shape[1]}x{new_img.shape[0]}, "
                        f"expected {W}x{H} — keeping current frame)"
                    )
                else:
                    img_box[0] = new_img
                    print("  (frame refreshed; clicks preserved)")
        if k in (13, 10):
            break

    cv2.destroyAllWindows()
    return points


def save(img: np.ndarray, points: list[tuple[int, int]]) -> None:
    H, W = img.shape[:2]
    payload = {
        "marked_points": {
            "source_image": str(SNAPSHOT.relative_to(REPO)),
            "frame_size": [int(W), int(H)],
            "points": [
                {"idx": i, "pixel": [int(x), int(y)]}
                for i, (x, y) in enumerate(points, 1)
            ],
        }
    }
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUT_YAML.write_text(yaml.safe_dump(payload, sort_keys=False))

    overlay = img.copy()
    for i, (x, y) in enumerate(points, 1):
        cv2.circle(overlay, (x, y), 12, (0, 255, 0), -1)
        cv2.circle(overlay, (x, y), 13, (0, 0, 0), 2)
        cv2.putText(
            overlay, str(i), (x + 14, y - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, str(i), (x + 14, y - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA,
        )
    cv2.imwrite(str(OUT_OVERLAY), overlay)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--frame",
        help="Use a saved JPEG instead of grabbing fresh (default: grab from main stream).",
    )
    ap.add_argument(
        "--with-grid", action="store_true",
        help="Overlay the current homography's 5ft grid + axes as visual reference "
             "while clicking. The overlay is for display only and isn't saved into "
             "the click positions.",
    )
    ap.add_argument(
        "--use-cached", action="store_true",
        help="Skip the RTSP grab and use the previously cached frame at "
             "events/calibration_main_frame.jpg. Useful when re-clicking against "
             "the same scene the homography was built on.",
    )
    args = ap.parse_args()

    url: str | None = None
    if args.frame:
        img = cv2.imread(args.frame)
        if img is None:
            sys.exit(f"failed to read {args.frame}")
        print(f"Loaded frame: {args.frame} ({img.shape[1]}x{img.shape[0]})")
    elif args.use_cached:
        if not SNAPSHOT.exists():
            sys.exit(f"--use-cached: {SNAPSHOT} doesn't exist; run without --use-cached first")
        img = cv2.imread(str(SNAPSHOT))
        if img is None:
            sys.exit(f"failed to read cached {SNAPSHOT}")
        print(f"Using cached frame: {SNAPSHOT} ({img.shape[1]}x{img.shape[0]})")
        url = build_main_url()
    else:
        url = build_main_url()
        img = grab_mainstream_frame(url)
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SNAPSHOT), img)
        print(f"Saved snapshot to {SNAPSHOT} ({img.shape[1]}x{img.shape[0]})")

    if args.with_grid:
        img = overlay_grid_from_homography(img)
        print("Overlaid 5ft grid + axes from current homography (display only).")

    def refresh_from_stream() -> np.ndarray:
        assert url is not None
        new_img = grab_mainstream_frame(url)
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SNAPSHOT), new_img)
        print(f"  (saved refreshed snapshot to {SNAPSHOT})")
        return new_img

    refresh = refresh_from_stream if (url is not None and not args.with_grid) else None
    points = click_loop(img, refresh=refresh)
    if not points:
        sys.exit("no points marked; nothing saved")
    save(img, points)
    print(f"\nWrote {OUT_YAML}")
    print(f"Wrote {OUT_OVERLAY}")


if __name__ == "__main__":
    main()
