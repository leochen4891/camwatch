"""Interactive calibration tool.

Subcommands:
  pick-roi          Drag-adjust the YOLO region-of-interest rectangle.
  annotate          Walk through unannotated passes, ask for known GPS speed.
  compute           Average implied distances per direction; write
                    line_distance_m_north / line_distance_m_south (legacy
                    speed conversion for pre-homography DB rows).
  report            Re-run each annotated pass through the legacy speed
                    math; print predicted vs. known speed.
  freeze / restore  Snapshot/recreate calibration_points across DB wipes.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import yaml

from .config import load_config
from .db import Database

log = logging.getLogger("camwatch.calibrate")

MPS_TO_MPH = 2.2369362920544


# ---------- shared helpers ----------

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _grab_one_frame(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit("could not open RTSP stream")
    for _ in range(8):  # let buffer settle
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise SystemExit("RTSP read failed")
    cap.release()
    return frame


def _scale_to_fit(frame, max_w: int = 1280):
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame, 1.0
    scale = max_w / w
    return cv2.resize(frame, (max_w, int(h * scale))), scale


# ---------- pick-roi ----------

def cmd_pick_roi(cfg) -> None:
    """Drag-adjust the YOLO region-of-interest rectangle.

    Loads any existing ROI from calibration.yaml as the starting rectangle
    (or a sensible default if none). Click-drag a corner to resize from that
    corner; click-drag an edge to resize axis-locked; click-drag inside to
    move the whole rectangle. `n` draws a fresh rectangle from two clicks,
    `r` reverts to the loaded ROI, `s` saves, `q` quits.
    """
    frame = _grab_one_frame(cfg.camera.rtsp_url)
    full_h, full_w = frame.shape[:2]
    disp, scale = _scale_to_fit(frame)
    disp_h, disp_w = disp.shape[:2]

    data = _load_yaml(cfg.calibration_path)

    def _initial_roi_disp() -> list[int]:
        # Existing ROI takes priority. Otherwise center a half-frame rectangle.
        rx1 = int(data.get("roi_x1") or 0)
        ry1 = int(data.get("roi_y1") or 0)
        rx2 = int(data.get("roi_x2") or 0)
        ry2 = int(data.get("roi_y2") or 0)
        if rx2 > rx1 and ry2 > ry1:
            return [
                int(round(rx1 * scale)), int(round(ry1 * scale)),
                int(round(rx2 * scale)), int(round(ry2 * scale)),
            ]
        return [disp_w // 4, disp_h // 4, 3 * disp_w // 4, 3 * disp_h // 4]

    initial_disp = _initial_roi_disp()
    roi = list(initial_disp)            # [x1, y1, x2, y2] in display coords
    drag = {"mode": None, "anchor": None, "orig": None}
    # Two-click "draw new" mode toggled by `n`
    draw_mode = {"on": False, "first": None}
    EDGE_TOL = 18                       # px tolerance for hit-test (display coords)

    def _hit_test(mx: int, my: int) -> str | None:
        x1, y1, x2, y2 = roi
        near_l = abs(mx - x1) <= EDGE_TOL
        near_r = abs(mx - x2) <= EDGE_TOL
        near_t = abs(my - y1) <= EDGE_TOL
        near_b = abs(my - y2) <= EDGE_TOL
        if near_l and near_t: return "tl"
        if near_r and near_t: return "tr"
        if near_l and near_b: return "bl"
        if near_r and near_b: return "br"
        if near_l and y1 <= my <= y2: return "left"
        if near_r and y1 <= my <= y2: return "right"
        if near_t and x1 <= mx <= x2: return "top"
        if near_b and x1 <= mx <= x2: return "bottom"
        if x1 < mx < x2 and y1 < my < y2: return "move"
        return None

    def on_mouse(event, x, y, _flags, _userdata):
        if draw_mode["on"]:
            if event == cv2.EVENT_LBUTTONDOWN:
                if draw_mode["first"] is None:
                    draw_mode["first"] = (x, y)
                else:
                    x0, y0 = draw_mode["first"]
                    roi[0], roi[1] = min(x0, x), min(y0, y)
                    roi[2], roi[3] = max(x0, x), max(y0, y)
                    draw_mode["on"] = False
                    draw_mode["first"] = None
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            mode = _hit_test(x, y)
            if mode is not None:
                drag["mode"] = mode
                drag["anchor"] = (x, y)
                drag["orig"] = list(roi)
        elif event == cv2.EVENT_MOUSEMOVE and drag["mode"]:
            ax, ay = drag["anchor"]
            ox1, oy1, ox2, oy2 = drag["orig"]
            dx, dy = x - ax, y - ay
            m = drag["mode"]
            if m == "move":
                roi[:] = [ox1 + dx, oy1 + dy, ox2 + dx, oy2 + dy]
            elif m == "tl":   roi[:] = [ox1 + dx, oy1 + dy, ox2, oy2]
            elif m == "tr":   roi[:] = [ox1, oy1 + dy, ox2 + dx, oy2]
            elif m == "bl":   roi[:] = [ox1 + dx, oy1, ox2, oy2 + dy]
            elif m == "br":   roi[:] = [ox1, oy1, ox2 + dx, oy2 + dy]
            elif m == "left":   roi[0] = ox1 + dx
            elif m == "right":  roi[2] = ox2 + dx
            elif m == "top":    roi[1] = oy1 + dy
            elif m == "bottom": roi[3] = oy2 + dy
            # Keep a minimum 20 px box; let the user release before fixing.
            if roi[2] - roi[0] < 20: roi[2] = roi[0] + 20
            if roi[3] - roi[1] < 20: roi[3] = roi[1] + 20
        elif event == cv2.EVENT_LBUTTONUP:
            drag["mode"] = None

    win = "pick ROI (drag corners/edges/inside; n=new  r=revert  s=save  q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    HANDLE = 8
    EDGE_COLOR = (0, 0, 255)
    HANDLE_COLOR = (0, 200, 255)
    while True:
        view = disp.copy()
        # Clamp to display before drawing.
        roi[0] = max(0, min(roi[0], disp_w - 1))
        roi[1] = max(0, min(roi[1], disp_h - 1))
        roi[2] = max(roi[0] + 20, min(roi[2], disp_w))
        roi[3] = max(roi[1] + 20, min(roi[3], disp_h))
        x1, y1, x2, y2 = roi
        # Dim outside the rectangle.
        mask = view.copy()
        cv2.rectangle(mask, (0, 0), (disp_w, disp_h), (0, 0, 0), -1)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), -1)
        view = cv2.addWeighted(view, 0.55, mask, 0.45, 0)
        cv2.rectangle(view, (x1, y1), (x2, y2), EDGE_COLOR, 2)
        # Corner + edge midpoint handles.
        for (hx, hy) in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
            cv2.rectangle(view, (hx - HANDLE, hy - HANDLE), (hx + HANDLE, hy + HANDLE),
                          HANDLE_COLOR, -1)
            cv2.rectangle(view, (hx - HANDLE, hy - HANDLE), (hx + HANDLE, hy + HANDLE),
                          (0, 0, 0), 1)
        mx_x, my_y = (x1 + x2) // 2, (y1 + y2) // 2
        for (hx, hy) in ((mx_x, y1), (mx_x, y2), (x1, my_y), (x2, my_y)):
            cv2.circle(view, (hx, hy), HANDLE - 1, HANDLE_COLOR, -1)
            cv2.circle(view, (hx, hy), HANDLE - 1, (0, 0, 0), 1)
        # Status label (full-res coords so the user knows what'll be saved).
        full_x1 = int(round(x1 / scale))
        full_y1 = int(round(y1 / scale))
        full_x2 = int(round(x2 / scale))
        full_y2 = int(round(y2 / scale))
        label = (
            f"ROI ({full_x1},{full_y1}) -> ({full_x2},{full_y2})  "
            f"= {full_x2 - full_x1} x {full_y2 - full_y1} px"
        )
        if draw_mode["on"]:
            label = ("NEW: " + ("click 2nd corner" if draw_mode["first"] is not None
                                else "click 1st corner")) + "    " + label
        cv2.rectangle(view, (0, 0), (disp_w, 36), (0, 0, 0), -1)
        cv2.putText(view, label, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return
        if key == ord("n"):
            draw_mode["on"] = True
            draw_mode["first"] = None
        if key == ord("r"):
            roi[:] = list(initial_disp)
            draw_mode["on"] = False
            draw_mode["first"] = None
        if key == ord("s"):
            cv2.destroyAllWindows()
            break

    roi_x1 = max(0, min(int(round(roi[0] / scale)), full_w))
    roi_y1 = max(0, min(int(round(roi[1] / scale)), full_h))
    roi_x2 = max(0, min(int(round(roi[2] / scale)), full_w))
    roi_y2 = max(0, min(int(round(roi[3] / scale)), full_h))

    data["roi_x1"] = roi_x1
    data["roi_y1"] = roi_y1
    data["roi_x2"] = roi_x2
    data["roi_y2"] = roi_y2
    data["frame_width"] = full_w
    data["frame_height"] = full_h
    _save_yaml(cfg.calibration_path, data)
    print(
        f"saved ROI ({roi_x1}, {roi_y1}) -> ({roi_x2}, {roi_y2}) "
        f"= {roi_x2 - roi_x1} x {roi_y2 - roi_y1} px to {cfg.calibration_path}"
    )


# ---------- annotate ----------

def cmd_annotate(cfg) -> None:
    db = Database()
    todo = [p for p in db.list_passes(limit=10000) if p.known_mph is None]
    if not todo:
        print("no unannotated passes found.")
        return

    print(f"{len(todo)} unannotated pass(es). At each prompt, type:")
    print("  <number>   GPS-known speed in mph (e.g. 30)")
    print("  open       open the clip in the default player (then re-prompt)")
    print("  skip / s   delete this pass (not your car)")
    print("  q          stop annotating now")
    print()

    stopped = False
    for n, p in enumerate(todo, 1):
        print(
            f"[{n}/{len(todo)}] id={p.id} track_id={p.track_id} dir={p.direction} "
            f"elapsed={p.elapsed_s:.3f}s captured_at={p.captured_at}"
        )
        if p.clip_path:
            print(f"  clip: {p.clip_path}")
        while True:
            ans = input("  > ").strip().lower()
            if ans == "q":
                stopped = True
                break
            if ans == "open":
                if p.clip_path and Path(p.clip_path).exists():
                    subprocess.run(["open", p.clip_path], check=False)
                else:
                    print("  no clip available for this pass")
                continue
            if ans in ("skip", "s"):
                db.soft_delete([p.id])
                break
            try:
                db.set_known_mph(p.id, float(ans))
                break
            except ValueError:
                print("  not a number; type a speed, 'open', 'skip', or 'q'")
        if stopped:
            break

    print(f"annotations saved to camwatch.db")


# ---------- compute ----------

def cmd_compute(cfg) -> None:
    db = Database()
    data = _load_yaml(cfg.calibration_path)

    by_dir: dict[str, list[float]] = {"N": [], "S": []}
    seen: set[tuple] = set()
    # Frozen YAML points first (always count, survive DB clears).
    for pt in (data.get("calibration_points") or []):
        try:
            elapsed = float(pt["elapsed_s"])
            known = float(pt["known_mph"])
        except (KeyError, TypeError, ValueError):
            continue
        if elapsed <= 0:
            continue
        seen.add((pt.get("track_id"), pt.get("captured_at")))
        by_dir[pt["direction"]].append((known / MPS_TO_MPH) * elapsed)
    # Then DB known passes (skip duplicates already in YAML).
    for p in db.list_passes(limit=10000):
        if p.known_mph is None:
            continue
        if (p.track_id, p.captured_at) in seen:
            continue
        if p.elapsed_s <= 0:
            continue
        by_dir[p.direction].append((float(p.known_mph) / MPS_TO_MPH) * p.elapsed_s)

    if not by_dir["N"] and not by_dir["S"]:
        raise SystemExit("no calibration data; annotate passes or freeze first")

    if by_dir["N"]:
        d = sum(by_dir["N"]) / len(by_dir["N"])
        data["line_distance_m_north"] = round(d, 3)
        print(f"northbound: avg distance = {d:.3f} m  (n={len(by_dir['N'])})")
    if by_dir["S"]:
        d = sum(by_dir["S"]) / len(by_dir["S"])
        data["line_distance_m_south"] = round(d, 3)
        print(f"southbound: avg distance = {d:.3f} m  (n={len(by_dir['S'])})")
    _save_yaml(cfg.calibration_path, data)
    print(f"saved to {cfg.calibration_path}")


# ---------- report ----------

def cmd_report(cfg) -> None:
    db = Database()
    annotated = [p for p in db.list_passes(limit=10000) if p.known_mph is not None]
    if not annotated:
        print("no annotated passes")
        return
    data = _load_yaml(cfg.calibration_path)
    dist_n = float(data.get("line_distance_m_north", 0))
    dist_s = float(data.get("line_distance_m_south", 0))
    print(f"calibrated distances: N={dist_n:.3f}m  S={dist_s:.3f}m\n")
    print(f"{'idx':>3}  {'dir':>3}  {'elapsed':>8}  {'known':>7}  {'pred':>7}  {'err':>6}")
    for i, p in enumerate(annotated, 1):
        d = dist_n if p.direction == "N" else dist_s
        if d <= 0 or p.elapsed_s <= 0:
            print(f"{i:>3}  {p.direction:>3}  {p.elapsed_s:>8.3f}  "
                  f"{p.known_mph:>7.1f}  {'-':>7}  {'-':>6}")
            continue
        pred = (d / p.elapsed_s) * MPS_TO_MPH
        err = pred - p.known_mph
        print(f"{i:>3}  {p.direction:>3}  {p.elapsed_s:>8.3f}  "
              f"{p.known_mph:>7.1f}  {pred:>7.1f}  {err:>+6.1f}")


# ---------- freeze / restore ----------

def cmd_freeze(cfg) -> None:
    """Snapshot all DB passes that have known_mph set into calibration.yaml.

    Survives DB clears. Use `restore` to recreate the rows if the DB is later
    wiped. Replaces any existing `calibration_points` list in the YAML.
    """
    db = Database()
    annotated = [p for p in db.list_passes(limit=10000) if p.known_mph is not None]
    if not annotated:
        print("no annotated passes in DB to freeze")
        return
    points = [
        {
            "direction": p.direction,
            "known_mph": float(p.known_mph),
            "elapsed_s": float(p.elapsed_s),
            "captured_at": p.captured_at,
            "track_id": int(p.track_id),
            "cls_name": p.cls_name,
            "clip_path": p.clip_path,
        }
        for p in annotated
    ]
    data = _load_yaml(cfg.calibration_path)
    data["calibration_points"] = points
    _save_yaml(cfg.calibration_path, data)
    print(f"froze {len(points)} calibration points to {cfg.calibration_path}")


def cmd_restore(cfg) -> None:
    """Recreate DB passes from `calibration_points` in calibration.yaml.

    Skips entries whose (track_id, captured_at) already exists in the DB so
    the command is idempotent.
    """
    data = _load_yaml(cfg.calibration_path)
    points = data.get("calibration_points") or []
    if not points:
        print(f"no calibration_points found in {cfg.calibration_path}")
        return

    db = Database()
    existing = {(p.track_id, p.captured_at) for p in db.list_passes(limit=10000)}
    n_new = n_skip = 0
    for pt in points:
        key = (int(pt["track_id"]), pt["captured_at"])
        if key in existing:
            n_skip += 1
            continue
        pid = db.insert_pass(
            captured_at=pt["captured_at"],
            track_id=int(pt["track_id"]),
            cls_name=pt.get("cls_name"),
            direction=pt["direction"],
            elapsed_s=float(pt["elapsed_s"]),
            clip_path=pt.get("clip_path"),
        )
        db.set_known_mph(pid, float(pt["known_mph"]))
        n_new += 1
    print(f"restore: inserted {n_new}, skipped {n_skip} already-present")


# ---------- entry ----------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="camwatch.calibrate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pick-roi")
    sub.add_parser("annotate")
    sub.add_parser("compute")
    sub.add_parser("report")
    sub.add_parser("freeze")
    sub.add_parser("restore")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.cmd == "pick-roi":
        cmd_pick_roi(cfg)
    elif args.cmd == "annotate":
        cmd_annotate(cfg)
    elif args.cmd == "compute":
        cmd_compute(cfg)
    elif args.cmd == "freeze":
        cmd_freeze(cfg)
    elif args.cmd == "restore":
        cmd_restore(cfg)
    elif args.cmd == "report":
        cmd_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
