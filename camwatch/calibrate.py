"""Interactive calibration tool.

Subcommands:
  pick-lines        Click two vertical lines on a still frame from the camera.
  capture --secs N  Watch the live stream for N seconds and record every full
                    A-then-B (or B-then-A) crossing into calibration.yaml as
                    an unannotated pass.
  annotate          Walk through unannotated passes, ask for known GPS speed.
  compute           Average implied distances per direction; write
                    line_distance_m_north / line_distance_m_south.
  report            Re-run each annotated pass through the speed math; print
                    predicted vs. known speed.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from .capture import RtspStream
from .config import load_config
from .crossing import CrossingDetector
from .db import Database
from .detect import Detector
from .recorder import ClipRecorder
from .speed import MPS_TO_MPH

log = logging.getLogger("camwatch.calibrate")


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


# ---------- pick-lines ----------

def cmd_pick_lines(cfg) -> None:
    frame = _grab_one_frame(cfg.camera.rtsp_url)
    full_h, full_w = frame.shape[:2]
    disp, scale = _scale_to_fit(frame)

    clicks_disp: list[int] = []  # x in display coords

    def on_mouse(event, x, _y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks_disp.append(x)

    win = "pick lines (click line A, then line B; r=reset, s=save, q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        view = disp.copy()
        for i, x in enumerate(clicks_disp[:2]):
            color = (0, 255, 0) if i == 0 else (0, 200, 255)
            cv2.line(view, (x, 0), (x, view.shape[0]), color, 2)
            cv2.putText(view, f"{'A' if i == 0 else 'B'} (x={int(x / scale)})",
                        (x + 6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return
        if key == ord("r"):
            clicks_disp.clear()
        if key == ord("s") and len(clicks_disp) >= 2:
            cv2.destroyAllWindows()
            break
        if len(clicks_disp) > 2:
            del clicks_disp[2:]

    line_a_x = int(round(min(clicks_disp[0], clicks_disp[1]) / scale))
    line_b_x = int(round(max(clicks_disp[0], clicks_disp[1]) / scale))

    data = _load_yaml(cfg.calibration_path)
    data.update({
        "line_a_x": line_a_x,
        "line_b_x": line_b_x,
        "frame_width": full_w,
        "frame_height": full_h,
        "line_distance_m_north": data.get("line_distance_m_north", 0.0),
        "line_distance_m_south": data.get("line_distance_m_south", 0.0),
    })
    _save_yaml(cfg.calibration_path, data)
    print(f"saved line_a_x={line_a_x} line_b_x={line_b_x} to {cfg.calibration_path}")


# ---------- pick-roi ----------

def cmd_pick_roi(cfg) -> None:
    """Pick a region of interest rectangle. YOLO will only see pixels inside.

    Click top-left, then bottom-right of the road belt. The two crossing lines
    must lie within the chosen rectangle for detection to work end-to-end.
    """
    frame = _grab_one_frame(cfg.camera.rtsp_url)
    full_h, full_w = frame.shape[:2]
    disp, scale = _scale_to_fit(frame)

    data = _load_yaml(cfg.calibration_path)
    line_a_disp = int(round(int(data.get("line_a_x", 0) or 0) * scale))
    line_b_disp = int(round(int(data.get("line_b_x", 0) or 0) * scale))

    clicks_disp: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks_disp.append((x, y))

    win = "pick ROI (click top-left, then bottom-right; r=reset, s=save, q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        view = disp.copy()
        # Show the existing crossing lines so the user knows where the ROI must cover.
        if line_a_disp:
            cv2.line(view, (line_a_disp, 0), (line_a_disp, view.shape[0]), (0, 220, 0), 1)
        if line_b_disp:
            cv2.line(view, (line_b_disp, 0), (line_b_disp, view.shape[0]), (220, 120, 0), 1)
        if len(clicks_disp) >= 1:
            cv2.circle(view, clicks_disp[0], 5, (0, 0, 255), -1)
        if len(clicks_disp) >= 2:
            p1 = (min(clicks_disp[0][0], clicks_disp[1][0]), min(clicks_disp[0][1], clicks_disp[1][1]))
            p2 = (max(clicks_disp[0][0], clicks_disp[1][0]), max(clicks_disp[0][1], clicks_disp[1][1]))
            # Dim everything outside the rectangle.
            mask = view.copy()
            cv2.rectangle(mask, (0, 0), (view.shape[1], view.shape[0]), (0, 0, 0), -1)
            cv2.rectangle(mask, p1, p2, (0, 0, 0), -1)
            view = cv2.addWeighted(view, 0.5, mask, 0.5, 0)
            cv2.rectangle(view, p1, p2, (0, 0, 255), 2)
            label = f"ROI {int(p1[0]/scale)},{int(p1[1]/scale)} -> {int(p2[0]/scale)},{int(p2[1]/scale)}"
            cv2.putText(view, label, (p1[0] + 6, max(20, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return
        if key == ord("r"):
            clicks_disp.clear()
        if key == ord("s") and len(clicks_disp) >= 2:
            cv2.destroyAllWindows()
            break
        if len(clicks_disp) > 2:
            del clicks_disp[2:]

    p1 = clicks_disp[0]
    p2 = clicks_disp[1]
    roi_x1 = int(round(min(p1[0], p2[0]) / scale))
    roi_y1 = int(round(min(p1[1], p2[1]) / scale))
    roi_x2 = int(round(max(p1[0], p2[0]) / scale))
    roi_y2 = int(round(max(p1[1], p2[1]) / scale))
    # Clamp to frame
    roi_x1 = max(0, min(roi_x1, full_w))
    roi_x2 = max(0, min(roi_x2, full_w))
    roi_y1 = max(0, min(roi_y1, full_h))
    roi_y2 = max(0, min(roi_y2, full_h))

    line_a = int(data.get("line_a_x") or 0)
    line_b = int(data.get("line_b_x") or 0)
    if line_a and not (roi_x1 <= line_a <= roi_x2):
        print(f"WARNING: line_a_x={line_a} is outside roi x range [{roi_x1}, {roi_x2}]")
    if line_b and not (roi_x1 <= line_b <= roi_x2):
        print(f"WARNING: line_b_x={line_b} is outside roi x range [{roi_x1}, {roi_x2}]")

    data["roi_x1"] = roi_x1
    data["roi_y1"] = roi_y1
    data["roi_x2"] = roi_x2
    data["roi_y2"] = roi_y2
    _save_yaml(cfg.calibration_path, data)
    print(
        f"saved ROI ({roi_x1}, {roi_y1}) -> ({roi_x2}, {roi_y2}) "
        f"= {roi_x2 - roi_x1} x {roi_y2 - roi_y1} px to {cfg.calibration_path}"
    )


# ---------- capture ----------

def cmd_capture(cfg, secs: int, recordings_dir: Path) -> None:
    cal = _load_yaml(cfg.calibration_path)
    if "line_a_x" not in cal or "line_b_x" not in cal:
        raise SystemExit("run `pick-lines` first")
    line_a = int(cal["line_a_x"])
    line_b = int(cal["line_b_x"])
    rx1, ry1 = int(cal.get("roi_x1") or 0), int(cal.get("roi_y1") or 0)
    rx2, ry2 = int(cal.get("roi_x2") or 0), int(cal.get("roi_y2") or 0)
    roi = (rx1, ry1, rx2, ry2) if (rx2 > rx1 and ry2 > ry1) else None

    cap = RtspStream(cfg.camera.rtsp_url)
    det = Detector(
        weights=cfg.model.weights,
        device=cfg.model.device,
        classes=cfg.model.classes,
        conf=cfg.model.conf,
        iou=cfg.model.iou,
        roi=roi,
    )
    recorder = ClipRecorder(recordings_dir)
    crossing = CrossingDetector(line_a, line_b, cfg.max_track_age_s)
    db = Database()

    n_inserted = 0
    t0 = time.monotonic()
    deadline = t0 + secs
    print(f"capture: watching for {secs}s. Drive at GPS-known speeds in each direction now.")
    print(f"clips will be written to {recordings_dir}/")
    print("ctrl+c to stop early; everything captured so far will be saved.\n")

    interrupted = False
    try:
        for fr in cap.frames():
            if time.monotonic() >= deadline:
                cap.stop()
                break
            tracks = det.track(fr.image)
            recorder.push(fr.image, fr.ts, tracks)
            for ev in crossing.update(tracks, fr.ts):
                captured_at = datetime.now().astimezone()
                stamp = captured_at.strftime("%Y%m%dT%H%M%S")
                clip_name = f"cal_{stamp}_id{ev.track_id}_{ev.direction}.mp4"
                clip_path = recorder.trigger(
                    name=clip_name,
                    focus_track_id=ev.track_id,
                    line_a_x=line_a,
                    line_b_x=line_b,
                    t_a=ev.t_a,
                    t_b=ev.t_b,
                )
                db.insert_pass(
                    captured_at=captured_at.isoformat(timespec="seconds"),
                    track_id=ev.track_id,
                    cls_name=ev.cls_name,
                    direction=ev.direction,
                    elapsed_s=round(ev.elapsed_s, 4),
                    clip_path=clip_path or None,
                )
                n_inserted += 1
                wallclock = captured_at.strftime("%H:%M:%S")
                print(
                    f"  [{wallclock}] pass: id={ev.track_id} {ev.cls_name} {ev.direction} "
                    f"elapsed={ev.elapsed_s:.3f}s -> {clip_name}"
                )
    except KeyboardInterrupt:
        interrupted = True
        cap.stop()
        print("\ninterrupted; flushing pending clips...")
    finally:
        recorder.flush()
        verb = "interrupted" if interrupted else "done"
        print(f"\ncapture {verb}. {n_inserted} new passes inserted into camwatch.db")
        if n_inserted:
            print("Next: run `python -m camwatch.calibrate annotate` to label your own drives.")


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
    annotated = [p for p in db.list_passes(limit=10000) if p.known_mph is not None]
    if not annotated:
        raise SystemExit("no annotated passes; run `annotate` first")

    by_dir: dict[str, list[float]] = {"N": [], "S": []}
    for p in annotated:
        mps = float(p.known_mph) / MPS_TO_MPH
        implied = mps * float(p.elapsed_s)
        by_dir.setdefault(p.direction, []).append(implied)

    data = _load_yaml(cfg.calibration_path)
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


# ---------- entry ----------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="camwatch.calibrate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pick-lines")
    sub.add_parser("pick-roi")
    p_cap = sub.add_parser("capture")
    p_cap.add_argument("--secs", type=int, default=300)
    p_cap.add_argument("--recordings-dir", type=Path, default=Path("recordings"))
    sub.add_parser("annotate")
    sub.add_parser("compute")
    sub.add_parser("report")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.cmd == "pick-lines":
        cmd_pick_lines(cfg)
    elif args.cmd == "pick-roi":
        cmd_pick_roi(cfg)
    elif args.cmd == "capture":
        cmd_capture(cfg, args.secs, args.recordings_dir)
    elif args.cmd == "annotate":
        cmd_annotate(cfg)
    elif args.cmd == "compute":
        cmd_compute(cfg)
    elif args.cmd == "report":
        cmd_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
