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
        "passes": data.get("passes") or [],
    })
    _save_yaml(cfg.calibration_path, data)
    print(f"saved line_a_x={line_a_x} line_b_x={line_b_x} to {cfg.calibration_path}")


# ---------- capture ----------

def cmd_capture(cfg, secs: int, recordings_dir: Path) -> None:
    cal = _load_yaml(cfg.calibration_path)
    if "line_a_x" not in cal or "line_b_x" not in cal:
        raise SystemExit("run `pick-lines` first")
    line_a = int(cal["line_a_x"])
    line_b = int(cal["line_b_x"])

    cap = RtspStream(cfg.camera.rtsp_url)
    det = Detector(
        weights=cfg.model.weights,
        device=cfg.model.device,
        classes=cfg.model.classes,
        conf=cfg.model.conf,
        iou=cfg.model.iou,
    )
    recorder = ClipRecorder(recordings_dir)

    state: dict[int, dict] = {}  # track_id -> {t_a, t_b, last_x, last_t, direction, cls_name}
    passes: list[dict] = []
    t0 = time.monotonic()
    deadline = t0 + secs
    print(f"capture: watching for {secs}s. Drive at GPS-known speeds in each direction now.")
    print(f"clips will be written to {recordings_dir}/")
    print("ctrl+c to stop early; everything captured so far will be saved.\n")

    def stop_at_deadline():
        return time.monotonic() >= deadline

    interrupted = False
    try:
        for fr in cap.frames():
            if stop_at_deadline():
                cap.stop()
                break
            tracks = det.track(fr.image)
            recorder.push(fr.image, fr.ts, tracks)
            for tr in tracks:
                x = tr.ground_point[0]
                st = state.setdefault(tr.track_id, {
                    "t_a": None, "t_b": None,
                    "last_x": None, "last_t": None,
                    "cls_name": tr.cls_name,
                })
                if st["last_x"] is not None:
                    xp, tp = st["last_x"], st["last_t"]
                    if st["t_a"] is None and (xp - line_a) * (x - line_a) <= 0 and xp != line_a:
                        span = x - xp
                        st["t_a"] = tp + (line_a - xp) / span * (fr.ts - tp) if span else fr.ts
                    if st["t_b"] is None and (xp - line_b) * (x - line_b) <= 0 and xp != line_b:
                        span = x - xp
                        st["t_b"] = tp + (line_b - xp) / span * (fr.ts - tp) if span else fr.ts
                    if st["t_a"] is not None and st["t_b"] is not None:
                        direction = "N" if st["t_a"] < st["t_b"] else "S"
                        elapsed = abs(st["t_b"] - st["t_a"])
                        captured_at = datetime.now().astimezone()
                        stamp = captured_at.strftime("%Y%m%dT%H%M%S")
                        clip_name = f"cal_{stamp}_id{tr.track_id}_{direction}.mp4"
                        clip_path = recorder.trigger(
                            name=clip_name,
                            focus_track_id=tr.track_id,
                            line_a_x=line_a,
                            line_b_x=line_b,
                            t_a=st["t_a"],
                            t_b=st["t_b"],
                        )
                        p = {
                            "captured_at": captured_at.isoformat(timespec="seconds"),
                            "track_id": tr.track_id,
                            "class": tr.cls_name,
                            "direction": direction,
                            "elapsed_s": round(elapsed, 4),
                            "known_mph": None,
                            "clip": clip_path or None,
                        }
                        passes.append(p)
                        wallclock = captured_at.strftime("%H:%M:%S")
                        print(
                            f"  [{wallclock}] pass: id={tr.track_id} {tr.cls_name} {direction} "
                            f"elapsed={elapsed:.3f}s -> {clip_name}"
                        )
                        del state[tr.track_id]
                        continue
                st["last_x"] = x
                st["last_t"] = fr.ts
    except KeyboardInterrupt:
        interrupted = True
        cap.stop()
        print("\ninterrupted; flushing pending clips and saving passes...")
    finally:
        recorder.flush()
        existing = _load_yaml(cfg.calibration_path)
        existing_passes = existing.get("passes") or []
        existing_passes.extend(passes)
        existing["passes"] = existing_passes
        _save_yaml(cfg.calibration_path, existing)
        verb = "interrupted" if interrupted else "done"
        print(f"\ncapture {verb}. {len(passes)} new passes appended to {cfg.calibration_path}")
        if not interrupted and passes:
            print("Next: run `python -m camwatch.calibrate annotate` to label your own drives.")
    print("Next: run `python -m camwatch.calibrate annotate` to label your own drives.")


# ---------- annotate ----------

def cmd_annotate(cfg) -> None:
    data = _load_yaml(cfg.calibration_path)
    passes = data.get("passes") or []
    todo = [(i, p) for i, p in enumerate(passes) if p.get("known_mph") is None]
    if not todo:
        print("no unannotated passes found.")
        return

    print(f"{len(todo)} unannotated pass(es). At each prompt, type:")
    print("  <number>   GPS-known speed in mph (e.g. 30)")
    print("  open       open the clip in the default player (then re-prompt)")
    print("  skip / s   discard this pass (not your car)")
    print("  q          stop annotating now (already-typed answers are saved)")
    print()

    discard_idxs: list[int] = []
    stopped = False
    for n, (idx, p) in enumerate(todo, 1):
        print(
            f"[{n}/{len(todo)}] track_id={p['track_id']} dir={p['direction']} "
            f"elapsed={p['elapsed_s']:.3f}s captured_at={p.get('captured_at', '?')}"
        )
        if p.get("clip"):
            print(f"  clip: {p['clip']}")
        while True:
            ans = input("  > ").strip().lower()
            if ans == "q":
                stopped = True
                break
            if ans == "open":
                clip = p.get("clip")
                if clip and Path(clip).exists():
                    subprocess.run(["open", clip], check=False)
                else:
                    print("  no clip available for this pass")
                continue
            if ans in ("skip", "s"):
                discard_idxs.append(idx)
                break
            try:
                p["known_mph"] = float(ans)
                break
            except ValueError:
                print("  not a number; type a speed, 'open', 'skip', or 'q'")
        if stopped:
            break

    if discard_idxs:
        keep = [p for i, p in enumerate(passes) if i not in set(discard_idxs)]
        data["passes"] = keep
    _save_yaml(cfg.calibration_path, data)
    print(f"saved annotations to {cfg.calibration_path}")


# ---------- compute ----------

def cmd_compute(cfg) -> None:
    data = _load_yaml(cfg.calibration_path)
    passes = data.get("passes") or []
    annotated = [p for p in passes if p.get("known_mph") is not None]
    if not annotated:
        raise SystemExit("no annotated passes; run `annotate` first")

    by_dir: dict[str, list[float]] = {"N": [], "S": []}
    for p in annotated:
        mps = float(p["known_mph"]) / MPS_TO_MPH
        implied = mps * float(p["elapsed_s"])
        p["implied_distance_m"] = round(implied, 3)
        by_dir.setdefault(p["direction"], []).append(implied)

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
    data = _load_yaml(cfg.calibration_path)
    passes = [p for p in (data.get("passes") or []) if p.get("known_mph") is not None]
    if not passes:
        print("no annotated passes")
        return
    dist_n = float(data.get("line_distance_m_north", 0))
    dist_s = float(data.get("line_distance_m_south", 0))
    print(f"calibrated distances: N={dist_n:.3f}m  S={dist_s:.3f}m\n")
    print(f"{'idx':>3}  {'dir':>3}  {'elapsed':>8}  {'known':>7}  {'pred':>7}  {'err':>6}")
    for i, p in enumerate(passes, 1):
        d = dist_n if p["direction"] == "N" else dist_s
        if d <= 0 or p["elapsed_s"] <= 0:
            print(f"{i:>3}  {p['direction']:>3}  {p['elapsed_s']:>8.3f}  "
                  f"{p['known_mph']:>7.1f}  {'-':>7}  {'-':>6}")
            continue
        pred = (d / p["elapsed_s"]) * MPS_TO_MPH
        err = pred - p["known_mph"]
        print(f"{i:>3}  {p['direction']:>3}  {p['elapsed_s']:>8.3f}  "
              f"{p['known_mph']:>7.1f}  {pred:>7.1f}  {err:>+6.1f}")


# ---------- entry ----------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="camwatch.calibrate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pick-lines")
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
