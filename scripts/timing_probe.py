"""Live timing diagnostic: compare time.monotonic() vs CV PTS vs OSD ticks.

Goal: figure out which timing source is the truthful representation of when
the camera actually captured each frame, so we can replace `time.monotonic()`
in the live capture path if needed.

What it does:
  1. Opens the configured RTSP stream (default sub; --stream main supported).
  2. For ~N seconds (default 60), reads frames in the same low-latency
     pattern the live capture worker uses.
  3. Per frame, records:
        - monotonic_ms          : time.monotonic() at cap.read() return
        - pts_ms                : cv2.CAP_PROP_POS_MSEC for the frame
        - osd_second            : seconds part of the burned-in OSD (optional)
  4. Computes inter-frame deltas for each series and prints summary stats
     (median, mean, p95, p99, stddev). Also prints per-OSD-second frame
     counts so we can see the camera's true fps vs what the deltas claim.

Usage:
    uv run python scripts/timing_probe.py                     # 60s of sub
    uv run python scripts/timing_probe.py --stream main       # main stream
    uv run python scripts/timing_probe.py --secs 30 --no-ocr  # skip OCR
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# capture.py sets the FFmpeg env var on import; keep that ordering.
import cv2  # noqa: E402

from camwatch.config import load_config  # noqa: E402
from camwatch.ts_reader import read_timestamp  # noqa: E402

# OSD region per stream — main is what we calibrated earlier.
# For sub (640x480) we proportionally scale the same scene-relative position.
OSD_REGION_MAIN = (700, 1810, 2000, 1910)
OSD_REGION_SUB = (175, 452, 500, 477)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def summarize(name: str, deltas_ms: list[float]) -> None:
    if not deltas_ms:
        print(f"  {name}: no samples")
        return
    print(
        f"  {name:18s}  "
        f"n={len(deltas_ms):4d}  "
        f"min={min(deltas_ms):6.1f}  "
        f"med={statistics.median(deltas_ms):6.1f}  "
        f"mean={statistics.fmean(deltas_ms):6.1f}  "
        f"std={statistics.pstdev(deltas_ms):6.1f}  "
        f"p95={percentile(deltas_ms, 95):6.1f}  "
        f"p99={percentile(deltas_ms, 99):6.1f}  "
        f"max={max(deltas_ms):6.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream",
        choices=("sub", "main"),
        default="sub",
        help="Which stream to probe. The capture worker uses 'sub'.",
    )
    parser.add_argument("--secs", type=float, default=60.0, help="seconds to record")
    parser.add_argument("--no-ocr", action="store_true", help="skip OSD OCR")
    args = parser.parse_args()

    cfg = load_config()
    if args.stream == "sub":
        url = cfg.camera.rtsp_url
        region = OSD_REGION_SUB
    else:
        url = cfg.camera.rtsp_url_thumb
        region = OSD_REGION_MAIN
        if url is None:
            raise SystemExit("camera.path_thumb not configured for main stream")

    safe_url = url.replace(cfg.camera.password, "***")
    print(f"opening: {safe_url}")
    print(f"OSD region: {region}  (omit OCR with --no-ocr)\n")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit("could not open stream")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Warm up: drop the first few frames so timestamps reflect a settled stream.
    for _ in range(5):
        cap.read()

    monotonic_ms: list[float] = []
    pts_ms: list[float] = []
    osd_seconds: list[int] = []
    rows: list[tuple[int, float, float, int | None]] = []

    t_wall_start = time.monotonic()
    last_mono: float | None = None
    last_pts: float | None = None
    frame_idx = 0

    try:
        while time.monotonic() - t_wall_start < args.secs:
            ok, image = cap.read()
            if not ok:
                continue
            now_mono = time.monotonic()
            pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            osd_sec: int | None = None
            if not args.no_ocr:
                ts = read_timestamp(image, region)
                if ts is not None:
                    osd_sec = ts.second

            if last_mono is not None:
                monotonic_ms.append((now_mono - last_mono) * 1000.0)
            last_mono = now_mono
            if last_pts is not None and pts > 0:
                pts_ms.append(pts - last_pts)
            last_pts = pts
            if osd_sec is not None:
                osd_seconds.append(osd_sec)

            rows.append((frame_idx, (now_mono - t_wall_start) * 1000.0, pts, osd_sec))
            frame_idx += 1
    finally:
        cap.release()

    if frame_idx == 0:
        print("no frames received")
        return 1

    elapsed_wall = time.monotonic() - t_wall_start
    print(f"recorded {frame_idx} frames in {elapsed_wall:.2f}s "
          f"(wall-clock fps={frame_idx / elapsed_wall:.2f})\n")

    print("per-frame inter-arrival deltas (ms):")
    summarize("monotonic_dt_ms", monotonic_ms)
    summarize("pts_dt_ms      ", pts_ms)

    if osd_seconds:
        per_second = Counter(osd_seconds)
        # Show only the inner OSD-seconds (drop first and last because they're
        # truncated at the probe's start/end).
        inner = sorted(per_second.keys())[1:-1] if len(per_second) >= 3 else sorted(per_second.keys())
        if inner:
            counts = [per_second[s] for s in inner]
            print(f"\nOSD ticks observed: {len(per_second)} distinct seconds")
            print(f"  full counts: {dict(sorted(per_second.items()))}")
            print(f"  inner only : {[per_second[s] for s in inner]}")
            print(
                f"  inner-second frames: "
                f"min={min(counts)}  med={statistics.median(counts):.1f}  "
                f"mean={statistics.fmean(counts):.2f}  max={max(counts)}"
            )
            print(f"  ⇒ camera observed fps (mean) = {statistics.fmean(counts):.2f}")
        else:
            print("\nnot enough OSD seconds to estimate fps (probe too short)")

    print("\nfirst 20 frames (idx, wall_ms, pts_ms, osd_sec):")
    for r in rows[:20]:
        print(f"  {r[0]:3d}  {r[1]:8.1f}  {r[2]:8.1f}  {r[3]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
