"""Smoke-test the RTSP stream: read N frames, print FPS, save one JPG."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camwatch.config import load_config


def main() -> int:
    cfg = load_config()
    url = cfg.camera.rtsp_url
    safe = url.replace(cfg.camera.password, "***")
    print(f"connecting: {safe}")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("ERROR: could not open RTSP stream", file=sys.stderr)
        return 1

    n_target = 100
    out_path = Path("/tmp/camwatch_test.jpg")
    t0 = time.monotonic()
    frames = 0
    last_frame = None

    while frames < n_target:
        ok, frame = cap.read()
        if not ok:
            print(f"read failed at frame {frames}", file=sys.stderr)
            break
        frames += 1
        last_frame = frame
        if frames % 25 == 0:
            elapsed = time.monotonic() - t0
            print(f"  {frames} frames in {elapsed:.2f}s ({frames / elapsed:.1f} fps)")

    cap.release()
    elapsed = time.monotonic() - t0
    print(f"done: {frames} frames in {elapsed:.2f}s ({frames / elapsed:.1f} fps)")

    if last_frame is not None:
        cv2.imwrite(str(out_path), last_frame)
        h, w = last_frame.shape[:2]
        print(f"saved {out_path}  ({w}x{h})")
    return 0 if frames == n_target else 2


if __name__ == "__main__":
    raise SystemExit(main())
