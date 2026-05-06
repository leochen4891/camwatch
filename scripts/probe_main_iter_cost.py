"""Time the inner reader loop's per-frame cost on the main stream.

Splits the loop into measurable phases:
  - cap.read()                    H.264 decode + memcpy
  - cap.get(CAP_PROP_POS_MSEC)    metadata fetch
  - rest of loop body             timestamp anchor, lock, etc.

If cap.read() dominates, decode is the bottleneck. If cap.get dominates,
that's a fixable hot spot. If sum is much less than wall-clock per iter,
something between iterations is blocking.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

_parser = argparse.ArgumentParser()
_parser.add_argument("--low-latency", action="store_true",
                     help="add fflags=nobuffer + flags=low_delay (live-capture style)")
_args, _rest = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _rest

if _args.low_latency:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    )
    print("opening with: rtsp_transport=tcp, nobuffer, low_delay")
else:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    print("opening with: rtsp_transport=tcp (no nobuffer/low_delay)")

import cv2  # noqa: E402

from camwatch.config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    url = cfg.camera.rtsp_url_thumb
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("could not open")
        return 1
    # Drain warmup.
    for _ in range(20):
        cap.read()

    n = 50
    read_times = []
    get_times = []
    iter_times = []
    last = time.monotonic()
    for _ in range(n):
        t0 = time.monotonic()
        ok, _img = cap.read()
        t1 = time.monotonic()
        if not ok:
            continue
        _pts = cap.get(cv2.CAP_PROP_POS_MSEC)
        t2 = time.monotonic()
        read_times.append(t1 - t0)
        get_times.append(t2 - t1)
        iter_times.append(t2 - last)
        last = t2
    cap.release()

    def stats(name, xs):
        xs = sorted(xs)
        n = len(xs)
        if n == 0:
            print(f"  {name}: no samples")
            return
        med = xs[n // 2]
        p95 = xs[min(n - 1, int(n * 0.95))]
        mn, mx = xs[0], xs[-1]
        print(f"  {name:24s}  median={med * 1000:6.1f}ms  p95={p95 * 1000:6.1f}ms  min={mn * 1000:5.1f}  max={mx * 1000:6.1f}")

    print(f"sampled {len(read_times)} iterations:")
    stats("cap.read()", read_times)
    stats("cap.get(POS_MSEC)", get_times)
    stats("full iter (start-to-start)", iter_times)
    avg_iter = sum(iter_times) / len(iter_times)
    print(f"\nimplied raw read fps: {1.0 / avg_iter:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
