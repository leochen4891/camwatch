"""Diagnostic: open the main RTSP stream and log every frame's PTS gap.

Goal: tell whether the multi-second gaps we see in the thumb-upgrader's
buffer are because (a) the camera/network actually skips frames, or
(b) our consumer is dropping intermediate frames between cap.read()
calls (the RtspStream uses a single-slot "keep latest" buffer that can
silently overwrite older frames if the consumer iterates slowly).

We bypass our RtspStream wrapper entirely and call cv2.VideoCapture
directly in a tight loop, capturing every frame ffmpeg hands us. For
each frame we record (fr.ts, monotonic_now). Any gap in fr.ts between
consecutive frames reveals what ffmpeg/the camera delivered:

  - Steady ~0.067s gaps at 15fps → no skipping anywhere
  - Occasional small spikes (~0.1-0.5s) → minor jitter, normal
  - Big multi-second gaps → frames truly lost upstream (camera or
    network), unrecoverable in our layer
  - Bursty arrivals (many fr.ts close in monotonic time) → consumer
    drops are possible if our wrapper consumer is slow, but THIS
    probe doesn't have that problem because it has no real consumer

Usage:
    uv run python scripts/probe_main_gaps.py [--secs 30]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Mirror the same ffmpeg options the thumb upgrader's TimestampedFrameBuffer
# now uses: TCP transport, NO nobuffer/low_delay (so ffmpeg keeps decoded-
# frame backlogs intact). This lets us measure whether the multi-second
# fr.ts gaps we saw in earlier probes were caused by those flags or by
# real upstream frame loss.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp",
)

import cv2  # noqa: E402

from camwatch.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secs", type=float, default=30.0,
                        help="seconds to probe")
    args = parser.parse_args()

    cfg = load_config()
    url = cfg.camera.rtsp_url_thumb or cfg.camera.rtsp_url
    print(f"opening: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("could not open stream")
        return 1
    # Try FORCING a large frame queue to test whether the "lost frames" are
    # ffmpeg's internal behavior or actually receivable if we ask for them.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)

    # Drain warmup.
    for _ in range(10):
        cap.read()

    samples: list[tuple[float, float]] = []  # (monotonic_now, fr_ts_seconds)
    start = time.monotonic()
    deadline = start + args.secs
    while time.monotonic() < deadline:
        ok, _img = cap.read()
        if not ok:
            continue
        pts_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        samples.append((time.monotonic(), pts_s))
    cap.release()

    if len(samples) < 2:
        print(f"only {len(samples)} samples; aborting")
        return 1

    print(f"received {len(samples)} frames in {time.monotonic() - start:.1f}s "
          f"(consumer rate {len(samples) / (time.monotonic() - start):.1f}fps)")

    # Compute fr.ts gaps between consecutive frames.
    pts_gaps = []
    mono_gaps = []
    for (mt0, p0), (mt1, p1) in zip(samples, samples[1:]):
        pts_gaps.append(p1 - p0)
        mono_gaps.append(mt1 - mt0)

    # Bucket pts gaps by size for a quick overview.
    def bucket(g: float) -> str:
        if g < 0:
            return "NEGATIVE (rare)"
        if g < 0.05:
            return "<50ms"
        if g < 0.1:
            return "50-100ms"
        if g < 0.2:
            return "100-200ms (1 frame at 15fps)"
        if g < 0.5:
            return "200-500ms"
        if g < 1.0:
            return "500-1000ms"
        if g < 5.0:
            return "1-5s"
        return ">=5s (large gap)"

    print("\nfr.ts gap distribution between consecutive frames:")
    counts = Counter(bucket(g) for g in pts_gaps)
    for label in [
        "NEGATIVE (rare)", "<50ms", "50-100ms",
        "100-200ms (1 frame at 15fps)", "200-500ms",
        "500-1000ms", "1-5s", ">=5s (large gap)",
    ]:
        n = counts.get(label, 0)
        if n:
            pct = 100.0 * n / len(pts_gaps)
            print(f"  {label:32s}  {n:5d}  ({pct:5.1f}%)")

    # Show the 5 largest gaps with surrounding context (when they happened).
    largest_idx = sorted(range(len(pts_gaps)), key=lambda i: pts_gaps[i], reverse=True)[:5]
    largest_idx.sort()
    print("\n5 largest fr.ts gaps:")
    for i in largest_idx:
        mono_at = samples[i + 1][0] - start
        print(
            f"  at +{mono_at:6.2f}s: pts_gap={pts_gaps[i]:7.3f}s "
            f"(mono_gap={mono_gaps[i]:6.3f}s, ratio={pts_gaps[i] / max(mono_gaps[i], 0.001):4.1f}x)"
        )

    print("\ninterpretation:")
    big_gaps = sum(1 for g in pts_gaps if g >= 1.0)
    if big_gaps == 0:
        print("  ✓ no multi-second gaps observed — frames are NOT being skipped")
        print("    upstream. Any gaps the upgrader sees must come from the")
        print("    consumer-side single-slot buffer overwriting frames.")
    else:
        print(f"  ⚠ {big_gaps} gaps >=1s observed — main stream IS losing")
        print("    frames somewhere upstream of our cap.read() loop.")
        print("    likely camera/network/ffmpeg internal drops; not recoverable.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
