"""Capture ~12 seconds of frames per stream and extract the seconds-ones
digit slot from each, so a single run sees every digit 0-9 and we can
fill the template library in one pass.

Approach:
  1. For each stream, open RTSP, drain warmup frames, then read frames at
     ~1-second wall-clock intervals for `n_seconds` seconds. Each captured
     frame's OSD seconds-ones digit will differ from the previous by 1.
  2. For each captured frame, run the same connected-component slot
     detection used by collect_digit_templates.py to find character boxes,
     and crop the seconds-ones slot (LAYOUT index 18, the rightmost digit
     before "DAY").
  3. Save individual slot crops + a labeled horizontal strip.
  4. The user reads off the digits in order from the strip; a follow-up
     command promotes each slot crop to templates/{stream}/digit_<X>.png.

Usage:
    uv run python scripts/scan_digit_seconds.py [--n-seconds 12]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Low-latency FFmpeg flags MUST be set before cv2 is imported. The same env
# var is applied in camwatch/capture.py for the live worker; we mirror it
# here so this script reads frames the same way (no deep buffering).
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from camwatch.config import load_config  # noqa: E402

# Re-import the helpers from the sibling collector so we share slot logic.
sys.path.insert(0, str(ROOT / "scripts"))
from collect_digit_templates import (  # noqa: E402
    LAYOUT,
    OSD_REGION_MAIN,
    OSD_REGION_SUB,
    _box_index_for_layout_pos,
    _detect_char_boxes,
)

OUT_ROOT = Path("/tmp/digit_templates")

# LAYOUT index 18 = seconds-ones digit (the last 'd' before " LLL").
SECONDS_ONES_LAYOUT_IDX = 18


def _open_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {url}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _capture_at_intervals(
    cap: cv2.VideoCapture,
    region: tuple[int, int, int, int],
    n_seconds: int,
    interval_s: float,
) -> list[np.ndarray]:
    """Capture one good frame per `interval_s` of wall-clock time. A "good"
    frame is one whose OSD region has plenty of bright pixels — guards
    against the occasional partial-decode garbage frame."""
    x1, y1, x2, y2 = region
    # Drain setup junk.
    for _ in range(40):
        cap.read()

    frames: list[np.ndarray] = []
    deadline = time.monotonic() + n_seconds + 1.0
    next_capture_at = time.monotonic()
    pending_best: tuple[float, np.ndarray] | None = None  # (score, frame)

    while time.monotonic() < deadline and len(frames) < n_seconds + 1:
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        rx2 = min(x2, w)
        ry2 = min(y2, h)
        crop = f[y1:ry2, x1:rx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        score = float((gray > 200).sum())

        # Track the highest-score frame seen since the last capture point.
        if pending_best is None or score > pending_best[0]:
            pending_best = (score, f.copy())

        # When we cross the next 1-second boundary, commit the best-scoring
        # frame from the last interval.
        if time.monotonic() >= next_capture_at and pending_best is not None:
            frames.append(pending_best[1])
            pending_best = None
            next_capture_at += interval_s
    return frames


def _seconds_ones_slot_via_projection(
    osd_gray: np.ndarray, layout: str
) -> tuple[int, int] | None:
    """Fallback slot detector for streams where the connected-component
    approach doesn't recover the expected box count (small/low-contrast
    OSDs). Detects the leftmost/rightmost bright columns, then divides
    that span uniformly by `len(layout)` characters."""
    bright = (osd_gray > 180).astype(np.uint8)
    cols = np.where(bright.sum(axis=0) >= 2)[0]
    if len(cols) == 0:
        return None
    text_x1, text_x2 = int(cols[0]), int(cols[-1] + 1)
    char_w = (text_x2 - text_x1) / len(layout)
    bx1 = int(round(text_x1 + SECONDS_ONES_LAYOUT_IDX * char_w))
    bx2 = int(round(text_x1 + (SECONDS_ONES_LAYOUT_IDX + 1) * char_w))
    return bx1, bx2


def _extract_seconds_ones_slot(
    frame: np.ndarray, region: tuple[int, int, int, int]
) -> np.ndarray | None:
    """Crop the OSD strip and return the seconds-ones digit slot, upscaled
    6x for visual clarity. Tries connected-component detection first; if
    the box count is wrong (common on the sub stream's 25-pixel-tall OSD),
    falls back to projection-based even-spacing slicing."""
    x1, y1, x2, y2 = region
    h, w = frame.shape[:2]
    x2 = min(x2, w)
    y2 = min(y2, h)
    osd = frame[y1:y2, x1:x2]
    crop_h, crop_w = osd.shape[:2]
    osd_gray = cv2.cvtColor(osd, cv2.COLOR_BGR2GRAY)

    boxes = _detect_char_boxes(osd_gray)
    expected = len(LAYOUT) - LAYOUT.count(" ")
    if len(boxes) == expected:
        box_idx = _box_index_for_layout_pos(LAYOUT, SECONDS_ONES_LAYOUT_IDX)
        if box_idx >= len(boxes):
            return None
        bx1, _, bx2, _ = boxes[box_idx]
    else:
        proj = _seconds_ones_slot_via_projection(osd_gray, LAYOUT)
        if proj is None:
            return None
        bx1, bx2 = proj

    pad = max(1, (bx2 - bx1) // 6)
    bx1p = max(0, bx1 - pad)
    bx2p = min(crop_w, bx2 + pad)
    upscale = 6
    osd_big = cv2.resize(
        osd, (crop_w * upscale, crop_h * upscale), interpolation=cv2.INTER_LANCZOS4
    )
    return osd_big[:, bx1p * upscale:bx2p * upscale]


def _build_strip(slots: list[np.ndarray], indices: list[int]) -> np.ndarray:
    """Horizontal labeled strip of all captured seconds-ones slots."""
    target_h = max(s.shape[0] for s in slots)
    label_h = 30
    cells: list[np.ndarray] = []
    for slot, idx in zip(slots, indices):
        h, w = slot.shape[:2]
        cell = np.full(
            (target_h + label_h, max(w, 60), 3), 32, dtype=np.uint8
        )
        cy = label_h + (target_h - h) // 2
        cx = (cell.shape[1] - w) // 2
        cell[cy:cy + h, cx:cx + w] = slot
        cv2.putText(
            cell, str(idx), (cell.shape[1] // 2 - 8, label_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )
        cells.append(cell)
    return np.hstack(cells)


def scan_stream(
    stream_name: str, url: str, region: tuple[int, int, int, int],
    n_seconds: int, interval: float = 1.0,
) -> None:
    out_dir = OUT_ROOT / f"{stream_name}_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any previous scan of this stream so file-name index = capture order.
    for old in out_dir.glob("scan_*.png"):
        old.unlink()
    for old in out_dir.glob("strip*.png"):
        old.unlink()

    print(f"\n[{stream_name}] capturing {n_seconds}s of frames at {interval}s intervals…")
    cap = _open_stream(url)
    try:
        frames = _capture_at_intervals(cap, region, n_seconds, interval)
    finally:
        cap.release()
    print(f"[{stream_name}] got {len(frames)} frames")

    slots: list[np.ndarray] = []
    indices: list[int] = []
    for i, fr in enumerate(frames):
        slot = _extract_seconds_ones_slot(fr, region)
        if slot is None:
            print(f"  [{stream_name}] frame {i}: slot extraction failed (skipped)")
            continue
        slots.append(slot)
        indices.append(i)
        cv2.imwrite(str(out_dir / f"scan_{i:02d}.png"), slot)

    if not slots:
        print(f"[{stream_name}] no usable slots extracted — bailing")
        return

    strip = _build_strip(slots, indices)
    strip_path = out_dir / "strip.png"
    cv2.imwrite(str(strip_path), strip)
    print(f"[{stream_name}] {len(slots)} slots saved to {out_dir}")
    print(f"[{stream_name}] composite strip -> {strip_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seconds", type=int, default=12,
                        help="seconds of capture per stream (default 12)")
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="seconds between captures (default 1.0). Use 0.7 to break "
             "lock-step with cameras that deliver at a steady ~1Hz, so "
             "consecutive captures hit different OSD seconds.",
    )
    args = parser.parse_args()

    cfg = load_config()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    scan_stream("sub", cfg.camera.rtsp_url, OSD_REGION_SUB,
                args.n_seconds, args.interval)
    if cfg.camera.rtsp_url_thumb:
        scan_stream("main", cfg.camera.rtsp_url_thumb, OSD_REGION_MAIN,
                    args.n_seconds, args.interval)
    else:
        print("\nno main-stream URL configured (camera.path_thumb)")

    print(f"\nReview the strips in {OUT_ROOT}/{{sub,main}}_scan/strip.png "
          "and report the digit visible in each numbered slot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
