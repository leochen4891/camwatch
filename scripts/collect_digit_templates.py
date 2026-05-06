"""Manually bootstrap an OCR digit-template library for the OSD timestamp.

The Reolink E1 burns "MM/DD/YYYY HH:MM:SS DAY" into every frame. Tesseract
reads it well most of the time but occasionally misreads visually-similar
digits (8↔6, 5↔3, etc.), and a single wrong digit can corrupt timestamp-
dependent logic for the entire session. The fix is to replace the general-
purpose OCR with a fixed-font template matcher: 10 small reference images,
one per digit, and per-frame normalized cross-correlation against each slot.

This script collects the raw material for that template library. It:

  1. Connects to both RTSP streams (sub and main) for one frame each.
  2. Crops each stream's OSD region.
  3. Splits the crop into per-character slots using the known character
     positions of the timestamp string (digits + separators + day-of-week).
  4. Runs Tesseract on each digit slot to *propose* a label.
  5. Saves to /tmp/digit_templates/{sub,main}/:
        osd_full.png         — the full OSD strip
        slot_<i>_pred_<X>.png — each digit slot, with Tesseract's guess in name
        review.png           — composite of all slots side by side, labeled

The user reviews review.png and confirms or corrects the labels. A second
step (a separate, simpler script) then copies the confirmed slot images to
templates/{sub,main}/digit_<X>.png to become the actual template library.

Usage:
    uv run python scripts/collect_digit_templates.py

Notes:
  - Spacing assumes a fixed-width font (true for the Reolink OSD).
  - One frame won't show every digit 0-9. Run the script several times at
    different times of day, or use the seconds digit's natural rollover to
    accumulate a full set across runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# capture.py sets the FFmpeg env var on import; keep that ordering.
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytesseract  # noqa: E402

from camwatch.config import load_config  # noqa: E402

OUT_ROOT = Path("/tmp/digit_templates")

# OSD region (x1, y1, x2, y2) for each stream — same regions used elsewhere
# in the codebase. The Reolink E1 puts the OSD at the bottom of the frame.
OSD_REGION_SUB = (175, 452, 500, 477)
# Main is 2048x1536 (4:3). Auto-detection of bright-pixel runs in the
# bottom band shows OSD timestamp at x=657..1360 with text height ~40px
# vertically centered around y=1493. The "corner" caption is isolated
# to x=1849+ and excluded.
OSD_REGION_MAIN = (650, 1469, 1370, 1517)

# Character layout of "MM/DD/YYYY HH:MM:SS DAY" — 23 positions.
# 'd' = digit slot we want a template of; '/', ':', ' ' = skip; 'L' = letter
# (day-of-week, not relevant for templates).
LAYOUT = "dd/dd/dddd dd:dd:dd LLL"
DIGIT_INDICES = [i for i, c in enumerate(LAYOUT) if c == "d"]


def _detect_char_boxes(osd_gray: np.ndarray, bright_thresh: int = 180) -> list[tuple[int, int, int, int]]:
    """Find character bounding boxes in the OSD strip via connected-component
    analysis. Returns boxes sorted by left-x.

    Vertical morphological closing first merges the two dots of a colon
    into one component, so each non-space character of the timestamp text
    yields exactly one box. Filters out small/short components (noise) and
    very wide ones (background blobs that survive thresholding)."""
    h, w = osd_gray.shape
    _, binary = cv2.threshold(osd_gray, bright_thresh, 255, cv2.THRESH_BINARY)
    # Vertical kernel as tall as the strip itself ensures colon dots merge
    # into a single component on any OSD height. Anything shorter risks
    # leaving a gap between the two dots when the strip is short and the
    # gap between dots is proportionally large.
    kernel_h = max(3, h)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n_comp, _labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n_comp):  # 0 is background
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        # Real characters span essentially the whole strip vertically
        # (after morph closing). Anything shorter is a partial-height
        # noise streak that survived thresholding.
        if ch < h * 0.8:
            continue
        # Filter narrow streaks (width=1-2 pixels passing height check).
        if cw < 4:
            continue
        # Stray giant blobs (e.g. background patches that crossed threshold).
        if cw > w * 0.15:
            continue
        boxes.append((x, y, x + cw, y + ch))
    boxes.sort(key=lambda b: b[0])
    return boxes


# Box index for each LAYOUT character position — boxes correspond to non-
# space characters in left-to-right order, so the box for LAYOUT index `i`
# is at box index `i - (number of spaces before i)`.
def _box_index_for_layout_pos(layout: str, idx: int) -> int:
    return idx - layout[:idx].count(" ")


def _ocr_slot(img: np.ndarray) -> str:
    """Tesseract a single character. Returns the first character of the
    OCR output stripped, or '?' if nothing came back."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # PSM 10 = "single character"
    config = "--psm 10 -c tessedit_char_whitelist=0123456789"
    raw = pytesseract.image_to_string(gray, config=config).strip()
    return raw[0] if raw else "?"


def _composite(slot_imgs: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """Build a horizontal strip showing each slot with its predicted label
    rendered above it. All slots get padded to the tallest slot's height."""
    target_h = max(s.shape[0] for s in slot_imgs)
    label_h = 30
    cells: list[np.ndarray] = []
    for img, label in zip(slot_imgs, labels):
        h, w = img.shape[:2]
        cell = np.full((target_h + label_h, max(w, 60), 3), 32, dtype=np.uint8)
        # center the slot image vertically below the label band
        cy = label_h + (target_h - h) // 2
        cx = (cell.shape[1] - w) // 2
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img
        cell[cy:cy + h, cx:cx + w] = img_bgr
        cv2.putText(
            cell, label, (cell.shape[1] // 2 - 8, label_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cells.append(cell)
    return np.hstack(cells)


def _grab_frame(url: str, region: tuple[int, int, int, int]) -> np.ndarray | None:
    """Open the stream and read frames until one with a high-contrast OSD
    region lands — guards against the partial / corrupted frames that arrive
    during the initial keyframe wait on the main stream."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    x1, y1, x2, y2 = region
    best_frame: np.ndarray | None = None
    best_score = 0.0
    # Read up to 80 frames; keep the one whose OSD region has the most
    # bright pixels (i.e., the OSD text is visible and not occluded by
    # h264 garbage).
    for _ in range(80):
        ok, f = cap.read()
        if not ok or f is None:
            continue
        h, w = f.shape[:2]
        rx2 = min(x2, w)
        ry2 = min(y2, h)
        if rx2 <= x1 or ry2 <= y1:
            continue
        crop = f[y1:ry2, x1:rx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # "score" = number of bright pixels — high when OSD digits are
        # cleanly rendered, low for green grass or partially-decoded frames.
        score = float((gray > 200).sum())
        if score > best_score:
            best_score = score
            best_frame = f
    cap.release()
    return best_frame


def collect(stream_name: str, url: str, region: tuple[int, int, int, int]) -> None:
    out_dir = OUT_ROOT / stream_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{stream_name}] grabbing one frame from {url[:48]}...")
    frame = _grab_frame(url, region)
    if frame is None:
        print(f"[{stream_name}] FAILED to read a frame")
        return

    h_full, w_full = frame.shape[:2]
    x1, y1, x2, y2 = region
    x2 = min(x2, w_full)
    y2 = min(y2, h_full)
    if x2 <= x1 or y2 <= y1:
        print(f"[{stream_name}] OSD region {region} is outside frame {w_full}x{h_full}")
        return

    osd = frame[y1:y2, x1:x2].copy()
    crop_h, crop_w = osd.shape[:2]
    cv2.imwrite(str(out_dir / "osd_full.png"), osd)
    print(f"[{stream_name}] OSD strip: {crop_w}x{crop_h}  -> {out_dir / 'osd_full.png'}")

    # Find character bounding boxes via connected components — robust to
    # the OSD being narrower than the crop and to neighbouring grass
    # reflections that fool a simple horizontal-projection bounds detection.
    osd_gray = cv2.cvtColor(osd, cv2.COLOR_BGR2GRAY)
    boxes = _detect_char_boxes(osd_gray)
    expected_non_space = len(LAYOUT) - LAYOUT.count(" ")
    print(f"[{stream_name}] found {len(boxes)} character boxes "
          f"(expected {expected_non_space} for layout {LAYOUT!r})")
    if len(boxes) != expected_non_space:
        print(f"[{stream_name}] WARNING: box count mismatch — slot mapping "
              "may be wrong. Check osd_full.png and threshold.")

    # Upscale for visual inspection.
    upscale = 6
    osd_big = cv2.resize(
        osd, (crop_w * upscale, crop_h * upscale), interpolation=cv2.INTER_LANCZOS4
    )
    cv2.imwrite(str(out_dir / "osd_full_6x.png"), osd_big)

    slot_imgs: list[np.ndarray] = []
    labels: list[str] = []
    for digit_pos, idx in enumerate(DIGIT_INDICES):
        box_idx = _box_index_for_layout_pos(LAYOUT, idx)
        if box_idx >= len(boxes):
            slot_imgs.append(np.zeros((20, 20, 3), dtype=np.uint8))
            labels.append("?")
            continue
        bx1, _, bx2, _ = boxes[box_idx]
        # Pad each side by ~10% of the character width before upscaling for
        # a nicer visual review crop.
        pad = max(1, (bx2 - bx1) // 6)
        bx1p = max(0, bx1 - pad)
        bx2p = min(crop_w, bx2 + pad)
        slot = osd_big[:, bx1p * upscale:bx2p * upscale]
        pred = _ocr_slot(slot)
        slot_imgs.append(slot)
        labels.append(pred)
        out_path = out_dir / f"slot{digit_pos:02d}_idx{idx:02d}_pred_{pred}.png"
        cv2.imwrite(str(out_path), slot)

    review = _composite(slot_imgs, labels)
    cv2.imwrite(str(out_dir / "review.png"), review)

    print(f"[{stream_name}] proposed labels (slot order): {''.join(labels)}")
    print(f"[{stream_name}] composite -> {out_dir / 'review.png'}")
    print(f"[{stream_name}] expected pattern:             dd  dddd  dd dd dd")


def main() -> int:
    cfg = load_config()
    sub_url = cfg.camera.rtsp_url
    main_url = cfg.camera.rtsp_url_thumb
    if main_url is None:
        print("no main-stream URL configured (camera.path_thumb)")
        return 1
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    collect("sub", sub_url, OSD_REGION_SUB)
    collect("main", main_url, OSD_REGION_MAIN)
    print(f"\nReview the per-slot crops + review.png in {OUT_ROOT} and "
          "confirm/correct the labels.")
    return 0


if __name__ == "__main__":
    # capture.py installs ffmpeg low-latency env vars on import; do the same
    # here so the streams open with the same settings as the live worker.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
    )
    sys.exit(main())
