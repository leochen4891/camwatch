"""Template-based OCR for the camera's burned-in OSD timestamp.

Replaces Tesseract for the OSD-reading path. The Reolink E1 OSD uses a
fixed digital-clock-like font that doesn't change frame-to-frame, so a
small library of per-digit reference images plus normalized cross-
correlation gives near-deterministic accuracy on this specific input —
none of the digit-vs-digit confusions Tesseract suffers (8↔6, 5↔3,
0↔O, etc.).

The library is bootstrapped once per camera installation by
`scripts/collect_digit_templates.py` (collects digit slots from a single
fresh frame) plus `scripts/scan_digit_seconds.py` (sweeps the seconds-
ones digit through 0-9 over a few seconds of capture).

Typical usage:

    matcher = DigitMatcher(templates_dir="templates/main")
    dt = matcher.read_timestamp(frame_bgr, OSD_REGION)
    if dt is not None:
        ...

Returns a naive `datetime` (the OSD doesn't carry timezone info) at
1-second resolution, or `None` if the OSD couldn't be parsed.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Character layout of "MM/DD/YYYY HH:MM:SS DAY" — 23 positions. Same layout
# the collector scripts use.
_LAYOUT = "dd/dd/dddd dd:dd:dd LLL"
_DIGIT_INDICES = [i for i, c in enumerate(_LAYOUT) if c == "d"]


def _detect_char_boxes(
    osd_gray: np.ndarray, bright_thresh: int = 180
) -> list[tuple[int, int, int, int]]:
    """Find character bounding boxes in the OSD strip via connected-component
    analysis. Vertical morphological closing first merges the two dots of a
    colon into one component, so each non-space character of the timestamp
    text yields exactly one box. Sorted left-to-right."""
    h, w = osd_gray.shape
    _, binary = cv2.threshold(osd_gray, bright_thresh, 255, cv2.THRESH_BINARY)
    # Vertical kernel as tall as the strip itself ensures colon dots merge
    # into a single component on any OSD height; shorter kernels can leave
    # the dots as two separate components.
    kernel_h = max(3, h)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n_comp, _labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n_comp):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        # Real characters span ~full strip height after morph closing;
        # 1-2px-wide streaks pass height check otherwise.
        if ch < h * 0.8:
            continue
        if cw < 4:
            continue
        if cw > w * 0.15:
            continue
        boxes.append((x, y, x + cw, y + ch))
    boxes.sort(key=lambda b: b[0])
    return boxes


def _detect_text_bounds(
    osd_gray: np.ndarray, bright_thresh: int = 180
) -> tuple[int, int]:
    """Projection-based fallback: leftmost/rightmost columns with bright
    pixels. Used when CC analysis under-counts (small/low-contrast OSDs
    where some characters merge or fall below the size filter)."""
    bright = (osd_gray > bright_thresh).astype(np.uint8)
    cols = np.where(bright.sum(axis=0) >= 2)[0]
    if len(cols) == 0:
        return 0, osd_gray.shape[1]
    return int(cols[0]), int(cols[-1] + 1)


def _box_index_for_layout_pos(idx: int) -> int:
    return idx - _LAYOUT[:idx].count(" ")


# Minimum NCC score to accept a digit match. Below this, we treat the read
# as failed rather than guessing — better to retry than to seed a bad
# offset calibration.
_MIN_MATCH_SCORE = 0.5


class DigitMatcher:
    def __init__(self, templates_dir: Path | str, bright_thresh: int = 180) -> None:
        self._templates_dir = Path(templates_dir)
        self._bright_thresh = bright_thresh
        # Templates are stored as 6x-upscaled BGR PNGs. We threshold them
        # at load time to isolate the digit silhouette from the variable
        # outdoor background — without this, NCC scores key on the
        # (similar) grass texture across all templates instead of the
        # digit shape, producing high-confidence wrong matches.
        self._templates: dict[int, np.ndarray] = {}
        for d in range(10):
            p = self._templates_dir / f"digit_{d}.png"
            if not p.exists():
                raise FileNotFoundError(f"missing digit template: {p}")
            tpl_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if tpl_bgr is None:
                raise ValueError(f"could not load template: {p}")
            tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
            _, tpl_bin = cv2.threshold(
                tpl_gray, self._bright_thresh, 255, cv2.THRESH_BINARY
            )
            self._templates[d] = tpl_bin
        log.info(
            "DigitMatcher: loaded 10 templates from %s "
            "(sample size %dx%d, threshold=%d)",
            self._templates_dir, *self._templates[0].shape[::-1],
            self._bright_thresh,
        )

    def _digit_slot_x_ranges(self, osd_gray: np.ndarray) -> list[tuple[int, int]] | None:
        """Compute the (x1, x2) pixel range of each of the 14 digit slots
        in the OSD strip. Tries connected-component detection first;
        falls back to projection-based even spacing for small OSDs where
        CC under-counts."""
        boxes = _detect_char_boxes(osd_gray)
        expected_non_space = len(_LAYOUT) - _LAYOUT.count(" ")
        if len(boxes) == expected_non_space:
            ranges: list[tuple[int, int]] = []
            for idx in _DIGIT_INDICES:
                box_idx = _box_index_for_layout_pos(idx)
                bx1, _, bx2, _ = boxes[box_idx]
                ranges.append((bx1, bx2))
            return ranges
        # Fallback: use projection-detected bounds + uniform spacing.
        text_x1, text_x2 = _detect_text_bounds(osd_gray)
        if text_x2 <= text_x1:
            return None
        char_w = (text_x2 - text_x1) / len(_LAYOUT)
        ranges = []
        for idx in _DIGIT_INDICES:
            bx1 = int(round(text_x1 + idx * char_w))
            bx2 = int(round(text_x1 + (idx + 1) * char_w))
            ranges.append((bx1, bx2))
        return ranges

    def _classify_slot(self, slot_gray: np.ndarray) -> tuple[int, float]:
        """Return (digit, score) for the highest-correlating template.
        The slot is binarized at the same threshold the templates were,
        so matching keys on the digit silhouette and not on background
        texture variation. Each template is resized to the slot's
        dimensions before NCC; the result is a single-position match."""
        sh, sw = slot_gray.shape
        if sh == 0 or sw == 0:
            return -1, -1.0
        _, slot_bin = cv2.threshold(
            slot_gray, self._bright_thresh, 255, cv2.THRESH_BINARY
        )
        best_digit = -1
        best_score = -1.0
        for d, tpl in self._templates.items():
            tpl_resized = cv2.resize(
                tpl, (sw, sh), interpolation=cv2.INTER_NEAREST
            )
            result = cv2.matchTemplate(slot_bin, tpl_resized, cv2.TM_CCOEFF_NORMED)
            score = float(result[0, 0])
            if score > best_score:
                best_score = score
                best_digit = d
        return best_digit, best_score

    def read_timestamp(
        self, frame: np.ndarray, region: tuple[int, int, int, int]
    ) -> datetime | None:
        """Slide each digit template across the binarized OSD strip and
        pick the 14 strongest non-overlapping peaks. Each peak's argmax
        template gives the digit at that position. Sort by x and assemble
        the timestamp.

        This sidesteps separate slot detection: the templates themselves
        locate their own positions, which is more robust than connected-
        component or projection-based slicing — the latter can fail on
        OSDs with merged colon dots, low contrast, or unusual spacing."""
        x1, y1, x2, y2 = region
        h, w = frame.shape[:2]
        x2 = min(x2, w)
        y2 = min(y2, h)
        if x2 <= x1 or y2 <= y1:
            return None

        osd = frame[y1:y2, x1:x2]
        osd_gray = cv2.cvtColor(osd, cv2.COLOR_BGR2GRAY) if osd.ndim == 3 else osd

        # Upscale to match the resolution at which templates were captured
        # so per-digit pixel widths align between osd and templates.
        upscale = 6
        crop_h, crop_w = osd_gray.shape
        osd_big = cv2.resize(
            osd_gray, (crop_w * upscale, crop_h * upscale),
            interpolation=cv2.INTER_LANCZOS4,
        )
        _, osd_bin = cv2.threshold(
            osd_big, self._bright_thresh, 255, cv2.THRESH_BINARY
        )

        # Resize each template to the OSD's full height (templates are
        # already at the right vertical scale by construction, but sub
        # vs. main calls may differ slightly).
        target_h = osd_bin.shape[0]
        tpl_resized: dict[int, np.ndarray] = {}
        for d, tpl in self._templates.items():
            th, tw = tpl.shape
            if th != target_h:
                scale = target_h / th
                new_w = max(1, int(round(tw * scale)))
                tpl_resized[d] = cv2.resize(
                    tpl, (new_w, target_h), interpolation=cv2.INTER_NEAREST
                )
            else:
                tpl_resized[d] = tpl

        # Slide each template; combine into a single (best_digit, best_score)
        # array indexed by horizontal position.
        n_cols = osd_bin.shape[1]
        # All templates should have the same width because they came from
        # uniform-width digit slots; if they differ, take the smallest so
        # the result arrays line up.
        tpl_w = min(t.shape[1] for t in tpl_resized.values())
        tpl_resized = {d: t[:, :tpl_w] for d, t in tpl_resized.items()}
        result_w = n_cols - tpl_w + 1
        if result_w <= 0:
            return None

        score_stack = np.zeros((10, result_w), dtype=np.float32)
        for d, tpl in tpl_resized.items():
            res = cv2.matchTemplate(osd_bin, tpl, cv2.TM_CCOEFF_NORMED)
            score_stack[d, :] = res[0, :]

        best_digit = score_stack.argmax(axis=0).astype(np.int8)
        best_score = score_stack.max(axis=0)

        # Greedy non-max-suppression: pick the highest-score position, claim
        # a window of width tpl_w around it, repeat until 14 picks or no
        # remaining positions clear threshold.
        picks: list[tuple[int, int, float]] = []  # (x_left, digit, score)
        scores = best_score.copy()
        suppression_w = int(tpl_w * 0.7)
        while len(picks) < 14:
            x = int(scores.argmax())
            score = float(scores[x])
            if score < _MIN_MATCH_SCORE:
                break
            d = int(best_digit[x])
            picks.append((x, d, score))
            lo = max(0, x - suppression_w)
            hi = min(len(scores), x + suppression_w + 1)
            scores[lo:hi] = -1.0

        if len(picks) != 14:
            log.debug(
                "DigitMatcher: found only %d/14 digit peaks (best score=%.2f)",
                len(picks), max((p[2] for p in picks), default=0.0),
            )
            return None

        picks.sort(key=lambda p: p[0])
        digits = [p[1] for p in picks]

        try:
            mo = digits[0] * 10 + digits[1]
            da = digits[2] * 10 + digits[3]
            yr = (
                digits[4] * 1000 + digits[5] * 100
                + digits[6] * 10 + digits[7]
            )
            hr = digits[8] * 10 + digits[9]
            mi = digits[10] * 10 + digits[11]
            se = digits[12] * 10 + digits[13]
            return datetime(yr, mo, da, hr, mi, se)
        except (ValueError, OverflowError):
            return None
