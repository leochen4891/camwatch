"""Read the burned-in OSD timestamp from a Reolink frame.

The camera's OSD shows `MM/DD/YYYY HH:MM:SS DAY` in 1-second resolution.
With the OSD positioned over a relatively uniform background (e.g. lawn),
a simple bright-pixel threshold + Tesseract reads it reliably (~50-70 ms).

This module exposes a `read_timestamp(frame, region)` helper that returns
either a `datetime` (with second precision) or None if the frame couldn't
be parsed. The region is configured per stream because the OSD's pixel
position depends on the camera's resolution.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np
import pytesseract

log = logging.getLogger(__name__)

# Format Reolink stamps: "MM/DD/YYYY HH:MM:SS DAY" — 24h clock.
# Tesseract sometimes drops separators or hallucinates an extra digit before
# the year (e.g. "05/05/7202619:11:45"), so the pattern is intentionally
# tolerant: between the day and the year we accept up to one stray digit, and
# between time fields any non-digit separator (or none).
_TS_RE = re.compile(
    r"(\d{2})/?(\d{2})/?\d?(\d{4})\D*(\d{2})\D?(\d{2})\D?(\d{2})"
)

_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789:/MONTUEWDFRISA"

# Tesseract regularly confuses digit ↔ letter on the OSD's small font.
# `O` for `0` is by far the most common (the day-of-week strings like MON/TUE
# train Tesseract to expect Os near digits). `I`/`l` for `1`, `S` for `5`,
# and `B` for `8` show up at lower thresholds. We collapse these before
# regex parsing — the substitution can't hurt the date/time portion (it
# only contains digits), and it can damage the day-of-week (e.g. MON → M0N)
# but we don't read the day-of-week anyway.
_DIGIT_FIXUPS = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})


def read_timestamp(
    frame: np.ndarray,
    region: Tuple[int, int, int, int],
    threshold: int | None = None,
) -> datetime | None:
    """Crop `region` (x1, y1, x2, y2), threshold, OCR, and parse the timestamp.

    Lighting varies (sun position, golden hour, etc.) so a single threshold
    isn't reliable. We try a small sequence and accept the first parseable
    result. If all fail, returns None.

    Returns a naive datetime (no tzinfo) — the camera's OSD doesn't carry one.
    """
    x1, y1, x2, y2 = region
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    thresholds = [threshold] if threshold is not None else [200, 170, 230, 140]
    last_raw = ""
    for t in thresholds:
        _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        inv = cv2.bitwise_not(th)
        raw = pytesseract.image_to_string(inv, config=_OCR_CONFIG).strip()
        last_raw = raw
        cleaned = raw.replace(" ", "").translate(_DIGIT_FIXUPS)
        m = _TS_RE.search(cleaned)
        if not m:
            continue
        mo, da, yr, h, mi, se = (int(g) for g in m.groups())
        try:
            return datetime(yr, mo, da, h, mi, se)
        except ValueError:
            continue
    log.debug("ts read failed across thresholds; last raw=%r", last_raw)
    return None
