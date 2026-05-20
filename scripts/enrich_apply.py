#!/usr/bin/env python3
"""Apply vehicle-enrichment JSON to camwatch.db.

Reads a JSON array from stdin where each item is:
    {"pass_id": int, "make": str|null, "model": str|null,
     "year_range": str|null, "color": str|null,
     "confidence": "high"|"medium"|"low", "note": str|null}

Writes vehicle_make/model/year_range/color/confidence/enriched_at on each row.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

COLORS = {"light", "grey", "dark", "red", "blue", "green", "brown", "yellow"}
CONFIDENCES = {"high", "medium", "low"}


def normalize(item: dict) -> dict:
    color = item.get("color")
    if color is not None and color not in COLORS:
        color = None
    conf = item.get("confidence")
    if conf not in CONFIDENCES:
        conf = "low"
    return {
        "pass_id": int(item["pass_id"]),
        "make": item.get("make") or None,
        "model": item.get("model") or None,
        "year_range": item.get("year_range") or None,
        "color": color,
        "confidence": conf,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="camwatch.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw = sys.stdin.read().strip()
    if not raw:
        print("no input", file=sys.stderr)
        return 1
    data = json.loads(raw)
    if not isinstance(data, list):
        print("expected JSON array", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db = sqlite3.connect(args.db)
    updated = 0
    for raw_item in data:
        item = normalize(raw_item)
        if args.dry_run:
            print(item)
            continue
        cur = db.execute(
            """UPDATE passes
                  SET vehicle_make = ?,
                      vehicle_model = ?,
                      vehicle_year_range = ?,
                      vehicle_color = ?,
                      vehicle_confidence = ?,
                      vehicle_enriched_at = ?,
                      vehicle_enriched_by = 'opus'
                WHERE id = ?""",
            (item["make"], item["model"], item["year_range"],
             item["color"], item["confidence"], now, item["pass_id"]),
        )
        updated += cur.rowcount
    if not args.dry_run:
        db.commit()
    db.close()
    print(f"updated {updated}/{len(data)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
