#!/usr/bin/env python3
"""Backtest report: how often does local_* agree with vehicle_* (Opus)?

Operates on rows where both columns are populated — i.e., the local
enricher has weighed in AND Opus has too. Surfaces overall agreement
rate, agreement by Opus-make, and the worst confusions (local says X,
Opus says Y, count).

Usage:
  python scripts/compare_local_vs_opus.py [--db camwatch.db] [--top 20]
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="camwatch.db")
    ap.add_argument("--top", type=int, default=20,
                    help="how many top confusions / per-make rows to print")
    ap.add_argument("--hour-min", type=int, default=None,
                    help="only include passes captured at-or-after this local hour (0-23)")
    ap.add_argument("--hour-max", type=int, default=None,
                    help="only include passes captured strictly-before this local hour (0-24)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"db not found: {db_path}")
        return 1

    # captured_at format: '2026-05-20T15:56:52-04:00'. Hour is chars 11-13.
    hour_clause = ""
    if args.hour_min is not None:
        hour_clause += f" AND CAST(substr(captured_at, 12, 2) AS INTEGER) >= {int(args.hour_min)}"
    if args.hour_max is not None:
        hour_clause += f" AND CAST(substr(captured_at, 12, 2) AS INTEGER) < {int(args.hour_max)}"

    with sqlite3.connect(db_path) as conn:
        rows = list(conn.execute(f"""
            SELECT vehicle_make, vehicle_model, vehicle_color,
                   local_make,   local_model,   local_color,
                   local_confidence
            FROM passes
            WHERE deleted = 0
              AND vehicle_make IS NOT NULL
              AND local_make IS NOT NULL
              {hour_clause}
        """))

    if not rows:
        print("no rows have both vehicle_* and local_* set — run "
              "scripts/build_embedding_index.py first to populate local_*")
        return 0

    total = len(rows)
    make_match = sum(1 for r in rows if r[0] == r[3])
    model_match = sum(1 for r in rows if r[0] == r[3] and r[1] == r[4])
    color_match = sum(1 for r in rows if r[2] == r[5])

    print(f"backtest: local vs Opus on {total} passes where both are set")
    print()
    print(f"  make agreement:        {make_match}/{total} = {make_match/total*100:.1f}%")
    print(f"  make+model agreement:  {model_match}/{total} = {model_match/total*100:.1f}%")
    print(f"  color agreement:       {color_match}/{total} = {color_match/total*100:.1f}%")

    # Per-Opus-make breakdown.
    per_make: dict[str, list[int]] = {}
    for v_mk, v_md, _, l_mk, l_md, _, _ in rows:
        bucket = per_make.setdefault(v_mk, [0, 0])
        bucket[0] += 1
        if v_mk == l_mk and v_md == l_md:
            bucket[1] += 1

    print()
    print(f"per-Opus-make make+model agreement (top {args.top} by count):")
    sorted_makes = sorted(per_make.items(), key=lambda kv: -kv[1][0])[:args.top]
    print(f"  {'make':<20}  {'count':>5}  {'match':>5}  {'agree%':>6}")
    for mk, (n, m) in sorted_makes:
        pct = m / n * 100 if n else 0.0
        print(f"  {mk:<20}  {n:5d}  {m:5d}  {pct:5.1f}%")

    # Top confusions: (Opus_label) -> (local_label) when they differ.
    confusions: Counter[tuple[str, str, str, str]] = Counter()
    for v_mk, v_md, _, l_mk, l_md, _, _ in rows:
        if (v_mk, v_md) != (l_mk, l_md):
            confusions[(v_mk or "?", v_md or "?", l_mk or "?", l_md or "?")] += 1

    print()
    print(f"top {args.top} confusions (Opus -> local, when they differ):")
    print(f"  {'count':>5}  Opus                     ->  local")
    for (v_mk, v_md, l_mk, l_md), c in confusions.most_common(args.top):
        print(f"  {c:5d}  {v_mk + ' ' + v_md:<24}  ->  {l_mk} {l_md}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
