#!/usr/bin/env python3
"""Leave-one-out calibration of the local enricher's decision rule.

For every labeled pass (vehicle_make IS NOT NULL), drop it from the index,
run KNN against the rest, and ask the decision rule what it would have
labeled. Sweep tau_high and k_agree to find the operating point that gives
the highest recall at the precision floor we care about.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from camwatch_enricher.config import load_config
from camwatch_enricher.decision import decide
from camwatch_enricher.index import KnnIndex, Neighbor


def _labeled_rows(db_path: Path, model_name: str):
    """(pass_id, make, model, embedding) for rows that are labeled AND embedded."""
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT p.id, p.vehicle_make, p.vehicle_model, pe.embedding
            FROM passes p
            JOIN pass_embeddings pe ON pe.pass_id = p.id AND pe.model_name = ?
            WHERE p.deleted = 0
              AND p.vehicle_enriched_at IS NOT NULL
              AND p.vehicle_make IS NOT NULL
              AND p.vehicle_model IS NOT NULL
            """,
            (model_name,),
        )
        rows = []
        for r in cur:
            rows.append((
                int(r["id"]),
                str(r["vehicle_make"]),
                str(r["vehicle_model"]),
                np.frombuffer(r["embedding"], dtype=np.float32),
            ))
        return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/enricher.yaml")
    ap.add_argument("--taus", default="0.75,0.80,0.85,0.90")
    ap.add_argument("--ks", default="2,3,4")
    args = ap.parse_args()

    cfg = load_config(args.config)
    db_path = Path(cfg.paths.db)

    rows = _labeled_rows(db_path, cfg.model.name)
    print(f"labeled+embedded rows: {len(rows)}")
    if len(rows) < 10:
        print("not enough labels to calibrate", file=sys.stderr)
        return 1

    # Build the full index once; we'll exclude one pass at a time via topk's
    # exclude_pass_id arg, which is exactly what LOO needs.
    index = KnnIndex(db_path=db_path, model_name=cfg.model.name)
    print(f"index size: {index.size()}")

    taus = [float(x) for x in args.taus.split(",")]
    ks = [int(x) for x in args.ks.split(",")]

    print()
    print(f"{'tau':>5}  {'k':>2}  {'preds':>5}  {'correct':>7}  {'precision':>9}  {'recall':>6}")
    best_at_precision: dict[float, tuple[float, int, float, float]] = {}
    for tau in taus:
        for k_agree in ks:
            preds = 0
            correct = 0
            for pid, true_make, true_model, vec in rows:
                neighbors = index.topk(vec, k=cfg.decision.k, exclude_pass_id=pid)
                d = decide(neighbors, k_agree_high=k_agree, tau_high=tau)
                if d.status == "high":
                    preds += 1
                    if d.make == true_make and d.model == true_model:
                        correct += 1
            precision = (correct / preds) if preds > 0 else 0.0
            recall = preds / len(rows)
            print(f"{tau:5.2f}  {k_agree:2d}  {preds:5d}  {correct:7d}  {precision:9.3f}  {recall:6.3f}")
            # Track the highest-recall config that hits each precision floor.
            for floor in (0.90, 0.95, 0.98, 1.00):
                if precision >= floor and recall >= best_at_precision.get(floor, (None, None, None, -1.0))[3]:
                    best_at_precision[floor] = (tau, k_agree, precision, recall)

    print()
    print("best operating point per precision floor:")
    for floor in sorted(best_at_precision):
        tau, k_agree, p, r = best_at_precision[floor]
        print(f"  precision>={floor:.2f}: tau={tau} k_agree={k_agree} precision={p:.3f} recall={r:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
