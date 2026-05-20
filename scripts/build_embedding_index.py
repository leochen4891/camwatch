#!/usr/bin/env python3
"""Backfill pass_embeddings + apply high-confidence local labels.

Idempotent. Safe to run any time the enricher has been offline for a while
and rows piled up without embeddings.

Steps:
  1. Encode every pass that has a .jpg on disk but no row in pass_embeddings.
  2. Refresh the in-memory index from labeled rows.
  3. For every vehicle_enriched_at IS NULL pass, run KNN and apply the
     high-confidence labels. Low-confidence rows are left for the existing
     Opus workflow.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from camwatch_enricher.config import load_config
from camwatch_enricher.decision import decide
from camwatch_enricher.embedder import Embedder
from camwatch_enricher.index import KnnIndex
from camwatch_enricher.store import apply_decision, thumb_path_from_clip, upsert_embedding


def _passes_needing_embedding(db_path: Path, model_name: str) -> list[tuple[int, str]]:
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT p.id, p.clip_path
            FROM passes p
            LEFT JOIN pass_embeddings pe
                ON pe.pass_id = p.id AND pe.model_name = ?
            WHERE p.deleted = 0
              AND p.clip_path IS NOT NULL
              AND pe.pass_id IS NULL
            ORDER BY p.id
            """,
            (model_name,),
        )
        return [(r["id"], r["clip_path"]) for r in cur]


def _passes_needing_local_label(db_path: Path, relabel_all: bool) -> list[int]:
    """Return ids of passes that need a local_* label attempt.

    By default: passes that have a clip_path but no local_make yet. This
    covers both fresh passes and historical Opus-labeled passes whose
    local_* is empty (useful for backtest: every Opus-labeled pass becomes
    a ground-truth comparison point once local has weighed in).

    With relabel_all=True, re-run on every pass with a clip_path, even
    those already locally labeled. Useful after threshold changes.
    """
    where = "deleted = 0 AND clip_path IS NOT NULL"
    if not relabel_all:
        where += " AND local_make IS NULL"
    with sqlite3.connect(db_path, timeout=10) as conn:
        cur = conn.execute(f"SELECT id FROM passes WHERE {where} ORDER BY id")
        return [int(r[0]) for r in cur]


def _load_embedding(db_path: Path, pass_id: int, model_name: str):
    with sqlite3.connect(db_path, timeout=10) as conn:
        row = conn.execute(
            "SELECT embedding FROM pass_embeddings WHERE pass_id = ? AND model_name = ?",
            (pass_id, model_name),
        ).fetchone()
    if not row:
        return None
    import numpy as np
    return np.frombuffer(row[0], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/enricher.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="encode + decide but don't write to DB")
    ap.add_argument("--encode-only", action="store_true",
                    help="just compute missing embeddings, skip the relabel pass")
    ap.add_argument("--relabel-all", action="store_true",
                    help="re-run local labeling for every pass, including those that "
                         "already have a local_* label (useful after threshold tuning)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap how many passes to encode this run (0 = no cap)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    db_path = Path(cfg.paths.db)
    if not db_path.exists():
        print(f"db not found: {db_path}", file=sys.stderr)
        return 1

    embedder = Embedder(model_name=cfg.model.name, device=cfg.model.device)

    pending = _passes_needing_embedding(db_path, cfg.model.name)
    if args.limit > 0:
        pending = pending[: args.limit]
    print(f"encoding {len(pending)} thumbnails ({cfg.model.name})")

    encoded = 0
    missing_thumb = 0
    failed = 0
    for pid, clip_path in pending:
        thumb = thumb_path_from_clip(clip_path)
        if not thumb.exists():
            missing_thumb += 1
            continue
        try:
            vec = embedder.encode_path(thumb)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  encode failed pass={pid}: {e}", file=sys.stderr)
            continue
        if not args.dry_run:
            upsert_embedding(db_path, pid, vec, embedder.model_name)
        encoded += 1
        if encoded % 100 == 0:
            print(f"  encoded {encoded}/{len(pending)}")
    print(f"encoded={encoded} missing_thumb={missing_thumb} failed={failed}")

    if args.encode_only:
        return 0

    # Relabel pass: query the index for every still-unlabeled pass.
    index = KnnIndex(db_path=db_path, model_name=cfg.model.name)
    print(f"index size: {index.size()} labeled embeddings")

    pending = _passes_needing_local_label(db_path, relabel_all=args.relabel_all)
    mode = "all passes" if args.relabel_all else "passes without local_make"
    print(f"local-labeling {len(pending)} {mode}")

    high = low = no_emb = 0
    for pid in pending:
        vec = _load_embedding(db_path, pid, cfg.model.name)
        if vec is None:
            no_emb += 1
            continue
        neighbors = index.topk(vec, k=cfg.decision.k, exclude_pass_id=pid)
        # Backfill only uses the high tier (single-view, no per-pass anchor
        # combining) since labeled embeddings in the index are thumb-only.
        d = decide(
            neighbors,
            k=cfg.decision.k,
            min_votes_high=cfg.decision.min_votes_high, tau_high=cfg.decision.tau_high,
        )
        if d.status == "high":
            high += 1
        else:
            low += 1
        if not args.dry_run:
            apply_decision(db_path, pid, d)

    print(f"relabel: high={high} low/no_match={low} skipped_no_embedding={no_emb}")
    if args.dry_run:
        print("(dry-run: nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
