"""DB writes for the enricher: persist embeddings + apply high-confidence labels."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .decision import Decision


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_clip_path(db_path: Path, pass_id: int) -> str | None:
    with sqlite3.connect(db_path, timeout=10) as conn:
        row = conn.execute(
            "SELECT clip_path FROM passes WHERE id = ? AND deleted = 0",
            (int(pass_id),),
        ).fetchone()
    return row[0] if row else None


def thumb_path_from_clip(clip_path: str) -> Path:
    base = clip_path[:-4] if clip_path.endswith(".mp4") else clip_path
    return Path(base + ".jpg")


def anchor_paths_from_clip(clip_path: str) -> dict[str, Path]:
    """Return {view_name: Path} for the three views camwatch writes per pass.

    Anchor JPEGs only exist for passes captured after the recorder
    anchor-image change shipped; callers should check `.exists()` before
    using each path. The 'thumb' view is the midpoint thumbnail (always
    present for any pass with a clip_path).
    """
    base = clip_path[:-4] if clip_path.endswith(".mp4") else clip_path
    return {
        "thumb": Path(base + ".jpg"),
        "entry": Path(base + ".entry.jpg"),
        "exit": Path(base + ".exit.jpg"),
    }


def upsert_embedding(
    db_path: Path,
    pass_id: int,
    vector: np.ndarray,
    model_name: str,
) -> None:
    blob = vector.astype(np.float32, copy=False).tobytes()
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.execute(
            """
            INSERT INTO pass_embeddings (pass_id, embedding, model_name, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(pass_id) DO UPDATE SET
                embedding = excluded.embedding,
                model_name = excluded.model_name,
                created_at = excluded.created_at
            """,
            (int(pass_id), blob, model_name, _now_iso()),
        )
        conn.commit()


def apply_decision(db_path: Path, pass_id: int, d: Decision) -> None:
    """Persist a decision back to the passes row.

    Local label is written to the parallel `local_*` columns so it can
    coexist with Opus's answer in `vehicle_*`. enrich_local_status +
    enrich_local_topk record the per-call diagnostics whether the label
    fired or not.
    """
    topk_json = json.dumps([
        {"pass_id": n.pass_id, "make": n.make, "model": n.model, "sim": round(n.sim, 4)}
        for n in d.topk
    ])
    with sqlite3.connect(db_path, timeout=10) as conn:
        if d.status == "high":
            conn.execute(
                """
                UPDATE passes SET
                    local_make          = ?,
                    local_model         = ?,
                    local_color         = ?,
                    local_confidence    = 'high',
                    local_enriched_at   = ?,
                    enrich_local_status = ?,
                    enrich_local_topk   = ?
                WHERE id = ?
                """,
                (d.make, d.model, d.color, _now_iso(), d.status, topk_json, int(pass_id)),
            )
        else:
            conn.execute(
                """
                UPDATE passes SET
                    enrich_local_status = ?,
                    enrich_local_topk   = ?
                WHERE id = ?
                """,
                (d.status, topk_json, int(pass_id)),
            )
        conn.commit()
