"""In-memory KNN over labeled pass embeddings.

Loads embeddings + (make, model) labels from the DB and answers
cosine-similarity top-K queries. Numpy matmul is sufficient at our scale
(a few thousand labels at most); no FAISS dependency.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Neighbor:
    pass_id: int
    make: str | None
    model: str | None
    color: str | None
    sim: float


class KnnIndex:
    """Read-only matrix of labeled embeddings + parallel metadata.

    All entries have a non-NULL `vehicle_make` AND `vehicle_model`. Rows
    that Opus tagged as null (false positives, animals, etc.) are skipped
    so they can't anchor a confident match.
    """

    def __init__(self, db_path: Path, model_name: str) -> None:
        self.db_path = Path(db_path)
        self.model_name = model_name
        self._lock = threading.RLock()
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._meta: list[Neighbor] = []
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            rows: list[tuple[int, bytes, str | None, str | None, str | None]] = []
            conn = sqlite3.connect(self.db_path, timeout=10)
            try:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    """
                    SELECT pe.pass_id, pe.embedding,
                           p.vehicle_make, p.vehicle_model, p.vehicle_color
                    FROM pass_embeddings pe
                    JOIN passes p ON p.id = pe.pass_id
                    WHERE pe.model_name = ?
                      AND p.deleted = 0
                      AND p.vehicle_enriched_at IS NOT NULL
                      AND p.vehicle_make IS NOT NULL
                      AND p.vehicle_model IS NOT NULL
                    """,
                    (self.model_name,),
                )
                for r in cur:
                    rows.append((r["pass_id"], r["embedding"], r["vehicle_make"],
                                 r["vehicle_model"], r["vehicle_color"]))
            finally:
                conn.close()

            if not rows:
                self._matrix = np.zeros((0, 0), dtype=np.float32)
                self._meta = []
                log.info("index refreshed: 0 labeled embeddings")
                return

            vectors = np.stack(
                [np.frombuffer(b, dtype=np.float32) for _, b, _, _, _ in rows]
            )
            self._matrix = vectors
            self._meta = [
                Neighbor(pass_id=pid, make=mk, model=mdl, color=col, sim=0.0)
                for pid, _, mk, mdl, col in rows
            ]
            log.info(
                "index refreshed: %d labeled embeddings (dim=%d)",
                len(self._meta), self._matrix.shape[1],
            )

    def size(self) -> int:
        return len(self._meta)

    def topk(self, query: np.ndarray, k: int = 5, exclude_pass_id: int | None = None) -> list[Neighbor]:
        with self._lock:
            if self._matrix.shape[0] == 0:
                return []
            # Embeddings are already L2-normalized so cosine = dot.
            sims = self._matrix @ query
            order = np.argsort(-sims)
            out: list[Neighbor] = []
            for idx in order:
                m = self._meta[int(idx)]
                if exclude_pass_id is not None and m.pass_id == exclude_pass_id:
                    continue
                out.append(Neighbor(
                    pass_id=m.pass_id, make=m.make, model=m.model,
                    color=m.color, sim=float(sims[idx]),
                ))
                if len(out) >= k:
                    break
            return out
