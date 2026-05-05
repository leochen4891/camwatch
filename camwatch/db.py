"""SQLite-backed pass storage.

Single writer (the capture worker thread) + multiple readers (FastAPI request
handlers). WAL mode keeps readers from blocking the writer. Each thread/request
gets its own short-lived connection via `connect()`.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("camwatch.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS passes (
    id           INTEGER PRIMARY KEY,
    captured_at  TEXT    NOT NULL,
    track_id     INTEGER NOT NULL,
    cls_name     TEXT,
    direction    TEXT    NOT NULL CHECK (direction IN ('N','S')),
    elapsed_s    REAL    NOT NULL,
    known_mph    REAL,
    clip_path    TEXT,
    deleted      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS passes_captured_at_idx ON passes(captured_at);
CREATE INDEX IF NOT EXISTS passes_deleted_idx ON passes(deleted);
"""


@dataclass
class Pass:
    id: int
    captured_at: str
    track_id: int
    cls_name: str | None
    direction: str
    elapsed_s: float
    known_mph: float | None
    clip_path: str | None
    deleted: bool

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Pass:
        return cls(
            id=row["id"],
            captured_at=row["captured_at"],
            track_id=row["track_id"],
            cls_name=row["cls_name"],
            direction=row["direction"],
            elapsed_s=row["elapsed_s"],
            known_mph=row["known_mph"],
            clip_path=row["clip_path"],
            deleted=bool(row["deleted"]),
        )


class Database:
    """Holds the db path and bootstraps the schema; hands out connections."""

    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self._init_lock = threading.Lock()
        self._initialized = False
        self.bootstrap()

    def bootstrap(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            with self.connect() as conn:
                conn.executescript(_SCHEMA)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.commit()
            self._initialized = True
            log.info("db ready: %s", self.path.resolve())

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    # ---------- writes ----------

    def insert_pass(
        self,
        captured_at: str,
        track_id: int,
        cls_name: str | None,
        direction: str,
        elapsed_s: float,
        clip_path: str | None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO passes (captured_at, track_id, cls_name, direction, elapsed_s, clip_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (captured_at, int(track_id), cls_name, direction, float(elapsed_s), clip_path),
            )
            return int(cur.lastrowid)

    def set_known_mph(self, pass_id: int, known_mph: float | None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE passes SET known_mph = ? WHERE id = ?",
                (None if known_mph is None else float(known_mph), int(pass_id)),
            )

    def soft_delete(self, ids: list[int]) -> int:
        if not ids:
            return 0
        with self.connect() as conn:
            placeholders = ",".join("?" * len(ids))
            cur = conn.execute(
                f"UPDATE passes SET deleted = 1 WHERE id IN ({placeholders})",
                [int(i) for i in ids],
            )
            return cur.rowcount

    def restore(self, ids: list[int]) -> int:
        if not ids:
            return 0
        with self.connect() as conn:
            placeholders = ",".join("?" * len(ids))
            cur = conn.execute(
                f"UPDATE passes SET deleted = 0 WHERE id IN ({placeholders})",
                [int(i) for i in ids],
            )
            return cur.rowcount

    def purge_older_than(self, days: int) -> tuple[int, list[str]]:
        """Hard-delete passes older than `days` and return (count, removed_clip_paths)
        so the caller can also clean up the clip + thumb files on disk."""
        if days <= 0:
            return 0, []
        from datetime import datetime, timedelta, timezone
        cutoff = (
            datetime.now(timezone.utc).astimezone() - timedelta(days=days)
        ).isoformat(timespec="seconds")
        with self.connect() as conn:
            rows = list(conn.execute(
                "SELECT id, clip_path FROM passes WHERE captured_at < ?",
                (cutoff,),
            ))
            ids = [r["id"] for r in rows]
            clip_paths = [r["clip_path"] for r in rows if r["clip_path"]]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"DELETE FROM passes WHERE id IN ({placeholders})",
                    ids,
                )
            return len(ids), clip_paths

    # ---------- reads ----------

    def list_passes(
        self,
        direction: str | None = None,
        alerts_only: bool = False,
        limit: int = 200,
        threshold_mph: float | None = None,
        line_distance_m_north: float | None = None,
        line_distance_m_south: float | None = None,
        include_deleted: bool = False,
        from_ts: str | None = None,
        to_ts: str | None = None,
        offset: int = 0,
    ) -> list[Pass]:
        sql = "SELECT * FROM passes" if include_deleted else "SELECT * FROM passes WHERE deleted = 0"
        params: list[Any] = []
        if direction in ("N", "S"):
            sql += (" AND " if "WHERE" in sql else " WHERE ") + "direction = ?"
            params.append(direction)
        if from_ts:
            sql += (" AND " if "WHERE" in sql else " WHERE ") + "captured_at >= ?"
            params.append(from_ts)
        if to_ts:
            sql += (" AND " if "WHERE" in sql else " WHERE ") + "captured_at <= ?"
            params.append(to_ts)
        sql += " ORDER BY captured_at DESC LIMIT ? OFFSET ?"
        params.append(int(limit))
        params.append(int(offset))
        with self.connect() as conn:
            rows = [Pass.from_row(r) for r in conn.execute(sql, params)]

        if alerts_only and threshold_mph is not None:
            mps_to_mph = 2.2369362920544
            kept: list[Pass] = []
            for p in rows:
                d = (
                    line_distance_m_north
                    if p.direction == "N"
                    else line_distance_m_south
                )
                if not d or d <= 0 or p.elapsed_s <= 0:
                    continue
                mph = (d / p.elapsed_s) * mps_to_mph
                if mph >= threshold_mph:
                    kept.append(p)
            rows = kept
        return rows

    def get_pass(self, pass_id: int) -> Pass | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM passes WHERE id = ?", (int(pass_id),)
            ).fetchone()
        return Pass.from_row(row) if row else None

    def passes_with_known(self, direction: str | None = None) -> list[Pass]:
        sql = "SELECT * FROM passes WHERE deleted = 0 AND known_mph IS NOT NULL"
        params: list[Any] = []
        if direction in ("N", "S"):
            sql += " AND direction = ?"
            params.append(direction)
        sql += " ORDER BY captured_at"
        with self.connect() as conn:
            return [Pass.from_row(r) for r in conn.execute(sql, params)]

    def count_passes(
        self,
        direction: str | None = None,
        include_deleted: bool = True,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> int:
        sql = "SELECT COUNT(*) FROM passes" if include_deleted else "SELECT COUNT(*) FROM passes WHERE deleted = 0"
        params: list[Any] = []
        if direction in ("N", "S"):
            sql += (" AND " if "WHERE" in sql else " WHERE ") + "direction = ?"
            params.append(direction)
        if from_ts:
            sql += (" AND " if "WHERE" in sql else " WHERE ") + "captured_at >= ?"
            params.append(from_ts)
        if to_ts:
            sql += (" AND " if "WHERE" in sql else " WHERE ") + "captured_at <= ?"
            params.append(to_ts)
        with self.connect() as conn:
            (n,) = conn.execute(sql, params).fetchone()
            return int(n)

    def count_known(self) -> int:
        with self.connect() as conn:
            (n,) = conn.execute(
                "SELECT COUNT(*) FROM passes WHERE deleted = 0 AND known_mph IS NOT NULL"
            ).fetchone()
            return int(n)
