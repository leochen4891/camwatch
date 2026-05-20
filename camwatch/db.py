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
    id                   INTEGER PRIMARY KEY,
    captured_at          TEXT    NOT NULL,
    track_id             INTEGER NOT NULL,
    cls_name             TEXT,
    direction            TEXT    NOT NULL CHECK (direction IN ('N','S')),
    elapsed_s            REAL    NOT NULL,
    known_mph            REAL,
    clip_path            TEXT,
    deleted              INTEGER NOT NULL DEFAULT 0,
    thumb_upgrade_status TEXT,  -- NULL=pending/none, 'ok'=upgraded, 'failed'=tried but failed
    speed_mph            REAL,  -- single source of truth for displayed speed (homography)
    speed_method         TEXT,  -- 'regression' (high confidence) | 'median_fallback' (low confidence) | NULL (no speed)
    vehicle_make         TEXT,
    vehicle_model        TEXT,
    vehicle_year_range   TEXT,
    vehicle_color        TEXT,  -- categorical bucket: light/grey/dark/red/blue/green/brown/yellow
    vehicle_confidence   TEXT,  -- 'high' | 'medium' | 'low'
    vehicle_enriched_at  TEXT,
    vehicle_enriched_by  TEXT,  -- 'local' | 'opus' | NULL (historical; new rows always 'opus' for vehicle_*)
    -- Local enricher writes its own label to a parallel set of columns so
    -- it can coexist with Opus's answer in vehicle_*. Enables backtest /
    -- agreement analysis without overwriting either side.
    local_make           TEXT,
    local_model          TEXT,
    local_color          TEXT,
    local_confidence     TEXT,  -- 'high' | 'medium' (only set when label fires)
    local_enriched_at    TEXT,
    enrich_local_status  TEXT,  -- 'high' | 'medium' | 'low' | 'no_match' | 'error' | NULL — local enricher's per-call status (always recorded once attempted)
    enrich_local_topk    TEXT   -- JSON [{pass_id, make, model, sim}, ...] from the thumbnail view, for debugging
);
CREATE INDEX IF NOT EXISTS passes_captured_at_idx ON passes(captured_at);
CREATE INDEX IF NOT EXISTS passes_deleted_idx ON passes(deleted);

-- pass_embeddings used to live here. It now belongs to the
-- camwatch-enricher service (separate repo, separate SQLite at
-- enricher.db). camwatch does not read or write embeddings directly —
-- the enricher pulls labeled passes from /api/labeled-passes.

CREATE TABLE IF NOT EXISTS metrics (
    ts    TEXT NOT NULL,        -- 5s bucket start, local-aware ISO seconds
    name  TEXT NOT NULL,        -- 'fps_sub' | 'fps_main' | 'fps_yolo'
                                -- | 'yolo_ms_p50' | 'yolo_ms_p95'
                                -- | 'lag_ms_p50' | 'lag_ms_p95'
                                -- | 'queue_depth'
    value REAL,
    PRIMARY KEY (ts, name)
);
CREATE INDEX IF NOT EXISTS metrics_ts_idx ON metrics(ts);
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
    thumb_upgrade_status: str | None  # None | 'ok' | 'failed'
    speed_mph: float | None  # canonical displayed speed (homography-based for new rows)
    speed_method: str | None  # 'regression' | 'median_fallback' | None — confidence indicator for the UI
    vehicle_make: str | None
    vehicle_model: str | None
    vehicle_year_range: str | None
    vehicle_color: str | None
    vehicle_confidence: str | None
    vehicle_enriched_at: str | None
    vehicle_enriched_by: str | None = None  # 'opus' | (historical 'local') | None
    local_make: str | None = None
    local_model: str | None = None
    local_color: str | None = None
    local_confidence: str | None = None     # 'high' | 'medium' | None
    local_enriched_at: str | None = None
    enrich_local_status: str | None = None  # 'high' | 'medium' | 'low' | 'no_match' | None
    enrich_local_topk: str | None = None    # JSON debug payload
    deleted_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Pass:
        # `thumb_upgrade_status`, `speed_mph`, `speed_method` only exist on
        # freshly bootstrapped tables; access via .keys() so we don't blow
        # up against an old schema.
        keys = row.keys() if hasattr(row, "keys") else []
        upgrade_status = row["thumb_upgrade_status"] if "thumb_upgrade_status" in keys else None
        speed_mph = row["speed_mph"] if "speed_mph" in keys else None
        speed_method = row["speed_method"] if "speed_method" in keys else None
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
            thumb_upgrade_status=upgrade_status,
            speed_mph=speed_mph,
            speed_method=speed_method,
            vehicle_make=row["vehicle_make"] if "vehicle_make" in keys else None,
            vehicle_model=row["vehicle_model"] if "vehicle_model" in keys else None,
            vehicle_year_range=row["vehicle_year_range"] if "vehicle_year_range" in keys else None,
            vehicle_color=row["vehicle_color"] if "vehicle_color" in keys else None,
            vehicle_confidence=row["vehicle_confidence"] if "vehicle_confidence" in keys else None,
            vehicle_enriched_at=row["vehicle_enriched_at"] if "vehicle_enriched_at" in keys else None,
            vehicle_enriched_by=row["vehicle_enriched_by"] if "vehicle_enriched_by" in keys else None,
            local_make=row["local_make"] if "local_make" in keys else None,
            local_model=row["local_model"] if "local_model" in keys else None,
            local_color=row["local_color"] if "local_color" in keys else None,
            local_confidence=row["local_confidence"] if "local_confidence" in keys else None,
            local_enriched_at=row["local_enriched_at"] if "local_enriched_at" in keys else None,
            enrich_local_status=row["enrich_local_status"] if "enrich_local_status" in keys else None,
            enrich_local_topk=row["enrich_local_topk"] if "enrich_local_topk" in keys else None,
            deleted_at=row["deleted_at"] if "deleted_at" in keys else None,
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
                # Add columns to existing tables created before they existed.
                for col, decl in [
                    ("thumb_upgrade_status", "TEXT"),
                    ("speed_mph", "REAL"),
                    ("speed_method", "TEXT"),
                    ("vehicle_make", "TEXT"),
                    ("vehicle_model", "TEXT"),
                    ("vehicle_year_range", "TEXT"),
                    ("vehicle_color", "TEXT"),
                    ("vehicle_confidence", "TEXT"),
                    ("vehicle_enriched_at", "TEXT"),
                    ("vehicle_enriched_by", "TEXT"),
                    ("local_make", "TEXT"),
                    ("local_model", "TEXT"),
                    ("local_color", "TEXT"),
                    ("local_confidence", "TEXT"),
                    ("local_enriched_at", "TEXT"),
                    ("enrich_local_status", "TEXT"),
                    ("enrich_local_topk", "TEXT"),
                    ("deleted_at", "TEXT"),
                ]:
                    try:
                        conn.execute(f"ALTER TABLE passes ADD COLUMN {col} {decl}")
                    except sqlite3.OperationalError:
                        pass  # already there
                # Backfill speed_mph for legacy passes that pre-date the
                # column. We use the old 2-line formula
                # (line_distance_m_* / elapsed_s) so historical passes still
                # display sensibly with the new single-source-of-truth read.
                # We can only do this once we know the line distances, so the
                # actual backfill happens via backfill_legacy_speed() called
                # after Config is loaded.
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
        speed_mph: float | None = None,
        speed_method: str | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO passes
                    (captured_at, track_id, cls_name, direction, elapsed_s,
                     clip_path, speed_mph, speed_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (captured_at, int(track_id), cls_name, direction,
                 float(elapsed_s), clip_path,
                 None if speed_mph is None else float(speed_mph),
                 speed_method),
            )
            return int(cur.lastrowid)

    def apply_local_enrichment(self, pass_id: int, response: dict) -> None:
        """Persist a camwatch-enricher /enrich response into the local_* columns.

        High-confidence responses set the label fields; everything records
        the per-call status + top-K debug payload so we can audit why local
        declined to label a pass.
        """
        import json
        from datetime import datetime, timezone

        status = str(response.get("status") or "low")
        topk_payload = {
            "status": status,
            "top_sim": response.get("top_sim"),
            "top_matches": response.get("top_matches") or [],
            "views": response.get("views") or [],
        }
        topk_json = json.dumps(topk_payload)
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.connect() as conn:
            if status == "high":
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
                    (
                        response.get("make"), response.get("model"),
                        response.get("color"),
                        now_iso, status, topk_json, int(pass_id),
                    ),
                )
            else:
                # Non-high: clear any prior local label so the row reflects
                # the latest call. Useful for re-runs against a new index /
                # model where a previously-high label is no longer supported.
                conn.execute(
                    """
                    UPDATE passes SET
                        local_make          = NULL,
                        local_model         = NULL,
                        local_color         = NULL,
                        local_confidence    = NULL,
                        local_enriched_at   = NULL,
                        enrich_local_status = ?,
                        enrich_local_topk   = ?
                    WHERE id = ?
                    """,
                    (status, topk_json, int(pass_id)),
                )

    def backfill_legacy_speed(
        self,
        line_distance_m_north: float,
        line_distance_m_south: float,
    ) -> int:
        """One-time migration: populate speed_mph for legacy passes that
        pre-date the column, using the old 2-line formula. Idempotent — only
        touches rows where speed_mph IS NULL. Returns the number of rows
        updated."""
        if line_distance_m_north <= 0 and line_distance_m_south <= 0:
            return 0
        mps_to_mph = 2.2369362920544
        with self.connect() as conn:
            cur = conn.execute(
                """
                UPDATE passes
                SET speed_mph = CASE
                    WHEN direction = 'N' AND ? > 0 AND elapsed_s > 0
                        THEN (? / elapsed_s) * ?
                    WHEN direction = 'S' AND ? > 0 AND elapsed_s > 0
                        THEN (? / elapsed_s) * ?
                    ELSE NULL
                END
                WHERE speed_mph IS NULL
                """,
                (line_distance_m_north, line_distance_m_north, mps_to_mph,
                 line_distance_m_south, line_distance_m_south, mps_to_mph),
            )
            return cur.rowcount

    def set_known_mph(self, pass_id: int, known_mph: float | None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE passes SET known_mph = ? WHERE id = ?",
                (None if known_mph is None else float(known_mph), int(pass_id)),
            )

    def soft_delete(self, ids: list[int]) -> int:
        if not ids:
            return 0
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.connect() as conn:
            placeholders = ",".join("?" * len(ids))
            cur = conn.execute(
                f"UPDATE passes SET deleted = 1, deleted_at = ? WHERE id IN ({placeholders})",
                [now, *[int(i) for i in ids]],
            )
            return cur.rowcount

    def restore(self, ids: list[int]) -> int:
        if not ids:
            return 0
        with self.connect() as conn:
            placeholders = ",".join("?" * len(ids))
            cur = conn.execute(
                f"UPDATE passes SET deleted = 0, deleted_at = NULL WHERE id IN ({placeholders})",
                [int(i) for i in ids],
            )
            return cur.rowcount

    def passes_with_clip_older_than(
        self, days: int
    ) -> list[tuple[int, str, float | None]]:
        """Return [(id, clip_path, speed_mph), ...] for passes captured more
        than `days` ago that still have a `clip_path` recorded. Used by the
        clips-retention sweep to delete .mp4 files. Does NOT mutate the row;
        the .mp4 deletion is idempotent so re-listing across sweeps is safe."""
        if days <= 0:
            return []
        from datetime import datetime, timedelta, timezone
        cutoff = (
            datetime.now(timezone.utc).astimezone() - timedelta(days=days)
        ).isoformat(timespec="seconds")
        with self.connect() as conn:
            rows = list(conn.execute(
                "SELECT id, clip_path, speed_mph FROM passes WHERE captured_at < ? AND clip_path IS NOT NULL",
                (cutoff,),
            ))
            return [(r["id"], r["clip_path"], r["speed_mph"]) for r in rows]

    def purge_thumbs_older_than(
        self, days: int
    ) -> list[tuple[int, str, float | None]]:
        """Drop thumbnail files for passes older than `days` and NULL their
        `clip_path` so future sweeps skip them. Returns [(id, clip_path, speed_mph)]
        captured at the moment of the sweep — callers use speed_mph to decide
        whether to archive (alarm passes) or hard-delete the .jpg."""
        if days <= 0:
            return []
        from datetime import datetime, timedelta, timezone
        cutoff = (
            datetime.now(timezone.utc).astimezone() - timedelta(days=days)
        ).isoformat(timespec="seconds")
        with self.connect() as conn:
            rows = list(conn.execute(
                "SELECT id, clip_path, speed_mph FROM passes WHERE captured_at < ? AND clip_path IS NOT NULL",
                (cutoff,),
            ))
            items: list[tuple[int, str, float | None]] = [
                (r["id"], r["clip_path"], r["speed_mph"]) for r in rows
            ]
            if items:
                placeholders = ",".join("?" * len(items))
                conn.execute(
                    f"UPDATE passes SET clip_path = NULL WHERE id IN ({placeholders})",
                    [i for i, _, _ in items],
                )
            return items

    def purge_older_than(self, days: int) -> tuple[int, list[tuple[int, str | None]]]:
        """Hard-delete passes older than `days` and return (count, [(id, clip_path), ...])
        so the caller can also clean up the clip + thumb + per-pass log files on disk.
        clip_path may be None for rows without a recorded clip."""
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
            items: list[tuple[int, str | None]] = [(r["id"], r["clip_path"]) for r in rows]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"DELETE FROM passes WHERE id IN ({placeholders})",
                    ids,
                )
            return len(ids), items

    # ---------- reads ----------

    def list_passes(
        self,
        direction: str | None = None,
        alerts_only: bool = False,
        limit: int = 200,
        threshold_mph: float | None = None,
        line_distance_m_north: float | None = None,  # noqa: ARG002 (legacy, unused)
        line_distance_m_south: float | None = None,  # noqa: ARG002 (legacy, unused)
        include_deleted: bool = False,
        time_ranges: list[tuple[str, str]] | None = None,
        offset: int = 0,
    ) -> list[Pass]:
        # time_ranges semantics: None = no time filter; [] = no rows match;
        # otherwise OR-joined half-open ranges [start, end).
        if time_ranges is not None and not time_ranges:
            return []
        sql = "SELECT * FROM passes" if include_deleted else "SELECT * FROM passes WHERE deleted = 0"
        params: list[Any] = []
        if direction in ("N", "S"):
            sql += (" AND " if "WHERE" in sql else " WHERE ") + "direction = ?"
            params.append(direction)
        if time_ranges:
            clause = " OR ".join(["(captured_at >= ? AND captured_at < ?)"] * len(time_ranges))
            sql += (" AND " if "WHERE" in sql else " WHERE ") + f"({clause})"
            for start, end in time_ranges:
                params.append(start)
                params.append(end)
        sql += " ORDER BY captured_at DESC LIMIT ? OFFSET ?"
        params.append(int(limit))
        params.append(int(offset))
        with self.connect() as conn:
            rows = [Pass.from_row(r) for r in conn.execute(sql, params)]

        if alerts_only and threshold_mph is not None:
            kept: list[Pass] = []
            for p in rows:
                if p.speed_mph is None or p.speed_mph <= 0:
                    continue
                if p.speed_mph >= threshold_mph:
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

    # ---------- metrics (performance timeline) ----------

    def insert_metric_samples(self, ts_iso: str, samples: dict[str, float]) -> None:
        """Upsert a row per (ts_iso, name). Uses ON CONFLICT REPLACE so a
        late re-flush for the same bucket overwrites rather than dropping."""
        if not samples:
            return
        rows = [
            (ts_iso, name, None if value is None else float(value))
            for name, value in samples.items()
        ]
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO metrics (ts, name, value) VALUES (?, ?, ?) "
                "ON CONFLICT(ts, name) DO UPDATE SET value=excluded.value",
                rows,
            )

    def recent_metrics(self, since_iso: str) -> dict[str, list[tuple[str, float | None]]]:
        """Returns {name: [(ts, value), ...]} for ts >= since_iso, ts-ascending."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT ts, name, value FROM metrics "
                "WHERE ts >= ? ORDER BY ts ASC",
                (since_iso,),
            ).fetchall()
        out: dict[str, list[tuple[str, float | None]]] = {}
        for r in rows:
            out.setdefault(r["name"], []).append(
                (r["ts"], None if r["value"] is None else float(r["value"]))
            )
        return out

    def purge_metrics_older_than(self, days: int) -> int:
        """Hard-delete metric rows older than `days`. Returns rowcount."""
        if days <= 0:
            return 0
        from datetime import datetime, timedelta, timezone
        cutoff = (
            datetime.now(timezone.utc).astimezone() - timedelta(days=days)
        ).isoformat(timespec="seconds")
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM metrics WHERE ts < ?", (cutoff,))
            return cur.rowcount
