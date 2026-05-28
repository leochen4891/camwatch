"""Upload finalized passes to camwatch-web cloud service.

Runs as a background thread alongside the capture worker. Periodically
scans for un-uploaded passes, assembles the multipart payload, and POSTs
to the cloud ingest endpoint. Also syncs enrichment data that arrives
after the initial upload.

Usage:
    from camwatch.uploader import Uploader
    uploader = Uploader(db, config, cloud_url, api_key)
    uploader.start()     # spawns daemon thread
    uploader.stop()      # signals shutdown
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import httpx

from .config import Config
from .db import Database, Pass

log = logging.getLogger(__name__)

POLL_INTERVAL_S = 30
BATCH_SIZE = 10
LOCAL_SERVER = "http://127.0.0.1:8000"
MIN_PASS_ID = 8651


class Uploader:
    def __init__(
        self,
        db: Database,
        config: Config,
        cloud_url: str,
        api_key: str,
    ) -> None:
        self.db = db
        self.config = config
        self.cloud_url = cloud_url.rstrip("/")
        self.api_key = api_key
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        with self.db.connect() as conn:
            for col, decl in [
                ("uploaded_at", "TEXT"),
                ("enrichment_synced_at", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE passes ADD COLUMN {col} {decl}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="uploader")
        self._thread.start()
        log.info("uploader started -> %s", self.cloud_url)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        self._fix_missing_media()
        while not self._stop.is_set():
            try:
                uploaded = self._upload_batch()
                synced = self._sync_enrichment_batch()
                if uploaded == 0 and synced == 0:
                    self._stop.wait(POLL_INTERVAL_S)
            except Exception:
                log.exception("uploader error")
                self._stop.wait(POLL_INTERVAL_S)

    def _fix_missing_media(self) -> None:
        """Re-upload passes that were sent before their thumbnail existed."""
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM passes
                   WHERE uploaded_at IS NOT NULL
                     AND clip_path IS NOT NULL
                     AND id >= ?
                   ORDER BY id DESC
                   LIMIT 200""",
                (MIN_PASS_ID,),
            ).fetchall()

        count = 0
        for row in rows:
            p = Pass.from_row(row)
            if not p.clip_path:
                continue
            thumb_path = Path(p.clip_path).with_suffix(".jpg")
            if not thumb_path.exists():
                continue
            if self._upload_pass(p):
                count += 1
                log.info("re-uploaded pass %d (media fix)", p.id)
        if count:
            log.info("fixed media for %d passes", count)

    def _upload_batch(self) -> int:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM passes
                   WHERE deleted = 0
                     AND uploaded_at IS NULL
                     AND clip_path IS NOT NULL
                     AND id >= ?
                   ORDER BY id ASC
                   LIMIT ?""",
                (MIN_PASS_ID, BATCH_SIZE),
            ).fetchall()

        count = 0
        for row in rows:
            p = Pass.from_row(row)
            if p.clip_path:
                thumb_path = Path(p.clip_path).with_suffix(".jpg")
                if not thumb_path.exists():
                    continue
            if self._upload_pass(p):
                with self.db.connect() as conn:
                    conn.execute(
                        "UPDATE passes SET uploaded_at = datetime('now') WHERE id = ?",
                        (p.id,),
                    )
                    conn.commit()
                count += 1
                log.info("uploaded pass %d", p.id)

        return count

    def _sync_enrichment_batch(self) -> int:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT id, vehicle_make, vehicle_model, vehicle_color,
                          vehicle_confidence, vehicle_enriched_at
                   FROM passes
                   WHERE uploaded_at IS NOT NULL
                     AND vehicle_enriched_at IS NOT NULL
                     AND (enrichment_synced_at IS NULL
                          OR enrichment_synced_at != vehicle_enriched_at)
                     AND id >= ?
                   ORDER BY id ASC
                   LIMIT ?""",
                (MIN_PASS_ID, BATCH_SIZE),
            ).fetchall()

        count = 0
        for row in rows:
            pass_id = row["id"]
            payload = {
                "vehicle_make": row["vehicle_make"],
                "vehicle_model": row["vehicle_model"],
                "vehicle_color": row["vehicle_color"],
                "vehicle_confidence": row["vehicle_confidence"],
            }
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(
                        f"{self.cloud_url}/api/passes/{pass_id}/enrich",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    if resp.status_code == 200:
                        with self.db.connect() as conn:
                            conn.execute(
                                "UPDATE passes SET enrichment_synced_at = ? WHERE id = ?",
                                (row["vehicle_enriched_at"], pass_id),
                            )
                            conn.commit()
                        count += 1
                        log.info("synced enrichment for pass %d", pass_id)
                    else:
                        log.warning("enrich sync pass %d: HTTP %d", pass_id, resp.status_code)
            except httpx.HTTPError as e:
                log.warning("enrich sync pass %d: %s", pass_id, e)

        return count

    def _upload_pass(self, p: Pass) -> bool:
        metadata = {
            "captured_at": p.captured_at,
            "engine_pass_id": p.id,
            "track_id": p.track_id,
            "cls_name": p.cls_name,
            "direction": p.direction,
            "elapsed_s": p.elapsed_s,
            "speed_mph": p.speed_mph,
            "speed_method": p.speed_method,
            "known_mph": p.known_mph,
            "is_alarm": p.speed_mph is not None and p.speed_mph >= self.config.alert_threshold_mph and p.known_mph is None,
            "threshold_mph": self.config.alert_threshold_mph,
            "vehicle_make": p.vehicle_make,
            "vehicle_model": p.vehicle_model,
            "vehicle_color": p.vehicle_color,
            "vehicle_confidence": p.vehicle_confidence,
        }

        files: dict[str, tuple[str | None, bytes | io.IOBase, str]] = {
            "metadata": (None, json.dumps(metadata).encode(), "application/json"),
        }

        clip_path = Path(p.clip_path) if p.clip_path else None
        thumb_path = Path(str(clip_path).rsplit(".", 1)[0] + ".jpg") if clip_path else None

        if thumb_path and thumb_path.exists():
            files["thumb"] = ("thumb.jpg", thumb_path.read_bytes(), "image/jpeg")

        if clip_path and clip_path.exists():
            files["clip"] = ("clip.mp4", clip_path.read_bytes(), "video/mp4")

        trajectory_path = self.config.events_dir / f"pass_{p.id}.jsonl"
        if trajectory_path.exists():
            files["trajectory"] = ("trajectory.jsonl", trajectory_path.read_bytes(), "application/x-jsonlines")

        for anchor in ("entry", "exit"):
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(f"{LOCAL_SERVER}/passes/{p.id}/thumb?anchor={anchor}")
                    if resp.status_code == 200:
                        files[f"thumb_{anchor}"] = (f"thumb_{anchor}.jpg", resp.content, "image/jpeg")
            except httpx.HTTPError:
                pass

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{self.cloud_url}/api/ingest",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                )
                if resp.status_code in (200, 201):
                    return True
                log.warning("upload pass %d failed: %d %s", p.id, resp.status_code, resp.text[:200])
                return False
        except httpx.HTTPError as e:
            log.warning("upload pass %d network error: %s", p.id, e)
            return False
