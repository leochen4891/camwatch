"""Upload finalized passes to camwatch-web cloud service.

Runs as a background thread alongside the capture worker. Periodically
scans for un-uploaded passes, assembles the multipart payload, and POSTs
to the cloud ingest endpoint.

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
        self._ensure_column()

    def _ensure_column(self) -> None:
        with self.db.connect() as conn:
            try:
                conn.execute("ALTER TABLE passes ADD COLUMN uploaded_at TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="uploader")
        self._thread.start()
        log.info("uploader started -> %s", self.cloud_url)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                uploaded = self._upload_batch()
                if uploaded == 0:
                    self._stop.wait(POLL_INTERVAL_S)
            except Exception:
                log.exception("uploader error")
                self._stop.wait(POLL_INTERVAL_S)

    def _upload_batch(self) -> int:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM passes
                   WHERE deleted = 0
                     AND uploaded_at IS NULL
                     AND clip_path IS NOT NULL
                   ORDER BY id ASC
                   LIMIT ?""",
                (BATCH_SIZE,),
            ).fetchall()

        count = 0
        for row in rows:
            p = Pass.from_row(row)
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

    def _upload_pass(self, p: Pass) -> bool:
        metadata = {
            "captured_at": p.captured_at,
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
                if resp.status_code == 201:
                    return True
                log.warning("upload pass %d failed: %d %s", p.id, resp.status_code, resp.text[:200])
                return False
        except httpx.HTTPError as e:
            log.warning("upload pass %d network error: %s", p.id, e)
            return False
