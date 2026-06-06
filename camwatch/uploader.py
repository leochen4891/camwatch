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

from . import metrics_push as mp
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
        enabled: bool = True,
    ) -> None:
        self.db = db
        self.config = config
        self.cloud_url = cloud_url.rstrip("/")
        self.api_key = api_key
        self._stop = threading.Event()
        # When clear, the poll loop idles without scanning or POSTing. Lets the
        # /settings switch pause/resume hub uploads live, without a restart.
        self._enabled = threading.Event()
        if enabled:
            self._enabled.set()
        self._thread: threading.Thread | None = None
        self._ensure_columns()

    def set_enabled(self, enabled: bool) -> None:
        """Flip hub uploading on/off at runtime (called from /settings)."""
        if enabled == self._enabled.is_set():
            return
        if enabled:
            self._enabled.set()
        else:
            self._enabled.clear()
        log.info("uploader %s", "resumed" if enabled else "paused")

    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

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
        log.info(
            "uploader thread started -> %s (%s)",
            self.cloud_url,
            "active" if self._enabled.is_set() else "paused",
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        fixed_media = False
        while not self._stop.is_set():
            if not self._enabled.is_set():
                # Paused via the /settings switch: idle without scanning the DB
                # or POSTing. Re-check after the poll interval.
                self._stop.wait(POLL_INTERVAL_S)
                continue
            try:
                # Runs the first time uploading is active — at startup if
                # enabled, otherwise the first poll after being switched on.
                if not fixed_media:
                    self._fix_missing_media()
                    fixed_media = True
                uploaded = self._upload_batch()
                synced = self._sync_enrichment_batch()
                if uploaded == 0 and synced == 0:
                    self._stop.wait(POLL_INTERVAL_S)
            except Exception:
                log.exception("uploader error")
                self._stop.wait(POLL_INTERVAL_S)

    def _fix_missing_media(self) -> None:
        """Verified media sweep, run once per upload activation.

        Asks the hub which recently-uploaded passes actually lack media
        (`has_thumb` / `has_clip` in the GET /api/passes list contract) and
        re-uploads only those — and only when the local media file still
        exists. The original sweep re-sent the latest 200 uploaded passes
        unconditionally; with the thumbnail-wait invariant in
        _upload_batch (passes wait for the .jpg before first upload) it
        healed nothing and starved fresh uploads for ~45 min after every
        service restart. Against a healthy hub this now re-uploads zero.
        """
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
        candidates = [Pass.from_row(r) for r in rows]
        if not candidates:
            return

        hub = self._hub_media_index(min(p.captured_at for p in candidates))
        if hub is None:
            # Hub unreachable: skip the sweep entirely — never block fresh
            # uploads (or boot) on it. The next restart tries again.
            log.warning("media verify: hub list unavailable; sweep skipped")
            return

        checked = 0
        fixed = 0
        for p in candidates:
            entry = hub.get(p.id)
            if entry is None:
                # Outside the hub's list window or deleted on the hub —
                # neither is a media gap; never blind-resend (a re-upload
                # would resurrect hub-side deletions).
                continue
            checked += 1
            clip = Path(p.clip_path)
            thumb = clip.with_suffix(".jpg")
            needs = (
                (not entry.get("has_thumb") and thumb.exists())
                or (not entry.get("has_clip") and clip.exists())
            )
            if not needs:
                continue
            if self._upload_pass(p):
                fixed += 1
                log.info("re-uploaded pass %d (media verify)", p.id)
        log.info("media verify: %d checked, %d re-uploaded", checked, fixed)

    def _hub_media_index(self, since: str) -> dict[int, dict] | None:
        """engine_pass_id → hub list row (with has_thumb / has_clip), for
        the window starting at `since` (clamped server-side to the Bearer
        tier's 30d range cap). Returns None when the hub can't be queried,
        so the caller skips the sweep instead of guessing.

        The list carries only the elected headline speed fields (ADR-014);
        gate exclusively on has_thumb / has_clip here.
        """
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    f"{self.cloud_url}/api/passes",
                    params={"range": "30d", "since": since},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            if resp.status_code != 200:
                log.warning("media verify: hub list HTTP %d", resp.status_code)
                return None
            passes = resp.json().get("passes") or []
        except (httpx.HTTPError, ValueError) as e:
            log.warning("media verify: hub list fetch failed: %s", e)
            return None
        return {
            int(row["engine_pass_id"]): row
            for row in passes
            if row.get("engine_pass_id") is not None
        }

    def _upload_batch(self) -> int:
        with self.db.connect() as conn:
            pending = conn.execute(
                """SELECT COUNT(*) FROM passes
                   WHERE deleted = 0
                     AND uploaded_at IS NULL
                     AND clip_path IS NOT NULL
                     AND id >= ?""",
                (MIN_PASS_ID,),
            ).fetchone()[0]
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
        mp.UPLOAD_PENDING.set(pending)

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

    def _pass_metadata(self, p: Pass) -> dict:
        """The ingest `metadata` JSON for one pass (split out for tests)."""
        return {
            "captured_at": p.captured_at,
            "engine_pass_id": p.id,
            "track_id": p.track_id,
            "cls_name": p.cls_name,
            "direction": p.direction,
            "elapsed_s": p.elapsed_s,
            "speed_mph": p.speed_mph,
            "speed_method": p.speed_method,
            # Camera provenance (ADR-013): the registry camera_id that
            # produced this pass; with speed_method it seeds the hub's
            # initial pass_speeds measurement (ADR-014). Rows from before
            # the local column all predate multi-camera and are cx810's.
            # The hub ignores unknown metadata fields until its migration
            # ships, so sending this early is harmless.
            "camera": p.camera or "cx810",
            "known_mph": p.known_mph,
            "is_alarm": p.speed_mph is not None and p.speed_mph >= self.config.alert_threshold_mph and p.known_mph is None,
            "threshold_mph": self.config.alert_threshold_mph,
            "vehicle_make": p.vehicle_make,
            "vehicle_model": p.vehicle_model,
            "vehicle_color": p.vehicle_color,
            "vehicle_confidence": p.vehicle_confidence,
        }

    def _upload_pass(self, p: Pass) -> bool:
        metadata = self._pass_metadata(p)

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

        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{self.cloud_url}/api/ingest",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                )
                if resp.status_code in (200, 201):
                    mp.UPLOADS.inc(status="ok")
                    mp.UPLOAD_SECONDS.observe(time.monotonic() - t0)
                    return True
                log.warning("upload pass %d failed: %d %s", p.id, resp.status_code, resp.text[:200])
                mp.UPLOADS.inc(status="http_error")
                return False
        except httpx.HTTPError as e:
            log.warning("upload pass %d network error: %s", p.id, e)
            mp.UPLOADS.inc(status="network_error")
            return False
