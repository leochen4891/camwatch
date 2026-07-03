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
from dataclasses import dataclass
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
# After this many consecutive *client-side* delivery rejections (HTTP 4xx: the
# hub understood the request and refused it — a poison pass, not an outage) a
# pending pass is dead-lettered so it stops holding the head of the id-ordered
# batch. Transient failures (network errors, 5xx) never count toward this — an
# outage must not quarantine deliverable passes; they retry indefinitely.
MAX_UPLOAD_ATTEMPTS = 20


@dataclass
class _Outcome:
    """Result of one delivery attempt. Truthy iff the hub accepted it, so
    existing `if self._upload_pass(p):` callers keep working unchanged."""

    ok: bool
    status: int | None = None   # HTTP status, or None if no response arrived
    transient: bool = False     # network error / 5xx — retry, never dead-letter

    def __bool__(self) -> bool:
        return self.ok


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
                # Dead-letter / retry bookkeeping (undeliverable-pass handling).
                # upload_state: NULL = deliverable/pending, 'dead_letter' =
                # quarantined (undeliverable, NEVER marked uploaded).
                ("upload_state", "TEXT"),
                ("upload_attempts", "INTEGER NOT NULL DEFAULT 0"),
                ("upload_error", "TEXT"),
                ("upload_last_attempt_at", "TEXT"),
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
        # Dead-lettered passes (upload_state IS NOT NULL) are excluded so an
        # undeliverable pass can never re-fill the head of the id-ordered window
        # and freeze the queue. Historically that freeze was "unblocked" by
        # manually stamping uploaded_at — which silently dropped real passes
        # from the hub. The batch now resolves every head pass to one of exactly
        # two honest outcomes: delivered (uploaded_at set) or dead-lettered
        # (quarantined, uploaded_at still NULL). It is never marked uploaded
        # without a confirmed hub accept.
        with self.db.connect() as conn:
            pending = conn.execute(
                """SELECT COUNT(*) FROM passes
                   WHERE deleted = 0
                     AND uploaded_at IS NULL
                     AND upload_state IS NULL
                     AND clip_path IS NOT NULL
                     AND id >= ?""",
                (MIN_PASS_ID,),
            ).fetchone()[0]
            rows = conn.execute(
                """SELECT * FROM passes
                   WHERE deleted = 0
                     AND uploaded_at IS NULL
                     AND upload_state IS NULL
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
            # A pass needs its thumbnail to upload. A thumbnail-less pass used to
            # be skipped forever and jam the queue; instead, regenerate the
            # thumbnail from the clip when the .mp4 is still on disk. If it
            # can't be produced (clip gone or undecodable) the pass is genuinely
            # undeliverable — dead-letter it (never uploaded) so the window
            # slides to the next deliverable pass in this same batch.
            if p.clip_path:
                clip_path = Path(p.clip_path)
                thumb_path = clip_path.with_suffix(".jpg")
                if not thumb_path.exists() and not self._regenerate_thumbnail(clip_path):
                    self._dead_letter(
                        p.id,
                        "no thumbnail and clip missing/undecodable — cannot "
                        "regenerate; quarantined (NOT marked uploaded)",
                    )
                    continue

            outcome = self._deliver(p)
            if outcome.ok:
                with self.db.connect() as conn:
                    conn.execute(
                        "UPDATE passes SET uploaded_at = datetime('now'), "
                        "upload_attempts = 0, upload_error = NULL WHERE id = ?",
                        (p.id,),
                    )
                    conn.commit()
                count += 1
                log.info("uploaded pass %d", p.id)
            elif not outcome.transient:
                # Client-side rejection (4xx): count it; a persistently-rejected
                # poison pass eventually dead-letters so it stops blocking.
                self._record_failed_attempt(p.id, outcome.status)
            # Transient (network / 5xx): leave pending, retry next poll. Never
            # counted, never dead-lettered — an outage must not drop passes.

        self._refresh_dead_letter_gauge()
        return count

    def _regenerate_thumbnail(self, clip_path: Path) -> bool:
        """Best-effort: write the sibling `.jpg` the uploader + hub expect by
        extracting a representative (mid-clip) frame from the recorder's `.mp4`.

        Returns True if the thumbnail exists afterward (already present, or
        regenerated), False if it can't be produced (clip gone/undecodable) —
        the caller then dead-letters the pass. Matches the recorder's output
        format (width<=800, JPEG q85). av/cv2 are imported lazily so the
        uploader stays light when no repair is needed."""
        thumb = clip_path.with_suffix(".jpg")
        if thumb.exists():
            return True
        if not clip_path.exists():
            return False
        try:
            import av
            import cv2

            # Two passes so we never hold every decoded frame in memory: count
            # cheaply, then decode again and convert only the middle frame.
            with av.open(str(clip_path)) as container:
                stream = container.streams.video[0]
                n = stream.frames or sum(1 for _ in container.decode(stream))
            if n <= 0:
                return False
            target = n // 2
            img = None
            with av.open(str(clip_path)) as container:
                stream = container.streams.video[0]
                for i, frame in enumerate(container.decode(stream)):
                    if i == target:
                        img = frame.to_ndarray(format="bgr24")
                        break
            if img is None:
                return False
            h, w = img.shape[:2]
            if w > 800:
                img = cv2.resize(img, (800, max(1, round(h * 800 / w))))
            cv2.imwrite(str(thumb), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            ok = thumb.exists()
            if ok:
                log.info("regenerated thumbnail from clip: %s", thumb.name)
            return ok
        except Exception:  # noqa: BLE001 — any decode/encode failure = undeliverable
            log.exception("thumbnail regen failed for %s", clip_path)
            return False

    def _dead_letter(self, pass_id: int, reason: str) -> None:
        """Quarantine an undeliverable pass: it leaves the pending window but is
        NEVER marked uploaded (uploaded_at stays NULL), so the hub-missing state
        is honest and the pass remains findable for a future restore."""
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE passes SET upload_state = 'dead_letter', upload_error = ?, "
                "upload_last_attempt_at = datetime('now') WHERE id = ?",
                (reason, pass_id),
            )
            conn.commit()
        log.warning("upload dead-letter pass %d: %s", pass_id, reason)

    def _record_failed_attempt(self, pass_id: int, status: int | None) -> None:
        """Count a client-side (4xx) delivery rejection; dead-letter once a pass
        has been rejected MAX_UPLOAD_ATTEMPTS times in a row."""
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE passes SET upload_attempts = upload_attempts + 1, "
                "upload_error = ?, upload_last_attempt_at = datetime('now') "
                "WHERE id = ?",
                (f"HTTP {status}" if status is not None else "client error", pass_id),
            )
            attempts = conn.execute(
                "SELECT upload_attempts FROM passes WHERE id = ?", (pass_id,)
            ).fetchone()[0]
            conn.commit()
        if attempts >= MAX_UPLOAD_ATTEMPTS:
            self._dead_letter(
                pass_id,
                f"{attempts} consecutive client-side rejections "
                f"(last HTTP {status}) — quarantined (NOT marked uploaded)",
            )

    def _refresh_dead_letter_gauge(self) -> None:
        with self.db.connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM passes WHERE upload_state = 'dead_letter'"
            ).fetchone()[0]
        mp.UPLOAD_DEAD_LETTER.set(n)

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
        """Deliver one pass; True iff the hub accepted it. Thin bool wrapper
        over `_deliver` for external callers (the media-verify sweep and the
        offline reprocessor) that only care whether it landed."""
        return self._deliver(p).ok

    def _deliver(self, p: Pass) -> _Outcome:
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
                    return _Outcome(ok=True, status=resp.status_code)
                log.warning("upload pass %d failed: %d %s", p.id, resp.status_code, resp.text[:200])
                mp.UPLOADS.inc(status="http_error")
                # 5xx = hub-side/transient (retry, never dead-letter); 4xx =
                # client-side rejection (a poison pass — count toward dead-letter).
                return _Outcome(ok=False, status=resp.status_code,
                                transient=resp.status_code >= 500)
        except httpx.HTTPError as e:
            log.warning("upload pass %d network error: %s", p.id, e)
            mp.UPLOADS.inc(status="network_error")
            return _Outcome(ok=False, status=None, transient=True)
