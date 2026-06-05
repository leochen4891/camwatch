"""FastAPI app: pass review, annotation, threshold, live capture worker.

Run:
    uv run python -m camwatch serve [--host 127.0.0.1] [--port 8000]

The capture worker is started in the lifespan handler and joined on shutdown.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .capture_worker import CaptureWorker
from .config import Config, load_config
from .db import Database, Pass
from .metrics import BUCKET_S as METRICS_BUCKET_S, MetricsCollector
from .preview import PreviewBuffer

log = logging.getLogger(__name__)

MPS_TO_MPH = 2.2369362920544
HERE = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=HERE / "templates")
STATIC_DIR = HERE / "static"


# ---------- helpers ----------

def _rotate_if_oversized(path: Path, max_bytes: int = 10 * 1024 * 1024, backups: int = 5) -> None:
    """Size-based rotation: path -> path.1 -> path.2 -> ... -> path.{backups} when path exceeds max_bytes.
    Best-effort; never raises so writes are not blocked."""
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        for i in range(backups, 0, -1):
            src = path if i == 1 else path.parent / f"{path.name}.{i-1}"
            dst = path.parent / f"{path.name}.{i}"
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
    except Exception:
        log.exception("log rotation failed for %s", path)


# Color compatibility map for "filter by this vehicle". Recall-favoring: each
# queried bucket expands to a small set of acceptable matches so lighting drift
# between visits doesn't drop the actual car. Hard exclusions (light vs dark,
# red vs blue/green) still fall out naturally.
COLOR_MATCHES: dict[str, set[str]] = {
    "light":  {"light", "yellow", "grey"},
    "grey":   {"grey", "light", "dark"},
    "dark":   {"dark", "grey"},
    "red":    {"red", "brown"},
    "blue":   {"blue", "dark", "grey"},
    "green":  {"green", "brown"},
    "brown":  {"brown", "red", "yellow"},
    "yellow": {"yellow", "light", "brown"},
}


def computed_mph(p: Pass, dist_n: float, dist_s: float) -> float | None:  # noqa: ARG001
    """Read the canonical speed from the DB. Returns None when the
    homography pipeline couldn't compute a reliable speed (e.g., the
    trajectory was too short to fit a centered-window regression).

    The dist_n/dist_s args are kept for signature stability with existing
    callers; they are no longer used. The legacy 2-line fallback was
    removed because on degenerate trajectories (4-frame phantom passes,
    partial-bbox detections at the frame edge) it produces wildly
    incorrect speeds, e.g., line_distance_south / 0.4 s ≈ 81 mph for a
    pass that wasn't a real vehicle traversal at all. None is the right
    signal for "we don't know" and the UI hides the speed accordingly."""
    if p.speed_mph is not None and p.speed_mph > 0:
        return float(p.speed_mph)
    return None


def thumb_url(pass_id: int) -> str:
    return f"/passes/{pass_id}/thumb"


def clip_url(pass_id: int) -> str:
    return f"/passes/{pass_id}/clip"


def render_pass(p: Pass, dist_n: float, dist_s: float, threshold: float) -> dict:
    mph = computed_mph(p, dist_n, dist_s)
    has_clip = False
    has_thumb = False
    if p.clip_path:
        has_clip = Path(p.clip_path).exists()
        thumb = p.clip_path[:-4] + ".jpg" if p.clip_path.endswith(".mp4") else p.clip_path + ".jpg"
        has_thumb = Path(thumb).exists()
    return {
        "id": p.id,
        "deleted": p.deleted,
        "captured_at": p.captured_at,
        "direction": p.direction,
        "elapsed_s": p.elapsed_s,
        "known_mph": p.known_mph,
        "computed_mph": mph,
        "speed_method": p.speed_method,
        "alert": (mph is not None and mph >= threshold and p.known_mph is None),
        "has_clip": has_clip,
        "has_thumb": has_thumb,
        "thumb_upgrade_status": p.thumb_upgrade_status,
        "vehicle_make": p.vehicle_make,
        "vehicle_model": p.vehicle_model,
        "vehicle_year_range": p.vehicle_year_range,
        "vehicle_color": p.vehicle_color,
        "vehicle_confidence": p.vehicle_confidence,
        "vehicle_enriched_by": p.vehicle_enriched_by,
        "local_make": p.local_make,
        "local_model": p.local_model,
        "local_color": p.local_color,
        "local_confidence": p.local_confidence,
        "deleted_at": p.deleted_at,
    }


def update_threshold(cfg_path: Path, value: float) -> None:
    """Rewrite config.yaml's alert.threshold_mph in place, preserving the rest."""
    with cfg_path.open() as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("alert", {})["threshold_mph"] = float(value)
    with cfg_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def recompute_calibration(cfg: Config, db: Database) -> tuple[float, float]:
    """Average implied distances per direction; rewrite calibration.yaml.

    Considers BOTH the DB's known passes AND any existing `calibration_points`
    in calibration.yaml (saved by `python -m camwatch.calibrate freeze` or by
    earlier auto-saves). Deduped by (track_id, captured_at).

    Side effect: every call refreshes the `calibration_points` list in the
    YAML to be the merged set, so any new annotation in the UI is durably
    saved on the next recompute (which the annotate endpoint triggers
    automatically). Points already in YAML are preserved even if their DB
    row got deleted — that's the whole point of freezing.

    Returns (line_distance_m_north, line_distance_m_south)."""
    cal_path = cfg.calibration_path
    with cal_path.open() as f:
        cal = yaml.safe_load(f) or {}

    by_dir: dict[str, list[float]] = {"N": [], "S": []}
    merged_points: list[dict] = []
    seen: set[tuple] = set()

    # Existing YAML points (durable, preserved even if DB row is gone).
    for pt in (cal.get("calibration_points") or []):
        try:
            elapsed = float(pt["elapsed_s"])
            known = float(pt["known_mph"])
        except (KeyError, TypeError, ValueError):
            continue
        if elapsed <= 0:
            continue
        seen.add((pt.get("track_id"), pt.get("captured_at")))
        merged_points.append(pt)
        by_dir[pt["direction"]].append((known / MPS_TO_MPH) * elapsed)

    # DB known passes that aren't already in YAML.
    for p in db.passes_with_known():
        key = (p.track_id, p.captured_at)
        if key in seen:
            continue
        if p.elapsed_s <= 0:
            continue
        merged_points.append({
            "direction": p.direction,
            "known_mph": float(p.known_mph),
            "elapsed_s": float(p.elapsed_s),
            "captured_at": p.captured_at,
            "track_id": int(p.track_id),
            "cls_name": p.cls_name,
            "clip_path": p.clip_path,
        })
        by_dir[p.direction].append((float(p.known_mph) / MPS_TO_MPH) * p.elapsed_s)

    if by_dir["N"]:
        cal["line_distance_m_north"] = round(sum(by_dir["N"]) / len(by_dir["N"]), 3)
    if by_dir["S"]:
        cal["line_distance_m_south"] = round(sum(by_dir["S"]) / len(by_dir["S"]), 3)
    cal["calibration_points"] = merged_points

    with cal_path.open("w") as f:
        yaml.safe_dump(cal, f, sort_keys=False)
    return (
        float(cal.get("line_distance_m_north", 0) or 0),
        float(cal.get("line_distance_m_south", 0) or 0),
    )


# ---------- app factory ----------

def make_app(
    cfg: Config | None = None,
    db_path: Path = Path("camwatch.db"),
    profile: bool = False,
) -> FastAPI:
    cfg = cfg or load_config()
    db = Database(db_path)

    # The one-time legacy-speed backfill (using the old 2-line formula for
    # passes that pre-dated the speed_mph column) was completed long ago and
    # must NOT run on every startup — it clobbers any newly-inserted row
    # whose speed_mph is legitimately NULL (e.g., when the centered-window
    # regression has too few samples for a confident estimate). The function
    # `db.backfill_legacy_speed` is kept available for manual one-shot use,
    # but is no longer invoked at boot.

    preview = PreviewBuffer()
    _apply_heatmap_days(cfg.heatmap_days)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        metrics = MetricsCollector(db)
        metrics.start()
        worker = CaptureWorker(
            cfg, db, preview=preview, profile=profile, metrics=metrics,
        )
        worker.start()
        app.state.worker = worker
        app.state.cfg = cfg
        app.state.db = db
        app.state.preview = preview
        app.state.metrics = metrics

        # The uploader thread starts whenever hub credentials are present, but
        # only POSTs while cfg.upload_enabled is true. That lets the /settings
        # switch pause/resume uploads live without a restart.
        uploader = None
        cloud_url = os.environ.get("CAMWATCH_CLOUD_URL")
        cloud_key = os.environ.get("CAMWATCH_CLOUD_KEY")
        if cloud_url and cloud_key:
            from .uploader import Uploader
            uploader = Uploader(db, cfg, cloud_url, cloud_key, enabled=cfg.upload_enabled)
            uploader.start()
        app.state.uploader = uploader

        log.info("server startup complete")
        try:
            yield
        finally:
            log.info("server shutdown: stopping capture worker")
            if uploader:
                uploader.stop()
            worker.stop()
            worker.join(timeout=10)
            metrics.stop()

    app = FastAPI(lifespan=lifespan, title="camwatch")
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    access_log_path = cfg.events_dir / "access.jsonl"

    @app.middleware("http")
    async def access_log_middleware(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") or path.endswith("/thumb") or path in ("/preview.jpg", "/preview/stream"):
            return response
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "email": request.headers.get("cf-access-authenticated-user-email", "-"),
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "duration_ms": round((time.monotonic() - start) * 1000.0, 1),
            "ip": request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "-"),
            "country": request.headers.get("cf-ipcountry", "-"),
            "ua": (request.headers.get("user-agent") or "")[:200],
        }
        try:
            access_log_path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_oversized(access_log_path)
            with access_log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            log.exception("access log write failed")
        return response

    # ---------- routes ----------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return _render_index(request, cfg, db)

    @app.get("/passes", response_class=HTMLResponse)
    async def list_passes_partial(
        request: Request,
        direction: str | None = None,
        alerts_only: bool = False,
        time_mask: str | None = None,
        time_ref_date: str | None = None,
        buckets: list[int] = Query(default=[]),
        page: int = 1,
        page_size: int | None = None,
        vehicle_make: str | None = None,
        vehicle_model: str | None = None,
        vehicle_color: str | None = None,
    ):
        return _render_pass_list(
            request, cfg, db, direction, alerts_only,
            time_mask=time_mask, time_ref_date=time_ref_date,
            selected_buckets=buckets, page=page, page_size=page_size,
            vehicle_make=vehicle_make, vehicle_model=vehicle_model,
            vehicle_color=vehicle_color,
        )

    @app.post("/passes/delete")
    async def delete_passes(request: Request):
        form = await request.form()
        ids = [int(v) for v in form.getlist("ids")]
        if not ids:
            return Response(status_code=204)
        db.soft_delete(ids)
        recompute_calibration(cfg, db)
        # HTMX trigger to refresh the list
        return Response(
            status_code=200,
            headers={"HX-Trigger": "passes-changed"},
        )

    @app.post("/passes/{pass_id}/delete", response_class=HTMLResponse)
    async def delete_one(request: Request, pass_id: int):
        p = db.get_pass(pass_id)
        if p is None:
            raise HTTPException(status_code=404)
        db.soft_delete([pass_id])
        recompute_calibration(cfg, db)
        return _render_pass_row(request, cfg, db, pass_id)

    @app.post("/passes/{pass_id}/restore", response_class=HTMLResponse)
    async def restore_one(request: Request, pass_id: int):
        p = db.get_pass(pass_id)
        if p is None:
            raise HTTPException(status_code=404)
        db.restore([pass_id])
        recompute_calibration(cfg, db)
        return _render_pass_row(request, cfg, db, pass_id)

    @app.get("/passes/{pass_id}/clip")
    async def get_clip(pass_id: int):
        p = db.get_pass(pass_id)
        if p is None or not p.clip_path:
            raise HTTPException(status_code=404)
        path = Path(p.clip_path)
        if not path.exists():
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="video/mp4")

    @app.get("/passes/{pass_id}/trajectory.jsonl")
    async def get_trajectory(pass_id: int):
        """Per-frame trajectory for the visualization view: t, ground (u,v),
        bbox, projected (X, Y), v_inst_mph. First line is the manifest."""
        path = Path("events") / f"pass_{pass_id}.jsonl"
        if not path.exists():
            raise HTTPException(status_code=404, detail="trajectory not recorded for this pass")
        return FileResponse(path, media_type="application/x-jsonlines")

    @app.get("/api/homography")
    async def get_homography():
        """Current homography matrix + frame size, from the elected main
        camera's registry profile (ADR-015). Used by the client to
        project meter grid → main-stream pixel for the video canvas
        overlay. Also returns the raw clicked anchor positions so the
        client can draw the calibration grid through the actual white
        marks rather than through the H matrix's least-squares fit
        (which has small per-point residuals)."""
        try:
            data = cfg.camera.profile.calibration()
        except Exception:  # noqa: BLE001 — CalibrationMissing or a bad artifact
            raise HTTPException(status_code=404, detail="homography not calibrated")
        return JSONResponse({
            "H": data["H"],
            "frame_size": data.get("frame_size", [2048, 1536]),
            "spacing_ft": data.get("spacing_ft", 5.0),
            "road_width_ft": data.get("road_width_ft", 30.0),
            "pixel_pts": data.get("pixel_pts", []),
            "meter_pts": data.get("meter_pts", []),
        })

    @app.get("/api/labeled-passes")
    async def get_labeled_passes(since: str | None = None, limit: int = 5000):
        """Feed of Opus-labeled passes for the local enrichment service.

        Returns one entry per pass that has an Opus-supplied (make, model)
        and a clip_path on disk. Image URLs are relative so the consumer
        can resolve them against whichever camwatch base it's calling.
        """
        n = max(1, min(int(limit), 10000))
        params: list[object] = []
        sql = (
            "SELECT id, captured_at, direction, vehicle_make, vehicle_model, "
            "       vehicle_color, vehicle_confidence, clip_path "
            "FROM passes "
            "WHERE deleted = 0 "
            "  AND vehicle_enriched_at IS NOT NULL "
            "  AND vehicle_make IS NOT NULL "
            "  AND vehicle_model IS NOT NULL "
            "  AND clip_path IS NOT NULL"
        )
        if since:
            sql += " AND captured_at >= ?"
            params.append(since)
        sql += " ORDER BY captured_at ASC LIMIT ?"
        params.append(n)
        with db.connect() as conn:
            rows = list(conn.execute(sql, params))

        out: list[dict] = []
        for r in rows:
            pid = int(r["id"])
            clip = r["clip_path"]
            base = clip[:-4] if clip.endswith(".mp4") else clip
            entry_jpg = Path(f"{base}.entry.jpg")
            exit_jpg = Path(f"{base}.exit.jpg")
            out.append({
                "pass_id": pid,
                "captured_at": r["captured_at"],
                "direction": r["direction"],
                "vehicle_make": r["vehicle_make"],
                "vehicle_model": r["vehicle_model"],
                "vehicle_color": r["vehicle_color"],
                "vehicle_confidence": r["vehicle_confidence"],
                "thumb_url": f"/passes/{pid}/thumb",
                "entry_url": f"/passes/{pid}/thumb?anchor=entry" if entry_jpg.exists() else None,
                "exit_url":  f"/passes/{pid}/thumb?anchor=exit"  if exit_jpg.exists()  else None,
            })
        return JSONResponse(out)

    @app.get("/passes/{pass_id}/thumb")
    async def get_thumb(pass_id: int, big: bool = False, anchor: str | None = None):
        p = db.get_pass(pass_id)
        if p is None or not p.clip_path:
            raise HTTPException(status_code=404)
        base = p.clip_path[:-4] if p.clip_path.endswith(".mp4") else p.clip_path
        # Entry / exit spot-check images (only present for passes captured
        # after the recorder anchor-image change shipped).
        if anchor in ("entry", "exit"):
            anchor_path = Path(f"{base}.{anchor}.jpg")
            if not anchor_path.exists():
                raise HTTPException(status_code=404)
            return FileResponse(anchor_path, media_type="image/jpeg")
        if big:
            big_path = Path(base + "_big.jpg")
            if big_path.exists():
                return FileResponse(big_path, media_type="image/jpeg")
            # Fall through to regular thumb when no _big variant on disk.
        thumb_path = Path(base + ".jpg")
        if not thumb_path.exists():
            raise HTTPException(status_code=404)
        return FileResponse(thumb_path, media_type="image/jpeg")

    @app.get("/preview.jpg")
    async def get_preview_frame():
        latest = preview.get_latest()
        if latest is None:
            raise HTTPException(status_code=503, detail="preview not ready")
        _, jpeg = latest
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            # Cloudflare aggressively caches image responses by default; without
            # an explicit no-store the tunnel serves stale frames to external
            # clients for hours while local consumers see the live feed.
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/preview/stream")
    async def get_preview_stream():
        boundary = b"--frame"

        async def gen():
            last_id = 0
            while True:
                # 5s timeout so a stalled stream still emits a keepalive frame.
                result = await asyncio.to_thread(preview.wait_for_next, last_id, 5.0)
                if result is None:
                    cached = preview.get_latest()
                    if cached is None:
                        await asyncio.sleep(0.2)
                        continue
                    last_id, jpeg = cached
                else:
                    last_id, jpeg = result
                chunk = (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                yield chunk

        return StreamingResponse(
            gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/calibration/threshold", response_class=HTMLResponse)
    async def set_threshold(request: Request, threshold_mph: float = Form(...)):
        update_threshold(Path("config/config.yaml"), threshold_mph)
        cfg.alert_threshold_mph = float(threshold_mph)
        return _render_status_panel(request, cfg, db)

    @app.post("/settings", response_class=HTMLResponse)
    async def save_settings(
        request: Request,
        threshold_mph: float = Form(...),
        clips_days: int = Form(default=0),
        thumbs_days: int = Form(default=0),
        passes_days: int = Form(default=0),
        heatmap_days: int = Form(default=8),
        clip_margin_s: float = Form(default=0.5),
        clip_capture_min_mph: float = Form(default=0.0),
        clip_capture_max_mph: float = Form(default=999.0),
        preview_show_grid: bool = Form(default=False),
        pause_at_night: bool = Form(default=False),
        upload_enabled: bool = Form(default=False),
    ):
        cfg_path = Path("config/config.yaml")
        margin = max(0.0, float(clip_margin_s))
        cap_min = max(0.0, float(clip_capture_min_mph))
        cap_max = max(cap_min, float(clip_capture_max_mph))
        clips_d = max(0, int(clips_days))
        thumbs_d = max(0, int(thumbs_days))
        passes_d = max(0, int(passes_days))
        heatmap_d = max(1, min(14, int(heatmap_days)))
        # Persist to config.yaml.
        with cfg_path.open() as f:
            data = yaml.safe_load(f) or {}
        data.setdefault("alert", {})["threshold_mph"] = float(threshold_mph)
        retention_section = data.setdefault("retention", {})
        retention_section["clips_days"] = clips_d
        retention_section["thumbs_days"] = thumbs_d
        retention_section["passes_days"] = passes_d
        retention_section.pop("recordings_days", None)  # superseded by clips/thumbs split
        retention_section.pop("days", None)  # legacy single-knob key
        data.setdefault("heatmap", {})["days"] = heatmap_d
        clip_section = data.setdefault("clip", {})
        clip_section["margin_s"] = margin
        clip_section["capture_min_mph"] = cap_min
        clip_section["capture_max_mph"] = cap_max
        data.setdefault("preview", {})["show_grid"] = bool(preview_show_grid)
        data.setdefault("capture", {})["pause_at_night"] = bool(pause_at_night)
        data.setdefault("upload", {})["enabled"] = bool(upload_enabled)
        with cfg_path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        cfg.alert_threshold_mph = float(threshold_mph)
        cfg.clips_days = clips_d
        cfg.thumbs_days = thumbs_d
        cfg.passes_days = passes_d
        cfg.heatmap_days = heatmap_d
        _apply_heatmap_days(heatmap_d)
        cfg.clip_margin_s = margin
        cfg.clip_capture_min_mph = cap_min
        cfg.clip_capture_max_mph = cap_max
        cfg.preview_show_grid = bool(preview_show_grid)
        cfg.pause_at_night = bool(pause_at_night)
        cfg.upload_enabled = bool(upload_enabled)
        # Push the new margin to the running capture worker without a restart.
        worker = getattr(request.app.state, "worker", None)
        if worker is not None:
            worker.update_clip_margin(margin)
        # Push the grid-overlay toggle straight to the live preview buffer.
        preview = getattr(request.app.state, "preview", None)
        if preview is not None:
            preview.set_show_grid(bool(preview_show_grid))
        # Flip hub uploading on/off live (no-op if no creds / uploader absent).
        uploader = getattr(request.app.state, "uploader", None)
        if uploader is not None:
            uploader.set_enabled(bool(upload_enabled))
        threshold = float(cfg.alert_threshold_mph)
        archive_dir = Path("recordings_archive")
        # Phase 1: clips_days — archive alarm .mp4 to recordings_archive/,
        # delete non-alarm .mp4.
        if cfg.clips_days > 0:
            clip_items = db.passes_with_clip_older_than(cfg.clips_days)
            if clip_items:
                archive_dir.mkdir(parents=True, exist_ok=True)
            archived_clips = 0
            removed_clips = 0
            for _pid, cp, speed in clip_items:
                p = Path(cp)
                if not p.exists():
                    continue
                try:
                    if speed is not None and speed >= threshold:
                        shutil.move(str(p), archive_dir / p.name)
                        archived_clips += 1
                    else:
                        p.unlink(missing_ok=True)
                        removed_clips += 1
                except Exception:
                    pass
            if archived_clips or removed_clips:
                log.info(
                    "retention: %d alarm clips archived, %d non-alarm deleted on settings save",
                    archived_clips, removed_clips,
                )
        # Phase 2: thumbs_days — delete .jpg (archive for alarm passes), NULL clip_path.
        if cfg.thumbs_days > 0:
            thumb_items = db.purge_thumbs_older_than(cfg.thumbs_days)
            if thumb_items:
                archive_dir.mkdir(parents=True, exist_ok=True)
            archived = 0
            deleted = 0
            for _pid, cp, speed in thumb_items:
                base = cp[:-4] if cp.endswith(".mp4") else cp
                thumb = base + ".jpg"
                # Belt-and-braces: archive (alarm) or remove (non-alarm) the
                # .mp4 if it somehow outlived clips_days.
                mp4 = Path(cp)
                if mp4.exists():
                    try:
                        if speed is not None and speed >= threshold:
                            shutil.move(str(mp4), archive_dir / mp4.name)
                        else:
                            mp4.unlink(missing_ok=True)
                    except Exception:
                        pass
                if speed is not None and speed >= threshold:
                    if Path(thumb).exists():
                        try:
                            shutil.move(thumb, archive_dir / Path(thumb).name)
                            archived += 1
                        except Exception:
                            pass
                else:
                    try:
                        Path(thumb).unlink(missing_ok=True)
                        deleted += 1
                    except Exception:
                        pass
                # Entry/exit spot-check images: always delete, never archive.
                for side in (".entry.jpg", ".exit.jpg"):
                    try:
                        Path(base + side).unlink(missing_ok=True)
                    except Exception:
                        pass
            if archived or deleted:
                log.info(
                    "retention: %d alarm thumbs archived, %d non-alarm deleted on settings save",
                    archived, deleted,
                )
        # Performance metrics share the pass-record retention. They are the
        # forensic record of the processing rate at capture time, which is
        # what bounds how far back a pass's speed can be recomputed (see
        # pts_timing_investigation.md) — a pass without its era's metrics is
        # unfixable. Storage is cheap (~17k rows per metric per day).
        # passes_days == 0 means keep passes forever, so keep metrics too.
        if cfg.passes_days > 0:
            purged_metrics = db.purge_metrics_older_than(cfg.passes_days)
            if purged_metrics:
                log.info(
                    "retention: purged %d metric rows older than %d days",
                    purged_metrics, cfg.passes_days,
                )
        if cfg.passes_days > 0:
            n, items = db.purge_older_than(cfg.passes_days)
            for pid, cp in items:
                paths_to_unlink: list[str] = []
                if cp:
                    base = cp[:-4] if cp.endswith(".mp4") else cp
                    paths_to_unlink.append(cp)
                    paths_to_unlink.append(base + ".jpg")
                    paths_to_unlink.append(base + ".entry.jpg")
                    paths_to_unlink.append(base + ".exit.jpg")
                paths_to_unlink.append(str(cfg.events_dir / f"pass_{pid}.jsonl"))
                for path in paths_to_unlink:
                    try:
                        Path(path).unlink(missing_ok=True)
                    except Exception:
                        pass
            if n:
                log.info("retention: purged %d passes on settings save", n)
        return _render_status_panel(request, cfg, db)

    @app.post("/calibration/recompute", response_class=HTMLResponse)
    async def calibration_recompute(request: Request):
        recompute_calibration(cfg, db)
        return _render_status_panel(request, cfg, db)

    @app.get("/api/status")
    async def status():
        cal = cfg.load_calibration()
        return JSONResponse({
            "running": app.state.worker.is_alive() if hasattr(app.state, "worker") else False,
            "threshold_mph": cfg.alert_threshold_mph,
            "line_distance_m_north": cal.line_distance_m_north if cal else 0,
            "line_distance_m_south": cal.line_distance_m_south if cal else 0,
            "known_count": len(cal.calibration_points) if cal else 0,
        })

    @app.get("/api/metrics")
    async def metrics_api(window: str = "5m"):
        """Recent performance samples for the perf panel charts.

        ?window= one of 5m, 1h, 6h, 24h. Returns:
          {
            "bucket_s": 5,
            "buckets": ["2026-05-11T13:00:00-04:00", ...],   # ascending
            "series":  {"fps_sub": [...], "fps_main": [...], ...}
          }
        Series and buckets are aligned: series[name][i] is the value at
        buckets[i], or null if that metric had no sample in that bucket.
        """
        window_s_map = {"5m": 300, "1h": 3600, "6h": 21600, "24h": 86400}
        window_s = window_s_map.get(window, 300)
        since_dt = (
            datetime.now(timezone.utc).astimezone()
            - timedelta(seconds=window_s)
        )
        since_iso = since_dt.isoformat(timespec="seconds")
        raw = db.recent_metrics(since_iso)
        # Build the union of bucket timestamps across all series so the
        # client can render aligned line charts without per-series x-axes.
        all_ts: set[str] = set()
        for points in raw.values():
            for ts, _ in points:
                all_ts.add(ts)
        buckets = sorted(all_ts)
        index = {ts: i for i, ts in enumerate(buckets)}
        series: dict[str, list[float | None]] = {}
        for name, points in raw.items():
            row: list[float | None] = [None] * len(buckets)
            for ts, value in points:
                row[index[ts]] = value
            series[name] = row
        return JSONResponse({
            "bucket_s": int(METRICS_BUCKET_S),
            "buckets": buckets,
            "series": series,
        })

    @app.get("/status-badge", response_class=HTMLResponse)
    async def status_badge(request: Request):
        worker = getattr(request.app.state, "worker", None)
        running = bool(worker and worker.is_alive())
        is_night = bool(worker and worker.is_night_mode())
        return TEMPLATES.TemplateResponse(
            request,
            "_status_badge.html",
            {
                "running": running,
                "paused_night": running and is_night and cfg.pause_at_night,
            },
        )

    return app


# ---------- render helpers (split out for clarity) ----------

BUCKET_CAP_MPH = 50           # everything above this is grouped in the last bar
BUCKET_OVERFLOW_IDX = BUCKET_CAP_MPH // 5  # = 10, the ">50" bucket


def _bucket_for(mph: float) -> int:
    """5-mph buckets keyed on the rounded mph (matches the row display, which
    formats with %.0f). 1..5 -> 0, 6..10 -> 1, …, 46..50 -> 9, >50 -> 10.
    Pre-1 mph (rounds to 0) clamps to bucket 0."""
    r = round(mph)
    if r < 1:
        return 0
    return min(BUCKET_OVERFLOW_IDX, int((r - 1) // 5))


def _bucket_label(idx: int) -> str:
    if idx >= BUCKET_OVERFLOW_IDX:
        return f">{BUCKET_CAP_MPH}"
    return f"{idx * 5 + 1}-{idx * 5 + 5}"


def _build_histogram(
    rows_for_hist: list[dict],
    threshold: float,
    selected_buckets: set[int],
) -> tuple[list[dict], int, bool]:
    """Returns (bars, total_count, all_default). Each bar dict has idx, label,
    count, height_pct, selected. Always emits buckets 0..BUCKET_OVERFLOW_IDX so
    the rightmost ">50" bucket is permanently clickable. `all_default` is True
    when no bucket has been explicitly picked (server treats this as "no
    filter / show all"); the template uses it to render every bar as visually
    selected so the default state doesn't look like everything is filtered out.
    """
    counts: dict[int, int] = {}
    total = 0
    for r in rows_for_hist:
        if r["computed_mph"] is None:
            continue
        idx = _bucket_for(r["computed_mph"])
        counts[idx] = counts.get(idx, 0) + 1
        total += 1

    max_count = max(counts.values(), default=0) or 1
    bars: list[dict] = []
    for idx in range(0, BUCKET_OVERFLOW_IDX + 1):
        c = counts.get(idx, 0)
        bars.append({
            "idx": idx,
            "label": _bucket_label(idx),
            "count": c,
            "height_pct": int(round(c / max_count * 100)),
            "selected": idx in selected_buckets,
        })
    return bars, total, not selected_buckets


# ---------- heatmap (week-view time filter) ----------

DAY_START_HOUR = 6   # heatmap rows start at 06:00 local
DAY_END_HOUR = 22    # exclusive: rows stop at 22:00 (last slot is 21:30-22:00)
SLOTS_PER_DAY = (DAY_END_HOUR - DAY_START_HOUR) * 2  # 32
DAYS_IN_WEEK = 8  # heatmap rolling window length; reassigned at startup
                  # and on settings save from cfg.heatmap_days (1..14).
TOTAL_SLOTS = DAYS_IN_WEEK * SLOTS_PER_DAY


def _apply_heatmap_days(n: int) -> None:
    """Rebind the heatmap window length. Called from create_app() at
    startup and from the settings POST handler when the user changes it."""
    global DAYS_IN_WEEK, TOTAL_SLOTS
    DAYS_IN_WEEK = max(1, min(14, int(n)))
    TOTAL_SLOTS = DAYS_IN_WEEK * SLOTS_PER_DAY


def _today_local() -> date:
    return datetime.now().astimezone().date()


def _week_window(today: date) -> tuple[datetime, datetime]:
    """Trailing 7 days ending today, as local-aware datetimes.
    Returns [start, end) where start=today-6 at 00:00 and end=today+1 at 00:00."""
    tz = datetime.now().astimezone().tzinfo
    start = datetime.combine(today - timedelta(days=DAYS_IN_WEEK - 1), datetime.min.time(), tzinfo=tz)
    end = datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=tz)
    return start, end


def _parse_time_mask(
    mask: str | None, ref_date: str | None, today: date,
) -> set[int] | None:
    """Returns selected slot indices, or None for 'no time filter (all selected)'.

    None / blank / all-1 / stale ref_date → None (treat as no filter, render all selected).
    Empty after parse → empty set (no rows match).
    """
    if not mask:
        return None
    if ref_date != today.isoformat():
        return None
    if len(mask) != TOTAL_SLOTS:
        return None
    if all(c == "1" for c in mask):
        return None
    return {i for i, c in enumerate(mask) if c == "1"}


def _slot_dt(slot_idx: int, week_start: datetime) -> datetime:
    """Local datetime at the START of the given slot.

    Edge slots act as overflow buckets:
      - slot 0 of any day starts at 00:00 (covers everything before 06:30).
      - all other slots N start at DAY_START_HOUR + (N-1)*0.5h + 0:30, i.e.
        the regular grid 06:30, 07:00, ..., 21:30.
      - the LAST slot's end is the next day at 00:00 (see _slot_end_dt).
    """
    day = slot_idx // SLOTS_PER_DAY
    slot_in_day = slot_idx % SLOTS_PER_DAY
    if slot_in_day == 0:
        return week_start + timedelta(days=day)
    return week_start + timedelta(days=day, hours=DAY_START_HOUR, minutes=30 * slot_in_day)


def _slot_end_dt(slot_idx: int, week_start: datetime) -> datetime:
    """Local datetime at the END (exclusive) of the given slot."""
    day = slot_idx // SLOTS_PER_DAY
    slot_in_day = slot_idx % SLOTS_PER_DAY
    if slot_in_day == SLOTS_PER_DAY - 1:
        return week_start + timedelta(days=day + 1)
    return _slot_dt(slot_idx + 1, week_start)


def _mask_to_time_ranges(
    selected: set[int] | None, today: date,
) -> list[tuple[str, str]] | None:
    """Convert mask to OR-able [start, end) ISO ranges for db.list_passes.

    None → 7 daily 24-hour ranges (the visible window with edge overflow).
    Empty set → [] (no rows match).
    Otherwise: contiguous slot runs collapsed into ranges. With overflow
    edges, slot 31 of day D (ending at day D+1 00:00) is time-contiguous
    with slot 0 of day D+1 (also starting at day D+1 00:00), so consecutive
    global indices are always contiguous in time.
    """
    week_start, _ = _week_window(today)
    if selected is None:
        return [
            (
                (week_start + timedelta(days=d)).isoformat(timespec="seconds"),
                (week_start + timedelta(days=d + 1)).isoformat(timespec="seconds"),
            )
            for d in range(DAYS_IN_WEEK)
        ]
    if not selected:
        return []
    sorted_idx = sorted(selected)
    ranges: list[tuple[str, str]] = []
    run_start = sorted_idx[0]
    prev = run_start
    for i in sorted_idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        ranges.append((
            _slot_dt(run_start, week_start).isoformat(timespec="seconds"),
            _slot_end_dt(prev, week_start).isoformat(timespec="seconds"),
        ))
        run_start = i
        prev = i
    ranges.append((
        _slot_dt(run_start, week_start).isoformat(timespec="seconds"),
        _slot_end_dt(prev, week_start).isoformat(timespec="seconds"),
    ))
    return ranges


def _slot_index_for(captured_at: str, week_start: datetime) -> int | None:
    """Map a captured_at ISO string to a slot index 0..TOTAL_SLOTS-1.

    Edge slots absorb overflow:
      - hour < DAY_START_HOUR or (hour == DAY_START_HOUR and minute < 30)
        → slot 0 of that day.
      - hour >= DAY_END_HOUR → last slot of that day.
    Returns None only if captured_at is malformed or outside the 7-day window.
    """
    try:
        dt = datetime.fromisoformat(captured_at)
    except ValueError:
        return None
    if dt < week_start:
        return None
    day_idx = (dt.date() - week_start.date()).days
    if day_idx < 0 or day_idx >= DAYS_IN_WEEK:
        return None
    if dt.hour < DAY_START_HOUR:
        slot_in_day = 0
    elif dt.hour >= DAY_END_HOUR:
        slot_in_day = SLOTS_PER_DAY - 1
    else:
        slot_in_day = (dt.hour - DAY_START_HOUR) * 2 + (1 if dt.minute >= 30 else 0)
    return day_idx * SLOTS_PER_DAY + slot_in_day


def _heat_class(count: int, max_count: int) -> int:
    if count <= 0 or max_count <= 0:
        return 0
    # 5 non-empty buckets; sqrt scaling so light traffic isn't all dark
    import math
    ratio = math.sqrt(count / max_count)
    return min(5, max(1, int(ratio * 5 + 0.999)))


def _heat_class_speed(value: float | None, min_v: float, max_v: float) -> int:
    """Linear bucketing for speed mode (top mph per slot). Spreads observed
    range over 5 buckets so the slowest non-empty cell is heat-1 and the
    fastest is heat-5."""
    if value is None or value <= 0 or max_v <= 0:
        return 0
    if max_v <= min_v:
        return 3
    ratio = (value - min_v) / (max_v - min_v)
    return min(5, max(1, int(ratio * 5 + 0.999)))


def _build_heatmap(
    rows_in_window: list[dict],  # already direction- and alerts_only-filtered, non-deleted
    today: date,
    selected: set[int] | None,
) -> dict:
    """Build heatmap context: cells with count + heat class + selected state,
    plus column/row headers and the serialized mask for the hidden input."""
    week_start, _ = _week_window(today)
    counts = [0] * TOTAL_SLOTS
    sum_mph = [0.0] * TOTAL_SLOTS
    valid_count = [0] * TOTAL_SLOTS
    top_mph: list[float | None] = [None] * TOTAL_SLOTS
    for r in rows_in_window:
        idx = _slot_index_for(r["captured_at"], week_start)
        if idx is None:
            continue
        counts[idx] += 1
        mph = r.get("computed_mph")
        if mph is not None and mph > 0:
            sum_mph[idx] += mph
            valid_count[idx] += 1
            if top_mph[idx] is None or mph > top_mph[idx]:
                top_mph[idx] = mph
    max_count = max(counts) if any(counts) else 0
    avg_mph: list[float | None] = [
        (sum_mph[i] / valid_count[i]) if valid_count[i] > 0 else None
        for i in range(TOTAL_SLOTS)
    ]
    valid_avgs = [a for a in avg_mph if a is not None]
    min_avg = min(valid_avgs) if valid_avgs else 0.0
    max_avg = max(valid_avgs) if valid_avgs else 0.0
    valid_tops = [t for t in top_mph if t is not None]
    min_top = min(valid_tops) if valid_tops else 0.0
    max_top = max(valid_tops) if valid_tops else 0.0

    is_selected = (lambda i: True) if selected is None else (lambda i: i in selected)

    def slot_time_label(slot: int) -> str:
        if slot == 0:
            return f"00:00-{DAY_START_HOUR:02d}:30"
        if slot == SLOTS_PER_DAY - 1:
            return f"{DAY_END_HOUR - 1:02d}:30-24:00"
        hour = DAY_START_HOUR + slot // 2
        minute = (slot % 2) * 30
        return f"{hour:02d}:{minute:02d}"

    def slot_time_range(slot: int) -> str:
        if slot == 0:
            return f"00:00-{DAY_START_HOUR:02d}:30"
        if slot == SLOTS_PER_DAY - 1:
            return f"{DAY_END_HOUR - 1:02d}:30-24:00"
        hour = DAY_START_HOUR + slot // 2
        minute = (slot % 2) * 30
        end_hour = hour + (1 if minute == 30 else 0)
        end_minute = 0 if minute == 30 else 30
        return f"{hour:02d}:{minute:02d}-{end_hour:02d}:{end_minute:02d}"

    day_label_for = {}
    for day in range(DAYS_IN_WEEK):
        d = today - timedelta(days=DAYS_IN_WEEK - 1 - day)
        day_label_for[day] = f"{d.strftime('%a')} {d.month}/{d.day}"

    cells: list[dict] = []
    for slot in range(SLOTS_PER_DAY):
        for day in range(DAYS_IN_WEEK):
            i = day * SLOTS_PER_DAY + slot
            cells.append({
                "day": day,
                "slot": slot,
                "index": i,
                "count": counts[i],
                "avg_mph": avg_mph[i],
                "top_mph": top_mph[i],
                "heat_count": _heat_class(counts[i], max_count),
                "heat_speed": _heat_class_speed(top_mph[i], min_top, max_top),
                "selected": is_selected(i),
                "time_label": slot_time_label(slot),
                "time_range": slot_time_range(slot),
                "day_label": day_label_for[day],
            })

    day_headers: list[dict] = []
    for day in range(DAYS_IN_WEEK):
        d = today - timedelta(days=DAYS_IN_WEEK - 1 - day)
        day_headers.append({
            "day": day,
            "weekday": d.strftime("%a"),
            "label": f"{d.month}/{d.day}",
            "iso": d.isoformat(),
        })

    slot_headers: list[dict] = []
    for slot in range(SLOTS_PER_DAY):
        # Show a "HH:00" label every 2 hours; blank on the in-between rows.
        hour = DAY_START_HOUR + slot // 2
        if slot % 4 == 0:
            label = f"{hour:02d}:00"
        else:
            label = ""
        slot_headers.append({
            "slot": slot,
            "label": label,
            "time_label": slot_time_label(slot),
        })

    initial_mask = "".join("1" if is_selected(i) else "0" for i in range(TOTAL_SLOTS))

    return {
        "heatmap_cells": cells,
        "heatmap_day_headers": day_headers,
        "heatmap_slot_headers": slot_headers,
        "heatmap_max_count": max_count,
        "heatmap_max_avg_mph": max_avg,
        "heatmap_max_top_mph": max_top,
        "heatmap_days": DAYS_IN_WEEK,
        "time_mask": initial_mask,
        "time_ref_date": today.isoformat(),
    }


def _static_version() -> str:
    """File-mtime-based cache buster so CSS/JS edits invalidate cached assets.

    Mobile Safari is aggressive about reusing cached static files without
    sending conditional GETs, which would otherwise leave users staring at
    a stale stylesheet after we change it. Appending ?v=<mtime> on the
    asset URLs gives each version a unique URL that bypasses the cache.
    """
    css = STATIC_DIR / "style.css"
    try:
        return str(int(css.stat().st_mtime))
    except OSError:
        return "0"


DEFAULT_PAGE_SIZE = 20
ALLOWED_PAGE_SIZES = (10, 20, 50, 100, 200)


def _normalize_page_size(value: int | None) -> int:
    """Clamp `value` to one of ALLOWED_PAGE_SIZES; fall back to default."""
    try:
        v = int(value) if value is not None else DEFAULT_PAGE_SIZE
    except (TypeError, ValueError):
        return DEFAULT_PAGE_SIZE
    return v if v in ALLOWED_PAGE_SIZES else DEFAULT_PAGE_SIZE


def _render_index(request: Request, cfg: Config, db: Database):
    cal = cfg.load_calibration()
    dist_n = cal.line_distance_m_north if cal else 0
    dist_s = cal.line_distance_m_south if cal else 0
    threshold = cfg.alert_threshold_mph
    today = _today_local()
    week_start, week_end = _week_window(today)

    # Initial paint: no direction, alerts-only ON, time selection = "today
    # only" (rightmost column of the week grid, day index 6). Mirrors the
    # default checkbox state in the template.
    today_slots = set(range(
        (DAYS_IN_WEEK - 1) * SLOTS_PER_DAY,
        DAYS_IN_WEEK * SLOTS_PER_DAY,
    ))
    week_rows = db.list_passes(
        alerts_only=True,
        threshold_mph=threshold,
        line_distance_m_north=dist_n,
        line_distance_m_south=dist_s,
        limit=10000,
        include_deleted=True,
        time_ranges=[(
            week_start.isoformat(timespec="seconds"),
            week_end.isoformat(timespec="seconds"),
        )],
    )
    rendered_all = [render_pass(p, dist_n, dist_s, threshold) for p in week_rows]
    week_non_deleted = [r for r in rendered_all if not r["deleted"]]

    # Filter list + histogram to today only; heatmap colors still reflect the
    # full week so the user can see which other days have traffic.
    rendered_filtered: list[dict] = []
    for r in rendered_all:
        idx = _slot_index_for(r["captured_at"], week_start)
        if idx is not None and idx in today_slots:
            rendered_filtered.append(r)
    hist_rows = [r for r in rendered_filtered if not r["deleted"]]
    histogram, hist_total, hist_all_default = _build_histogram(hist_rows, threshold, set())
    heatmap_ctx = _build_heatmap(week_non_deleted, today, selected=today_slots)

    total_filtered = len(rendered_filtered)
    page_size = DEFAULT_PAGE_SIZE
    total_pages = max(1, (total_filtered + page_size - 1) // page_size)
    rows = rendered_filtered[:page_size]
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "static_v": _static_version(),
            "rows": rows,
            "threshold": threshold,
            "clips_days": cfg.clips_days,
            "thumbs_days": cfg.thumbs_days,
            "passes_days": cfg.passes_days,
            "heatmap_days": cfg.heatmap_days,
            "clip_margin_s": cfg.clip_margin_s,
            "clip_capture_min_mph": cfg.clip_capture_min_mph,
            "clip_capture_max_mph": cfg.clip_capture_max_mph,
            "preview_show_grid": cfg.preview_show_grid,
            "pause_at_night": cfg.pause_at_night,
            "upload_enabled": cfg.upload_enabled,
            "running": True,
            "paused_night": _is_paused_night(request, cfg),
            **_load_homography_meta(cfg),
            "histogram": histogram,
            "histogram_total": hist_total,
            "histogram_all_default": hist_all_default,
            "page": 1,
            "page_size": page_size,
            "page_size_options": ALLOWED_PAGE_SIZES,
            "total_pages": total_pages,
            "total_filtered": total_filtered,
            **heatmap_ctx,
        },
    )


def _render_pass_list(
    request: Request, cfg: Config, db: Database,
    direction: str | None, alerts_only: bool,
    time_mask: str | None = None, time_ref_date: str | None = None,
    selected_buckets: list[int] | None = None,
    page: int = 1,
    page_size: int | None = None,
    vehicle_make: str | None = None,
    vehicle_model: str | None = None,
    vehicle_color: str | None = None,
):
    cal = cfg.load_calibration()
    dist_n = cal.line_distance_m_north if cal else 0
    dist_s = cal.line_distance_m_south if cal else 0
    threshold = cfg.alert_threshold_mph

    direction = direction if direction in ("N", "S") else None
    selected_buckets_set: set[int] = set(selected_buckets or [])
    page = max(1, int(page or 1))

    today = _today_local()
    week_start, week_end = _week_window(today)
    selected_slots = _parse_time_mask(time_mask, time_ref_date, today)

    # Heatmap counts use the trailing-7-day window with direction + alerts_only
    # applied, but ignore the slot mask (so users see what's selectable, not
    # just what's selected). One DB fetch over the week is enough; we filter
    # the same rows by mask in Python for the histogram + list.
    week_rows = db.list_passes(
        direction=direction,
        alerts_only=alerts_only,
        threshold_mph=threshold,
        line_distance_m_north=dist_n,
        line_distance_m_south=dist_s,
        limit=10000,
        include_deleted=True,
        time_ranges=[(
            week_start.isoformat(timespec="seconds"),
            week_end.isoformat(timespec="seconds"),
        )],
    )
    rendered_week = [render_pass(p, dist_n, dist_s, threshold) for p in week_rows]

    # Speed-bucket + vehicle predicates, extracted so they can narrow the
    # heatmap input as well as the list. The slot mask is deliberately NOT
    # applied to the heatmap (the heatmap is its own picker for that axis).
    veh_make = (vehicle_make or "").strip() or None
    veh_model = (vehicle_model or "").strip() or None
    veh_color = (vehicle_color or "").strip() or None
    color_allowed = (
        COLOR_MATCHES.get(veh_color, {veh_color}) if veh_color else None
    )
    vehicle_active = bool(veh_make or veh_model or veh_color)

    def speed_bucket_matches(r: dict) -> bool:
        if not selected_buckets_set:
            return True
        mph = r["computed_mph"]
        return mph is not None and _bucket_for(mph) in selected_buckets_set

    def vehicle_matches(r: dict) -> bool:
        if not vehicle_active:
            return True
        if veh_make and r.get("vehicle_make") != veh_make:
            return False
        if veh_model and r.get("vehicle_model") != veh_model:
            return False
        if color_allowed is not None and r.get("vehicle_color") not in color_allowed:
            return False
        return True

    # Heatmap rows: everything except the slot mask. Deleted rows are dropped
    # for cell counts (heatmap is a live-density view, not a deletion log).
    heatmap_rows = [
        r for r in rendered_week
        if not r["deleted"] and speed_bucket_matches(r) and vehicle_matches(r)
    ]
    heatmap_ctx = _build_heatmap(heatmap_rows, today, selected_slots)

    # Filter the rendered week rows by the slot mask in Python.
    if selected_slots is None:
        rendered_filtered = rendered_week
    elif not selected_slots:
        rendered_filtered = []
    else:
        rendered_filtered = []
        for r in rendered_week:
            idx = _slot_index_for(r["captured_at"], week_start)
            if idx is not None and idx in selected_slots:
                rendered_filtered.append(r)

    hist_rows = [r for r in rendered_filtered if not r["deleted"]]
    histogram, hist_total, hist_all_default = _build_histogram(hist_rows, threshold, selected_buckets_set)

    # Speed-bucket selection narrows further (deleted rows pass through so the
    # ↩ Revert affordance stays visible regardless of facet).
    if selected_buckets_set:
        rendered_filtered = [r for r in rendered_filtered if r["deleted"] or speed_bucket_matches(r)]

    # Vehicle filter: make + model are exact match; color expands via
    # COLOR_MATCHES (recall-favoring against lighting drift).
    if vehicle_active:
        rendered_filtered = [r for r in rendered_filtered if r["deleted"] or vehicle_matches(r)]

    # Auto-hide stale soft-deletes: keep the ↩ Revert affordance visible for
    # 60s after deletion, then drop the row from the list entirely. Rows
    # without a deleted_at (pre-migration deletes) are treated as already stale.
    stale_cutoff = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(timespec="seconds")
    rendered_filtered = [
        r for r in rendered_filtered
        if not r["deleted"] or (r.get("deleted_at") and r["deleted_at"] >= stale_cutoff)
    ]

    page_size = _normalize_page_size(page_size)
    total_filtered = len(rendered_filtered)
    total_pages = max(1, (total_filtered + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    rendered = rendered_filtered[start:start + page_size]

    return TEMPLATES.TemplateResponse(
        request,
        "_pass_list.html",
        {
            "rows": rendered,
            "histogram": histogram,
            "histogram_total": hist_total,
            "histogram_all_default": hist_all_default,
            "include_oob_histogram": True,
            "include_oob_heatmap": True,
            "page": page,
            "page_size": page_size,
            "page_size_options": ALLOWED_PAGE_SIZES,
            "total_pages": total_pages,
            "total_filtered": total_filtered,
            "vehicle_make": veh_make,
            "vehicle_model": veh_model,
            "vehicle_color": veh_color,
            **heatmap_ctx,
        },
    )


def _render_pass_row(request: Request, cfg: Config, db: Database, pass_id: int):
    cal = cfg.load_calibration()
    dist_n = cal.line_distance_m_north if cal else 0
    dist_s = cal.line_distance_m_south if cal else 0
    threshold = cfg.alert_threshold_mph
    p = db.get_pass(pass_id)
    if p is None:
        raise HTTPException(status_code=404)
    return TEMPLATES.TemplateResponse(
        request,
        "_pass_row.html",
        {"row": render_pass(p, dist_n, dist_s, threshold)},
    )


def _is_paused_night(request: Request, cfg: Config) -> bool:
    """True iff the live worker reports night-mode AND the gate is enabled.
    Used by the initial page render so the status badge shows the correct
    state on first paint, before the HTMX poll fires."""
    worker = getattr(request.app.state, "worker", None) if hasattr(request, "app") else None
    if worker is None or not worker.is_alive():
        return False
    return bool(cfg.pause_at_night and worker.is_night_mode())


def _load_homography_meta(cfg: Config) -> dict:
    """Best-effort read of the elected main camera's calibration metadata
    (registry profile, ADR-015) for display in the header. Returns empty
    dict if no homography is calibrated."""
    try:
        data = cfg.camera.profile.calibration()
    except Exception:  # noqa: BLE001 — CalibrationMissing or a bad artifact
        return {}
    n_pts = len(data.get("pixel_pts") or data.get("pixel_pts_sub") or [])
    mean_m = float(data.get("mean_reprojection_error_m") or 0.0)
    return {
        "homog_n_pts": n_pts,
        "homog_mean_err_cm": mean_m * 100.0,
    }


def _render_status_panel(request: Request, cfg: Config, db: Database):
    return TEMPLATES.TemplateResponse(
        request,
        "_status.html",
        {
            "threshold": cfg.alert_threshold_mph,
            "running": True,
            **_load_homography_meta(cfg),
        },
    )


# ---------- entrypoint ----------

def serve(host: str = "127.0.0.1", port: int = 8000, profile: bool = False) -> None:
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = make_app(profile=profile)
    uvicorn.run(app, host=host, port=port, log_level="info")
