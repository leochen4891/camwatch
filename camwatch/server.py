"""FastAPI app: pass review, annotation, threshold, live capture worker.

Run:
    uv run python -m camwatch serve [--host 127.0.0.1] [--port 8000]

The capture worker is started in the lifespan handler and joined on shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .capture_worker import CaptureWorker
from .config import Config, load_config
from .db import Database, Pass
from .preview import PreviewBuffer

log = logging.getLogger(__name__)

MPS_TO_MPH = 2.2369362920544
HERE = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=HERE / "templates")
STATIC_DIR = HERE / "static"


# ---------- helpers ----------

def computed_mph(p: Pass, dist_n: float, dist_s: float) -> float | None:
    if p.elapsed_s <= 0:
        return None
    d = dist_n if p.direction == "N" else dist_s
    if not d or d <= 0:
        return None
    return (d / p.elapsed_s) * MPS_TO_MPH


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
        "alert": (mph is not None and mph >= threshold and p.known_mph is None),
        "has_clip": has_clip,
        "has_thumb": has_thumb,
        "thumb_upgrade_status": p.thumb_upgrade_status,
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

    preview = PreviewBuffer()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker = CaptureWorker(cfg, db, preview=preview, profile=profile)
        worker.start()
        app.state.worker = worker
        app.state.cfg = cfg
        app.state.db = db
        app.state.preview = preview
        log.info("server startup complete")
        try:
            yield
        finally:
            log.info("server shutdown: stopping capture worker")
            worker.stop()
            worker.join(timeout=10)

    app = FastAPI(lifespan=lifespan, title="camwatch")
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---------- routes ----------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return _render_index(request, cfg, db)

    @app.get("/passes", response_class=HTMLResponse)
    async def list_passes_partial(
        request: Request,
        direction: str | None = None,
        alerts_only: bool = False,
        from_ts: str | None = None,
        to_ts: str | None = None,
        buckets: list[int] = Query(default=[]),
        page: int = 1,
    ):
        return _render_pass_list(
            request, cfg, db, direction, alerts_only,
            from_ts=from_ts, to_ts=to_ts, selected_buckets=buckets, page=page,
        )

    @app.post("/passes/{pass_id}/annotate", response_class=HTMLResponse)
    async def annotate_pass(request: Request, pass_id: int, known_mph: str = Form(...)):
        p = db.get_pass(pass_id)
        if p is None or p.deleted:
            raise HTTPException(status_code=404, detail="not found")
        try:
            value: float | None = float(known_mph) if known_mph.strip() else None
        except ValueError:
            raise HTTPException(status_code=400, detail="known_mph must be a number")
        db.set_known_mph(pass_id, value)
        recompute_calibration(cfg, db)
        return _render_pass_row(request, cfg, db, pass_id)

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

    @app.get("/passes/{pass_id}/thumb")
    async def get_thumb(pass_id: int, big: bool = False):
        p = db.get_pass(pass_id)
        if p is None or not p.clip_path:
            raise HTTPException(status_code=404)
        base = p.clip_path[:-4] if p.clip_path.endswith(".mp4") else p.clip_path
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
        return Response(content=jpeg, media_type="image/jpeg")

    @app.post("/preview/show_lines")
    async def set_preview_show_lines(show: bool = Form(False)):
        preview.set_show_lines(show)
        return Response(status_code=204)

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
        show_lines: str | None = Form(default=None),
        retention_days: int = Form(default=0),
        clip_margin_s: float = Form(default=0.5),
        clip_capture_min_mph: float = Form(default=0.0),
        clip_capture_max_mph: float = Form(default=999.0),
    ):
        cfg_path = Path("config/config.yaml")
        margin = max(0.0, float(clip_margin_s))
        cap_min = max(0.0, float(clip_capture_min_mph))
        cap_max = max(cap_min, float(clip_capture_max_mph))
        # Persist threshold, retention, and clip settings to config.yaml
        with cfg_path.open() as f:
            data = yaml.safe_load(f) or {}
        data.setdefault("alert", {})["threshold_mph"] = float(threshold_mph)
        data.setdefault("retention", {})["days"] = max(0, int(retention_days))
        clip_section = data.setdefault("clip", {})
        clip_section["margin_s"] = margin
        clip_section["capture_min_mph"] = cap_min
        clip_section["capture_max_mph"] = cap_max
        with cfg_path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        cfg.alert_threshold_mph = float(threshold_mph)
        cfg.retention_days = max(0, int(retention_days))
        cfg.clip_margin_s = margin
        cfg.clip_capture_min_mph = cap_min
        cfg.clip_capture_max_mph = cap_max
        # Push the new margin to the running capture worker without a restart.
        worker = getattr(request.app.state, "worker", None)
        if worker is not None:
            worker.update_clip_margin(margin)
        # Show-lines is runtime-only (resets on server restart). Unchecked
        # checkboxes don't post a value, so a missing field means False.
        preview.set_show_lines(show_lines is not None)
        # Trigger an immediate retention sweep if enabled
        if cfg.retention_days > 0:
            n, clips = db.purge_older_than(cfg.retention_days)
            for cp in clips:
                try:
                    Path(cp).unlink(missing_ok=True)
                    Path(cp[:-4] + ".jpg").unlink(missing_ok=True)
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

    return app


# ---------- render helpers (split out for clarity) ----------

def _bucket_for(mph: float) -> int:
    """5-mph buckets: 1..5 -> 0, 6..10 -> 1, …. Pre-1 mph clamps to 0."""
    if mph < 1:
        return 0
    return int((mph - 1) // 5)


def _bucket_label(idx: int) -> str:
    return f"{idx * 5 + 1}-{idx * 5 + 5}"


def _build_histogram(
    rows_for_hist: list[dict],
    threshold: float,
    selected_buckets: set[int],
) -> tuple[list[dict], int]:
    """Returns (bars, total_count). Each bar dict has idx, label, count,
    height_pct, selected. Bars are emitted from 0 up through max non-empty
    bucket, with at least 2 buckets past the alarm threshold so high-speed
    buckets remain visible/clickable."""
    counts: dict[int, int] = {}
    total = 0
    for r in rows_for_hist:
        if r["computed_mph"] is None:
            continue
        idx = _bucket_for(r["computed_mph"])
        counts[idx] = counts.get(idx, 0) + 1
        total += 1

    threshold_bucket = _bucket_for(threshold)
    max_observed = max(counts.keys(), default=-1)
    last = max(max_observed, threshold_bucket + 2)
    if last < 0:
        return [], 0

    max_count = max(counts.values(), default=0) or 1
    bars: list[dict] = []
    for idx in range(0, last + 1):
        c = counts.get(idx, 0)
        bars.append({
            "idx": idx,
            "label": _bucket_label(idx),
            "count": c,
            "height_pct": int(round(c / max_count * 100)),
            "selected": idx in selected_buckets,
        })
    return bars, total


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


def _render_index(request: Request, cfg: Config, db: Database):
    cal = cfg.load_calibration()
    dist_n = cal.line_distance_m_north if cal else 0
    dist_s = cal.line_distance_m_south if cal else 0
    threshold = cfg.alert_threshold_mph
    all_rows = [
        render_pass(p, dist_n, dist_s, threshold)
        for p in db.list_passes(limit=10000, include_deleted=True)
    ]
    hist_rows = [r for r in all_rows if not r["deleted"]]
    histogram, hist_total = _build_histogram(hist_rows, threshold, set())
    total_filtered = len(all_rows)
    total_pages = max(1, (total_filtered + PAGE_SIZE - 1) // PAGE_SIZE)
    rows = all_rows[:PAGE_SIZE]
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "static_v": _static_version(),
            "rows": rows,
            "dist_n": dist_n,
            "dist_s": dist_s,
            "threshold": threshold,
            "retention_days": cfg.retention_days,
            "clip_margin_s": cfg.clip_margin_s,
            "clip_capture_min_mph": cfg.clip_capture_min_mph,
            "clip_capture_max_mph": cfg.clip_capture_max_mph,
            "known_count": len(cal.calibration_points) if cal else 0,
            "running": True,
            "histogram": histogram,
            "histogram_total": hist_total,
            "page": 1,
            "total_pages": total_pages,
            "total_filtered": total_filtered,
        },
    )


PAGE_SIZE = 100


def _render_pass_list(
    request: Request, cfg: Config, db: Database,
    direction: str | None, alerts_only: bool,
    from_ts: str | None = None, to_ts: str | None = None,
    selected_buckets: list[int] | None = None,
    page: int = 1,
):
    cal = cfg.load_calibration()
    dist_n = cal.line_distance_m_north if cal else 0
    dist_s = cal.line_distance_m_south if cal else 0
    threshold = cfg.alert_threshold_mph
    direction = direction if direction in ("N", "S") else None
    selected_set: set[int] = set(selected_buckets or [])
    page = max(1, int(page or 1))

    # Histogram needs the FULL filtered (by direction/alerts/time) set, not
    # just the current page. Pull a wide list with a generous cap.
    all_filtered = db.list_passes(
        direction=direction,
        alerts_only=alerts_only,
        threshold_mph=threshold,
        line_distance_m_north=dist_n,
        line_distance_m_south=dist_s,
        limit=10000,
        include_deleted=True,
        from_ts=from_ts,
        to_ts=to_ts,
    )
    rendered_all = [render_pass(p, dist_n, dist_s, threshold) for p in all_filtered]

    hist_rows = [r for r in rendered_all if not r["deleted"]]
    histogram, hist_total = _build_histogram(hist_rows, threshold, selected_set)

    # List filter: bucket selection narrows further (deleted rows pass through).
    if selected_set:
        def in_selected(r: dict) -> bool:
            if r["deleted"]:
                return True
            mph = r["computed_mph"]
            if mph is None:
                return False
            return _bucket_for(mph) in selected_set
        rendered_all = [r for r in rendered_all if in_selected(r)]

    total_filtered = len(rendered_all)
    total_pages = max(1, (total_filtered + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, total_pages)
    start = (page - 1) * PAGE_SIZE
    rendered = rendered_all[start:start + PAGE_SIZE]

    return TEMPLATES.TemplateResponse(
        request,
        "_pass_list.html",
        {
            "rows": rendered,
            "histogram": histogram,
            "histogram_total": hist_total,
            "include_oob_histogram": True,
            "page": page,
            "total_pages": total_pages,
            "total_filtered": total_filtered,
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


def _render_status_panel(request: Request, cfg: Config, db: Database):
    cal = cfg.load_calibration()
    dist_n = cal.line_distance_m_north if cal else 0
    dist_s = cal.line_distance_m_south if cal else 0
    return TEMPLATES.TemplateResponse(
        request,
        "_status.html",
        {
            "dist_n": dist_n,
            "dist_s": dist_s,
            "threshold": cfg.alert_threshold_mph,
            "known_count": len(cal.calibration_points) if cal else 0,
            "running": True,
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
