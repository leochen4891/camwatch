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
from fastapi import FastAPI, Form, HTTPException, Request
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
    return {
        "id": p.id,
        "deleted": p.deleted,
        "captured_at": p.captured_at,
        "direction": p.direction,
        "elapsed_s": p.elapsed_s,
        "known_mph": p.known_mph,
        "computed_mph": mph,
        "alert": (mph is not None and mph >= threshold and p.known_mph is None),
        "has_clip": bool(p.clip_path),
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

def make_app(cfg: Config | None = None, db_path: Path = Path("camwatch.db")) -> FastAPI:
    cfg = cfg or load_config()
    db = Database(db_path)

    preview = PreviewBuffer()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker = CaptureWorker(cfg, db, preview=preview)
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
    ):
        return _render_pass_list(request, cfg, db, direction, alerts_only)

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
    async def get_thumb(pass_id: int):
        p = db.get_pass(pass_id)
        if p is None or not p.clip_path:
            raise HTTPException(status_code=404)
        thumb_path = Path(p.clip_path.replace(".mp4", ".jpg"))
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
        # mutate the in-memory cfg too
        cfg.alert_threshold_mph = float(threshold_mph)
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

def _render_index(request: Request, cfg: Config, db: Database):
    cal = cfg.load_calibration()
    dist_n = cal.line_distance_m_north if cal else 0
    dist_s = cal.line_distance_m_south if cal else 0
    threshold = cfg.alert_threshold_mph
    rows = [
        render_pass(p, dist_n, dist_s, threshold)
        for p in db.list_passes(limit=200, include_deleted=True)
    ]
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "rows": rows,
            "dist_n": dist_n,
            "dist_s": dist_s,
            "threshold": threshold,
            "known_count": len(cal.calibration_points) if cal else 0,
            "running": True,
        },
    )


def _render_pass_list(
    request: Request, cfg: Config, db: Database,
    direction: str | None, alerts_only: bool,
):
    cal = cfg.load_calibration()
    dist_n = cal.line_distance_m_north if cal else 0
    dist_s = cal.line_distance_m_south if cal else 0
    threshold = cfg.alert_threshold_mph
    direction = direction if direction in ("N", "S") else None
    passes = db.list_passes(
        direction=direction,
        alerts_only=alerts_only,
        threshold_mph=threshold,
        line_distance_m_north=dist_n,
        line_distance_m_south=dist_s,
        limit=200,
        include_deleted=True,
    )
    rows = [render_pass(p, dist_n, dist_s, threshold) for p in passes]
    return TEMPLATES.TemplateResponse(request, "_pass_list.html", {"rows": rows})


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

def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = make_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
