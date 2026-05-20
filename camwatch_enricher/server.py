"""FastAPI app for the local enrichment service."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import EnricherConfig, load_config
from .decision import Decision, decide
from .embedder import Embedder
from .index import KnnIndex, Neighbor
from .store import anchor_paths_from_clip, apply_decision, get_clip_path, upsert_embedding

log = logging.getLogger(__name__)


class EnrichRequest(BaseModel):
    pass_id: int


class TopMatch(BaseModel):
    pass_id: int
    make: str | None
    model: str | None
    sim: float


class ViewResult(BaseModel):
    view: str               # 'thumb' | 'entry' | 'exit'
    status: str
    make: str | None
    model: str | None
    top_sim: float


class EnrichResponse(BaseModel):
    pass_id: int
    status: str
    make: str | None = None
    model: str | None = None
    color: str | None = None
    top_sim: float
    top_matches: list[TopMatch]
    views: list[ViewResult]


class HealthResponse(BaseModel):
    ok: bool
    indexed_n: int
    device: str
    model: str


def _combine_views(per_view: dict[str, Decision]) -> tuple[Decision, list[ViewResult]]:
    """Combine per-view decisions into a single decision.

    Rules (in order):
      1. If ANY view fires `high`, take that view's label — one confident
         view is enough on its own.
      2. Otherwise, if EVERY available view lands at >= medium AND they
         all agree on (make, model), take that agreed label.
      3. Otherwise, `low` — defer to the Opus workflow.

    The thumb view drives top_sim / topk / color of the combined decision
    so downstream UI keeps a single canonical preview.
    """
    view_results = [
        ViewResult(
            view=v, status=d.status, make=d.make, model=d.model, top_sim=d.top_sim
        )
        for v, d in per_view.items()
    ]
    if not per_view:
        return Decision("no_match", None, None, None, 0.0, 0, []), view_results

    thumb_d = per_view.get("thumb") or next(iter(per_view.values()))

    # Rule 1: any view at high → accept that view.
    for d in per_view.values():
        if d.status == "high":
            return Decision(
                "high", d.make, d.model, d.color,
                thumb_d.top_sim, d.agree_count, thumb_d.topk,
            ), view_results

    # Rule 2: every available view at >= medium AND unanimous label.
    all_at_least_medium = all(d.status in ("high", "medium") for d in per_view.values())
    labels = {(d.make, d.model) for d in per_view.values()}
    if all_at_least_medium and len(labels) == 1:
        mk, md = labels.pop()
        return Decision(
            "high", mk, md, thumb_d.color,
            thumb_d.top_sim, thumb_d.agree_count, thumb_d.topk,
        ), view_results

    # Rule 3: not confident — let Opus handle it.
    return Decision(
        "low", None, None, None,
        thumb_d.top_sim, 0, thumb_d.topk,
    ), view_results


def create_app(cfg: EnricherConfig | None = None) -> FastAPI:
    cfg = cfg or load_config()
    db_path = Path(cfg.paths.db)

    embedder = Embedder(model_name=cfg.model.name, device=cfg.model.device)
    index = KnnIndex(db_path=db_path, model_name=cfg.model.name)

    app = FastAPI(title="camwatch-enricher")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            ok=True,
            indexed_n=index.size(),
            device=embedder.device,
            model=embedder.model_name,
        )

    @app.post("/index/refresh", response_model=HealthResponse)
    def refresh() -> HealthResponse:
        index.refresh()
        return health()

    @app.post("/enrich", response_model=EnrichResponse)
    def enrich(req: EnrichRequest) -> EnrichResponse:
        clip_path = get_clip_path(db_path, req.pass_id)
        if not clip_path:
            raise HTTPException(status_code=404, detail="pass not found or has no clip_path")
        paths = anchor_paths_from_clip(clip_path)
        if not paths["thumb"].exists():
            raise HTTPException(status_code=404, detail=f"thumbnail missing: {paths['thumb']}")

        per_view: dict[str, Decision] = {}
        thumb_vec = None
        for view, p in paths.items():
            if not p.exists():
                continue
            try:
                vec = embedder.encode_path(p)
            except Exception as e:
                log.warning("encode failed for pass %s view %s: %s", req.pass_id, view, e)
                continue
            if view == "thumb":
                thumb_vec = vec
            neighbors = index.topk(vec, k=cfg.decision.k, exclude_pass_id=req.pass_id)
            per_view[view] = decide(
                neighbors,
                k_agree_high=cfg.decision.k_agree, tau_high=cfg.decision.tau_high,
                k_agree_medium=cfg.decision.k_agree_medium, tau_medium=cfg.decision.tau_medium,
            )

        if thumb_vec is None:
            raise HTTPException(status_code=500, detail="thumb encode failed")
        # Only the thumb embedding goes in the index (so labeled-set growth
        # mirrors what Opus-labeled rows store today). Entry/exit are used
        # for confirmation only.
        upsert_embedding(db_path, req.pass_id, thumb_vec, embedder.model_name)

        combined, view_results = _combine_views(per_view)
        apply_decision(db_path, req.pass_id, combined)

        return EnrichResponse(
            pass_id=req.pass_id,
            status=combined.status,
            make=combined.make,
            model=combined.model,
            color=combined.color,
            top_sim=combined.top_sim,
            top_matches=[
                TopMatch(pass_id=n.pass_id, make=n.make, model=n.model, sim=n.sim)
                for n in combined.topk
            ],
            views=view_results,
        )

    return app
