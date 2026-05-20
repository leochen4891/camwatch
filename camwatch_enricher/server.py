"""FastAPI app for the local enrichment service."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import EnricherConfig, load_config
from .decision import decide
from .embedder import Embedder
from .index import KnnIndex
from .store import apply_decision, get_clip_path, thumb_path_from_clip, upsert_embedding

log = logging.getLogger(__name__)


class EnrichRequest(BaseModel):
    pass_id: int


class TopMatch(BaseModel):
    pass_id: int
    make: str | None
    model: str | None
    sim: float


class EnrichResponse(BaseModel):
    pass_id: int
    status: str
    make: str | None = None
    model: str | None = None
    color: str | None = None
    top_sim: float
    top_matches: list[TopMatch]


class HealthResponse(BaseModel):
    ok: bool
    indexed_n: int
    device: str
    model: str


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
        thumb = thumb_path_from_clip(clip_path)
        if not thumb.exists():
            raise HTTPException(status_code=404, detail=f"thumbnail missing: {thumb}")

        try:
            vec = embedder.encode_path(thumb)
        except Exception as e:
            log.exception("encode failed for pass %s", req.pass_id)
            raise HTTPException(status_code=500, detail=f"encode failed: {e}") from e

        upsert_embedding(db_path, req.pass_id, vec, embedder.model_name)

        neighbors = index.topk(vec, k=cfg.decision.k, exclude_pass_id=req.pass_id)
        d = decide(neighbors, k_agree=cfg.decision.k_agree, tau_high=cfg.decision.tau_high)
        apply_decision(db_path, req.pass_id, d)

        return EnrichResponse(
            pass_id=req.pass_id,
            status=d.status,
            make=d.make,
            model=d.model,
            color=d.color,
            top_sim=d.top_sim,
            top_matches=[
                TopMatch(pass_id=n.pass_id, make=n.make, model=n.model, sim=n.sim)
                for n in d.topk
            ],
        )

    return app
