"""Enricher config loader. Reads config/enricher.yaml with defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path("config/enricher.yaml")


@dataclass
class ServiceCfg:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class ModelCfg:
    name: str = "dinov2_vits14"
    device: str = "auto"
    embed_dim: int = 384


@dataclass
class PathsCfg:
    db: str = "camwatch.db"
    recordings: str = "recordings"


@dataclass
class DecisionCfg:
    k: int = 5
    # High tier: single-view confidence threshold. A view meeting these
    # bounds on its own can label the pass.
    min_votes_high: int = 4
    tau_high: float = 0.85
    # Medium tier: looser rule used only by the multi-view combiner — a
    # pass is labeled when every available view lands at >= medium and
    # all views agree on (make, model).
    min_votes_medium: int = 3
    tau_medium: float = 0.80


@dataclass
class EnricherConfig:
    service: ServiceCfg = field(default_factory=ServiceCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    decision: DecisionCfg = field(default_factory=DecisionCfg)


def load_config(path: Path | str = DEFAULT_PATH) -> EnricherConfig:
    p = Path(path)
    if not p.exists():
        return EnricherConfig()
    data: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
    cfg = EnricherConfig()
    if "service" in data:
        s = data["service"]
        cfg.service = ServiceCfg(
            host=s.get("host", cfg.service.host),
            port=int(s.get("port", cfg.service.port)),
        )
    if "model" in data:
        m = data["model"]
        cfg.model = ModelCfg(
            name=m.get("name", cfg.model.name),
            device=m.get("device", cfg.model.device),
            embed_dim=int(m.get("embed_dim", cfg.model.embed_dim)),
        )
    if "paths" in data:
        pp = data["paths"]
        cfg.paths = PathsCfg(
            db=pp.get("db", cfg.paths.db),
            recordings=pp.get("recordings", cfg.paths.recordings),
        )
    if "decision" in data:
        d = data["decision"]
        cfg.decision = DecisionCfg(
            k=int(d.get("k", cfg.decision.k)),
            # Back-compat: older configs used `k_agree` to mean "unanimous
            # in top-k_agree". Map that to min_votes_high when min_votes_high
            # isn't explicitly set.
            min_votes_high=int(
                d.get("min_votes_high", d.get("k_agree", cfg.decision.min_votes_high))
            ),
            tau_high=float(d.get("tau_high", cfg.decision.tau_high)),
            min_votes_medium=int(
                d.get("min_votes_medium", d.get("k_agree_medium", cfg.decision.min_votes_medium))
            ),
            tau_medium=float(d.get("tau_medium", cfg.decision.tau_medium)),
        )
    return cfg
