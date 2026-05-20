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
    k_agree: int = 3
    tau_high: float = 0.85


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
            k_agree=int(d.get("k_agree", cfg.decision.k_agree)),
            tau_high=float(d.get("tau_high", cfg.decision.tau_high)),
        )
    return cfg
