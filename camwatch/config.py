from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import yaml
from dotenv import load_dotenv


@dataclass
class CameraConfig:
    host: str
    port: int
    path: str
    user: str
    password: str

    @property
    def rtsp_url(self) -> str:
        u = quote(self.user, safe="")
        p = quote(self.password, safe="")
        return f"rtsp://{u}:{p}@{self.host}:{self.port}{self.path}"


@dataclass
class ModelConfig:
    weights: str
    device: str
    conf: float
    iou: float
    classes: list[int]


@dataclass
class CalibrationConfig:
    line_a_x: int
    line_b_x: int
    frame_width: int
    frame_height: int
    line_distance_m_north: float
    line_distance_m_south: float
    passes: list[dict] = field(default_factory=list)


@dataclass
class Config:
    camera: CameraConfig
    model: ModelConfig
    alert_threshold_mph: float
    events_dir: Path
    calibration_path: Path
    max_track_age_s: float

    def load_calibration(self) -> CalibrationConfig | None:
        if not self.calibration_path.exists():
            return None
        with self.calibration_path.open() as f:
            data = yaml.safe_load(f) or {}
        return CalibrationConfig(
            line_a_x=int(data["line_a_x"]),
            line_b_x=int(data["line_b_x"]),
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
            line_distance_m_north=float(data.get("line_distance_m_north", 0.0)),
            line_distance_m_south=float(data.get("line_distance_m_south", 0.0)),
            passes=list(data.get("passes") or []),
        )


def load_config(path: str | Path = "config/config.yaml") -> Config:
    load_dotenv()

    user = os.environ.get("REOLINK_USER")
    pw = os.environ.get("REOLINK_PASS")
    if not user or not pw:
        raise SystemExit(
            "REOLINK_USER and REOLINK_PASS must be set (copy .env.example to .env and fill them in)"
        )

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise SystemExit(
            f"Config not found at {cfg_path}. Copy config/config.example.yaml to {cfg_path}."
        )
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)

    cam = raw["camera"]
    mdl = raw["model"]
    return Config(
        camera=CameraConfig(
            host=cam["host"],
            port=int(cam["port"]),
            path=cam["path"],
            user=user,
            password=pw,
        ),
        model=ModelConfig(
            weights=mdl["weights"],
            device=mdl["device"],
            conf=float(mdl["conf"]),
            iou=float(mdl["iou"]),
            classes=list(mdl["classes"]),
        ),
        alert_threshold_mph=float(raw["alert"]["threshold_mph"]),
        events_dir=Path(raw["paths"]["events_dir"]),
        calibration_path=Path(raw["paths"]["calibration"]),
        max_track_age_s=float(raw["speed"]["max_track_age_s"]),
    )
