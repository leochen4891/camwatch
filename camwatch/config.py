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
    path_sub: str | None = None  # optional low-res stream for detection

    @property
    def rtsp_url(self) -> str:
        u = quote(self.user, safe="")
        p = quote(self.password, safe="")
        return f"rtsp://{u}:{p}@{self.host}:{self.port}{self.path}"

    @property
    def rtsp_url_sub(self) -> str | None:
        if not self.path_sub:
            return None
        u = quote(self.user, safe="")
        p = quote(self.password, safe="")
        return f"rtsp://{u}:{p}@{self.host}:{self.port}{self.path_sub}"


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
    # Region of interest fed to YOLO. Anything outside is invisible to detection
    # but still saved in clips and visible to any future security view. Set all
    # four to 0 (or omit) to feed the full frame to YOLO.
    roi_x1: int = 0
    roi_y1: int = 0
    roi_x2: int = 0
    roi_y2: int = 0
    passes: list[dict] = field(default_factory=list)
    calibration_points: list[dict] = field(default_factory=list)

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        """ROI as (x1, y1, x2, y2) in full-frame coords, or None if disabled."""
        if self.roi_x2 > self.roi_x1 and self.roi_y2 > self.roi_y1:
            return (self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2)
        return None


@dataclass
class Config:
    camera: CameraConfig
    model: ModelConfig
    alert_threshold_mph: float
    events_dir: Path
    calibration_path: Path
    max_track_age_s: float
    retention_days: int = 0  # 0 = no auto-delete
    clip_margin_s: float = 0.5  # pre/post-roll padding around the crossing window
    clip_capture_min_mph: float = 0.0  # passes below this speed are logged but skip clip
    clip_capture_max_mph: float = 999.0  # passes above this speed are logged but skip clip

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
            roi_x1=int(data.get("roi_x1", 0) or 0),
            roi_y1=int(data.get("roi_y1", 0) or 0),
            roi_x2=int(data.get("roi_x2", 0) or 0),
            roi_y2=int(data.get("roi_y2", 0) or 0),
            passes=list(data.get("passes") or []),
            calibration_points=list(data.get("calibration_points") or []),
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
            path_sub=cam.get("path_sub"),
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
        retention_days=int((raw.get("retention") or {}).get("days", 0) or 0),
        clip_margin_s=float((raw.get("clip") or {}).get("margin_s", 0.5) or 0.5),
        clip_capture_min_mph=float((raw.get("clip") or {}).get("capture_min_mph", 0.0) or 0.0),
        clip_capture_max_mph=float((raw.get("clip") or {}).get("capture_max_mph", 999.0) or 999.0),
    )
