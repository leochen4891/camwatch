from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    host: str
    port: int
    path: str                       # main-stream RTSP path on the camera
    user: str
    password: str
    # When set, the capture worker reads frames from this JPEG (looped at the
    # camera's nominal rate) instead of opening RTSP. Used for dev/test when
    # the real camera isn't available or shouldn't be contended with.
    static_frame_path: str | None = None

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
    # Optional per-class confidence minimums keyed by COCO class name
    # (e.g. {"motorcycle": 0.6}). Detections of a listed class below its
    # threshold are dropped. Values at or below `conf` have no effect:
    # YOLO already filters everything below `conf` at inference.
    conf_per_class: dict[str, float] = field(default_factory=dict)


@dataclass
class CalibrationConfig:
    frame_width: int
    frame_height: int
    # Legacy per-direction "meters between the deprecated screen lines".
    # Speed for new passes comes from the homography running-avg; these are
    # retained only as a fallback conversion factor for pre-grid_crossing rows
    # in the DB that recorded `elapsed_s` but no `speed_mph`.
    line_distance_m_north: float = 0.0
    line_distance_m_south: float = 0.0
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
    enrich_offset_mph: float  # passes >= (threshold - this) get vehicle make/model enrichment
    events_dir: Path
    calibration_path: Path
    max_track_age_s: float
    # Three-phase retention. Set any to 0 to disable that phase.
    # Typical layout: clips_days < thumbs_days < passes_days.
    clips_days: int = 14   # delete .mp4 only; the thumb + DB row survive
    thumbs_days: int = 90  # delete .jpg too; NULLs clip_path so the UI hides playback affordances
    passes_days: int = 365 # hard-delete the DB row + per-pass jsonl
    heatmap_days: int = 8  # trailing window for the activity heatmap (max 14)
    clip_margin_s: float = 0.5  # pre/post-roll padding around the crossing window
    clip_capture_min_mph: float = 0.0  # passes below this speed are logged but skip clip
    clip_capture_max_mph: float = 999.0  # passes above this speed are logged but skip clip
    preview_show_grid: bool = True  # draw the calibrated measurement grid on the live preview
    pause_at_night: bool = True  # skip YOLO/triggering when the camera is in IR/night mode
    upload_enabled: bool = True  # when False, the hub uploader stays idle (creds still required to run at all)
    # Inset (in feet) for the entry/exit anchor images relative to the grid
    # edges. The speed-measurement grid is unchanged; only the anchor image
    # picker shifts. Positive values move the anchor inward from the grid
    # boundary. Set both to 0 for the original "anchor at grid edge" behavior.
    recorder_south_anchor_inset_ft: float = 0.0  # north of Y_MIN
    recorder_north_anchor_inset_ft: float = 0.0  # south of Y_MAX

    def load_calibration(self) -> CalibrationConfig | None:
        if not self.calibration_path.exists():
            return None
        with self.calibration_path.open() as f:
            data = yaml.safe_load(f) or {}
        return CalibrationConfig(
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


def _resolve_device(requested: str | None) -> str:
    """Return a concrete torch device string from a config value.

    `"auto"` (or missing) probes in order: cuda → cpu. Any other value
    (`"cuda"`, `"cuda:0"`, `"cpu"`) is passed through untouched so
    existing configs keep working.
    """
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_retention_days(raw: dict, key: str, legacy_default: int) -> int:
    """Read `retention.<key>` from a parsed config, falling back to the
    Mac-era `recordings_days` (or even older `days`) single-knob if the new
    split keys are absent. Returns 0 only if the user explicitly set 0."""
    r = raw.get("retention") or {}
    if key in r and r[key] is not None:
        return int(r[key])
    legacy = r.get("recordings_days", r.get("days"))
    if legacy is not None:
        return int(legacy)
    return legacy_default


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
    device_requested = mdl.get("device")
    device_resolved = _resolve_device(device_requested)
    if device_requested in (None, "auto") and device_resolved != (device_requested or "auto"):
        log.info("model.device=%s resolved to %s", device_requested or "auto", device_resolved)
    return Config(
        camera=CameraConfig(
            host=cam["host"],
            port=int(cam["port"]),
            path=cam["path"],
            static_frame_path=cam.get("static_frame_path"),
            user=user,
            password=pw,
        ),
        model=ModelConfig(
            weights=mdl["weights"],
            device=device_resolved,
            conf=float(mdl["conf"]),
            iou=float(mdl["iou"]),
            classes=list(mdl["classes"]),
            conf_per_class={
                str(k): float(v)
                for k, v in (mdl.get("conf_per_class") or {}).items()
            },
        ),
        alert_threshold_mph=float(raw["alert"]["threshold_mph"]),
        enrich_offset_mph=float(raw["alert"].get("enrich_offset_mph", 5.0)),
        events_dir=Path(raw["paths"]["events_dir"]),
        calibration_path=Path(raw["paths"]["calibration"]),
        max_track_age_s=float(raw["speed"]["max_track_age_s"]),
        clips_days=_pick_retention_days(raw, "clips_days", legacy_default=14),
        thumbs_days=_pick_retention_days(raw, "thumbs_days", legacy_default=90),
        passes_days=int((raw.get("retention") or {}).get("passes_days", 365) or 0),
        heatmap_days=max(1, min(14, int((raw.get("heatmap") or {}).get("days", 8) or 8))),
        clip_margin_s=float((raw.get("clip") or {}).get("margin_s", 0.5) or 0.5),
        clip_capture_min_mph=float((raw.get("clip") or {}).get("capture_min_mph", 0.0) or 0.0),
        clip_capture_max_mph=float((raw.get("clip") or {}).get("capture_max_mph", 999.0) or 999.0),
        preview_show_grid=bool((raw.get("preview") or {}).get("show_grid", True)),
        pause_at_night=bool((raw.get("capture") or {}).get("pause_at_night", True)),
        upload_enabled=bool((raw.get("upload") or {}).get("enabled", True)),
        recorder_south_anchor_inset_ft=float(
            ((raw.get("recorder") or {}).get("anchor_inset_ft") or {}).get("south", 0.0) or 0.0
        ),
        recorder_north_anchor_inset_ft=float(
            ((raw.get("recorder") or {}).get("anchor_inset_ft") or {}).get("north", 0.0) or 0.0
        ),
    )
