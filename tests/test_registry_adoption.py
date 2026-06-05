"""Acceptance tests for the camwatch-cameras registry adoption (ADR-013/015).

Covers the work order's acceptance criteria:

  * Regression — with the loader-provided homography + cadence, a replayed
    pass produces the same speed as before the switch (the registry's cx810
    artifacts were migrated verbatim from this repo's config).
  * Election rejects a camera whose profile lacks a calibrated `speed`
    capability (and one not flagged electable).
  * The ingest payload carries `camera`.

The verbatim-migration tests read this repo's legacy
`config/homography.yaml`; when that file is removed in the planned
follow-up, drop the LEGACY_YAML-based tests here (from_profile is then
guarded by `test_replayed_pass_speed_*` alone).

Runs under pytest, or standalone: `python tests/test_registry_adoption.py`.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from camwatch.capture_worker import _cadence_speed
from camwatch.config import elect_main_camera
from camwatch.db import Database, Pass
from camwatch.homography import MPH_PER_MPS, Homography
from camwatch.uploader import Uploader
from camwatch_cameras import load_camera

LEGACY_YAML = Path(__file__).resolve().parent.parent / "config" / "homography.yaml"


# ---------- the registry's cx810 artifact is this repo's, verbatim ----------

def test_cx810_artifact_migrated_verbatim():
    legacy = yaml.safe_load(LEGACY_YAML.read_text())["homography"]
    registry = load_camera("cx810").calibration()
    for key in ("H", "K", "D", "frame_size"):
        assert legacy[key] == registry[key], key


def test_from_profile_matches_legacy_loader():
    old = Homography.load(LEGACY_YAML)
    new = Homography.from_profile(load_camera("cx810"))
    assert old is not None and new is not None
    assert np.array_equal(old.H, new.H)
    assert np.array_equal(old.K, new.K)
    assert np.array_equal(old.D, new.D)
    assert tuple(old.frame_size) == tuple(new.frame_size)


# ---------- replayed pass: same speed before/after the switch ----------

def _replay_trajectory(homog: Homography, mph: float, rate_fps: float):
    """A synthetic northbound crossing as the capture worker records it:
    (ts, u, v, bbox, seq, epoch) tuples whose *pixels* trace a constant
    `mph` run along the road, sampled at the cadence `rate_fps`. Stays
    within the calibrated grid (|Y| ≲ 6 m) — beyond it the lens model is
    extrapolating and world↔pixel roundtrips degrade, exactly why the
    live path only accumulates in-grid samples."""
    mps = mph / MPH_PER_MPS
    traj = []
    for i in range(12):
        t = i / rate_fps
        X, Y = -2.0, -6.0 + mps * t          # in-grid, west of the east curb
        u, v = homog.world_to_pixel(X, Y)     # distorted main-stream pixels
        traj.append((1000.0 + t, float(u), float(v), (0.0, 0.0, 1.0, 1.0), i, 1))
    return traj


def test_replayed_pass_speed_unchanged_by_loader_switch():
    old = Homography.load(LEGACY_YAML)
    new = Homography.from_profile(load_camera("cx810"))
    rate = load_camera("cx810").cadence_fps()
    traj = _replay_trajectory(old, mph=30.0, rate_fps=rate)
    s_old, n_old = _cadence_speed(traj, rate, old)
    s_new, n_new = _cadence_speed(traj, rate, new)
    assert s_old is not None
    assert s_new == s_old          # identical H/K/D + identical math
    assert n_new == n_old
    assert abs(s_new - 30.0) < 0.2  # and it is the speed we synthesized


def test_replayed_pass_speed_on_registry_cadence_fallback():
    """Warm-up fallback: the registry's measured cadence stands in for the
    live received-frame rate, so the same trajectory still yields a speed
    (previously: speed unknown) — and the value tracks the cadence."""
    homog = Homography.from_profile(load_camera("cx810"))
    registry_rate = load_camera("cx810").cadence_fps()
    traj = _replay_trajectory(homog, mph=30.0, rate_fps=registry_rate)
    s, _n = _cadence_speed(traj, registry_rate, homog)
    assert s is not None and abs(s - 30.0) < 0.2


def test_profile_projection_agrees_with_cv2_pipeline():
    """The registry's numpy px_to_world and this repo's cv2 undistort+H
    pipeline must agree — they are the 'identical homography' ADR-015
    exists to guarantee, just through two implementations. The solvers
    iterate the same fixed-point undistortion a different number of times
    (cv2 ~5, registry 20), leaving ~5 mm of residual at the anchor
    extremes — far below the fit's 13 cm mean reprojection error."""
    cam = load_camera("cx810")
    homog = Homography.from_profile(cam)
    doc = cam.calibration()
    pts = np.array([[p["u"], p["v"]] for p in doc["pixel_pts"]], dtype=np.float64)
    ours = np.array([homog.project(u, v) for u, v in pts])
    theirs = np.asarray(cam.px_to_world(pts))
    assert np.abs(ours - theirs).max() < 0.02  # meters


# ---------- main-camera election (ADR-013) ----------

def test_election_accepts_calibrated_main_camera():
    profile = elect_main_camera("cx810")
    assert profile.camera_id == "cx810"


@pytest.mark.parametrize("camera_id", ["e1", "cx410w", "no_such_camera"])
def test_election_refuses_ineligible_cameras(camera_id):
    # e1: no calibrated speed capability (zoomed plate camera).
    # cx410w: speed-calibrated but not flagged electable (pass_creation).
    # no_such_camera: not in the registry at all.
    with pytest.raises(SystemExit):
        elect_main_camera(camera_id)


# ---------- boot path: config + registry resolve with no network ----------

def test_config_boot_path_is_local_only(tmp_path, monkeypatch):
    """End-to-end load_config with the new camera.main_id key: election,
    profile, homography and cadence all resolve from local data — no hub,
    no camera, no network. (The uploader is unchanged: fire-and-forget,
    so a down hub never blocks boot.)"""
    from camwatch.config import load_config

    monkeypatch.setenv("REOLINK_USER", "u")
    monkeypatch.setenv("REOLINK_PASS", "p")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "camera:\n"
        "  main_id: cx810\n"
        "model:\n"
        "  weights: yolo11l.pt\n"
        "  device: cpu\n"
        "  conf: 0.35\n"
        "  iou: 0.5\n"
        "  classes: [2, 3, 5, 7]\n"
        "alert:\n"
        "  threshold_mph: 40\n"
        "paths:\n"
        "  events_dir: events\n"
        "  calibration: config/calibration.yaml\n"
        "speed:\n"
        "  max_track_age_s: 5.0\n"
    )
    cfg = load_config(cfg_path)
    assert cfg.camera.main_id == "cx810"
    assert cfg.camera.rtsp_url.startswith("rtsp://u:p@")
    assert Homography.from_profile(cfg.camera.profile) is not None
    assert cfg.camera.profile.cadence_fps() == pytest.approx(13.8)


# ---------- ingest payload carries camera provenance ----------

def _pass(**overrides) -> Pass:
    base = dict(
        id=1, captured_at="2026-06-05T12:00:00-04:00", track_id=7,
        cls_name="car", direction="N", elapsed_s=1.5, known_mph=None,
        clip_path=None, deleted=False, thumb_upgrade_status=None,
        speed_mph=31.2, speed_method="cadence_seq", vehicle_make=None,
        vehicle_model=None, vehicle_year_range=None, vehicle_color=None,
        vehicle_confidence=None, vehicle_enriched_at=None,
    )
    base.update(overrides)
    return Pass(**base)


def _uploader(tmp_path) -> Uploader:
    db = Database(path=tmp_path / "test.db")
    cfg = SimpleNamespace(alert_threshold_mph=40.0, events_dir=tmp_path)
    return Uploader(db=db, config=cfg, cloud_url="http://hub.invalid", api_key="k")


def test_ingest_metadata_carries_camera(tmp_path):
    meta = _uploader(tmp_path)._pass_metadata(_pass(camera="cx810"))
    assert meta["camera"] == "cx810"
    assert meta["speed_method"] == "cadence_seq"


def test_ingest_metadata_camera_fallback_for_legacy_rows(tmp_path):
    # Rows from before the local `camera` column all predate multi-camera
    # and were produced by the cx810.
    meta = _uploader(tmp_path)._pass_metadata(_pass(camera=None))
    assert meta["camera"] == "cx810"


def test_pass_camera_roundtrips_through_db(tmp_path):
    db = Database(path=tmp_path / "test.db")
    pid = db.insert_pass(
        captured_at="2026-06-05T12:00:00-04:00", track_id=7, cls_name="car",
        direction="N", elapsed_s=1.5, clip_path=None,
        speed_mph=31.2, speed_method="cadence_seq", camera="cx810",
    )
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM passes WHERE id = ?", (pid,)).fetchone()
    assert Pass.from_row(row).camera == "cx810"


if __name__ == "__main__":
    import sys

    import pytest as _pytest

    sys.exit(_pytest.main([__file__, "-v"]))
