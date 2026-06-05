"""Tests for the verified media sweep (`Uploader._fix_missing_media`).

Work order (2026-06-05): the startup sweep must re-upload only passes the
hub reports as media-missing (has_thumb / has_clip in the list contract)
and whose local media still exists — against a healthy hub it re-uploads
nothing, so fresh passes upload within one batch interval of a restart.

Runs under pytest, or standalone: `python tests/test_media_verify.py`.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from camwatch.db import Database
from camwatch.uploader import MIN_PASS_ID, Uploader


def _make_uploader(tmp_path) -> Uploader:
    db = Database(path=tmp_path / "test.db")
    cfg = SimpleNamespace(alert_threshold_mph=40.0, events_dir=tmp_path)
    return Uploader(db=db, config=cfg, cloud_url="http://hub.invalid", api_key="k")


def _insert_uploaded_pass(
    up: Uploader, tmp_path, n: int, *, thumb: bool = True, clip: bool = True
) -> int:
    """An already-uploaded pass whose local media files exist as requested.
    Rows get ids above MIN_PASS_ID so the sweep's window includes them."""
    clip_path = tmp_path / f"cal_test_{n}.mp4"
    if clip:
        clip_path.write_bytes(b"mp4")
    if thumb:
        clip_path.with_suffix(".jpg").write_bytes(b"jpg")
    pid = MIN_PASS_ID + n
    with up.db.connect() as conn:
        conn.execute(
            """INSERT INTO passes
                   (id, captured_at, track_id, cls_name, direction, elapsed_s,
                    clip_path, speed_mph, speed_method, camera, uploaded_at)
               VALUES (?, ?, ?, 'car', 'N', 1.5, ?, 30.0, 'cadence_seq',
                       'cx810', datetime('now'))""",
            (pid, f"2026-06-05T08:{n:02d}:00-04:00", n, str(clip_path)),
        )
        conn.commit()
    return pid


def _hub_row(has_thumb: int = 1, has_clip: int = 1) -> dict:
    return {"has_thumb": has_thumb, "has_clip": has_clip}


def _run_sweep(up: Uploader, hub_index) -> list[int]:
    """Run the sweep against a stubbed hub; return re-uploaded pass ids."""
    sent: list[int] = []
    up._hub_media_index = lambda since: hub_index  # type: ignore[method-assign]
    up._upload_pass = lambda p: sent.append(p.id) or True  # type: ignore[method-assign]
    up._fix_missing_media()
    return sent


def test_healthy_hub_reuploads_nothing(tmp_path):
    up = _make_uploader(tmp_path)
    pids = [_insert_uploaded_pass(up, tmp_path, n) for n in range(1, 4)]
    hub = {pid: _hub_row() for pid in pids}
    assert _run_sweep(up, hub) == []


def test_only_flagged_pass_is_resent(tmp_path):
    up = _make_uploader(tmp_path)
    ok1 = _insert_uploaded_pass(up, tmp_path, 1)
    bad = _insert_uploaded_pass(up, tmp_path, 2)
    ok2 = _insert_uploaded_pass(up, tmp_path, 3)
    hub = {ok1: _hub_row(), bad: _hub_row(has_thumb=0), ok2: _hub_row()}
    assert _run_sweep(up, hub) == [bad]


def test_missing_hub_clip_resent_only_while_local_clip_exists(tmp_path):
    up = _make_uploader(tmp_path)
    # Clip flagged missing on the hub, but the local .mp4 is gone too
    # (retention, or a thumb-only pass) — nothing useful to send.
    aged = _insert_uploaded_pass(up, tmp_path, 1, clip=False)
    # Same hub state with the local .mp4 still on disk — re-send.
    fresh = _insert_uploaded_pass(up, tmp_path, 2)
    hub = {aged: _hub_row(has_clip=0), fresh: _hub_row(has_clip=0)}
    assert _run_sweep(up, hub) == [fresh]


def test_pass_absent_from_hub_is_never_resent(tmp_path):
    # Absent = outside the hub's window or deleted hub-side; a blind
    # re-send would resurrect deletions.
    up = _make_uploader(tmp_path)
    _insert_uploaded_pass(up, tmp_path, 1)
    assert _run_sweep(up, {}) == []


def test_unreachable_hub_skips_sweep(tmp_path):
    up = _make_uploader(tmp_path)
    _insert_uploaded_pass(up, tmp_path, 1)
    assert _run_sweep(up, None) == []


def test_hub_media_index_url_shape(tmp_path, monkeypatch):
    """The hub query hits GET /api/passes with Bearer auth and a range
    window — the admin-tier machine-read contract."""
    up = _make_uploader(tmp_path)
    seen: dict = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"passes": [
                {"engine_pass_id": 9001, "has_thumb": 1, "has_clip": 1},
                {"engine_pass_id": None, "has_thumb": 0, "has_clip": 0},
            ]}

    class _Client:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None, headers=None):
            seen.update(url=url, params=params, headers=headers)
            return _Resp()

    monkeypatch.setattr("camwatch.uploader.httpx.Client", _Client)
    index = up._hub_media_index("2026-06-05T00:00:00-04:00")
    assert seen["url"] == "http://hub.invalid/api/passes"
    assert seen["params"]["range"] == "30d"
    assert seen["params"]["since"] == "2026-06-05T00:00:00-04:00"
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert index == {9001: {"engine_pass_id": 9001, "has_thumb": 1, "has_clip": 1}}


if __name__ == "__main__":
    import sys

    import pytest as _pytest

    sys.exit(_pytest.main([__file__, "-v"]))
