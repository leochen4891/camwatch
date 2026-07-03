"""Durable `_upload_batch` behavior: an undeliverable head-of-line pass must
never freeze the queue, and must never be marked uploaded without a confirmed
hub accept (the failure mode behind the 2026-07-01 jam, where thumbnail-less
head passes were manually stamped uploaded_at and silently dropped from the hub).

The batch now resolves every head pass to exactly one honest outcome: delivered
(uploaded_at set) or dead-lettered (quarantined, uploaded_at still NULL).

Runs under pytest, or standalone: `python tests/test_upload_dead_letter.py`.
"""
from __future__ import annotations

from types import SimpleNamespace

from camwatch.db import Database
from camwatch.uploader import MAX_UPLOAD_ATTEMPTS, MIN_PASS_ID, Uploader, _Outcome


def _make_uploader(tmp_path) -> Uploader:
    db = Database(path=tmp_path / "test.db")
    cfg = SimpleNamespace(alert_threshold_mph=40.0, events_dir=tmp_path)
    return Uploader(db=db, config=cfg, cloud_url="http://hub.invalid", api_key="k")


def _insert_pending(up: Uploader, tmp_path, n: int, *, thumb: bool, clip: bool) -> int:
    """A pending (not-yet-uploaded) pass. clip_path is always set (the batch
    requires it); `clip`/`thumb` control whether the files exist on disk."""
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
                    clip_path, speed_mph, speed_method, camera)
               VALUES (?, ?, ?, 'car', 'N', 1.5, ?, 30.0, 'cadence_seq', 'cx810')""",
            (pid, f"2026-06-05T08:{n:02d}:00-04:00", n, str(clip_path)),
        )
        conn.commit()
    return pid


def _state(up: Uploader, pid: int) -> dict:
    with up.db.connect() as conn:
        r = conn.execute(
            "SELECT uploaded_at, upload_state, upload_attempts FROM passes WHERE id = ?",
            (pid,),
        ).fetchone()
    return {"uploaded_at": r[0], "upload_state": r[1], "upload_attempts": r[2]}


def _always(outcome: _Outcome):
    return lambda p: outcome


def test_media_gone_head_is_deadlettered_and_window_slides(tmp_path):
    """A thumbnail-less, clip-gone pass at the head is dead-lettered (never
    uploaded), and a deliverable pass later in the same batch still uploads."""
    up = _make_uploader(tmp_path)
    bad = _insert_pending(up, tmp_path, 1, thumb=False, clip=False)  # lower id = head
    good = _insert_pending(up, tmp_path, 2, thumb=True, clip=True)
    sent: list[int] = []
    up._deliver = lambda p: (sent.append(p.id), _Outcome(ok=True, status=200))[1]

    count = up._upload_batch()

    assert count == 1
    assert sent == [good]                    # window slid past the dead head
    assert _state(up, bad)["upload_state"] == "dead_letter"
    assert _state(up, bad)["uploaded_at"] is None     # NEVER marked uploaded
    assert _state(up, good)["uploaded_at"] is not None


def test_deadlettered_pass_leaves_pending_set(tmp_path):
    up = _make_uploader(tmp_path)
    _insert_pending(up, tmp_path, 1, thumb=False, clip=False)
    up._deliver = _always(_Outcome(ok=True, status=200))
    up._upload_batch()                        # dead-letters it
    # Second batch: it's excluded, so nothing to do and it stays un-uploaded.
    up._deliver = lambda p: (_ for _ in ()).throw(AssertionError("should not deliver"))
    assert up._upload_batch() == 0


def test_thumbnail_regenerated_from_clip_then_delivered(tmp_path):
    """Thumbnail missing but clip present → regenerate, then deliver (no
    dead-letter). Regen is stubbed (real av/cv2 decode is covered elsewhere)."""
    up = _make_uploader(tmp_path)
    pid = _insert_pending(up, tmp_path, 1, thumb=False, clip=True)
    regen_calls: list[str] = []

    def fake_regen(clip_path):
        regen_calls.append(str(clip_path))
        clip_path.with_suffix(".jpg").write_bytes(b"jpg")
        return True

    up._regenerate_thumbnail = fake_regen
    up._deliver = _always(_Outcome(ok=True, status=200))

    assert up._upload_batch() == 1
    assert regen_calls                         # regen was attempted
    assert _state(up, pid)["uploaded_at"] is not None
    assert _state(up, pid)["upload_state"] is None


def test_persistent_4xx_deadletters_after_max_attempts(tmp_path):
    """A poison pass (hub keeps returning 4xx) dead-letters after
    MAX_UPLOAD_ATTEMPTS, never marked uploaded."""
    up = _make_uploader(tmp_path)
    pid = _insert_pending(up, tmp_path, 1, thumb=True, clip=True)
    up._deliver = _always(_Outcome(ok=False, status=422, transient=False))

    for _ in range(MAX_UPLOAD_ATTEMPTS):
        up._upload_batch()

    st = _state(up, pid)
    assert st["upload_state"] == "dead_letter"
    assert st["uploaded_at"] is None
    assert st["upload_attempts"] >= MAX_UPLOAD_ATTEMPTS


def test_transient_failures_never_deadletter(tmp_path):
    """Network / 5xx failures are an outage, not a poison pass: retry forever,
    never dead-letter, never marked uploaded."""
    up = _make_uploader(tmp_path)
    pid = _insert_pending(up, tmp_path, 1, thumb=True, clip=True)
    up._deliver = _always(_Outcome(ok=False, status=503, transient=True))

    for _ in range(MAX_UPLOAD_ATTEMPTS * 2):
        up._upload_batch()

    st = _state(up, pid)
    assert st["upload_state"] is None          # still pending, not quarantined
    assert st["uploaded_at"] is None
    assert st["upload_attempts"] == 0          # transient never counts


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
                print(f"ok   {fn.__name__}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
