"""Tests for the ADR-024 gated retention sweep (`camwatch.retention_sweep`).

The delete-gate contract (frozen in /srv/camwatch/SCHEMA.md): a hot
(engine_pass_id, kind) file is deletable iff the archive ledger row exists
AND verified_at IS NOT NULL; clips must additionally match the ledger's
sha256. DB rows are never touched.

Runs under pytest, or standalone: `python tests/test_retention_sweep.py`.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from camwatch.retention_sweep import sweep


def _setup(tmp_path: Path, *, verified: bool, sha_of: bytes | None = None):
    """One 100-day-old pass with clip+thumb+trajectory on disk, one ledger."""
    root = tmp_path
    (root / "recordings").mkdir()
    (root / "events").mkdir()

    clip = root / "recordings" / "cal_x_id1_N.mp4"
    clip.write_bytes(b"clipdata")
    (root / "recordings" / "cal_x_id1_N.jpg").write_bytes(b"thumbdata")
    (root / "events" / "pass_1.jsonl").write_bytes(b"{}\n")

    eng = root / "camwatch.db"
    with sqlite3.connect(eng) as c:
        c.execute(
            "CREATE TABLE passes (id INTEGER PRIMARY KEY, captured_at TEXT, clip_path TEXT)"
        )
        c.execute(
            "INSERT INTO passes VALUES (1, '2026-01-01T00:00:00-04:00', 'recordings/cal_x_id1_N.mp4')"
        )

    arc = root / "archive.db"
    clip_sha = hashlib.sha256(sha_of if sha_of is not None else b"clipdata").hexdigest()
    with sqlite3.connect(arc) as c:
        c.execute(
            "CREATE TABLE archive_files (engine_pass_id INTEGER, kind TEXT, path TEXT,"
            " sha256 TEXT, bytes INTEGER, source TEXT, archived_at TEXT, verified_at TEXT,"
            " PRIMARY KEY (engine_pass_id, kind))"
        )
        v = "2026-07-01T00:00:00Z" if verified else None
        for kind, sha in (("clip", clip_sha), ("thumb", "x"), ("trajectory", "x")):
            c.execute(
                "INSERT INTO archive_files VALUES (1, ?, 'p', ?, 8, 'producer', 'a', ?)",
                (kind, sha, v),
            )
    return root, eng, arc, clip


def _run(root, eng, arc, delete):
    return sweep(
        root=root, engine_db=eng, archive_db=arc,
        events_dir=Path("events"), age_days=90, delete=delete,
    )


def test_dry_run_deletes_nothing(tmp_path):
    root, eng, arc, clip = _setup(tmp_path, verified=True)
    stats = _run(root, eng, arc, delete=False)
    assert stats["clip"]["deletable"] == 1
    assert stats["clip"]["deleted"] == 0
    assert clip.exists()


def test_verified_files_deleted_rows_kept(tmp_path):
    root, eng, arc, clip = _setup(tmp_path, verified=True)
    stats = _run(root, eng, arc, delete=True)
    assert stats["clip"]["deleted"] == 1
    assert stats["thumb"]["deleted"] == 1
    assert stats["trajectory"]["deleted"] == 1
    assert not clip.exists()
    with sqlite3.connect(eng) as c:
        assert c.execute("SELECT COUNT(*) FROM passes").fetchone()[0] == 1


def test_unverified_gate_blocks(tmp_path):
    root, eng, arc, clip = _setup(tmp_path, verified=False)
    stats = _run(root, eng, arc, delete=True)
    assert stats["clip"]["unverified"] == 1
    assert stats["clip"]["deleted"] == 0
    assert clip.exists()


def test_clip_sha_mismatch_kept(tmp_path):
    root, eng, arc, clip = _setup(tmp_path, verified=True, sha_of=b"otherdata")
    stats = _run(root, eng, arc, delete=True)
    assert stats["clip"]["sha_mismatch"] == 1
    assert clip.exists()
    # thumbs are gate-only: still deleted
    assert stats["thumb"]["deleted"] == 1


def test_young_pass_untouched(tmp_path):
    root, eng, arc, clip = _setup(tmp_path, verified=True)
    with sqlite3.connect(eng) as c:
        c.execute("UPDATE passes SET captured_at = '2099-01-01T00:00:00-04:00'")
    stats = _run(root, eng, arc, delete=True)
    assert not stats  # no candidates at all
    assert clip.exists()


def test_missing_ledger_aborts(tmp_path):
    root, eng, arc, clip = _setup(tmp_path, verified=True)
    arc.unlink()
    try:
        _run(root, eng, arc, delete=True)
    except SystemExit:
        pass
    else:  # pragma: no cover
        raise AssertionError("sweep must refuse to run without the ledger")
    assert clip.exists()


if __name__ == "__main__":
    import tempfile

    for fn in (
        test_dry_run_deletes_nothing,
        test_verified_files_deleted_rows_kept,
        test_unverified_gate_blocks,
        test_clip_sha_mismatch_kept,
        test_young_pass_untouched,
        test_missing_ledger_aborts,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
        print(f"ok {fn.__name__}")
