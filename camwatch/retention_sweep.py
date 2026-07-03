"""ADR-024 gated hot-media retention sweep.

Deletes engine-local media files (clip/thumb/entry/exit/trajectory) for
passes older than a cutoff, but ONLY when the cold archive's ledger
(`archive_files` in /srv/camwatch/archive.db, schema camwatch-cold-archive/1)
holds a read-back-verified copy of that exact (engine_pass_id, kind) — the
delete-gate contract frozen in /srv/camwatch/SCHEMA.md. DB rows are never
touched: this sweep unlinks files, nothing else.

Dry-run is the default; deletion requires an explicit --delete. Clips are
additionally sha256-compared against the ledger before deletion (the
"paranoid sweeper" the schema recommends); a mismatch skips the file and
logs a warning rather than deleting either side.

Runs standalone (stdlib only) so it needs neither the server process nor
its config loader:

    .venv/bin/python -m camwatch.retention_sweep --age-days 90 [--delete]

Supersedes the in-engine `retention.clips_days` phase for media deletion —
see README "Retention & the cold archive" for how the two are reconciled.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("camwatch.retention_sweep")

# Engine file layout -> ledger `kind`. The stem is clip_path minus ".mp4";
# trajectories are keyed by pass id, not stem.
_STEM_KINDS = (
    (".mp4", "clip"),
    (".jpg", "thumb"),
    (".entry.jpg", "entry"),
    (".exit.jpg", "exit"),
)


def _candidate_files(
    root: Path, events_dir: Path, pid: int, clip_path: str | None
) -> list[tuple[str, Path]]:
    """All engine-local files this pass may still own, as (kind, path)."""
    out: list[tuple[str, Path]] = []
    if clip_path:
        rel = Path(clip_path)
        base = rel.name[: -len(".mp4")] if rel.name.endswith(".mp4") else rel.name
        # recordings_archive/ holds alarm copies moved by the legacy purge —
        # same kind, alternate location; both are hot copies under the gate.
        for parent in (root / rel.parent, root / "recordings_archive"):
            for suffix, kind in _STEM_KINDS:
                out.append((kind, parent / f"{base}{suffix}"))
    out.append(("trajectory", root / events_dir / f"pass_{pid}.jsonl"))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sweep(
    root: Path,
    engine_db: Path,
    archive_db: Path,
    events_dir: Path,
    age_days: int,
    delete: bool,
) -> dict[str, dict[str, int]]:
    if not archive_db.exists():
        # No ledger -> nothing is provably archived -> nothing is deletable.
        raise SystemExit(f"archive ledger not found at {archive_db}; refusing to sweep")

    cutoff = (
        datetime.now(timezone.utc).astimezone() - timedelta(days=age_days)
    ).isoformat(timespec="seconds")

    eng = sqlite3.connect(f"file:{engine_db}?mode=ro", uri=True)
    arc = sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True)
    try:
        # Soft-deleted rows are included on purpose: their files are still
        # on disk and age out under the same gate.
        rows = eng.execute(
            "SELECT id, clip_path FROM passes WHERE captured_at < ?", (cutoff,)
        ).fetchall()

        stats: dict[str, dict[str, int]] = {}

        def bump(kind: str, key: str, n: int = 1, size: int = 0) -> None:
            s = stats.setdefault(
                kind,
                {"deletable": 0, "bytes": 0, "deleted": 0,
                 "unverified": 0, "sha_mismatch": 0},
            )
            s[key] += n
            if size:
                s["bytes"] += size

        for pid, clip_path in rows:
            ledger = {
                kind: sha
                for kind, sha in arc.execute(
                    "SELECT kind, sha256 FROM archive_files "
                    "WHERE engine_pass_id = ? AND verified_at IS NOT NULL",
                    (pid,),
                )
            }
            for kind, path in _candidate_files(root, events_dir, pid, clip_path):
                if not path.is_file():
                    continue
                if kind not in ledger:
                    bump(kind, "unverified")
                    continue
                size = path.stat().st_size
                if kind == "clip" and _sha256(path) != ledger[kind]:
                    bump(kind, "sha_mismatch")
                    log.warning(
                        "pass %d: hot %s does not match ledger sha256 — kept",
                        pid, path,
                    )
                    continue
                bump(kind, "deletable", size=size)
                if delete:
                    path.unlink()
                    bump(kind, "deleted")
        return stats
    finally:
        eng.close()
        arc.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path.cwd(),
                    help="engine root (holds recordings/, camwatch.db); default cwd")
    ap.add_argument("--engine-db", type=Path, default=None,
                    help="engine sqlite db (default <root>/camwatch.db)")
    ap.add_argument("--archive-db", type=Path,
                    default=Path("/srv/camwatch/archive.db"),
                    help="cold-archive ledger db")
    ap.add_argument("--events-dir", type=Path, default=Path("events"),
                    help="per-pass jsonl dir, relative to root")
    ap.add_argument("--age-days", type=int, default=90,
                    help="sweep passes captured more than this many days ago")
    ap.add_argument("--delete", action="store_true",
                    help="actually delete; without this the sweep is a dry run")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    if args.age_days < 1 and args.delete:
        # A cutoff in the future (or "now") with --delete would raze the
        # entire verified store; only dry runs may probe with age 0.
        raise SystemExit("--delete requires --age-days >= 1")

    stats = sweep(
        root=args.root,
        engine_db=args.engine_db or args.root / "camwatch.db",
        archive_db=args.archive_db,
        events_dir=args.events_dir,
        age_days=args.age_days,
        delete=args.delete,
    )

    mode = "DELETED" if args.delete else "would delete (dry run)"
    total_n = sum(s["deletable"] for s in stats.values())
    total_b = sum(s["bytes"] for s in stats.values())
    for kind in sorted(stats):
        s = stats[kind]
        log.info(
            "%s: %s %d files (%.2f GB); gate-blocked unverified=%d sha_mismatch=%d",
            kind, mode, s["deletable"], s["bytes"] / 1e9,
            s["unverified"], s["sha_mismatch"],
        )
    log.info(
        "sweep complete: age>%dd, %s %d files, %.2f GB total",
        args.age_days, mode, total_n, total_b / 1e9,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
