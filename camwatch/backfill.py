"""Backfill passes with anchor images to camwatch-web cloud.

Run from the camwatch working directory on the production machine:
    python -m camwatch.backfill --since-id 6693

Uses the same ingest route as the live uploader. Only uploads passes
that have entry/exit anchor images (CX810 camera era).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import httpx

from .config import load_config
from .db import Database, Pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

LOCAL_SERVER = "http://127.0.0.1:8000"


def has_anchors(pass_id: int) -> bool:
    try:
        r = httpx.head(f"{LOCAL_SERVER}/passes/{pass_id}/thumb?anchor=entry", timeout=3.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def upload_pass(
    p: Pass,
    config_threshold: float,
    events_dir: Path,
    cloud_url: str,
    api_key: str,
) -> bool:
    metadata = {
        "captured_at": p.captured_at,
        "track_id": p.track_id,
        "cls_name": p.cls_name,
        "direction": p.direction,
        "elapsed_s": p.elapsed_s,
        "speed_mph": p.speed_mph,
        "speed_method": p.speed_method,
        "known_mph": p.known_mph,
        "is_alarm": (
            p.speed_mph is not None
            and p.speed_mph >= config_threshold
            and p.known_mph is None
        ),
        "threshold_mph": config_threshold,
        "vehicle_make": p.vehicle_make,
        "vehicle_model": p.vehicle_model,
        "vehicle_color": p.vehicle_color,
        "vehicle_confidence": p.vehicle_confidence,
    }

    files: dict[str, tuple[str | None, bytes, str]] = {
        "metadata": (None, json.dumps(metadata).encode(), "application/json"),
    }

    clip_path = Path(p.clip_path) if p.clip_path else None
    if clip_path:
        thumb_path = clip_path.with_suffix(".jpg")
        if thumb_path.exists():
            files["thumb"] = ("thumb.jpg", thumb_path.read_bytes(), "image/jpeg")
        if clip_path.exists():
            files["clip"] = ("clip.mp4", clip_path.read_bytes(), "video/mp4")

    traj_path = events_dir / f"pass_{p.id}.jsonl"
    if traj_path.exists():
        files["trajectory"] = (
            "trajectory.jsonl",
            traj_path.read_bytes(),
            "application/x-jsonlines",
        )

    for anchor in ("entry", "exit"):
        try:
            resp = httpx.get(
                f"{LOCAL_SERVER}/passes/{p.id}/thumb?anchor={anchor}",
                timeout=5.0,
            )
            if resp.status_code == 200 and len(resp.content) > 100:
                files[f"thumb_{anchor}"] = (
                    f"thumb_{anchor}.jpg",
                    resp.content,
                    "image/jpeg",
                )
        except httpx.HTTPError:
            pass

    try:
        resp = httpx.post(
            f"{cloud_url}/api/ingest",
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            timeout=60.0,
        )
        if resp.status_code == 201:
            return True
        log.warning("pass %d: HTTP %d %s", p.id, resp.status_code, resp.text[:200])
        return False
    except httpx.HTTPError as e:
        log.warning("pass %d: network error: %s", p.id, e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill passes to camwatch-web")
    parser.add_argument(
        "--since-id",
        type=int,
        default=6693,
        help="First pass ID to backfill (default: 6693, first CX810 pass)",
    )
    parser.add_argument("--cloud-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--batch", type=int, default=50, help="Batch size before pause")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    db = Database()

    with db.connect() as conn:
        rows = conn.execute(
            """SELECT * FROM passes
               WHERE deleted = 0 AND id >= ?
               ORDER BY id ASC""",
            (args.since_id,),
        ).fetchall()

    passes = [Pass.from_row(r) for r in rows]
    log.info("found %d passes since id %d", len(passes), args.since_id)

    if args.dry_run:
        for p in passes[:20]:
            anchor = has_anchors(p.id)
            print(f"  #{p.id} {p.captured_at} {p.direction} "
                  f"{p.speed_mph or 0:.0f}mph anchors={anchor}")
        print(f"  ... ({len(passes)} total)")
        return

    ok = 0
    fail = 0
    skip = 0
    for i, p in enumerate(passes):
        if not has_anchors(p.id):
            skip += 1
            continue

        log.info("[%d/%d] pass #%d %s", i + 1, len(passes), p.id, p.captured_at)
        if upload_pass(p, cfg.alert_threshold_mph, cfg.events_dir, args.cloud_url, args.api_key):
            ok += 1
        else:
            fail += 1

        if (ok + fail) % args.batch == 0 and (ok + fail) > 0:
            log.info("progress: %d ok, %d fail, %d skip — pausing 2s", ok, fail, skip)
            time.sleep(2)

    log.info("done: %d uploaded, %d failed, %d skipped (no anchors)", ok, fail, skip)


if __name__ == "__main__":
    main()
