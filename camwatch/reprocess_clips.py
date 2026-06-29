"""Offline reprocessor: recover native vehicle passes from recorded cx810 clips.

WHY THIS EXISTS
---------------
A router power-cut took the box offline 2026-06-26 ~01:00 -> 2026-06-28
17:39 ET; the live engine created no passes in that window. The cx810 main
camera kept recording motion clips to its SD card, which a separate agent
mirrors into the FTP tree:

    /srv/nas/files/cx810/{YYYY}/{MM}/{DD}/cx810_00_{YYYYMMDDHHMMSS}.mp4

This tool consumes those FILES and reconstructs the *native* per-pass output
the live engine would have produced: a pass row PLUS its media set --
a main thumbnail, an ENTRY anchor image, an EXIT anchor image, and the
per-frame trajectory.jsonl that drives the speed chart.

DESIGN: drive the engine's real pipeline from a FILE frame-source
-----------------------------------------------------------------
The live pass machinery lives in `camwatch.capture_worker.CaptureWorker._run`:
YOLO detect -> BotSORT track -> grid crossing -> trajectory accumulation ->
`_cadence_speed` -> captured_at anchored to the grid-exit frame -> recorder
(thumb + entry/exit anchors + clip) -> `_save_pass_trajectory_jsonl`.

We reuse those *component pieces* directly rather than the monolithic
`_run` thread (which owns an infinite loop, a retention sweep, and a direct
`db.insert_pass`). The per-frame and per-pass logic here is a faithful
transcription of the relevant section of `_run` (the event handling at
capture_worker.py ~line 1026-1223), wired to:

  - a `ClipFrameStream` file source (cv2.VideoCapture over the clip, frame
    timestamps = clip_true_start + clean container PTS) standing in for
    RtspStream. It exposes the same `Frame(image, ts, seq, epoch)` shape and
    a `received_fps()` that returns the clip's MEASURED container cadence,
    so the engine's Layer-1 cadence-seq speed runs on honest timing (the SD
    recordings carry clean PTS, unlike live RTSP per ADR-010).
  - cx810 calibration (homography @ 3840x2160 + measured cadence) from the
    camwatch-cameras registry, and camera='cx810' provenance.
  - a real (borrowed) `CaptureWorker` instance ONLY to call its
    `_save_pass_trajectory_jsonl` and its anchor-inset picker -- so the
    trajectory jsonl and the entry/exit anchor capture points are produced
    by the exact same code paths as live passes.

GATING (this phase = TEST ONLY)
-------------------------------
DB inserts and hub uploads are gated behind `--commit`, which is OFF by
default. In dry-run we print what we would create and write all generated
artifacts (thumb + entry anchor + exit anchor + trajectory.jsonl) into a
scratch dir for eyeballing. NOTHING is written to camwatch.db and NOTHING is
uploaded in this phase.

IDEMPOTENCY / DEDUP
-------------------
Re-running creates no duplicates and never collides with real passes that
already exist at the gap edges (1 real pass at 2026-06-26T01:00:05 and live
capture resumed 2026-06-28T17:39:11). Before emitting a candidate we query
camwatch.db for any existing pass within `--dedup-window-s` of the candidate
captured_at in the SAME direction; a hit means "skip". The recovery window
itself is bounded to (last-real-before-gap, first-real-after-gap) exclusive.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .capture import Frame
from .capture_worker import (
    _FT_TO_M,
    _GRID_X_MAX,
    _GRID_X_MIN,
    _GRID_Y_MAX,
    _GRID_Y_MIN,
    _GRID_TOLERANCE_M,
    _MIN_RUNNING_SAMPLES,
    CaptureWorker,
    _cadence_speed,
)
from .config import Config, load_config
from .db import Database
from .detect import Detector
from .grid_crossing import GridCrossingDetector
from .homography import Homography
from .recorder import ClipRecorder

log = logging.getLogger(__name__)

# Camera tz (the cx810 records local wall-clock in its filenames). The DB
# rows are stored as local ISO with offset (e.g. -04:00), which is what
# captured_at must look like for the recovered passes too.
_CAMERA_TZ = ZoneInfo("America/New_York")

# Filename time pattern: cx810_00_YYYYMMDDHHMMSS.mp4 -> clip_true_start.
# Same recipe the speed-refiner uses (recordings.parse_start_from_filename).
import re

_FILENAME_TS = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?!\d)")


def parse_clip_start(path: Path, tz: ZoneInfo = _CAMERA_TZ) -> datetime | None:
    """Local wall-clock start of the clip, from its filename. tz-aware."""
    m = _FILENAME_TS.search(path.stem)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    return datetime(y, mo, d, h, mi, s, tzinfo=tz)


def extract_pts(path: Path) -> list[float]:
    """Per-frame container PTS in seconds, decode order (clean for SD clips).

    The validated recipe from camwatch-speed-refiner's PTS investigation:
    ffprobe packet pts_time. These IP-camera encodes have no B-frames, so
    decode order == presentation order; we sort defensively.
    """
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True, timeout=120, check=True,
    ).stdout
    pts = [float(line) for line in out.split() if line.strip()]
    pts.sort()
    return pts


@dataclass
class ClipFrameStream:
    """File-backed frame source with the RtspStream/StaticFrameStream contract.

    Yields `Frame(image, ts, seq, epoch)` for each decoded frame, pairing
    cv2-decoded frames with the ffprobe PTS list by index (the speed-refiner
    recipe). `ts` = clip_true_start_monotonic_base + container PTS, so the
    inter-frame intervals are the clip's real cadence. `received_fps()`
    returns the clip's measured container cadence -- the engine's cadence-seq
    speed path consumes this exactly as it consumes the live received rate.

    One clip == one session (epoch fixed): the cadence speed path rejects a
    pass that spans an epoch change, which never happens within a single clip.
    """

    path: Path
    pts_times: list[float]
    epoch: int = 1
    _stop: bool = field(default=False, init=False)

    @property
    def session_epoch(self) -> int:
        return self.epoch

    def measured_fps(self) -> float | None:
        p = self.pts_times
        if len(p) < 2 or p[-1] <= p[0]:
            return None
        return (len(p) - 1) / (p[-1] - p[0])

    def received_fps(self, window_s: float = 60.0) -> float | None:
        # Whole-clip measured cadence. The clip is only ~45 s, so the live
        # 30 s-coverage warm-up floor is irrelevant: the clip's own PTS span
        # is the honest, complete cadence for every pass in it.
        return self.measured_fps()

    def stop(self) -> None:
        self._stop = True

    def frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open {self.path}")
        try:
            seq = 0
            n = len(self.pts_times)
            i = 0
            while i < n and not self._stop:
                ok, image = cap.read()
                if not ok:
                    break
                seq += 1
                # ts in seconds on an arbitrary-but-monotonic base (PTS is
                # clip-relative; the engine only uses ts ordering + the
                # cadence rate for intervals, never ts as wall-clock).
                yield Frame(
                    image=image, ts=float(self.pts_times[i]),
                    seq=seq, epoch=self.epoch,
                )
                i += 1
            if i != n:
                log.debug(
                    "%s: decoded %d frames but have %d PTS entries",
                    self.path.name, i, n,
                )
        finally:
            cap.release()


@dataclass
class Candidate:
    """A recovered pass before any DB/hub write."""
    clip: str
    captured_at: datetime          # tz-aware local
    direction: str
    speed_mph: float | None
    elapsed_s: float
    track_id: int
    cls_name: str
    n_speed_samples: int
    rate_fps: float
    # Artifact paths actually written to the scratch dir (dry-run) or the
    # recordings/events dirs (commit). None if the recorder declined to write.
    thumb_path: str | None = None
    entry_path: str | None = None
    exit_path: str | None = None
    trajectory_path: str | None = None
    skipped_reason: str | None = None


def _existing_pass_nearby(
    db: Database, captured_at: datetime, direction: str, window_s: float,
) -> str | None:
    """Return a short description of an existing camwatch.db pass within
    `window_s` of `captured_at` in the SAME direction, or None.

    captured_at is stored as local ISO with offset; we compare on the
    absolute instant via SQLite's datetime() (offset-aware), so the
    comparison is tz-correct regardless of the stored offset string.
    """
    lo = (captured_at - timedelta(seconds=window_s)).astimezone(timezone.utc)
    hi = (captured_at + timedelta(seconds=window_s)).astimezone(timezone.utc)
    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT id, captured_at FROM passes
            WHERE deleted = 0
              AND direction = ?
              AND datetime(captured_at) BETWEEN datetime(?) AND datetime(?)
            ORDER BY abs(julianday(captured_at) - julianday(?))
            LIMIT 1
            """,
            (
                direction,
                lo.isoformat(timespec="seconds"),
                hi.isoformat(timespec="seconds"),
                captured_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
            ),
        ).fetchone()
    if row is None:
        return None
    return f"pass #{row['id']} @ {row['captured_at']}"


class _CaptureWorkerShim(CaptureWorker):
    """A CaptureWorker we instantiate ONLY to borrow its instance methods
    (`_save_pass_trajectory_jsonl`, the anchor-inset picker logic) without
    starting its thread. We never call .start()/.run(); we point its
    `_recordings_dir` and `_homog` at our offline config so the jsonl lands
    where we want.
    """

    def __init__(self, cfg: Config, db: Database, recordings_dir: Path) -> None:
        super().__init__(cfg=cfg, db=db, recordings_dir=recordings_dir,
                         preview=None, profile=False, metrics=None)


def reprocess_clip(
    clip_path: Path,
    cfg: Config,
    db: Database,
    worker: _CaptureWorkerShim,
    rec_dir: Path,
    detector: Detector,
    homog: Homography,
    *,
    gap_start: datetime,
    gap_end: datetime,
    dedup_window_s: float,
) -> list[Candidate]:
    """Run the full detect/grid/trajectory/speed/anchor pipeline over one
    clip. Returns the candidate passes (deduped, gap-bounded). This mirrors
    the per-frame and per-event logic of CaptureWorker._run for cx810.

    A FRESH ClipRecorder is built per clip: the recorder's ring buffer is
    time-indexed by PTS and each clip's PTS restarts near 0, so a shared
    recorder would not evict the previous clip's frames (cutoff goes
    negative) and would contaminate the next clip's thumb/anchor picking.
    """
    clip_start = parse_clip_start(clip_path)
    if clip_start is None:
        log.warning("%s: cannot parse start time from filename; skipping", clip_path.name)
        return []

    pts = extract_pts(clip_path)
    if len(pts) < 2:
        log.warning("%s: too few PTS samples (%d); skipping", clip_path.name, len(pts))
        return []
    recorder = ClipRecorder(
        rec_dir,
        pre_seconds_before_a=cfg.clip_margin_s,
        post_seconds_after_b=cfg.clip_margin_s,
        homography=homog,
        grid_x_min=_GRID_X_MIN, grid_x_max=_GRID_X_MAX,
        grid_y_min=_GRID_Y_MIN, grid_y_max=_GRID_Y_MAX,
        grid_tolerance_m=_GRID_TOLERANCE_M,
        min_running_samples=_MIN_RUNNING_SAMPLES,
    )
    stream = ClipFrameStream(path=clip_path, pts_times=pts)
    measured = stream.received_fps()
    rate_fps = measured or cfg.camera.profile.cadence_fps()
    rate_source = "measured_pts" if measured else "registry"

    crossing = GridCrossingDetector(
        homography=homog,
        grid_x_min=_GRID_X_MIN, grid_x_max=_GRID_X_MAX,
        grid_y_min=_GRID_Y_MIN, grid_y_max=_GRID_Y_MAX,
        tolerance_m=_GRID_TOLERANCE_M,
        max_track_age_s=cfg.max_track_age_s,
    )

    # Per-track trajectory accumulation (mirrors CaptureWorker state).
    trajectories: dict[int, deque] = {}
    entered_strict: dict[int, bool] = {}

    def project(u, v):
        return homog.project(float(u), float(v))

    def t_to_captured_at(exit_pts: float) -> datetime:
        """captured_at = clip wall-clock start + in-clip grid-exit PTS.

        For the OFFLINE file path there is no processing-staleness term (we
        are not behind a live RTSP reader), so the live `_stamp_captured_at`
        correction is exactly 0: the exit-frame PTS offset from the clip
        start IS the road time. Result is tz-aware local with offset, like
        every stored row.
        """
        return clip_start + timedelta(seconds=float(exit_pts) - pts[0])

    candidates: list[Candidate] = []

    for fr in stream.frames():
        all_tracks = detector.track(fr.image)
        tracks = [
            t for t in all_tracks
            if _in_grid_track(t, homog)
        ]

        # Trajectory accumulation with strict-entry hysteresis (verbatim
        # from CaptureWorker._run).
        for tr in tracks:
            tid = int(tr.track_id)
            gx, gy = tr.ground_point
            X, Y = project(gx, gy)
            strict_in = (
                _GRID_X_MIN <= X <= _GRID_X_MAX
                and _GRID_Y_MIN <= Y <= _GRID_Y_MAX
            )
            if not entered_strict.get(tid):
                if not strict_in:
                    continue
                entered_strict[tid] = True
            traj = trajectories.setdefault(tid, deque(maxlen=200))
            bb = tuple(float(x) for x in tr.bbox)
            traj.append((fr.ts, float(gx), float(gy), bb, fr.seq, fr.epoch))

        # Recorder sees every in-grid track for overlay/anchor purposes;
        # crossing sees ALL tracks so it can observe the in->out transition.
        recorder.push(fr.image, fr.ts, tracks, seq=fr.seq)
        events = crossing.update(all_tracks, fr.ts)

        for ev in events:
            captured_at = t_to_captured_at(ev.t_b)

            traj_for_speed = list(trajectories.get(ev.track_id, ()))
            speed_mph, n_speed = _cadence_speed(traj_for_speed, rate_fps, homog)
            speed_method = "cadence_seq" if speed_mph is not None else None

            # Gap-bounding + dedup BEFORE we emit anything.
            skipped = None
            if not (gap_start < captured_at < gap_end):
                skipped = "outside recovery window"
            else:
                hit = _existing_pass_nearby(
                    db, captured_at, ev.direction, dedup_window_s
                )
                if hit is not None:
                    skipped = f"dedup: collides with existing {hit}"

            cand = Candidate(
                clip=clip_path.name,
                captured_at=captured_at,
                direction=ev.direction,
                speed_mph=speed_mph,
                elapsed_s=ev.elapsed_s,
                track_id=ev.track_id,
                cls_name=ev.cls_name,
                n_speed_samples=n_speed,
                rate_fps=rate_fps,
                skipped_reason=skipped,
            )

            if skipped is not None:
                log.info(
                    "  SKIP %s %s @ %s (%s)",
                    clip_path.name, ev.direction,
                    captured_at.isoformat(timespec="seconds"), skipped,
                )
                # Still pop the trajectory so it doesn't leak into a later
                # recycled track id.
                trajectories.pop(ev.track_id, None)
                entered_strict.pop(ev.track_id, None)
                candidates.append(cand)
                continue

            # --- produce the native media set via the engine's recorder ---
            in_range = (
                speed_mph is None
                or (cfg.clip_capture_min_mph <= speed_mph <= cfg.clip_capture_max_mph)
            )
            # Anchor-inset capture points (verbatim from _run): shift the
            # entry/exit anchor capture moments inward from the grid edges.
            entry_anchor_ts, exit_anchor_ts = _anchor_inset_ts(
                cfg, homog, trajectories.get(ev.track_id), ev.direction,
            )

            stamp = captured_at.strftime("%Y%m%dT%H%M%S")
            clip_name = f"recov_{stamp}_id{ev.track_id}_{ev.direction}.mp4"
            recorder.trigger(
                name=clip_name,
                focus_track_id=ev.track_id,
                t_a=ev.t_a,
                t_b=ev.t_b,
                speed_mph=speed_mph,
                record_video=in_range,
                entry_anchor_ts=entry_anchor_ts,
                exit_anchor_ts=exit_anchor_ts,
                rate_fps=rate_fps,
            )

            # Assign a synthetic, deterministic pass id for the jsonl name so
            # the dry-run artifacts are stable + inspectable and never collide
            # with real positive ids. Negative, derived from captured_at.
            synthetic_pid = -int(captured_at.timestamp())
            try:
                worker._save_pass_trajectory_jsonl(
                    pid=synthetic_pid, ev=ev, traj=traj_for_speed,
                    speed_mph=speed_mph, speed_method=speed_method,
                    rate_fps=rate_fps, rate_source=rate_source,
                    stamp_correction_s=0.0,
                )
                cand.trajectory_path = str(
                    cfg.events_dir / f"pass_{synthetic_pid}.jsonl"
                )
            except Exception as e:  # noqa: BLE001
                log.warning("  trajectory jsonl failed for %s: %s", clip_name, e)

            base = str(recorder._dir / clip_name)[:-4]
            cand.thumb_path = base + ".jpg"
            cand.entry_path = base + ".entry.jpg"
            cand.exit_path = base + ".exit.jpg"

            log.info(
                "  PASS %s -> %s %s %.1f mph  elapsed=%.2fs  (%d samples @ %.2f fps [%s])",
                clip_path.name,
                captured_at.isoformat(timespec="seconds"),
                ev.direction,
                speed_mph if speed_mph is not None else float("nan"),
                ev.elapsed_s, n_speed, rate_fps, rate_source,
            )
            candidates.append(cand)
            trajectories.pop(ev.track_id, None)
            entered_strict.pop(ev.track_id, None)

    # Flush the recorder so any still-active clip (and its thumb/anchors)
    # gets finalized for this clip before we move on.
    recorder.flush()
    return candidates


def _in_grid_track(t, homog: Homography) -> bool:
    X, Y = homog.project(t.ground_point[0], t.ground_point[1])
    return (
        _GRID_X_MIN - _GRID_TOLERANCE_M <= X <= _GRID_X_MAX + _GRID_TOLERANCE_M
        and _GRID_Y_MIN - _GRID_TOLERANCE_M <= Y <= _GRID_Y_MAX + _GRID_TOLERANCE_M
    )


def _anchor_inset_ts(cfg, homog, traj, direction):
    """Entry/exit anchor capture timestamps with the configured inset, copied
    verbatim from CaptureWorker._run so the anchor capture points match live.
    """
    entry_anchor_ts = None
    exit_anchor_ts = None
    south_inset_ft = cfg.recorder_south_anchor_inset_ft
    north_inset_ft = cfg.recorder_north_anchor_inset_ft
    if (south_inset_ft or north_inset_ft) and homog is not None and traj:
        samples_xy = [
            (ts, homog.project(u, v)[1])
            for ts, u, v, _bb, _seq, _epoch in traj
        ]
        y_min = min(y for _, y in samples_xy)
        y_max = max(y for _, y in samples_xy)
        south_ts = None
        north_ts = None
        if south_inset_ft > 0:
            south_target = _GRID_Y_MIN + south_inset_ft * _FT_TO_M
            if y_min <= south_target:
                south_ts = min(samples_xy, key=lambda s: abs(s[1] - south_target))[0]
        if north_inset_ft > 0:
            north_target = _GRID_Y_MAX - north_inset_ft * _FT_TO_M
            if y_max >= north_target:
                north_ts = min(samples_xy, key=lambda s: abs(s[1] - north_target))[0]
        if direction == "N":
            entry_anchor_ts = south_ts
            exit_anchor_ts = north_ts
        elif direction == "S":
            entry_anchor_ts = north_ts
            exit_anchor_ts = south_ts
    return entry_anchor_ts, exit_anchor_ts


def discover_clips(root: Path, day_dirs: list[str]) -> list[Path]:
    """All cx810 clips under root for the given YYYY/MM/DD day dirs, sorted."""
    clips: list[Path] = []
    for d in day_dirs:
        clips.extend(sorted((root / d).glob("cx810_00_*.mp4")))
    return sorted(clips, key=lambda p: p.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clips-root", default="/srv/nas/files/cx810",
        help="root of the cx810 FTP tree (default: /srv/nas/files/cx810)",
    )
    ap.add_argument(
        "--days", nargs="+", default=["2026/06/28"],
        help="YYYY/MM/DD day dirs under clips-root to scan (default: 2026/06/28)",
    )
    ap.add_argument(
        "--limit", type=int, default=20,
        help="max clips to process this run (default: 20)",
    )
    ap.add_argument(
        "--gap-start", default="2026-06-26T01:00:05-04:00",
        help="recovery window start (exclusive); the last real pass before the gap",
    )
    ap.add_argument(
        "--gap-end", default="2026-06-28T17:39:11-04:00",
        help="recovery window end (exclusive); the first real pass after resume",
    )
    ap.add_argument(
        "--dedup-window-s", type=float, default=20.0,
        help="skip a candidate within this many seconds of an existing same-direction pass",
    )
    ap.add_argument(
        "--scratch", default="recover_scratch",
        help="dir for dry-run artifacts (recordings + events go under here)",
    )
    ap.add_argument(
        "--commit", action="store_true",
        help="DANGER: actually write passes to camwatch.db (OFF by default). "
             "This phase: leave OFF. Hub upload is a separate later step.",
    )
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.commit:
        raise SystemExit(
            "--commit is intentionally NOT supported in this phase. DB inserts "
            "and hub uploads are gated off pending review. Remove --commit."
        )

    cfg = load_config(args.config)
    db = Database()  # read-only use here (dedup query only); no inserts.
    homog = Homography.from_profile(cfg.camera.profile)
    if homog is None:
        raise SystemExit("cx810 homography failed to load; cannot reprocess")

    gap_start = datetime.fromisoformat(args.gap_start)
    gap_end = datetime.fromisoformat(args.gap_end)

    # Artifacts land under the scratch dir so nothing touches the live
    # recordings/ or events/ trees.
    scratch = Path(args.scratch)
    rec_dir = scratch / "recordings"
    events_dir = scratch / "events"
    rec_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)
    # Point the borrowed worker + recorder at the scratch dirs. The worker's
    # _save_pass_trajectory_jsonl writes to recordings_dir.parent / "events".
    cfg.events_dir = events_dir  # used only for our Candidate bookkeeping

    cal = cfg.load_calibration()
    if cal is None:
        raise SystemExit("calibration.yaml not found; cannot reprocess")

    detector = Detector(
        weights=cfg.model.weights, device=cfg.model.device,
        classes=cfg.model.classes, conf=cfg.model.conf, iou=cfg.model.iou,
        roi=cal.roi, conf_per_class=cfg.model.conf_per_class,
    )
    worker = _CaptureWorkerShim(cfg=cfg, db=db, recordings_dir=rec_dir)

    clips = discover_clips(Path(args.clips_root), args.days)[: args.limit]
    log.info(
        "DRY-RUN: %d clip(s); recovery window (%s, %s); scratch=%s",
        len(clips), gap_start.isoformat(), gap_end.isoformat(), scratch.resolve(),
    )

    all_cands: list[Candidate] = []
    for clip in clips:
        log.info("clip %s", clip.name)
        try:
            cands = reprocess_clip(
                clip, cfg, db, worker, rec_dir, detector, homog,
                gap_start=gap_start, gap_end=gap_end,
                dedup_window_s=args.dedup_window_s,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("clip %s failed: %s", clip.name, e)
            continue
        all_cands.extend(cands)

    emitted = [c for c in all_cands if c.skipped_reason is None]
    skipped = [c for c in all_cands if c.skipped_reason is not None]

    print("\n==================== DRY-RUN SUMMARY ====================")
    print(f"clips processed : {len(clips)}")
    print(f"passes emitted  : {len(emitted)}")
    print(f"events skipped  : {len(skipped)} "
          f"(dedup/out-of-window; no artifacts written)")
    if emitted:
        n = sum(1 for c in emitted if c.direction == "N")
        s = sum(1 for c in emitted if c.direction == "S")
        speeds = [c.speed_mph for c in emitted if c.speed_mph is not None]
        print(f"direction split : N={n}  S={s}")
        if speeds:
            print(
                f"speed mph       : min={min(speeds):.1f} "
                f"mean={sum(speeds)/len(speeds):.1f} max={max(speeds):.1f} "
                f"(n_with_speed={len(speeds)}/{len(emitted)})"
            )
    print("\nclip -> captured_at | dir | mph | elapsed_s")
    for c in emitted:
        print(
            f"  {c.clip} -> {c.captured_at.isoformat(timespec='seconds')} | "
            f"{c.direction} | "
            f"{('%.1f' % c.speed_mph) if c.speed_mph is not None else 'None':>6} | "
            f"{c.elapsed_s:.2f}"
        )
    print(f"\nartifacts under: {scratch.resolve()}")
    print("  recordings/<clip>.jpg / .entry.jpg / .exit.jpg   (thumb + 3 anchors)")
    print("  events/pass_<negid>.jsonl                        (speed-chart trajectory)")
    print("\nNOTHING written to camwatch.db; NOTHING uploaded to the hub.")

    # Machine-readable manifest of the dry-run for review tooling.
    manifest = scratch / "dry_run_manifest.json"
    manifest.write_text(json.dumps(
        [
            {
                "clip": c.clip,
                "captured_at": c.captured_at.isoformat(timespec="seconds"),
                "direction": c.direction,
                "speed_mph": c.speed_mph,
                "elapsed_s": c.elapsed_s,
                "track_id": c.track_id,
                "cls_name": c.cls_name,
                "n_speed_samples": c.n_speed_samples,
                "rate_fps": c.rate_fps,
                "thumb_path": c.thumb_path,
                "entry_path": c.entry_path,
                "exit_path": c.exit_path,
                "trajectory_path": c.trajectory_path,
                "skipped_reason": c.skipped_reason,
            }
            for c in all_cands
        ],
        indent=2,
    ))
    print(f"manifest       : {manifest.resolve()}")


if __name__ == "__main__":
    main()
