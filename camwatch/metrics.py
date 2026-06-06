"""Performance metrics collection for the capture pipeline.

The capture worker and its RtspStreams hand raw samples to a single
`MetricsCollector` instance. The collector buckets them into 5-second
windows and a background thread flushes one row per (bucket, metric)
to the `metrics` table every BUCKET_S seconds.

What's recorded:
  * fps_main               — frames/sec decoded by the main-stream RtspStream
  * fps_yolo               — frames/sec consumed by the YOLO loop
                             (≈ fps_main under healthy load; gap signals
                             frames dropped at the reader→consumer queue)
  * yolo_ms_p50/p95        — per-frame inference latency
  * lag_ms_p50/p95         — wallclock - frame.ts for frames reaching
                             the consumer (how far behind realtime)
  * queue_depth_main       — mean depth of the reader→consumer FIFO
                             (0–3 healthy; sustained higher = backlog)
  * frame_gap_ms_max_main  — longest inter-arrival gap between decoded
                             frames in the bucket (Reolink burst signal)

All `record_*` calls are O(1) and lock only briefly. The flush thread
runs once per bucket — ~8 inserts every 5s, totally negligible load.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from . import metrics_push as mp

if TYPE_CHECKING:
    from .db import Database

log = logging.getLogger(__name__)

BUCKET_S = 5.0  # bucket width in seconds


def _bucket_iso_for(now_unix: float) -> str:
    """Floor `now_unix` to the next BUCKET_S boundary and return as
    local-aware ISO seconds (matches `passes.captured_at` formatting)."""
    bucket_unix = (int(now_unix) // int(BUCKET_S)) * int(BUCKET_S)
    return datetime.fromtimestamp(bucket_unix, tz=timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )


class _Bucket:
    """Per-bucket scratch state. All fields touched only under the
    collector's lock."""

    __slots__ = ("frame_counts", "stage_durations_ms", "lag_ms",
                 "queue_depths", "buffer_lag_ms", "frame_gap_max_ms")

    def __init__(self) -> None:
        self.frame_counts: dict[str, int] = {}      # label → count
        self.stage_durations_ms: dict[str, list[float]] = {}  # name → [ms,...]
        self.lag_ms: list[float] = []
        self.queue_depths: dict[str, list[int]] = {}  # stream label → [depth,...]
        self.buffer_lag_ms: list[float] = []
        self.frame_gap_max_ms: dict[str, float] = {}  # stream label → max ms in bucket


def _percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1))))
    return s[k]


class MetricsCollector:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------- ingest ----------

    def _bucket(self, now_unix: float | None = None) -> _Bucket:
        ts = _bucket_iso_for(now_unix if now_unix is not None else time.time())
        b = self._buckets.get(ts)
        if b is None:
            b = _Bucket()
            self._buckets[ts] = b
        return b

    def record_frame(self, label: str) -> None:
        """One decoded frame on the named stream ('sub', 'main', 'yolo')."""
        mp.FRAMES.inc(stream=label)
        with self._lock:
            b = self._bucket()
            b.frame_counts[label] = b.frame_counts.get(label, 0) + 1

    def record_stage(self, name: str, dt_s: float) -> None:
        """A per-frame stage duration ('yolo' is the one we surface; others
        are accepted but currently ignored by the flush)."""
        if name == "yolo":
            mp.INFERENCE.observe(dt_s)
        with self._lock:
            b = self._bucket()
            b.stage_durations_ms.setdefault(name, []).append(dt_s * 1000.0)

    def record_lag(self, dt_s: float) -> None:
        """now - frame.ts at the moment the consumer picked it up."""
        mp.FRAME_LAG.observe(dt_s)
        with self._lock:
            b = self._bucket()
            b.lag_ms.append(dt_s * 1000.0)

    def record_queue_depth(self, label: str, depth: int) -> None:
        """Reader→consumer queue occupancy at drain time, per stream.
        Emitted as `queue_depth_<label>` so main and sub can be charted
        side-by-side."""
        mp.QUEUE_DEPTH.set(depth, queue=label)
        with self._lock:
            b = self._bucket()
            b.queue_depths.setdefault(label, []).append(int(depth))

    def record_reconnect(self, reason: str) -> None:
        """An RTSP session reopen, by bounded cause ('open_failed',
        'decode_failures', 'demux_error'). Push-registry only — the SQLite
        buckets never carried this and the UI doesn't read it."""
        mp.RTSP_RECONNECTS.inc(reason=reason)

    def record_buffer_lag(self, dt_s: float) -> None:
        """Legacy: was written by the dual-stream main-stream buffer (now
        removed). The field stays so historical rows still chart; no caller
        writes to it post-single-stream migration."""
        with self._lock:
            b = self._bucket()
            b.buffer_lag_ms.append(dt_s * 1000.0)

    def record_frame_gap(self, label: str, dt_s: float) -> None:
        """Wallclock interval between consecutive decoded frames on a stream.
        We keep only the MAX per bucket — that's the burstiness signal: a
        healthy stream maxes out near its frame interval, a stalled-then-burst
        stream spikes to multi-second values."""
        mp.FRAME_GAP.observe(dt_s, stream=label)
        ms = dt_s * 1000.0
        with self._lock:
            b = self._bucket()
            cur = b.frame_gap_max_ms.get(label, 0.0)
            if ms > cur:
                b.frame_gap_max_ms[label] = ms

    # ---------- background flush ----------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="metrics-flush", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2 * BUCKET_S)
        self._flush_closed_buckets(force_all=True)

    def _run(self) -> None:
        # Wake half a bucket past each boundary so the bucket we flush is
        # definitely closed (no more samples landing in it).
        while not self._stop.wait(BUCKET_S / 2.0):
            try:
                self._flush_closed_buckets()
            except Exception:  # noqa: BLE001
                log.exception("metrics flush failed")

    def _flush_closed_buckets(self, force_all: bool = False) -> None:
        now_ts = _bucket_iso_for(time.time())
        with self._lock:
            keys = sorted(self._buckets.keys())
            ready = [k for k in keys if force_all or k < now_ts]
            drained: list[tuple[str, _Bucket]] = [
                (k, self._buckets.pop(k)) for k in ready
            ]
        for ts_iso, b in drained:
            samples: dict[str, float] = {}
            # Frame rates: count / BUCKET_S. Keys we promote to columns:
            for label, count in b.frame_counts.items():
                samples[f"fps_{label}"] = count / BUCKET_S
            yolo_durs = b.stage_durations_ms.get("yolo")
            if yolo_durs:
                samples["yolo_ms_p50"] = _percentile(yolo_durs, 50)
                samples["yolo_ms_p95"] = _percentile(yolo_durs, 95)
            if b.lag_ms:
                samples["lag_ms_p50"] = _percentile(b.lag_ms, 50)
                samples["lag_ms_p95"] = _percentile(b.lag_ms, 95)
            for label, depths in b.queue_depths.items():
                if depths:
                    samples[f"queue_depth_{label}"] = sum(depths) / len(depths)
            if b.buffer_lag_ms:
                samples["buffer_lag_ms_p50"] = _percentile(b.buffer_lag_ms, 50)
                samples["buffer_lag_ms_p95"] = _percentile(b.buffer_lag_ms, 95)
            for label, max_ms in b.frame_gap_max_ms.items():
                samples[f"frame_gap_ms_max_{label}"] = max_ms
            if samples:
                self._db.insert_metric_samples(ts_iso, samples)
