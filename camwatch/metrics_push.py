"""Push-mode metrics for VictoriaMetrics (the observability contract).

Minimal Prometheus-style primitives (Counter / Gauge / Histogram) plus a
background pusher that renders the registry in Prometheus text exposition
format and POSTs it to VictoriaMetrics' import endpoint
(`/api/v1/import/prometheus`) on a fixed interval. VM stamps each push
with its arrival time, so series carry no client timestamps.

Contract (camwatch-system, 2026-06-observability effort):
  * metric names are prefixed `camwatch_engine_` with unit suffixes;
    counters end `_total`
  * label values come from BOUNDED sets only (`camera`, `stream`,
    `queue`, `direction`, `method`, `status`, `reason`) — never ids,
    filenames, or timestamps
  * the elected main camera is applied to every series as a base
    `camera` label by the pusher (ADR-013 provenance)
  * fail-open: a down or slow VM must never break or block the capture
    path — pushes are try/except'd, time-bounded, and run on their own
    thread; record calls are O(1) dict updates under a per-metric lock

Endpoint + interval come from config (`metrics:` section, untracked
config.yaml); no endpoint configured = the pusher never starts and the
record calls are just cheap in-memory updates.

Instrumentation convention (mirrors prometheus_client): metrics are
module-level objects; call sites import and update them directly, e.g.

    from . import metrics_push as mp
    mp.PASSES.inc(direction="N", method="cadence_seq")
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

log = logging.getLogger(__name__)

_IMPORT_PATH = "/api/v1/import/prometheus"


def _fmt(v: float) -> str:
    """Shortest exact-ish float form Prometheus parsers accept."""
    if v == float("inf"):
        return "+Inf"
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


def _label_str(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(
        '{}="{}"'.format(k, str(v).replace("\\", "\\\\").replace('"', '\\"'))
        for k, v in sorted(labels.items())
    )
    return "{" + inner + "}"


class _Metric:
    """Shared base: a named metric with label-keyed series."""

    kind = "untyped"

    def __init__(self, name: str, help_: str, label_names: tuple[str, ...]) -> None:
        self.name = name
        self.help = help_
        self._label_names = label_names
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if set(labels) != set(self._label_names):
            raise ValueError(
                f"{self.name}: labels {sorted(labels)} != declared {sorted(self._label_names)}"
            )
        return tuple(str(labels[k]) for k in self._label_names)

    def _labels_of(self, key: tuple[str, ...]) -> dict[str, str]:
        return dict(zip(self._label_names, key))

    def render(self, base_labels: dict[str, str]) -> list[str]:  # pragma: no cover
        raise NotImplementedError


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name: str, help_: str, label_names: tuple[str, ...] = ()) -> None:
        super().__init__(name, help_, label_names)
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        k = self._key(labels)
        with self._lock:
            self._values[k] = self._values.get(k, 0.0) + amount

    def render(self, base_labels: dict[str, str]) -> list[str]:
        with self._lock:
            items = list(self._values.items())
        lines = [f"# TYPE {self.name} {self.kind}"]
        for key, v in items:
            lines.append(f"{self.name}{_label_str(base_labels | self._labels_of(key))} {_fmt(v)}")
        return lines


class Gauge(_Metric):
    kind = "gauge"

    def __init__(self, name: str, help_: str, label_names: tuple[str, ...] = ()) -> None:
        super().__init__(name, help_, label_names)
        self._values: dict[tuple[str, ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        k = self._key(labels)
        with self._lock:
            self._values[k] = float(value)

    def render(self, base_labels: dict[str, str]) -> list[str]:
        with self._lock:
            items = list(self._values.items())
        lines = [f"# TYPE {self.name} {self.kind}"]
        for key, v in items:
            lines.append(f"{self.name}{_label_str(base_labels | self._labels_of(key))} {_fmt(v)}")
        return lines


class Histogram(_Metric):
    """Fixed-bucket cumulative histogram (Grafana: histogram_quantile)."""

    kind = "histogram"

    def __init__(
        self,
        name: str,
        help_: str,
        buckets: tuple[float, ...],
        label_names: tuple[str, ...] = (),
    ) -> None:
        super().__init__(name, help_, label_names)
        self._buckets = tuple(sorted(buckets))
        # key → (per-bucket counts, sum, count)
        self._series: dict[tuple[str, ...], tuple[list[int], float, int]] = {}

    def observe(self, value: float, **labels: str) -> None:
        k = self._key(labels)
        v = float(value)
        with self._lock:
            counts, total, n = self._series.get(k) or ([0] * len(self._buckets), 0.0, 0)
            for i, le in enumerate(self._buckets):
                if v <= le:
                    counts[i] += 1
            self._series[k] = (counts, total + v, n + 1)

    def render(self, base_labels: dict[str, str]) -> list[str]:
        with self._lock:
            items = [(k, (list(c), s, n)) for k, (c, s, n) in self._series.items()]
        lines = [f"# TYPE {self.name} {self.kind}"]
        for key, (counts, total, n) in items:
            labels = base_labels | self._labels_of(key)
            for le, c in zip(self._buckets, counts):
                lines.append(
                    f"{self.name}_bucket{_label_str(labels | {'le': _fmt(le)})} {c}"
                )
            lines.append(f"{self.name}_bucket{_label_str(labels | {'le': '+Inf'})} {n}")
            lines.append(f"{self.name}_sum{_label_str(labels)} {_fmt(total)}")
            lines.append(f"{self.name}_count{_label_str(labels)} {n}")
        return lines


class Registry:
    def __init__(self) -> None:
        self._metrics: list[_Metric] = []
        self._lock = threading.Lock()

    def register(self, metric: _Metric) -> _Metric:
        with self._lock:
            self._metrics.append(metric)
        return metric

    def render(self, base_labels: dict[str, str] | None = None) -> str:
        base = base_labels or {}
        with self._lock:
            metrics = list(self._metrics)
        out: list[str] = []
        for m in metrics:
            out.extend(m.render(base))
        return "\n".join(out) + "\n"


REGISTRY = Registry()


def _counter(name: str, help_: str, labels: tuple[str, ...] = ()) -> Counter:
    return REGISTRY.register(Counter(name, help_, labels))  # type: ignore[return-value]


def _gauge(name: str, help_: str, labels: tuple[str, ...] = ()) -> Gauge:
    return REGISTRY.register(Gauge(name, help_, labels))  # type: ignore[return-value]


def _histogram(
    name: str, help_: str, buckets: tuple[float, ...], labels: tuple[str, ...] = ()
) -> Histogram:
    return REGISTRY.register(Histogram(name, help_, buckets, labels))  # type: ignore[return-value]


# ---------- the engine's metrics ----------

FRAMES = _counter(
    "camwatch_engine_frames_total",
    "Frames decoded (stream=main) / consumed by the detector loop (stream=yolo).",
    ("stream",),
)
FRAME_GAP = _histogram(
    "camwatch_engine_frame_gap_seconds",
    "Wallclock interval between consecutive decoded frames (burst signal).",
    (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    ("stream",),
)
INFERENCE = _histogram(
    "camwatch_engine_inference_seconds",
    "Per-frame YOLO inference latency.",
    (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0),
)
FRAME_LAG = _histogram(
    "camwatch_engine_frame_lag_seconds",
    "Frame age (now - capture ts) when the detector loop picks it up.",
    (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)
QUEUE_DEPTH = _gauge(
    "camwatch_engine_queue_depth",
    "Reader-to-consumer FIFO occupancy at drain time (0-3 healthy).",
    ("queue",),
)
RTSP_RECONNECTS = _counter(
    "camwatch_engine_rtsp_reconnects_total",
    "Stream reconnect causes (open_failed / decode_failures / demux_error).",
    ("reason",),
)
PASSES = _counter(
    "camwatch_engine_passes_total",
    "Vehicle passes recorded, by direction and speed method ('none' = no speed).",
    ("direction", "method"),
)
PASS_SPEED = _histogram(
    "camwatch_engine_pass_speed_mph",
    "Recorded pass speeds (histogram_quantile gives the p95 speeding signal).",
    (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 75.0),
    ("direction",),
)
ALARMS = _counter(
    "camwatch_engine_alarms_total",
    "Passes at or above the configured alert threshold.",
    ("direction",),
)
UPLOADS = _counter(
    "camwatch_engine_uploads_total",
    "Hub ingest attempts by outcome (ok / http_error / network_error).",
    ("status",),
)
UPLOAD_SECONDS = _histogram(
    "camwatch_engine_upload_seconds",
    "Hub ingest POST latency (successful uploads).",
    (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)
UPLOAD_PENDING = _gauge(
    "camwatch_engine_upload_pending",
    "Passes awaiting first upload to the hub.",
)
ENRICHMENT = _counter(
    "camwatch_engine_enrichment_total",
    "Local enricher handoffs by outcome (ok / error / skipped_night).",
    ("status",),
)
STAMP_CORRECTION = _histogram(
    "camwatch_engine_stamp_correction_seconds",
    "Per-pass captured_at correction (now - grid-entry frame ts): transit "
    "plus processing staleness. Uptime-stable small values = stamp fix OK.",
    (1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0),
)
CAPTURE_PAUSED = _gauge(
    "camwatch_engine_capture_paused",
    "1 while detection is paused (night/IR mode), else 0.",
)
START_TIME = _gauge(
    "camwatch_engine_start_time_seconds",
    "Unix time the capture service started (uptime = time() - this).",
)


# ---------- the pusher ----------

class MetricsPusher:
    """Renders REGISTRY and POSTs it to VictoriaMetrics every interval.

    Owns a daemon thread; both the loop and each push are fail-open. To
    keep a down VM from spamming the log at every interval, only state
    *transitions* (ok→failing, failing→ok) log above DEBUG.
    """

    def __init__(
        self,
        endpoint: str,
        camera: str,
        interval_s: float = 15.0,
        registry: Registry = REGISTRY,
    ) -> None:
        base = endpoint.rstrip("/")
        self._url = base if base.endswith(_IMPORT_PATH) else base + _IMPORT_PATH
        self._base_labels = {"camera": camera}
        self._interval_s = max(1.0, float(interval_s))
        self._registry = registry
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failing = False

    def start(self) -> None:
        if self._thread is not None:
            return
        START_TIME.set(time.time())
        self._thread = threading.Thread(
            target=self._run, name="metrics-push", daemon=True
        )
        self._thread.start()
        log.info(
            "metrics pusher started -> %s every %.0fs", self._url, self._interval_s
        )

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        self.push_once()  # best-effort final flush

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.push_once()

    def push_once(self) -> bool:
        """One render+POST. Never raises; returns success for tests/logs."""
        try:
            body = self._registry.render(self._base_labels)
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(
                    self._url,
                    content=body.encode(),
                    headers={"Content-Type": "text/plain; charset=utf-8"},
                )
            ok = resp.status_code in (200, 204)
            if not ok and not self._failing:
                log.warning("metrics push: VM HTTP %d (will keep retrying quietly)",
                            resp.status_code)
        except Exception as e:  # noqa: BLE001 — fail-open by contract
            ok = False
            if not self._failing:
                log.warning("metrics push failed: %s (will keep retrying quietly)", e)
        if ok and self._failing:
            log.info("metrics push recovered")
        self._failing = not ok
        return ok
