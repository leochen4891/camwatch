"""Tests for the VictoriaMetrics push layer (`camwatch.metrics_push`).

Covers the observability contract (2026-06-observability effort):
exposition format VM's /api/v1/import/prometheus accepts, the base
`camera` label on every series, cumulative histogram buckets, and —
load-bearing — fail-open behavior: an unreachable VM must never raise
into the capture path.

Runs under pytest, or standalone: `python tests/test_metrics_push.py`.
"""
from __future__ import annotations

import pytest

from camwatch import metrics_push as mp
from camwatch.metrics import MetricsCollector
from camwatch.metrics_push import (
    Counter,
    Gauge,
    Histogram,
    MetricsPusher,
    Registry,
)


def _registry_with(*metrics):
    reg = Registry()
    for m in metrics:
        reg.register(m)
    return reg


# ---------- exposition format ----------

def test_counter_render_with_base_label():
    c = Counter("camwatch_engine_passes_total", "t", ("direction",))
    c.inc(direction="N")
    c.inc(direction="N")
    c.inc(direction="S")
    out = _registry_with(c).render({"camera": "cx810"})
    assert "# TYPE camwatch_engine_passes_total counter" in out
    assert 'camwatch_engine_passes_total{camera="cx810",direction="N"} 2' in out
    assert 'camwatch_engine_passes_total{camera="cx810",direction="S"} 1' in out


def test_gauge_render_float_and_unlabeled():
    g = Gauge("camwatch_engine_queue_depth", "t", ("queue",))
    g.set(2.5, queue="main")
    u = Gauge("camwatch_engine_capture_paused", "t")
    u.set(1)
    out = _registry_with(g, u).render({"camera": "cx810"})
    assert 'camwatch_engine_queue_depth{camera="cx810",queue="main"} 2.5' in out
    assert 'camwatch_engine_capture_paused{camera="cx810"} 1' in out


def test_histogram_buckets_are_cumulative():
    h = Histogram("camwatch_engine_pass_speed_mph", "t", (20.0, 30.0, 40.0),
                  ("direction",))
    for v in (18.0, 28.0, 33.0, 55.0):
        h.observe(v, direction="N")
    out = _registry_with(h).render({"camera": "cx810"})
    assert 'mph_bucket{camera="cx810",direction="N",le="20"} 1' in out
    assert 'mph_bucket{camera="cx810",direction="N",le="30"} 2' in out
    assert 'mph_bucket{camera="cx810",direction="N",le="40"} 3' in out
    assert 'mph_bucket{camera="cx810",direction="N",le="+Inf"} 4' in out
    assert 'camwatch_engine_pass_speed_mph_sum{camera="cx810",direction="N"} 134' in out
    assert 'camwatch_engine_pass_speed_mph_count{camera="cx810",direction="N"} 4' in out


def test_label_mismatch_is_a_programming_error():
    c = Counter("camwatch_engine_uploads_total", "t", ("status",))
    with pytest.raises(ValueError):
        c.inc(direction="N")  # wrong label name — caught in dev, not prod


# ---------- pusher ----------

class _CapturingClient:
    """Stand-in for httpx.Client capturing the POST."""

    seen: dict = {}
    status_code = 204

    def __init__(self, timeout):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, content=None, headers=None):
        _CapturingClient.seen = {
            "url": url, "content": content, "headers": headers,
        }

        class _Resp:
            status_code = _CapturingClient.status_code

        return _Resp()


def test_pusher_posts_rendered_registry(monkeypatch):
    c = Counter("camwatch_engine_frames_total", "t", ("stream",))
    c.inc(stream="main")
    reg = _registry_with(c)
    monkeypatch.setattr("camwatch.metrics_push.httpx.Client", _CapturingClient)
    pusher = MetricsPusher("http://localhost:8428", camera="cx810",
                           registry=reg)
    assert pusher.push_once() is True
    seen = _CapturingClient.seen
    assert seen["url"] == "http://localhost:8428/api/v1/import/prometheus"
    assert seen["headers"]["Content-Type"].startswith("text/plain")
    body = seen["content"].decode()
    assert 'camwatch_engine_frames_total{camera="cx810",stream="main"} 1' in body


def test_pusher_accepts_full_import_url(monkeypatch):
    monkeypatch.setattr("camwatch.metrics_push.httpx.Client", _CapturingClient)
    pusher = MetricsPusher(
        "http://localhost:8428/api/v1/import/prometheus",
        camera="cx810", registry=Registry(),
    )
    pusher.push_once()
    # The import path must not be doubled when config carries the full URL.
    assert _CapturingClient.seen["url"] == (
        "http://localhost:8428/api/v1/import/prometheus"
    )


def test_pusher_fail_open_on_unreachable_vm():
    """A down VM yields False, never an exception (capture path safety)."""
    pusher = MetricsPusher(
        # RFC 5737 TEST-NET address with an immediate-fail timeout via
        # the .invalid TLD: connection fails fast inside httpx.
        "http://vm.invalid:8428",
        camera="cx810", registry=_registry_with(
            Gauge("camwatch_engine_capture_paused", "t")
        ),
    )
    assert pusher.push_once() is False  # and no exception propagated


def test_pusher_fail_open_on_http_error(monkeypatch):
    monkeypatch.setattr("camwatch.metrics_push.httpx.Client", _CapturingClient)
    _CapturingClient.status_code = 503
    try:
        pusher = MetricsPusher("http://localhost:8428", camera="cx810",
                               registry=Registry())
        assert pusher.push_once() is False
    finally:
        _CapturingClient.status_code = 204


# ---------- collector dual-write into the module registry ----------

def test_collector_feeds_push_registry(tmp_path):
    """The record_* surface the capture path already calls must land in
    the push registry — frames as counters, yolo stage as the inference
    histogram, queue depth as a gauge, reconnects as reason counters."""
    from camwatch.db import Database

    coll = MetricsCollector(Database(path=tmp_path / "t.db"))
    coll.record_frame("pytest_stream")
    coll.record_stage("yolo", 0.025)
    coll.record_queue_depth("pytest_stream", 3)
    coll.record_frame_gap("pytest_stream", 0.07)
    coll.record_lag(0.12)
    coll.record_reconnect("pytest_reason")
    out = mp.REGISTRY.render({"camera": "cx810"})
    assert 'camwatch_engine_frames_total{camera="cx810",stream="pytest_stream"} 1' in out
    assert 'camwatch_engine_queue_depth{camera="cx810",queue="pytest_stream"} 3' in out
    assert 'camwatch_engine_rtsp_reconnects_total{camera="cx810",reason="pytest_reason"} 1' in out
    assert 'camwatch_engine_inference_seconds_bucket' in out
    assert 'camwatch_engine_frame_gap_seconds_bucket{camera="cx810",le="0.1",stream="pytest_stream"} 1' in out


def test_default_config_disables_push():
    """No `metrics:` section in config = no endpoint = the server never
    starts a pusher (Config default is None)."""
    from camwatch.config import Config

    assert Config.__dataclass_fields__["metrics_endpoint"].default is None


if __name__ == "__main__":
    import sys

    import pytest as _pytest

    sys.exit(_pytest.main([__file__, "-v"]))
