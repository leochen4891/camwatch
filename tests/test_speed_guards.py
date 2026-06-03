"""Regression tests for the headline-speed trustworthiness guards in
`Homography.running_avg_speed`.

Two real failure modes once produced phantom over-speeds on a residential
street (a 100 mph Mazda CX-90, a 79 mph Tesla, a 67 mph "two cars meeting"):

  * Timing compression — a variable-frame-rate camera delivered frames with
    bunched presentation timestamps, collapsing the trajectory's time span
    (the speed denominator) and inflating the result.
  * Spatial jump — the focus track's box merged with an oncoming vehicle, so
    its ground point leapt sideways/backward and inflated cumulative arc length.

The guards reject both (speed -> NaN, surfaced as NULL) instead of reporting a
fabricated number, while leaving clean crossings untouched.

Runs under pytest, or standalone: `python tests/test_speed_guards.py`.
"""
from __future__ import annotations

import math

from camwatch.capture_worker import (
    _MAX_ARC_DISPLACEMENT_RATIO,
    _MAX_EXIT_DESCENT,
    _MAX_PLAUSIBLE_FPS,
    _MAX_PLAUSIBLE_MPH,
    _MIN_RUNNING_SAMPLES,
)
from camwatch.homography import MPH_PER_MPS, Homography


def _homog() -> Homography:
    """A Homography whose `project` is the identity, so test inputs are fed
    directly as ground-plane (X, Y) metres and the geometry is exact."""
    h = Homography.__new__(Homography)
    h.project = lambda u, v: (u, v)  # type: ignore[method-assign]
    return h


def _speed(samples, **overrides):
    kwargs = dict(
        min_samples=_MIN_RUNNING_SAMPLES,
        max_plausible_fps=_MAX_PLAUSIBLE_FPS,
        max_arc_displacement_ratio=_MAX_ARC_DISPLACEMENT_RATIO,
        max_exit_descent=_MAX_EXIT_DESCENT,
        max_plausible_mph=_MAX_PLAUSIBLE_MPH,
    )
    kwargs.update(overrides)
    final, _per_frame, _n = _homog().running_avg_speed(samples, **kwargs)
    return final


def _straight(n, dt, mps):
    """A straight northbound crossing: n samples, fixed dt, constant speed."""
    return [(i * dt, 0.0, i * dt * mps) for i in range(n)]


def test_clean_crossing_reports_true_speed():
    # 30 mph straight crossing at 20 fps — well within both guards.
    mps = 30.0 / MPH_PER_MPS
    final = _speed(_straight(n=10, dt=0.05, mps=mps))
    assert not math.isnan(final)
    assert abs(final - 30.0) < 0.5


def test_timing_compressed_is_rejected():
    # A 30 mph crossing whose timestamps are compressed 6x (real dt 0.06 s
    # recorded as 0.01 s): the distance is real but the span collapses, so the
    # headline inflates to ~180 mph — both suspicious (≈900 fps) and over the
    # ceiling, so it is rejected.
    real_mps = 30.0 / MPH_PER_MPS
    samples = [(i * 0.01, 0.0, i * real_mps * 0.06) for i in range(10)]
    assert (len(samples) - 1) / (samples[-1][0] - samples[0][0]) > _MAX_PLAUSIBLE_FPS
    assert math.isnan(_speed(samples))


def test_spatial_jump_is_rejected():
    # Clean cadence, but one sample leaps far sideways then returns (a box
    # merge with an oncoming car), inflating arc length past displacement.
    mps = 30.0 / MPH_PER_MPS
    samples = _straight(n=10, dt=0.05, mps=mps)
    t, x, y = samples[5]
    samples[5] = (t, x + 12.0, y)  # 12 m lateral excursion
    assert math.isnan(_speed(samples))


def _early_burst():
    """An acquisition burst (7 frames bunched in time but spread in space) then
    a normally-timed tail at the true speed. The per-pass average frame rate
    stays under the fps cap, but the burst inflates the running average so it is
    still descending at grid exit — the partial-burst case (cf. pass 12708)."""
    samples = []
    t = 0.0
    for i in range(7):           # burst: dt 0.006 s, 2 m apart  → ~fast
        samples.append((t, 0.0, 2.0 * i))
        t += 0.006
    y = 2.0 * 6
    for _ in range(6):           # tail: dt 0.07 s, 1 m apart  → true speed
        y += 1.0
        t += 0.07
        samples.append((t, 0.0, y))
    return samples


def test_partial_burst_not_converged_is_rejected():
    samples = _early_burst()
    # The global fps guard must NOT be what fires here (the tail dilutes it)...
    fps = (len(samples) - 1) / (samples[-1][0] - samples[0][0])
    assert fps < _MAX_PLAUSIBLE_FPS
    assert not math.isnan(_speed(samples, max_exit_descent=None))  # only guard C rejects
    # ...the convergence guard is (the headline ~87 mph is over the ceiling).
    assert math.isnan(_speed(samples))


def test_magnitude_gate_keeps_plausible_suspicious_pass():
    # The same suspicious-shape burst is KEPT when its headline is below the
    # ceiling: raising the ceiling above the ~87 mph headline means the
    # convergence flag no longer rejects it. A suspicious shape alone is not
    # enough — the speed must also be implausibly high.
    samples = _early_burst()
    assert math.isnan(_speed(samples, max_plausible_mph=55.0))
    assert not math.isnan(_speed(samples, max_plausible_mph=200.0))


def test_too_few_samples_is_nan():
    mps = 30.0 / MPH_PER_MPS
    assert math.isnan(_speed(_straight(n=_MIN_RUNNING_SAMPLES - 1, dt=0.05, mps=mps)))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all speed-guard tests passed")
