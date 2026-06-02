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
    # Same straight path but at 100 fps (bunched PTS) — denominator collapsed.
    mps = 30.0 / MPH_PER_MPS
    samples = _straight(n=10, dt=0.01, mps=mps)
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
    # ...the convergence guard is.
    assert math.isnan(_speed(samples))


def test_too_few_samples_is_nan():
    mps = 30.0 / MPH_PER_MPS
    assert math.isnan(_speed(_straight(n=_MIN_RUNNING_SAMPLES - 1, dt=0.05, mps=mps)))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all speed-guard tests passed")
