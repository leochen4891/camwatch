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
    _cadence_speed,
    _trim_stationary_dwell,
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


# ---------------------------------------------------------------------------
# Cadence-path tests. `_cadence_speed` is the live headline path: per-frame
# time is reconstructed from received-frame sequence numbers and the stream's
# measured received rate; the camera's PTS (first tuple element) is never
# consulted, because it is untrustworthy (see pts_timing_investigation.md).


def _cadence_traj(n, rate, mps, *, seq0=100, epoch=3, scrambled_ts=True):
    """Trajectory tuples (ts, u, v, bbox, seq, epoch) for a straight crossing
    at constant speed. Position advances with seq (i.e. with real time); ts is
    deliberately a constant garbage value by default, mimicking the camera's
    broken PTS — the cadence path must not care."""
    out = []
    for i in range(n):
        seq = seq0 + i
        y = mps * (seq - seq0) / rate
        ts = 999.0 if scrambled_ts else (seq - seq0) / rate
        out.append((ts, 0.0, y, (0.0, 0.0, 1.0, 1.0), seq, epoch))
    return out


def test_cadence_clean_pass():
    mps = 30.0 / MPH_PER_MPS
    mph, n = _cadence_speed(_cadence_traj(12, rate=13.8, mps=mps), 13.8, _homog())
    assert mph is not None
    assert abs(mph - 30.0) < 0.5
    assert n == 12


def test_cadence_ignores_scrambled_pts():
    # Identical positions and seqs, totally different ts values -> same speed.
    mps = 25.0 / MPH_PER_MPS
    a, _ = _cadence_speed(
        _cadence_traj(10, rate=14.0, mps=mps, scrambled_ts=True), 14.0, _homog())
    b, _ = _cadence_speed(
        _cadence_traj(10, rate=14.0, mps=mps, scrambled_ts=False), 14.0, _homog())
    assert a is not None and b is not None
    assert abs(a - b) < 1e-9


def test_cadence_seq_gap_adds_time():
    # A missed detection (or a frame dropped before YOLO) leaves a seq gap.
    # The car covered two frame-periods of distance in two frame-periods of
    # inferred time, so the speed stays true.
    mps = 30.0 / MPH_PER_MPS
    traj = _cadence_traj(12, rate=14.0, mps=mps)
    gapped = traj[:5] + traj[6:]  # drop one mid-pass detection
    mph, _ = _cadence_speed(gapped, 14.0, _homog())
    assert mph is not None
    assert abs(mph - 30.0) < 0.5


def test_cadence_epoch_change_returns_none():
    # A pass spanning an RTSP reconnect has no meaningful seq spacing.
    mps = 30.0 / MPH_PER_MPS
    traj = _cadence_traj(12, rate=14.0, mps=mps)
    t = traj[-1]
    traj[-1] = (t[0], t[1], t[2], t[3], t[4], t[5] + 1)
    mph, _ = _cadence_speed(traj, 14.0, _homog())
    assert mph is None


def test_cadence_no_rate_returns_none():
    # Stream warm-up: received_fps() has <10s of data and returns None.
    mps = 30.0 / MPH_PER_MPS
    traj = _cadence_traj(12, rate=14.0, mps=mps)
    assert _cadence_speed(traj, None, _homog())[0] is None
    assert _cadence_speed(traj, 0.0, _homog())[0] is None


def test_cadence_track_merge_rejected_on_shape_alone():
    # Arc length far beyond net displacement (a box merge) corrupts the
    # measured distance, so the speed is unknown even when the resulting
    # number is plausible — no magnitude gate on the cadence path.
    mps = 8.0 / MPH_PER_MPS  # slow, so the inflated headline stays under 55
    traj = _cadence_traj(12, rate=14.0, mps=mps)
    t = traj[6]
    traj[6] = (t[0], t[1] + 3.0, t[2], t[3], t[4], t[5])  # 3 m lateral leap
    mph, _ = _cadence_speed(traj, 14.0, _homog())
    assert mph is None


def test_cadence_too_few_samples_returns_none():
    mps = 30.0 / MPH_PER_MPS
    traj = _cadence_traj(_MIN_RUNNING_SAMPLES - 1, rate=14.0, mps=mps)
    assert _cadence_speed(traj, 14.0, _homog())[0] is None


# ---------------------------------------------------------------------------
# Stationary-dwell trim. A track acquired (or lost) while parked inside the grid
# leaves a run of near-motionless samples that add elapsed time but no distance,
# dragging the cumulative-average headline below the true crossing speed (the
# engine-cadence-underreport pass 26256: a ~1.5 s dwell pulled a ~45 mph
# crossing down to 19). `_cadence_speed` trims such a run before averaging.


def _cadence_dwell_then_cross(dwell_n, move_n, rate, mps, *, seq0=100, epoch=3):
    """A track parked at Y=0 for `dwell_n` frames, then a straight crossing for
    `move_n` frames at constant `mps`. Seq is contiguous throughout (the vehicle
    was tracked the whole time), so the dwell frames sit inside the measured
    span exactly as they did for pass 26256."""
    out = []
    seq = seq0
    for _ in range(dwell_n):  # confined within the stationary box at the origin
        out.append((999.0, 0.0, 0.0, (0.0, 0.0, 1.0, 1.0), seq, epoch))
        seq += 1
    for i in range(move_n):   # straight crossing, Y advancing each frame
        out.append((999.0, 0.0, mps * (i + 1) / rate,
                    (0.0, 0.0, 1.0, 1.0), seq, epoch))
        seq += 1
    return out


def test_cadence_stationary_lead_in_trimmed():
    # 20 parked frames then a 12-frame 40 mph crossing. Untrimmed, the ~1.4 s of
    # dwell time would pull the headline to ~15; the trim recovers ~40.
    mps = 40.0 / MPH_PER_MPS
    traj = _cadence_dwell_then_cross(dwell_n=20, move_n=12, rate=14.0, mps=mps)
    mph, n = _cadence_speed(traj, 14.0, _homog())
    assert mph is not None
    assert abs(mph - 40.0) < 1.0
    assert n == 12  # only the moving core is measured


def test_cadence_stationary_tail_trimmed():
    # Mirror image: the crossing happens first, then the vehicle stops inside
    # the grid (waiting to turn) and is tracked stationary for 20 frames.
    mps = 35.0 / MPH_PER_MPS
    cross = _cadence_dwell_then_cross(dwell_n=0, move_n=12, rate=14.0, mps=mps)
    last_y = cross[-1][2]
    seq = cross[-1][4] + 1
    tail = [(999.0, 0.0, last_y, (0.0, 0.0, 1.0, 1.0), seq + i, 3)
            for i in range(20)]
    mph, n = _cadence_speed(cross + tail, 14.0, _homog())
    assert mph is not None
    assert abs(mph - 35.0) < 1.0
    # The 20 stationary tail frames (and the arrival frame that shares their Y)
    # are cut; only the moving core is measured, far short of the 32 total.
    assert 10 <= n <= 12


def test_cadence_short_lead_in_not_trimmed():
    # A confined run shorter than the dwell floor (the ordinary frame or two a
    # moving car spends near its start) is left alone: a clean uniform pass is
    # returned whole, unchanged in speed and sample count.
    mps = 30.0 / MPH_PER_MPS
    traj = _cadence_traj(12, rate=14.0, mps=mps)
    trimmed = _trim_stationary_dwell(
        [((s[4] - traj[0][4]) / 14.0, s[1], s[2]) for s in traj], _homog())
    assert len(trimmed) == 12  # nothing trimmed
    mph, n = _cadence_speed(traj, 14.0, _homog())
    assert abs(mph - 30.0) < 0.5 and n == 12


def test_cadence_all_dwell_returns_none():
    # A track that never moves (curb-parked, bbox jitter only) has no moving
    # core to measure: speed is unknown, not a fabricated crawl.
    traj = [(999.0, 0.0, 0.0, (0.0, 0.0, 1.0, 1.0), 100 + i, 3) for i in range(30)]
    assert _cadence_speed(traj, 14.0, _homog())[0] is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all speed-guard tests passed")
