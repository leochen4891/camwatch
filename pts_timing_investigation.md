# PTS Timing Investigation

**Date:** June 3, 2026 (Wednesday)
**Camera under investigation:** Reolink CX810, 4K main stream, the active CamWatch producer
**Status:** Investigation complete. Root cause settled. Fix direction decided; implementation pending.

---

## TL;DR

CamWatch speed measurements have been corrupted by unreliable per-frame timestamps (PTS) on the live RTSP stream. After a full day of controlled experiments across three cameras, the verdict is:

1. **The corruption is a firmware flaw in the camera's RTP packetizer**, present only on the live RTSP stream. It is model-dependent (E1 clean, CX410W mild, CX810 severe) and amplified by encoding load (4K HEVC).
2. **The camera's own recordings (SD card, delivered via FTP) always carry clean, honest timestamps**, on all three camera models. The sensor and encoder are fine.
3. **CamWatch code, the NVDEC decode path, and the network are all exonerated** by direct measurement.
4. **No camera setting can fix it.** Codec, frame rate, recording mode, on-camera recording, and resolution were each ruled out.
5. The CX810's true capture cadence is **about 13.8 fps**, not the configured "15 fps Constant". This was confirmed by three independent methods.
6. The speed corruption is **bidirectional**: timestamp micro-bursts inflate speeds (a 40 mph car stored as 63.8), and fake stalls deflate them (a ~32 mph car stored as 16.5). The existing trustworthiness guards only catch a subset of the inflated cases and none of the deflated ones.
7. **The durable fix is software:** compute speed from the per-frame displacement (geometry, which is reliable) multiplied by the measured cadence, instead of trusting RTSP timestamps. The FTP recordings serve as a continuous validation oracle.

---

## 1. Background

### Camera and method history

The project has used three cameras (see blog posts [camwatch](https://leidevs.com/blog/camwatch), [camwatch-2](https://leidevs.com/blog/camwatch-2), [camwatch-3](https://leidevs.com/blog/camwatch-3/), [camwatch-4](https://leidevs.com/blog/camwatch-4/)):

| era | camera | main stream | notes |
|---|---|---|---|
| 1 | E1 Outdoor | 1440p H.264 | Started with frame arrival time, then switched to PTS, which was found to be stable ("camera-side std=0") |
| 2 | CX410W | 1440p H.264 | Swapped for low-light performance; PTS bursts first appeared; speed method moved toward cumulative-distance over cumulative-time |
| 3 | CX810 (current, since ~pass 6693) | 4K HEVC | Swapped for 4K resolution; PTS jitter got significantly worse |

Relevant code history: commit `bbca4dd` (May 9) replaced the 2-line crossing-time measurement with a grid trigger plus trajectory method; the live-RTSP/NVDEC cutover landed mid-May (`0af741b`); the current scene-fit K+D+H calibration for the CX810 at 4K landed May 14 (`ead1228`, mean reprojection error 0.13 m). Note: the comment in `homography.py` attributing K/D to the CX410W is stale; the actual matrix is calibrated for the CX810 at 3840x2160 (cx,cy = 1920,1080).

### The speed calculation and guards (state at start of day)

Headline speed is `cum_arc_length / (t_last - t_0)` over the in-grid trajectory (running average at grid exit), using PTS-derived timestamps. After phantom over-speeds (passes 12533, 12542, 12588, 12708), trustworthiness guards were added (branch `fix/speed-trustworthiness-guards`, PR #2):

- implied fps ceiling (35), arc/displacement ratio (1.4), exit-descent convergence check (8%)
- a magnitude gate: reject only when the shape is suspicious AND the headline exceeds 55 mph

These guards were known to catch the extreme inflated phantoms. This investigation revealed they cannot catch deflated speeds at all (section 8).

---

## 2. The symptom

Per-frame PTS within a pass shows physically impossible spacing. Typical patterns, measured repeatedly:

- **Micro-bursts:** many frames stamped within milliseconds of each other. Example, pass 12848: 12 frames crammed into the first 48 ms of the pass. Pass 13009: 8 frames within 29 ms.
- **Fake stalls:** single inter-frame gaps of 400-850 ms, with the car's motion proving no frames were actually missing (section 4).
- The instantaneous-speed chart spikes to 600-2000 mph during bursts; the running average starts wildly inflated and decays (or never converges).

The burst location moves around (pass 12848: at the start; pass 12842: at the end), and on a 1-2 s pass it does not average out.

---

## 3. What was ruled out

### Camera settings (all tested on the CX810, live RTSP measured before/after)

| lever | tested | result |
|---|---|---|
| Recording mode (continuous vs car-sensing) | yes | no effect; burst moved location, persisted |
| Codec H.265 vs H.264 (at 4K) | yes, verified live (both RTSP paths served h264 after the switch) | byte-for-byte identical burst |
| Frame rate 25 / 20 / 15 / 12 | yes, each | no effect on the scramble |
| On-camera recording off | yes | reduced micro-clustering (38% to 2.7% of frames under 10 ms apart in a 25 s capture) but fabrication persisted; passes still corrupted |
| Frame Rate Mode | already Constant | the recording is still effectively variable (~13.8 fps); "Constant" is a target, not a guarantee |
| Resolution to 1440p | not needed | the three-camera comparison (section 7) shows the CX410W bursts at 1440p H.264, so resolution is not the cause |

### Network

Ping from lei-ubuntu to the camera: 20 packets, 0% loss, rtt min/avg/max/mdev = 0.204/0.365/1.376/0.238 ms. A clean wired LAN cannot cause this, and a clean network faithfully delivers whatever clumping the camera produces.

### Our own code (decisive, section 6)

- **Stock ffprobe** (none of our code or options) reading the raw RTSP packets sees the identical scramble: dt min 5.6 ms, median 31.5 ms, max 2104 ms, 16/180 packets under 10 ms apart.
- **PyAV `packet.pts` (pre-decode) equals `frame.pts` (post-NVDEC)** in the same capture: min 3.6 / med 15.4 / max 1571.7 ms in both; 71 vs 72 sub-10 ms gaps (a single boundary frame). The decode is timestamp-transparent. There is nothing in CamWatch to fix.

### Side finding: pipeline headroom

While testing frame rates, the metrics table showed the desktop pipeline (yolo11l on the 3060 Ti, ~35-49 ms p95 per frame) keeps up at 15 fps (lag ~0) but goes underwater at 20-25 fps: `fps_yolo` 16.6 vs 18.8 delivered, `lag_ms_p95` growing monotonically to 6.7 s (queue overflow). Conclusion: with the current model, do not run the camera above ~15 fps, independent of the PTS issue.

---

## 4. The key analytical insight: geometry is honest, PTS is not

Pass 12986 provided the smoking gun. Two consecutive frames showed a 0.78 s PTS gap, while the car moved only 1.07 m between them, the same ~1.1 m it moves between frames stamped 11.6 ms apart:

```
 i   t_rel   dt(ms)    step_m
 0   0.000     0.0      0.00
 1   0.776   775.8      1.07   <- "0.78 s" gap, one normal frame-step of motion
 2   1.206   430.3      1.09
 3   1.218    11.6      1.14   <- "11.6 ms" gap, same motion
 ...all 17 steps: 0.76-1.50 m
```

Two conclusions follow directly:

1. **The frames are captured at a uniform real-time cadence** (equal displacement per frame at constant vehicle speed), and the PTS values are fabricated in both directions (compressed clusters and stretched gaps).
2. **No frames were dropped inside the fake gaps.** If ~11 frames were missing in the 0.78 s gap, the car would have moved ~12 m, not 1.07 m.

This generalizes: across every pass measured, the per-frame displacement is steady (`step_cv` 0.09-0.22) while the PTS intervals swing wildly (`dt_cv` 0.5-2.7). A useful per-pass diagnostic is therefore: **steady motion plus wild intervals equals fabricated timing.** Note that counting only sub-10 ms bursts undercounts the damage; pass 13067 had zero micro-bursts but a 737 ms fake stall.

The projected geometry itself (homography X/Y) is trustworthy. It is the clock that lies.

---

## 5. Measuring the true cadence

Since the timestamps cannot be trusted, the camera's burn-in OSD clock (the date/time the camera bakes into the pixels at capture, e.g. `06/03/2026 11:03:24 am WED`) was used as an independent time reference. It is rendered by the ISP before encoding, so it appears identically in every output and is immune to both PTS fabrication and delivery bursts.

### Method

1. Decode a recorded clip (the recorder writes every decoded frame into the mp4, one for one, so frame counts are preserved).
2. Crop the OSD region (top-center; our own overlays are top-left and top-right).
3. Detect second-ticks as frames where the OSD digits change. Robust recipe: downscale the crop hard (e.g. to 80x8, which averages away H.264 edge-ringing noise), mean-absolute-diff consecutive crops, threshold ~0.6, and drop ticks within 3 frames of the clip edges (keyframe artifacts).
4. **Count only fully-bounded seconds**: a second is valid only if both its entering and leaving ticks are visible. The first and last seconds of every clip are truncated by the clip boundary and must be discarded. (Counting truncated seconds was an early mistake in this investigation and produced a spurious "wobble".)

### Results

- Pass 13009's clip, read manually frame-by-frame: second `:24` contains exactly **14 frames** (frames 8-21).
- Sweep across 35 clips, 84 fully-bounded seconds: **mode 14, median 14**, with 68% of samples in 13-15. High outliers at 28-29 are detector merge errors (exactly 2x14, which itself confirms the rate); the low tail is split errors.
- Independently, the camera-native FTP recordings (next section) carry container PTS averaging **13.75-13.79 fps** across every clip measured.

**Conclusion: the CX810, configured "15 fps Constant", actually delivers ~13.8 fps.** The camera quantizes recording timestamps to ~45.5 ms units (4096 ticks at the 90 kHz clock) and alternates 45.5/91 ms intervals to approximate 15 fps, landing at ~13.8 effective with occasional 137 ms (3-unit) gaps. Never use the nominal setting as a cadence constant; measure it.

---

## 6. Camera-native recordings vs the live stream

All three cameras were configured to record passing cars and upload via FTP (vsftpd on lei-ubuntu, root `/srv/nas/files/<camera-name>/YYYY/MM/DD/`). These clips are the camera's own SD-card encode of the same main stream, never touching the RTSP packetizer, the network, or our decode.

### CX810: same stream, two different timing outcomes

| source | frames | fps | dt min/med/max (ms) | frames <10 ms apart |
|---|---|---|---|---|
| camera FTP clip | 360-450 | 13.75-13.79 | 25 / 90 / 137 | **0** (every clip) |
| live RTSP | varies | apparent 8-13 | 2 / 15-46 / 737-2104 | up to 50% |

### Same-car verification (three matched events)

Each FTP clip was matched to the CamWatch pass inside its time window:

| event | camera FTP | camwatch RTSP |
|---|---|---|
| pass 13066 | 390 frames, 13.79 fps, 0/389 bursts | 19 frames, max gap 813 ms |
| pass 13067 | 390 frames, 13.79 fps, 0/389 bursts | 20 frames, max gap 737 ms |
| pass 13069 | 390 frames, 13.77 fps, 0/389 bursts | 24 frames, max gap 854 ms |

The camera can timestamp perfectly. It simply does not do so on the live RTSP stream.

---

## 7. Localizing the root cause: the three-camera comparison

With all three cameras online simultaneously (same `leochen4891` credentials), live RTSP and FTP clips were probed for each:

| camera | IP | stream type | live RTSP timing | frames <10 ms | max gap | FTP clip |
|---|---|---|---|---|---|---|
| E1 | 192.168.0.217 | 1440p H.264 | perfectly uniform 66.6 ms | 0% | 67 ms | clean |
| CX410W | 192.168.0.235 | 1440p H.264 | mildly scrambled | 4.4% (13/296, min 2.3 ms) | 156 ms | clean (0/149) |
| CX810 | 192.168.0.158 | 4K HEVC | heavily scrambled | 23% (51/223) | 875 ms | clean (0/449) |

This validates the historical narrative with hard numbers (E1 stable, CX410W introduced bursts, CX810 worse) and settles the causal question:

- The CX410W bursts at **the same stream spec where the E1 is perfect** (1440p H.264). Therefore resolution and codec are not the root cause.
- The flaw is **in the newer models' RTP packetizer firmware**, and the CX810's 4K HEVC load **amplifies** it (4.4% to 23%, 156 ms to 875 ms).
- Dropping the CX810 to 1440p would at best mitigate to CX410W levels, which is the level of corruption that originally caused phantom speeds in the CX410W era. It is not a cure.

(Side note: the E1's live stream, while perfectly timed, delivered packets unusually slowly during probing, roughly 13 packets in 22 s. Treat the E1 as a timing reference, not a viable producer.)

---

## 8. Impact on stored speeds

The fabrication corrupts the headline speed in both directions, depending on whether the pass window catches compressed clusters or stretched gaps:

- **Inflation (front-burst):** pass 13009, 8 frames in 29 ms, running average started at 558 mph and decayed to a stored **63.8 mph**; geometry puts the true speed near **37-41 mph**. The exit-descent guard missed it because the curve flattened to a 2.8% final step (under the 8% threshold) while still settled on an inflated value. This was a false alarm (threshold is 40).
- **Deflation (fake stalls):** an afternoon batch (passes 13098-13111) was dominated by stalls (`dt_cv` ~2). Stored speeds ran at roughly half the true value: pass 13108 stored **16.5 mph** vs geometry **34.9**; 13104 stored 17.3 vs 31.3. Deflated speeds trigger no guard and look unremarkable, so they pass silently.
- **Validation case:** pass 13105 was the one cleanly-timed pass in that batch (`dt_cv` 0.48); there the stored PTS headline (36.4) and geometry at 13.8 fps (37.2) agree within 0.8 mph. When the clock is honest the two methods converge, which is exactly the property a replacement method must have.

The guards (PR #2) remain useful as a stopgap for extreme inflation but are structurally unable to address this problem: they test shape heuristics on a corrupted clock and have no visibility into deflation.

---

## 9. The fix

### Decided direction

Compute speed from the reliable signal (geometry) against the measured cadence, ignoring RTSP timestamps:

```
speed = median(per-frame displacement) x measured_cadence_fps x 2.2369
```

- The median absorbs both detection jitter and genuinely dropped frames (a dropped frame doubles one step, which the median discards). Displacement-based drop detection (a step that is a clean multiple of the local median) can refine the time base if needed.
- `measured_cadence_fps` is **13.8** for the CX810 today, confirmed by three independent methods (burn-in counting, FTP container PTS, and consistency across 4+ clips). It must be a measured value, re-checked when camera settings change, never the nominal setting.
- The FTP recordings remain flowing and act as a continuous validation oracle: camera-native timestamps are exact, so any drift between the geometry method and FTP-derived speeds flags a cadence change.

### Open design questions (for the implementation discussion)

1. Estimator detail: median-step times cadence vs total-distance over (frame-count/cadence); behavior for accelerating or very slow vehicles (e.g. pass 13007 at ~5 mph over 90 frames).
2. Where the cadence constant lives (config value vs periodically measured) and how a camera-settings change invalidates it.
3. Validity in low light: auto-exposure can lengthen integration time and change the true cadence (out of scope for v1 daytime use, but should fail loudly, not silently).
4. Whether to recompute historical passes (the per-era cadence for E1/CX410W eras is unknown) or fix forward only.
5. Disposition of the shape guards once geometry-speed ships (retire vs keep as anomaly flags).
6. Whether to also store the PTS-based value for comparison, and how corrected speeds reach the hub.

### Alternatives considered and rejected

- **Fix our decode:** nothing to fix; decode is timestamp-transparent (section 3).
- **Use frame arrival time:** measured as bursty as PTS (libav delivery buffering); this was also the pre-E1-era method abandoned for the same reason.
- **Camera settings:** all ruled out (section 3, section 7).
- **Use FTP clip timestamps as the primary source:** exact but post-hoc and async; kept as the validation oracle instead of the hot path.

---

## 10. Side changes shipped during the investigation (June 3)

- **Hub upload switch:** new `upload.enabled` config + "Hub upload" toggle in the local Settings dialog, applied live to the uploader thread without a restart (uploader starts whenever creds exist but only POSTs while enabled). Currently **off** so corrupted-speed passes do not reach camwatch-web. Watermark: passes after **12980** were captured during the experiment window and will backfill on re-enable unless scrubbed first.
- **Chart fix:** guard-rejected passes wrote `v_homog_mph: NaN` into the trajectory manifest, which is invalid JSON and silently broke the entire speed chart in the browser (first seen on pass 12982). The manifest now writes `null`. Already-written manifests with `NaN` still need a one-shot backfill.
- Both changes are deployed on lei-ubuntu (branch `fix/speed-trustworthiness-guards`, uncommitted at time of writing).

---

## Appendix A: How to re-measure (recipes)

**Live RTSP packet timing (no CamWatch code involved):**
```
ffprobe -rtsp_transport tcp -i "rtsp://USER:PASS@CAMERA:554/h265Preview_01_main" \
  -select_streams v:0 -show_entries packet=pts_time -of csv=p=0
```
Compute consecutive deltas; healthy is a tight cluster at 1/fps, corrupted shows sub-10 ms clusters and/or multi-hundred-ms gaps.

**Camera-native clip timing:** run the same `ffprobe -show_entries packet=pts_time` on a clip from `/srv/nas/files/<camera>/YYYY/MM/DD/`.

**True cadence from burn-in:** decode a clip, crop the top-center OSD, downscale crop to ~80x8, mean-abs-diff consecutive frames, threshold ~0.6 for ticks, discard ticks within 3 frames of the clip edges, and count frames only between two visible ticks (fully-bounded seconds). Validate the detector against a manually-read clip before trusting an aggregate.

**Per-pass fabrication check (from `events/pass_N.jsonl`):** compute `step_cv` (per-frame displacement variation) and `dt_cv` (PTS interval variation). Steady motion (`step_cv` ~0.15) with `dt_cv` above ~0.6 means the timing is fabricated.

## Appendix B: Key reference numbers (June 3, 2026)

- CX810 true cadence: **~13.8 fps** (recording PTS 13.75-13.79; burn-in mode 14; nominal setting 15).
- CX810 recording dt quantization: ~45.5 ms units (4096 ticks at 90 kHz); observed dt values 25/45/91/137 ms.
- CX810 camera settings at time of measurement: Clear 3840x2160, 15 fps, 6144 kbps, H.265; Fluent 640x360, 10 fps, H.264; Frame Rate Mode Constant.
- Guards: implied-fps 35, arc ratio 1.4, exit descent 8%, magnitude ceiling 55 mph, min samples 5.
- Pipeline limit: yolo11l sustains ~15 fps on the 3060 Ti; 20-25 fps causes unbounded lag growth.
- Homography: scene-fit May 14 (`ead1228`), 4K, mean reprojection error 0.134 m.
