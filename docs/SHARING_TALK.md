# Camwatch: 3 Days, AI-Paired

A personal residential speed camera, built end-to-end by pair-programming with Claude Code. Below: how it was built, where it broke, and what an AI partner *changed* about the development experience.

---

## The Need

> "Can you find a video stream from an IP camera in the same LAN?"
> *— first message of the session, May 4*

I had a Reolink E1 Outdoor camera and a residential street with traffic that sometimes felt too fast. I wanted:

- A live speed detector for cars in front of my house
- A web UI to review passes (thumbnail, speed, video clip)
- Email alerts above a threshold (deferred)

No prior code. No CV/ML expertise. Just a camera, a MacBook Air, and Claude Code.

---

## The Outcome (3 days, 53 commits)

```
camwatch/
├── camwatch/             core service: capture → YOLO+BotSORT → crossing → DB → UI
├── scripts/              8 diagnostic / bootstrap tools
├── templates/            digit reference images for OSD OCR (per stream)
├── config/               YAML config + per-direction calibration
└── README.md             ~500 lines of docs
```

- **Day 1 (May 4)**: 1 commit — initial scaffold (RTSP, YOLO, two-line crossing math)
- **Day 2 (May 5)**: 32 commits — recording, web UI, dual stream, calibration drives, full feature set
- **Day 3 (May 6)**: 20 commits — the rabbit hole

---

## Day 2 morning: from zero to a working system

Within a few hours of starting:

```python
# Linear interpolation between adjacent samples gives the precise crossing time.
def _interp_cross(xp, x, tp, t, line_x):
    if x == xp: return t
    return tp + (line_x - xp) / (x - xp) * (t - tp)
```

**Calibration approach** I would never have arrived at alone: instead of measuring the road, drive the car at GPS-known speeds in each direction and let the system back-fit the line distance. Solves the perspective + camera-angle problem in one stroke.

**My role here**: domain framing ("two-lane residential street, north and south are at slightly different distances from the camera"). **Claude's role**: turning that framing into a per-direction calibration system in one shot.

---

## Day 2 evening: the bug that defined Day 3

A car passed. The system logged **71 mph**. I knew the car was doing maybe 25.

```
[17:01] pass: id=4 car S elapsed=0.268s -> cal_20260505T072942_id4_S.mp4
```

> "I'm 100% sure the car wasn't going 53mph, 35mph at most. Also the driveway where I marked the two lines are 21ft wide. How is the time measured?"

**Hypothesis I formed (later proven correct)**: ffmpeg's RTSP buffer occasionally bursts, so consecutive `cap.read()` returns get `time.monotonic()` values much closer than the camera actually captured them. That compresses elapsed_s, inflates speed.

**This was a turning point**. Until this moment, AI had been writing code from my requirements. Here, I was the one with the *empirical hypothesis*.

---

## The forensic investigation

Frame-by-frame analysis of the recorded clip:

1. Last frame showing OSD `:24` was at overlay-time `+0.169s`
2. First frame showing OSD `:25` was at overlay-time `+0.198s`
3. Last frame showing OSD `:25` was at overlay-time `+0.772s`

Math: 0.574s of overlay-time spans up to 1.0s of real wallclock → the recorder's monotonic timestamps were running at ~57% of real time. **Speed was 1.74× over-stated.**

Independent verification — a satellite-view distance measurement:

> "in the 25s, it moved from [position A] to [position B], it roughly translate to about 29ft from the map. what's the speed for 29ft in 1s?"

29 ft / 1.0 s = ~20 mph. The 71 mph reading was actually ~20-25 mph. **Confirmed, by measuring the same event two completely independent ways.**

---

## The fix: PTS instead of monotonic

```python
# Before:
ts = time.monotonic()  # vulnerable to ffmpeg burst delivery

# After:
pts_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
if pts_s > 0:
    if pts_offset is None:
        pts_offset = time.monotonic() - pts_s  # anchor once
    ts = pts_s + pts_offset
```

Camera-side PTS is stamped before the frame leaves the encoder; immune to ffmpeg's delivery jitter. **Diagnostic confirmed**: monotonic intervals had std=48ms; PTS std=0.0ms.

---

## Day 3: the thumbnail upgrade rabbit hole

The sub-stream thumbnails (640×480) were grainy. Idea: also pull a high-res frame from the main stream when a pass triggers, swap the thumbnail.

This took **all of Day 3**. Below is the cascade of issues we hit and how each got resolved.

---

## Issue 1: streams aren't aligned frame-by-frame

The main and sub RTSP sessions each have their own PTS counter starting at zero on session open. Same camera-instant produces *different* `*_ts` values in each stream.

**First attempt** (pure PTS lookup): `find_frame_at(trigger.sub_ts)` in main buffer.
**Result**: matched frames were ~5-7 seconds off. The PTS spaces are anchored at different camera-instants because each session's first frame was captured at different times.

**Fix**: measure the constant cross-stream offset once via OCR on each stream's burned-in OSD timestamp:

```
drift_main = main_ts(F) - wallclock_unix(F)    # OCR'd main frame
drift_sub  = sub_ts(F') - wallclock_unix(F')   # OCR'd sub frame
cross_stream_offset = drift_main - drift_sub
target_main_ts = trigger.target_ts + cross_stream_offset
```

After the offset is learned, every subsequent lookup is pure PTS arithmetic.

---

## Issue 2: Tesseract is unreliable on this OSD

OCR on a small fixed-font OSD ("05/06/2026 13:15:56 WED"):

- 8 ↔ 6 (top loop dropped)
- 0 ↔ O (whitelist trains it to expect letters near digits)
- 5 ↔ 3, 0 ↔ 9, etc.

A single misread caused `cross_stream_offset = +7205.393s` (exactly 2 hours off — Tesseract read `08:23:27` as `06:23:12`). That poison value cached, and every subsequent lookup tried to match a frame ~2 hours in the future.

**Fix path** (multiple attempts):
1. Substitute confusable letters → digits (`O → 0`, `I → 1`, `S → 5`, `B → 8`) before regex parsing — recovered ~70%.
2. Sanity-check OCR'd datetime against `datetime.now()` ±60s — rejects bad readings.
3. Build a **template-matching reader** with per-digit reference images. Sub-millisecond, deterministic for this font. Failure rate: 0%.

---

## Bootstrapping the template library — a key human moment

I needed clean digit images (0-9). The first scan-based collection caught only some digits (the seconds-ones cycles through 0-9, but at 1.0s capture intervals we hit even-numbered seconds twice and odd-numbered ones once due to lock-step).

> "let's get back to the drawing board and try to understand the gap"
> "I have been watching the main stream on a different device and there is not much skipped frames if at all. I'm pretty sure the main stream is stable enough to extract the image"
> "Can we add a framecounter on both sub and main stream as early as possible?"

**That third quote was decisive.** Up to that point, AI had been theorizing about *where* frames went. Once I asked for a measurement at the lowest level, the answer became visible:

```
stream sub:  raw read fps=14.73, pts advance=0.98x  ← perfectly fine
stream main: raw read fps=3.48,  pts advance=1.76x  ← problem
```

Sub gets 15 fps clean; main gets 3.5 fps and is draining a backlog (`pts advance > 1`).

---

## Issue 3: where did the main-stream frames go?

Probe: time `cap.read()` per call on main:

| resolution | median | p95 |
|---|---|---|
| 2560×1920 | 197ms | 2.48s |
| 2048×1536 | 86ms | 1.44s |
| Same, under live load | ~430ms | (unknown) |

The same call that takes 86ms in isolation takes 430ms when sub-stream YOLO + tracker + recorder are running concurrently. **CPU + GIL contention on H.264 software decode of large frames.**

When `cap.read()` falls behind the camera's frame rate, ffmpeg accumulates a backlog and eventually dumps it in a burst. We recovered "only" ~20 of ~140 frames in such a burst because of:

---

## Issue 4: single-slot reader buffer

The original `RtspStream` design used a "keep latest, drop older" pattern between the inner reader thread and the consumer. Right for live (latency wins), wrong for the upgrader (every frame matters).

**Fix**: bounded `deque(maxlen=N)`:
- `queue_size=1` → identical to single-slot (live capture)
- `queue_size=200` → bounded FIFO that absorbs bursts (upgrader)

This stopped frame loss in our pipeline. The remaining gaps were the camera/ffmpeg I-frame interval — a recovery wait that loses up to one full GOP — which on a Reolink E1 default is ~15 seconds.

---

## Issue 5: camera-side encode settings

Once we proved the bottleneck was decode time, lowering camera settings was the obvious win:

| setting | from | to | reasoning |
|---|---|---|---|
| Resolution | 2560×1920 | 2048×1536 | fewer pixels per decode |
| Aspect | 4:3 | 4:3 | match sub stream for accurate bbox projection |
| Frame rate | 15 fps | 4 fps | thumbnail doesn't need 15fps |
| I-frame interval | 15s default | shortest UI allows | reduces recovery gap from 15s to 1s |

End state: ~70-80% of passes successfully upgrade to a high-res thumbnail. The remaining failures are documented and the sub-stream thumbnail stays in place when upgrade fails — no data lost.

---

## Where AI accelerated me dramatically

**Code volume**: 53 commits across 3 days for a single hobbyist would normally be a 3-week project. Most commits are non-trivial (~100 LOC of focused work each).

**Cross-domain knowledge**: I never had to learn — only direct.
- *YOLO + BotSORT integration*: AI wrote the wrapper.
- *RTSP/H.264/PTS internals*: AI explained the layers, including which OpenCV property exposes PTS.
- *OpenCV connected components, NCC matching*: AI drafted the slot-detection and template-matcher code.
- *FastAPI + Jinja2 + HTMX*: zero learning curve on my side.
- *ONVIF SOAP requests*: AI used `onvif-zeep` directly, parsed responses.

**Diagnostic infrastructure**: I would not have built `timing_probe.py`, `verify_speed.py`, `probe_main_gaps.py`, `probe_main_iter_cost.py`, `collect_digit_templates.py`, `scan_digit_seconds.py` on my own. AI made each one in ~50-200 lines exactly when needed for the next debugging step.

**Subagent research**: when I asked about PyAV / RTCP / ONVIF as alternative architectures, AI spawned two parallel research agents that returned with concrete library recommendations and gotchas (Reolink ONVIF metadata stream not implemented; PyAV doesn't expose `start_time_realtime`; GStreamer is the canonical RTCP path). 30 minutes of research that would have taken me a weekend.

---

## Where the human played the decisive role

These are moments where the AI alone would have gone in the wrong direction:

1. **Domain knowledge as ground truth** — *"I'm 100% sure the car wasn't going 53 mph"*. AI accepts the recorded number as fact; a human knows when reality and the system disagree.

2. **Empirical hypothesis** — *"the framerate may fluctuate from time to time which results in significant inaccuracy"*. AI would have checked the math first; human noticed the visual stuttering on the live preview and made the right guess about timing.

3. **Cross-verification with external evidence** — *"in the 25s, it moved from X to Y, that's 29 ft from the map"*. Independent confirmation via satellite imagery, completely outside the system. AI doesn't volunteer this kind of triangulation.

4. **Challenging AI's conclusions** — *"Are you sure the frames from main stream were skipped?"*. I had told the AI the gap was upstream; the human pushed back and we found out the issue was actually inside our pipeline.

5. **Calling for the right diagnostic** — *"Can we add a frame counter on both sub and main stream as early as possible?"*. AI was theorizing; human asked for measurement at the lowest possible level. That was the breakthrough.

6. **Knowing when good enough is good enough** — *"thumbnail upgrade is like a nice-to-have feature. I will accept the current status."*. AI will keep optimizing forever; human draws the line.

---

## Lessons for AI-paired product development

1. **AI is a force multiplier on volume and breadth, not on judgment.** The product shipped because the human stayed in the seat for *what to build* and *when to stop*.

2. **Diagnostic tools are first-class deliverables**. The 8 probe / scan / collect scripts were essential to making the project tractable, and AI built them on demand without protest. None of them shipped to production but all of them existed when needed.

3. **Empirical measurement beats theoretical reasoning, every time.** When AI theorizes and human measures, the correct answer arrives faster. The opposite (AI measures, human theorizes) didn't happen in this project.

4. **Documentation has to be paired with the work.** README updates landed in the same commits as feature changes. Three days later, the doc reflects current state, not stale plans.

5. **The "one more layer of debugging" reflex is dangerous in both directions.** AI will go arbitrarily deep without prompting. Human can either let it (cost: time) or call it (cost: a known floor in production).

---

## Closing

> "back to the drawing board"

That phrase, used twice in 3 days, was the single most productive prompt of the project.

The product works. There's a UI on Tailscale and a Cloudflare tunnel queued for a future session. The code is on GitHub. The README is comprehensive. None of it would exist as it does without AI in the loop — and none of it would have shipped without a human ready to push back when the AI was confidently wrong.
