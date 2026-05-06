# camwatch

Local traffic-speed monitor. Pulls a live RTSP feed from a Reolink IP camera, detects passing cars with YOLO, estimates each car's speed using a two-line crossing method, and writes an alert when the speed exceeds a threshold (default 40 mph).

Two ways to run it:

- **Web UI** (recommended): `uv run python -m camwatch serve` starts an always-on capture worker plus a FastAPI/HTMX web app at `http://localhost:8000`. Browse passes, play overlay-rendered clips, set known speeds inline, batch-delete, configure threshold.
- **Headless**: `uv run python -m camwatch` runs the live alert pipeline and writes events.jsonl + alert JPGs.

Both modes write into the same SQLite database (`camwatch.db`).

---

## What it does

```
RTSP stream  →  YOLO detect  →  BotSORT track  →  two-line speed math  →  events.jsonl + JPG
```

For every car in the camera's view:
1. **Detect**: YOLO11n picks out cars/trucks/buses/motorcycles in each frame.
2. **Track**: BotSORT assigns a stable `track_id` so the same car is followed across frames.
3. **Time the crossing**: when a tracked car's bbox bottom-center crosses two pre-defined vertical screen lines, the elapsed time between crossings is recorded. Linear interpolation between adjacent frames cancels most of the frame-quantization noise.
4. **Convert to mph**: divide a calibrated real-world distance by the elapsed time. The distance is calibrated separately for each direction (see "Why per-direction" below).
5. **Log + alert**: every pass appends a JSON line to `events/events.jsonl`. Passes at or above the threshold also save a bbox-annotated JPG snapshot.

---

## How the speed math works

### The two-line crossing method

Pick two vertical lines on a still frame from the camera, line A on the left and line B on the right. Both lines are at fixed pixel x-coordinates. For each tracked car, watch when its ground point (the bottom-center of its bounding box, where the wheels meet the road) crosses each line:

```
                 line A           line B
                   │                │
   ───────────────────────────────────────────  far lane (N→S)
   ───────────────────────────────────────────  near lane (S→N)
                   │                │
                   x_A              x_B
                   t_A              t_B   ← interpolated crossing times
```

Then `speed = distance_between_lines / |t_B − t_A|`.

The crossing times are interpolated linearly between adjacent frames so timing precision isn't quantized to the camera's frame interval. At ~10 fps that interpolation typically gives ~10 ms of timing precision per crossing, about ±0.4 mph at 30 mph and ±1.6 mph at 60 mph. Frame rate is **not** the practical bottleneck at residential speeds.

### Why per-direction calibration

The two lanes of the road are at different distances from the camera. A vertical line in screen space (constant x in pixels) hits the near lane and the far lane at different real-world points, so the actual road distance between line A and line B is different for each direction:

```
camera ─┐
        │
        ▼
═════════════════  far lane (N→S)   ← distance ≈  9 m between lines
═════════════════  near lane (S→N)  ← distance ≈ 11 m between lines
```

Ignoring this would produce a 10-25% per-direction error. So calibration stores two distances:

```yaml
line_distance_m_north: 11.5    # used when a car crosses A then B (left→right)
line_distance_m_south:  9.1    # used when a car crosses B then A (right→left)
```

The live system picks the right distance based on which line was crossed first.

### How calibration finds those distances

Mathematically, `distance = speed × time`. So if you drive past at a known GPS-confirmed speed and the system measures elapsed time between line A and line B, the implied real-world distance is just `(known_mph in m/s) × elapsed_s`.

You drive multiple passes per direction, each at a known GPS speed. The tool records elapsed times automatically. You then label each pass with the speed you were driving (skipping any other cars that happened to pass during the recording window). The averaged implied distance per direction goes into `calibration.yaml`.

A typical calibration plan: 20, 30, 40 mph in each direction, two passes each = 12 passes total, ~10 minutes.

---

## Architecture

Two pipelines share most of the building blocks:

**Headless** (`python -m camwatch`):
```
RTSP → detect → speed (crossing) → sink (events.jsonl + alert JPGs)
```

**Web UI** (`python -m camwatch serve`):
```
RTSP → detect → crossing → recorder (mp4 + thumb) → SQLite (passes)
                       └→ preview buffer → FastAPI → browser
```

Modules:

```
camwatch/
├── capture.py         RTSP frame source with reconnect. Yields (frame, pts_ts).
├── detect.py          YOLO + BotSORT wrapper. Returns Track records per frame.
├── crossing.py        Two-line crossing state machine (web UI source of truth).
├── speed.py           Parallel crossing detector retained for headless mode.
├── sink.py            Headless: append events.jsonl, save annotated JPG on alert.
├── main.py            Headless wiring; handles SIGINT.
├── recorder.py        Rolling ring buffer; on trigger writes a .mp4 (+ overlay) and a .jpg thumb.
├── preview.py         Latest-frame MJPEG buffer for the live preview.
├── db.py              SQLite (WAL) pass storage; single writer + many readers.
├── capture_worker.py  Web UI: long-running thread driving detect → crossing → recorder → DB.
├── thumb_upgrader.py  Background main-stream thumbnail upgrader, OSD-time-keyed.
├── ts_reader.py       OCRs the camera's burned-in OSD timestamp (Tesseract).
├── server.py          FastAPI app + Jinja2/HTMX rendering.
├── calibrate.py       Interactive calibration CLI (subcommands below).
├── config.py          Loads config.yaml + .env into a typed dataclass.
└── __main__.py        Dispatches `python -m camwatch [serve]`.
```

### `capture.py`
- `cv2.VideoCapture(url, CAP_FFMPEG)` with `CAP_PROP_BUFFERSIZE=1` to keep latency low.
- Reconnect loop after consecutive read failures.
- Yields `Frame(image, ts, seq)` where `ts` is the frame's **camera-side PTS** (`cv2.CAP_PROP_POS_MSEC`) re-anchored to `time.monotonic()` at startup. PTS is immune to ffmpeg's RTSP buffer-burst delivery, which we confirmed compresses `time.monotonic()` intervals badly enough to over-state speeds by 3× during traffic spikes. See `scripts/timing_probe.py` for the diagnostic that established this.

### `detect.py`
- Wraps `ultralytics.YOLO('yolo11n.pt').track(...)`.
- Filters to COCO classes `[2 car, 3 motorcycle, 5 bus, 7 truck]`.
- Tracker: BotSORT (Ultralytics built-in, default).
- Runs on Apple Silicon MPS by default.
- Returns `Track(track_id, cls_idx, cls_name, bbox, conf, ground_point)` where `ground_point = ((x1+x2)/2, y2)`. That's the bbox bottom-center, the most stable feature for road-crossing geometry.

### `speed.py`
- Per `track_id`, holds the most recent `(t, x)` sample.
- On each new sample, checks whether x has crossed line A or line B since the previous sample. A crossing is detected by sign change in `(x_prev − line_x) × (x_curr − line_x)`.
- Linear interpolation between the two adjacent samples gives the precise crossing time.
- When both A and B have been crossed, emits a `SpeedEvent` and clears state for that track.
- Stale tracks (no update for `max_track_age_s`, default 5s) are GC'd silently. They didn't complete a full crossing.

### `sink.py`
- Appends one JSON line per `SpeedEvent`:
  ```json
  {"ts":"2026-05-04T21:45:12-04:00","track_id":47,"class":"car","direction":"N","speed_mph":43.2,"alert":true,"snapshot":"events/2026-05-04T21-45-12_id47_N_43mph.jpg"}
  ```
- For events at or above the threshold, saves a JPG with the bbox and label drawn on, named with timestamp + track_id + direction + integer mph.

---

## Prerequisites

- macOS, Apple Silicon (the project runs on MPS for inference)
- Reolink camera with **RTSP enabled** (Reolink mobile app → Settings → Network → Advanced → Port Settings → enable RTSP)
- Homebrew packages: `brew install uv ffmpeg`

---

## Setup

```sh
git clone git@github.com:leochen4891/camwatch.git
cd camwatch
uv venv --python 3.12
uv sync

cp .env.example .env                        # edit: REOLINK_USER, REOLINK_PASS
cp config/config.example.yaml config/config.yaml
cp config/calibration.example.yaml config/calibration.yaml
```

Quick smoke test against the camera:

```sh
uv run python scripts/test_stream.py
```

Expected: ~10-25 fps printed and one frame written to `/tmp/camwatch_test.jpg`.

---

## Calibration walkthrough

Five subcommands, run in order. The first three involve you doing something physical; the last two are pure analysis.

### 1. `pick-lines`: pick the two screen lines

```sh
uv run python -m camwatch.calibrate pick-lines
```

Opens a live still from the camera in a window. **Click line A (left), then line B (right).** Press `s` to save, `r` to reset, `q` to quit. The pixel x-coordinates of A and B (mapped back to the source resolution) get written to `config/calibration.yaml`.

Pick lines that are clearly perpendicular to the road and far enough apart that even a fast-moving car spends a noticeable fraction of a second between them. ~30-60% of the frame width is a good target.

### 1b. `pick-roi`: limit YOLO to the road belt (optional but recommended)

```sh
uv run python -m camwatch.calibrate pick-roi
```

Opens the same still and lets you drag a rectangle around just the road. **Click the top-left corner, then the bottom-right corner of the road belt.** The two crossing lines from `pick-lines` are drawn on the frame so you can verify they fall inside the rectangle.

The ROI is the only region YOLO sees; lawn / sky / driveways outside it cannot generate detections. Two benefits:
- Faster inference (often 2-3x): YOLO processes a smaller image.
- Fewer false positives: parked cars in driveways stop appearing as tracks.

The full frame is still saved in clips and shown in any future security view; ROI only affects what the detector looks at. Saved as `roi_x1/y1/x2/y2` in `calibration.yaml`. Leave it un-set (all zeros) to feed the full frame to YOLO.

### 2. `capture`: record calibration drives

```sh
uv run python -m camwatch.calibrate capture --secs 600
```

Watches the live stream for 600 seconds. For every car that crosses both lines, the tool:
1. Records a short mp4 clip into `recordings/cal_<timestamp>_id<id>_<dir>.mp4` covering `clip.margin_s` seconds before the first line crossing through the same after the last. With the default sub stream (640×480) no downscaling is needed; from a higher-res source the recorder caps the saved frame width to keep file size sane. The clip is rendered with a debugging overlay:
   - both vertical lines (line A in green, line B in blue), drawn dim before the focus car has crossed them and bright after, with the crossing timestamp labelled next to each
   - the focus car's bounding box in red, with `id=N` label, plus a red dot at its `ground_point` (bbox bottom-center, the feature the speed math anchors to)
   - any other detected cars in the frame in gray (faint bbox + small dot), so you can see when the tracker confused two cars
   - a top header strip with the focus track ID and the total span (`B - A`)
   - a bottom strip with `t = +X.XXXs (relative to line A)`, so you can scrub the clip and see exactly when each crossing fires
2. Prints a line including the wall-clock time and the clip name.
3. Appends a `passes:` entry to `calibration.yaml`.

```
[06:59:52] pass: id=47 car N elapsed=0.823s -> cal_20260505T065952_id47_N.mp4
[06:59:54] pass: id=48 car S elapsed=1.105s -> cal_20260505T065954_id48_S.mp4
```

While this is running, **drive past at GPS-confirmed speeds** in both directions. A reasonable plan, ~10 min:

| Direction | Speeds | Passes per speed |
|---|---|---|
| S → N | 20, 30, 40 mph | 2 |
| N → S | 20, 30, 40 mph | 2 |

Tips:
- Use a phone GPS speedometer (Waze, Google Maps, etc.). Far more accurate than a car speedometer.
- Hold a steady speed from before line A all the way past line B. Don't accelerate or decelerate during the crossing window.
- Stay in a consistent lane position across passes. Don't hug the curb on one pass and the centerline on the next.
- Stagger your passes by ~15 seconds so they don't overlap with each other or with random traffic.
- Keep a paper note: `21:46 S→N 20 mph`, `21:48 N→S 20 mph`, etc. You'll match these against `track_id`s in the next step.

The tool can't tell which passes are you and which are random other cars on the road; they all get recorded with `known_mph: null`.

### 3. `annotate`: label each pass

```sh
uv run python -m camwatch.calibrate annotate
```

Walks through every unannotated pass, one at a time. Each prompt shows the elapsed time, the wall-clock timestamp, and the clip path:

```
[1/14] track_id=47 dir=N elapsed=1.214s captured_at=2026-05-04T21:46:12-04:00
  clip: recordings/cal_20260504T214612_id47_N.mp4
  > 20
[2/14] track_id=48 dir=S elapsed=1.105s captured_at=2026-05-04T21:46:54-04:00
  clip: recordings/cal_20260504T214654_id48_S.mp4
  > open
  > skip
```

At each prompt:
- Type a number to record the GPS speed in mph.
- Type `open` to play the clip in the default video player, then re-prompt.
- Type `skip` (or `s`) to discard the pass (not your car, or a tracker artifact).
- Type `q` to stop annotating now; already-typed answers are saved.

The clip preview makes it easy to filter out tracker re-ID artifacts (a single car split into two `track_id`s by BotSORT mid-crossing). Those show up as suspiciously short elapsed times (e.g. 0.3 sec across a span that should take 0.8 sec).

### 4. `compute`: average the implied distances

```sh
uv run python -m camwatch.calibrate compute
```

For each annotated pass, computes `implied_distance_m = (known_mph in m/s) × elapsed_s`. Averages those per direction and writes:

```yaml
line_distance_m_north: 11.483
line_distance_m_south:  9.142
```

### 5. `report`: sanity check

```sh
uv run python -m camwatch.calibrate report
```

Re-runs each annotated pass through the same speed math the live system will use. You should see something like:

```
calibrated distances: N=11.483m  S=9.142m

idx  dir  elapsed   known    pred    err
  1    N    1.214    20.0    21.2   +1.2
  2    N    0.821    30.0    31.4   +1.4
  3    N    0.621    40.0    41.5   +1.5
  4    S    1.105    20.0    18.5   -1.5
  5    S    0.731    30.0    28.0   -2.0
  6    S    0.547    40.0    37.4   -2.6
```

Healthy result: errors within ±2-3 mph and roughly random in sign. If errors are systematically biased one way (e.g. all N too high, all S too low), revisit the drives. Usually that's lane-position drift.

---

## Run

### Web UI (recommended)

```sh
uv run python -m camwatch serve            # bind 127.0.0.1:8000
uv run python -m camwatch serve --host 0.0.0.0 --port 8000   # LAN/phone access
uv run python -m camwatch serve --profile  # log per-stage capture-loop timings every 30s
```

Open `http://localhost:8000`. The page shows:

- **Status panel**: live capture indicator, current per-direction calibration, # known points, alert threshold (editable inline).
- **Filters**: by direction, alerts-only.
- **Filters**: by direction, alerts-only, time range (from/to), and speed bucket via the histogram.
- **Speed histogram**: 5-mph buckets summarising the filtered set. Click a bar to narrow the list to that bucket.
- **Pass list**: each row has a thumbnail, timestamp, direction, speed (blank until calibration is set), inline ▶ play, ✕ delete (revertable), and a `⋯` menu for "Set known mph". Click anywhere on the row to toggle inline playback.
- **Settings dialog**: alarm threshold, live-preview lines toggle, clip margin (pre/post-roll seconds), clip capture speed range (passes outside still get a thumbnail but no .mp4), and storage retention in days.

Setting a known mph automatically recomputes per-direction calibration (writes `line_distance_m_north` / `line_distance_m_south` into `config/calibration.yaml`). All other passes in that direction immediately re-display computed mph on next render.

### Headless

```sh
uv run python -m camwatch
```

Foreground process, logs to stderr, appends events to `events/events.jsonl`, saves snapshots in `events/` for alerts. Ctrl-C to stop.

### Remote access (outside your LAN)

The web UI binds to `0.0.0.0:8000` by default, so it's reachable on any network interface the host has. To reach it from your phone away from home, you have two reasonable options:

- **Tailscale** (recommended for personal use). Install Tailscale on the host Mac and on your phone/laptop. Both devices show up on a private mesh network with `*.ts.net` hostnames; the camwatch UI is reachable at `http://<host>.ts.net:8000` from any logged-in device. Nothing is exposed to the public internet, no port forwarding, no DNS to manage. Free for personal use.

- **Cloudflare Tunnel** (use if you want a public URL). Run `cloudflared` on the host; it opens an outbound connection to Cloudflare's edge and routes a hostname like `camwatch.yourdomain.com` to `localhost:8000`. No inbound port opened on your router; HTTPS terminates at Cloudflare's edge. **The camwatch UI has no auth**, so you should layer Cloudflare Access (free, email/Google login) on top before exposing it publicly.

For a single-user "check it from my phone" use case, Tailscale is simpler and more private. Use Cloudflare Tunnel only when you need to share a URL with someone whose device you don't control.

---

## Configuration

All defaults live in `config/config.yaml`:

```yaml
camera:
  host: 192.168.0.227
  port: 554
  path: /h264Preview_01_sub        # 640x480 sub stream — recommended for detection + clips.
                                   # _main is full-res but pushes CPU and adds latency.

model:
  weights: yolo11n.pt              # nano. Bump to yolo11s.pt for more accuracy.
  device: mps                      # mps / cuda / cpu
  conf: 0.35                       # daytime default
  iou: 0.5
  classes: [2, 3, 5, 7]            # COCO: car, motorcycle, bus, truck

alert:
  threshold_mph: 40

paths:
  events_dir: events
  calibration: config/calibration.yaml

speed:
  max_track_age_s: 5.0

retention:
  days: 7                          # auto-delete passes older than this. 0 disables.

clip:
  margin_s: 0.5                    # pre/post-roll seconds around each crossing
  capture_min_mph: 0               # passes outside [min, max] are logged with a thumb only
  capture_max_mph: 999
```

The `retention.*`, `clip.*`, and `alert.threshold_mph` keys are also editable in the in-app Settings dialog; saving there persists back to `config.yaml`.

Camera credentials live in `.env`:

```
REOLINK_USER=...
REOLINK_PASS=...
```

---

## Project layout

```
camwatch/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example                   ← .env (gitignored) holds the real creds
├── .gitignore
├── config/
│   ├── config.example.yaml        ← config.yaml (gitignored) is the live one
│   └── calibration.example.yaml   ← calibration.yaml (gitignored) is the live one
├── camwatch/
│   ├── __init__.py
│   ├── __main__.py                # entrypoint: dispatches headless vs `serve`
│   ├── config.py                  # YAML + .env loader, typed dataclasses
│   ├── capture.py                 # RTSP frame source with reconnect
│   ├── detect.py                  # YOLO + BotSORT wrapper
│   ├── crossing.py                # Two-line crossing state machine (web UI source of truth)
│   ├── speed.py                   # Crossing detector for headless mode (parallel to crossing.py)
│   ├── sink.py                    # Headless: events.jsonl + alert JPGs
│   ├── main.py                    # Headless wiring (capture → detect → speed → sink)
│   ├── calibrate.py               # Interactive calibration CLI
│   ├── recorder.py                # Rolling clip recorder + thumbnail writer
│   ├── preview.py                 # Live preview MJPEG buffer
│   ├── db.py                      # SQLite (WAL) pass storage
│   ├── capture_worker.py          # Always-on capture thread for the web UI
│   ├── server.py                  # FastAPI app + render helpers
│   ├── static/style.css
│   └── templates/                 # Jinja2 partials for the HTMX frontend
│       ├── index.html
│       ├── _status.html
│       ├── _pass_list.html
│       ├── _pass_row.html
│       └── _histogram.html
├── scripts/
│   ├── test_stream.py             # quick RTSP smoke test
│   ├── regen_thumbs.py            # rebuild thumbnail JPEGs from existing clips
│   ├── timing_probe.py            # compare monotonic vs PTS vs OSD ticks (timing diagnostic)
│   └── verify_speed.py            # verify a stored pass's speed using OSD ticks in its clip
├── camwatch.db                    # gitignored: SQLite db (created on first run)
├── events/                        # gitignored: events.jsonl + alert snapshots (headless mode)
├── recordings/                    # gitignored: clip mp4s + thumbnail jpgs
└── models/                        # gitignored: yolo11n.pt downloaded by ultralytics
```

---

## Scope and limitations

- **Daytime only.** Night and dawn/dusk are out of scope. Motion blur in low light makes the bbox bottom-center jitter by enough to introduce ±3-8 mph noise per pass, and the IR-cut filter transition at dawn/dusk drops frames and breaks tracker IDs.
- **Single line pair, two lanes.** v1 uses one calibration distance per direction. Cars that drift across the lane width during a single crossing window will measure incorrectly. Rare in residential traffic.
- **No retention policy.** `events.jsonl` and the snapshot folder grow forever. Clean up by hand for now.
- **Camera must not move.** Calibration is tied to the specific pixel positions of line A and line B. If the camera is bumped or repositioned, redo `pick-lines` and the drives.

## v2 ideas

- Email alerts (SMTP / Gmail app password)
- launchd daemon for unattended operation
- Per-lane (not just per-direction) calibration
- Persist filter/pagination state across reloads
- SSE-based pass list updates (push instead of 10s poll)
