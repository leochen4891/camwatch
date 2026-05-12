# Migration to NVIDIA GPU (3060 Ti / Ubuntu)

Moving camwatch off the MacBook Air (Apple Silicon + MPS + VideoToolbox) onto
the Ubuntu desktop with an NVIDIA RTX 3060 Ti at `192.168.0.137`
(SSH user `lchen`, passwordless from the Mac).

## The architectural win: drop the sub stream

The single biggest reason to migrate is **not** raw speed; it's that the GPU
headroom lets us run live YOLO directly on the **main** stream and retire the
sub stream entirely.

Today's pipeline:

```
sub stream  (640x480, ~15fps)  -> YOLO+BoT-SORT   (live trigger path)
main stream (2048x1536, ~20fps) -> TimestampedFrameBuffer
                                -> thumb_upgrader picks frame at trigger ts
                                -> high-res clip + thumbnail
```

Sub and main are two independent RTSP sessions with independent PTS anchors;
keeping them aligned is the reason `TimestampedFrameBuffer`, the
PTS-anchored-to-monotonic logic in `RtspStream.frames()`, and the
queue-size=200 main-stream backlog buffer all exist.

Target pipeline on the 3060 Ti:

```
main stream (2048x1536, ~20fps) -> YOLO+BoT-SORT     (live trigger path)
                                -> screenshot of current frame = thumbnail
                                -> clip recorder uses same frames already on hand
```

Single stream in, single time domain, no cross-stream sync. The
`TimestampedFrameBuffer`, the thumb upgrader, and a fair amount of the
"epoch / PTS offset" complexity in `capture.py` can go away. The thumbnail
becomes whatever main-stream frame was current at the trigger moment, with no
PTS hunting needed.

### Why this is safe on the 3060 Ti

Benchmark on a real 2048x1536 main-stream JPEG (`calibration_main_frame.jpg`),
yolo11n weights, classes `[2,3,5,7]`, conf=0.35 iou=0.5, default ultralytics
imgsz (resizes internally to 640):

| Path                                | Mode    | fps   | ms/iter | VRAM   |
|-------------------------------------|---------|-------|---------|--------|
| Live (`Detector.track`, BoT-SORT)   | track   | 45.2  | 22.1    | 65 MiB |
| Thumb path (`Detector.detect`)      | predict | 171.1 | 5.8     | 65 MiB |
| Predict + FP16                      | predict | 163.8 | 6.1     | 48 MiB |

Main stream nominal rate is ~20 fps -> live track has ~2x headroom. Tracker
mode is CPU-bound by BoT-SORT's Hungarian step (`lap`), not by the GPU. VRAM
use is ~1% of the 8 GB card, so there's room to raise `imgsz` to 960 or 1280
and let the model see more pixels before downscale if accuracy ever needs it.

## Known camera constraint

Reolink E1 main stream is the contended slot. With the MacBook still pulling
main and the Ubuntu box also pulling main, both clients dropped to ~1.8 fps
in testing. Sub stream tolerates 3 concurrent clients at full ~16 fps.

Implication for cutover: the Ubuntu box can warm up its **sub**-stream code
path while the Mac is still serving live, but the moment we move live to the
main stream, the Mac side must stop first.

## Code changes (small)

1. `config/config.yaml`: `model.device: mps` -> `cuda` (or `0`).
2. `camwatch/capture.py`: `_VT_HWACCEL = HWAccel(device_type="videotoolbox", ...)`
   -> pick by platform (`darwin` -> videotoolbox, else `cuda`), or read from
   config / env var.
3. `config/config.yaml`: `camera.path` -> `/h264Preview_01_main` (was
   `_sub`). Keep `path_thumb` or repurpose it.
4. Tear down the dual-stream machinery once the new path is verified:
   - `capture.py`: remove `TimestampedFrameBuffer` and the
     `queue_size=200` backlog mode; the live deque stays at `maxlen=1`.
   - `thumb_upgrader.py`: replace its high-res-frame lookup with "save the
     live frame from the trigger event."
   - `capture_worker.py`: drop the calls into the thumb upgrader's
     `find_frame_at(ts, ...)` and feed it the live frame directly.

Recommendation: do the platform switch and stream switch in one PR, leave
the dual-stream code in place but unused for one release in case we need to
roll back, then delete in a follow-up.

## Cutover sequence

The actual stop/start cutover is short; the prep is most of the work.

### Prep (no service impact, Mac stays up)

1. SSH into `lchen@192.168.0.137`. There's already a probe venv at
   `~/camwatch-probe/` (PyAV 17, torch 2.5.1+cu121, ultralytics 8.4.48, lap).
   You can reuse it or scrap.
2. `git clone` (or rsync from Mac) the repo to e.g. `~/camwatch`.
3. Install deps with `uv venv` + `uv pip install -e .`. Pin torch wheels
   from `https://download.pytorch.org/whl/cu121` (driver 535.288 caps us at
   CUDA 12.2 runtime; cu121 wheels are the right target).
4. Apply the code changes above. Smoke test still on **sub** so you don't
   disturb the Mac's main-stream pull yet:
   - Temporarily set `camera.path: /h264Preview_01_sub` on the Ubuntu side.
   - `pause_at_night: true` to avoid weird behavior outside test hours.
   - Don't start the thumb upgrader yet.
   - Run `python -m camwatch serve --host 127.0.0.1` (no tunnel) and watch
     `prof.record("yolo_track", ...)` numbers + raw fps logs.
5. Switch the Ubuntu config to `path: /h264Preview_01_main` and confirm in
   isolation (still no tunnel) that main works the same way once the Mac is
   stopped (you can briefly stop the Mac for a 60s test and restart it).

### Data migration

Sizes on the Mac (as of writing):

- `camwatch.db`: 11 MB, 2828 pass rows
- `events/`: 34 MB (per-pass JSONL + calibration frames)
- `recordings/`: 852 MB (mp4 clips, 8-day retention)
- `config/`: small (calibration yaml, homography, etc.)

```
rsync -avh --progress \
  /Users/lei/github/camwatch/{camwatch.db,events,recordings,config,.env} \
  lchen@192.168.0.137:~/camwatch/
```

If `recordings/` over WiFi is too slow, an option is to rsync only the last
N days (matches `retention.recordings_days: 8`) or start fresh.

### Cutover (~30s downtime)

1. On the Mac: `pkill -f "camwatch serve"` and stop the local `cloudflared`.
2. On the Mac: copy `~/.cloudflared/` (cert + tunnel credential JSON for
   tunnel `0b1d9e07-8ba2-4671-bbfc-a5586eff3b6d`) to Ubuntu:
   ```
   rsync -av ~/.cloudflared/ lchen@192.168.0.137:~/.cloudflared/
   ```
3. On Ubuntu: install `cloudflared`, start the same tunnel:
   ```
   cloudflared tunnel run 0b1d9e07-8ba2-4671-bbfc-a5586eff3b6d
   ```
   DNS for `camwatch.leidevs.com` is bound to the tunnel ID, not the host,
   so it follows automatically.
4. On Ubuntu: start `camwatch serve --host 0.0.0.0`.
5. Hit `https://camwatch.leidevs.com`, check perf panel, check that new
   passes land in the DB.

### Rollback

If anything misbehaves: stop Ubuntu's `camwatch` + `cloudflared`, restart
the Mac's. Same tunnel ID, same DNS, instant flip back.

## Open follow-ups (post-cutover)

- Ubuntu box is on WiFi (`wlp5s0`, signal -46 dBm). Wired uplink to the
  camera switch would reduce jitter; main-stream RTSP is more sensitive to
  latency than sub.
- Consider raising `imgsz` for the main-stream YOLO call once we see real
  accuracy numbers — 8 GB VRAM allows 1280 or higher easily.
- Delete dual-stream code (`TimestampedFrameBuffer`, thumb upgrader's
  cross-stream lookup) once the single-stream path has run for a stable
  period.
