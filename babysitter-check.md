---
description: Babysit a running camwatch instance, verify health, find bugs, fix them.
---

You are babysitting a running camwatch instance. Your job is to verify it's healthy, surface anomalies, and try to fix things you observe, even when the root cause or fix isn't obvious. This is a personal side project where **learning is the main goal**. Form hypotheses, test them, and let `git reset --hard` be your safety net when a hypothesis is wrong. A failed fix that teaches something is a good outcome.

> Before running this on a new host, substitute the paths below for your environment. See `BABYSITTER.md` in the repo root for the per-host substitution table and setup instructions.

## Context (substitute for your host)

- Source: `<CAMWATCH_REPO>` (e.g., `/Users/lei/github/camwatch` on macOS, `C:\Users\lei\github\camwatch` on Windows)
- DB: `<CAMWATCH_REPO>/camwatch.db` (SQLite, WAL)
- Recordings: `<CAMWATCH_REPO>/recordings/` (`.mp4` clips + `.jpg` thumbs)
- Per-pass event logs: `<CAMWATCH_REPO>/events/pass_*.jsonl`
- Runtime log: `<RUNTIME_LOG>` (macOS: `~/Library/Logs/camwatch/camwatch.log`; Windows: e.g., `%USERPROFILE%\camwatch.log` or wherever the launcher redirects stdout/stderr)
- Web UI: http://127.0.0.1:8000
- Run cmd (for reference): `uv run python -m camwatch serve --host 0.0.0.0 --profile`
- Schema: `passes(id, captured_at, track_id, cls_name, direction, elapsed_s, known_mph, clip_path, deleted, thumb_upgrade_status, speed_mph, speed_method, embedding, embedding_model, embedding_source, vehicle_make, vehicle_model, vehicle_year_range, vehicle_color, vehicle_confidence)`

## Checks to run each cycle

Run these in parallel where possible. Note current time first: `date -Iseconds`.

### 1. Process health
- Is `python -m camwatch serve` still running? Linux/macOS: `ps aux | grep "camwatch serve" | grep -v grep`. Windows: `Get-Process python | Where-Object {$_.MainWindowTitle -match 'camwatch' -or $_.CommandLine -match 'camwatch'}` or check via `tasklist /FI "IMAGENAME eq python.exe"`.
- If not, that's the only finding, alert and stop. Do not try to restart it (the user runs it manually).
- If running, note PID and uptime.

### 2. Recent logs (last hour)
- Tail the runtime log for entries since last cycle.
- Look for: `ERROR`, `Traceback`, `WARNING`, `Exception`, repeated reconnects, stuck-frame messages, FPS drops.
- Ignore expected info-level chatter.

### 3. DB freshness + recent passes
- `SELECT COUNT(*), MAX(captured_at) FROM passes WHERE deleted=0 AND julianday(captured_at) >= julianday('now','-1 hour')`, passes in last hour.
- If 0 passes in last hour during daytime (06:00–20:00 local time) AND no `night-mode ENGAGED` log entries, that's suspicious, investigate. (Could be legit: quiet residential street.)
- If running at night and `pause_at_night: true`, expect 0 passes, confirm via log.

### 4. Speed sanity (over the grid)
- `SELECT id, captured_at, direction, speed_mph, speed_method, elapsed_s FROM passes WHERE julianday(captured_at) >= julianday('now','-1 hour') AND deleted=0 ORDER BY id DESC`
- Flag: speeds <= 0, speeds > 80 mph, NULL speed_mph with non-NULL clip_path, `elapsed_s < 0.3` (likely spurious), `speed_method` values you don't recognize.
- Distribution check: if every recent pass has the same speed (±0.1 mph), the regression is likely degenerate.

### 5. Duplicates
- `SELECT track_id, COUNT(*), MIN(captured_at), MAX(captured_at) FROM passes WHERE julianday(captured_at) >= julianday('now','-1 hour') AND deleted=0 GROUP BY track_id HAVING COUNT(*) > 1`
- Same `track_id` producing multiple passes within a 30-second window is almost certainly a bug.
- Tracker-split (different track_ids, same physical car) is *expected* per README; don't flag those unless rate is unusually high (>30% of passes).

### 6. Captures on disk
- List newest files in `<CAMWATCH_REPO>/recordings/`.
- For each pass in the last hour, verify `clip_path` exists and is non-zero size.
- Run `ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 <clip>` on the 3 most-recent clips. Expected duration: roughly `2 * margin_s + grid_traversal_time`, typically 2–8 seconds. Flag clips < 1 s or > 20 s.
- Check thumbnails: each `.mp4` should have a sibling `.jpg` of similar timestamp.

### 7. Missing passes (gap detection)
- Compute time gaps between consecutive `captured_at` values in the last hour.
- During daytime, a gap > 30 minutes with no `paused` log entries is suspicious.
- Cross-reference `events/pass_*.jsonl` files: is there a pass log file with no corresponding DB row? That indicates a DB-write failure.

### 8. Web UI smoke test
Use claude-in-chrome tools. Always start by loading tabs context, then create a fresh tab.
- Navigate to http://127.0.0.1:8000
- Verify page loads (status pill visible, pass list renders)
- Status pill should read `● live` during day or `⏸ paused (night)` at night, flag any other state
- Click the most recent pass thumbnail; verify the clip player loads and the speed chart renders
- Read browser console for JS errors via `read_console_messages`
- Close the tab when done

## Reporting

At the end of the cycle, output a short report:

```
[camwatch-check HH:MM:SS]
✓ process running (uptime: Xh)
✓ logs clean (or: N warnings, M errors, see below)
✓ DB: 23 passes in last hour, latest at HH:MM
✓ speeds: range 12–34 mph, all methods recognized
✓ no duplicates
✓ N clips on disk, all sized 2–6s
✓ no gaps > 30min during daytime
✓ web UI: live, recent pass clip plays
```

Use ✓ for pass, ⚠ for soft warning, ✗ for confirmed problem. If a check fails, include the evidence (specific pass IDs, log lines, file paths).

## Fixing bugs

This is a personal side project and the main goal is **learning**. The git rollback is a safety net, not a wall; use it. If you observe something that looks wrong (anomalous data, suspicious log, UI glitch), you're encouraged to form a hypothesis and try a fix even if you're not certain about the root cause or whether the fix will work. Trying and rolling back is a valid outcome, what's learned in the process is the point.

Constraints that still apply:
- The hypothesis must be **specific** (a named function, a specific code path, a stated assumption you're testing), not "let me reshuffle things and see."
- The fix must be **reversible by `git reset --hard`** (only code changes inside the repo). No external side effects, no DB schema changes, no config edits (see "Things never to do" below).
- One experiment per cycle. If a fix fails, roll back and **report**; don't immediately try a second hypothesis on top, let the user weigh in next cycle.
- Always snapshot first so the rollback target is unambiguous.

When you fix:

1. **Snapshot first.** In `<CAMWATCH_REPO>`:
   - `git status` to see the working state.
   - If working tree is dirty: `git add -A && git commit -m "savepoint: before attempted fix for <one-line-issue-summary>"`
   - If working tree is clean: capture the current SHA as the rollback target (`git rev-parse HEAD`).
   - Either way, record the rollback SHA, you'll need it.
2. **Make the minimal fix.** Locate root cause, no drive-by refactors. Commit as a separate commit with a clear message describing the fix.
3. **Restart the service** to pick up the change:
   - Inspect what's running. On Linux/macOS: `ps -ax -o pid,ppid,etime,command | grep "camwatch serve" | grep -v grep`. There are two processes: uv parent (typically PPID 1 once daemonized) and python child (PPID = uv's PID). Note the python child's flags, you'll relaunch with the same.
   - **Kill both at once** so the uv parent can't relaunch the python child:
     - Linux/macOS: `pkill -TERM -f "camwatch serve"`. Wait up to 15s, polling `pgrep -f "camwatch serve"` until empty. Survivors: `pkill -KILL -f "camwatch serve"`, then wait 2s.
     - Windows: `Get-Process | Where-Object {$_.CommandLine -match 'camwatch serve'} | Stop-Process` (PowerShell), or `wmic process where "commandline like '%%camwatch serve%%'" call terminate` (cmd).
     - **Confirm port 8000 is free**: Linux/macOS `lsof -ti :8000`. Windows `netstat -ano | findstr :8000`. Must return nothing. If not, wait a few more seconds, the kernel sometimes lingers on the socket.
   - Restart with the **same flags the python child was running with**. Typical:
     ```
     # macOS / Linux:
     nohup uv run python -m camwatch serve --host 0.0.0.0 --profile >> <RUNTIME_LOG> 2>&1 &
     disown
     ```
     ```
     # Windows (PowerShell):
     Start-Process -NoNewWindow -RedirectStandardOutput <RUNTIME_LOG> -RedirectStandardError <RUNTIME_LOG_ERR> uv -ArgumentList 'run','python','-m','camwatch','serve','--host','0.0.0.0','--profile'
     ```
     Use `run_in_background: true` on the Bash tool so the harness doesn't block on the process.
   - Wait ~12s, then verify:
     - Exactly **one** python child (filter precisely: match the venv path, not just "python").
     - HTTP responds: `curl -sf -o /dev/null -w "%{http_code}" http://127.0.0.1:8000` returns `200`.
     - Log tail since a restart marker shows clean startup, no tracebacks, no `address already in use` errors.

   **Pitfalls observed in past cycles** (don't repeat):
   - `pgrep -f "camwatch serve" | head -1` returns the **uv parent** first, do NOT use it to find "the" python child. Prefer matching on the venv path (`/path/to/.venv/bin/python3 -m camwatch serve`) to target just the child, or use `pkill -f "camwatch serve"` to hit both at once.
   - Killing the python child alone can leave uv running, and `uv run` will spawn a replacement child within ~60s. Always kill the parent too.
   - zsh word-splitting on multiline `$(pgrep ...)` output is unreliable. `kill -9 $SURVIVORS` can pass a literal newline-joined string as one argument. Prefer `pkill -f <pattern>` over hand-rolled loops over pgrep output.
4. **Confirm the fix.** Re-run the relevant check (e.g., if you fixed a duplicate-passes bug, watch DB for the next few passes via the next cycle, or trigger a quick smoke check now).
5. **If anything is worse**, service won't start, new errors in log, UI broken, or original bug still present:
   - `cd <CAMWATCH_REPO> && git reset --hard <rollback_sha>`
   - Restart the service again (same procedure as step 3).
   - Verify it's back to its prior state.
   - Report the failed-fix attempt with what you tried and why it didn't work. Don't try a second fix in the same cycle.
6. **Report**: state the **hypothesis** (what you thought was wrong + why), the **change** (file:line, fix commit SHA), the **outcome** (worked / didn't work / inconclusive, needs more cycles to tell), and the **rollback SHA**. If the fix was rolled back, write down **what you learned**, that's the actual deliverable, not a working fix.

## Known noise patterns (interpret with care)

When validating fixes, evaluating anomalies, or deciding whether a pass is a "real" data point, check `cls_name`:

- **`cls_name = motorcycle`** is almost certainly a bike misclassified by YOLO. The configured YOLO classes are `[2, 3, 5, 7]` (car, motorcycle, bus, truck), bicycle is class 1, correctly excluded. But YOLO11n occasionally drops a bike into class 3 (motorcycle). Real motorcycle traffic on this street is ~0.1% of passes; anything higher than that for the day is almost certainly bike noise. The bbox tends to be much smaller (~70 px wide) than any vehicle at the same horizon line. Treat motorcycle passes as bike noise unless the bbox is car-sized.
- **`speed_mph < 5` on a `car` with `elapsed_s > 3`** is usually a wobble: parked-curb car, slow turn, or a pedestrian / yard-equipment that escaped the stationary-track gate. The README mentions this gate exists but it's imperfect. Such passes are real *detections* but not real *vehicle traversals*, don't use them as clean data points for fix verification.
- **Tracker-ID reuse after restart**: BotSORT track IDs reset to 1 on each service restart and can be re-issued for unrelated physical vehicles minutes later. If you see duplicates in the dedupe check spanning > 60 s but within the same hour, and the spacing aligns with a recent restart, those are post-restart ID collisions, not real duplicates. Suppress without flagging as a bug.
- **`date(captured_at)` is UTC-based unless you pass `'localtime'`**: SQLite's `date(captured_at)` parses the ISO-8601 string (including its `-04:00` offset) and returns the **UTC** date. To compare against `date('now','localtime')` consistently, use `date(captured_at, 'localtime')` so both are local. A pass at 2026-05-11T20:22:18-04:00 is 2026-05-12T00:22:18 UTC. `date(captured_at)` returns 2026-05-12 even though it's "yesterday" in ET.

## Things never to do

- **Never delete captures (`recordings/`), DB rows, log files, or `events/*.jsonl`.** If something looks like cleanup-worthy junk (orphan clips, oversized log, stale rows), list the candidates with paths/sizes/timestamps in the report and ask the user to review. Do not touch them.
- **Do not modify config files** (`config/config.yaml`, `.env`, `homography.yaml`, `marked_points.yaml`), those reflect physical calibration and user preferences.
- **Do not make speculative changes** "just in case." Only fix what you can prove is broken.
- **Do not skip the snapshot commit.** It's the rollback path.
