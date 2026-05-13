# Babysitter handover

A scheduled Claude Code agent that periodically inspects a running camwatch
instance, surfaces anomalies, and ships fixes under documented constraints.
The babysitter ran on the MacBook from 2026-05-10 through 2026-05-12 (~36
cycles). This doc captures what's there and how to re-stand it up on the
Windows / Ubuntu desktop now hosting the service.

## What this is

The babysitter is a slash command (`/camwatch-check`) plus a `/loop` running
once an hour. Each cycle does the same set of inspections: process health,
recent log scan, DB freshness + speed sanity, duplicates, gap detection,
captures-on-disk, web UI smoke test. The exact specification lives in
`babysitter-check.md` at the repo root.

The cycle has documented rules for *fixing* bugs it finds: snapshot the repo
into git first, make the smallest possible change, restart the service,
verify the fix, and roll back via `git reset --hard` if anything got worse.
One experiment per cycle.

What worked: the babysitter caught and shipped fixes for the truncated-clip
recorder bug, the orphan `_big.jpg` leak from retention, two log rotation
mechanisms (size-based for `access.jsonl`, daily for `camwatch.log`), and
the retention split. It also surfaced and validated noise patterns
(motorcycle misclassifications, wobbles, post-restart track-ID reuse) that
should be filtered before flagging an anomaly.

What didn't work in the original setup, worth knowing for the new host:
- Cron-fired Claude Code sessions cannot resolve user-level slash commands
  via the `/name` syntax. The cron prompt must be a *file-reference prompt*
  that tells the spawned session to read `babysitter-check.md` and follow
  it. The exact prompt I used is at the bottom of this doc.
- `pgrep -f "camwatch serve" | head -1` returns the uv parent, not the
  python child. Killing the wrong one leaves the service unkillable until
  you target both. The fix is `pkill -f "camwatch serve"` (one shot, both
  procs).
- The `claude-in-chrome` MCP refs go stale fast under HTMX UI updates. The
  web-UI smoke step often had to settle for "page loads + status pill +
  console errors" rather than "click the most recent pass and verify the
  player." Not fatal.

## What was built (commit summary)

All commits in this list are on `origin/main`. They reflect work done by the
babysitter during operation, alongside fixes the user authored directly.

Babysitter-authored:

- `92f8f1b`: extend retention sweep to also delete `_big.jpg` and
  `events/pass_<id>.jsonl`
- `20df1a3`: rotate `access.jsonl` when it exceeds 10 MB (in-process)
- `98cc612`: launchd-based daily rotation for `camwatch.log` (macOS-only;
  see below for Ubuntu / Windows equivalents)
- `40ee7c1`: split storage retention into `recordings_days` and
  `passes_days` (recordings expire fast, DB rows + per-pass jsonl live
  longer for stats)
- `26d81cd`: rescue alarm passes from recordings retention into
  `recordings_archive/` (later scoped down)
- `df4a05c`: scope alarm archive to thumbnails only, delete the .mp4 at
  retention
- `f23a6d3`: heatmap speed mode: color by top mph instead of average

User-authored (alongside the babysitter session):

- `5484b9e`: fix truncated clips (time-based ring buffer), camera-reboot
  resilience, dual-track IoU dedup
- `5db98aa`: add vehicle make/model/color enrichment + per-row filter
- `49172c5`: add `enrich_offset_mph` config, bump rolling window to 8 days
- `a8d6e14`: drop refresh button spinner
- `cbc9125`: perf panel
- `f97fb96`: NVIDIA GPU migration plan
- `5fd907a`: car make/model enrichment plan

## Setting up on a new host

### Files you need

1. `babysitter-check.md`: the spec. Already in the repo root.
2. A way to invoke it on a schedule.

### On the new Claude Code host

Claude Code runs the same on macOS, Windows, and Linux. The slash command
just needs to live at the platform-specific user-commands directory:

| Host | Slash-command directory |
|------|-------------------------|
| macOS / Linux | `~/.claude/commands/` |
| Windows | `%USERPROFILE%\.claude\commands\` |

Copy `babysitter-check.md` from the repo root into that directory, named
`camwatch-check.md` (drop the `babysitter-` prefix so the slash command
matches the existing pattern).

Edit the **Context** section at the top to use the new host's paths. Quick
substitutions:

| Token | macOS | Ubuntu (3060 Ti host) | Windows |
|-------|-------|----------------------|---------|
| `<CAMWATCH_REPO>` | `/Users/lei/github/camwatch` | `/home/lchen/camwatch` | `C:\Users\lei\github\camwatch` |
| `<RUNTIME_LOG>` | `~/Library/Logs/camwatch/camwatch.log` | `~/.local/state/camwatch/camwatch.log` (or wherever your service script redirects stdout) | `C:\Users\lei\camwatch.log` (or wherever) |

Also edit the restart-procedure block to match the host's process tools.
The spec already includes Linux/macOS (`pkill`, `pgrep`, `lsof`) and
Windows (PowerShell / `wmic`) snippets side by side.

### Starting the loop

From Claude Code on the new host:

```
/loop 1h <the file-reference prompt below>
```

When asked "cloud schedule or this session only?", pick **this session
only** for an hourly local loop. Cloud schedule won't work for this
babysitter because the agent can't reach `127.0.0.1:8000`, the local
runtime log, or the DB.

The cron prompt I used (paste this in place of `/camwatch-check`):

```
You are running a scheduled camwatch babysitter cycle. Read the full
instructions at <full path to camwatch-check.md> and execute them
end-to-end. That file describes what to inspect (process, logs, DB,
captures, web UI, etc.), the rules for fixing bugs (snapshot first,
restart, verify, rollback on failure), and the report format. Do not
improvise, follow that file as the source of truth, and update it if
you learn something the next cycle should know.
```

Substitute `<full path to camwatch-check.md>` with the absolute path. On
macOS that was `/Users/lei/.claude/commands/camwatch-check.md`. On Windows
it'll be something like `C:\Users\lei\.claude\commands\camwatch-check.md`.

The reason this prompt works while `/camwatch-check` does not: cron-fired
sessions don't resolve user-level slash commands. The new session starts
without the harness having loaded `~/.claude/commands/*.md`. Reading the
file by absolute path sidesteps that.

## Log rotation on the new host

The macOS babysitter set up `~/Library/LaunchAgents/com.lei.camwatch-logrotate.plist`
to rotate `camwatch.log` daily at 03:17. That's macOS-specific. Equivalents:

- **Ubuntu**: add `camwatch.log` to a `/etc/logrotate.d/` config, or write a
  systemd timer that runs `scripts/camwatch_log_rotate.sh` (already in the
  repo at `scripts/camwatch_log_rotate.sh`. Linux-compatible bash).
- **Windows**: Task Scheduler running the same shell script via Git Bash /
  WSL, or rewrite as a PowerShell script. The rotation logic is just
  `cp + gzip + truncate-in-place`, ~10 lines.

The in-process `access.jsonl` rotation (commit `20df1a3`) is pure Python
and works everywhere. Nothing to migrate.

## Retention behavior to know about

Two knobs in `config.yaml` under `retention:`:

- `recordings_days` (current setting: 8). After this many days, the .mp4
  + thumbnails get deleted; the pass row stays in the DB with `clip_path`
  NULLed. If the pass was an alarm (`speed_mph >= alert.threshold_mph`),
  the thumbnails (`.jpg` + `_big.jpg`) get moved to `recordings_archive/`
  instead of deleted. The .mp4 is always deleted at retention regardless
  of alarm status.
- `passes_days` (current setting: 365). After this many days, the pass
  row is hard-deleted from the DB along with its `events/pass_<id>.jsonl`.

Backward compat: if the config has the legacy `retention.days: N` key, it
gets read as `recordings_days = N` with `passes_days = 0` (no auto-purge
of rows).

`recordings_archive/` is gitignored. It accumulates ~2-3 thumbnails per
alarm pass per day, ~100-200 KB each. Estimated steady-state growth: ~84
MB/year. Not auto-cleaned.

## Known noise patterns (filter these before flagging)

The babysitter's spec includes these but worth restating:

- **`cls_name = motorcycle`** is almost always a bike misclassified by
  YOLO. Real motorcycle traffic is ~0.1% of passes. Treat as bike noise
  unless the bbox is car-sized.
- **`speed_mph < 5` on a `car` with `elapsed_s > 3`** is a wobble: parked
  car, slow turn, pedestrian / yard equipment that slipped past the
  stationary-track gate. Real detection, not a real pass.
- **Tracker-ID reuse after restart**: BotSORT IDs reset on each restart.
  Duplicates spanning > 60 s within the same hour are usually post-restart
  ID collisions, not tracker bugs. The dedup spec's "<30 s, same direction"
  rule still catches real tracker bugs cleanly.
- **`date(captured_at)` is UTC-based** in SQLite. Use
  `date(captured_at, 'localtime')` to compare against `date('now','localtime')`
  consistently. A pass at 20:22 ET is 00:22 UTC the next day, so the naive
  query puts it on the wrong day.

## Open follow-ups

Things observed during babysitter operation but not yet shipped:

1. **HD upgrade success rate is ~67%, not the README's 70-80%.** The root
   cause: the Reolink main stream delivers frames in bursts (`pts_advance`
   swings between 0.5× and 3×), and the upgrader's ±1.5 s tolerance
   sometimes lands on a moment with no sampled frame in the current epoch.
   Candidate fixes: widen tolerance to ±3 s (cheap, mostly safe on a
   low-density residential street); raise the camera's main-stream fps in
   the Reolink web UI from current 2-3 fps to 4-5 fps (no code change);
   wait for the GPU migration to retire the cross-stream sync entirely
   (per MIGRATION_TO_NVIDIA_GPU.md).
2. **Archived alarm thumbnails are not surfaced in the UI.** When a pass's
   recordings expire, `clip_path` is NULLed and the UI shows it without a
   playback button. The archived `.jpg` / `_big.jpg` in `recordings_archive/`
   could be served via a small route or marked with a "view archived
   thumb" badge. Babysitter spec calls this out as a follow-up.
3. **Spec false-positive on `elapsed_s < 0.3`** for legit fast-truncated
   passes. The rule treats it as anomaly but the truncated-clip fix is
   designed to handle exactly that case. Refinement: "elapsed_s < 0.3 AND
   not in trajectory-truncated log."
4. **Restart procedure rough edges.** Even the corrected `pkill`-based
   sequence sometimes produced transient `address already in use` errors
   during the new-instance startup. End state is always healthy; the
   intermediate state has port overlap during the kill-respawn window.
   The launcher (`nohup uv run …`) plus a longer `sleep` before the new
   instance binds would fix this if it matters.
5. **Babysitter SQL bug**: today-count queries use `date(captured_at)`
   without `'localtime'`, so they bucket passes by UTC date and miscount
   late-evening / early-morning passes by one day. Spec has been updated
   to recommend `date(captured_at, 'localtime')` going forward.

## File map

- `babysitter-check.md`: the babysitter's per-cycle spec. Copy to
  `~/.claude/commands/camwatch-check.md` (or Windows equivalent) on the
  new host and edit the paths.
- `scripts/camwatch_log_rotate.sh`: the log rotation script. Linux/macOS
  bash; portable to WSL or Git Bash on Windows.
- `scripts/com.lei.camwatch-logrotate.plist`: macOS launchd plist. Not
  portable; rewrite as systemd / Task Scheduler config on the new host.
