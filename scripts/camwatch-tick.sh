#!/usr/bin/env bash
# camwatch-tick.sh — one headless observer cycle.
#
# Spawned by cron (or run manually for a dry run). Each invocation is a brand
# new Claude Code session; nothing persists between ticks except what the
# spec writes to ~/.claude/camwatch-check-status.md.
#
# Schedule (cron, every 6h):
#   0 */6 * * * /home/lchen/git/camwatch/scripts/camwatch-tick.sh
#
# Manual dry run:
#   /home/lchen/git/camwatch/scripts/camwatch-tick.sh
#   tail -f ~/.claude/camwatch-tick.log

set -euo pipefail

# Cron runs with a minimal PATH; make sure ~/.local/bin (where `claude`
# lives) and the standard tool dirs are reachable.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

REPO_DIR="/home/lchen/git/camwatch"
SPEC_PATH="/home/lchen/.claude/commands/camwatch-check.md"
TICK_LOG="$HOME/.claude/camwatch-tick.log"
MAX_LOG_BYTES=$((5 * 1024 * 1024))  # 5 MB

# Lightweight log rotation: if the tick log exceeds 5 MB, rotate it to
# camwatch-tick.log.1 (discarding any prior .1). Keeps the active file small
# without depending on logrotate.
if [[ -f "$TICK_LOG" ]]; then
  size=$(stat -c %s "$TICK_LOG" 2>/dev/null || echo 0)
  if (( size > MAX_LOG_BYTES )); then
    mv -f "$TICK_LOG" "${TICK_LOG}.1"
  fi
fi

# Make sure the log file exists so the redirection below can append.
: > "$TICK_LOG.lock" 2>/dev/null || true
mkdir -p "$(dirname "$TICK_LOG")"

{
  echo
  echo "=========================================="
  echo "tick start: $(date -Iseconds)"
  echo "=========================================="
} >> "$TICK_LOG"

# Run a single headless Claude session. The prompt is a file-reference (the
# cron-resolves-slash-commands gotcha documented in BABYSITTER.md). Tools run
# unattended via --permission-mode bypassPermissions; the spec is observer-
# only so the blast radius is bounded.
cd "$REPO_DIR"
claude -p \
  --permission-mode bypassPermissions \
  "Read $SPEC_PATH and execute the cycle end-to-end. Follow the observer-only constraint strictly: do not edit code, do not run git mutations, do not restart the service. Enrichment via scripts/enrich_apply.py is the only DB write permitted. End by Writing ~/.claude/camwatch-check-status.md per the spec's Step 10." \
  >> "$TICK_LOG" 2>&1 \
  || echo "tick FAILED with exit $? at $(date -Iseconds)" >> "$TICK_LOG"

{
  echo "------------------------------------------"
  echo "tick end:   $(date -Iseconds)"
  echo
} >> "$TICK_LOG"
