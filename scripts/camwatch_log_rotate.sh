#!/bin/sh
# Rotate ~/Library/Logs/camwatch/camwatch.log daily.
# Snapshot via cp (so the running camwatch process keeps its append fd valid),
# gzip the snapshot, then truncate the live log in place.
#
# Run via launchd; see scripts/com.lei.camwatch-logrotate.plist.

set -eu

LOG="${HOME}/Library/Logs/camwatch/camwatch.log"
KEEP=7

[ -f "$LOG" ] || exit 0
SIZE=$(stat -f%z "$LOG" 2>/dev/null || echo 0)
[ "$SIZE" -gt 0 ] || exit 0

DATE=$(date +%Y-%m-%d)
ARCHIVE="${LOG}.${DATE}"

cp "$LOG" "$ARCHIVE"
gzip -f "$ARCHIVE"
: > "$LOG"

# Keep only the $KEEP most recent .gz archives; drop older.
ls -1t "${LOG}".*.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -I {} rm -f {}
