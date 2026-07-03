#!/bin/sh
# deploy.sh — the ONE audited deploy action for the camwatch engine.
#
# This script is a security boundary: the operator's session is
# allowlisted to run exactly this file, so it must do exactly one thing.
# Hardcoded target, no arguments, no environment knobs:
#
#   ssh lei-ubuntu →
#     git pull --ff-only on main in ~/github/camwatch
#     uv sync (the repo .venv the systemd unit runs from)
#     sudo systemctl restart camwatch.service   (this unit ONLY)
#     health check: is-active + UI HTTP 200 on 127.0.0.1:8000
#
# Fails loudly on any step. The branch guard exists because a checkout
# parked on a feature branch makes `uv sync` resolve a pre-pin lockfile
# (observed 2026-06-05: torch silently reverted to a CUDA-incompatible
# build). Deploys happen from main, full stop.
set -eu

[ "$#" -eq 0 ] || {
    echo "deploy.sh takes no arguments (fixed target: camwatch.service on lei-ubuntu)" >&2
    exit 2
}

# Single SSH session; the remote script re-asserts every invariant rather
# than trusting local state. `sh -e` aborts on the first failed step.
ssh lei-ubuntu /bin/sh -e <<'REMOTE'
cd "$HOME/github/camwatch"

branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "main" ] || {
    echo "DEPLOY REFUSED: checkout is on '$branch', not main" >&2
    exit 1
}

echo "== git pull --ff-only =="
git pull --ff-only

echo "== uv sync =="
"$HOME/.local/bin/uv" sync

echo "== restart camwatch.service =="
sudo systemctl restart camwatch.service

# Health: unit active AND the UI answering. Startup loads YOLO weights,
# so poll up to 60s before declaring failure.
state=unknown
http=000
i=0
while [ "$i" -lt 30 ]; do
    state=$(systemctl is-active camwatch.service || true)
    [ "$state" = "failed" ] && break
    http=$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 3 \
        http://127.0.0.1:8000/ 2>/dev/null || echo 000)
    [ "$state" = "active" ] && [ "$http" = "200" ] && break
    i=$((i + 1))
    sleep 2
done

rev=$(git rev-parse --short HEAD)
if [ "$state" = "active" ] && [ "$http" = "200" ]; then
    echo "deploy OK: $rev | camwatch.service active | UI HTTP 200"
else
    echo "DEPLOY UNHEALTHY: $rev | unit=$state | UI HTTP $http" >&2
    exit 1
fi
REMOTE
