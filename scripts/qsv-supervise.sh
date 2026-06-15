#!/usr/bin/env bash
# Uptime supervisor for the Quant-Stack Video service on :8400.
#
# Installed in cron twice: @reboot (start on boot) and every minute (watchdog).
# Idempotent — a no-op when the service is already healthy; (re)starts it within
# ~1 min of a crash/wedge. No sudo / systemd required — matches this box's
# cron-based supervision convention (see crontab + /mnt/a/logs/cron).
#
# The dominant production failure mode for history-shorts renders is this service
# being DOWN (every shot's LTX call then falls back to gradients and the flat-gate
# refuses to ship), so keeping :8400 up is the point.
set -uo pipefail

PORT=8400
REPO=/mnt/a/submodules/quant-stack-video
VENV_PY="$HOME/.venvs/qsv/bin/python3"
LOG_DIR=/mnt/a/logs/qsv
HEALTH="http://127.0.0.1:${PORT}/health"
LOCK=/tmp/qsv-supervise.lock

mkdir -p "$LOG_DIR"
log() { echo "$(date -Is) $*" >> "$LOG_DIR/supervise.log"; }
healthy() { curl -fsS -m 5 "$HEALTH" >/dev/null 2>&1; }

# Serialize supervisor runs so overlapping cron ticks (or @reboot + watchdog)
# can never double-start the server.
exec 9>"$LOCK"
flock -n 9 || exit 0

# Retry before declaring it down — /health stays responsive even mid-render
# (asyncio.to_thread), so a single miss is a transient blip, not a dead server.
# This avoids killing a live generation on a hiccup.
for _ in 1 2 3; do
  healthy && exit 0
  sleep 3
done

log "health check failed 3x — (re)starting service on :$PORT"
pkill -f "uvicorn src.agent.server:app" 2>/dev/null || true
sleep 2

cd "$REPO" || { log "ERROR: repo $REPO missing"; exit 1; }
if [ ! -x "$VENV_PY" ]; then log "ERROR: venv python missing at $VENV_PY"; exit 1; fi
export HF_HOME="${HF_HOME:-/mnt/a/cache/huggingface}"

nohup "$VENV_PY" -m uvicorn src.agent.server:app \
    --host 127.0.0.1 --port "$PORT" --log-level info \
    >> "$LOG_DIR/service.log" 2>&1 &
pid=$!
log "started uvicorn pid $pid"

# Brief confirm; /health comes up fast (the model loads lazily on first generate).
for _ in $(seq 1 10); do
  sleep 2
  if healthy; then log "service healthy (pid $pid)"; exit 0; fi
done
log "WARN: started pid $pid but /health not up after ~20s — next watchdog tick re-checks"
exit 0
