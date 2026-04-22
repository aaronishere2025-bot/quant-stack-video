#!/bin/bash
# LTX Thompson-sampling gen session launcher
#
# Starts overnight_apollo11.py in the background, detached from the terminal
# (safe to call from cron or by hand).
#
# PID file:  /tmp/gen-session.pid
# Logs:      /tmp/gen-session-YYYYMMDD.log
#
# Stop the session (finishes current clip cleanly):
#   kill $(cat /tmp/gen-session.pid)

set -euo pipefail

PID_FILE="/tmp/gen-session.pid"
LOG_FILE="/tmp/gen-session-$(date +%Y%m%d).log"
REPO="/mnt/d/ai-workspace/projects/quant-stack-video"
PYTHON="$REPO/.venv/bin/python3"

# ── Guard: refuse to start if a previous session is still running ────────────
if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Gen session already running (pid $OLD_PID). Aborting." >&2
    exit 1
  else
    echo "Stale PID file (pid $OLD_PID not running). Cleaning up."
    rm -f "$PID_FILE"
  fi
fi

# ── Launch ───────────────────────────────────────────────────────────────────
echo "=== $(date) === Starting LTX gen session ===" >> "$LOG_FILE"

# Load GEMINI_API_KEY from workspace env file if not already set
if [ -z "${GEMINI_API_KEY:-}" ] && [ -f "/mnt/d/ai-workspace/.env" ]; then
  export GEMINI_API_KEY=$(grep '^GEMINI_API_KEY=' /mnt/d/ai-workspace/.env | cut -d= -f2-)
fi

nohup "$PYTHON" -u "$REPO/scripts/overnight_apollo11.py" \
  >> "$LOG_FILE" 2>&1 &

GEN_PID=$!
echo "$GEN_PID" > "$PID_FILE"

echo "Started gen session: pid=$GEN_PID log=$LOG_FILE"
echo "Stop with:  kill \$(cat $PID_FILE)"
