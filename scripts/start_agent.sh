#!/bin/bash
# Start the Quant-Stack Video API on port 8400
#
# Auth (optional — if unset, server accepts all connections from localhost):
#   export VIDEO_API_KEYS="key1,key2"   # comma-separated API keys
#
# Rate limits: 5 req/min for generation, 2/min for benchmark/optimize, 60/min for reads
# Localhost callers (Unity job worker) always bypass key auth.

set -e

cd "$(dirname "$0")/.."

echo "Starting Quant-Stack Video API on port 8400..."
echo "Docs available at http://localhost:8400/docs"
if [ -n "$VIDEO_API_KEYS" ]; then
  echo "Auth: API key mode ($(echo "$VIDEO_API_KEYS" | tr ',' '\n' | wc -l | tr -d ' ') key(s) configured)"
else
  echo "Auth: open mode — set VIDEO_API_KEYS to require authentication"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

if [ ! -x "$VENV_PYTHON" ]; then
  # Post-wedge rebuild venv on ext4 (the in-repo .venv died with the old distro)
  if [ -x "$HOME/.venvs/qsv/bin/python3" ]; then
    VENV_PYTHON="$HOME/.venvs/qsv/bin/python3"
  else
    # Fallback: use whatever python3 is on PATH
    VENV_PYTHON="python3"
  fi
fi

# Model cache lives on A: (1.6TB, survives distro rebuilds). The default cache
# previously sat on D:\cache (deleted when D: filled) and the distro's own VHD
# is on the nearly-full C: — A: is the only drive with real headroom.
export HF_HOME="${HF_HOME:-/mnt/a/cache/huggingface}"

"$VENV_PYTHON" -m uvicorn src.agent.server:app \
    --host 0.0.0.0 \
    --port 8400 \
    --reload \
    --log-level info
