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

python -m uvicorn src.agent.server:app \
    --host 0.0.0.0 \
    --port 8400 \
    --reload \
    --log-level info
