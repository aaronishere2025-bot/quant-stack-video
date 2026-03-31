#!/bin/bash
# Start the autonomous quantization optimization agent on port 8400

set -e

cd "$(dirname "$0")/.."

echo "Starting Quant-Stack Video Agent on port 8400..."
echo "Docs available at http://localhost:8400/docs"

python -m uvicorn src.agent.server:app \
    --host 0.0.0.0 \
    --port 8400 \
    --reload \
    --log-level info
