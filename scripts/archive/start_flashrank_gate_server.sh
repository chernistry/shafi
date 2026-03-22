#!/bin/bash
# Start FlashRank gate test server on port 8001.
# Sources .env.local (V7 config) then overlays flashrank_gate_test.env.
# Runs current HEAD (0176381 = G-guard + FlashRank adapter).
# Does NOT affect V7 server on port 8000.

set -e
cd "$(dirname "$0")/.."

# Load V7 base config
export $(grep -v '^#' .env.local | grep -v '^\s*$' | xargs)

# Overlay FlashRank overrides
export $(grep -v '^#' profiles/flashrank_gate_test.env | grep -v '^\s*$' | xargs)

# Kill any existing server on port 8001
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
sleep 1

echo "Starting FlashRank gate server on port 8001..."
echo "  RERANK_PROVIDER_MODE=$RERANK_PROVIDER_MODE"
echo "  RERANK_LOCAL_MODEL_PATH=$RERANK_LOCAL_MODEL_PATH"
echo "  QDRANT_COLLECTION=$QDRANT_COLLECTION"

nohup uv run uvicorn "rag_challenge.api.app:create_app" --factory --host 0.0.0.0 --port 8001 \
    > /tmp/server_flashrank_gate.log 2>&1 &
echo $! > server_flashrank_gate.pid
echo "Server starting with PID=$! — log: /tmp/server_flashrank_gate.log"
echo "Wait ~15s for model download/load, then run: uv run python scripts/tzuf_flashrank_gate_30q.py"
