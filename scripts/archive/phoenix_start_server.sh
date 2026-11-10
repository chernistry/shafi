#!/bin/bash
# Export all variables from .env.local
export $(grep -v '^#' .env.local | xargs)
# Force port 8000
export APP_PORT=8000
# Kill existing
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 2
# Start
nohup uv run uvicorn "rag_challenge.api.app:create_app" --factory --host 0.0.0.0 --port 8000 > /tmp/server_v10.log 2>&1 &
echo $! > server_v10.pid
echo "Server starting with PID=$!"
