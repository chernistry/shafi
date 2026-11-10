#!/bin/bash
# Start dashboard server on port 8050
cd "$(dirname "$0")/.." || exit 1

echo "🚀 Starting dashboard server on http://localhost:8050"
python3 -m dashboard.server
