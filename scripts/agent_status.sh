#!/bin/bash
# Quick status update for agents. Usage: bash scripts/agent_status.sh AGENT_NAME "what I'm doing"
# Updates STATUS.json in ONE command — no thinking needed
AGENT="${1:?Usage: agent_status.sh AGENT_NAME \"task description\"}"
TASK="${2:?Provide task description}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
python3 -c "
import json
from datetime import datetime, timezone
f='$REPO/.sdd/agents/$AGENT/STATUS.json'
d={'agent':'$AGENT','status':'working','current_task':'$TASK','heartbeat_ts':datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
json.dump(d, open(f,'w'), indent=2)
print('OK: $AGENT → $TASK')
"
