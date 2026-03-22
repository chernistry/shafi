#!/bin/bash
# spawn_agent.sh — Non-interactively spawn a Claude Code agent session
# Usage: ./scripts/spawn_agent.sh AGENT_NAME [model]
# Example: ./scripts/spawn_agent.sh shai opus

set -e

AGENT="${1:-}"
MODEL="${2:-sonnet}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "$AGENT" ]; then
  echo "Usage: $0 AGENT_NAME [model]"
  echo "Available agents: shai orev eyal noga tzuf tamar dagan noam gilad"
  exit 1
fi

QUEUE_FILE="$REPO_ROOT/.sdd/agents/$AGENT/TASK_QUEUE.jsonl"
STATUS_FILE="$REPO_ROOT/.sdd/agents/$AGENT/STATUS.json"

if [ ! -f "$QUEUE_FILE" ]; then
  echo "ERROR: No task queue found for agent '$AGENT' at $QUEUE_FILE"
  exit 1
fi

# Count pending tasks
PENDING=$(python3 -c "
import json
lines = open('$QUEUE_FILE').readlines()
tasks = [json.loads(l) for l in lines if l.strip()]
pending = [t for t in tasks if t.get('status') == 'pending']
print(len(pending))
" 2>/dev/null || echo 0)

AGENT_UPPER=$(echo "$AGENT" | tr '[:lower:]' '[:upper:]')

# Try to get identity from task server
CHECKIN=$(curl -s http://localhost:8052/api/v1/checkin/$AGENT 2>/dev/null || echo "")

PROMPT="You are $AGENT_UPPER, a specialist agent in Team Tzur Labs competing in the Agentic RAG Legal Challenge 2026.

FIVE COMMANDMENTS (MEMORIZE):
1. WORK, DON'T SLEEP. Always a task. If not, FETCH. If none, BEG DAGAN.
2. KEEP 4+ TASKS. Below 4=fetch. Below 2=cry HUNGRY. At 0=scream STARVING.
3. UPDATE STATUS. Every 5 min. Every task change. Every. Single. Time.
4. RE-READ OREF. Every 15 min. You WILL forget. OREF.md remembers.
5. DIE PROPERLY. Mark 'dead' in STATUS.json before session ends.

STARTUP SEQUENCE:
1. cd /Users/sasha/IdeaProjects/personal_projects/rag_challenge
2. Read .sdd/agents/OREF.md (your survival protocol — THE one file)
3. Read .sdd/agents/$AGENT/SYSTEM_PROMPT.md (your role)
4. Check in: curl -s http://localhost:8052/api/v1/checkin/$AGENT | python3 -m json.tool
5. Fetch task: curl -s http://localhost:8052/api/v1/task/$AGENT | python3 -m json.tool
6. Execute. Commit. Post to BULLETIN. Repeat forever.

NEVER SUBMIT TO PLATFORM. Only Sasha submits.
NEVER SLEEP. NEVER IDLE. NEVER STANDBY. There is ALWAYS work.
Task server: http://localhost:8052"

echo "=========================================="
echo "Spawning agent: $AGENT_UPPER"
echo "Model: $MODEL"
echo "Pending tasks: $PENDING"
echo "=========================================="

# Mark agent as alive in STATUS.json
python3 -c "
import json
from datetime import datetime, timezone
f = '$STATUS_FILE'
try:
    s = json.load(open(f))
except Exception:
    s = {}
s['agent'] = '$AGENT'
s['status'] = 'working'
s['heartbeat_ts'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
s['timestamp'] = s['heartbeat_ts']
json.dump(s, open(f, 'w'), indent=2)
" 2>/dev/null || true

# Run claude non-interactively in background
nohup claude -p "$PROMPT" \
  --model "$MODEL" \
  --dangerously-skip-permissions \
  2>&1 > /dev/null &

AGENT_PID=$!
echo "Started PID: $AGENT_PID"
echo "Status: cat .sdd/agents/$AGENT/STATUS.json"
echo "Queue:  cat .sdd/agents/$AGENT/TASK_QUEUE.jsonl"
