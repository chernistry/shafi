#!/bin/bash
# spawn_with_task.sh — Spawn an agent with a SPECIFIC TASK injected into the prompt.
# This implements the AutoGen "handoff" pattern: orchestrator passes task context
# directly to the worker instead of the worker reading a shared queue.
#
# Usage:
#   ./scripts/spawn_with_task.sh AGENT_NAME TASK_ID [model]
#   ./scripts/spawn_with_task.sh shai shai-12c sonnet
#
# If TASK_ID is omitted or "auto", picks the first pending task from the queue.

set -e

AGENT="${1:-}"
TASK_ID="${2:-auto}"
MODEL="${3:-sonnet}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "$AGENT" ]; then
  echo "Usage: $0 AGENT_NAME [TASK_ID|auto] [model]"
  exit 1
fi

QUEUE_FILE="$REPO_ROOT/.sdd/agents/$AGENT/TASK_QUEUE.jsonl"
if [ ! -f "$QUEUE_FILE" ]; then
  echo "ERROR: No task queue for agent '$AGENT'"
  exit 1
fi

AGENT_UPPER=$(echo "$AGENT" | tr '[:lower:]' '[:upper:]')

# Extract task JSON (first pending if TASK_ID=auto, else specific task)
TASK_JSON=$(python3 -c "
import json, sys
lines = open('$QUEUE_FILE').readlines()
tasks = [json.loads(l) for l in lines if l.strip()]
pending = [t for t in tasks if t.get('status') == 'pending']
if not pending:
    print('NONE')
    sys.exit(0)
if '$TASK_ID' == 'auto':
    print(json.dumps(pending[0]))
else:
    for t in pending:
        if t.get('task_id') == '$TASK_ID':
            print(json.dumps(t))
            sys.exit(0)
    print('NONE')
" 2>/dev/null)

if [ "$TASK_JSON" = "NONE" ] || [ -z "$TASK_JSON" ]; then
  echo "No pending task found for $AGENT (TASK_ID=$TASK_ID). Spawning in standby mode."
  exec "$(dirname "$0")/spawn_agent.sh" "$AGENT" "$MODEL"
fi

TASK_DESCRIPTION=$(echo "$TASK_JSON" | python3 -c "import json,sys; t=json.load(sys.stdin); print(t.get('description',''))")
TASK_DETAILS=$(echo "$TASK_JSON" | python3 -c "import json,sys; t=json.load(sys.stdin); print(t.get('details','')[:800])")
ACTUAL_TASK_ID=$(echo "$TASK_JSON" | python3 -c "import json,sys; t=json.load(sys.stdin); print(t.get('task_id',''))")

SESSION_TS=$(date +%Y%m%d_%H%M%S)
SESSION_LOG="$REPO_ROOT/.sdd/agents/$AGENT/session_${SESSION_TS}.log"

# Build a focused prompt that includes the specific task — no file-reading overhead
PROMPT="You are $AGENT_UPPER, a specialist agent in Team Tzur Labs competing in the Agentic RAG Legal Challenge 2026.

YOUR IMMEDIATE TASK (already dispatched, start NOW):
Task ID: $ACTUAL_TASK_ID
Description: $TASK_DESCRIPTION
Details: $TASK_DETAILS

STARTUP SEQUENCE:
1. cd to $REPO_ROOT
2. Mark task '$ACTUAL_TASK_ID' as 'active' in .sdd/agents/$AGENT/TASK_QUEUE.jsonl
3. Update .sdd/agents/$AGENT/STATUS.json: status=working, current_task=$ACTUAL_TASK_ID, heartbeat_ts=NOW
4. EXECUTE THE TASK above immediately — do not re-read protocol unless stuck
5. After task: mark 'done', update STATUS.json, commit, post finding to BULLETIN.jsonl
6. Check queue for next pending task and continue
7. If queue empty: enter STANDBY (re-read queue on every user message)

AGENT PROTOCOL (quick reference, only if needed):
- Read .sdd/agents/AGENT_INSTRUCTIONS_V2.md for full protocol
- Write heartbeat_ts to STATUS.json every 3-5 minutes or watchdog respawns you
- Agent-to-agent dispatch: see §2b in AGENT_INSTRUCTIONS_V2.md

IMPORTANT: Work autonomously. Commit frequently. Post findings to BULLETIN.jsonl."

echo "=========================================="
echo "Spawning: $AGENT_UPPER (task-injected)"
echo "Task: $ACTUAL_TASK_ID — $TASK_DESCRIPTION"
echo "Model: $MODEL"
echo "Log: $SESSION_LOG"
echo "=========================================="

nohup claude -p "$PROMPT" \
  --model "$MODEL" \
  --dangerously-skip-permissions \
  2>&1 | tee "$SESSION_LOG" &

AGENT_PID=$!
echo "Started PID: $AGENT_PID"
echo "Monitor: tail -f $SESSION_LOG"
