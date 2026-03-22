#!/usr/bin/env bash
# dagan_check.sh — Live agent status monitor
# Run: bash scripts/dagan_check.sh [--watch]
# With --watch: refreshes every 30 seconds

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS_DIR="$ROOT/.sdd/agents"

print_status() {
  echo ""
  echo "══════════════════════════════════════════════════════"
  echo "  DAGAN STATUS CHECK  —  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "══════════════════════════════════════════════════════"

  local any_idle=0
  local any_stuck=0

  for agent in shai orev eyal noga tzuf tamar dagan; do
    local dir="$AGENTS_DIR/$agent"
    [[ -d "$dir" ]] || continue

    # Read status
    local status="?"
    local task="?"
    if [[ -f "$dir/STATUS.json" ]]; then
      status=$(python3 -c "import json; d=json.load(open('$dir/STATUS.json')); print(d.get('status','?'))" 2>/dev/null)
      task=$(python3 -c "import json; d=json.load(open('$dir/STATUS.json')); print(d.get('current_task', d.get('last_completed', d.get('last_updated','?')))[:40])" 2>/dev/null)
    fi

    # Read queue
    local pending=0
    local active=0
    local next_task="—"
    if [[ -f "$dir/TASK_QUEUE.jsonl" ]]; then
      pending=$(python3 -c "
import json
lines = open('$dir/TASK_QUEUE.jsonl').readlines()
tasks = [json.loads(l) for l in lines if l.strip()]
print(sum(1 for t in tasks if t.get('status') == 'pending'))
" 2>/dev/null)
      active=$(python3 -c "
import json
lines = open('$dir/TASK_QUEUE.jsonl').readlines()
tasks = [json.loads(l) for l in lines if l.strip()]
print(sum(1 for t in tasks if t.get('status') == 'active'))
" 2>/dev/null)
      next_task=$(python3 -c "
import json
lines = open('$dir/TASK_QUEUE.jsonl').readlines()
tasks = [json.loads(l) for l in lines if l.strip()]
pending = [t for t in tasks if t.get('status') == 'pending']
if pending:
    t = pending[0]
    print(t['task_id'] + ': ' + t.get('description','')[:50])
else:
    print('QUEUE EMPTY')
" 2>/dev/null)
    fi

    # Status icon
    local icon="⚡"
    if [[ "$status" == "idle" || "$pending" == "0" ]]; then
      icon="🔴"
      any_idle=1
    elif [[ "$status" == "working" || "$status" == "active" ]]; then
      icon="🟢"
    elif [[ "$status" == "pending_activation" ]]; then
      icon="⬜"
    fi

    printf "  %s %-8s │ queue: %s pending, %s active\n" "$icon" "${agent^^}" "$pending" "$active"
    printf "         %-8s │ status: %-10s\n" "" "$status"
    if [[ "$next_task" != "—" ]]; then
      printf "         %-8s │ next: %s\n" "" "$next_task"
    fi
    echo ""
  done

  # Recent commits
  echo "──────────────────────────────────────────────────────"
  echo "  RECENT COMMITS:"
  git -C "$ROOT" log --oneline -6 2>/dev/null | while read line; do echo "    $line"; done

  # Last 3 BULLETIN entries
  echo ""
  echo "──────────────────────────────────────────────────────"
  echo "  BULLETIN (last 3):"
  tail -3 "$AGENTS_DIR/BULLETIN.jsonl" 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        d = json.loads(line)
        ts = d.get('timestamp','')[:16]
        frm = d.get('from','?')
        msg = d.get('message','')[:120]
        print(f'  [{ts}] {frm}: {msg}')
    except: pass
" 2>/dev/null

  if [[ "$any_idle" == "1" ]]; then
    echo ""
    echo "  ⚠️  IDLE AGENTS DETECTED — DAGAN needs to dispatch tasks!"
    echo "  Run: claude 'Read .sdd/agents/dagan/DAGAN_INSTRUCTIONS.md and GO'"
  fi

  echo "══════════════════════════════════════════════════════"
}

if [[ "$1" == "--watch" ]]; then
  echo "DAGAN WATCH MODE — refreshing every 30s (Ctrl+C to stop)"
  while true; do
    clear
    print_status
    sleep 30
  done
else
  print_status
fi
