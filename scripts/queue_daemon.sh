#!/bin/bash
# Queue daemon — runs forever, fills empty agent queues every 60s
# Launch: nohup bash scripts/queue_daemon.sh > /tmp/queue_daemon.log 2>&1 &

REPO="/Users/sasha/IdeaProjects/personal_projects/rag_challenge"
cd "$REPO"

CYCLE=0
while true; do
    CYCLE=$((CYCLE + 1))

    for agent in orev eyal shai tamar tzuf noga dagan keshet liron; do
        QUEUE=".sdd/agents/$agent/TASK_QUEUE.jsonl"
        [ -f "$QUEUE" ] || continue

        PENDING=$(python3 -c "
import json
tasks=[json.loads(l) for l in open('$QUEUE') if l.strip()]
print(sum(1 for t in tasks if t.get('status')=='pending'))
" 2>/dev/null)

        if [ "${PENDING:-0}" = "0" ]; then
            # Agent has 0 pending — inject work
            NOW=$(python3 -c "from datetime import datetime,timezone;print(datetime.now(timezone.utc).isoformat())")

            # Find their assigned ticket
            TICKET=$(grep -l "${agent^^}" .sdd/backlog/open/3*.md 2>/dev/null | head -1)
            if [ -n "$TICKET" ]; then
                DESC="Read your ticket: $TICKET. Execute it NOW."
            else
                DESC="No ticket assigned. Check POOL: .sdd/agents/POOL.jsonl. Or self-direct: read OREF.md section 4."
            fi

            python3 -c "
import json
task = {'task_id': 'auto-d${CYCLE}-${agent}', 'priority': 0, 'status': 'pending',
        'description': '$DESC', 'assigned_by': 'daemon', 'assigned_at': '$NOW'}
with open('$QUEUE', 'a') as f:
    f.write(json.dumps(task) + '\n')
" 2>/dev/null
            echo "$(date -u +%H:%M:%S) Filled $agent (was 0 pending)"
        fi
    done

    sleep 60
done
