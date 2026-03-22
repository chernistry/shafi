import json
import time

def get_pending_tasks():
    try:
        with open(".sdd/agents/noam/TASK_QUEUE.jsonl") as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        return [t for t in tasks if t.get("status") == "pending" and "TRIGGER" not in t.get("description", "")]
    except Exception:
        return []

print("Polling for new tasks in TASK_QUEUE.jsonl (sleep 30)...")
for _ in range(10):
    tasks = get_pending_tasks()
    if tasks:
        print(f"NEW_TASK_FOUND: {tasks[0]['task_id']}")
        break
    time.sleep(30)
else:
    print("NO_NEW_TASKS_YET")
