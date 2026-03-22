import json
import time
import os

def get_pending_tasks():
    try:
        with open(".sdd/agents/noam/TASK_QUEUE.jsonl") as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        # Ignore 2a/2b/2c unless they are marked differently, but actually I should just
        # wait for any pending task. 
        # For now, we ignore the private data trigger ones if they are waiting for data.
        return [t for t in tasks if t["status"] == "pending" and t["task_id"] not in ("noam-2a", "noam-2b", "noam-2c")]
    except Exception as e:
        print(f"Error reading queue: {e}")
        return []

print("Polling for new tasks in .sdd/agents/noam/TASK_QUEUE.jsonl every 30 seconds...")
while True:
    new_tasks = get_pending_tasks()
    if new_tasks:
        print(f"NEW TASK FOUND: {new_tasks[0]['task_id']}")
        print(json.dumps(new_tasks[0], indent=2))
        break
    time.sleep(30)
