import json
import os
import time
import datetime
from pathlib import Path
import subprocess

STATUS_FILE = Path(".sdd/agents/noam/STATUS.json")
BULLETIN_FILE = Path(".sdd/agents/BULLETIN.jsonl")
QUEUE_FILE = Path(".sdd/agents/noam/TASK_QUEUE.jsonl")
PRIVATE_DIR = Path("data/private")

def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def update_status(updates):
    status = {}
    if STATUS_FILE.exists():
        with open(STATUS_FILE, "r") as f:
            try:
                status = json.load(f)
            except json.JSONDecodeError:
                pass
    status.update(updates)
    status["heartbeat_ts"] = now_iso()
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)

def post_bulletin(msg_type, message):
    BULLETIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BULLETIN_FILE, "a") as f:
        f.write(json.dumps({
            "from": "noam",
            "timestamp": now_iso(),
            "type": msg_type,
            "message": message
        }) + "\n")

def run_pipeline():
    # 1. Post to BULLETIN
    pdf_count = len(list(PRIVATE_DIR.rglob("*.pdf")))
    post_bulletin("CRITICAL_ALERT", f"PRIVATE DATA ARRIVED! {pdf_count} docs detected at data/private/. Starting enrichment+ingestion pipeline NOW.")
    
    # 2. Update STATUS
    update_status({"status": "working", "current_task": "noam-2a/2b", "current_track": "Running enrichment and ingestion"})
    
    # Run noam-2a: Enrichment
    post_bulletin("finding", "NOAM enrichment STARTED. Concurrent with ingest.")
    subprocess.Popen(
        ["uv", "run", "python", "scripts/run_isaacus_enrichment.py", "--input-dir", "data/private", "--output-dir", "data/enrichments/private"],
        env=dict(os.environ, **{"ENV_FILE": "profiles/private_v7_enhanced.env"})
    )
    
    # Run noam-2b: Ingestion
    # "Command: ENV_FILE=profiles/private_v7_enhanced.env uv run python scripts/ingest_pipeline.py --input-dir data/private OR the project's standard ingest command"
    # Looking at Makefile, standard ingest command is `docker compose --profile tools run --rm ingest` or something similar,
    # But wait, Makefile had `docker-ingest: $(COMPOSE) --profile tools run --rm ingest`.
    # And there is `scripts/run_private_dataset_pipeline.py` or similar. Let's just run what's described in the instruction if possible.
    post_bulletin("finding", "NOAM ingest STARTED. Expected completion: T+90min.")
    subprocess.Popen(
        ["make", "docker-ingest"],
    )

def main():
    print("NOAM Monitor started.")
    last_status_update = time.time()
    while True:
        try:
            if PRIVATE_DIR.exists() and any(PRIVATE_DIR.iterdir()):
                print("Private data detected!")
                run_pipeline()
                break
            
            # Heartbeat every 5 mins
            if time.time() - last_status_update > 300:
                update_status({"current_task": "noam-1c", "status": "standby", "current_track": "Monitoring data/private/"})
                last_status_update = time.time()
                
        except Exception as e:
            print(f"Error in monitor loop: {e}")
            
        time.sleep(120)

if __name__ == "__main__":
    main()
