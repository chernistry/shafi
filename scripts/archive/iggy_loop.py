#!/usr/bin/env python3
"""NOAM endless monitoring loop — updates dashboard every 5 minutes."""
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = ROOT / ".sdd" / "agents"
BULLETIN = AGENTS_DIR / "BULLETIN.jsonl"
DIRECTIVE = AGENTS_DIR / "DIRECTIVE.md"

def gather() -> str:
    """Gather current state and return Russian status note."""
    parts = []
    
    # Recent commits
    try:
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-3"],
            cwd=ROOT, text=True, timeout=5
        ).strip().split("\n")
        if log:
            parts.append(f"Коммиты: {log[0][:60]}")
    except Exception:
        pass
    
    # Agent statuses
    alive = []
    for agent in ["orev", "eyal", "shai", "tzuf", "noga", "dagan", "keshet", "liron", "tamar"]:
        status_file = AGENTS_DIR / agent / "STATUS.json"
        if status_file.exists():
            try:
                data = json.loads(status_file.read_text())
                task = data.get("current_task", "?")[:40]
                alive.append(f"{agent.upper()}: {task}")
            except Exception:
                pass
    
    if alive:
        parts.append(f"{len(alive)}/9 агентов живы")
    
    # Last bulletin
    if BULLETIN.exists():
        try:
            lines = BULLETIN.read_text().strip().split("\n")
            if lines:
                last = json.loads(lines[-1])
                msg = last.get("message", "")[:80]
                if msg:
                    parts.append(f"Последнее: {msg}")
        except Exception:
            pass
    
    # Submission files
    try:
        final = ROOT / "data" / "private_submission_FINAL_SUBMISSION.json"
        if final.exists():
            size_kb = final.stat().st_size // 1024
            mtime = datetime.fromtimestamp(final.stat().st_mtime, tz=timezone.utc)
            parts.append(f"FINAL_SUBMISSION: {size_kb}KB, обновлён {mtime.strftime('%H:%M')}")
    except Exception:
        pass
    
    return " | ".join(parts) if parts else "Всё тихо."

def post_note(text: str) -> None:
    """Post note using helper script."""
    subprocess.run(
        ["python3", "scripts/post_dashboard_note.py", text],
        cwd=ROOT, check=True
    )

def commit() -> None:
    """Commit dashboard changes."""
    subprocess.run(
        ["git", "add", "-f", "dashboard/static/status_notes.json"],
        cwd=ROOT, check=True
    )
    subprocess.run(
        ["git", "commit", "-m", "NOAM: dashboard update"],
        cwd=ROOT, check=False  # OK if nothing to commit
    )

def loop():
    """Endless monitoring loop."""
    print("🔥 NOAM loop started. Ctrl+C to stop.")
    cycle = 0
    while True:
        cycle += 1
        now = datetime.now(timezone.utc).strftime("%H:%M")
        
        try:
            status = gather()
            note = f"NOAM ({now}): {status}"
            print(f"[{cycle}] {note}")
            
            post_note(note)
            commit()
            
        except Exception as e:
            print(f"[{cycle}] ERROR: {e}")
        
        print(f"[{cycle}] Sleeping 5 min...")
        time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    loop()
