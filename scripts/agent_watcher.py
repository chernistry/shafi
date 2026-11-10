#!/usr/bin/env python3
"""
Agent orchestration watcher — polls .sdd/agents/*/STATUS.json for changes
and monitors BULLETIN.jsonl for cross-agent messages.

Usage:  python scripts/agent_watcher.py
        Ctrl+C to stop (prints final summary).
"""

import json
import signal
import sys
import time
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent.parent / ".sdd" / "agents"
BULLETIN = AGENTS_DIR / "BULLETIN.jsonl"
POLL_INTERVAL = 10  # seconds


# ---- state ----
prev_status: dict[str, dict] = {}   # agent_name -> last-seen STATUS.json content
prev_mtime: dict[str, float] = {}   # agent_name -> last-seen mtime of STATUS.json
bulletin_lines_read: int = 0


def discover_agents() -> list[str]:
    """Return names of agent subdirectories."""
    return sorted(
        d.name for d in AGENTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fmt_status(data: dict) -> str:
    status = data.get("status", "unknown")
    last = data.get("last_completed", "n/a")
    agent = data.get("agent", "???").upper()
    return f"{agent}: status={status}  last_completed={last!r}"


def print_startup_summary():
    agents = discover_agents()
    print("=" * 64)
    print(f"  Agent Watcher started — monitoring {len(agents)} agents")
    print(f"  Agents dir: {AGENTS_DIR}")
    print("=" * 64)
    for name in agents:
        status_path = AGENTS_DIR / name / "STATUS.json"
        data = read_json(status_path)
        if data:
            prev_status[name] = data
            prev_mtime[name] = status_path.stat().st_mtime
            print(f"  [{data.get('status', '?').upper():>7}] {fmt_status(data)}")
        else:
            print(f"  [     --] {name.upper()}: no STATUS.json yet")
    print("-" * 64)


def check_bulletin():
    global bulletin_lines_read
    if not BULLETIN.exists():
        return
    lines = BULLETIN.read_text().splitlines()
    new_lines = lines[bulletin_lines_read:]
    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            sender = msg.get("from", "?").upper()
            mtype = msg.get("type", "msg")
            text = msg.get("message", line)
            print(f"[BULLETIN] {sender} ({mtype}): {text}")
        except json.JSONDecodeError:
            print(f"[BULLETIN] (raw) {line}")
    bulletin_lines_read = len(lines)


def check_agents():
    for name in discover_agents():
        status_path = AGENTS_DIR / name / "STATUS.json"
        if not status_path.exists():
            continue

        mtime = status_path.stat().st_mtime
        if name in prev_mtime and mtime == prev_mtime[name]:
            continue  # no change

        data = read_json(status_path)
        if data is None:
            continue

        old = prev_status.get(name, {})
        old_status = old.get("status")
        new_status = data.get("status")
        agent_label = name.upper()

        # Detect transition or first appearance
        if new_status != old_status or name not in prev_status:
            if new_status == "idle":
                last = data.get("last_completed", "n/a")
                print(f"[NOTIFY] {agent_label} is idle — last completed: {last!r}")
                # Check for pending task
                task_path = AGENTS_DIR / name / "TASK.json"
                task = read_json(task_path)
                if task:
                    assigned = task.get("assigned_at", "")
                    picked_up = data.get("picked_up_at", "")
                    if assigned and assigned > picked_up:
                        print(f"  -> Pending task found: {task.get('task_id', '?')}")
                    else:
                        print(f"[DISPATCH NEEDED] {agent_label} needs a new task")
                else:
                    print(f"[DISPATCH NEEDED] {agent_label} needs a new task")

            elif new_status == "blocked":
                needs = data.get("needs_from_others", [])
                print(f"[BLOCKED] {agent_label} blocked — needs: {needs}")

            elif new_status == "working":
                print(f"[WORKING] {agent_label} picked up work")

            else:
                print(f"[STATUS] {agent_label} -> {new_status}")

        prev_status[name] = data
        prev_mtime[name] = mtime


def print_final_summary():
    print()
    print("=" * 64)
    print("  Final agent states")
    print("=" * 64)
    for name in discover_agents():
        data = prev_status.get(name)
        if data:
            print(f"  [{data.get('status', '?').upper():>7}] {fmt_status(data)}")
        else:
            print(f"  [     --] {name.upper()}: never reported")
    print("=" * 64)


def handle_exit(_sig, _frame):
    print_final_summary()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    if not AGENTS_DIR.is_dir():
        print(f"ERROR: agents dir not found: {AGENTS_DIR}", file=sys.stderr)
        sys.exit(1)

    print_startup_summary()

    # Catch up on existing bulletin
    check_bulletin()

    print(f"\nPolling every {POLL_INTERVAL}s … (Ctrl+C to stop)\n")

    while True:
        check_agents()
        check_bulletin()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
