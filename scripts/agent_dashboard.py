#!/usr/bin/env python3
"""Agent coordination dashboard — polls STATUS.json files and shows overview.

Usage:
    python scripts/agent_dashboard.py              # One-shot status check
    python scripts/agent_dashboard.py --watch 30   # Poll every 30 seconds
    python scripts/agent_dashboard.py --bulletin    # Show recent bulletin messages
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent.parent / ".sdd" / "agents"
AGENTS = ["orev", "eyal", "noga", "tamar", "tzuf", "shai", "noam"]

MODEL_MAP = {
    "orev": "Opus Max",
    "eyal": "Opus Max",
    "noga": "Opus",
    "tamar": "Sonnet",
    "tzuf": "Sonnet",
    "shai": "Opus Max",
    "noam": "Sonnet",
}

def _read_status(agent: str) -> dict | None:
    path = AGENTS_DIR / agent / "STATUS.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

def _read_task(agent: str) -> dict | None:
    for fname in ("TASK.json", "TASK.md"):
        path = AGENTS_DIR / agent / fname
        if path.exists():
            if fname.endswith(".json"):
                try:
                    return json.loads(path.read_text())
                except (json.JSONDecodeError, OSError):
                    return {"file": fname, "error": "parse error"}
            return {"file": fname, "size": path.stat().st_size}
    return None

def _git_branch_info(branch: str) -> str:
    """Get last commit info for a branch."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-1", branch],
            capture_output=True, text=True, timeout=5,
            cwd=AGENTS_DIR.parent.parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "?"
    except Exception:
        return "?"

def _time_ago(ts_str: str) -> str:
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - ts
        minutes = int(delta.total_seconds() / 60)
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        return f"{hours}h {minutes % 60}m ago"
    except Exception:
        return "?"

def print_dashboard() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*72}")
    print(f"  AGENT DASHBOARD — {now}")
    print(f"{'='*72}\n")

    for agent in AGENTS:
        status = _read_status(agent)
        model = MODEL_MAP.get(agent, "?")

        if status:
            st = status.get("status", "?")
            icon = {"idle": "⏸", "working": "🔄", "blocked": "🚫"}.get(st, "❓")
            last = status.get("last_completed", "")[:50]
            ts = _time_ago(status.get("timestamp", ""))
            tests = status.get("tests_passing", "?")
            commit = status.get("committed_at", "?")[:8]
            branch = status.get("branch", "?")
            needs = status.get("needs_from_others", [])

            print(f"  {icon} {agent.upper():8s} [{model:9s}] {st:8s}  ({ts})")
            print(f"    Last: {last}")
            print(f"    Branch: {branch}  Commit: {commit}  Tests: {tests}")
            if needs:
                print(f"    ⚠ Needs: {', '.join(needs)}")
        else:
            # No STATUS.json — check if TASK.md exists
            task = _read_task(agent)
            if task:
                print(f"  ❓ {agent.upper():8s} [{model:9s}] no status  (has task file)")
            else:
                print(f"  ⬜ {agent.upper():8s} [{model:9s}] no status, no task")
        print()

    # Bulletin
    bulletin_path = AGENTS_DIR / "BULLETIN.jsonl"
    if bulletin_path.exists():
        lines = bulletin_path.read_text().strip().split("\n")
        recent = lines[-5:] if len(lines) > 5 else lines
        if recent and recent[0]:
            print(f"  {'─'*60}")
            print(f"  BULLETIN (last {len(recent)} messages):")
            for line in recent:
                try:
                    msg = json.loads(line)
                    fr = msg.get("from", "?")
                    tp = msg.get("type", "?")
                    txt = msg.get("message", "")[:60]
                    print(f"    [{fr}] ({tp}) {txt}")
                except json.JSONDecodeError:
                    pass
            print()

def main() -> None:
    parser = argparse.ArgumentParser(description="Agent coordination dashboard")
    parser.add_argument("--watch", type=int, metavar="SECONDS",
                       help="Poll every N seconds")
    parser.add_argument("--bulletin", action="store_true",
                       help="Show full bulletin board")
    args = parser.parse_args()

    if args.bulletin:
        bulletin_path = AGENTS_DIR / "BULLETIN.jsonl"
        if bulletin_path.exists():
            print(bulletin_path.read_text())
        else:
            print("No bulletin messages yet.")
        return

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")  # clear screen
                print_dashboard()
                print(f"  Refreshing every {args.watch}s... (Ctrl+C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_dashboard()

if __name__ == "__main__":
    main()
