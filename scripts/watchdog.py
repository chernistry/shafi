#!/usr/bin/env python3
"""
Watchdog daemon — keeps agents alive permanently.

Runs as a background process. Every POLL_INTERVAL seconds:
  1. Checks each agent's TASK_QUEUE.jsonl for pending tasks
  2. Reads STATUS.json heartbeat_ts
  3. If pending tasks + heartbeat stale → spawns agent via spawn_agent.sh

Usage:
    python scripts/watchdog.py &           # background
    python scripts/watchdog.py --once      # single check, then exit (for cron)
    python scripts/watchdog.py --dry-run   # show what would be spawned, don't spawn
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / ".sdd" / "agents"
SPAWN_SCRIPT = REPO_ROOT / "scripts" / "spawn_agent.sh"
WATCHDOG_LOG = AGENTS_DIR / "watchdog.log"
WATCHDOG_STATUS = AGENTS_DIR / "watchdog_status.json"

AGENTS = ["shai", "orev", "eyal", "noga", "tzuf", "tamar", "dagan"]
MODEL = "sonnet"

POLL_INTERVAL = 60          # seconds between full checks
HEARTBEAT_STALE_SEC = 480   # 8 min: stale heartbeat → assume dead
SPAWN_COOLDOWN_SEC = 120    # 2 min: don't respawn same agent twice in a row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(WATCHDOG_LOG)),
    ],
)
log = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────
_last_spawn: dict[str, float] = {}   # agent → monotonic time of last spawn


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pending_tasks(agent: str) -> list[dict]:
    """Return tasks that need work: 'pending' OR 'active' (may be stale/crashed)."""
    queue_file = AGENTS_DIR / agent / "TASK_QUEUE.jsonl"
    if not queue_file.exists():
        return []
    tasks = []
    try:
        for line in queue_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("status") in ("pending", "active"):
                tasks.append(t)
    except Exception:
        pass
    return tasks


def _heartbeat_age_sec(agent: str) -> float:
    """Seconds since last heartbeat_ts in STATUS.json. Returns inf if missing."""
    status_file = AGENTS_DIR / agent / "STATUS.json"
    if not status_file.exists():
        return float("inf")
    try:
        s = json.loads(status_file.read_text())
        ts_str = s.get("heartbeat_ts") or s.get("timestamp") or ""
        if not ts_str:
            return float("inf")
        # Parse ISO 8601 with Z suffix
        ts_str = ts_str.replace("Z", "+00:00")
        ts = datetime.fromisoformat(ts_str)
        now = datetime.now(timezone.utc)
        return (now - ts).total_seconds()
    except Exception:
        return float("inf")


def _agent_status(agent: str) -> str:
    status_file = AGENTS_DIR / agent / "STATUS.json"
    if not status_file.exists():
        return "unknown"
    try:
        s = json.loads(status_file.read_text())
        return s.get("status", "unknown")
    except Exception:
        return "unknown"


def _in_cooldown(agent: str) -> bool:
    last = _last_spawn.get(agent, 0.0)
    return (time.monotonic() - last) < SPAWN_COOLDOWN_SEC


def _first_pending_task(agent: str) -> dict | None:
    """Return the first pending task for an agent, or None."""
    queue_file = AGENTS_DIR / agent / "TASK_QUEUE.jsonl"
    if not queue_file.exists():
        return None
    try:
        for line in queue_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("status") == "pending":
                return t
    except Exception:
        pass
    return None


def _spawn(agent: str, dry_run: bool = False) -> bool:
    """Spawn an agent. Uses task-injected script if a pending task exists (handoff pattern)."""
    # Prefer task-injected spawn (AutoGen handoff pattern: pass task context directly)
    task_script = REPO_ROOT / "scripts" / "spawn_with_task.sh"
    fallback_script = SPAWN_SCRIPT

    pending_task = _first_pending_task(agent)
    use_task_inject = pending_task is not None and task_script.exists()
    script = task_script if use_task_inject else fallback_script

    if not script.exists():
        log.error("spawn script not found: %s", script)
        return False

    task_info = f" task={pending_task['task_id']}" if pending_task else ""
    if dry_run:
        log.info("[DRY-RUN] Would spawn %s%s via %s", agent, task_info, script.name)
        return True

    log.info("Spawning %s%s via %s ...", agent, task_info, script.name)
    cmd = [str(script), agent, MODEL] if not use_task_inject else [str(script), agent, "auto", MODEL]
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            log.info("Spawned %s OK%s", agent, task_info)
            _last_spawn[agent] = time.monotonic()
            return True
        else:
            log.error("Spawn %s failed (rc=%d): %s", agent, result.returncode, result.stderr[-200:])
            return False
    except subprocess.TimeoutExpired:
        log.info("Spawned %s (timeout — process started)%s", agent, task_info)
        _last_spawn[agent] = time.monotonic()
        return True
    except Exception as e:
        log.error("Spawn %s error: %s", agent, e)
        return False


def _write_watchdog_status(checks: list[dict]) -> None:
    WATCHDOG_STATUS.write_text(json.dumps({
        "last_check": _now_utc(),
        "agents": checks,
    }, indent=2))


def check_all(dry_run: bool = False) -> list[dict]:
    log.info("── Checking all agents ──────────────────────────────")
    results = []
    for agent in AGENTS:
        pending = _pending_tasks(agent)
        age_sec = _heartbeat_age_sec(agent)
        status = _agent_status(agent)
        stale = age_sec > HEARTBEAT_STALE_SEC
        cooldown = _in_cooldown(agent)

        action = "ok"
        spawned = False
        reason = ""

        if pending and stale and not cooldown:
            reason = (
                f"{len(pending)} pending task(s), heartbeat {age_sec:.0f}s ago "
                f"(>{HEARTBEAT_STALE_SEC}s threshold), status={status}"
            )
            log.warning("STALE: %s — %s → spawning", agent, reason)
            spawned = _spawn(agent, dry_run=dry_run)
            action = "spawned" if spawned else "spawn_failed"
        elif pending and stale and cooldown:
            reason = f"pending tasks but in cooldown ({SPAWN_COOLDOWN_SEC}s)"
            action = "cooldown"
            log.info("COOLDOWN: %s — skipping spawn", agent)
        elif pending and not stale:
            reason = f"{len(pending)} pending, heartbeat fresh ({age_sec:.0f}s ago)"
            action = "running"
            log.info("ACTIVE: %s — %s", agent, reason)
        else:
            reason = f"no pending tasks, status={status}"
            log.info("IDLE: %s — %s", agent, reason)

        results.append({
            "agent": agent,
            "pending": len(pending),
            "heartbeat_age_sec": round(age_sec, 1) if age_sec != float("inf") else -1,
            "status": status,
            "action": action,
            "spawned": spawned,
            "reason": reason,
        })

    _write_watchdog_status(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent watchdog daemon")
    parser.add_argument("--once", action="store_true", help="Single check then exit")
    parser.add_argument("--dry-run", action="store_true", help="Print would-spawn without spawning")
    args = parser.parse_args()

    log.info("Watchdog starting. POLL=%ds STALE=%ds COOLDOWN=%ds",
             POLL_INTERVAL, HEARTBEAT_STALE_SEC, SPAWN_COOLDOWN_SEC)
    log.info("Repo: %s", REPO_ROOT)
    log.info("Mode: %s", "DRY-RUN" if args.dry_run else ("ONCE" if args.once else "DAEMON"))

    if args.once:
        check_all(dry_run=args.dry_run)
        return

    while True:
        try:
            check_all(dry_run=args.dry_run)
        except Exception as e:
            log.error("check_all error: %s", e)
        log.info("Next check in %ds ...", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
