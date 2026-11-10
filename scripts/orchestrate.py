#!/usr/bin/env python3
"""
orchestrate.py — Lightweight multi-agent orchestrator for Team Tzur Labs.

This implements the core AutoGen/Swarms "handoff" pattern without external deps:
  - Reads each agent's TASK_QUEUE.jsonl
  - For idle agents with pending tasks: spawns with task context injected
  - Monitors heartbeats and respawns dead agents
  - Supports sequential task chaining and dependency resolution

This is our "AutoGen light" — same pattern, zero new dependencies.

Usage:
    python scripts/orchestrate.py                    # single dispatch cycle
    python scripts/orchestrate.py --watch            # continuous orchestration
    python scripts/orchestrate.py --agent tzuf      # only orchestrate tzuf
    python scripts/orchestrate.py --dry-run          # show what would be spawned
    python scripts/orchestrate.py --spawn-dagan       # only start dagan
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

# ── Config ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / ".sdd" / "agents"
SPAWN_TASK_SCRIPT = REPO_ROOT / "scripts" / "spawn_with_task.sh"
SPAWN_SCRIPT = REPO_ROOT / "scripts" / "spawn_agent.sh"
ORCH_LOG = AGENTS_DIR / "orchestrator.log"
ORCH_STATUS = AGENTS_DIR / "orchestrator_status.json"

AGENTS = ["shai", "orev", "eyal", "noga", "tzuf", "tamar", "dagan"]
MODEL = "sonnet"
WATCH_INTERVAL = 90   # seconds between orchestration cycles in --watch mode
HEARTBEAT_STALE = 480  # seconds — agent assumed dead if no heartbeat
SPAWN_COOLDOWN = 150   # seconds — don't respawn same agent twice quickly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ORCH] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(ORCH_LOG)),
    ],
)
log = logging.getLogger(__name__)

_last_spawn: dict[str, float] = {}


# ── Queue helpers ──────────────────────────────────────────────────────────────

def _read_queue(agent: str) -> list[dict]:
    f = AGENTS_DIR / agent / "TASK_QUEUE.jsonl"
    if not f.exists():
        return []
    tasks = []
    try:
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    except Exception:
        pass
    return tasks


def _pending(agent: str) -> list[dict]:
    return [t for t in _read_queue(agent) if t.get("status") == "pending"]


def _active(agent: str) -> list[dict]:
    return [t for t in _read_queue(agent) if t.get("status") == "active"]


def _needs_work(agent: str) -> bool:
    q = _read_queue(agent)
    return any(t.get("status") in ("pending", "active") for t in q)


def _heartbeat_age(agent: str) -> float:
    f = AGENTS_DIR / agent / "STATUS.json"
    if not f.exists():
        return float("inf")
    try:
        s = json.loads(f.read_text())
        ts = (s.get("heartbeat_ts") or s.get("timestamp") or "").replace("Z", "+00:00")
        if not ts:
            return float("inf")
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
    except Exception:
        return float("inf")


def _agent_status(agent: str) -> str:
    f = AGENTS_DIR / agent / "STATUS.json"
    if not f.exists():
        return "unknown"
    try:
        return json.loads(f.read_text()).get("status", "unknown")
    except Exception:
        return "unknown"


def _in_cooldown(agent: str) -> bool:
    return (time.monotonic() - _last_spawn.get(agent, 0.0)) < SPAWN_COOLDOWN


# ── Spawn ──────────────────────────────────────────────────────────────────────

def _spawn_agent(agent: str, task: dict | None = None, dry_run: bool = False) -> bool:
    """Spawn agent with task-injected prompt (AutoGen handoff) or fallback to generic."""
    if task and SPAWN_TASK_SCRIPT.exists():
        script = SPAWN_TASK_SCRIPT
        cmd = [str(script), agent, task.get("task_id", "auto"), MODEL]
        tag = f"task={task['task_id']}: {task.get('description','')[:60]}"
    else:
        script = SPAWN_SCRIPT
        cmd = [str(script), agent, MODEL]
        tag = "(generic)"

    if dry_run:
        log.info("[DRY-RUN] %s ← %s", agent.upper(), tag)
        return True

    log.info("Dispatching %s ← %s", agent.upper(), tag)
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            _last_spawn[agent] = time.monotonic()
            return True
        log.error("Spawn %s failed: %s", agent, result.stderr[-300:])
        return False
    except subprocess.TimeoutExpired:
        _last_spawn[agent] = time.monotonic()
        return True
    except Exception as e:
        log.error("Spawn %s error: %s", agent, e)
        return False


# ── Core dispatch cycle ─────────────────────────────────────────────────────────

def dispatch_cycle(
    agents: list[str] = AGENTS,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict]:
    """
    One orchestration cycle — the AutoGen GroupManager equivalent.

    For each agent:
    - No pending tasks → idle, skip
    - Has pending tasks + live heartbeat → already running, skip
    - Has pending tasks + stale heartbeat → dead/sleeping → SPAWN with task context
    - In cooldown → skip (prevent rapid respawn)
    """
    log.info("── Orchestration cycle (%d agents) ──────────────────", len(agents))
    results = []

    for agent in agents:
        pending = _pending(agent)
        active_tasks = _active(agent)
        work = pending or active_tasks
        age = _heartbeat_age(agent)
        stale = age > HEARTBEAT_STALE
        cooldown = _in_cooldown(agent)
        status = _agent_status(agent)

        if not work:
            log.info("  %-8s idle (no pending tasks)", agent)
            results.append({"agent": agent, "action": "idle", "reason": "no tasks"})
            continue

        if not stale and not force:
            log.info("  %-8s running (heartbeat %ds ago, %d pending)", agent, int(age), len(pending))
            results.append({"agent": agent, "action": "running", "heartbeat_age": age})
            continue

        if cooldown and not force:
            log.info("  %-8s cooldown (respawned recently)", agent)
            results.append({"agent": agent, "action": "cooldown"})
            continue

        # Dead/sleeping agent with work → dispatch with handoff
        task = pending[0] if pending else active_tasks[0]
        reason = f"stale heartbeat ({age:.0f}s), status={status}, {len(pending)} pending"
        log.warning("  %-8s DISPATCHING — %s", agent.upper(), reason)

        spawned = _spawn_agent(agent, task=task, dry_run=dry_run)
        results.append({
            "agent": agent,
            "action": "spawned" if spawned else "spawn_failed",
            "task": task.get("task_id"),
            "reason": reason,
        })

    # Write orchestrator status
    ORCH_STATUS.write_text(json.dumps({
        "last_cycle": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }, indent=2))

    spawned_count = sum(1 for r in results if r["action"] == "spawned")
    log.info("Cycle complete: %d/%d agents dispatched", spawned_count, len(agents))
    return results


# ── Papa-specific helper ────────────────────────────────────────────────────────

def spawn_dagan(dry_run: bool = False) -> bool:
    """Start Papa with full management prompt — Papa then dispatches other agents."""
    script = SPAWN_SCRIPT
    if not script.exists():
        log.error("spawn_agent.sh not found")
        return False
    if dry_run:
        log.info("[DRY-RUN] Would spawn DAGAN")
        return True
    log.info("Starting DAGAN (manager) ...")
    try:
        result = subprocess.run(
            [str(script), "dagan", "sonnet"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=20,
        )
        if result.returncode == 0:
            _last_spawn["dagan"] = time.monotonic()
            log.info("DAGAN started")
            return True
        log.error("DAGAN spawn failed: %s", result.stderr[-200:])
        return False
    except subprocess.TimeoutExpired:
        _last_spawn["dagan"] = time.monotonic()
        log.info("DAGAN started (timeout)")
        return True
    except Exception as e:
        log.error("DAGAN error: %s", e)
        return False


# ── Status report ──────────────────────────────────────────────────────────────

def print_status() -> None:
    """Print a quick team status table."""
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          Team Tzur Labs — Agent Status               ║")
    print("╠══════════════════════════════════════════════════════╣")
    for agent in AGENTS:
        pending = _pending(agent)
        active = _active(agent)
        age = _heartbeat_age(agent)
        status = _agent_status(agent)
        stale = age > HEARTBEAT_STALE

        age_str = f"{age:.0f}s" if age != float("inf") else "∞"
        staleness = " ⚠️ STALE" if stale and (pending or active) else ""
        next_task = pending[0].get("task_id", "?") if pending else (
            active[0].get("task_id", "?") + " [active]" if active else "—"
        )
        print(f"║ {agent.upper():<7} {status:<10} hb={age_str:<8} pending={len(pending)} next={next_task[:20]:<20}{staleness}")
    print("╚══════════════════════════════════════════════════════╝\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Team Tzur Labs agent orchestrator")
    parser.add_argument("--watch", action="store_true", help=f"Continuous mode (every {WATCH_INTERVAL}s)")
    parser.add_argument("--dry-run", action="store_true", help="Show dispatches without spawning")
    parser.add_argument("--agent", help="Only orchestrate specific agent")
    parser.add_argument("--spawn-dagan", action="store_true", help="Start Papa manager and exit")
    parser.add_argument("--status", action="store_true", help="Print status table and exit")
    parser.add_argument("--force", action="store_true", help="Spawn even if heartbeat is fresh")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.spawn_dagan:
        spawn_dagan(dry_run=args.dry_run)
        return

    agents = [args.agent] if args.agent else AGENTS
    mode = "WATCH" if args.watch else "ONCE"
    log.info("Orchestrator starting. mode=%s agents=%s dry_run=%s", mode, agents, args.dry_run)

    if args.watch:
        while True:
            try:
                dispatch_cycle(agents=agents, dry_run=args.dry_run, force=args.force)
            except Exception as e:
                log.error("Cycle error: %s", e)
            log.info("Next cycle in %ds ...", WATCH_INTERVAL)
            time.sleep(WATCH_INTERVAL)
    else:
        print_status()
        dispatch_cycle(agents=agents, dry_run=args.dry_run, force=args.force)
        print_status()


if __name__ == "__main__":
    main()
