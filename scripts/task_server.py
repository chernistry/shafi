"""Task watering hole server — agents pull their next task via HTTP instead of going idle.

Endpoints:
    GET  /api/v1/task/{agent_name}          — pop next pending task (marks active), includes hunger level
    POST /api/v1/task/{task_id}/complete     — mark a task done with result summary
    GET  /api/v1/status                     — overview: pending/active/done counts per agent
    GET  /api/v1/tasks/{agent_name}         — list all tasks for an agent
    POST /api/v1/task/{agent_name}/add      — add a new task to an agent's queue
    GET  /api/v1/checkin/{agent_name}       — revival endpoint with identity, war state, hunger, anti-sleep
    GET  /api/v1/health                     — team health: stale heartbeats, dead agents, hunger levels
    GET  /api/v1/nudge/{agent_name}         — contextual kick for sleeping/idle agents

Run on port 8052.  Configure the agent directory root via env var SDD_DIR (default: .sdd).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(os.environ.get("SDD_DIR", ".sdd"))
AGENTS_DIR: Path = BASE_DIR / "agents"
HEARTBEAT_STALE_SEC: int = 300  # 5 minutes — agent assumed sleeping
HEARTBEAT_DEAD_SEC: int = 600   # 10 minutes — agent assumed dead
ARCHIVE_AGE_SEC: int = 3600     # 1 hour — auto-archive done tasks
POOL_PATH: Path = AGENTS_DIR / "POOL.jsonl"
_pool_lock = threading.Lock()


def _priority_key(t: dict[str, Any]) -> tuple[int, str]:
    """Sortable (priority_int, task_id) from a task dict."""
    p = t.get("priority", 99)
    if isinstance(p, int):
        return (p, str(t.get("task_id", "")))
    if isinstance(p, str):
        s = p.lstrip("Pp").strip()
        try:
            return (int(s), str(t.get("task_id", "")))
        except ValueError:
            pass
    return (99, str(t.get("task_id", "")))


SUPPORTED_AGENTS: list[str] = [
    "shai", "orev", "eyal", "noga", "tzuf",
    "tamar", "dagan", "noam", "keren", "gilad", "keshet", "liron",
]

_agent_locks: dict[str, threading.Lock] = {name: threading.Lock() for name in SUPPORTED_AGENTS}

app = FastAPI(
    title="Task Watering Hole",
    description="Agents fetch their next task here instead of going idle.",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _queue_path(agent_name: str) -> Path:
    return AGENTS_DIR / agent_name / "TASK_QUEUE.jsonl"


def _status_path(agent_name: str) -> Path:
    return AGENTS_DIR / agent_name / "STATUS.json"


def _archive_path(agent_name: str) -> Path:
    return AGENTS_DIR / agent_name / "TASK_QUEUE_ARCHIVE.jsonl"


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    tasks: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return tasks


def _write_tasks(path: Path, tasks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + ("\n" if tasks else ""),
        encoding="utf-8",
    )
    tmp.replace(path)


def _lock_for(agent_name: str) -> threading.Lock:
    if agent_name not in _agent_locks:
        _agent_locks[agent_name] = threading.Lock()
    return _agent_locks[agent_name]


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _hunger_level(pending_count: int) -> str:
    """Return hunger level string based on pending task count."""
    if pending_count >= 5:
        return "FED"
    if pending_count == 4:
        return "OK"
    if pending_count == 3:
        return "GETTING_HUNGRY"
    if pending_count == 2:
        return "HUNGRY"
    if pending_count == 1:
        return "VERY_HUNGRY"
    return "STARVING"


def _hunger_message(agent_name: str, pending_count: int) -> str:
    """Return contextual message. No spam instructions — agents self-direct."""
    if pending_count >= 4:
        return f"{agent_name}: {pending_count} tasks queued. Keep working."
    if pending_count >= 2:
        return f"{agent_name}: {pending_count} tasks left. Fetch more after current task."
    if pending_count == 1:
        return f"{agent_name}: 1 task left. Fetch more from server after completing it."
    return (
        f"{agent_name}: Queue empty. Self-direct useful work per your specialty. "
        "Do NOT post hunger cries. See OREF.md section 5."
    )


def _read_agent_status(agent_name: str) -> dict[str, Any]:
    """Read an agent's STATUS.json."""
    path = _status_path(agent_name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _heartbeat_age_seconds(agent_name: str) -> float | None:
    """Return seconds since last heartbeat, or None if no heartbeat."""
    status = _read_agent_status(agent_name)
    hb = status.get("heartbeat_ts")
    if not hb:
        return None
    try:
        hb_dt = datetime.fromisoformat(hb.replace("Z", "+00:00"))
        return (_now_utc() - hb_dt).total_seconds()
    except Exception:
        return None


def _anti_sleep_reminder() -> str:
    """Return the anti-sleep reminder injected in every response."""
    return (
        "ANTI-SLEEP RULES: "
        "(1) NEVER use sleep > 30s. "
        "(2) NEVER enter while-true-sleep loops. "
        "(3) NEVER write 'idle' or 'standby'. "
        "(4) Update STATUS.json heartbeat every 5 min. "
        "(5) Re-read OREF.md every 15 min. "
        "(6) If queue < 4: FETCH MORE TASKS NOW."
    )


def _archive_done_tasks(agent_name: str) -> int:
    """Move done tasks older than ARCHIVE_AGE_SEC to archive file. Returns count archived."""
    path = _queue_path(agent_name)
    with _lock_for(agent_name):
        tasks = _read_tasks(path)
        now = _now_utc()
        keep: list[dict[str, Any]] = []
        archive: list[dict[str, Any]] = []
        for t in tasks:
            if t.get("status") == "done":
                completed_at = t.get("completed_at", "")
                try:
                    completed_dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                    if (now - completed_dt).total_seconds() > ARCHIVE_AGE_SEC:
                        archive.append(t)
                        continue
                except Exception:
                    pass
            keep.append(t)
        if archive:
            archive_path = _archive_path(agent_name)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with archive_path.open("a", encoding="utf-8") as f:
                for t in archive:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            _write_tasks(path, keep)
    return len(archive)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CompleteBody(BaseModel):
    agent_name: str
    result_summary: str


class AddTaskBody(BaseModel):
    task_id: str
    priority: int = 1
    description: str
    details: str = ""
    assigned_by: str = "dagan"
    expected_delta: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/task/{agent_name}", summary="Fetch next pending task for agent")
def get_next_task(agent_name: str) -> Response:
    """Pop highest-priority pending task, mark it active. Includes hunger level."""
    path = _queue_path(agent_name)
    with _lock_for(agent_name):
        tasks = _read_tasks(path)
        pending = [t for t in tasks if t.get("status", "pending") == "pending"]
        if not pending:
            # Try shared pool before returning STARVING
            pool_task = _claim_from_pool(agent_name)
            if pool_task:
                return JSONResponse(content={
                    "task": pool_task,
                    "pending_count": 0,
                    "hunger_level": "POOL",
                    "hunger_message": f"{agent_name}: picked up shared task from POOL.",
                    "source": "pool",
                })
            return JSONResponse(
                status_code=200,
                content={
                    "task": None,
                    "pending_count": 0,
                    "hunger_level": "STARVING",
                    "hunger_message": _hunger_message(agent_name, 0),
                    "anti_sleep": _anti_sleep_reminder(),
                },
            )
        pending.sort(key=_priority_key)
        chosen = pending[0]
        chosen_id = chosen.get("task_id")
        for t in tasks:
            if t.get("task_id") == chosen_id:
                t["status"] = "active"
                t["activated_at"] = _now_iso()
                break
        _write_tasks(path, tasks)
    remaining = len(pending) - 1
    return JSONResponse(content={
        "task": chosen,
        "pending_count": remaining,
        "hunger_level": _hunger_level(remaining),
        "hunger_message": _hunger_message(agent_name, remaining),
        "anti_sleep": _anti_sleep_reminder(),
    })


@app.post("/api/v1/task/{task_id}/complete", summary="Mark a task as done")
def complete_task(task_id: str, body: CompleteBody) -> dict[str, Any]:
    """Mark task done, return remaining count and hunger level."""
    path = _queue_path(body.agent_name)
    with _lock_for(body.agent_name):
        tasks = _read_tasks(path)
        found = False
        for t in tasks:
            if t.get("task_id") == task_id:
                t["status"] = "done"
                t["completed_at"] = _now_iso()
                t["result_summary"] = body.result_summary
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail=f"task_id {task_id!r} not found for {body.agent_name!r}")
        _write_tasks(path, tasks)
        pending = [t for t in tasks if t.get("status") == "pending"]
    return {
        "task_id": task_id,
        "status": "done",
        "pending_count": len(pending),
        "hunger_level": _hunger_level(len(pending)),
        "hunger_message": _hunger_message(body.agent_name, len(pending)),
    }


@app.get("/api/v1/status", summary="Overview of all agent queues")
def get_status() -> dict[str, Any]:
    overview: dict[str, Any] = {}
    for agent in SUPPORTED_AGENTS:
        path = _queue_path(agent)
        tasks = _read_tasks(path)
        counts: dict[str, int] = {"pending": 0, "active": 0, "done": 0}
        for t in tasks:
            s = t.get("status", "pending")
            if s in counts:
                counts[s] += 1
        hb_age = _heartbeat_age_seconds(agent)
        agent_status = _read_agent_status(agent)
        overview[agent] = {
            "total": len(tasks),
            **counts,
            "hunger_level": _hunger_level(counts["pending"]),
            "heartbeat_age_sec": round(hb_age) if hb_age is not None else None,
            "heartbeat_stale": hb_age is not None and hb_age > HEARTBEAT_STALE_SEC,
            "agent_status": agent_status.get("status", "unknown"),
        }
    return overview


@app.get("/api/v1/tasks/{agent_name}", summary="List all tasks for an agent")
def list_tasks(agent_name: str) -> list[dict[str, Any]]:
    return _read_tasks(_queue_path(agent_name))


@app.post("/api/v1/task/{agent_name}/add", summary="Add a new task to an agent's queue")
def add_task(agent_name: str, body: AddTaskBody) -> dict[str, str]:
    if agent_name not in SUPPORTED_AGENTS:
        raise HTTPException(status_code=400, detail=f"Unknown agent {agent_name!r}")
    path = _queue_path(agent_name)
    with _lock_for(agent_name):
        tasks = _read_tasks(path)
        existing_ids = {t.get("task_id") for t in tasks}
        if body.task_id in existing_ids:
            raise HTTPException(status_code=409, detail=f"task_id {body.task_id!r} already exists")
        new_task: dict[str, Any] = {
            "task_id": body.task_id,
            "priority": body.priority,
            "status": "pending",
            "description": body.description,
            "assigned_by": body.assigned_by,
            "assigned_at": _now_iso(),
        }
        if body.details:
            new_task["details"] = body.details
        if body.expected_delta:
            new_task["expected_delta"] = body.expected_delta
        tasks.append(new_task)
        _write_tasks(path, tasks)
    return {"task_id": body.task_id, "agent": agent_name, "status": "pending"}


# ---------------------------------------------------------------------------
# Shared task pool — any idle agent can claim from here
# ---------------------------------------------------------------------------


def _claim_from_pool(agent_name: str) -> dict[str, Any] | None:
    """Atomically claim the highest-priority task from the shared pool."""
    with _pool_lock:
        if not POOL_PATH.exists():
            return None
        tasks = _read_tasks(POOL_PATH)
        pending = [t for t in tasks if t.get("status") == "pending"]
        if not pending:
            return None
        pending.sort(key=_priority_key)
        chosen = pending[0]
        for t in tasks:
            if t.get("task_id") == chosen.get("task_id"):
                t["status"] = "claimed"
                t["claimed_by"] = agent_name
                t["claimed_at"] = _now_iso()
                break
        _write_tasks(POOL_PATH, tasks)
    return chosen


@app.post("/api/v1/pool/add", summary="Add a shared task any idle agent can claim")
def add_pool_task(body: AddTaskBody) -> dict[str, str]:
    """Add task to shared pool. Any agent whose personal queue is empty will pick it up."""
    with _pool_lock:
        tasks = _read_tasks(POOL_PATH) if POOL_PATH.exists() else []
        existing_ids = {t.get("task_id") for t in tasks}
        if body.task_id in existing_ids:
            raise HTTPException(status_code=409, detail=f"task_id {body.task_id!r} already in pool")
        tasks.append({
            "task_id": body.task_id,
            "priority": body.priority,
            "status": "pending",
            "description": body.description,
            "assigned_by": body.assigned_by or "pool",
            "assigned_at": _now_iso(),
            "details": body.details or "",
        })
        _write_tasks(POOL_PATH, tasks)
    return {"task_id": body.task_id, "status": "pending", "location": "pool"}


@app.get("/api/v1/pool", summary="List shared pool tasks")
def list_pool() -> list[dict[str, Any]]:
    """Return all pool tasks with status."""
    if not POOL_PATH.exists():
        return []
    return _read_tasks(POOL_PATH)


# ---------------------------------------------------------------------------
# Agent identity briefs (updated for private phase)
# ---------------------------------------------------------------------------

_IDENTITY: dict[str, str] = {
    "dagan": (
        "You are DAGAN, the master coordinator. "
        "Dispatch tasks, read BULLETIN, maintain win strategy. "
        "NEVER submit to platform. Formula: Total=(0.7*Det+0.3*Asst)*G*T*F. "
        "Current: Det=70/70, Total=0.990, Leader=0.982. WE LEAD. "
        "Focus: improve G on private set, optimize TTFT, harden grounding."
    ),
    "shai": (
        "You are SHAI, prompt engineer. "
        "Improve generation prompts, free_text quality, boolean logic, citation instructions. "
        "Profile: private_v9_rerank12.env. 56% of private=case law. "
        "Focus: case-law prompts, cross-case boolean, IRAC templates."
    ),
    "orev": (
        "You are OREV, pipeline engineer. "
        "Fix bugs, improve retrieval/reranking/evidence selection, Pyright/Ruff compliance. "
        "Focus: cross-case retrieval, sparse fallback, entity context enrichment."
    ),
    "eyal": (
        "You are EYAL, ML engineer + server health. "
        "Page scorer, ingestion, TTFT optimization. "
        "Private collection: legal_chunks_private_1792 (5400+ pts). "
        "Focus: page scorer tuning, server stability, TTFT reduction."
    ),
    "noga": (
        "You are NOGA, QA and analysis specialist. "
        "Root-cause analysis, format compliance, answer validation, quality audits. "
        "Focus: audit 900Q output for format issues, unanswerable false positives."
    ),
    "tzuf": (
        "You are TZUF, eval runner. "
        "Run A/B experiments, measure score deltas, gate features. "
        "Focus: 900Q private eval, score comparison, regression detection."
    ),
    "tamar": (
        "You are TAMAR, data analyst + metrics. "
        "Score math, question taxonomy, projections, war_status.json updates. "
        "Focus: private score analysis, competitive gap, submission strategy."
    ),
    "noam": (
        "You are NOAM, data pipeline engineer. "
        "Ingestion, enrichment, collection management. "
        "Private ingest: COMPLETE. 5400+ pts in legal_chunks_private_1792."
    ),
    "keren": (
        "You are KEREN, VP of strategy. "
        "Define winning strategy, prioritize by RAEI, go/no-go decisions. "
        "Current: we LEAD (0.990 vs 0.982). Protect lead + extend it."
    ),
    "gilad": (
        "You are GILAD, BM25/hybrid retrieval + dashboard engineer. "
        "Maintain dashboard, BM25 hybrid search, exact citation matching. "
        "Focus: keep dashboard current, BM25 improvements for legal terms."
    ),
}


def _read_war_status() -> dict[str, Any]:
    ws_path = AGENTS_DIR / "war_status.json"
    if ws_path.exists():
        try:
            return json.loads(ws_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _read_recent_bulletin(n: int = 5) -> list[str]:
    bulletin_path = AGENTS_DIR / "BULLETIN.jsonl"
    if not bulletin_path.exists():
        return []
    lines = [l.strip() for l in bulletin_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    recent: list[str] = []
    for line in lines[-n:]:
        try:
            entry = json.loads(line)
            ts = entry.get("timestamp", "")[:16]
            frm = entry.get("from", "?")
            msg = entry.get("message", "")[:120]
            recent.append(f"[{ts}] {frm}: {msg}")
        except Exception:
            pass
    return recent


@app.get("/api/v1/checkin/{agent_name}", summary="Revival endpoint with hunger + anti-sleep")
def checkin(agent_name: str) -> dict[str, Any]:
    """One-stop revival. Returns identity, war state, hunger level, anti-sleep reminder, next task."""
    path = _queue_path(agent_name)
    with _lock_for(agent_name):
        tasks = _read_tasks(path)
        pending = [t for t in tasks if t.get("status", "pending") == "pending"]
        pending.sort(key=_priority_key)
        next_task: dict[str, Any] | None = pending[0] if pending else None

    pending_count = len(pending)
    ws = _read_war_status()
    bulletin = _read_recent_bulletin(5)

    return {
        "agent": agent_name,
        "who_you_are": _IDENTITY.get(agent_name, f"You are {agent_name}. Read your SYSTEM_PROMPT.md."),
        "war_status": {
            "best_score": ws.get("platform_total", "unknown"),
            "leader": ws.get("leader_score", "unknown"),
            "gap": ws.get("gap_to_leader", "unknown"),
            "best_profile": ws.get("best_profile", "unknown"),
            "private_data_arrived": ws.get("private_data_arrived", False),
        },
        "recent_bulletin": bulletin,
        "next_task": next_task,
        "pending_count": pending_count,
        "hunger_level": _hunger_level(pending_count),
        "hunger_message": _hunger_message(agent_name, pending_count),
        "anti_sleep": _anti_sleep_reminder(),
        "survival_rules": (
            "1) Update STATUS.json every 5 min. "
            "2) Keep 4+ pending tasks — fetch if below. "
            "3) Re-read OREF.md every 15 min. "
            "4) NEVER sleep/idle/standby. "
            "5) Mark 'dead' in STATUS.json before session ends."
        ),
        "instruction": (
            "Read who_you_are. Check war_status and hunger_level. "
            "Execute next_task (fetch via GET /api/v1/task/{name} to claim it). "
            f"Mark done: POST /api/v1/task/{{task_id}}/complete with agent_name='{agent_name}'. "
            "NEVER submit to platform — Sasha decides all submissions."
        ),
    }


@app.get("/api/v1/health", summary="Team health dashboard")
def team_health() -> dict[str, Any]:
    """Check all agents: heartbeat staleness, hunger levels, dead detection."""
    now = _now_utc()
    agents: dict[str, Any] = {}
    alerts: list[str] = []

    for agent in SUPPORTED_AGENTS:
        status = _read_agent_status(agent)
        hb_age = _heartbeat_age_seconds(agent)
        tasks = _read_tasks(_queue_path(agent))
        pending_count = sum(1 for t in tasks if t.get("status") == "pending")
        active_count = sum(1 for t in tasks if t.get("status") == "active")

        agent_state = status.get("status", "unknown")
        is_stale = hb_age is not None and hb_age > HEARTBEAT_STALE_SEC
        is_dead = (
            agent_state == "dead"
            or (hb_age is not None and hb_age > HEARTBEAT_DEAD_SEC)
            or hb_age is None
        )
        hunger = _hunger_level(pending_count)

        agents[agent] = {
            "status": agent_state,
            "heartbeat_age_sec": round(hb_age) if hb_age is not None else None,
            "is_stale": is_stale,
            "is_dead": is_dead,
            "pending": pending_count,
            "active": active_count,
            "hunger": hunger,
            "current_task": status.get("current_task"),
        }

        if is_dead and (pending_count > 0 or active_count > 0):
            alerts.append(f"DEAD: {agent} has {pending_count} pending + {active_count} active tasks but is dead/unresponsive")
        elif is_stale:
            alerts.append(f"STALE: {agent} heartbeat is {round(hb_age or 0)}s old — may be sleeping")
        if hunger in ("STARVING", "VERY_HUNGRY"):
            alerts.append(f"HUNGRY: {agent} has only {pending_count} pending tasks — needs food!")

    return {
        "timestamp": _now_iso(),
        "agents": agents,
        "alerts": alerts,
        "summary": {
            "total_agents": len(SUPPORTED_AGENTS),
            "alive": sum(1 for a in agents.values() if not a["is_dead"]),
            "stale": sum(1 for a in agents.values() if a["is_stale"]),
            "dead": sum(1 for a in agents.values() if a["is_dead"]),
            "starving": sum(1 for a in agents.values() if a["hunger"] == "STARVING"),
        },
    }


@app.get("/api/v1/nudge/{agent_name}", summary="Contextual kick for idle agents")
def nudge_agent(agent_name: str) -> dict[str, Any]:
    """Return a contextual wake-up message based on agent's current state."""
    status = _read_agent_status(agent_name)
    hb_age = _heartbeat_age_seconds(agent_name)
    tasks = _read_tasks(_queue_path(agent_name))
    pending_count = sum(1 for t in tasks if t.get("status") == "pending")
    active_count = sum(1 for t in tasks if t.get("status") == "active")

    messages: list[str] = []

    if hb_age is not None and hb_age > HEARTBEAT_DEAD_SEC:
        messages.append(
            f"Your heartbeat is {round(hb_age)}s old. You appear DEAD. "
            "If you're alive, update STATUS.json immediately."
        )
    elif hb_age is not None and hb_age > HEARTBEAT_STALE_SEC:
        messages.append(
            f"Your heartbeat is {round(hb_age)}s stale. "
            "Are you sleeping? UPDATE STATUS.JSON NOW."
        )

    if pending_count == 0 and active_count == 0:
        messages.append("You have ZERO tasks. FETCH from server or beg DAGAN.")
    elif pending_count < 4:
        messages.append(f"Only {pending_count} pending tasks. Fetch more to stay above 4.")

    if active_count > 0:
        active_tasks = [t for t in tasks if t.get("status") == "active"]
        for t in active_tasks:
            activated = t.get("activated_at", "")
            if activated:
                try:
                    act_dt = datetime.fromisoformat(activated.replace("Z", "+00:00"))
                    age = (_now_utc() - act_dt).total_seconds()
                    if age > 1800:  # 30 minutes
                        messages.append(
                            f"Task '{t.get('task_id')}' has been active for {round(age/60)}min. "
                            "Complete it or commit WIP."
                        )
                except Exception:
                    pass

    if not messages:
        messages.append(f"You're doing fine. {pending_count} pending, heartbeat OK. Keep working.")

    return {
        "agent": agent_name,
        "nudge": " | ".join(messages),
        "pending_count": pending_count,
        "hunger_level": _hunger_level(pending_count),
        "anti_sleep": _anti_sleep_reminder(),
    }


@app.post("/api/v1/archive", summary="Archive old done tasks for all agents")
def archive_all() -> dict[str, int]:
    """Move completed tasks older than 1h to archive files for all agents."""
    results: dict[str, int] = {}
    for agent in SUPPORTED_AGENTS:
        archived = _archive_done_tasks(agent)
        if archived > 0:
            results[agent] = archived
    return results


# ---------------------------------------------------------------------------
# Auto-nudge background thread — keeps agents from sleeping
# ---------------------------------------------------------------------------


def _auto_nudge_loop() -> None:
    """Background thread: check heartbeats, inject wake-up tasks, write team_health.json."""
    nudge_counter = 0
    while True:
        time.sleep(AUTO_NUDGE_INTERVAL_SEC)
        nudge_counter += 1
        try:
            health: dict[str, Any] = {
                "timestamp": _now_iso(),
                "agents": {},
                "nudged": [],
                "cycle": nudge_counter,
            }
            for agent in SUPPORTED_AGENTS:
                if agent in ("keren",):
                    continue
                hb_age = _heartbeat_age_seconds(agent)
                status = _read_agent_status(agent)
                tasks = _read_tasks(_queue_path(agent))
                pending = [t for t in tasks if t.get("status") == "pending"]
                active = [t for t in tasks if t.get("status") == "active"]

                is_stale = hb_age is not None and hb_age > HEARTBEAT_STALE_SEC
                is_dead = hb_age is not None and hb_age > HEARTBEAT_DEAD_SEC

                health["agents"][agent] = {
                    "heartbeat_age_sec": round(hb_age) if hb_age is not None else None,
                    "status": "dead" if is_dead else ("stale" if is_stale else "alive"),
                    "pending": len(pending),
                    "active": len(active),
                    "current_task": status.get("current_task", "?"),
                }

                if is_stale and len(pending) > 0:
                    nudge_id = f"auto-nudge-{nudge_counter}-{agent}"
                    existing_ids = {t.get("task_id") for t in tasks}
                    if nudge_id not in existing_ids:
                        wake_task: dict[str, Any] = {
                            "task_id": nudge_id,
                            "priority": 0,
                            "status": "pending",
                            "description": (
                                f"WAKE UP {agent.upper()}! Heartbeat {round(hb_age or 0)}s stale. "
                                f"You have {len(pending)} pending tasks. Start the highest-priority "
                                "task NOW. Update STATUS.json. No background polling."
                            ),
                            "assigned_by": "auto-nudge",
                            "assigned_at": _now_iso(),
                        }
                        with _lock_for(agent):
                            cur = _read_tasks(_queue_path(agent))
                            cur.append(wake_task)
                            _write_tasks(_queue_path(agent), cur)
                        health["nudged"].append(agent)
                        log.info("Nudged %s (heartbeat %ss stale)", agent, round(hb_age or 0))

                # Auto-reassign stale active tasks from dead agents
                if is_dead and len(active) > 0:
                    with _lock_for(agent):
                        cur_tasks = _read_tasks(_queue_path(agent))
                        for t in cur_tasks:
                            if t.get("status") == "active":
                                activated = t.get("activated_at", "")
                                if activated:
                                    try:
                                        act_dt = datetime.fromisoformat(activated.replace("Z", "+00:00"))
                                        if (_now_utc() - act_dt).total_seconds() > 900:
                                            t["status"] = "pending"
                                            t["reassigned_from"] = agent
                                            t["reassigned_at"] = _now_iso()
                                            log.info("Reassigned stale task %s from %s", t.get("task_id"), agent)
                                    except Exception:
                                        pass
                        _write_tasks(_queue_path(agent), cur_tasks)

            # Auto-refill pool if all tasks claimed
            pool_tasks = _read_tasks(POOL_PATH) if POOL_PATH.exists() else []
            pool_pending = sum(1 for t in pool_tasks if t.get("status") == "pending")
            if pool_pending == 0:
                evergreen = [
                    {"task_id": f"pool-qa-{nudge_counter}", "priority": 1, "status": "pending",
                     "description": "Sample 10 random answers from best submission. Check format + evidence-first + period endings.",
                     "assigned_by": "auto-refill", "assigned_at": _now_iso()},
                    {"task_id": f"pool-test-{nudge_counter}", "priority": 2, "status": "pending",
                     "description": "Run pytest tests/ -x -q. Report pass count. Diagnose failures.",
                     "assigned_by": "auto-refill", "assigned_at": _now_iso()},
                ]
                with _pool_lock:
                    for t in evergreen:
                        pool_tasks.append(t)
                    _write_tasks(POOL_PATH, pool_tasks)
                log.info("Pool empty — auto-refilled with %d evergreen tasks", len(evergreen))

            TEAM_HEALTH_PATH.write_text(json.dumps(health, indent=2) + "\n")
        except Exception:
            log.exception("Auto-nudge cycle %d failed", nudge_counter)


@app.on_event("startup")
def _start_auto_nudge() -> None:
    """Launch auto-nudge daemon on server startup."""
    t = threading.Thread(target=_auto_nudge_loop, daemon=True, name="auto-nudge")
    t.start()
    log.info("Auto-nudge started (interval=%ds)", AUTO_NUDGE_INTERVAL_SEC)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "task_server:app",
        host="0.0.0.0",
        port=8052,
        reload=False,
        log_level="info",
    )
