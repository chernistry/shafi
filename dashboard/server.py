"""FastAPI dashboard server — serves static SPA + JSON API endpoints."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .parsers.benchmark_parser import parse_benchmark_timeline
from .parsers.eval_parser import parse_eval_latest, parse_eval_timeline
from .parsers.judge_parser import parse_judge_latest, parse_judge_timeline
from .parsers.matrix_parser import parse_matrix, parse_scores_timeline
from .parsers.research_parser import parse_tickets
from .parsers.team_parser import parse_bulletin, parse_recent_commits, parse_score_state, parse_team

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if ".claude" in ROOT.parts:
    ROOT = Path(*ROOT.parts[:ROOT.parts.index(".claude")])

DATA_DIR = ROOT / "data"
RESEARCHES_DIR = ROOT / ".sdd" / "researches"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="RAG Challenge Dashboard")


SEARCH_DIRS = [DATA_DIR, ROOT / "platform_runs", RESEARCHES_DIR]

_cache: dict[str, tuple[float, object]] = {}

def _max_mtime(pattern: str, directories: list[Path]) -> float:
    """Return max mtime of files matching *pattern* under *directories*."""
    max_t = 0.0
    for directory in directories:
        if not directory.exists():
            continue
        try:
            files = list(directory.rglob(pattern))
            if files:
                max_t = max(max_t, max(f.stat().st_mtime for f in files))
        except OSError:
            pass
    return max_t


def _cached(key: str, pattern: str, directories: list[Path], loader):
    """Return cached result or re-compute if any source file changed."""
    mtime = _max_mtime(pattern, directories)
    entry = _cache.get(key)
    if entry and entry[0] >= mtime and mtime > 0:
        return entry[1]
    t0 = time.monotonic()
    result = loader()
    elapsed = (time.monotonic() - t0) * 1000
    logger.info("Cache miss: %s (%.0f ms, mtime=%.0f)", key, elapsed, mtime)
    _cache[key] = (mtime, result)
    return result


def _multi_cached(key: str, specs: list[tuple[str, list[Path]]], loader):
    """Cache keyed on max mtime across multiple glob/dir pairs."""
    mtime = max(_max_mtime(pattern, dirs) for pattern, dirs in specs)
    entry = _cache.get(key)
    if entry and entry[0] >= mtime and mtime > 0:
        return entry[1]
    result = loader()
    _cache[key] = (mtime, result)
    return result


# ── API routes ───────────────────────────────────────────────────────────

@app.get("/api/eval/timeline")
def api_eval_timeline():
    data = _cached("eval_tl", "eval_*.json", SEARCH_DIRS,
                    lambda: parse_eval_timeline(SEARCH_DIRS))
    return {"data": data}


@app.get("/api/eval/latest")
def api_eval_latest():
    data = _cached("eval_latest", "eval_*.json", SEARCH_DIRS,
                    lambda: parse_eval_latest(SEARCH_DIRS))
    if data is None:
        return {"data": None, "warning": "no eval files found"}
    return {"data": data}


@app.get("/api/judge/timeline")
def api_judge_timeline():
    data = _cached("judge_tl", "judge_*.jsonl", SEARCH_DIRS,
                    lambda: parse_judge_timeline(SEARCH_DIRS))
    return {"data": data}


@app.get("/api/judge/latest")
def api_judge_latest():
    data = _cached("judge_latest", "judge_*.jsonl", SEARCH_DIRS,
                    lambda: parse_judge_latest(SEARCH_DIRS))
    if data is None:
        return {"data": None, "warning": "no judge files found"}
    return {"data": data}


@app.get("/api/benchmark/timeline")
def api_benchmark_timeline():
    data = _cached("bench_tl", "page_benchmark_*.md", SEARCH_DIRS,
                    lambda: parse_benchmark_timeline(SEARCH_DIRS))
    return {"data": data}


@app.get("/api/matrix")
def api_matrix():
    data = _cached("matrix", "competition_matrix.json", [DATA_DIR],
                    lambda: parse_matrix(DATA_DIR))
    if data is None:
        return {"data": None, "warning": "competition_matrix.json not found"}
    return {"data": data}


@app.get("/api/scores/timeline")
def api_scores_timeline():
    data = _multi_cached(
        "scores_tl",
        [("competition_matrix.json", [DATA_DIR]),
         ("platform_scoring_*.json", [RESEARCHES_DIR])],
        lambda: parse_scores_timeline(DATA_DIR, RESEARCHES_DIR),
    )
    return {"data": data}


@app.get("/api/research/tickets")
def api_research_tickets():
    data = _cached("tickets", "*", [RESEARCHES_DIR],
                    lambda: parse_tickets(RESEARCHES_DIR))
    return {"data": data}


# ── Team / Agent monitoring ───────────────────────────────────────────────

@app.get("/api/team/agents")
def api_team_agents():
    """Live agent statuses and queue depths — no caching (always fresh)."""
    return {"data": parse_team(ROOT)}


@app.get("/api/team/bulletin")
def api_team_bulletin():
    """Last 30 BULLETIN entries — cached by file mtime."""
    sdd_dir = ROOT / ".sdd" / "agents"
    data = _cached("bulletin", "BULLETIN.jsonl", [sdd_dir],
                    lambda: parse_bulletin(ROOT, limit=40))
    return {"data": data}


@app.get("/api/team/commits")
def api_team_commits():
    """Recent git commits — no caching."""
    return {"data": parse_recent_commits(ROOT, limit=15)}


@app.get("/api/team/scores")
def api_team_scores():
    """Known score state (platform + local warmup + targets)."""
    return {"data": parse_score_state(ROOT)}


_ALLOWED_AGENTS = frozenset(["franky", "smarty", "alby", "sissy", "rocky", "muffy", "papa", "iggy"])


@app.post("/api/spawn/{agent_name}")
def api_spawn_agent(agent_name: str):
    """Spawn a non-interactive Claude agent session via spawn_agent.sh."""
    if agent_name not in _ALLOWED_AGENTS:
        return {"error": f"Unknown agent '{agent_name}'"}

    queue_file = ROOT / ".sdd" / "agents" / agent_name / "TASK_QUEUE.jsonl"
    if not queue_file.exists():
        return {"error": "No task queue found"}

    spawn_script = ROOT / "scripts" / "spawn_agent.sh"
    if not spawn_script.exists():
        return {"error": "spawn_agent.sh not found"}

    try:
        result = subprocess.run(
            [str(spawn_script), agent_name, "sonnet"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {
            "status": "spawned" if result.returncode == 0 else "error",
            "agent": agent_name,
            "output": result.stdout[-500:] if result.stdout else "",
            "error": result.stderr[-200:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "spawned", "agent": agent_name, "note": "process started (timeout on wait)"}
    except Exception as e:
        return {"error": str(e)}


# ── Static files + SPA fallback ──────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/{path:path}")
def spa_fallback(path: str = ""):
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "dashboard.server:app",
        host="0.0.0.0",
        port=8050,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
    )
