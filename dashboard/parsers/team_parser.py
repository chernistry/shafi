"""Team parser — reads agent STATUS.json, TASK_QUEUE.jsonl, and BULLETIN.jsonl."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _read_latest_local_eval(root: Path) -> dict:
    """Parse the most recent warmup_score_*.md file for G and Det numbers."""
    eval_dir = root / "data" / "eval"
    if not eval_dir.exists():
        return {"g": 0.379, "det_correct": 54, "det_total": 70, "det_pct": 54 / 70, "source": "hardcoded"}

    # Try JSON score files first (more reliable), sorted by mtime newest last
    json_files = sorted(eval_dir.glob("warmup_score_*.json"), key=lambda f: f.stat().st_mtime)
    if json_files:
        try:
            data = json.loads(json_files[-1].read_text())
            s = data.get("summary", {})
            g = (s.get("overall_grounding_f_beta") or s.get("grounding_fbeta")
                 or s.get("g_score") or s.get("overall_g"))
            det_pct = (s.get("overall_exact_match_rate") or s.get("exact_match_rate")
                       or s.get("det_pct"))
            det_n = s.get("exact_match_correct") or s.get("exact_match_n") or s.get("det_correct")
            det_total = (s.get("exact_match_evaluated") or s.get("exact_match_total")
                         or s.get("det_total") or 70)
            if g is not None:
                return {
                    "g": round(float(g), 4),
                    "det_correct": int(det_n) if det_n is not None else int(round(float(det_pct or 0) * det_total)),
                    "det_total": int(det_total),
                    "det_pct": round(float(det_pct or 0), 4),
                    "source": json_files[-1].name,
                }
        except Exception:
            pass

    # Fall back to parsing the latest markdown file
    md_files = sorted(eval_dir.glob("warmup_score_*.md"), key=lambda f: f.stat().st_mtime)
    if not md_files:
        return {"g": 0.379, "det_correct": 54, "det_total": 70, "det_pct": 54 / 70, "source": "hardcoded"}
    try:
        text = md_files[-1].read_text()
        g_m = re.search(r"Overall Grounding F-beta.*?:\s*([\d.]+)", text)
        det_m = re.search(r"Exact Match Rate.*?:\s*([\d.]+)\s*\((\d+)/(\d+)\)", text)
        g = float(g_m.group(1)) if g_m else 0.379
        det_correct = int(det_m.group(2)) if det_m else 54
        det_total = int(det_m.group(3)) if det_m else 70
        return {
            "g": g,
            "det_correct": det_correct,
            "det_total": det_total,
            "det_pct": round(det_correct / det_total, 4),
            "source": md_files[-1].name,
        }
    except Exception:
        return {"g": 0.379, "det_correct": 54, "det_total": 70, "det_pct": 54 / 70, "source": "hardcoded"}


def parse_team(root: Path) -> dict:
    agents_dir = root / ".sdd" / "agents"
    agents = ["franky", "smarty", "alby", "sissy", "rocky", "muffy", "papa", "benny", "cocky"]
    result = []

    for name in agents:
        agent_dir = agents_dir / name
        status_file = agent_dir / "STATUS.json"
        queue_file = agent_dir / "TASK_QUEUE.jsonl"

        # Read status
        status_raw: dict = {}
        if status_file.exists():
            try:
                status_raw = json.loads(status_file.read_text())
            except Exception:
                pass

        # Read queue
        pending, active, done = 0, 0, 0
        next_task = ""
        if queue_file.exists():
            try:
                for line in queue_file.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        s = t.get("status", "")
                        if s == "pending":
                            pending += 1
                            if not next_task:
                                next_task = f"{t.get('task_id', '')} — {t.get('description', '')[:60]}"
                        elif s == "active":
                            active += 1
                            if not next_task:
                                next_task = f"[active] {t.get('task_id', '')} — {t.get('description', '')[:50]}"
                        elif s == "done":
                            done += 1
                    except Exception:
                        pass
            except Exception:
                pass

        agent_status = status_raw.get("status", "unknown")
        is_idle = agent_status in ("idle", "pending_activation", "unknown") and pending == 0
        has_work_idle = agent_status in ("idle", "standby") and pending > 0

        result.append({
            "name": name,
            "status": agent_status,
            "is_idle": is_idle,
            "has_work_idle": has_work_idle,
            "current_task": status_raw.get("current_task") or status_raw.get("last_completed") or status_raw.get("last_action") or "",
            "progress": status_raw.get("progress") or "",
            "recent_findings": status_raw.get("recent_findings") or [],
            "timestamp": status_raw.get("timestamp") or status_raw.get("last_updated") or "",
            "pending": pending,
            "active": active,
            "done": done,
            "next_task": next_task,
            "tests_passing": status_raw.get("tests_passing"),
            "cycle": status_raw.get("cycle"),  # for papa
        })

    return result


def parse_bulletin(root: Path, limit: int = 30) -> list[dict]:
    bulletin_file = root / ".sdd" / "agents" / "BULLETIN.jsonl"
    if not bulletin_file.exists():
        return []

    entries = []
    try:
        lines = bulletin_file.read_text().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                entries.append({
                    "from": e.get("from", "?"),
                    "timestamp": e.get("timestamp", ""),
                    "type": e.get("type", ""),
                    "message": e.get("message", ""),
                })
            except Exception:
                pass
    except Exception:
        pass

    return entries[-limit:]


def parse_recent_commits(root: Path, limit: int = 12) -> list[dict]:
    try:
        result = subprocess.run(
            ["git", "log", f"--oneline", f"-{limit}"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        commits = []
        for line in result.stdout.splitlines():
            if line.strip():
                sha, _, msg = line.partition(" ")
                commits.append({"sha": sha, "message": msg})
        return commits
    except Exception:
        return []


def parse_score_state(root: Path | None = None) -> dict:
    """Local state for the score panel: timestamp and latest local eval summary."""
    if root is None:
        root = Path(__file__).resolve().parent.parent.parent
    return {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "current_best": {},
        "run_900q": {},
        "ingest": {},
        "eval_progress": {},
        "local_warmup": _read_latest_local_eval(root),
    }
