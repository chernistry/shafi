"""Parse data/competition_matrix.json for leaderboard data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_matrix(data_dir: Path) -> dict | None:
    """Load competition matrix and return leaderboard summary."""
    f = data_dir / "competition_matrix.json"
    if not f.exists():
        return None
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Cannot parse competition_matrix.json")
        return None

    lb = raw.get("leaderboard_summary", {})
    summary = raw.get("summary", {})

    return {
        "team": lb.get("team_name", summary.get("team", "")),
        "rank": lb.get("rank"),
        "total": lb.get("total"),
        "s": lb.get("s") or summary.get("current_s"),
        "g": lb.get("g") or summary.get("current_g"),
        "t": lb.get("t"),
        "f": lb.get("f"),
        "latency_ms": lb.get("latency_ms"),
        "submissions": lb.get("submissions"),
        "submissions_remaining": summary.get("submissions_remaining"),
        "gap_targets": raw.get("gap_targets", []),
    }


def parse_scores_timeline(data_dir: Path, researches_dir: Path) -> list[dict]:
    """Build G/S/T/F timeline from platform_scoring_*.json files in researches."""
    results = []

    # Scan researches for platform scoring files
    if researches_dir.is_dir():
        for f in researches_dir.rglob("platform_scoring_*.json"):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            lb = raw.get("leaderboard_summary", {})
            if not lb:
                continue

            # Try to get a date from the parent dir name or file
            from .timestamp import parse_iso_date, parse_filename_timestamp
            ts = parse_filename_timestamp(f.name) or parse_iso_date(f.parent.name)
            if ts is None:
                continue

            label = f.parent.name if f.parent != researches_dir else f.stem
            results.append({
                "timestamp": ts.isoformat(),
                "label": label.replace("_", " "),
                "total": lb.get("total"),
                "s": lb.get("s"),
                "g": lb.get("g"),
                "t": lb.get("t"),
                "f": lb.get("f"),
            })

    results.sort(key=lambda e: e["timestamp"])
    return results
