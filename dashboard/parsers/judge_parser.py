"""Parse data/judge_*.jsonl files into timeline and per-case data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .timestamp import extract_label, parse_filename_timestamp

logger = logging.getLogger(__name__)


def _parse_single_judge(path: Path) -> list[dict]:
    """Parse one JSONL file into a list of case dicts."""
    cases = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
            except json.JSONDecodeError:
                continue
            jr = obj.get("judge_result") or {}
            scores = jr.get("scores") or {}
            cases.append({
                "case_id": obj.get("case_id", ""),
                "question_id": obj.get("question_id", ""),
                "answer_type": obj.get("answer_type", ""),
                "verdict": jr.get("verdict", ""),
                "accuracy": scores.get("accuracy", 0),
                "grounding": scores.get("grounding", 0),
                "clarity": scores.get("clarity", 0),
                "uncertainty": scores.get("uncertainty_handling", 0),
            })
    except OSError:
        logger.warning("Cannot read judge file: %s", path.name)
    return cases


def parse_judge_timeline(data_dirs: list[Path]) -> list[dict]:
    """Scan judge_*.jsonl files across multiple directories and return sorted timeline of aggregate scores."""
    results = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for f in data_dir.rglob("judge_*.jsonl"):
            ts = parse_filename_timestamp(f.name)
            if ts is None:
                try:
                    ts = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                except OSError:
                    continue
            cases = _parse_single_judge(f)
            if not cases:
                continue

            n = len(cases)
            pass_count = sum(1 for c in cases if c["verdict"] == "PASS")

            # Per-type grounding averages
            by_type: dict[str, list[float]] = {}
            for c in cases:
                at = c["answer_type"]
                if at:
                    by_type.setdefault(at, []).append(c["grounding"])

            results.append({
                "timestamp": ts.isoformat(),
                "label": extract_label(f.name),
                "filename": f.name,
                "filepath": str(f),
                "n_cases": n,
                "pass_rate": round(pass_count / n, 4) if n else 0,
                "avg_accuracy": round(sum(c["accuracy"] for c in cases) / n, 2) if n else 0,
                "avg_grounding": round(sum(c["grounding"] for c in cases) / n, 2) if n else 0,
                "avg_clarity": round(sum(c["clarity"] for c in cases) / n, 2) if n else 0,
                "avg_uncertainty": round(sum(c["uncertainty"] for c in cases) / n, 2) if n else 0,
                "grounding_by_type": {
                    t: round(sum(vals) / len(vals), 2) for t, vals in by_type.items()
                },
                "count_by_type": {t: len(vals) for t, vals in by_type.items()},
            })

    results.sort(key=lambda e: e["timestamp"])
    return results


def parse_judge_latest(data_dirs: list[Path]) -> dict | None:
    """Return the latest judge result with per-case detail."""
    timeline = parse_judge_timeline(data_dirs)
    if not timeline:
        return None
    latest = timeline[-1]
    filepath = latest.get("filepath")
    if filepath:
        f = Path(filepath)
        latest["cases"] = _parse_single_judge(f)
    else:
        latest["cases"] = []
    return latest
