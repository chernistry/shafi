"""Parse data/eval_*.json files into timeline data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .timestamp import extract_label, parse_filename_timestamp

logger = logging.getLogger(__name__)


def parse_eval_timeline(data_dirs: list[Path]) -> list[dict]:
    """Scan eval_*.json files and return sorted timeline entries."""
    results = []
    skipped = 0
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        patterns = ["eval_*.json", "warmup_score_*.json"]
        seen: set[Path] = set()
        for pattern in patterns:
            for f in data_dir.rglob(pattern):
                if f in seen:
                    continue
                seen.add(f)
        for f in seen:
            ts = parse_filename_timestamp(f.name)
            if ts is None:
                try:
                    ts = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                except OSError:
                    skipped += 1
                    continue
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping malformed eval file: %s", f.name)
                skipped += 1
                continue

            if not isinstance(raw, dict):
                logger.warning("Skipping non-dict eval file: %s", f.name)
                skipped += 1
                continue

            summary = raw.get("summary", {})
            # Support both old eval_*.json keys and newer warmup_score_*.json keys
            def _s(*keys):
                for k in keys:
                    v = summary.get(k)
                    if v is not None:
                        return v
                return None
            entry = {
                "timestamp": ts.isoformat(),
                "label": extract_label(f.name),
                "filename": f.name,
                "filepath": str(f),
                "total_cases": _s("total_cases", "matched_count", "golden_count") or 0,
                "g": _s("overall_grounding_f_beta", "grounding_fbeta", "g_score"),
                "det_pct": _s("overall_exact_match_rate", "exact_match_rate", "det_pct"),
                "det_correct": _s("exact_match_correct"),
                "det_total": _s("exact_match_evaluated"),
                "citation_coverage": _s("citation_coverage"),
                "doc_ref_hit_rate": _s("doc_ref_hit_rate"),
                "format_compliance": _s("answer_type_format_compliance"),
                "ttft_p50": _s("ttft_p50_ms"),
                "ttft_p95": _s("ttft_p95_ms"),
                "ttft_by_type": summary.get("ttft_by_answer_type", {}),
                "compliance_by_type": summary.get("format_compliance_by_answer_type", {}),
                "by_type": summary.get("by_answer_type", {}),
            }
            results.append(entry)

    results.sort(key=lambda e: e["timestamp"])
    return results


def parse_eval_latest(data_dirs: list[Path]) -> dict | None:
    """Return the latest eval entry with full detail."""
    timeline = parse_eval_timeline(data_dirs)
    if not timeline:
        return None
    latest = timeline[-1]

    # Also load per-case data from the file
    filepath = latest.get("filepath")
    if filepath:
        f = Path(filepath)
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            latest["cases"] = raw.get("cases", []) if isinstance(raw, dict) else []
        except (json.JSONDecodeError, OSError):
            latest["cases"] = []
    else:
        latest["cases"] = []

    # Compute delta vs previous
    if len(timeline) >= 2:
        prev = timeline[-2]
        latest["delta"] = {
            k: _safe_delta(latest.get(k), prev.get(k))
            for k in ("citation_coverage", "doc_ref_hit_rate", "format_compliance", "ttft_p50")
        }
    else:
        latest["delta"] = {}

    return latest


def _safe_delta(a, b):
    if a is not None and b is not None:
        try:
            return round(float(a) - float(b), 4)
        except (TypeError, ValueError):
            pass
    return None
