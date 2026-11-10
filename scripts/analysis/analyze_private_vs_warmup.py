#!/usr/bin/env python3
"""Compare private dataset submission against warmup baseline.

Produces a markdown report highlighting:
  - Question count and answer type distribution differences
  - Null answer rate comparison
  - Page projection statistics
  - Per-type answer length distributions

Usage:
    PYTHONPATH=src python scripts/analyze_private_vs_warmup.py \
        --private-submission platform_runs/final/submission_private.json \
        --warmup-submission platform_runs/warmup/submission_warmup.json \
        --out-dir platform_runs/final/analysis
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


def _load_submission(path: Path) -> JsonDict:
    """Load and validate a submission JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "answers" not in data:
        raise SystemExit(f"Invalid submission (no 'answers' key): {path}")
    return data


def _answer_type(answer: JsonDict) -> str:
    """Extract the answer type from telemetry, defaulting to 'unknown'."""
    return str(answer.get("telemetry", {}).get("answer_type", "unknown"))


def _used_page_count(answer: JsonDict) -> int:
    """Count used pages from telemetry."""
    pages = answer.get("telemetry", {}).get("retrieval", {}).get("used_page_ids", [])
    return len(pages) if isinstance(pages, list) else 0


def _answer_length(answer: JsonDict) -> int:
    """Return length of the answer string (0 for null answers)."""
    val = answer.get("answer")
    if val is None:
        return 0
    return len(json.dumps(val))


def _type_distribution(answers: list[JsonDict]) -> dict[str, int]:
    """Count answers by type."""
    counts: dict[str, int] = {}
    for a in answers:
        t = _answer_type(a)
        counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))


def _safe_mean(values: list[int | float]) -> float:
    """Return mean or 0.0 for empty lists."""
    return statistics.mean(values) if values else 0.0


def _safe_median(values: list[int | float]) -> float:
    """Return median or 0.0 for empty lists."""
    return statistics.median(values) if values else 0.0


def compare(
    private_sub: JsonDict,
    warmup_sub: JsonDict,
) -> str:
    """Generate comparison report as markdown string."""
    priv_answers = private_sub.get("answers", [])
    warm_answers = warmup_sub.get("answers", [])

    lines: list[str] = ["# Private vs Warmup Comparison", ""]

    # Basic counts
    lines.append("## Overview")
    lines.append("")
    lines.append(f"| Metric | Private | Warmup |")
    lines.append(f"|--------|---------|--------|")
    lines.append(f"| Total questions | {len(priv_answers)} | {len(warm_answers)} |")

    priv_null = sum(1 for a in priv_answers if a.get("answer") is None)
    warm_null = sum(1 for a in warm_answers if a.get("answer") is None)
    lines.append(f"| Null answers | {priv_null} ({100*priv_null/max(len(priv_answers),1):.1f}%) | {warm_null} ({100*warm_null/max(len(warm_answers),1):.1f}%) |")

    priv_pages = [_used_page_count(a) for a in priv_answers]
    warm_pages = [_used_page_count(a) for a in warm_answers]
    lines.append(f"| Mean used pages | {_safe_mean(priv_pages):.1f} | {_safe_mean(warm_pages):.1f} |")
    lines.append(f"| Median used pages | {_safe_median(priv_pages):.1f} | {_safe_median(warm_pages):.1f} |")
    lines.append("")

    # Type distribution
    lines.append("## Answer Type Distribution")
    lines.append("")
    priv_types = _type_distribution(priv_answers)
    warm_types = _type_distribution(warm_answers)
    all_types = sorted(set(priv_types) | set(warm_types))
    lines.append("| Type | Private | Warmup |")
    lines.append("|------|---------|--------|")
    for t in all_types:
        lines.append(f"| `{t}` | {priv_types.get(t, 0)} | {warm_types.get(t, 0)} |")
    lines.append("")

    # Per-type answer length stats
    lines.append("## Answer Length by Type")
    lines.append("")
    lines.append("| Type | Private Mean | Warmup Mean | Private Median | Warmup Median |")
    lines.append("|------|-------------|-------------|----------------|---------------|")
    for t in all_types:
        priv_lens = [_answer_length(a) for a in priv_answers if _answer_type(a) == t]
        warm_lens = [_answer_length(a) for a in warm_answers if _answer_type(a) == t]
        lines.append(
            f"| `{t}` | {_safe_mean(priv_lens):.0f} | {_safe_mean(warm_lens):.0f} "
            f"| {_safe_median(priv_lens):.0f} | {_safe_median(warm_lens):.0f} |"
        )
    lines.append("")

    # Page count distribution
    lines.append("## Used Page Count Distribution")
    lines.append("")
    lines.append("| Pages | Private | Warmup |")
    lines.append("|-------|---------|--------|")
    max_pages = max(max(priv_pages, default=0), max(warm_pages, default=0))
    for n in range(max_pages + 1):
        pc = sum(1 for p in priv_pages if p == n)
        wc = sum(1 for p in warm_pages if p == n)
        if pc > 0 or wc > 0:
            lines.append(f"| {n} | {pc} | {wc} |")
    lines.append("")

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--private-submission",
        type=Path,
        required=True,
        help="Path to private dataset submission JSON",
    )
    parser.add_argument(
        "--warmup-submission",
        type=Path,
        required=True,
        help="Path to warmup dataset submission JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for output report (default: same dir as private submission)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.private_submission.exists():
        print(f"ERROR: Private submission not found: {args.private_submission}", file=sys.stderr)
        return 1
    if not args.warmup_submission.exists():
        print(f"ERROR: Warmup submission not found: {args.warmup_submission}", file=sys.stderr)
        return 1

    private_sub = _load_submission(args.private_submission)
    warmup_sub = _load_submission(args.warmup_submission)

    report = compare(private_sub, warmup_sub)

    out_dir = args.out_dir or args.private_submission.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "private_vs_warmup_comparison.md"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
