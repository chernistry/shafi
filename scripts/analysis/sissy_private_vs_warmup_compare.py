#!/usr/bin/env python3
"""NOGA: Private eval vs warmup baseline comparison script.

Compares a private-set eval result against the warmup baseline (tzuf33a_v2_v7_full70.json)
to identify regressions, improvements, and per-type breakdowns.

Usage:
    uv run python3 scripts/noga_private_vs_warmup_compare.py <private_eval.json>

Private eval JSON format (same as tzuf33a_v2 output):
    {
        "eval_id": "...",
        "head": "...",
        "det_correct": N,
        "det_total": N,
        "results": [{"case_id": "...", "answer_type": "...", "answer": ..., "gold_answer": ..., "det_correct": bool, ...}]
    }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WARMUP_BASELINE = REPO / "data" / "tzuf33a_v2_v7_full70.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_by_type(results: list[dict]) -> dict[str, dict[str, int]]:
    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        if "error" in r:
            continue
        t = r.get("answer_type", "unknown")
        entry = by_type.setdefault(t, {"correct": 0, "total": 0})
        entry["total"] += 1
        if r.get("det_correct"):
            entry["correct"] += 1
    return by_type


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <private_eval.json>")
        print(f"Baseline: {WARMUP_BASELINE}")
        sys.exit(1)

    private_path = Path(sys.argv[1])
    if not private_path.exists():
        print(f"ERROR: {private_path} not found")
        sys.exit(1)

    warmup = load_json(WARMUP_BASELINE)
    private = load_json(private_path)

    warmup_results = {r["case_id"][:8]: r for r in warmup["results"]}
    private_results = {r["case_id"][:8]: r for r in private["results"]}

    w_correct = warmup["det_correct"]
    w_total = warmup["det_total"]
    p_correct = private["det_correct"]
    p_total = private["det_total"]

    print("=" * 60)
    print("NOGA: PRIVATE vs WARMUP COMPARISON")
    print("=" * 60)
    print(f"Warmup baseline: {WARMUP_BASELINE.name}")
    print(f"  HEAD={warmup.get('head')} profile={warmup.get('profile')}")
    print(f"  Det={w_correct}/{w_total} ({w_correct/w_total*100:.1f}%)")
    print()
    print(f"Private eval: {private_path.name}")
    print(f"  HEAD={private.get('head')} profile={private.get('profile', '?')}")
    print(f"  Det={p_correct}/{p_total} ({p_correct/p_total*100:.1f}%)")
    print()

    # Per-type breakdown
    w_by_type = summarize_by_type(warmup["results"])
    p_by_type = summarize_by_type(private["results"])
    all_types = sorted(set(w_by_type) | set(p_by_type))

    print("By type:")
    print(f"  {'Type':<10} {'Warmup':>12} {'Private':>12} {'Delta':>8}")
    for t in all_types:
        w = w_by_type.get(t, {"correct": 0, "total": 0})
        p = p_by_type.get(t, {"correct": 0, "total": 0})
        w_pct = w["correct"] / w["total"] * 100 if w["total"] else 0
        p_pct = p["correct"] / p["total"] * 100 if p["total"] else 0
        delta = p_pct - w_pct
        sign = "+" if delta >= 0 else ""
        print(f"  {t:<10} {w['correct']}/{w['total']} ({w_pct:.1f}%) {p['correct']}/{p['total']} ({p_pct:.1f}%) {sign}{delta:.1f}pp")
    print()

    # Find overlapping questions
    common_ids = set(warmup_results) & set(private_results)
    if common_ids:
        regressions = []
        improvements = []
        for qid in common_ids:
            w_r = warmup_results[qid]
            p_r = private_results[qid]
            if w_r.get("det_correct") and not p_r.get("det_correct"):
                regressions.append((qid, w_r, p_r))
            elif not w_r.get("det_correct") and p_r.get("det_correct"):
                improvements.append((qid, w_r, p_r))

        if regressions:
            print(f"REGRESSIONS ({len(regressions)}) — was correct in warmup, wrong in private:")
            for qid, w, p in regressions:
                print(f"  {qid} type={w.get('answer_type')} warmup_ans={str(w.get('answer'))[:20]!r} private_ans={str(p.get('answer'))[:20]!r} gold={w.get('gold_answer')!r}")
        else:
            print("REGRESSIONS: none ✓")
        print()

        if improvements:
            print(f"IMPROVEMENTS ({len(improvements)}) — wrong in warmup, correct in private:")
            for qid, w, p in improvements:
                print(f"  {qid} type={w.get('answer_type')} warmup_ans={str(w.get('answer'))[:20]!r} private_ans={str(p.get('answer'))[:20]!r} gold={w.get('gold_answer')!r}")
        else:
            print("IMPROVEMENTS: none")
        print()

    # Private-only failures
    private_fails = [r for r in private["results"] if not r.get("det_correct") and "error" not in r]
    print(f"All private failures ({len(private_fails)}):")
    for r in private_fails:
        qid = r["case_id"][:8]
        in_warmup = qid in warmup_results
        warmup_status = "was-correct" if in_warmup and warmup_results[qid].get("det_correct") else ("was-wrong" if in_warmup else "private-only")
        print(f"  {qid} type={r.get('answer_type')} pred={str(r.get('answer'))[:20]!r} gold={r.get('gold_answer')!r} [{warmup_status}] model={r.get('model','?')}")
    print()

    # Model usage
    from collections import Counter
    models = Counter(r.get("model", "?") for r in private["results"] if "error" not in r)
    print("Model usage in private eval:")
    for model, count in models.most_common():
        print(f"  {model}: {count}")


if __name__ == "__main__":
    main()
