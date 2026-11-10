#!/usr/bin/env python3
"""Compare V16 (with OREV dedup + title boost) vs V15 baseline.

Checks gate conditions and identifies wins/losses from the retrieval changes.

Usage:
    uv run python scripts/v16_vs_v15_gate.py
    uv run python scripts/v16_vs_v15_gate.py --v16 data/tzuf_v16_full900.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V15_PATH = REPO / "data" / "tzuf_v15_full900.json"
QUESTIONS_PATH = REPO / "dataset" / "private" / "questions.json"


def load_eval(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {r["id"]: r for r in data["results"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16", type=Path, default=REPO / "data" / "tzuf_v16_full900.json")
    args = parser.parse_args()

    if not args.v16.exists():
        print(f"V16 file not found: {args.v16}")
        sys.exit(1)
    if not V15_PATH.exists():
        print(f"V15 file not found: {V15_PATH}")
        sys.exit(1)

    v15 = load_eval(V15_PATH)
    v16 = load_eval(args.v16)
    qs = {q["id"]: q for q in json.loads(QUESTIONS_PATH.read_text())}

    print(f"V15: {len(v15)} results, V16: {len(v16)} results")

    # Gate checks
    v15_nulls = sum(1 for r in v15.values() if r.get("answer") in [None, "", "null"])
    v16_nulls = sum(1 for r in v16.values() if r.get("answer") in [None, "", "null"])
    v15_nopg = sum(1 for r in v15.values() if not r.get("used_page_ids"))
    v16_nopg = sum(1 for r in v16.values() if not r.get("used_page_ids"))
    v15_ttft = sum(r.get("ttft_ms", 0) for r in v15.values()) / max(len(v15), 1)
    v16_ttft = sum(r.get("ttft_ms", 0) for r in v16.values()) / max(len(v16), 1)

    def f_coeff(ttft: float) -> float:
        if ttft < 1000:
            return 1.05
        if ttft < 2000:
            return 1.02
        if ttft < 3000:
            return 1.00
        if ttft < 5000:
            return 0.99 - (ttft - 3000) * 0.14 / 2000
        return 0.85

    v15_f = sum(f_coeff(r.get("ttft_ms", 0)) for r in v15.values()) / max(len(v15), 1)
    v16_f = sum(f_coeff(r.get("ttft_ms", 0)) for r in v16.values()) / max(len(v16), 1)

    print(f"\n{'Metric':<15} {'V15':>10} {'V16':>10} {'Delta':>10} {'Gate':>8}")
    print("-" * 55)

    null_ok = "PASS" if v16_nulls <= v15_nulls else "FAIL"
    nopg_ok = "PASS" if v16_nopg <= v15_nopg else "FAIL"
    ttft_ok = "PASS" if v16_ttft <= v15_ttft * 1.1 else "WARN"
    f_ok = "PASS" if v16_f >= v15_f * 0.99 else "FAIL"

    print(f"{'Nulls':<15} {v15_nulls:>10} {v16_nulls:>10} {v16_nulls-v15_nulls:>+10} {null_ok:>8}")
    print(f"{'No-pages':<15} {v15_nopg:>10} {v16_nopg:>10} {v16_nopg-v15_nopg:>+10} {nopg_ok:>8}")
    print(f"{'Avg TTFT':<15} {v15_ttft:>10.0f} {v16_ttft:>10.0f} {v16_ttft-v15_ttft:>+10.0f} {ttft_ok:>8}")
    print(f"{'F avg':<15} {v15_f:>10.4f} {v16_f:>10.4f} {v16_f-v15_f:>+10.4f} {f_ok:>8}")

    # Answer differences
    changed = 0
    improved = 0
    regressed = 0
    for qid in v15:
        if qid not in v16:
            continue
        v15_ans = v15[qid].get("answer")
        v16_ans = v16[qid].get("answer")
        if str(v15_ans) != str(v16_ans):
            changed += 1
            # If V15 was null and V16 is not, that's an improvement
            if v15_ans in [None, "", "null"] and v16_ans not in [None, "", "null"]:
                improved += 1
                q = qs.get(qid, {})
                print(f"\n  RECOVERED: {qid[:12]} type={q.get('answer_type','?')}")
                print(f"    V15={v15_ans} → V16={v16_ans}")
                print(f"    Q: {q.get('question','')[:80]}")
            # If V15 had answer and V16 is null, that's a regression
            elif v15_ans not in [None, "", "null"] and v16_ans in [None, "", "null"]:
                regressed += 1
                q = qs.get(qid, {})
                print(f"\n  REGRESSION: {qid[:12]} type={q.get('answer_type','?')}")
                print(f"    V15={str(v15_ans)[:40]} → V16={v16_ans}")
                print(f"    Q: {q.get('question','')[:80]}")

    print(f"\n--- Answer Changes ---")
    print(f"Changed: {changed}, Recovered: {improved}, Regressed: {regressed}")

    # Page changes
    v15_pages_total = sum(len(r.get("used_page_ids", [])) for r in v15.values())
    v16_pages_total = sum(len(r.get("used_page_ids", [])) for r in v16.values())
    print(f"\n--- Page Coverage ---")
    print(f"V15 total pages: {v15_pages_total}")
    print(f"V16 total pages: {v16_pages_total} ({v16_pages_total-v15_pages_total:+d})")

    # Verdict
    all_pass = null_ok == "PASS" and nopg_ok == "PASS" and f_ok == "PASS"
    print(f"\n{'='*55}")
    print(f"VERDICT: {'PASS — V16 is safe to build' if all_pass else 'FAIL — investigate regressions'}")


if __name__ == "__main__":
    main()
