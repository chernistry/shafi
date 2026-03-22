#!/usr/bin/env python3
"""V13 vs V9.1 regression gate — run when V13 checkpoint reaches 900/900.

Usage: python3 scripts/v13_vs_v91_gate.py
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V13_CHECKPOINT = REPO / "data" / "tzuf_private1_checkpoint.jsonl"
V91_SUBMISSION = REPO / "data" / "private_submission_v9_1_final.json"
V91_FALLBACK = REPO / "data" / "private_submission_v9_1_cee8dc5.json"
QUESTIONS = REPO / "dataset" / "private" / "questions.json"

NULL_TOKENS = frozenset({"null", "none", ""})


def is_null(v: object) -> bool:
    return v is None or str(v).strip().lower() in NULL_TOKENS


def load_v13() -> dict[str, dict]:
    v13: dict[str, dict] = {}
    for line in V13_CHECKPOINT.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            qid = r.get("id")
            if not qid or "error" in r:
                continue
            v13[qid] = r
        except (json.JSONDecodeError, KeyError):
            pass
    return v13


def load_v91() -> dict[str, dict]:
    path = V91_SUBMISSION if V91_SUBMISSION.exists() else V91_FALLBACK
    data = json.loads(path.read_text())
    return {a["question_id"]: a for a in data["answers"]}


def main() -> None:
    v13 = load_v13()
    v91 = load_v91()
    questions = json.loads(QUESTIONS.read_text())
    type_map = {q["id"]: q.get("answer_type", "free_text") for q in questions}

    # === METRICS ===
    nulls = sum(1 for r in v13.values() if is_null(r.get("answer")))
    nopg = sum(
        1 for r in v13.values()
        if not is_null(r.get("answer")) and not r.get("used_page_ids")
    )
    ttfts = [
        float(r.get("ttft_ms", 0) or 0)
        for r in v13.values()
        if float(r.get("ttft_ms", 0) or 0) > 0
    ]
    f_vals = [
        1.05 if t < 1000 else 1.02 if t < 2000 else 1.00 if t < 3000
        else max(0.85, 0.99 - (t - 3000) * 0.14 / 2000) if t < 5000
        else 0.85
        for t in ttfts
    ]

    print(f"=== V13 GATE ({len(v13)}/900) ===")
    print(f"Nulls: {nulls} (≤5? {'PASS' if nulls <= 5 else 'FAIL'})")
    print(f"No-pages: {nopg}")
    print(f"TTFT: avg={statistics.mean(ttfts):.0f}ms (≤1300? {'PASS' if statistics.mean(ttfts) <= 1300 else 'FAIL'})")
    print(f"F-coeff: {statistics.mean(f_vals):.4f}")
    print(f">5s: {sum(1 for t in ttfts if t > 5000)}")

    # === REGRESSIONS ===
    null_regs = []
    null_imps = []
    for qid in v13:
        if qid not in v91:
            continue
        v91_null = v91[qid].get("answer") is None
        v13_null = is_null(v13[qid].get("answer"))
        if not v91_null and v13_null:
            null_regs.append((qid, type_map.get(qid, "?"), str(v91[qid].get("answer"))[:40]))
        elif v91_null and not v13_null:
            null_imps.append((qid, type_map.get(qid, "?"), str(v13[qid].get("answer"))[:40]))

    print(f"\n=== VS V9.1 ===")
    print(f"Null regressions: {len(null_regs)} (≤10? {'PASS' if len(null_regs) <= 10 else 'FAIL'})")
    for qid, atype, ans in null_regs:
        print(f"  {qid[:16]}... [{atype}] was \"{ans}\"")
    print(f"Null improvements: {len(null_imps)}")
    for qid, atype, ans in null_imps:
        print(f"  {qid[:16]}... [{atype}] now \"{ans}\"")

    # === GATE DECISION ===
    gates = {
        "nulls_le_5": nulls <= 5,
        "ttft_le_1300": statistics.mean(ttfts) <= 1300,
        "regressions_le_10": len(null_regs) <= 10,
        "no_catastrophic": len(null_regs) <= 10,
    }
    all_pass = all(gates.values())
    net = len(null_imps) - len(null_regs)

    print(f"\n=== GATE DECISION ===")
    for gate, passed in gates.items():
        print(f"  {gate}: {'PASS' if passed else 'FAIL'}")
    print(f"Net improvement: {net:+d}")
    print(f"\nVERDICT: {'PASS — V13 is submission candidate' if all_pass else 'FAIL — V9.1 remains safer'}")


if __name__ == "__main__":
    main()
