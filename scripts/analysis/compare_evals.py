#!/usr/bin/env python3
"""Quick before/after comparison of two eval score JSON files.

Produces per-type G/Det breakdown, per-QID diff (improved/regressed/unchanged),
and TTFT comparison from raw eval files.

Usage:
    python scripts/compare_evals.py \\
        --baseline data/eval/warmup_score_tzuf8a_20260320.json \\
        --candidate data/eval/warmup_score_tzuf9b_gate_20260320.json \\
        [--raw-baseline data/eval/warmup_raw_20260320_180420.json] \\
        [--raw-candidate data/eval/warmup_raw_20260320_204104.json] \\
        [--golden .sdd/golden/reviewed/reviewed_all_100.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BETA = 2.5


def _load_score(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if "per_case" not in data:
        raise ValueError(f"{path} missing 'per_case' — pass a score JSON, not raw JSON")
    return data


def _case_map(score_data: dict) -> dict[str, dict]:
    """Map question_id → per_case entry."""
    return {c["question_id"]: c for c in score_data["per_case"]}


def _load_ttft(raw_path: Path | None) -> dict[str, float]:
    if not raw_path or not raw_path.exists():
        return {}
    raw = json.loads(raw_path.read_text())
    out = {}
    for r in raw:
        cid = r.get("case", {}).get("case_id")
        ttft = r.get("telemetry", {}).get("ttft_ms", 0)
        if cid:
            out[cid] = ttft
    return out


def _print_table(rows: list[list], col_widths: list[int]) -> None:
    for row in rows:
        parts = []
        for i, cell in enumerate(row):
            w = col_widths[i] if i < len(col_widths) else 12
            parts.append(str(cell).ljust(w))
        print("  ".join(parts))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two eval score JSONs")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline score JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate score JSON")
    parser.add_argument("--raw-baseline", type=Path, default=None, help="Raw eval JSON for baseline TTFT")
    parser.add_argument("--raw-candidate", type=Path, default=None, help="Raw eval JSON for candidate TTFT")
    parser.add_argument("--golden", type=Path, default=None, help="Golden labels for answer type lookup")
    args = parser.parse_args(argv)

    base = _load_score(args.baseline)
    cand = _load_score(args.candidate)
    base_map = _case_map(base)
    cand_map = _case_map(cand)
    base_ttft = _load_ttft(args.raw_baseline)
    cand_ttft = _load_ttft(args.raw_candidate)

    # Load golden for answer types
    atype_map: dict[str, str] = {}
    if args.golden and args.golden.exists():
        for item in json.loads(args.golden.read_text()):
            qid = item.get("question_id") or item.get("id")
            atype = item.get("answer_type", "?")
            if qid:
                atype_map[qid] = atype

    common = sorted(set(base_map) & set(cand_map))
    if not common:
        print("ERROR: No common question_ids between baseline and candidate")
        sys.exit(1)

    # -- Summary --
    base_s = base.get("summary", {})
    cand_s = cand.get("summary", {})
    base_G = base_s.get("overall_grounding_f_beta", 0)
    cand_G = cand_s.get("overall_grounding_f_beta", 0)
    base_det = base_s.get("exact_match_correct", 0)
    cand_det = cand_s.get("exact_match_correct", 0)
    base_det_n = base_s.get("exact_match_evaluated", 70)
    cand_det_n = cand_s.get("exact_match_evaluated", 70)

    dG = cand_G - base_G
    dDet = cand_det - base_det

    print("=" * 60)
    print(f"BASELINE:  {args.baseline.name}")
    print(f"CANDIDATE: {args.candidate.name}")
    print("=" * 60)
    print(f"\n{'METRIC':<20} {'BASELINE':>10} {'CANDIDATE':>10} {'DELTA':>10}")
    print("-" * 52)
    sign = "+" if dG >= 0 else ""
    print(f"{'G (grounding)':<20} {base_G:>10.4f} {cand_G:>10.4f} {sign+f'{dG:.4f}':>10}")
    sign = "+" if dDet >= 0 else ""
    print(f"{'Det (correct)':<20} {base_det:>10}/{base_det_n} {cand_det:>10}/{cand_det_n} {sign+str(dDet):>10}")

    # TTFT averages from raw
    if base_ttft and cand_ttft:
        base_avg_ttft = sum(base_ttft.values()) / len(base_ttft)
        cand_avg_ttft = sum(v for qid, v in cand_ttft.items()) / len(cand_ttft)
        dT = cand_avg_ttft - base_avg_ttft
        sign = "+" if dT >= 0 else ""
        print(f"{'TTFT avg (ms)':<20} {base_avg_ttft:>10.0f} {cand_avg_ttft:>10.0f} {sign+f'{dT:.0f}':>10}")
        base_over5 = sum(1 for v in base_ttft.values() if v > 5000)
        cand_over5 = sum(1 for v in cand_ttft.values() if v > 5000)
        print(f"{'TTFT >5s count':<20} {base_over5:>10} {cand_over5:>10} {cand_over5-base_over5:>+10}")

    # -- Per-type G breakdown --
    type_stats: dict[str, dict[str, list]] = {}
    det_changes: list[tuple[str, str, str, str, str]] = []
    improved: list[tuple] = []
    regressed: list[tuple] = []
    unchanged: list[tuple] = []

    for qid in common:
        bc = base_map[qid]
        cc = cand_map[qid]
        atype = atype_map.get(qid) or bc.get("answer_type") or cc.get("answer_type") or "?"

        bg = bc.get("grounding_f_beta", 0)
        cg = cc.get("grounding_f_beta", 0)
        dg = cg - bg

        if atype not in type_stats:
            type_stats[atype] = {"bg": [], "cg": []}
        type_stats[atype]["bg"].append(bg)
        type_stats[atype]["cg"].append(cg)

        # Det comparison
        b_correct = bc.get("exact_match_correct")
        c_correct = cc.get("exact_match_correct")
        if b_correct is not None and c_correct is not None and b_correct != c_correct:
            b_ans = bc.get("predicted_answer", "?")
            c_ans = cc.get("predicted_answer", "?")
            det_changes.append((qid[:8], atype, str(b_correct), str(c_correct), f"{b_ans} → {c_ans}"))

        ttft_b = base_ttft.get(qid, 0)
        ttft_c = cand_ttft.get(qid, 0)
        row = (qid[:8], atype, bg, cg, dg, ttft_b, ttft_c)
        if abs(dg) < 0.001:
            unchanged.append(row)
        elif dg > 0:
            improved.append(row)
        else:
            regressed.append(row)

    print(f"\n{'Per-type G breakdown':}")
    print(f"  {'TYPE':<14} {'N':>4} {'BASE_G':>8} {'CAND_G':>8} {'DELTA':>8}")
    print("  " + "-" * 44)
    for atype, stats in sorted(type_stats.items()):
        n = len(stats["bg"])
        bg_avg = sum(stats["bg"]) / n
        cg_avg = sum(stats["cg"]) / n
        dg_avg = cg_avg - bg_avg
        sign = "+" if dg_avg >= 0 else ""
        print(f"  {atype:<14} {n:>4} {bg_avg:>8.4f} {cg_avg:>8.4f} {sign+f'{dg_avg:.4f}':>8}")

    print(f"\nPer-QID changes: {len(improved)} improved, {len(regressed)} regressed, {len(unchanged)} unchanged")

    if regressed:
        print(f"\nREGRESSIONS (G decreased):")
        print(f"  {'QID':>8}  {'TYPE':<10} {'BASE_G':>7} {'CAND_G':>7} {'DELTA':>8}")
        for row in sorted(regressed, key=lambda x: x[4]):
            qid, atype, bg, cg, dg, _, _ = row
            print(f"  {qid:>8}  {atype:<10} {bg:>7.4f} {cg:>7.4f} {dg:>+8.4f}")

    if improved:
        print(f"\nIMPROVEMENTS (G increased):")
        print(f"  {'QID':>8}  {'TYPE':<10} {'BASE_G':>7} {'CAND_G':>7} {'DELTA':>8}")
        for row in sorted(improved, key=lambda x: -x[4]):
            qid, atype, bg, cg, dg, _, _ = row
            print(f"  {qid:>8}  {atype:<10} {bg:>7.4f} {cg:>7.4f} {dg:>+8.4f}")

    if det_changes:
        print(f"\nDet changes ({len(det_changes)}):")
        for qid, atype, bc_v, cc_v, ans_diff in det_changes:
            marker = "✓" if cc_v == "True" else "✗"
            print(f"  {marker} {qid} [{atype}] {bc_v}→{cc_v}: {ans_diff}")

    print(f"\nSUMMARY: G {'+' if dG>=0 else ''}{dG:.4f} | Det {'+' if dDet>=0 else ''}{dDet} | "
          f"{len(improved)} improved, {len(regressed)} regressed")


if __name__ == "__main__":
    main()
