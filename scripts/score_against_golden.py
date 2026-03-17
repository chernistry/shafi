#!/usr/bin/env python3
"""Score pipeline raw results against golden_labels_v2.json.

Computes:
  - Deterministic exact match for boolean/number/date/name/names
  - Page grounding F-beta (beta=2.5) for used_page_ids vs golden_page_ids
  - Per-confidence-tier and per-answer-type breakdowns
  - Overall estimated S and G scores

Usage:
    python scripts/score_against_golden.py \
        --raw-results platform_runs/warmup/raw_results_hybrid_v1.3.json \
        --golden .sdd/golden/synthetic-ai-generated/golden_labels_v2.json \
        [--out-json /tmp/golden_score.json] \
        [--out-md /tmp/golden_score.md]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

BETA = 2.5


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _extract_used_page_ids(telemetry: dict[str, Any]) -> list[str]:
    pages = _coerce_str_list(telemetry.get("used_page_ids"))
    if pages:
        return pages
    pages = _coerce_str_list(telemetry.get("cited_page_ids"))
    if pages:
        return pages
    chunk_ids = _coerce_str_list(telemetry.get("retrieved_chunk_ids"))
    seen: set[str] = set()
    out: list[str] = []
    for cid in chunk_ids:
        parts = cid.split(":")
        if len(parts) < 2:
            continue
        doc_id = parts[0].strip()
        page_raw = parts[1].strip()
        if not doc_id or not page_raw.isdigit():
            continue
        pid = f"{doc_id}_{int(page_raw) + 1}"
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _fbeta(predicted: set[str], gold: set[str], beta: float = BETA) -> tuple[float, float, float]:
    if not gold:
        return (1.0, 1.0, 1.0) if not predicted else (0.0, 1.0, 0.0)
    if not predicted:
        return (0.0, 0.0, 0.0)
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return (0.0, precision, recall)
    b2 = beta * beta
    score = (1 + b2) * precision * recall / (b2 * precision + recall)
    return (score, precision, recall)


def _normalize_answer(answer: Any, answer_type: str) -> Any:
    if answer is None or (isinstance(answer, str) and answer.strip().lower() in ("null", "none", "")):
        return None

    if answer_type == "boolean":
        if isinstance(answer, bool):
            return answer
        s = str(answer).strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
        return None

    if answer_type == "number":
        if isinstance(answer, (int, float)) and not isinstance(answer, bool):
            return float(answer)
        s = str(answer).strip().replace(",", "").replace(" ", "")
        s = re.sub(r"[^\d.\-]", "", s)
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    if answer_type == "date":
        s = str(answer).strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return date.fromisoformat(s) if fmt == "%Y-%m-%d" else None
            except (ValueError, TypeError):
                pass
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        return s.lower()

    if answer_type in ("name", "names"):
        s = str(answer).strip()
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    return str(answer).strip()


def _answers_match(our_answer: Any, golden_answer: Any, answer_type: str) -> bool | None:
    """Returns True/False for deterministic types, None for free_text (needs semantic eval)."""
    if answer_type == "free_text":
        return None

    norm_ours = _normalize_answer(our_answer, answer_type)
    norm_gold = _normalize_answer(golden_answer, answer_type)

    if norm_ours is None and norm_gold is None:
        return True
    if norm_ours is None or norm_gold is None:
        return False

    if answer_type == "number":
        if norm_ours == 0 and norm_gold == 0:
            return True
        return math.isclose(norm_ours, norm_gold, rel_tol=1e-3)

    return norm_ours == norm_gold


def score(raw_results_path: Path, golden_path: Path) -> dict[str, Any]:
    raw_results = json.loads(raw_results_path.read_text(encoding="utf-8"))
    golden_labels = json.loads(golden_path.read_text(encoding="utf-8"))

    gold_by_qid = {str(g["question_id"]): g for g in golden_labels}

    results_by_qid: dict[str, dict[str, Any]] = {}
    for r in raw_results:
        case = r.get("case", {})
        qid = str(case.get("case_id") or "").strip()
        if not qid:
            continue
        results_by_qid[qid] = r

    per_case: list[dict[str, Any]] = []
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_confidence: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    matched_count = 0
    for qid, gold in gold_by_qid.items():
        result = results_by_qid.get(qid)
        if result is None:
            continue
        matched_count += 1

        answer_type = str(gold.get("answer_type") or "free_text")
        confidence = str(gold.get("confidence") or "unknown")
        golden_answer = gold.get("golden_answer")
        golden_pages = set(gold.get("golden_page_ids") or [])

        our_answer = result.get("answer_text")
        telemetry = result.get("telemetry", {})
        our_pages = set(_extract_used_page_ids(telemetry))

        answer_match = _answers_match(our_answer, golden_answer, answer_type)
        f_score, precision, recall = _fbeta(our_pages, golden_pages)

        case_result = {
            "question_id": qid,
            "answer_type": answer_type,
            "confidence": confidence,
            "golden_answer": golden_answer,
            "our_answer": our_answer,
            "answer_match": answer_match,
            "grounding_f_beta": round(f_score, 4),
            "grounding_precision": round(precision, 4),
            "grounding_recall": round(recall, 4),
            "our_page_count": len(our_pages),
            "gold_page_count": len(golden_pages),
            "overlap_count": len(our_pages & golden_pages),
        }
        per_case.append(case_result)

        by_type[answer_type]["f_beta"].append(f_score)
        by_confidence[confidence]["f_beta"].append(f_score)

        if answer_match is not None:
            by_type[answer_type]["exact_match"].append(1.0 if answer_match else 0.0)
            by_confidence[confidence]["exact_match"].append(1.0 if answer_match else 0.0)

    def _agg(vals: list[float]) -> dict[str, Any]:
        if not vals:
            return {"mean": 0.0, "count": 0}
        return {"mean": round(sum(vals) / len(vals), 4), "count": len(vals)}

    type_summary = {}
    for t, metrics in sorted(by_type.items()):
        type_summary[t] = {k: _agg(v) for k, v in metrics.items()}

    conf_summary = {}
    for c, metrics in sorted(by_confidence.items()):
        conf_summary[c] = {k: _agg(v) for k, v in metrics.items()}

    all_f = [c["grounding_f_beta"] for c in per_case]
    all_exact = [c["answer_match"] for c in per_case if c["answer_match"] is not None]
    overall_g = sum(all_f) / len(all_f) if all_f else 0.0
    overall_exact = sum(1 for m in all_exact if m) / len(all_exact) if all_exact else 0.0

    conf_weights = {"high": 1.0, "medium": 0.5, "low": 0.25}
    weighted_f_num = sum(c["grounding_f_beta"] * conf_weights.get(c["confidence"], 0.25) for c in per_case)
    weighted_f_den = sum(conf_weights.get(c["confidence"], 0.25) for c in per_case)
    weighted_g = weighted_f_num / weighted_f_den if weighted_f_den > 0 else 0.0

    trusted_cases = [c for c in per_case if c["confidence"] == "high"]
    trusted_g = sum(c["grounding_f_beta"] for c in trusted_cases) / len(trusted_cases) if trusted_cases else 0.0
    trusted_exact = [c for c in trusted_cases if c["answer_match"] is not None]
    trusted_exact_rate = sum(1 for c in trusted_exact if c["answer_match"]) / len(trusted_exact) if trusted_exact else 0.0

    def _infer_family(q: str, atype: str) -> str:
        ql = (q or "").lower()
        if any(t in ql for t in ("jury", "parole", "miranda", "plea bargain")):
            return "unsupported_trap"
        if any(t in ql for t in ("ruling", "outcome", "order", "cost", "award", "dismiss")):
            return "outcome_costs"
        if any(t in ql for t in ("common", "compare", "both", "same")):
            return "compare"
        if any(t in ql for t in ("enact", "commence", "force")):
            return "enactment"
        if "administ" in ql:
            return "administration"
        if atype in ("boolean",):
            return "boolean"
        if atype in ("name", "names"):
            return "names"
        if atype == "number":
            return "number"
        return "free_text_other"

    by_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for c in per_case:
        q_text = gold_by_qid.get(c["question_id"], {}).get("question", "")
        family = _infer_family(q_text, c["answer_type"])
        c["family"] = family
        by_family[family]["f_beta"].append(c["grounding_f_beta"])
        if c["answer_match"] is not None:
            by_family[family]["exact_match"].append(1.0 if c["answer_match"] else 0.0)

    family_summary = {}
    for fam, metrics in sorted(by_family.items()):
        family_summary[fam] = {k: _agg(v) for k, v in metrics.items()}

    trusted_regressions = [c for c in trusted_cases if c["answer_match"] is False]

    summary = {
        "golden_count": len(gold_by_qid),
        "matched_count": matched_count,
        "unmatched_count": len(gold_by_qid) - matched_count,
        "overall_grounding_f_beta": round(overall_g, 4),
        "weighted_grounding_f_beta": round(weighted_g, 4),
        "trusted_grounding_f_beta": round(trusted_g, 4),
        "overall_exact_match_rate": round(overall_exact, 4),
        "trusted_exact_match_rate": round(trusted_exact_rate, 4),
        "exact_match_evaluated": len(all_exact),
        "exact_match_correct": sum(1 for m in all_exact if m),
        "trusted_regressions": len(trusted_regressions),
        "by_answer_type": type_summary,
        "by_confidence": conf_summary,
        "by_family": family_summary,
    }

    per_case.sort(key=lambda c: c["grounding_f_beta"])

    return {"summary": summary, "per_case": per_case}


def _render_markdown(result: dict[str, Any]) -> str:
    s = result["summary"]
    lines = [
        "# Golden Labels Scoring Report\n",
        f"- **Matched questions**: {s['matched_count']} / {s['golden_count']}",
        f"- **Unmatched**: {s['unmatched_count']}",
        f"- **Overall Grounding F-beta (beta={BETA})**: {s['overall_grounding_f_beta']}",
        f"- **Weighted Grounding F-beta**: {s.get('weighted_grounding_f_beta', '-')} (high=1.0, medium=0.5, low=0.25)",
        f"- **Trusted-only Grounding F-beta**: {s.get('trusted_grounding_f_beta', '-')} (high confidence only)",
        f"- **Exact Match Rate (deterministic types)**: {s['overall_exact_match_rate']} "
        f"({s['exact_match_correct']}/{s['exact_match_evaluated']})",
        f"- **Trusted Exact Match Rate**: {s.get('trusted_exact_match_rate', '-')}",
        f"- **Trusted Regressions**: {s.get('trusted_regressions', '-')}\n",
        "## By Answer Type\n",
        "| Type | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |",
        "|------|--------------|------------|-------------|-----------|",
    ]
    for t, metrics in sorted(s["by_answer_type"].items()):
        fb = metrics.get("f_beta", {})
        em = metrics.get("exact_match", {})
        em_mean = em.get("mean")
        em_str = f"{em_mean:.4f}" if isinstance(em_mean, (int, float)) else "-"
        lines.append(
            f"| {t} | {fb.get('mean', 0):.4f} | {fb.get('count', 0)} | "
            f"{em_str} | {em.get('count', '-')} |"
        )

    lines += [
        "\n## By Confidence Tier\n",
        "| Confidence | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |",
        "|------------|--------------|------------|-------------|-----------|",
    ]
    for c, metrics in sorted(s["by_confidence"].items()):
        fb = metrics.get("f_beta", {})
        em = metrics.get("exact_match", {})
        em_mean_c = em.get("mean")
        em_str_c = f"{em_mean_c:.4f}" if isinstance(em_mean_c, (int, float)) else "-"
        lines.append(
            f"| {c} | {fb.get('mean', 0):.4f} | {fb.get('count', 0)} | "
            f"{em_str_c} | {em.get('count', '-')} |"
        )

    fam_summary = s.get("by_family", {})
    if fam_summary:
        lines += [
            "\n## By Question Family\n",
            "| Family | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |",
            "|--------|--------------|------------|-------------|-----------|",
        ]
        for fam, metrics in sorted(fam_summary.items()):
            fb = metrics.get("f_beta", {})
            em = metrics.get("exact_match", {})
            em_mean_f = em.get("mean")
            em_str_f = f"{em_mean_f:.4f}" if isinstance(em_mean_f, (int, float)) else "-"
            lines.append(
                f"| {fam} | {fb.get('mean', 0):.4f} | {fb.get('count', 0)} | "
                f"{em_str_f} | {em.get('count', '-')} |"
            )

    worst = [c for c in result["per_case"] if c["grounding_f_beta"] < 0.5]
    if worst:
        lines += [
            "\n## Worst Grounding Cases (F-beta < 0.5)\n",
            "| QID (short) | Type | F-beta | P | R | Our Pages | Gold Pages | Match? |",
            "|-------------|------|--------|---|---|-----------|------------|--------|",
        ]
        for c in worst[:20]:
            qid_short = c["question_id"][:8]
            lines.append(
                f"| {qid_short} | {c['answer_type']} | {c['grounding_f_beta']:.4f} | "
                f"{c['grounding_precision']:.2f} | {c['grounding_recall']:.2f} | "
                f"{c['our_page_count']} | {c['gold_page_count']} | {c.get('answer_match', '-')} |"
            )

    mismatches = [c for c in result["per_case"] if c["answer_match"] is False]
    if mismatches:
        lines += [
            "\n## Answer Mismatches\n",
            "| QID (short) | Type | Golden | Ours |",
            "|-------------|------|--------|------|",
        ]
        for c in mismatches:
            qid_short = c["question_id"][:8]
            g = str(c["golden_answer"])[:60]
            o = str(c["our_answer"])[:60]
            lines.append(f"| {qid_short} | {c['answer_type']} | {g} | {o} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw results against golden labels")
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    result = score(args.raw_results, args.golden)

    md = _render_markdown(result)
    print(md)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        print(f"\nJSON written to {args.out_json}")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
        print(f"Markdown written to {args.out_md}")

    sys.exit(0)


if __name__ == "__main__":
    main()
