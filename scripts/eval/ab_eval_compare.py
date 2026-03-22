#!/usr/bin/env python3
"""A/B comparison of two pipeline runs against a golden evaluation set.

Produces:
  - Per-question comparison table (sorted by largest G delta)
  - Aggregate G metrics (mean, median, G=0/1 counts)
  - Statistical significance (Wilcoxon signed-rank, bootstrap CI)
  - Regression analysis (questions where candidate is worse)
  - Config diff (from preflight summaries)

Both JSON and Markdown outputs.

Usage:
    PYTHONPATH=src python scripts/ab_eval_compare.py \
        --baseline-submission platform_runs/warmup/submission_v6.json \
        --candidate-submission platform_runs/warmup/submission_v7.json \
        --golden eval_golden_warmup.json \
        --output comparison_report
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

# ---------------------------------------------------------------------------
# G-score (F-beta 2.5) — replicates eval/harness.py logic
# ---------------------------------------------------------------------------


def g_score(predicted: set[str], gold: set[str], beta: float = 2.5) -> float:
    """Compute grounding F-beta score.

    Args:
        predicted: Set of predicted page IDs.
        gold: Set of gold page IDs.
        beta: Beta parameter (default 2.5, recall-weighted).

    Returns:
        F-beta score between 0.0 and 1.0.
    """
    if not gold:
        return 1.0 if not predicted else 0.0
    true_positives = len(predicted & gold)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(gold)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom <= 0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denom


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _answers_by_id(submission: JsonDict) -> dict[str, JsonDict]:
    return {a["question_id"]: a for a in submission.get("answers", [])}


def _used_pages_from_answer(answer: JsonDict) -> set[str]:
    """Extract used page IDs from a submission answer's telemetry."""
    tel = answer.get("telemetry", {})

    # Direct used_page_ids in telemetry (raw results format)
    used = tel.get("used_page_ids")
    if isinstance(used, list) and used:
        return set(str(p) for p in used if p)

    # Submission format: retrieval.retrieved_chunk_pages → flatten to page IDs
    retrieval = tel.get("retrieval", {})
    pages_list = retrieval.get("retrieved_chunk_pages", [])
    page_ids: set[str] = set()
    for entry in pages_list:
        if isinstance(entry, dict):
            doc_id = str(entry.get("doc_id", ""))
            for pn in entry.get("page_numbers", []):
                page_ids.add(f"{doc_id}_{pn}")
    return page_ids


def _load_golden_map(golden_path: Path) -> dict[str, set[str]]:
    """Load golden labels as {question_id: set of gold page IDs}."""
    data = _load_json(golden_path)
    result: dict[str, set[str]] = {}
    for case in data:
        qid = case["id"]
        gold_ids = set(str(g) for g in case.get("gold_chunk_ids", []) if g)
        result[qid] = gold_ids
    return result


def _load_golden_types(golden_path: Path) -> dict[str, str]:
    """Load golden answer types as {question_id: answer_type}."""
    data = _load_json(golden_path)
    return {case["id"]: case.get("answer_type", "unknown") for case in data}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _wilcoxon_approx(diffs: list[float]) -> dict[str, Any]:
    """Simplified Wilcoxon signed-rank test (without scipy dependency).

    Returns a dict with test statistic and approximate p-value using
    the normal approximation for large samples.
    """
    nonzero = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0.0]
    if len(nonzero) < 5:
        return {"statistic": None, "p_value": None, "n": len(nonzero), "note": "too few non-zero pairs"}

    ranked = sorted(nonzero, key=lambda x: x[0])
    w_plus = 0.0
    w_minus = 0.0
    for rank_idx, (_, sign) in enumerate(ranked, 1):
        if sign > 0:
            w_plus += rank_idx
        else:
            w_minus += rank_idx

    n = len(ranked)
    t_stat = min(w_plus, w_minus)
    mean_t = n * (n + 1) / 4
    std_t = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if std_t == 0:
        return {"statistic": t_stat, "p_value": 1.0, "n": n}

    z = (t_stat - mean_t) / std_t
    # Two-tailed p-value via normal CDF approximation
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    return {"statistic": round(t_stat, 4), "p_value": round(p_value, 6), "z": round(z, 4), "n": n}


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using the error function approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bootstrap_ci(
    baseline_scores: list[float],
    candidate_scores: list[float],
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap 95% CI on mean G difference (candidate - baseline)."""
    rng = random.Random(seed)
    n = len(baseline_scores)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    diffs: list[float] = []
    for _ in range(n_boot):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        boot_diff = sum(candidate_scores[i] - baseline_scores[i] for i in indices) / n
        diffs.append(boot_diff)

    diffs.sort()
    alpha = 1 - ci
    lo_idx = max(0, int(n_boot * alpha / 2))
    hi_idx = min(n_boot - 1, int(n_boot * (1 - alpha / 2)))
    mean_diff = sum(diffs) / len(diffs)
    return {
        "mean_diff": round(mean_diff, 6),
        "ci_lower": round(diffs[lo_idx], 6),
        "ci_upper": round(diffs[hi_idx], 6),
    }


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def compare(
    baseline_sub: JsonDict,
    candidate_sub: JsonDict,
    golden_map: dict[str, set[str]],
    golden_types: dict[str, str],
    baseline_preflight: JsonDict | None = None,
    candidate_preflight: JsonDict | None = None,
) -> JsonDict:
    """Run full A/B comparison and return structured results."""
    baseline_by_id = _answers_by_id(baseline_sub)
    candidate_by_id = _answers_by_id(candidate_sub)

    common_qids = sorted(set(baseline_by_id) & set(candidate_by_id) & set(golden_map))

    per_question: list[JsonDict] = []
    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    regressions: list[JsonDict] = []

    for qid in common_qids:
        gold = golden_map[qid]
        b_pages = _used_pages_from_answer(baseline_by_id[qid])
        c_pages = _used_pages_from_answer(candidate_by_id[qid])
        b_g = g_score(b_pages, gold)
        c_g = g_score(c_pages, gold)
        delta = c_g - b_g
        answer_type = golden_types.get(qid, "unknown")

        row: JsonDict = {
            "question_id": qid,
            "answer_type": answer_type,
            "baseline_G": round(b_g, 4),
            "candidate_G": round(c_g, 4),
            "delta": round(delta, 4),
            "winner": "candidate" if delta > 0 else ("baseline" if delta < 0 else "tie"),
            "baseline_pages": sorted(b_pages),
            "candidate_pages": sorted(c_pages),
            "gold_pages": sorted(gold),
        }
        per_question.append(row)
        baseline_scores.append(b_g)
        candidate_scores.append(c_g)

        if delta < -0.001:
            regressions.append(row)

    # Sort by absolute delta descending
    per_question.sort(key=lambda r: -abs(r["delta"]))
    regressions.sort(key=lambda r: r["delta"])

    # Aggregate metrics
    aggregate: JsonDict = {
        "question_count": len(common_qids),
        "baseline_mean_G": round(_mean(baseline_scores), 4),
        "candidate_mean_G": round(_mean(candidate_scores), 4),
        "baseline_median_G": round(_median(baseline_scores), 4),
        "candidate_median_G": round(_median(candidate_scores), 4),
        "baseline_G_eq_0": sum(1 for s in baseline_scores if s == 0.0),
        "candidate_G_eq_0": sum(1 for s in candidate_scores if s == 0.0),
        "baseline_G_eq_1": sum(1 for s in baseline_scores if s == 1.0),
        "candidate_G_eq_1": sum(1 for s in candidate_scores if s == 1.0),
        "candidate_wins": sum(1 for r in per_question if r["winner"] == "candidate"),
        "baseline_wins": sum(1 for r in per_question if r["winner"] == "baseline"),
        "ties": sum(1 for r in per_question if r["winner"] == "tie"),
        "regression_count": len(regressions),
    }

    # Statistical tests
    diffs = [c - b for b, c in zip(baseline_scores, candidate_scores, strict=True)]
    wilcoxon = _wilcoxon_approx(diffs)
    bootstrap = _bootstrap_ci(baseline_scores, candidate_scores)

    # Config diff
    config_diff: JsonDict | None = None
    if baseline_preflight and candidate_preflight:
        config_diff = _config_diff(baseline_preflight, candidate_preflight)

    return {
        "aggregate": aggregate,
        "statistical_tests": {
            "wilcoxon_signed_rank": wilcoxon,
            "bootstrap_95_ci": bootstrap,
        },
        "regressions": regressions,
        "per_question": per_question,
        "config_diff": config_diff,
    }


def _config_diff(base_pf: JsonDict, cand_pf: JsonDict) -> JsonDict:
    """Find keys that differ between two preflight summaries."""
    skip_keys = {"submission_sha256", "code_archive_sha256", "raw_results_path",
                 "truth_audit_workbook_path", "anomaly_report", "support_shape_report",
                 "truth_audit_report"}
    diffs: JsonDict = {}
    all_keys = sorted(set(base_pf) | set(cand_pf))
    for key in all_keys:
        if key in skip_keys:
            continue
        bv = base_pf.get(key)
        cv = cand_pf.get(key)
        if json.dumps(bv, sort_keys=True) != json.dumps(cv, sort_keys=True):
            diffs[key] = {"baseline": bv, "candidate": cv}
    return diffs


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(result: JsonDict) -> str:
    """Render comparison results as readable markdown."""
    agg = result["aggregate"]
    stats = result["statistical_tests"]
    lines: list[str] = ["# A/B Evaluation Comparison", ""]

    # Aggregate
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Delta |")
    lines.append("|--------|----------|-----------|-------|")
    lines.append(f"| Mean G | {agg['baseline_mean_G']:.4f} | {agg['candidate_mean_G']:.4f} | {agg['candidate_mean_G'] - agg['baseline_mean_G']:+.4f} |")
    lines.append(f"| Median G | {agg['baseline_median_G']:.4f} | {agg['candidate_median_G']:.4f} | {agg['candidate_median_G'] - agg['baseline_median_G']:+.4f} |")
    lines.append(f"| G=0 count | {agg['baseline_G_eq_0']} | {agg['candidate_G_eq_0']} | {agg['candidate_G_eq_0'] - agg['baseline_G_eq_0']:+d} |")
    lines.append(f"| G=1 count | {agg['baseline_G_eq_1']} | {agg['candidate_G_eq_1']} | {agg['candidate_G_eq_1'] - agg['baseline_G_eq_1']:+d} |")
    lines.append(f"| Questions | {agg['question_count']} | {agg['question_count']} | — |")
    lines.append("")
    lines.append(f"**Wins**: Candidate {agg['candidate_wins']} / Baseline {agg['baseline_wins']} / Ties {agg['ties']}")
    lines.append(f"**Regressions**: {agg['regression_count']}")
    lines.append("")

    # Statistical tests
    lines.append("## Statistical Significance")
    lines.append("")
    wil = stats["wilcoxon_signed_rank"]
    boot = stats["bootstrap_95_ci"]
    if wil["p_value"] is not None:
        sig = "YES" if wil["p_value"] < 0.05 else "no"
        lines.append(f"- Wilcoxon signed-rank: T={wil['statistic']}, z={wil.get('z', '—')}, p={wil['p_value']} (significant at 0.05: **{sig}**)")
    else:
        lines.append(f"- Wilcoxon signed-rank: {wil.get('note', 'N/A')}")
    lines.append(f"- Bootstrap 95% CI on mean G diff: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}] (mean={boot['mean_diff']:.4f})")
    lines.append("")

    # Regressions
    regressions = result["regressions"]
    if regressions:
        lines.append(f"## Regressions ({len(regressions)} questions)")
        lines.append("")
        lines.append("| question_id | type | baseline_G | candidate_G | delta | pages_changed |")
        lines.append("|-------------|------|------------|-------------|-------|---------------|")
        for r in regressions[:30]:  # cap at 30
            pages_changed = "yes" if r["baseline_pages"] != r["candidate_pages"] else "no"
            qid_short = r["question_id"][:12] + "..."
            lines.append(f"| `{qid_short}` | {r['answer_type']} | {r['baseline_G']:.3f} | {r['candidate_G']:.3f} | {r['delta']:+.3f} | {pages_changed} |")
        lines.append("")
    else:
        lines.append("## Regressions")
        lines.append("")
        lines.append("No regressions detected.")
        lines.append("")

    # Config diff
    config_diff = result.get("config_diff")
    if config_diff:
        lines.append("## Config Differences")
        lines.append("")
        if config_diff:
            for key, vals in config_diff.items():
                lines.append(f"- **{key}**: `{vals['baseline']}` → `{vals['candidate']}`")
        else:
            lines.append("No config differences detected.")
        lines.append("")

    # Top 20 per-question
    per_q = result["per_question"]
    lines.append(f"## Per-Question Comparison (top 20 by |delta|)")
    lines.append("")
    lines.append("| question_id | type | baseline_G | candidate_G | delta | winner |")
    lines.append("|-------------|------|------------|-------------|-------|--------|")
    for row in per_q[:20]:
        qid_short = row["question_id"][:12] + "..."
        lines.append(f"| `{qid_short}` | {row['answer_type']} | {row['baseline_G']:.3f} | {row['candidate_G']:.3f} | {row['delta']:+.3f} | {row['winner']} |")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline-submission", type=Path, required=True,
                        help="Path to baseline submission JSON")
    parser.add_argument("--candidate-submission", type=Path, required=True,
                        help="Path to candidate submission JSON")
    parser.add_argument("--golden", type=Path, required=True,
                        help="Path to golden evaluation labels JSON")
    parser.add_argument("--baseline-preflight", type=Path, default=None,
                        help="Optional baseline preflight JSON (for config diff)")
    parser.add_argument("--candidate-preflight", type=Path, default=None,
                        help="Optional candidate preflight JSON (for config diff)")
    parser.add_argument("--output", type=str, default="comparison_report",
                        help="Output base path (writes .json and .md)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for label, path in [("baseline", args.baseline_submission),
                        ("candidate", args.candidate_submission),
                        ("golden", args.golden)]:
        if not path.exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            return 1

    baseline_sub = _load_json(args.baseline_submission)
    candidate_sub = _load_json(args.candidate_submission)
    golden_map = _load_golden_map(args.golden)
    golden_types = _load_golden_types(args.golden)

    baseline_pf = _load_json(args.baseline_preflight) if args.baseline_preflight and args.baseline_preflight.exists() else None
    candidate_pf = _load_json(args.candidate_preflight) if args.candidate_preflight and args.candidate_preflight.exists() else None

    result = compare(baseline_sub, candidate_sub, golden_map, golden_types, baseline_pf, candidate_pf)

    # Write outputs
    out_json = Path(args.output + ".json")
    out_md = Path(args.output + ".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(result), encoding="utf-8")

    # Print summary
    agg = result["aggregate"]
    mean_delta = agg["candidate_mean_G"] - agg["baseline_mean_G"]
    print(f"A/B Comparison: {agg['question_count']} questions")
    print(f"  Mean G: {agg['baseline_mean_G']:.4f} → {agg['candidate_mean_G']:.4f} ({mean_delta:+.4f})")
    print(f"  Wins: Candidate {agg['candidate_wins']} / Baseline {agg['baseline_wins']} / Ties {agg['ties']}")
    print(f"  Regressions: {agg['regression_count']}")
    print(f"  Written: {out_json}, {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
