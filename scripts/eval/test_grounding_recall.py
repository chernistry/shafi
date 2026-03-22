"""Grounding recall funnel analyzer.

Computes the oracle funnel:
  oracle@retrieval → oracle@context → oracle@evidence_selected → oracle@cited → actual G

Reads from cached pipeline output (warmup_raw_*.json) + eval_golden_warmup_verified.json +
optionally warmup_score_*.json.

Usage:
    uv run python scripts/test_grounding_recall.py
    uv run python scripts/test_grounding_recall.py --raw data/eval/warmup_raw_20260320_133338.json
    uv run python scripts/test_grounding_recall.py --show-misses

Expected baseline (TAMAR Phase 2, v6_regime, warmup_raw_20260320_133338.json):
    oracle@retrieval   ~94%
    oracle@context     ~83%
    oracle@cited       ~75%
    actual G mean      ~46.8%

After page_budget=2 + dual-case fix, expect:
    oracle@context     ~88%+
    oracle@cited       ~78%+
    actual G mean      ~55%+
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
GOLDEN_PATH = ROOT / "eval_golden_warmup_verified.json"
EVAL_DIR = ROOT / "data" / "eval"

# Miss categories (aligned with g_audit_findings.md taxonomy)
CAT_A = "A"         # multi-page gold, partial citation (0 < G < 1)
CAT_C1 = "C1"       # gold in context, correct, zero citations
CAT_C2 = "C2"       # gold NOT in context, zero citations
CAT_D = "D"         # wrong doc or wrong answer
CAT_UNANSWERABLE = "U"  # empty gold (unanswerable, should return null)
CAT_OK = "OK"       # G == 1.0


def _latest_raw_file() -> Path | None:
    """Return most recent warmup_raw_*.json in data/eval/."""
    candidates = sorted(EVAL_DIR.glob("warmup_raw_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _latest_score_file() -> Path | None:
    """Return most recent warmup_score_*.json in data/eval/."""
    candidates = sorted(EVAL_DIR.glob("warmup_score_*.json"), reverse=True)
    return candidates[0] if candidates else None


def load_golden(path: Path) -> dict[str, dict[str, Any]]:
    """Load golden labels. Returns {question_id: {question, answer_type, gold_page_ids}}."""
    data: list[dict[str, Any]] = json.loads(path.read_text())
    result: dict[str, dict[str, Any]] = {}
    for entry in data:
        qid: str = entry["id"]
        result[qid] = {
            "question": entry.get("question", ""),
            "answer_type": entry.get("answer_type", ""),
            "gold_page_ids": set(entry.get("gold_chunk_ids") or []),
        }
    return result


def load_raw(path: Path) -> dict[str, dict[str, Any]]:
    """Load raw pipeline output. Returns {question_id: telemetry_dict}."""
    data: list[dict[str, Any]] = json.loads(path.read_text())
    result: dict[str, dict[str, Any]] = {}
    for entry in data:
        qid: str = entry["case"]["case_id"]
        telemetry: dict[str, Any] = entry.get("telemetry", {})
        result[qid] = {
            "retrieved_page_ids": set(telemetry.get("retrieved_page_ids") or []),
            "context_page_ids": set(telemetry.get("context_page_ids") or []),
            "used_page_ids": set(telemetry.get("used_page_ids") or []),
            "cited_page_ids": set(telemetry.get("cited_page_ids") or []),
            "answer_text": entry.get("answer_text", ""),
        }
    return result


def load_scores(path: Path) -> dict[str, float]:
    """Load per-question grounding F-beta scores. Returns {question_id: g_score}."""
    data: dict[str, Any] = json.loads(path.read_text())
    result: dict[str, float] = {}
    for case in data.get("per_case", []):
        qid: str = case.get("question_id", "")
        g: float = float(case.get("grounding_f_beta", 0.0))
        result[qid] = g
    return result


def classify_miss(
    gold_pages: set[str],
    retrieved: set[str],
    context: set[str],
    used: set[str],
    cited: set[str],
    g_score: float | None,
) -> str:
    """Assign miss category for a single question."""
    if not gold_pages:
        return CAT_UNANSWERABLE

    g = g_score if g_score is not None else 0.0

    if g >= 0.99:
        return CAT_OK

    if g > 0.0:
        # Some gold cited but not full set → Category A (partial multi-page citation)
        return CAT_A

    # G == 0
    gold_in_context = bool(gold_pages & context)
    if gold_in_context:
        # Gold was in LLM context but nothing cited → Category C1
        return CAT_C1

    gold_in_retrieval = bool(gold_pages & retrieved)
    if gold_in_retrieval:
        # Gold retrieved but dropped by evidence selector before LLM → also C1-ish but context-filtered
        # Technically between C1 and C2; classify as C2 (budget miss at context stage)
        return CAT_C2

    # Gold never retrieved at all → Category D
    return CAT_D


def run(
    raw_path: Path,
    golden_path: Path,
    score_path: Path | None,
    show_misses: bool,
) -> None:
    print(f"Golden:  {golden_path}")
    print(f"Raw:     {raw_path}")
    print(f"Scores:  {score_path or '(not found, G set to 0)'}")
    print()

    golden = load_golden(golden_path)
    raw = load_raw(raw_path)
    scores = load_scores(score_path) if score_path else {}

    # Align on golden question IDs (92 answerable + 8 unanswerable = 100)
    qids = list(golden.keys())

    oracle_retrieval = 0
    oracle_context = 0
    oracle_evidence = 0
    oracle_cited = 0
    g_total = 0.0
    g_count = 0

    category_counts: dict[str, int] = {
        CAT_OK: 0, CAT_A: 0, CAT_C1: 0, CAT_C2: 0, CAT_D: 0, CAT_UNANSWERABLE: 0,
    }
    miss_details: list[dict[str, Any]] = []

    answerable_count = 0  # questions with non-empty gold_chunk_ids

    for qid in qids:
        gold_info = golden[qid]
        gold_pages = gold_info["gold_page_ids"]
        pipeline = raw.get(qid, {})

        retrieved = pipeline.get("retrieved_page_ids", set())
        context = pipeline.get("context_page_ids", set())
        used = pipeline.get("used_page_ids", set())
        cited = pipeline.get("cited_page_ids", set())
        g_score = scores.get(qid)

        if not gold_pages:
            # Unanswerable
            category_counts[CAT_UNANSWERABLE] += 1
            continue

        answerable_count += 1

        # Oracle flags
        if gold_pages & retrieved:
            oracle_retrieval += 1
        if gold_pages & context:
            oracle_context += 1
        if gold_pages & used:
            oracle_evidence += 1
        if gold_pages & cited:
            oracle_cited += 1

        # G accumulation
        g = g_score if g_score is not None else 0.0
        g_total += g
        g_count += 1

        cat = classify_miss(gold_pages, retrieved, context, used, cited, g_score)
        category_counts[cat] += 1

        if cat != CAT_OK:
            miss_details.append({
                "qid": qid[:8],
                "question": gold_info["question"][:70],
                "answer_type": gold_info["answer_type"],
                "g": round(g, 4),
                "category": cat,
                "oracle_ret": bool(gold_pages & retrieved),
                "oracle_ctx": bool(gold_pages & context),
                "oracle_evid": bool(gold_pages & used),
                "oracle_cited": bool(gold_pages & cited),
                "gold_count": len(gold_pages),
                "cited_gold_count": len(gold_pages & cited),
            })

    total = len(qids)
    ans = answerable_count

    print("=" * 70)
    print("GROUNDING RECALL FUNNEL")
    print("=" * 70)
    print(f"Total questions:   {total}")
    print(f"Answerable:        {ans}")
    print(f"Unanswerable:      {total - ans}")
    print()
    print(f"oracle@retrieval:  {oracle_retrieval}/{ans} = {oracle_retrieval/ans*100:.1f}%")
    print(f"oracle@context:    {oracle_context}/{ans}  = {oracle_context/ans*100:.1f}%")
    print(f"oracle@evidence:   {oracle_evidence}/{ans}  = {oracle_evidence/ans*100:.1f}%  (used_page_ids; KEREN calls this oracle@used)")
    print(f"oracle@cited:      {oracle_cited}/{ans}  = {oracle_cited/ans*100:.1f}%  (cited_page_ids; LLM actually cited gold)")
    g_mean = g_total / g_count if g_count else 0.0
    print(f"avg G (all):       {g_mean:.4f} = {g_mean*100:.1f}%")
    if oracle_cited > 0:
        g_eff = (g_total / g_count) / (oracle_cited / ans)
        print(f"G efficiency (cited|oracle@cited): {g_eff:.3f}")

    print()
    print("MISS CATEGORY BREAKDOWN")
    print("-" * 40)
    cat_labels = {
        CAT_OK: "G=1.0 (perfect)",
        CAT_A: "A: multi-page, partial citation",
        CAT_C1: "C1: gold in context, zero citations",
        CAT_C2: "C2: gold not in context (budget miss)",
        CAT_D: "D: wrong doc / never retrieved",
        CAT_UNANSWERABLE: "U: unanswerable (empty gold)",
    }
    for cat, label in cat_labels.items():
        n = category_counts[cat]
        pct = n / total * 100
        print(f"  {cat}: {n:3d} ({pct:5.1f}%)  {label}")

    if show_misses and miss_details:
        print()
        print("MISS DETAILS")
        print("-" * 90)
        header = f"{'QID':10} {'Type':10} {'G':7} {'Cat':4} {'Ret':4} {'Ctx':4} {'Ev':4} {'Cit':4} {'GoldN':6} {'Question'}"
        print(header)
        print("-" * 90)
        for m in sorted(miss_details, key=lambda x: (x["category"], -x["g"])):
            r = "Y" if m["oracle_ret"] else "N"
            c = "Y" if m["oracle_ctx"] else "N"
            e = "Y" if m["oracle_evid"] else "N"
            ci = "Y" if m["oracle_cited"] else "N"
            print(
                f"{m['qid']:10} {m['answer_type']:10} {m['g']:7.4f} {m['category']:4} "
                f"{r:4} {c:4} {e:4} {ci:4} {m['gold_count']:6}  {m['question']}"
            )

    # Comparison to Phase 2 baseline (warmup_raw_20260320_133338.json, v6_regime)
    # Note: avg_G baseline is LOCAL F-beta (27.8%), not platform G score (46.8%)
    print()
    print("COMPARISON TO PHASE 2 BASELINE (v6_regime, warmup_raw_20260320_133338)")
    print("-" * 65)
    baseline = {
        "oracle@retrieval": 93.5,
        "oracle@context": 81.5,
        "oracle@evidence":  75.0,
        "oracle@cited": 67.4,
        "avg_G (local F-beta)": 27.8,
    }
    current = {
        "oracle@retrieval": oracle_retrieval / ans * 100,
        "oracle@context": oracle_context / ans * 100,
        "oracle@evidence": oracle_evidence / ans * 100,
        "oracle@cited": oracle_cited / ans * 100,
        "avg_G (local F-beta)": g_mean * 100,
    }
    for key in baseline:
        delta = current[key] - baseline[key]
        sign = "+" if delta >= 0 else ""
        print(f"  {key:30s}: {current[key]:5.1f}%  (baseline {baseline[key]:5.1f}%, delta {sign}{delta:.1f}pp)")

    print()
    print("NOTE: platform G (46.8% baseline) ≠ local F-beta (27.8% baseline); different formulas.")
    print("      oracle@context delta → OREV fix impact.")
    print("      oracle@cited + avg_G delta → SHAI cite-all impact.")
    print("      oracle@evidence drop → evidence selector regression (investigate if <70%).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounding recall funnel analyzer")
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Path to warmup_raw_*.json. Defaults to most recent in data/eval/",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Path to warmup_score_*.json. Defaults to most recent in data/eval/",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_PATH,
        help="Path to eval_golden_warmup_verified.json",
    )
    parser.add_argument(
        "--show-misses",
        action="store_true",
        help="Print per-question miss table",
    )
    args = parser.parse_args()

    raw_path = args.raw or _latest_raw_file()
    if raw_path is None or not raw_path.exists():
        print(f"ERROR: No raw pipeline output found in {EVAL_DIR}", file=sys.stderr)
        print("Run the pipeline first: uv run python scripts/batch_eval_warmup.py", file=sys.stderr)
        sys.exit(1)

    score_path = args.scores or _latest_score_file()
    if score_path and not score_path.exists():
        score_path = None

    if not args.golden.exists():
        print(f"ERROR: Golden file not found: {args.golden}", file=sys.stderr)
        sys.exit(1)

    run(raw_path, args.golden, score_path, args.show_misses)


if __name__ == "__main__":
    main()
