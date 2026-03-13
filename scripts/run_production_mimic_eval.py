from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

try:
    from analyze_leaderboard import build_summary as build_leaderboard_summary
    from analyze_leaderboard import load_rows as load_leaderboard_rows
except ModuleNotFoundError:  # pragma: no cover
    from scripts.analyze_leaderboard import build_summary as build_leaderboard_summary
    from scripts.analyze_leaderboard import load_rows as load_leaderboard_rows

from rag_challenge.eval.production_mimic import (
    JsonDict,
    JsonList,
    build_page_trace_summary,
    build_public_history_calibration,
    estimate_production_mimic,
)


def _load_json(path: Path | None) -> JsonDict | None:
    if path is None or not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _load_json_list(path: Path | None) -> JsonList | None:
    if path is None or not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list in {path}")
    out: JsonList = []
    for raw in cast("list[object]", obj):
        if isinstance(raw, dict):
            out.append(cast("JsonDict", raw))
    return out


def _load_history_rows(path: Path | None) -> list[JsonDict]:
    payload = _load_json(path)
    if payload is None:
        return []
    rows_obj = payload.get("rows")
    if not isinstance(rows_obj, list):
        return []
    out: list[JsonDict] = []
    for raw in cast("list[object]", rows_obj):
        if isinstance(raw, dict):
            out.append(cast("JsonDict", raw))
    return out


def _load_candidate_row(path: Path, *, label: str) -> JsonDict:
    payload = _load_json(path)
    if payload is None:
        raise ValueError(f"Missing candidate cycle JSON: {path}")
    ranked_obj = payload.get("ranked_candidates")
    if not isinstance(ranked_obj, list):
        raise ValueError(f"Candidate cycle JSON missing ranked_candidates: {path}")
    for raw in cast("list[object]", ranked_obj):
        if not isinstance(raw, dict):
            continue
        row = cast("JsonDict", raw)
        if str(row.get("label") or "").strip() == label:
            return row
    raise ValueError(f"Candidate label not found in {path}: {label}")


def _rank_for_total(
    *,
    leaderboard_path: Path,
    team_name: str,
    total: float,
) -> int:
    rows = load_leaderboard_rows(leaderboard_path)
    return 1 + sum(1 for row in rows if row.team_name != team_name and row.total > total + 1e-9)


def _render_markdown(
    *,
    label: str,
    leaderboard_path: Path,
    team_name: str,
    production_mimic: JsonDict,
) -> str:
    hidden_g_trusted = cast("JsonDict", production_mimic.get("hidden_g_trusted") or {})
    hidden_g_all = cast("JsonDict", production_mimic.get("hidden_g_all") or {})
    exactness = cast("JsonDict", production_mimic.get("exactness") or {})
    judge = cast("JsonDict", production_mimic.get("judge") or {})
    eval_block = cast("JsonDict", production_mimic.get("eval") or {})
    page_trace = cast("JsonDict", production_mimic.get("page_trace") or {})
    platform_like_total = cast("float", production_mimic.get("platform_like_total_estimate", 0.0))
    strict_total = cast("float", production_mimic.get("strict_total_estimate", 0.0))
    paranoid_total = cast("float", production_mimic.get("paranoid_total_estimate", 0.0))
    lines = [
        "# Production-Mimic Local Eval",
        "",
        f"- label: `{label}`",
        f"- leaderboard: `{leaderboard_path}`",
        f"- team: `{team_name}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "## Candidate Envelope",
        "",
        f"- candidate_class: `{production_mimic.get('candidate_class')}`",
        f"- lineage_confidence: `{production_mimic.get('lineage_confidence')}`",
        f"- submit_eligibility: `{production_mimic.get('submit_eligibility')}`",
        f"- no_submit_reason: `{production_mimic.get('no_submit_reason') or 'none'}`",
        "",
        "## Hidden-G",
        "",
        f"- trusted delta: `{cast('float', hidden_g_trusted.get('delta', 0.0)):.4f}`",
        f"- all-case delta: `{cast('float', hidden_g_all.get('delta', 0.0)):.4f}`",
        "",
        "## Exactness",
        "",
        f"- resolved_incorrect_qids: `{len(cast('list[object]', exactness.get('resolved_incorrect_qids') or []))}`",
        f"- unresolved_incorrect_qids: `{len(cast('list[object]', exactness.get('still_mismatched_incorrect_qids') or []))}`",
        "",
        "## Hybrid Strict Judge",
        "",
        f"- pass_rate: `{judge.get('pass_rate')}`",
        f"- avg_grounding: `{judge.get('avg_grounding')}`",
        f"- avg_accuracy: `{judge.get('avg_accuracy')}`",
        f"- judge_failures: `{judge.get('judge_failures')}`",
        f"- disagreement: `{judge.get('disagreement')}`",
        "",
        "## Eval Aggregates",
        "",
        f"- citation_coverage: `{eval_block.get('citation_coverage')}`",
        f"- answer_type_format_compliance: `{eval_block.get('answer_type_format_compliance')}`",
        f"- grounding_g_score_beta_2_5: `{eval_block.get('grounding_g_score_beta_2_5')}`",
        "",
        "## Page Trace",
        "",
        f"- cases_scored: `{page_trace.get('cases_scored')}`",
        f"- trusted_case_count: `{page_trace.get('trusted_case_count')}`",
        f"- gold_in_retrieved_count: `{page_trace.get('gold_in_retrieved_count')}`",
        f"- gold_in_reranked_count: `{page_trace.get('gold_in_reranked_count')}`",
        f"- gold_in_used_count: `{page_trace.get('gold_in_used_count')}`",
        f"- false_positive_case_count: `{page_trace.get('false_positive_case_count')}`",
        f"- explained_ratio: `{page_trace.get('explained_ratio')}`",
        "",
        "## Score Envelope",
        "",
        f"- platform_like_total_estimate: `{platform_like_total:.6f}`",
        f"- strict_total_estimate: `{strict_total:.6f}`",
        f"- paranoid_total_estimate: `{paranoid_total:.6f}`",
        f"- platform_like_rank_estimate: `{_rank_for_total(leaderboard_path=leaderboard_path, team_name=team_name, total=platform_like_total)}`",
        f"- strict_rank_estimate: `{_rank_for_total(leaderboard_path=leaderboard_path, team_name=team_name, total=strict_total)}`",
        f"- paranoid_rank_estimate: `{_rank_for_total(leaderboard_path=leaderboard_path, team_name=team_name, total=paranoid_total)}`",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict production-like local evaluation envelope for one candidate.")
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--candidate-cycle-json", type=Path, required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--exactness-json", type=Path, default=None)
    parser.add_argument("--equivalence-json", type=Path, default=None)
    parser.add_argument("--cheap-eval-json", type=Path, default=None)
    parser.add_argument("--strict-eval-json", type=Path, default=None)
    parser.add_argument("--history-json", type=Path, default=None)
    parser.add_argument("--scaffold-json", type=Path, default=None)
    parser.add_argument("--page-trace-json", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subject_summary = build_leaderboard_summary(load_leaderboard_rows(args.leaderboard), team_name=args.team)
    candidate_row = _load_candidate_row(args.candidate_cycle_json, label=args.candidate_label)
    calibration = build_public_history_calibration(_load_history_rows(args.history_json))
    raw_results_path = Path(str(candidate_row.get("raw_results") or "")).expanduser()
    raw_results_payload = _load_json_list(raw_results_path if raw_results_path.exists() else None)
    result = estimate_production_mimic(
        subject_summary=subject_summary,
        candidate_row=candidate_row,
        exactness_report=_load_json(args.exactness_json),
        equivalence_report=_load_json(args.equivalence_json),
        cheap_eval_payload=_load_json(args.cheap_eval_json),
        strict_eval_payload=_load_json(args.strict_eval_json),
        calibration=calibration,
        raw_results_payload=raw_results_payload,
        scaffold_payload=_load_json(args.scaffold_json),
    )
    result["platform_like_rank_estimate"] = _rank_for_total(
        leaderboard_path=args.leaderboard,
        team_name=args.team,
        total=cast("float", result.get("platform_like_total_estimate", 0.0)),
    )
    result["strict_rank_estimate"] = _rank_for_total(
        leaderboard_path=args.leaderboard,
        team_name=args.team,
        total=cast("float", result.get("strict_total_estimate", 0.0)),
    )
    result["paranoid_rank_estimate"] = _rank_for_total(
        leaderboard_path=args.leaderboard,
        team_name=args.team,
        total=cast("float", result.get("paranoid_total_estimate", 0.0)),
    )
    result["page_trace"] = build_page_trace_summary(_load_json(args.page_trace_json))
    payload: JsonDict = {
        "leaderboard": str(args.leaderboard),
        "team_name": args.team,
        "candidate_label": args.candidate_label,
        "candidate_row": candidate_row,
        "calibration": calibration,
        "production_mimic": result,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        _render_markdown(
            label=args.candidate_label,
            leaderboard_path=args.leaderboard,
            team_name=args.team,
            production_mimic=result,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
