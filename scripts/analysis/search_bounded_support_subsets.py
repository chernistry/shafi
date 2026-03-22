from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

try:
    from build_counterfactual_candidate import (
        _build_preflight,
        _load_json_dict,
        _load_json_list,
        _merge_records,
    )
    from run_experiment_gate import (
        BenchmarkSummary,
        SeedCaseDelta,
        _answer_changed_count,
        _coerce_str_list,
        _page_p95,
        _page_title_equivalent_hit,
        _page_title_map,
        _raw_results_by_id,
        _recommendation,
        _record_title_set,
        _retrieval_projection_changed_count,
        _scaffold_records_by_id,
        _submission_answers_by_id,
    )
    from score_page_benchmark import _load_benchmark, _score_case
    from select_anchor_slice_qids import load_anchor_slice_rows, select_qids
except ModuleNotFoundError:  # pragma: no cover - direct script execution uses different import root
    from scripts.build_counterfactual_candidate import (
        _build_preflight,
        _load_json_dict,
        _load_json_list,
        _merge_records,
    )
    from scripts.run_experiment_gate import (
        BenchmarkSummary,
        SeedCaseDelta,
        _answer_changed_count,
        _coerce_str_list,
        _page_p95,
        _page_title_equivalent_hit,
        _page_title_map,
        _raw_results_by_id,
        _recommendation,
        _record_title_set,
        _retrieval_projection_changed_count,
        _scaffold_records_by_id,
        _submission_answers_by_id,
    )
    from scripts.score_page_benchmark import _load_benchmark, _score_case
    from scripts.select_anchor_slice_qids import load_anchor_slice_rows, select_qids

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class SubsetEvaluation:
    qids: list[str]
    optional_qids: list[str]
    fixed_qids: list[str]
    recommendation: str
    answer_changed_count: int
    retrieval_page_projection_changed_count: int
    benchmark_all_baseline: float
    benchmark_all_candidate: float
    benchmark_trusted_baseline: float
    benchmark_trusted_candidate: float
    baseline_page_p95: int | None
    candidate_page_p95: int | None
    improved_seed_cases: list[str]
    equivalent_seed_cases: list[str]
    regressed_seed_cases: list[str]
    submission_policy: str = "NO_SUBMIT_WITHOUT_USER_APPROVAL"


def _eval_case_from_raw(raw_case: JsonDict | None) -> JsonDict | None:
    if raw_case is None:
        return None
    case_obj = raw_case.get("case")
    case = cast("JsonDict", case_obj) if isinstance(case_obj, dict) else {}
    return {
        "question_id": str(case.get("case_id") or case.get("question_id") or "").strip(),
        "answer": raw_case.get("answer_text"),
        "telemetry": raw_case.get("telemetry") if isinstance(raw_case.get("telemetry"), dict) else {},
    }


def _summarize_scores(raw_by_id: dict[str, JsonDict], *, benchmark_path: Path) -> tuple[BenchmarkSummary, BenchmarkSummary]:
    all_scores: list[float] = []
    trusted_scores: list[float] = []
    for benchmark_case in _load_benchmark(benchmark_path):
        score = _score_case(benchmark_case, _eval_case_from_raw(raw_by_id.get(benchmark_case.question_id)), beta=2.5)
        all_scores.append(score.f_beta)
        if score.trust_tier == "trusted":
            trusted_scores.append(score.f_beta)
    all_summary = BenchmarkSummary(
        cases=len(all_scores),
        page_f_beta=sum(all_scores) / len(all_scores) if all_scores else 0.0,
    )
    trusted_summary = BenchmarkSummary(
        cases=len(trusted_scores),
        page_f_beta=sum(trusted_scores) / len(trusted_scores) if trusted_scores else 0.0,
    )
    return all_summary, trusted_summary


def _seed_case_deltas_in_memory(
    *,
    baseline_scaffold_records: dict[str, JsonDict],
    candidate_scaffold_records: dict[str, JsonDict],
    baseline_raw_by_id: dict[str, JsonDict],
    candidate_raw_by_id: dict[str, JsonDict],
    seed_qids: list[str],
) -> list[SeedCaseDelta]:
    deltas: list[SeedCaseDelta] = []
    for qid in seed_qids:
        baseline_scaffold_record = baseline_scaffold_records.get(qid)
        if baseline_scaffold_record is None:
            continue
        candidate_scaffold_record = candidate_scaffold_records.get(qid, {})
        gold_page_ids = _coerce_str_list(baseline_scaffold_record.get("minimal_required_support_pages"))
        baseline_title_map = _page_title_map(baseline_scaffold_record)
        candidate_title_map = _page_title_map(candidate_scaffold_record)
        gold_record_titles = _record_title_set(baseline_scaffold_record)
        baseline_case = baseline_raw_by_id.get(qid, {})
        candidate_case = candidate_raw_by_id.get(qid, {})
        baseline_telemetry = cast("JsonDict", baseline_case.get("telemetry")) if isinstance(baseline_case.get("telemetry"), dict) else {}
        candidate_telemetry = cast("JsonDict", candidate_case.get("telemetry")) if isinstance(candidate_case.get("telemetry"), dict) else {}
        baseline_used = _coerce_str_list(baseline_telemetry.get("used_page_ids"))
        candidate_used = _coerce_str_list(candidate_telemetry.get("used_page_ids"))
        baseline_context = _coerce_str_list(baseline_telemetry.get("context_page_ids"))
        candidate_context = _coerce_str_list(candidate_telemetry.get("context_page_ids"))
        gold_set = set(gold_page_ids)
        deltas.append(
            SeedCaseDelta(
                question_id=qid,
                gold_page_ids=gold_page_ids,
                baseline_used_page_ids=baseline_used,
                candidate_used_page_ids=candidate_used,
                baseline_context_page_ids=baseline_context,
                candidate_context_page_ids=candidate_context,
                baseline_used_hit=bool(gold_set.intersection(baseline_used)),
                baseline_used_equivalent_hit=_page_title_equivalent_hit(
                    gold_page_ids=gold_page_ids,
                    candidate_page_ids=baseline_used,
                    gold_title_map=baseline_title_map,
                    candidate_title_map=baseline_title_map,
                    gold_record_titles=gold_record_titles,
                ),
                candidate_used_hit=bool(gold_set.intersection(candidate_used)),
                baseline_context_hit=bool(gold_set.intersection(baseline_context)),
                baseline_context_equivalent_hit=_page_title_equivalent_hit(
                    gold_page_ids=gold_page_ids,
                    candidate_page_ids=baseline_context,
                    gold_title_map=baseline_title_map,
                    candidate_title_map=baseline_title_map,
                    gold_record_titles=gold_record_titles,
                ),
                candidate_context_hit=bool(gold_set.intersection(candidate_context)),
                candidate_used_equivalent_hit=_page_title_equivalent_hit(
                    gold_page_ids=gold_page_ids,
                    candidate_page_ids=candidate_used,
                    gold_title_map=baseline_title_map,
                    candidate_title_map=candidate_title_map,
                    gold_record_titles=gold_record_titles,
                ),
                candidate_context_equivalent_hit=_page_title_equivalent_hit(
                    gold_page_ids=gold_page_ids,
                    candidate_page_ids=candidate_context,
                    gold_title_map=baseline_title_map,
                    candidate_title_map=candidate_title_map,
                    gold_record_titles=gold_record_titles,
                ),
            )
        )
    return deltas


def _improved_seed_cases(seed_deltas: list[SeedCaseDelta]) -> list[str]:
    return [
        delta.question_id
        for delta in seed_deltas
        if (
            (delta.candidate_used_hit and not delta.baseline_used_hit)
            or (delta.candidate_context_hit and not delta.baseline_context_hit)
        )
    ]


def _equivalent_seed_cases(seed_deltas: list[SeedCaseDelta]) -> list[str]:
    return [
        delta.question_id
        for delta in seed_deltas
        if (
            (delta.candidate_used_equivalent_hit and not delta.candidate_used_hit)
            or (delta.candidate_context_equivalent_hit and not delta.candidate_context_hit)
        )
    ]


def _regressed_seed_cases(seed_deltas: list[SeedCaseDelta]) -> list[str]:
    return [
        delta.question_id
        for delta in seed_deltas
        if (
            (delta.baseline_used_hit and not (delta.candidate_used_hit or delta.candidate_used_equivalent_hit))
            or (
                delta.baseline_context_hit
                and not (delta.candidate_context_hit or delta.candidate_context_equivalent_hit)
            )
        )
    ]


def _recommendation_rank(value: str) -> int:
    if value == "PROMISING":
        return 2
    if value == "EXPERIMENTAL_NO_SUBMIT":
        return 1
    return 0


def _result_sort_key(result: SubsetEvaluation) -> tuple[int, int, float, float, int, int, int, int, tuple[str, ...]]:
    return (
        _recommendation_rank(result.recommendation),
        len(result.improved_seed_cases),
        result.benchmark_trusted_candidate - result.benchmark_trusted_baseline,
        result.benchmark_all_candidate - result.benchmark_all_baseline,
        -len(result.equivalent_seed_cases),
        -result.answer_changed_count,
        -result.retrieval_page_projection_changed_count,
        -(result.candidate_page_p95 or 0),
        tuple(result.qids),
    )


def _render_report(
    *,
    baseline_label: str,
    page_source_label: str,
    optional_qids: list[str],
    fixed_qids: list[str],
    results: list[SubsetEvaluation],
    limit: int,
) -> str:
    lines = [
        "# Bounded Support Subset Search",
        "",
        f"- baseline_label: `{baseline_label}`",
        f"- page_source_label: `{page_source_label}`",
        f"- optional_qids: `{len(optional_qids)}`",
        f"- fixed_qids: `{len(fixed_qids)}`",
        f"- evaluated_subsets: `{len(results)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
    ]
    for index, row in enumerate(sorted(results, key=_result_sort_key, reverse=True)[:limit], start=1):
        lines.extend(
            [
                f"## Rank {index}",
                "",
                f"- recommendation: `{row.recommendation}`",
                f"- qids: `{', '.join(row.qids)}`",
                f"- answer_changed_count: `{row.answer_changed_count}`",
                f"- retrieval_page_projection_changed_count: `{row.retrieval_page_projection_changed_count}`",
                f"- hidden_g_all: `{row.benchmark_all_baseline:.4f} -> {row.benchmark_all_candidate:.4f}`",
                f"- hidden_g_trusted: `{row.benchmark_trusted_baseline:.4f} -> {row.benchmark_trusted_candidate:.4f}`",
                f"- page_p95: `{row.baseline_page_p95} -> {row.candidate_page_p95}`",
                f"- improved_seed_cases: `{len(row.improved_seed_cases)}`",
                f"- equivalent_seed_cases: `{len(row.equivalent_seed_cases)}`",
                f"- regressed_seed_cases: `{len(row.regressed_seed_cases)}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _page_p95_from_preflight(preflight: JsonDict) -> int:
    distribution_obj = preflight.get("page_count_distribution")
    distribution = cast("JsonDict", distribution_obj) if isinstance(distribution_obj, dict) else {}
    value = distribution.get("p95")
    return int(value) if isinstance(value, int | float) else 0


def _write_best_candidate(
    *,
    out_dir: Path,
    label: str,
    merged_submission: JsonDict,
    merged_raw_results: list[JsonDict],
    merged_preflight: JsonDict,
) -> tuple[Path, Path, Path]:
    submission_path = out_dir / f"submission_{label}.json"
    raw_results_path = out_dir / f"raw_results_{label}.json"
    preflight_path = out_dir / f"preflight_summary_{label}.json"
    submission_path.write_text(json.dumps(merged_submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_results_path.write_text(json.dumps(merged_raw_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preflight_path.write_text(json.dumps(merged_preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return submission_path, raw_results_path, preflight_path


def _load_seed_qids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search bounded support-only page-swap subsets against local grounding gates.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--page-source-label", required=True)
    parser.add_argument("--anchor-slice-json", type=Path, required=True)
    parser.add_argument("--include-status", action="append", default=[])
    parser.add_argument("--exclude-status", action="append", default=[])
    parser.add_argument("--exclude-qid", action="append", default=[])
    parser.add_argument("--fixed-qid", action="append", default=[])
    parser.add_argument("--seed-qids-file", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--baseline-preflight", type=Path, required=True)
    parser.add_argument("--baseline-scaffold", type=Path, required=True)
    parser.add_argument("--page-source-submission", type=Path, required=True)
    parser.add_argument("--page-source-raw-results", type=Path, required=True)
    parser.add_argument("--page-source-preflight", type=Path, required=True)
    parser.add_argument("--candidate-scaffold", type=Path, required=True)
    parser.add_argument("--require-no-answer-change", action="store_true")
    parser.add_argument("--require-used-support", action="store_true")
    parser.add_argument("--min-optional", type=int, default=0)
    parser.add_argument("--max-optional", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--best-label", default=None)
    parser.add_argument("--best-out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = load_anchor_slice_rows(args.anchor_slice_json)
    optional_qids, selection_report = select_qids(
        rows=rows,
        include_statuses={str(status).strip() for status in args.include_status if str(status).strip()},
        exclude_statuses={str(status).strip() for status in args.exclude_status if str(status).strip()},
        require_no_answer_change=bool(args.require_no_answer_change),
        require_used_support=bool(args.require_used_support),
        excluded_qids={str(qid).strip() for qid in args.exclude_qid if str(qid).strip()},
    )
    fixed_qids = sorted({str(qid).strip() for qid in args.fixed_qid if str(qid).strip()})
    min_optional = max(0, args.min_optional)
    max_optional = len(optional_qids) if args.max_optional is None else min(len(optional_qids), args.max_optional)

    all_subsets: list[tuple[str, ...]] = []
    for size in range(min_optional, max_optional + 1):
        all_subsets.extend(itertools.combinations(optional_qids, size))
    if len(all_subsets) > args.max_combinations:
        raise ValueError(
            f"Subset search would evaluate {len(all_subsets)} combinations; cap is {args.max_combinations}. "
            "Reduce optional qids or set a higher limit explicitly."
        )

    answer_source_submission = _load_json_dict(args.baseline_submission)
    answer_source_raw_results = _load_json_list(args.baseline_raw_results)
    answer_source_preflight = _load_json_dict(args.baseline_preflight)
    page_source_submission = _load_json_dict(args.page_source_submission)
    page_source_raw_results = _load_json_list(args.page_source_raw_results)
    page_source_preflight = _load_json_dict(args.page_source_preflight)
    baseline_submission_by_id = _submission_answers_by_id(args.baseline_submission)
    baseline_raw_by_id = _raw_results_by_id(args.baseline_raw_results)
    baseline_scaffold_records = _scaffold_records_by_id(args.baseline_scaffold)
    candidate_scaffold_records = _scaffold_records_by_id(args.candidate_scaffold)
    baseline_all, baseline_trusted = _summarize_scores(baseline_raw_by_id, benchmark_path=args.benchmark)
    baseline_page_p95 = _page_p95(args.baseline_preflight)
    seed_qids = _load_seed_qids(args.seed_qids_file)

    results: list[SubsetEvaluation] = []
    merged_artifacts: dict[tuple[str, ...], tuple[JsonDict, list[JsonDict], JsonDict]] = {}

    for optional_subset in all_subsets:
        qids = list(dict.fromkeys([*optional_subset, *fixed_qids]))
        merged_submission, merged_raw_results, _ = _merge_records(
            answer_source_submission=answer_source_submission,
            answer_source_raw_results=answer_source_raw_results,
            page_source_submission=page_source_submission,
            page_source_raw_results=page_source_raw_results,
            allowlisted_qids=set(),
            page_allowlisted_qids=set(qids),
            page_source_pages_default="none",
        )
        merged_preflight = _build_preflight(
            merged_payload=merged_submission,
            answer_source_preflight=answer_source_preflight,
            page_source_preflight=page_source_preflight,
            answer_source_submission=args.baseline_submission,
            page_source_submission=args.page_source_submission,
            allowlisted_qids=set(),
            page_allowlisted_qids=set(qids),
        )
        candidate_submission_by_id = _submission_answers_by_id_from_payload(merged_submission)
        answer_changed_count = _answer_changed_count(baseline_submission_by_id, candidate_submission_by_id)
        retrieval_projection_changed_count = _retrieval_projection_changed_count(
            baseline_submission_by_id,
            candidate_submission_by_id,
        )
        candidate_raw_by_id = _raw_results_by_id_from_records(merged_raw_results)
        candidate_all, candidate_trusted = _summarize_scores(candidate_raw_by_id, benchmark_path=args.benchmark)
        seed_deltas = _seed_case_deltas_in_memory(
            baseline_scaffold_records=baseline_scaffold_records,
            candidate_scaffold_records=candidate_scaffold_records,
            baseline_raw_by_id=baseline_raw_by_id,
            candidate_raw_by_id=candidate_raw_by_id,
            seed_qids=seed_qids,
        )
        recommendation, _notes, _staged_eval = _recommendation(
            static_safety_status="assumed_passed",
            static_safety_reason=None,
            impact_canary_status="assumed_passed",
            impact_canary_reason=None,
            impact_canary_pack="bounded_support_subset_search",
            baseline_trusted=baseline_trusted,
            candidate_trusted=candidate_trusted,
            answer_changed_count=answer_changed_count,
            baseline_page_p95=baseline_page_p95,
            candidate_page_p95=_page_p95_from_preflight(merged_preflight),
        )
        result = SubsetEvaluation(
            qids=qids,
            optional_qids=list(optional_subset),
            fixed_qids=fixed_qids,
            recommendation=recommendation,
            answer_changed_count=answer_changed_count,
            retrieval_page_projection_changed_count=retrieval_projection_changed_count,
            benchmark_all_baseline=baseline_all.page_f_beta,
            benchmark_all_candidate=candidate_all.page_f_beta,
            benchmark_trusted_baseline=baseline_trusted.page_f_beta,
            benchmark_trusted_candidate=candidate_trusted.page_f_beta,
            baseline_page_p95=baseline_page_p95,
            candidate_page_p95=_page_p95_from_preflight(merged_preflight),
            improved_seed_cases=_improved_seed_cases(seed_deltas),
            equivalent_seed_cases=_equivalent_seed_cases(seed_deltas),
            regressed_seed_cases=_regressed_seed_cases(seed_deltas),
        )
        results.append(result)
        merged_artifacts[tuple(qids)] = (merged_submission, merged_raw_results, merged_preflight)

    ranked_results = sorted(results, key=_result_sort_key, reverse=True)
    report = _render_report(
        baseline_label=args.baseline_label,
        page_source_label=args.page_source_label,
        optional_qids=optional_qids,
        fixed_qids=fixed_qids,
        results=ranked_results,
        limit=args.top_k,
    )
    payload: JsonDict = {
        "baseline_label": args.baseline_label,
        "page_source_label": args.page_source_label,
        "selection_report": selection_report,
        "fixed_qids": fixed_qids,
        "evaluated_subsets": len(results),
        "results": [asdict(row) for row in ranked_results],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out.write_text(report, encoding="utf-8")
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.best_label is not None:
        best_qids = tuple(ranked_results[0].qids)
        best_artifacts = merged_artifacts[best_qids]
        out_dir = args.best_out_dir or args.baseline_submission.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_best_candidate(
            out_dir=out_dir,
            label=args.best_label,
            merged_submission=best_artifacts[0],
            merged_raw_results=best_artifacts[1],
            merged_preflight=best_artifacts[2],
        )


def _submission_answers_by_id_from_payload(payload: JsonDict) -> dict[str, JsonDict]:
    answers_obj = payload.get("answers")
    answers = cast("list[object]", answers_obj) if isinstance(answers_obj, list) else []
    out: dict[str, JsonDict] = {}
    for raw in answers:
        if not isinstance(raw, dict):
            continue
        qid = str(raw.get("question_id") or "").strip()
        if qid:
            out[qid] = cast("JsonDict", raw)
    return out


def _raw_results_by_id_from_records(records: list[JsonDict]) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    for raw in records:
        case = cast("JsonDict", raw.get("case")) if isinstance(raw.get("case"), dict) else {}
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


if __name__ == "__main__":
    main()
