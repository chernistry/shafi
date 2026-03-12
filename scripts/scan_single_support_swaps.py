from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

try:
    from build_counterfactual_candidate import _build_preflight, _load_json_dict, _load_json_list, _merge_records
    from evaluate_candidate_debug_signal import (
        _build_compare_markdown,
        _evaluate_artifact,
        _load_questions,
        _load_raw_results,
        _select_qids,
    )
    from run_experiment_gate import (
        _answer_changed_count,
        _page_p95,
        _retrieval_projection_changed_count,
        _submission_answers_by_id,
    )
    from search_bounded_support_subsets import _summarize_scores
except ModuleNotFoundError:  # pragma: no cover
    from scripts.build_counterfactual_candidate import (
        _build_preflight,
        _load_json_dict,
        _load_json_list,
        _merge_records,
    )
    from scripts.evaluate_candidate_debug_signal import (
        _build_compare_markdown,
        _evaluate_artifact,
        _load_questions,
        _load_raw_results,
        _select_qids,
    )
    from scripts.run_experiment_gate import (
        _answer_changed_count,
        _page_p95,
        _retrieval_projection_changed_count,
        _submission_answers_by_id,
    )
    from scripts.search_bounded_support_subsets import _summarize_scores

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class SingleSwapResult:
    question_id: str
    recommendation: str
    answer_changed_count: int
    retrieval_page_projection_changed_count: int
    benchmark_all_baseline: float
    benchmark_all_candidate: float
    benchmark_trusted_baseline: float
    benchmark_trusted_candidate: float
    baseline_page_p95: int | None
    candidate_page_p95: int | None
    judge_pass_rate_baseline: float | None = None
    judge_pass_rate_candidate: float | None = None
    judge_grounding_baseline: float | None = None
    judge_grounding_candidate: float | None = None
    changed_qids: list[str] | None = None
    submission_policy: str = "NO_SUBMIT_WITHOUT_USER_APPROVAL"


def _raw_records_by_qid(records: list[JsonDict]) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    for raw in records:
        case = cast("JsonDict", raw.get("case")) if isinstance(raw.get("case"), dict) else {}
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


def _recommendation_rank(value: str) -> int:
    if value == "PROMISING":
        return 2
    if value == "EXPERIMENTAL_NO_SUBMIT":
        return 1
    return 0


def _candidate_sort_key(result: SingleSwapResult) -> tuple[int, float, float, float, float, int, str]:
    judge_pass_delta = (result.judge_pass_rate_candidate or 0.0) - (result.judge_pass_rate_baseline or 0.0)
    judge_grounding_delta = (result.judge_grounding_candidate or 0.0) - (result.judge_grounding_baseline or 0.0)
    return (
        _recommendation_rank(result.recommendation),
        result.benchmark_trusted_candidate - result.benchmark_trusted_baseline,
        result.benchmark_all_candidate - result.benchmark_all_baseline,
        judge_pass_delta,
        judge_grounding_delta,
        -result.retrieval_page_projection_changed_count,
        result.question_id,
    )


def _eligible_qids(*, baseline_submission: dict[str, JsonDict], page_submission: dict[str, JsonDict]) -> list[str]:
    eligible: list[str] = []
    for qid, baseline_record in baseline_submission.items():
        page_record = page_submission.get(qid)
        if page_record is None:
            continue
        baseline_answer = baseline_record.get("answer")
        page_answer = page_record.get("answer")
        if baseline_answer != page_answer:
            continue
        baseline_retrieval = cast("JsonDict", cast("JsonDict", baseline_record.get("telemetry", {})).get("retrieval", {}))
        page_retrieval = cast("JsonDict", cast("JsonDict", page_record.get("telemetry", {})).get("retrieval", {}))
        if baseline_retrieval.get("retrieved_chunk_pages") == page_retrieval.get("retrieved_chunk_pages"):
            continue
        eligible.append(qid)
    return sorted(eligible)


def _extract_judge_metrics(payload: JsonDict) -> tuple[float | None, float | None]:
    summary_obj = payload.get("summary")
    summary = cast("JsonDict", summary_obj) if isinstance(summary_obj, dict) else {}
    judge_obj = summary.get("judge")
    judge = cast("JsonDict", judge_obj) if isinstance(judge_obj, dict) else {}
    pass_rate = judge.get("pass_rate")
    grounding = judge.get("avg_grounding")
    return (
        float(pass_rate) if isinstance(pass_rate, int | float) else None,
        float(grounding) if isinstance(grounding, int | float) else None,
    )


def _render_report(
    *,
    baseline_label: str,
    page_source_label: str,
    results: list[SingleSwapResult],
    judged_qids: set[str],
    limit: int,
) -> str:
    ranked = sorted(results, key=_candidate_sort_key, reverse=True)
    lines = [
        "# Single Support Swap Scan",
        "",
        f"- baseline_label: `{baseline_label}`",
        f"- page_source_label: `{page_source_label}`",
        f"- candidates_scanned: `{len(results)}`",
        f"- judge_top_k_evaluated: `{len(judged_qids)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Rank | QID | Recommendation | Answer Drift | Page Drift | Hidden-G Trusted | Hidden-G All | Judge Pass Δ | Judge Grounding Δ |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(ranked[:limit], start=1):
        judge_pass_delta = (
            None
            if row.judge_pass_rate_baseline is None or row.judge_pass_rate_candidate is None
            else row.judge_pass_rate_candidate - row.judge_pass_rate_baseline
        )
        judge_grounding_delta = (
            None
            if row.judge_grounding_baseline is None or row.judge_grounding_candidate is None
            else row.judge_grounding_candidate - row.judge_grounding_baseline
        )
        lines.append(
            "| "
            f"{index} | `{row.question_id}` | `{row.recommendation}` | "
            f"{row.answer_changed_count} | {row.retrieval_page_projection_changed_count} | "
            f"{row.benchmark_trusted_candidate - row.benchmark_trusted_baseline:.4f} | "
            f"{row.benchmark_all_candidate - row.benchmark_all_baseline:.4f} | "
            f"{'n/a' if judge_pass_delta is None else f'{judge_pass_delta:.4f}'} | "
            f"{'n/a' if judge_grounding_delta is None else f'{judge_grounding_delta:.4f}'} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan single support-only page swaps and rank them by local gates.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--page-source-label", required=True)
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--baseline-preflight", type=Path, required=True)
    parser.add_argument("--page-source-submission", type=Path, required=True)
    parser.add_argument("--page-source-raw-results", type=Path, required=True)
    parser.add_argument("--page-source-preflight", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--judge-top-k", type=int, default=8)
    parser.add_argument("--judge-scope", choices=("all", "free_text", "none"), default="all")
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_submission_payload = _load_json_dict(args.baseline_submission.resolve())
    baseline_raw_results_payload = _load_json_list(args.baseline_raw_results.resolve())
    baseline_preflight_payload = _load_json_dict(args.baseline_preflight.resolve())
    page_source_submission_payload = _load_json_dict(args.page_source_submission.resolve())
    page_source_raw_results_payload = _load_json_list(args.page_source_raw_results.resolve())
    page_source_preflight_payload = _load_json_dict(args.page_source_preflight.resolve())

    baseline_submission_by_id = _submission_answers_by_id(args.baseline_submission.resolve())
    page_source_submission_by_id = _submission_answers_by_id(args.page_source_submission.resolve())
    eligible_qids = _eligible_qids(
        baseline_submission=baseline_submission_by_id,
        page_submission=page_source_submission_by_id,
    )
    baseline_raw_by_id = _raw_records_by_qid(baseline_raw_results_payload)
    baseline_all, baseline_trusted = _summarize_scores(baseline_raw_by_id, benchmark_path=args.benchmark.resolve())
    baseline_page_p95 = _page_p95(args.baseline_preflight.resolve()) or 0

    results: list[SingleSwapResult] = []
    artifacts_by_qid: dict[str, tuple[JsonDict, list[JsonDict], JsonDict]] = {}
    for qid in eligible_qids:
        merged_submission, merged_raw_results, _ = _merge_records(
            answer_source_submission=baseline_submission_payload,
            answer_source_raw_results=baseline_raw_results_payload,
            page_source_submission=page_source_submission_payload,
            page_source_raw_results=page_source_raw_results_payload,
            allowlisted_qids=set(),
            page_allowlisted_qids={qid},
        )
        merged_preflight = _build_preflight(
            merged_payload=merged_submission,
            answer_source_preflight=baseline_preflight_payload,
            page_source_preflight=page_source_preflight_payload,
            answer_source_submission=args.baseline_submission.resolve(),
            page_source_submission=args.page_source_submission.resolve(),
            allowlisted_qids=set(),
            page_allowlisted_qids={qid},
        )
        merged_submission_path = out_dir / f"submission_single_swap_{qid}.json"
        merged_raw_path = out_dir / f"raw_results_single_swap_{qid}.json"
        merged_preflight_path = out_dir / f"preflight_summary_single_swap_{qid}.json"
        merged_submission_path.write_text(json.dumps(merged_submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        merged_raw_path.write_text(json.dumps(merged_raw_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        merged_preflight_path.write_text(json.dumps(merged_preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        candidate_submission_by_id = _submission_answers_by_id(merged_submission_path)
        answer_changed_count = _answer_changed_count(baseline_submission_by_id, candidate_submission_by_id)
        retrieval_projection_changed_count = _retrieval_projection_changed_count(
            baseline_submission_by_id,
            candidate_submission_by_id,
        )
        candidate_raw_by_id = _raw_records_by_qid(merged_raw_results)
        candidate_all, candidate_trusted = _summarize_scores(candidate_raw_by_id, benchmark_path=args.benchmark.resolve())
        candidate_page_p95 = _page_p95(merged_preflight_path) or 0
        recommendation = "PROMISING" if (
            answer_changed_count == 0
            and retrieval_projection_changed_count <= 2
            and candidate_trusted.page_f_beta >= baseline_trusted.page_f_beta
            and candidate_page_p95 <= max(baseline_page_p95, 4)
        ) else "EXPERIMENTAL_NO_SUBMIT"
        results.append(
            SingleSwapResult(
                question_id=qid,
                recommendation=recommendation,
                answer_changed_count=answer_changed_count,
                retrieval_page_projection_changed_count=retrieval_projection_changed_count,
                benchmark_all_baseline=baseline_all.page_f_beta,
                benchmark_all_candidate=candidate_all.page_f_beta,
                benchmark_trusted_baseline=baseline_trusted.page_f_beta,
                benchmark_trusted_candidate=candidate_trusted.page_f_beta,
                baseline_page_p95=baseline_page_p95,
                candidate_page_p95=candidate_page_p95,
                changed_qids=[qid],
            )
        )
        artifacts_by_qid[qid] = (merged_submission, merged_raw_results, merged_preflight)

    ranked = sorted(results, key=_candidate_sort_key, reverse=True)
    judged_qids: set[str] = set()
    top_to_judge = ranked[: max(0, args.judge_top_k)]
    if args.judge_scope != "none":
        questions = _load_questions(args.questions.resolve())
        baseline_cases = _load_raw_results(args.baseline_raw_results.resolve(), questions=questions)
        for row in top_to_judge:
            qid = row.question_id
            merged_submission, merged_raw_results, _merged_preflight = artifacts_by_qid[qid]
            merged_raw_path = out_dir / f"raw_results_single_swap_{qid}.json"
            merged_raw_path.write_text(json.dumps(merged_raw_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            candidate_cases = _load_raw_results(merged_raw_path, questions=questions)
            selected_qids = _select_qids(
                baseline_cases=baseline_cases,
                candidate_cases=candidate_cases,
                scope="changed",
                include_qids={qid},
            )
            baseline_eval = await _evaluate_artifact(
                label=f"{args.baseline_label}_{qid}",
                cases_by_qid=baseline_cases,
                selected_qids=selected_qids,
                judge_scope=args.judge_scope,
                docs_dir=args.docs_dir.resolve(),
                out_dir=out_dir,
            )
            candidate_eval = await _evaluate_artifact(
                label=f"single_swap_{qid}",
                cases_by_qid=candidate_cases,
                selected_qids=selected_qids,
                judge_scope=args.judge_scope,
                docs_dir=args.docs_dir.resolve(),
                out_dir=out_dir,
            )
            compare_md = out_dir / f"candidate_debug_compare_single_swap_{qid}_vs_{args.baseline_label}.md"
            compare_md.write_text(
                _build_compare_markdown(
                    baseline=baseline_eval,
                    candidate=candidate_eval,
                    selected_qids=selected_qids,
                ),
                encoding="utf-8",
            )
            baseline_pass, baseline_grounding = _extract_judge_metrics(baseline_eval.payload)
            candidate_pass, candidate_grounding = _extract_judge_metrics(candidate_eval.payload)
            judged_qids.add(qid)
            for index, existing in enumerate(results):
                if existing.question_id != qid:
                    continue
                results[index] = SingleSwapResult(
                    question_id=existing.question_id,
                    recommendation=existing.recommendation,
                    answer_changed_count=existing.answer_changed_count,
                    retrieval_page_projection_changed_count=existing.retrieval_page_projection_changed_count,
                    benchmark_all_baseline=existing.benchmark_all_baseline,
                    benchmark_all_candidate=existing.benchmark_all_candidate,
                    benchmark_trusted_baseline=existing.benchmark_trusted_baseline,
                    benchmark_trusted_candidate=existing.benchmark_trusted_candidate,
                    baseline_page_p95=existing.baseline_page_p95,
                    candidate_page_p95=existing.candidate_page_p95,
                    judge_pass_rate_baseline=baseline_pass,
                    judge_pass_rate_candidate=candidate_pass,
                    judge_grounding_baseline=baseline_grounding,
                    judge_grounding_candidate=candidate_grounding,
                    changed_qids=existing.changed_qids,
                )
                break

    ranked = sorted(results, key=_candidate_sort_key, reverse=True)
    report = _render_report(
        baseline_label=args.baseline_label,
        page_source_label=args.page_source_label,
        results=ranked,
        judged_qids=judged_qids,
        limit=args.top_k,
    )
    payload: JsonDict = {
        "baseline_label": args.baseline_label,
        "page_source_label": args.page_source_label,
        "candidates_scanned": len(results),
        "judge_top_k": args.judge_top_k,
        "results": [asdict(row) for row in ranked],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    (out_dir / "single_support_swap_scan.md").write_text(report, encoding="utf-8")
    (out_dir / "single_support_swap_scan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    asyncio.run(_async_main(parse_args()))


if __name__ == "__main__":
    main()
