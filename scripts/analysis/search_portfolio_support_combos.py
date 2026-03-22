from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
import argparse
import asyncio
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

try:
    from build_counterfactual_candidate import _build_preflight, _load_json_dict, _load_json_list, _merge_records
    from evaluate_candidate_debug_signal import (
        CandidateEvalArtifacts,
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
        CandidateEvalArtifacts,
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
class PortfolioItem:
    qid: str
    label: str
    submission_path: Path
    raw_results_path: Path
    preflight_path: Path
    notes: str


@dataclass(frozen=True)
class ComboEvaluation:
    qids: list[str]
    labels: list[str]
    recommendation: str
    answer_changed_count: int
    retrieval_page_projection_changed_count: int
    benchmark_all_baseline: float
    benchmark_all_candidate: float
    benchmark_trusted_baseline: float
    benchmark_trusted_candidate: float
    baseline_page_p95: int
    candidate_page_p95: int
    judge_pass_rate_baseline: float | None = None
    judge_pass_rate_candidate: float | None = None
    judge_grounding_baseline: float | None = None
    judge_grounding_candidate: float | None = None
    changed_qids: list[str] | None = None
    submission_policy: str = "NO_SUBMIT_WITHOUT_USER_APPROVAL"


def _raw_results_by_id_in_memory(records: list[JsonDict]) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    for row in records:
        case = cast("JsonDict", row.get("case")) if isinstance(row.get("case"), dict) else {}
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _load_portfolio(path: Path) -> list[PortfolioItem]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array in {path}")
    items: list[PortfolioItem] = []
    seen_qids: set[str] = set()
    for row_obj in cast("list[object]", obj):
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        qid = str(row.get("qid") or "").strip()
        if not qid:
            continue
        if qid in seen_qids:
            raise ValueError(f"Duplicate qid in portfolio: {qid}")
        seen_qids.add(qid)
        items.append(
            PortfolioItem(
                qid=qid,
                label=str(row.get("label") or qid).strip() or qid,
                submission_path=Path(str(row.get("submission_path") or "")).expanduser().resolve(),
                raw_results_path=Path(str(row.get("raw_results_path") or "")).expanduser().resolve(),
                preflight_path=Path(str(row.get("preflight_path") or "")).expanduser().resolve(),
                notes=str(row.get("notes") or "").strip(),
            )
        )
    if not items:
        raise ValueError(f"No valid portfolio items in {path}")
    return items


def _load_required_qids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            out.add(text)
    return out


def _recommendation(
    *,
    answer_changed_count: int,
    retrieval_page_projection_changed_count: int,
    baseline_all: float,
    candidate_all: float,
    baseline_trusted: float,
    candidate_trusted: float,
    baseline_page_p95: int,
    candidate_page_p95: int,
) -> str:
    if (
        answer_changed_count == 0
        and retrieval_page_projection_changed_count <= 6
        and candidate_trusted >= baseline_trusted
        and candidate_all >= baseline_all
        and candidate_page_p95 <= max(4, baseline_page_p95)
    ):
        return "PROMISING"
    if (
        answer_changed_count <= 1
        and retrieval_page_projection_changed_count <= 12
        and candidate_all >= baseline_all
        and candidate_page_p95 <= max(4, baseline_page_p95 + 1)
    ):
        return "EXPERIMENTAL_NO_SUBMIT"
    return "REJECT"


def _combo_sort_key(row: ComboEvaluation) -> tuple[int, float, float, float, float, int, tuple[str, ...]]:
    judge_pass_delta = (row.judge_pass_rate_candidate or 0.0) - (row.judge_pass_rate_baseline or 0.0)
    judge_grounding_delta = (row.judge_grounding_candidate or 0.0) - (row.judge_grounding_baseline or 0.0)
    recommendation_rank = {"PROMISING": 2, "EXPERIMENTAL_NO_SUBMIT": 1}.get(row.recommendation, 0)
    return (
        recommendation_rank,
        row.benchmark_trusted_candidate - row.benchmark_trusted_baseline,
        row.benchmark_all_candidate - row.benchmark_all_baseline,
        judge_pass_delta,
        judge_grounding_delta,
        -row.retrieval_page_projection_changed_count,
        tuple(row.qids),
    )


def _combo_label(qids: list[str]) -> str:
    joined = "|".join(sorted(qids))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:10]
    prefixes = "_".join(qid[:6] for qid in qids[:3])
    return f"combo_{prefixes}_{digest}"


def _merge_combo(
    *,
    baseline_submission_payload: JsonDict,
    baseline_raw_results_payload: list[JsonDict],
    baseline_preflight_payload: JsonDict,
    baseline_submission_path: Path,
    items: list[PortfolioItem],
    out_dir: Path,
    label: str,
) -> tuple[Path, Path, Path]:
    merged_submission = json.loads(json.dumps(baseline_submission_payload, ensure_ascii=False))
    merged_raw_results = json.loads(json.dumps(baseline_raw_results_payload, ensure_ascii=False))
    merged_preflight = json.loads(json.dumps(baseline_preflight_payload, ensure_ascii=False))

    answer_source_submission = cast("JsonDict", merged_submission)
    answer_source_raw_results = cast("list[JsonDict]", merged_raw_results)

    for item in items:
        page_source_submission = _load_json_dict(item.submission_path)
        page_source_raw_results = _load_json_list(item.raw_results_path)
        answer_source_submission, answer_source_raw_results, _report = _merge_records(
            answer_source_submission=answer_source_submission,
            answer_source_raw_results=answer_source_raw_results,
            page_source_submission=page_source_submission,
            page_source_raw_results=page_source_raw_results,
            allowlisted_qids=set(),
            page_allowlisted_qids={item.qid},
            page_source_pages_default="all",
        )
        merged_preflight = _build_preflight(
            merged_payload=answer_source_submission,
            answer_source_preflight=baseline_preflight_payload,
            page_source_preflight=baseline_preflight_payload,
            answer_source_submission=baseline_submission_path,
            page_source_submission=baseline_submission_path,
            allowlisted_qids=set(),
            page_allowlisted_qids={candidate_item.qid for candidate_item in items},
        )
        merged_preflight["counterfactual_projection"] = {
            "answer_source_submission": str(baseline_submission_path),
            "page_source_submission": str(baseline_submission_path),
            "page_source_answer_qids": [],
            "page_source_page_qids": [candidate_item.qid for candidate_item in items],
            "page_source_labels": {candidate_item.qid: candidate_item.label for candidate_item in items},
            "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        }

    submission_path = out_dir / f"submission_{label}.json"
    raw_results_path = out_dir / f"raw_results_{label}.json"
    preflight_path = out_dir / f"preflight_summary_{label}.json"
    submission_path.write_text(json.dumps(answer_source_submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_results_path.write_text(json.dumps(answer_source_raw_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preflight_path.write_text(json.dumps(merged_preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return submission_path, raw_results_path, preflight_path


def _render_report(
    *,
    baseline_label: str,
    portfolio_path: Path,
    results: list[ComboEvaluation],
    limit: int,
) -> str:
    lines = [
        "# Portfolio Support Combo Search",
        "",
        f"- baseline_label: `{baseline_label}`",
        f"- portfolio_path: `{portfolio_path}`",
        f"- evaluated_combos: `{len(results)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Rank | QIDs | Recommendation | Answer Drift | Page Drift | Hidden-G Trusted Δ | Hidden-G All Δ | Judge Pass Δ | Judge Grounding Δ |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked = sorted(results, key=_combo_sort_key, reverse=True)
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
            f"{index} | `{','.join(row.qids)}` | `{row.recommendation}` | "
            f"{row.answer_changed_count} | {row.retrieval_page_projection_changed_count} | "
            f"{row.benchmark_trusted_candidate - row.benchmark_trusted_baseline:.4f} | "
            f"{row.benchmark_all_candidate - row.benchmark_all_baseline:.4f} | "
            f"{'n/a' if judge_pass_delta is None else f'{judge_pass_delta:.4f}'} | "
            f"{'n/a' if judge_grounding_delta is None else f'{judge_grounding_delta:.4f}'} |"
        )
    return "\n".join(lines) + "\n"


def _summary_from_case_rows(case_rows: list[JsonDict]) -> JsonDict:
    ttft_values: list[float] = []
    citation_sum = 0.0
    format_sum = 0.0
    judge_cases = 0
    judge_passes = 0
    judge_accuracy_sum = 0.0
    judge_grounding_sum = 0.0
    judge_clarity_sum = 0.0
    judge_uncertainty_sum = 0.0
    judge_failures = 0

    for row in case_rows:
        citation_sum += float(row.get("citation_coverage") or 0.0)
        format_sum += float(row.get("format_compliance") or 0.0)
        ttft = _coerce_float(row.get("ttft_ms"))
        if ttft is not None:
            ttft_values.append(ttft)
        judge_obj = row.get("judge")
        if isinstance(judge_obj, dict):
            judge = cast("JsonDict", judge_obj)
            judge_cases += 1
            if str(judge.get("verdict") or "").strip().upper() == "PASS":
                judge_passes += 1
            scores_obj = judge.get("scores")
            scores = cast("JsonDict", scores_obj) if isinstance(scores_obj, dict) else {}
            judge_accuracy_sum += float(scores.get("accuracy") or 0.0)
            judge_grounding_sum += float(scores.get("grounding") or 0.0)
            judge_clarity_sum += float(scores.get("clarity") or 0.0)
            judge_uncertainty_sum += float(scores.get("uncertainty_handling") or 0.0)
        elif row.get("judge_failure"):
            judge_failures += 1

    summary: JsonDict = {
        "total_cases": len(case_rows),
        "answer_type_cases": len(case_rows),
        "citation_coverage": round(citation_sum / max(1, len(case_rows)), 4),
        "answer_type_format_compliance": round(format_sum / max(1, len(case_rows)), 4),
        "ttft_p50_ms": None if not ttft_values else round(sorted(ttft_values)[len(ttft_values) // 2], 1),
        "ttft_p95_ms": None if not ttft_values else round(sorted(ttft_values)[min(len(ttft_values) - 1, int((len(ttft_values) - 1) * 0.95))], 1),
        "failures": 0,
    }
    if judge_cases > 0 or judge_failures > 0:
        summary["judge"] = {
            "cases": judge_cases,
            "pass_rate": None if judge_cases <= 0 else round(judge_passes / judge_cases, 4),
            "avg_accuracy": None if judge_cases <= 0 else round(judge_accuracy_sum / judge_cases, 4),
            "avg_grounding": None if judge_cases <= 0 else round(judge_grounding_sum / judge_cases, 4),
            "avg_clarity": None if judge_cases <= 0 else round(judge_clarity_sum / judge_cases, 4),
            "avg_uncertainty_handling": None if judge_cases <= 0 else round(judge_uncertainty_sum / judge_cases, 4),
            "judge_failures": judge_failures,
        }
    return summary


def _filter_eval_payload(*, label: str, payload: JsonDict, selected_qids: list[str]) -> JsonDict:
    case_rows_obj = payload.get("cases")
    case_rows = cast("list[JsonDict]", case_rows_obj) if isinstance(case_rows_obj, list) else []
    rows_by_qid = {
        str(row.get("question_id") or row.get("case_id") or "").strip(): row
        for row in case_rows
    }
    filtered_rows = [rows_by_qid[qid] for qid in selected_qids if qid in rows_by_qid]
    return {
        "label": label,
        "selected_qids": selected_qids,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        "summary": _summary_from_case_rows(filtered_rows),
        "cases": filtered_rows,
        "failures": [],
    }


async def _judge_top_combos(
    *,
    results: list[ComboEvaluation],
    artifacts_by_label: dict[str, tuple[Path, Path, Path]],
    baseline_label: str,
    baseline_raw_results_path: Path,
    questions_path: Path,
    docs_dir: Path,
    out_dir: Path,
    top_k: int,
) -> list[ComboEvaluation]:
    if top_k <= 0:
        return results

    questions = _load_questions(questions_path)
    baseline_cases = _load_raw_results(baseline_raw_results_path, questions=questions)
    ranked = sorted(results, key=_combo_sort_key, reverse=True)
    top_rows = ranked[: min(top_k, len(ranked))]
    union_qids = sorted({qid for row in top_rows for qid in row.qids})
    baseline_eval_union = await _evaluate_artifact(
        label=baseline_label,
        cases_by_qid=baseline_cases,
        selected_qids=union_qids,
        judge_scope="all",
        docs_dir=docs_dir,
        out_dir=out_dir,
    )
    evaluated: dict[str, ComboEvaluation] = {}

    for row in top_rows:
        label = _combo_label(row.qids)
        _submission_path, raw_results_path, _preflight_path = artifacts_by_label[label]
        candidate_cases = _load_raw_results(raw_results_path, questions=questions)
        selected_qids = _select_qids(
            baseline_cases=baseline_cases,
            candidate_cases=candidate_cases,
            scope="changed",
            include_qids=set(),
        )
        candidate_eval = await _evaluate_artifact(
            label=label,
            cases_by_qid=candidate_cases,
            selected_qids=selected_qids,
            judge_scope="all",
            docs_dir=docs_dir,
            out_dir=out_dir,
        )
        baseline_eval = CandidateEvalArtifacts(
            label=baseline_label,
            eval_path=baseline_eval_union.eval_path,
            judge_path=baseline_eval_union.judge_path,
            payload=_filter_eval_payload(
                label=baseline_label,
                payload=baseline_eval_union.payload,
                selected_qids=selected_qids,
            ),
        )
        compare_md = out_dir / f"candidate_debug_compare_{label}_vs_{baseline_label}.md"
        compare_md.write_text(
            _build_compare_markdown(
                baseline=baseline_eval,
                candidate=candidate_eval,
                selected_qids=selected_qids,
            ),
            encoding="utf-8",
        )

        baseline_judge = cast("JsonDict", cast("JsonDict", baseline_eval.payload.get("summary", {})).get("judge", {}))
        candidate_judge = cast("JsonDict", cast("JsonDict", candidate_eval.payload.get("summary", {})).get("judge", {}))
        evaluated[label] = ComboEvaluation(
            qids=row.qids,
            labels=row.labels,
            recommendation=row.recommendation,
            answer_changed_count=row.answer_changed_count,
            retrieval_page_projection_changed_count=row.retrieval_page_projection_changed_count,
            benchmark_all_baseline=row.benchmark_all_baseline,
            benchmark_all_candidate=row.benchmark_all_candidate,
            benchmark_trusted_baseline=row.benchmark_trusted_baseline,
            benchmark_trusted_candidate=row.benchmark_trusted_candidate,
            baseline_page_p95=row.baseline_page_p95,
            candidate_page_p95=row.candidate_page_p95,
            judge_pass_rate_baseline=_coerce_float(baseline_judge.get("pass_rate")),
            judge_pass_rate_candidate=_coerce_float(candidate_judge.get("pass_rate")),
            judge_grounding_baseline=_coerce_float(baseline_judge.get("avg_grounding")),
            judge_grounding_candidate=_coerce_float(candidate_judge.get("avg_grounding")),
            changed_qids=selected_qids,
        )

    out: list[ComboEvaluation] = []
    for row in results:
        label = _combo_label(row.qids)
        out.append(evaluated.get(label, row))
    return out


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search portfolio support-only combos across multiple source artifacts.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--baseline-preflight", type=Path, required=True)
    parser.add_argument("--portfolio-json", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-size", type=int, default=1)
    parser.add_argument("--max-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--judge-top-k", type=int, default=5)
    parser.add_argument("--required-qids-file", type=Path, default=None)
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    baseline_submission_path = args.baseline_submission.resolve()
    baseline_raw_results_path = args.baseline_raw_results.resolve()
    baseline_preflight_path = args.baseline_preflight.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_submission_payload = _load_json_dict(baseline_submission_path)
    baseline_raw_results_payload = _load_json_list(baseline_raw_results_path)
    baseline_preflight_payload = _load_json_dict(baseline_preflight_path)
    baseline_submission_by_id = _submission_answers_by_id(baseline_submission_path)
    baseline_all, baseline_trusted = _summarize_scores(
        _raw_results_by_id_in_memory(baseline_raw_results_payload),
        benchmark_path=args.benchmark.resolve(),
    )
    baseline_page_p95 = _page_p95(baseline_preflight_path) or 0

    items = _load_portfolio(args.portfolio_json.resolve())
    required_qids = _load_required_qids(args.required_qids_file.resolve() if args.required_qids_file is not None else None)
    if required_qids:
        known_qids = {item.qid for item in items}
        missing = sorted(required_qids.difference(known_qids))
        if missing:
            raise ValueError(f"Required qids missing from portfolio: {missing}")

    results: list[ComboEvaluation] = []
    artifacts_by_label: dict[str, tuple[Path, Path, Path]] = {}
    min_size = max(1, int(args.min_size))
    max_size = max(min_size, int(args.max_size))
    for size in range(min_size, min(max_size, len(items)) + 1):
        for combo_items in itertools.combinations(items, size):
            qids = [item.qid for item in combo_items]
            if required_qids and not required_qids.issubset(set(qids)):
                continue
            labels = [item.label for item in combo_items]
            label = _combo_label(qids)
            submission_path, raw_results_path, preflight_path = _merge_combo(
                baseline_submission_payload=baseline_submission_payload,
                baseline_raw_results_payload=baseline_raw_results_payload,
                baseline_preflight_payload=baseline_preflight_payload,
                baseline_submission_path=baseline_submission_path,
                items=list(combo_items),
                out_dir=out_dir,
                label=label,
            )
            artifacts_by_label[label] = (submission_path, raw_results_path, preflight_path)

            candidate_submission_by_id = _submission_answers_by_id(submission_path)
            answer_changed_count = _answer_changed_count(baseline_submission_by_id, candidate_submission_by_id)
            page_drift = _retrieval_projection_changed_count(baseline_submission_by_id, candidate_submission_by_id)
            candidate_raw_results_payload = _load_json_list(raw_results_path)
            candidate_all, candidate_trusted = _summarize_scores(
                _raw_results_by_id_in_memory(candidate_raw_results_payload),
                benchmark_path=args.benchmark.resolve(),
            )
            candidate_page_p95 = _page_p95(preflight_path) or 0
            results.append(
                ComboEvaluation(
                    qids=qids,
                    labels=labels,
                    recommendation=_recommendation(
                        answer_changed_count=answer_changed_count,
                        retrieval_page_projection_changed_count=page_drift,
                        baseline_all=baseline_all.page_f_beta,
                        candidate_all=candidate_all.page_f_beta,
                        baseline_trusted=baseline_trusted.page_f_beta,
                        candidate_trusted=candidate_trusted.page_f_beta,
                        baseline_page_p95=baseline_page_p95,
                        candidate_page_p95=candidate_page_p95,
                    ),
                    answer_changed_count=answer_changed_count,
                    retrieval_page_projection_changed_count=page_drift,
                    benchmark_all_baseline=baseline_all.page_f_beta,
                    benchmark_all_candidate=candidate_all.page_f_beta,
                    benchmark_trusted_baseline=baseline_trusted.page_f_beta,
                    benchmark_trusted_candidate=candidate_trusted.page_f_beta,
                    baseline_page_p95=baseline_page_p95,
                    candidate_page_p95=candidate_page_p95,
                )
            )

    results = await _judge_top_combos(
        results=results,
        artifacts_by_label=artifacts_by_label,
        baseline_label=str(args.baseline_label),
        baseline_raw_results_path=baseline_raw_results_path,
        questions_path=args.questions.resolve(),
        docs_dir=args.docs_dir.resolve(),
        out_dir=out_dir,
        top_k=int(args.judge_top_k),
    )

    payload = {
        "baseline_label": str(args.baseline_label),
        "portfolio_path": str(args.portfolio_json.resolve()),
        "results": [asdict(row) for row in sorted(results, key=_combo_sort_key, reverse=True)],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    (out_dir / "portfolio_support_combo_search.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "portfolio_support_combo_search.md").write_text(
        _render_report(
            baseline_label=str(args.baseline_label),
            portfolio_path=args.portfolio_json.resolve(),
            results=results,
            limit=int(args.top_k),
        ),
        encoding="utf-8",
    )


def main() -> None:
    asyncio.run(_async_main(parse_args()))


if __name__ == "__main__":
    main()
