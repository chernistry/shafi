"""Evaluate the selective ICR shadow reranker against recorded candidate pools."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, cast

from rag_challenge.core.selective_icr_reranker import SelectiveICRConfig, SelectiveICRReranker
from rag_challenge.eval.failure_cartography import load_reviewed_golden

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class RecordedCandidate:
    """A chunk-like candidate reconstructed from raw-results telemetry."""

    chunk_id: str
    text: str
    retrieval_score: float = 0.0


@dataclass(frozen=True, slots=True)
class ShadowCase:
    """Benchmark case extracted from recorded raw results."""

    question_id: str
    question: str
    answer_type: str
    gold_page_ids: tuple[str, ...]
    baseline_page_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    chunk_snippets: dict[str, str]
    baseline_rerank_ms: int
    family: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Per-case benchmark result."""

    question_id: str
    family: str
    gold_page_ids: list[str]
    baseline_page_ids: list[str]
    shadow_page_ids: list[str]
    baseline_hit: bool
    shadow_hit: bool
    baseline_precision: float
    shadow_precision: float
    baseline_rerank_ms: int
    shadow_rerank_ms: int


@dataclass(frozen=True, slots=True)
class SliceSummary:
    """Aggregate metrics for one benchmark slice."""

    name: str
    case_count: int
    baseline_hit_rate: float
    shadow_hit_rate: float
    hit_delta: float
    baseline_precision: float
    shadow_precision: float
    precision_delta: float
    baseline_latency_ms_mean: float
    shadow_latency_ms_mean: float
    baseline_latency_ms_p50: float
    shadow_latency_ms_p50: float
    family_metrics: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class ShadowBenchmarkReport:
    """Full benchmark report for one or more reviewed slices."""

    slices: list[SliceSummary]
    cases: list[CaseResult]


def load_shadow_cases(raw_results_path: Path, reviewed_path: Path) -> list[ShadowCase]:
    """Load benchmark cases from a recorded raw-results artifact.

    Args:
        raw_results_path: Path to the recorded raw results JSON.
        reviewed_path: Path to the reviewed gold JSON.

    Returns:
        Filtered benchmark cases aligned to the reviewed slice.
    """

    reviewed = load_reviewed_golden(reviewed_path)
    payload = json.loads(raw_results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected raw results list: {raw_results_path}")

    cases: list[ShadowCase] = []
    for raw in cast("list[object]", payload):
        if not isinstance(raw, dict):
            continue
        raw_case = cast("dict[str, object]", raw)
        case_obj = _as_dict(raw_case.get("case"))
        telemetry_obj = _as_dict(raw_case.get("telemetry"))
        question_id = str(case_obj.get("case_id") or telemetry_obj.get("question_id") or raw_case.get("question_id") or "").strip()
        if question_id not in reviewed:
            continue
        golden = reviewed[question_id]
        question = str(case_obj.get("question") or golden.question).strip()
        answer_type = str(case_obj.get("answer_type") or golden.answer_type).strip() or golden.answer_type
        chunk_snippets = _as_str_dict(telemetry_obj.get("chunk_snippets"))
        retrieved_chunk_ids = tuple(_as_str_list(telemetry_obj.get("retrieved_chunk_ids")))
        baseline_page_ids = tuple(_as_str_list(telemetry_obj.get("context_page_ids")) or _as_str_list(telemetry_obj.get("used_page_ids")))
        if not baseline_page_ids:
            baseline_page_ids = tuple(_project_pages_from_chunk_ids(list(retrieved_chunk_ids), limit=6))
        family = _route_family(question)
        rerank_ms_raw = telemetry_obj.get("rerank_ms")
        baseline_rerank_ms = max(0, int(float(rerank_ms_raw))) if isinstance(rerank_ms_raw, (int, float, str)) else 0
        cases.append(
            ShadowCase(
                question_id=question_id,
                question=question,
                answer_type=answer_type,
                gold_page_ids=tuple(golden.golden_page_ids),
                baseline_page_ids=baseline_page_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                chunk_snippets=chunk_snippets,
                baseline_rerank_ms=baseline_rerank_ms,
                family=family,
            )
        )
    return cases


def evaluate_shadow_cases(
    cases: list[ShadowCase],
    *,
    reranker: SelectiveICRReranker,
    slice_name: str,
) -> tuple[SliceSummary, list[CaseResult]]:
    """Run the selective ICR shadow reranker on recorded benchmark cases.

    Args:
        cases: Recorded benchmark cases.
        reranker: Selective ICR scorer.
        slice_name: Human-readable slice label.

    Returns:
        Aggregate slice summary and per-case results.
    """

    results: list[CaseResult] = []
    per_family: dict[str, list[CaseResult]] = defaultdict(list)
    for case in cases:
        candidate_pool = [
            RecordedCandidate(chunk_id=chunk_id, text=case.chunk_snippets.get(chunk_id, ""))
            for chunk_id in case.retrieved_chunk_ids
        ]
        top_n = max(1, len(case.baseline_page_ids) or 6)
        shadow_start = perf_counter()
        ranked = reranker.rank(case.question, candidate_pool, top_n=len(candidate_pool))
        shadow_latency_ms = max(0, int((perf_counter() - shadow_start) * 1000.0))
        shadow_page_ids = _project_pages_from_ranked(ranked, limit=top_n)
        baseline_page_ids = list(case.baseline_page_ids)
        gold_page_ids = list(case.gold_page_ids)
        baseline_hit = bool(set(baseline_page_ids) & set(gold_page_ids))
        shadow_hit = bool(set(shadow_page_ids) & set(gold_page_ids))
        baseline_precision = _precision(baseline_page_ids, gold_page_ids)
        shadow_precision = _precision(shadow_page_ids, gold_page_ids)
        result = CaseResult(
            question_id=case.question_id,
            family=case.family,
            gold_page_ids=gold_page_ids,
            baseline_page_ids=baseline_page_ids,
            shadow_page_ids=shadow_page_ids,
            baseline_hit=baseline_hit,
            shadow_hit=shadow_hit,
            baseline_precision=baseline_precision,
            shadow_precision=shadow_precision,
            baseline_rerank_ms=case.baseline_rerank_ms,
            shadow_rerank_ms=shadow_latency_ms,
        )
        results.append(result)
        per_family[case.family].append(result)

    summary = _summarize_slice(slice_name=slice_name, results=results, per_family=per_family)
    return summary, results


def _summarize_slice(
    *,
    slice_name: str,
    results: list[CaseResult],
    per_family: dict[str, list[CaseResult]],
) -> SliceSummary:
    """Summarize one benchmark slice.

    Args:
        slice_name: Human-readable slice label.
        results: Per-case benchmark results.
        per_family: Per-family case breakdown.

    Returns:
        Aggregated slice summary.
    """

    case_count = len(results)
    baseline_hit_rate = _mean([1.0 if result.baseline_hit else 0.0 for result in results])
    shadow_hit_rate = _mean([1.0 if result.shadow_hit else 0.0 for result in results])
    baseline_precision = _mean([result.baseline_precision for result in results])
    shadow_precision = _mean([result.shadow_precision for result in results])
    family_metrics: dict[str, dict[str, float]] = {}
    for family, family_results in sorted(per_family.items()):
        family_metrics[family] = {
            "case_count": float(len(family_results)),
            "baseline_hit_rate": _mean([1.0 if result.baseline_hit else 0.0 for result in family_results]),
            "shadow_hit_rate": _mean([1.0 if result.shadow_hit else 0.0 for result in family_results]),
            "hit_delta": _mean([1.0 if result.shadow_hit else 0.0 for result in family_results])
            - _mean([1.0 if result.baseline_hit else 0.0 for result in family_results]),
            "baseline_precision": _mean([result.baseline_precision for result in family_results]),
            "shadow_precision": _mean([result.shadow_precision for result in family_results]),
            "precision_delta": _mean([result.shadow_precision for result in family_results])
            - _mean([result.baseline_precision for result in family_results]),
        }
    return SliceSummary(
        name=slice_name,
        case_count=case_count,
        baseline_hit_rate=baseline_hit_rate,
        shadow_hit_rate=shadow_hit_rate,
        hit_delta=shadow_hit_rate - baseline_hit_rate,
        baseline_precision=baseline_precision,
        shadow_precision=shadow_precision,
        precision_delta=shadow_precision - baseline_precision,
        baseline_latency_ms_mean=_mean([float(result.baseline_rerank_ms) for result in results]),
        shadow_latency_ms_mean=_mean([float(result.shadow_rerank_ms) for result in results]),
        baseline_latency_ms_p50=_p50([float(result.baseline_rerank_ms) for result in results]),
        shadow_latency_ms_p50=_p50([float(result.shadow_rerank_ms) for result in results]),
        family_metrics=family_metrics,
    )


def _project_pages_from_ranked(ranked: Sequence[object], *, limit: int) -> list[str]:
    """Project ranked candidates to unique platform-valid page IDs."""

    page_ids: list[str] = []
    seen: set[str] = set()
    for item in ranked:
        chunk_id = str(getattr(item, "chunk_id", "") or "").strip()
        page_id = str(getattr(item, "page_id", "") or "").strip() or _chunk_id_to_page_id(chunk_id)
        if not page_id or page_id in seen:
            continue
        seen.add(page_id)
        page_ids.append(page_id)
        if len(page_ids) >= max(0, int(limit)):
            break
    return page_ids


def _project_pages_from_chunk_ids(chunk_ids: list[str], *, limit: int) -> list[str]:
    """Project raw chunk IDs to unique platform-valid page IDs."""

    page_ids: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        page_id = _chunk_id_to_page_id(chunk_id)
        if not page_id or page_id in seen:
            continue
        seen.add(page_id)
        page_ids.append(page_id)
        if len(page_ids) >= max(0, int(limit)):
            break
    return page_ids


def _chunk_id_to_page_id(chunk_id: str) -> str:
    """Convert a chunk identifier into a platform-valid page identifier."""

    if not chunk_id:
        return ""
    if ":" not in chunk_id and "_" in chunk_id:
        return chunk_id
    parts = chunk_id.split(":")
    if len(parts) < 2:
        return ""
    doc_id = parts[0].strip()
    page_raw = parts[1].strip()
    if not doc_id or not page_raw.isdigit():
        return ""
    return f"{doc_id}_{int(page_raw) + 1}"


def _precision(page_ids: list[str], gold_page_ids: list[str]) -> float:
    if not page_ids:
        return 0.0
    return len(set(page_ids) & set(gold_page_ids)) / len(page_ids)


def _route_family(question: str) -> str:
    text = question.casefold()
    if any(term in text for term in ("claimant", "respondent", "party", "caption")):
        return "title_caption_claimant"
    if any(term in text for term in ("article", "section", "schedule", "provision", "clause")):
        return "exact_article_provision_schedule"
    if any(term in text for term in ("authority", "law number", "law no", "date", "commencement", "enactment")):
        return "authority_date_law_number"
    if any(term in text for term in ("compare", "common", "judge", "same", "both")):
        return "compare"
    return "other"


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _p50(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _as_dict(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    return [str(item).strip() for item in items if str(item).strip()]


def _as_str_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    items = cast("dict[str, object]", value)
    return {str(key).strip(): str(raw).strip() for key, raw in items.items() if str(key).strip() and str(raw).strip()}


def _write_markdown(path: Path, report: ShadowBenchmarkReport) -> None:
    """Write a markdown summary for the benchmark report."""

    lines = [
        "# Selective ICR Shadow Benchmark",
        "",
        f"- slices: `{len(report.slices)}`",
        f"- cases: `{len(report.cases)}`",
        "",
        "## Summary",
        "",
        "| slice | cases | baseline hit | shadow hit | delta | baseline precision | shadow precision | baseline rerank ms | shadow rerank ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in report.slices:
        lines.append(
            "| {name} | {case_count} | {baseline_hit_rate:.4f} | {shadow_hit_rate:.4f} | {hit_delta:.4f} | {baseline_precision:.4f} | {shadow_precision:.4f} | {baseline_latency_ms_mean:.1f} | {shadow_latency_ms_mean:.1f} |".format(
                **asdict(summary)
            )
        )
    lines.append("")
    lines.append("## Family Metrics")
    lines.append("")
    for summary in report.slices:
        lines.append(f"### {summary.name}")
        lines.append("")
        lines.append("| family | cases | baseline hit | shadow hit | delta | baseline precision | shadow precision |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for family, metrics in summary.family_metrics.items():
            lines.append(
                "| {family} | {case_count:.0f} | {baseline_hit_rate:.4f} | {shadow_hit_rate:.4f} | {hit_delta:.4f} | {baseline_precision:.4f} | {shadow_precision:.4f} |".format(
                    family=family, **metrics
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Entry point for the selective ICR shadow benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-results",
        type=Path,
        default=Path(
            "/Users/sasha/IdeaProjects/personal_projects/rag_challenge/.sdd/researches/explicit_profile_baseline_2026-03-19/main_vs_explicit_profile/raw_results_answer_stable_replay.json"
        ),
        help="Path to the recorded raw results JSON.",
    )
    parser.add_argument(
        "--reviewed-all",
        type=Path,
        default=Path("/Users/sasha/IdeaProjects/personal_projects/rag_challenge/.sdd/golden/reviewed/reviewed_all_100.json"),
        help="Path to the reviewed all_100 gold slice.",
    )
    parser.add_argument(
        "--reviewed-high",
        type=Path,
        default=Path("/Users/sasha/IdeaProjects/personal_projects/rag_challenge/.sdd/golden/reviewed/reviewed_high_confidence_81.json"),
        help="Path to the reviewed high_confidence_81 gold slice.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/Users/sasha/IdeaProjects/personal_projects/rag_challenge/.sdd/researches/1065_selective_icr_local_rerank_and_provider_exit_r1_2026-03-19"
        ),
        help="Directory for JSON and markdown outputs.",
    )
    parser.add_argument("--model-path", default="", help="Optional local model path for the selective ICR scorer.")
    args = parser.parse_args()

    scorer = SelectiveICRReranker(
        config=SelectiveICRConfig(
            model_path=str(args.model_path).strip(),
            max_chars=1800,
            normalize_scores=True,
            provider_exit=False,
        )
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases = load_shadow_cases(args.raw_results, args.reviewed_all)
    all_summary, all_results = evaluate_shadow_cases(all_cases, reranker=scorer, slice_name="reviewed_all_100")
    high_cases = load_shadow_cases(args.raw_results, args.reviewed_high)
    high_summary, high_results = evaluate_shadow_cases(high_cases, reranker=scorer, slice_name="reviewed_high_confidence_81")

    report = ShadowBenchmarkReport(slices=[all_summary, high_summary], cases=[*all_results, *high_results])
    json_path = output_dir / "selective_icr_shadow_benchmark.json"
    md_path = output_dir / "selective_icr_shadow_benchmark.md"
    json_path.write_text(json.dumps({
        "slices": [asdict(summary) for summary in report.slices],
        "cases": [asdict(case) for case in report.cases],
    }, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
