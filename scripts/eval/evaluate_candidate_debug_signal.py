from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from shafi.config import get_settings
from shafi.eval.judge import JudgeClient, JudgeOutcome
from shafi.eval.metrics import AnswerTypeFormatCompliance, CitationCoverage
from shafi.eval.sources import PdfPageTextProvider, build_sources_text, select_used_pages

if TYPE_CHECKING:
    from collections.abc import Callable

JsonDict = dict[str, Any]
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./-]*")
_OVERLAP_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "into",
    "this",
    "there",
    "question",
    "according",
    "under",
    "law",
    "article",
    "page",
    "pages",
    "judgment",
    "specific",
    "claim",
    "number",
    "case",
    "cases",
    "legal",
    "entity",
    "entities",
    "individual",
    "individuals",
    "party",
    "parties",
    "date",
    "court",
    "appeal",
    "originated",
    "involve",
    "same",
    "yes",
    "no",
    "null",
}


def _coerce_float(value: object, *, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    return [text for item in items if (text := str(item).strip())]


def _coerce_object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return cast("list[object]", value)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    q = min(1.0, max(0.0, float(quantile)))
    index = q * (len(ordered) - 1)
    lo = int(index)
    hi = min(len(ordered) - 1, lo + 1)
    frac = index - lo
    return ordered[lo] + ((ordered[hi] - ordered[lo]) * frac)


@dataclass(frozen=True)
class CandidateCase:
    question_id: str
    question: str
    answer_type: str
    answer_text: str
    telemetry: JsonDict
    ttft_ms: float | None


@dataclass(frozen=True)
class CandidateEvalArtifacts:
    label: str
    eval_path: Path
    judge_path: Path
    payload: JsonDict


@dataclass(frozen=True)
class AttributionSignalRow:
    question_id: str
    gold_pages: list[str]
    false_positive_pages: list[str]
    gold_scores: list[float]
    false_positive_scores: list[float]
    signal_terms: list[str]
    signal_source: str


@dataclass(frozen=True)
class AttributionSignalSummary:
    verdict: str
    evaluated_cases: int
    pairwise_comparisons: int
    gold_beats_false_positive_rate: float
    mean_gold_overlap: float
    mean_false_positive_overlap: float
    mean_gap: float
    rows: list[AttributionSignalRow]


def _load_questions(path: Path) -> dict[str, tuple[str, str]]:
    rows_obj = json.loads(path.read_text(encoding="utf-8"))
    rows = _coerce_object_list(rows_obj)
    out: dict[str, tuple[str, str]] = {}
    for row_obj in rows:
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        qid = str(row.get("id") or row.get("question_id") or "").strip()
        if not qid:
            continue
        out[qid] = (
            str(row.get("question") or "").strip(),
            str(row.get("answer_type") or "").strip(),
        )
    return out


def _normalize_answer_text(value: object) -> str:
    if value is None:
        return "null"
    text = str(value).strip()
    return text if text else "null"


def _normalize_signal_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = raw.strip().lower().strip(".,;:()[]{}")
        if len(token) < 2 or token in _OVERLAP_STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _page_overlap_score(*, answer_text: str, page_text: str) -> tuple[float, list[str]]:
    answer_terms = _normalize_signal_tokens(answer_text)
    if not answer_terms:
        return 0.0, []
    page_tokens = set(_normalize_signal_tokens(page_text))
    hits = [token for token in answer_terms if token in page_tokens]
    return len(hits) / max(1, len(answer_terms)), hits


def _is_unanswerable_answer(answer_text: str) -> bool:
    normalized = answer_text.strip().lower()
    return (
        normalized in {"", "null", "none"}
        or normalized.startswith("there is no information on this question")
        or "insufficient sources retrieved" in normalized
    )


def _signal_text_for_case(case: CandidateCase) -> tuple[str, list[str], str]:
    answer_text = case.answer_text.strip()
    answer_terms = _normalize_signal_tokens(answer_text)
    answer_type = case.answer_type.strip().lower()
    if answer_terms and not (answer_type == "boolean" and answer_text.lower() in {"yes", "no", "true", "false"}):
        return answer_text, answer_terms, "answer"

    combined_text = " ".join(part for part in (case.question.strip(), answer_text) if part).strip()
    combined_terms = _normalize_signal_tokens(combined_text)
    if combined_terms:
        return combined_text, combined_terms, "question+answer"

    return answer_text, answer_terms, "answer"


def _load_page_benchmark(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows_obj = payload.get("cases", [])
    rows = _coerce_object_list(rows_obj)
    out: dict[str, list[str]] = {}
    for row_obj in rows:
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        if str(row.get("trust_tier") or "").strip().lower() != "trusted":
            continue
        qid = str(row.get("question_id") or "").strip()
        gold_pages = _coerce_str_list(row.get("gold_page_ids"))
        if qid and gold_pages:
            out[qid] = gold_pages
    return out


def _parse_page_id(page_id: str) -> tuple[str, int] | None:
    raw = page_id.strip()
    if not raw or "_" not in raw:
        return None
    doc_id, _, page_raw = raw.rpartition("_")
    if not doc_id or not page_raw.isdigit():
        return None
    page = int(page_raw)
    if page <= 0:
        return None
    return doc_id, page


def _load_raw_results(path: Path, *, questions: dict[str, tuple[str, str]]) -> dict[str, CandidateCase]:
    rows_obj = json.loads(path.read_text(encoding="utf-8"))
    rows = _coerce_object_list(rows_obj)
    out: dict[str, CandidateCase] = {}
    for row_obj in rows:
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        case_obj = row.get("case")
        case = cast("JsonDict", case_obj) if isinstance(case_obj, dict) else {}
        qid = str(case.get("case_id") or row.get("question_id") or "").strip()
        if not qid:
            continue
        question, answer_type = questions.get(qid, ("", ""))
        question = str(case.get("question") or question or "").strip()
        answer_type = str(case.get("answer_type") or answer_type or "").strip()
        telemetry_obj = row.get("telemetry")
        telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        ttft_ms = _coerce_float(telemetry.get("ttft_ms"))
        out[qid] = CandidateCase(
            question_id=qid,
            question=question,
            answer_type=answer_type,
            answer_text=_normalize_answer_text(row.get("answer_text")),
            telemetry=telemetry,
            ttft_ms=ttft_ms,
        )
    return out


def _used_pages(case: CandidateCase) -> list[str]:
    return select_used_pages(case.telemetry, max_pages=12)


def _answer_changed(baseline: CandidateCase, candidate: CandidateCase) -> bool:
    return baseline.answer_text.strip() != candidate.answer_text.strip()


def _pages_changed(baseline: CandidateCase, candidate: CandidateCase) -> bool:
    return _used_pages(baseline) != _used_pages(candidate)


def _select_qids(
    *,
    baseline_cases: dict[str, CandidateCase],
    candidate_cases: dict[str, CandidateCase],
    scope: str,
    include_qids: set[str],
) -> list[str]:
    baseline_qids = set(baseline_cases)
    candidate_qids = set(candidate_cases)
    common_qids = baseline_qids.intersection(candidate_qids)
    scope_key = scope.strip().lower()
    if scope_key == "all":
        selected = sorted(common_qids)
    else:
        selected = sorted(
            qid
            for qid in common_qids
            if _answer_changed(baseline_cases[qid], candidate_cases[qid])
            or _pages_changed(baseline_cases[qid], candidate_cases[qid])
        )
    if include_qids:
        selected = sorted(set(selected).union(include_qids).intersection(common_qids))
    return selected


def _citation_coverage(case: CandidateCase) -> float:
    answer = case.answer_text.strip()
    telemetry = case.telemetry
    is_null = answer.lower() == "null"
    is_no_info = answer.lower().startswith("there is no information on this question")
    if is_null or is_no_info:
        return 1.0
    cited_ids = _coerce_str_list(telemetry.get("cited_chunk_ids"))
    context_ids = _coerce_str_list(telemetry.get("context_chunk_ids"))
    if cited_ids and context_ids:
        return len(set(cited_ids).intersection(set(context_ids))) / max(1, len(set(cited_ids)))
    parsed_cited_ids = CitationCoverage.extract_cited_chunk_ids(answer)
    if parsed_cited_ids and context_ids:
        return len(parsed_cited_ids.intersection(set(context_ids))) / max(1, len(parsed_cited_ids))
    return 0.0


def _should_judge(*, answer_type: str, judge_scope: str) -> bool:
    scope_key = judge_scope.strip().lower()
    if scope_key == "none":
        return False
    if scope_key == "all":
        return True
    return answer_type.strip().lower() == "free_text"


async def _evaluate_single_case(
    *,
    case: CandidateCase,
    judge_scope: str,
    judge_client: JudgeClient | None,
    pdf_provider: PdfPageTextProvider,
) -> tuple[JsonDict, JsonDict | None]:
    answer_type_key = case.answer_type.strip().lower() or "free_text"
    used_pages = _used_pages(case)
    coverage = _citation_coverage(case)
    format_compliance = 1.0 if AnswerTypeFormatCompliance.is_answer_format_compliant(case.answer_text, case.answer_type) else 0.0

    judge_payload: JsonDict | None = None
    judge_record: JsonDict | None = None
    judge_failure = ""
    judge_sources_sha256 = ""
    if judge_client is not None and _should_judge(answer_type=answer_type_key, judge_scope=judge_scope):
        sources_text = await build_sources_text(
            used_pages,
            pdf_provider=pdf_provider,
            qdrant_fallback=None,
            max_chars_total=int(getattr(get_settings().judge, "sources_max_chars_total", 60000)),
        )
        judge_sources_sha256 = hashlib.sha256(sources_text.encode("utf-8")).hexdigest()
        judge_outcome: JudgeOutcome = await judge_client.evaluate(
            question=case.question,
            answer_type=answer_type_key,
            answer=case.answer_text,
            used_pages=used_pages,
            sources_text=sources_text,
        )
        judge_failure = judge_outcome.failure
        if judge_outcome.result is not None:
            judge_payload = judge_outcome.result.model_dump(mode="json")
        judge_record = {
            "case_id": case.question_id,
            "question_id": case.question_id,
            "answer_type": answer_type_key,
            "question": case.question,
            "answer": case.answer_text,
            "used_pages": used_pages,
            "sources_text_sha256": judge_sources_sha256,
            "judge_model": judge_outcome.model,
            "judge_failure": judge_outcome.failure,
            "judge_result": judge_payload,
        }

    record: JsonDict = {
        "case_id": case.question_id,
        "question_id": case.question_id,
        "question": case.question,
        "answer_type": case.answer_type,
        "answer": case.answer_text,
        "ttft_ms": None if case.ttft_ms is None else round(case.ttft_ms, 1),
        "citation_coverage": round(coverage, 4),
        "format_compliance": round(format_compliance, 4),
        "telemetry": case.telemetry,
        "used_pages": used_pages,
        "judge_sources_sha256": judge_sources_sha256,
    }
    if judge_payload is not None:
        record["judge"] = judge_payload
    if judge_failure:
        record["judge_failure"] = judge_failure
    return record, judge_record


def _classify_attribution_signal(
    *,
    pairwise_comparisons: int,
    gold_beats_false_positive_rate: float,
    mean_gap: float,
) -> str:
    if pairwise_comparisons >= 5 and gold_beats_false_positive_rate >= 0.75 and mean_gap >= 0.10:
        return "real signal"
    if pairwise_comparisons >= 3 and gold_beats_false_positive_rate >= 0.60 and mean_gap >= 0.03:
        return "weak signal"
    return "noise"


def _build_answer_to_page_attribution_signal(
    *,
    cases_by_qid: dict[str, CandidateCase],
    selected_qids: list[str],
    gold_pages_by_qid: dict[str, list[str]],
    page_text_for: Callable[[str], str | None],
) -> AttributionSignalSummary | None:
    rows: list[AttributionSignalRow] = []
    gold_scores_flat: list[float] = []
    false_positive_scores_flat: list[float] = []
    pairwise_wins = 0
    pairwise_total = 0

    for qid in selected_qids:
        gold_pages = gold_pages_by_qid.get(qid, [])
        case = cases_by_qid.get(qid)
        if case is None or not gold_pages or _is_unanswerable_answer(case.answer_text):
            continue

        used_pages = _used_pages(case)
        false_positive_pages = [page_id for page_id in used_pages if page_id not in set(gold_pages)]
        if not false_positive_pages:
            continue

        signal_text, signal_terms, signal_source = _signal_text_for_case(case)
        if not signal_terms:
            continue

        gold_scores: list[float] = []
        for page_id in gold_pages:
            page_text = page_text_for(page_id) or ""
            score, _ = _page_overlap_score(answer_text=signal_text, page_text=page_text)
            gold_scores.append(round(score, 4))

        false_positive_scores: list[float] = []
        for page_id in false_positive_pages:
            page_text = page_text_for(page_id) or ""
            score, _ = _page_overlap_score(answer_text=signal_text, page_text=page_text)
            false_positive_scores.append(round(score, 4))

        if not gold_scores or not false_positive_scores:
            continue

        for gold_score in gold_scores:
            for false_positive_score in false_positive_scores:
                if gold_score > false_positive_score:
                    pairwise_wins += 1
                pairwise_total += 1

        gold_scores_flat.extend(gold_scores)
        false_positive_scores_flat.extend(false_positive_scores)
        rows.append(
            AttributionSignalRow(
                question_id=qid,
                gold_pages=gold_pages,
                false_positive_pages=false_positive_pages,
                gold_scores=gold_scores,
                false_positive_scores=false_positive_scores,
                signal_terms=signal_terms,
                signal_source=signal_source,
            )
        )

    if not rows or not gold_scores_flat or not false_positive_scores_flat:
        return None

    mean_gold = sum(gold_scores_flat) / len(gold_scores_flat)
    mean_false = sum(false_positive_scores_flat) / len(false_positive_scores_flat)
    mean_gap = mean_gold - mean_false
    pairwise_rate = pairwise_wins / max(1, pairwise_total)
    return AttributionSignalSummary(
        verdict=_classify_attribution_signal(
            pairwise_comparisons=pairwise_total,
            gold_beats_false_positive_rate=pairwise_rate,
            mean_gap=mean_gap,
        ),
        evaluated_cases=len(rows),
        pairwise_comparisons=pairwise_total,
        gold_beats_false_positive_rate=pairwise_rate,
        mean_gold_overlap=mean_gold,
        mean_false_positive_overlap=mean_false,
        mean_gap=mean_gap,
        rows=rows,
    )


async def _evaluate_artifact(
    *,
    label: str,
    cases_by_qid: dict[str, CandidateCase],
    selected_qids: list[str],
    judge_scope: str,
    docs_dir: Path,
    out_dir: Path,
) -> CandidateEvalArtifacts:
    max_chars_per_page = 20000
    judge_enabled = False
    if judge_scope.strip().lower() != "none":
        settings = get_settings()
        max_chars_per_page = int(getattr(settings.judge, "sources_max_chars_per_page", 20000))
        judge_enabled = bool(settings.judge.enabled) and bool(settings.judge.api_key.get_secret_value().strip())
    judge_client: JudgeClient | None = JudgeClient() if judge_enabled else None
    pdf_provider = PdfPageTextProvider(
        docs_dir,
        max_chars_per_page=max_chars_per_page,
    )
    ttft_values: list[float] = []
    coverage_sum = 0.0
    format_sum = 0.0
    judge_cases = 0
    judge_passes = 0
    judge_failures = 0
    judge_accuracy_sum = 0
    judge_grounding_sum = 0
    judge_clarity_sum = 0
    judge_uncertainty_sum = 0
    case_records: list[JsonDict] = []
    judge_rows: list[JsonDict] = []
    try:
        for qid in selected_qids:
            case = cases_by_qid[qid]
            record, judge_row = await _evaluate_single_case(
                case=case,
                judge_scope=judge_scope,
                judge_client=judge_client,
                pdf_provider=pdf_provider,
            )
            case_records.append(record)
            coverage_sum += float(record["citation_coverage"])
            format_sum += float(record["format_compliance"])
            ttft_ms = _coerce_float(record.get("ttft_ms"))
            if ttft_ms is not None:
                ttft_values.append(ttft_ms)
            judge_payload = record.get("judge")
            if isinstance(judge_payload, dict):
                judge_payload_dict = cast("JsonDict", judge_payload)
                judge_cases += 1
                verdict = str(judge_payload_dict.get("verdict") or "").strip().upper()
                if verdict == "PASS":
                    judge_passes += 1
                scores_obj = judge_payload_dict.get("scores")
                scores = cast("JsonDict", scores_obj) if isinstance(scores_obj, dict) else {}
                judge_accuracy_sum += int(scores.get("accuracy") or 0)
                judge_grounding_sum += int(scores.get("grounding") or 0)
                judge_clarity_sum += int(scores.get("clarity") or 0)
                judge_uncertainty_sum += int(scores.get("uncertainty_handling") or 0)
            elif record.get("judge_failure"):
                judge_failures += 1
            if judge_row is not None:
                judge_rows.append(judge_row)
    finally:
        if judge_client is not None:
            await judge_client.close()
        pdf_provider.close()

    summary: JsonDict = {
        "total_cases": len(selected_qids),
        "answer_type_cases": len(selected_qids),
        "citation_coverage": round(coverage_sum / max(1, len(selected_qids)), 4),
        "answer_type_format_compliance": round(format_sum / max(1, len(selected_qids)), 4),
        "ttft_p50_ms": None if not ttft_values else round(_percentile(ttft_values, 0.5) or 0.0, 1),
        "ttft_p95_ms": None if not ttft_values else round(_percentile(ttft_values, 0.95) or 0.0, 1),
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

    payload: JsonDict = {
        "label": label,
        "selected_qids": selected_qids,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        "summary": summary,
        "cases": case_records,
        "failures": [],
    }
    eval_path = out_dir / f"eval_candidate_debug_{label}.json"
    judge_path = out_dir / f"judge_candidate_debug_{label}.jsonl"
    eval_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    judge_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in judge_rows),
        encoding="utf-8",
    )
    return CandidateEvalArtifacts(label=label, eval_path=eval_path, judge_path=judge_path, payload=payload)


def _diff_case_rows(
    *,
    qid: str,
    baseline_eval: JsonDict,
    candidate_eval: JsonDict,
) -> str:
    baseline_cases = {str(row.get("question_id") or row.get("case_id") or "").strip(): row for row in cast("list[JsonDict]", baseline_eval.get("cases", []))}
    candidate_cases = {str(row.get("question_id") or row.get("case_id") or "").strip(): row for row in cast("list[JsonDict]", candidate_eval.get("cases", []))}
    baseline_row = baseline_cases[qid]
    candidate_row = candidate_cases[qid]
    baseline_judge = cast("JsonDict", baseline_row.get("judge")) if isinstance(baseline_row.get("judge"), dict) else {}
    candidate_judge = cast("JsonDict", candidate_row.get("judge")) if isinstance(candidate_row.get("judge"), dict) else {}
    baseline_scores = cast("JsonDict", baseline_judge.get("scores")) if isinstance(baseline_judge.get("scores"), dict) else {}
    candidate_scores = cast("JsonDict", candidate_judge.get("scores")) if isinstance(candidate_judge.get("scores"), dict) else {}
    lines = [
        f"### `{qid}`",
        "",
        f"- `question`: {candidate_row.get('question') or baseline_row.get('question') or 'n/a'}",
        f"- `baseline_answer`: `{baseline_row.get('answer')}`",
        f"- `candidate_answer`: `{candidate_row.get('answer')}`",
        f"- `baseline_used_pages`: `{baseline_row.get('used_pages')}`",
        f"- `candidate_used_pages`: `{candidate_row.get('used_pages')}`",
        f"- `baseline_judge_verdict`: `{baseline_judge.get('verdict', 'n/a')}` | `candidate_judge_verdict`: `{candidate_judge.get('verdict', 'n/a')}`",
        f"- `baseline_scores`: `{baseline_scores}`",
        f"- `candidate_scores`: `{candidate_scores}`",
        "",
    ]
    return "\n".join(lines)


def _build_compare_markdown(
    *,
    baseline: CandidateEvalArtifacts,
    candidate: CandidateEvalArtifacts,
    selected_qids: list[str],
    baseline_attribution: JsonDict | None,
    candidate_attribution: JsonDict | None,
) -> str:
    baseline_summary = cast("JsonDict", baseline.payload.get("summary", {}))
    candidate_summary = cast("JsonDict", candidate.payload.get("summary", {}))
    baseline_judge = cast("JsonDict", baseline_summary.get("judge")) if isinstance(baseline_summary.get("judge"), dict) else {}
    candidate_judge = cast("JsonDict", candidate_summary.get("judge")) if isinstance(candidate_summary.get("judge"), dict) else {}
    lines = [
        "# Candidate Debug Signal",
        "",
        f"- `baseline`: `{baseline.label}`",
        f"- `candidate`: `{candidate.label}`",
        f"- `selected_qids`: `{selected_qids}`",
        "- `submission_policy`: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "## Summary",
        "",
        "| Artifact | Cases | Judge Pass Rate | Avg Grounding | Avg Accuracy | Avg Clarity | Citation | Format | TTFT p50 |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
        f"| `{baseline.label}` | {baseline_summary.get('total_cases', 0)} | {baseline_judge.get('pass_rate', 'n/a')} | {baseline_judge.get('avg_grounding', 'n/a')} | {baseline_judge.get('avg_accuracy', 'n/a')} | {baseline_judge.get('avg_clarity', 'n/a')} | {baseline_summary.get('citation_coverage', 'n/a')} | {baseline_summary.get('answer_type_format_compliance', 'n/a')} | {baseline_summary.get('ttft_p50_ms', 'n/a')} |",
        f"| `{candidate.label}` | {candidate_summary.get('total_cases', 0)} | {candidate_judge.get('pass_rate', 'n/a')} | {candidate_judge.get('avg_grounding', 'n/a')} | {candidate_judge.get('avg_accuracy', 'n/a')} | {candidate_judge.get('avg_clarity', 'n/a')} | {candidate_summary.get('citation_coverage', 'n/a')} | {candidate_summary.get('answer_type_format_compliance', 'n/a')} | {candidate_summary.get('ttft_p50_ms', 'n/a')} |",
        "",
        "## Attribution Falsifier",
        "",
        "| Artifact | Verdict | Cases | Pairwise | Gold>FP Rate | Mean Gold | Mean FP | Mean Gap |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|",
        _attribution_markdown_row(label=baseline.label, payload=baseline_attribution),
        _attribution_markdown_row(label=candidate.label, payload=candidate_attribution),
        "",
        "## Per-Case Delta",
        "",
    ]
    for qid in selected_qids:
        lines.append(_diff_case_rows(qid=qid, baseline_eval=baseline.payload, candidate_eval=candidate.payload))
    return "\n".join(lines).rstrip() + "\n"


def _attribution_markdown_row(*, label: str, payload: JsonDict | None) -> str:
    if payload is None:
        return f"| `{label}` | `not_run` | 0 | 0 | n/a | n/a | n/a | n/a |"
    return (
        f"| `{label}` | `{payload.get('verdict', 'n/a')}` | {payload.get('evaluated_cases', 0)} | "
        f"{payload.get('pairwise_comparisons', 0)} | {payload.get('gold_beats_false_positive_rate', 'n/a')} | "
        f"{payload.get('mean_gold_overlap', 'n/a')} | {payload.get('mean_false_positive_overlap', 'n/a')} | "
        f"{payload.get('mean_gap', 'n/a')} |"
    )


def _load_qids_file(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw-results artifacts with the local debug judge loop.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--candidate-raw-results", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-scope", choices=("changed", "all"), default="changed")
    parser.add_argument("--judge-scope", choices=("all", "free_text", "none"), default="all")
    parser.add_argument("--include-qids-file", type=Path, default=None)
    parser.add_argument("--page-benchmark", type=Path, default=None)
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = _load_questions(args.questions.resolve())
    baseline_cases = _load_raw_results(args.baseline_raw_results.resolve(), questions=questions)
    candidate_cases = _load_raw_results(args.candidate_raw_results.resolve(), questions=questions)
    include_qids = _load_qids_file(args.include_qids_file.resolve() if args.include_qids_file is not None else None)
    selected_qids = _select_qids(
        baseline_cases=baseline_cases,
        candidate_cases=candidate_cases,
        scope=str(args.case_scope),
        include_qids=include_qids,
    )

    baseline_eval = await _evaluate_artifact(
        label=str(args.baseline_label),
        cases_by_qid=baseline_cases,
        selected_qids=selected_qids,
        judge_scope=str(args.judge_scope),
        docs_dir=args.docs_dir.resolve(),
        out_dir=out_dir,
    )
    candidate_eval = await _evaluate_artifact(
        label=str(args.candidate_label),
        cases_by_qid=candidate_cases,
        selected_qids=selected_qids,
        judge_scope=str(args.judge_scope),
        docs_dir=args.docs_dir.resolve(),
        out_dir=out_dir,
    )

    baseline_attribution_payload: JsonDict | None = None
    candidate_attribution_payload: JsonDict | None = None
    page_benchmark = getattr(args, "page_benchmark", None)
    if page_benchmark is not None:
        gold_pages_by_qid = _load_page_benchmark(page_benchmark.resolve())
        pdf_provider = PdfPageTextProvider(
            args.docs_dir.resolve(),
            max_chars_per_page=20000,
        )
        try:
            def _page_text_for(page_id: str) -> str | None:
                parsed = _parse_page_id(page_id)
                if parsed is None:
                    return None
                doc_id, page = parsed
                return pdf_provider.get_page_text(doc_id=doc_id, page=page)

            baseline_attribution = _build_answer_to_page_attribution_signal(
                cases_by_qid=baseline_cases,
                selected_qids=selected_qids,
                gold_pages_by_qid=gold_pages_by_qid,
                page_text_for=_page_text_for,
            )
            candidate_attribution = _build_answer_to_page_attribution_signal(
                cases_by_qid=candidate_cases,
                selected_qids=selected_qids,
                gold_pages_by_qid=gold_pages_by_qid,
                page_text_for=_page_text_for,
            )
        finally:
            pdf_provider.close()

        if baseline_attribution is not None:
            baseline_attribution_payload = {
                "verdict": baseline_attribution.verdict,
                "evaluated_cases": baseline_attribution.evaluated_cases,
                "pairwise_comparisons": baseline_attribution.pairwise_comparisons,
                "gold_beats_false_positive_rate": round(baseline_attribution.gold_beats_false_positive_rate, 4),
                "mean_gold_overlap": round(baseline_attribution.mean_gold_overlap, 4),
                "mean_false_positive_overlap": round(baseline_attribution.mean_false_positive_overlap, 4),
                "mean_gap": round(baseline_attribution.mean_gap, 4),
                "cases": [
                    {
                        "question_id": row.question_id,
                        "gold_pages": row.gold_pages,
                        "false_positive_pages": row.false_positive_pages,
                        "gold_scores": row.gold_scores,
                        "false_positive_scores": row.false_positive_scores,
                        "signal_terms": row.signal_terms,
                        "signal_source": row.signal_source,
                    }
                    for row in baseline_attribution.rows
                ],
            }
        if candidate_attribution is not None:
            candidate_attribution_payload = {
                "verdict": candidate_attribution.verdict,
                "evaluated_cases": candidate_attribution.evaluated_cases,
                "pairwise_comparisons": candidate_attribution.pairwise_comparisons,
                "gold_beats_false_positive_rate": round(candidate_attribution.gold_beats_false_positive_rate, 4),
                "mean_gold_overlap": round(candidate_attribution.mean_gold_overlap, 4),
                "mean_false_positive_overlap": round(candidate_attribution.mean_false_positive_overlap, 4),
                "mean_gap": round(candidate_attribution.mean_gap, 4),
                "cases": [
                    {
                        "question_id": row.question_id,
                        "gold_pages": row.gold_pages,
                        "false_positive_pages": row.false_positive_pages,
                        "gold_scores": row.gold_scores,
                        "false_positive_scores": row.false_positive_scores,
                        "signal_terms": row.signal_terms,
                        "signal_source": row.signal_source,
                    }
                    for row in candidate_attribution.rows
                ],
            }

    compare_payload: JsonDict = {
        "baseline_label": baseline_eval.label,
        "candidate_label": candidate_eval.label,
        "selected_qids": selected_qids,
        "case_scope": str(args.case_scope),
        "judge_scope": str(args.judge_scope),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        "baseline_eval": str(baseline_eval.eval_path),
        "candidate_eval": str(candidate_eval.eval_path),
        "baseline_judge": str(baseline_eval.judge_path),
        "candidate_judge": str(candidate_eval.judge_path),
        "baseline_answer_to_page_signal": baseline_attribution_payload,
        "candidate_answer_to_page_signal": candidate_attribution_payload,
    }
    compare_json = out_dir / f"candidate_debug_compare_{args.candidate_label}_vs_{args.baseline_label}.json"
    compare_md = out_dir / f"candidate_debug_compare_{args.candidate_label}_vs_{args.baseline_label}.md"
    compare_json.write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compare_md.write_text(
        _build_compare_markdown(
            baseline=baseline_eval,
            candidate=candidate_eval,
            selected_qids=selected_qids,
            baseline_attribution=baseline_attribution_payload,
            candidate_attribution=candidate_attribution_payload,
        ),
        encoding="utf-8",
    )


def main() -> None:
    asyncio.run(_async_main(parse_args()))


if __name__ == "__main__":
    main()
