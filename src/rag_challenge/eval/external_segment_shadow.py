"""Offline shadow benchmark for the external segment payload."""

from __future__ import annotations

import json
import re
from collections import Counter
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from rag_challenge.ingestion.external_segment_payload import (
    ExternalSegmentPayload,
    ExternalSegmentRecord,
    load_external_segment_payload,
    normalize_shadow_text,
    tokenize_shadow_text,
)
from rag_challenge.ingestion.rich_segment_text import SegmentTextMode, analyze_segment_noise, compose_segment_text

if TYPE_CHECKING:
    from pathlib import Path


type JsonObject = dict[str, object]
type JsonList = list[object]


_ARTICLE_RE = re.compile(
    r"\b(article|section|schedule|part|clause|provision|rule|regulation)\b",
    re.IGNORECASE,
)
_AUTHORITY_RE = re.compile(
    r"\b(authority|issued by|legislative authority|law no\.?|date of issue|date of enactment|commencement|enactment)\b",
    re.IGNORECASE,
)
_CLAIMANT_RE = re.compile(r"\b(claimant|claimants|caption|party|parties|appellant|respondent)\b", re.IGNORECASE)


class ExternalSegmentFamily(StrEnum):
    """Question-family routes for the shadow benchmark."""

    TITLE_CAPTION_CLAIMANT = "title_caption_claimant"
    EXACT_PROVISION = "exact_article_provision_schedule"
    AUTHORITY_DATE_LAW_NUMBER = "authority_date_law_number"
    OTHER = "other"


class ExternalSegmentShadowCase(BaseModel):
    """One offline benchmark case for segment projection.

    Args:
        question_id: Stable question identifier.
        question: Question text.
        doc_refs: Declared doc refs from telemetry.
        gold_page_ids: Platform-valid target pages.
        baseline_page_ids: Current chunk/page path projection.
    """

    question_id: str
    question: str
    doc_refs: list[str] = Field(default_factory=list)
    gold_page_ids: list[str] = Field(default_factory=list)
    baseline_page_ids: list[str] = Field(default_factory=list)


class ExternalSegmentRetrievalHit(BaseModel):
    """Per-case retrieval outcome."""

    question_id: str
    family: ExternalSegmentFamily
    gold_page_ids: list[str]
    baseline_page_ids: list[str]
    projected_page_ids: list[str]
    matched_segment_ids: list[str]
    baseline_hit: bool
    projected_hit: bool
    projected_precision: float


class ExternalSegmentFamilyMetrics(BaseModel):
    """Aggregate metrics for one routed family."""

    family: ExternalSegmentFamily
    case_count: int = 0
    baseline_hit_rate: float = 0.0
    projected_hit_rate: float = 0.0
    projected_precision: float = 0.0
    hit_delta: float = 0.0


class ExternalSegmentShadowSummary(BaseModel):
    """Full benchmark summary."""

    composer_mode: SegmentTextMode
    payload_schema: dict[str, list[str] | str]
    family_metrics: list[ExternalSegmentFamilyMetrics]
    cases: list[ExternalSegmentRetrievalHit]


class ExternalSegmentAblationSummary(BaseModel):
    """Plain-vs-rich segment composition comparison."""

    plain_summary: ExternalSegmentShadowSummary
    rich_summary: ExternalSegmentShadowSummary
    title_header_noise_rate: dict[str, float]
    avg_projected_page_count: dict[str, float]


def load_shadow_benchmark_cases(path: Path) -> list[ExternalSegmentShadowCase]:
    """Load benchmark cases from an eval or raw-results artifact.

    Args:
        path: Input JSON artifact path.

    Returns:
        Parsed benchmark cases.
    """

    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    records_obj: object
    if isinstance(raw, dict):
        raw_object = cast("JsonObject", raw)
        records_obj = raw_object.get("cases") or raw_object.get("results") or raw_object.get("rows") or []
    else:
        records_obj = raw
    records = cast("JsonList", records_obj) if isinstance(records_obj, list) else []

    cases: list[ExternalSegmentShadowCase] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        item_obj = cast("JsonObject", item)
        case = item_obj.get("case")
        telemetry = item_obj.get("telemetry")
        if isinstance(case, dict):
            case_obj = cast("JsonObject", case)
            telemetry_obj = cast("JsonObject", telemetry) if isinstance(telemetry, dict) else {}
            question_id = _coerce_str(case_obj.get("question_id")) or _coerce_str(case_obj.get("case_id"))
            question = _coerce_str(case_obj.get("question"))
            doc_refs = _coerce_str_list(telemetry_obj.get("doc_refs"))
            gold_page_ids = _coerce_str_list(telemetry_obj.get("used_page_ids"))
            baseline_page_ids = _coerce_str_list(telemetry_obj.get("retrieved_page_ids"))
        else:
            case_telemetry = item_obj.get("telemetry")
            telemetry_obj = cast("JsonObject", case_telemetry) if isinstance(case_telemetry, dict) else {}
            question_id = _coerce_str(item_obj.get("question_id")) or _coerce_str(item_obj.get("case_id"))
            question = _coerce_str(item_obj.get("question"))
            doc_refs = _coerce_str_list(telemetry_obj.get("doc_refs"))
            gold_page_ids = (
                _coerce_str_list(item_obj.get("gold_page_ids"))
                or _coerce_str_list(item_obj.get("used_pages"))
                or _coerce_str_list(telemetry_obj.get("used_page_ids"))
            )
            baseline_page_ids = (
                _coerce_str_list(item_obj.get("retrieved_page_ids"))
                or _coerce_str_list(telemetry_obj.get("retrieved_page_ids"))
            )
        if not question_id or not question or not gold_page_ids:
            continue
        cases.append(
            ExternalSegmentShadowCase(
                question_id=question_id,
                question=question,
                doc_refs=doc_refs,
                gold_page_ids=gold_page_ids,
                baseline_page_ids=baseline_page_ids,
            )
        )
    return cases


def route_external_segment_family(question: str, *, doc_refs: list[str]) -> ExternalSegmentFamily:
    """Route a question into one of the family clusters.

    Args:
        question: Question text.
        doc_refs: Declared document references.

    Returns:
        Routed family label.
    """

    text = f"{question}\n{' '.join(doc_refs)}"
    if _CLAIMANT_RE.search(text):
        return ExternalSegmentFamily.TITLE_CAPTION_CLAIMANT
    if _ARTICLE_RE.search(text):
        return ExternalSegmentFamily.EXACT_PROVISION
    if _AUTHORITY_RE.search(text):
        return ExternalSegmentFamily.AUTHORITY_DATE_LAW_NUMBER
    return ExternalSegmentFamily.OTHER


def evaluate_external_segment_shadow(
    *,
    payload: ExternalSegmentPayload,
    cases: list[ExternalSegmentShadowCase],
    projected_top_k: int = 3,
    candidate_pool_size: int = 24,
    composer_mode: SegmentTextMode = SegmentTextMode.RICH,
) -> ExternalSegmentShadowSummary:
    """Evaluate routed shadow retrieval against platform-valid pages.

    Args:
        payload: Loaded external segment payload.
        cases: Benchmark cases.
        projected_top_k: Max projected pages per case.
        candidate_pool_size: Max segment candidates scored per case.

    Returns:
        Benchmark summary with per-family metrics and per-case outcomes.
    """

    hits: list[ExternalSegmentRetrievalHit] = []
    for case in cases:
        family = route_external_segment_family(case.question, doc_refs=case.doc_refs)
        ranked_segments = retrieve_external_segments(
            payload=payload,
            case=case,
            family=family,
            limit=candidate_pool_size,
            composer_mode=composer_mode,
        )
        projected_pages = project_segment_pages(ranked_segments, limit=projected_top_k)
        gold_pages = set(case.gold_page_ids)
        matched_segment_ids = [segment.segment_id for segment in ranked_segments if segment.page_id in gold_pages]
        projected_hit = any(page_id in gold_pages for page_id in projected_pages)
        precision = (
            sum(1 for page_id in projected_pages if page_id in gold_pages) / len(projected_pages)
            if projected_pages
            else 0.0
        )
        hits.append(
            ExternalSegmentRetrievalHit(
                question_id=case.question_id,
                family=family,
                gold_page_ids=list(case.gold_page_ids),
                baseline_page_ids=list(case.baseline_page_ids),
                projected_page_ids=projected_pages,
                matched_segment_ids=matched_segment_ids,
                baseline_hit=any(page_id in gold_pages for page_id in case.baseline_page_ids),
                projected_hit=projected_hit,
                projected_precision=precision,
            )
        )

    return ExternalSegmentShadowSummary(
        composer_mode=composer_mode,
        payload_schema={
            "root_keys": ["embedding_model", "segments_path", "output_cache_name", "segments"],
            "segment_keys": [
                "segment_id",
                "doc_id",
                "page_number",
                "text",
                "title",
                "structure_type",
                "hierarchy",
                "context_text",
                "embedding_text",
                "metadata",
            ],
            "metadata_keys": ["case_refs", "law_refs", "token_count", "document_descriptor"],
        },
        family_metrics=_aggregate_family_metrics(hits),
        cases=hits,
    )


def retrieve_external_segments(
    *,
    payload: ExternalSegmentPayload,
    case: ExternalSegmentShadowCase,
    family: ExternalSegmentFamily,
    limit: int,
    composer_mode: SegmentTextMode,
) -> list[ExternalSegmentRecord]:
    """Score and rank external segments for one benchmark case.

    Args:
        payload: Loaded payload.
        case: Benchmark case.
        family: Routed family label.
        limit: Max segment count to return.

    Returns:
        Ranked segment list.
    """

    filtered = _filter_segments_for_case(payload.segments, case=case)
    query_tokens = Counter(tokenize_shadow_text(case.question))
    ref_tokens = Counter(tokenize_shadow_text(" ".join(case.doc_refs)))
    ranked = sorted(
        filtered,
        key=lambda segment: _segment_rank_key(
            segment=segment,
            family=family,
            query_tokens=query_tokens,
            ref_tokens=ref_tokens,
            composer_mode=composer_mode,
        ),
        reverse=True,
    )
    return ranked[:limit]


def project_segment_pages(segments: list[ExternalSegmentRecord], *, limit: int) -> list[str]:
    """Project ranked segments back to unique platform-valid pages.

    Args:
        segments: Ranked segments.
        limit: Maximum number of unique page IDs.

    Returns:
        Ordered unique page IDs.
    """

    page_ids: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        if segment.page_id in seen:
            continue
        seen.add(segment.page_id)
        page_ids.append(segment.page_id)
        if len(page_ids) >= limit:
            break
    return page_ids


def render_external_segment_shadow_markdown(summary: ExternalSegmentShadowSummary) -> str:
    """Render a compact markdown report.

    Args:
        summary: Benchmark summary.

    Returns:
        Markdown report text.
    """

    lines = [
        "# External Segment Shadow Benchmark",
        "",
        "## Routed families",
        "",
        "| family | cases | baseline_hit_rate | projected_hit_rate | hit_delta | projected_precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metrics in summary.family_metrics:
        lines.append(
            f"| {metrics.family.value} | {metrics.case_count} | "
            f"{metrics.baseline_hit_rate:.3f} | {metrics.projected_hit_rate:.3f} | "
            f"{metrics.hit_delta:.3f} | {metrics.projected_precision:.3f} |"
        )
    return "\n".join(lines) + "\n"


def load_and_evaluate_external_segment_shadow(
    *,
    payload_path: Path,
    benchmark_path: Path,
    projected_top_k: int = 3,
    candidate_pool_size: int = 24,
    composer_mode: SegmentTextMode = SegmentTextMode.RICH,
) -> ExternalSegmentShadowSummary:
    """Convenience wrapper for CLI usage."""

    payload = load_external_segment_payload(payload_path)
    cases = load_shadow_benchmark_cases(benchmark_path)
    return evaluate_external_segment_shadow(
        payload=payload,
        cases=cases,
        projected_top_k=projected_top_k,
        candidate_pool_size=candidate_pool_size,
        composer_mode=composer_mode,
    )


def run_external_segment_shadow_ablation(
    *,
    payload_path: Path,
    benchmark_path: Path,
    projected_top_k: int = 3,
    candidate_pool_size: int = 24,
) -> ExternalSegmentAblationSummary:
    """Run plain-vs-rich text composition ablation on the same benchmark.

    Args:
        payload_path: External segment payload path.
        benchmark_path: Shadow benchmark path.
        projected_top_k: Max projected pages per case.
        candidate_pool_size: Max ranked segments per case.

    Returns:
        Ablation summary across plain and rich composers.
    """

    payload = load_external_segment_payload(payload_path)
    cases = load_shadow_benchmark_cases(benchmark_path)
    plain_summary = evaluate_external_segment_shadow(
        payload=payload,
        cases=cases,
        projected_top_k=projected_top_k,
        candidate_pool_size=candidate_pool_size,
        composer_mode=SegmentTextMode.PLAIN,
    )
    rich_summary = evaluate_external_segment_shadow(
        payload=payload,
        cases=cases,
        projected_top_k=projected_top_k,
        candidate_pool_size=candidate_pool_size,
        composer_mode=SegmentTextMode.RICH,
    )
    return ExternalSegmentAblationSummary(
        plain_summary=plain_summary,
        rich_summary=rich_summary,
        title_header_noise_rate={
            mode.value: _noise_rate(payload=payload, composer_mode=mode)
            for mode in (SegmentTextMode.PLAIN, SegmentTextMode.RICH)
        },
        avg_projected_page_count={
            SegmentTextMode.PLAIN.value: _avg_projected_page_count(plain_summary),
            SegmentTextMode.RICH.value: _avg_projected_page_count(rich_summary),
        },
    )


def _segment_rank_key(
    *,
    segment: ExternalSegmentRecord,
    family: ExternalSegmentFamily,
    query_tokens: Counter[str],
    ref_tokens: Counter[str],
    composer_mode: SegmentTextMode,
) -> tuple[int, int, int, int, int]:
    search_blob = normalize_shadow_text(compose_segment_text(segment, mode=composer_mode))
    token_hits = sum(count for token, count in query_tokens.items() if token in search_blob)
    ref_hits = sum(count for token, count in ref_tokens.items() if token in search_blob)
    family_score = _family_surface_score(segment=segment, family=family)
    evidence_score = len(segment.metadata.law_refs) + len(segment.metadata.case_refs)
    return family_score, ref_hits, token_hits, evidence_score, -segment.page_number


def _family_surface_score(*, segment: ExternalSegmentRecord, family: ExternalSegmentFamily) -> int:
    blob = normalize_shadow_text(segment.search_blob)
    if family is ExternalSegmentFamily.TITLE_CAPTION_CLAIMANT:
        score = 0
        if "claimant" in blob or "caption" in blob:
            score += 4
        if segment.page_number == 1:
            score += 2
        return score
    if family is ExternalSegmentFamily.EXACT_PROVISION:
        return int(bool(_ARTICLE_RE.search(segment.search_blob))) * 5 + int(segment.page_number <= 4)
    if family is ExternalSegmentFamily.AUTHORITY_DATE_LAW_NUMBER:
        score = 0
        if "legislative authority" in blob or "issued by" in blob:
            score += 3
        if "date of issue" in blob or "date of enactment" in blob or "commencement" in blob:
            score += 3
        return score
    return 0


def _filter_segments_for_case(
    segments: list[ExternalSegmentRecord],
    *,
    case: ExternalSegmentShadowCase,
) -> list[ExternalSegmentRecord]:
    if not case.doc_refs:
        return list(segments)
    normalized_refs = [normalize_shadow_text(ref) for ref in case.doc_refs if normalize_shadow_text(ref)]
    matched = [
        segment
        for segment in segments
        if any(ref in normalize_shadow_text(segment.search_blob) for ref in normalized_refs)
    ]
    return matched if matched else list(segments)


def _aggregate_family_metrics(hits: list[ExternalSegmentRetrievalHit]) -> list[ExternalSegmentFamilyMetrics]:
    grouped: dict[ExternalSegmentFamily, list[ExternalSegmentRetrievalHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.family, []).append(hit)
    metrics: list[ExternalSegmentFamilyMetrics] = []
    for family in ExternalSegmentFamily:
        rows = grouped.get(family, [])
        if not rows:
            continue
        baseline_hit_rate = sum(int(row.baseline_hit) for row in rows) / len(rows)
        projected_hit_rate = sum(int(row.projected_hit) for row in rows) / len(rows)
        projected_precision = sum(row.projected_precision for row in rows) / len(rows)
        metrics.append(
            ExternalSegmentFamilyMetrics(
                family=family,
                case_count=len(rows),
                baseline_hit_rate=baseline_hit_rate,
                projected_hit_rate=projected_hit_rate,
                projected_precision=projected_precision,
                hit_delta=projected_hit_rate - baseline_hit_rate,
            )
        )
    return metrics


def _noise_rate(*, payload: ExternalSegmentPayload, composer_mode: SegmentTextMode) -> float:
    analyses = [analyze_segment_noise(segment, mode=composer_mode) for segment in payload.segments]
    if not analyses:
        return 0.0
    noisy = sum(
        1
        for analysis in analyses
        if analysis.title_repeated or analysis.hierarchy_repeated or analysis.duplicate_line_count > 0
    )
    return noisy / len(analyses)


def _avg_projected_page_count(summary: ExternalSegmentShadowSummary) -> float:
    if not summary.cases:
        return 0.0
    return sum(len(case.projected_page_ids) for case in summary.cases) / len(summary.cases)


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    values = cast("list[object]", value)
    return [item.strip() for item in values if isinstance(item, str) and item.strip()]
