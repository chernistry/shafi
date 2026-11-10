# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed retrieval helpers for the pipeline hot path."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shafi.models import RankedChunk, RetrievedChunk

    from .builder import RAGPipelineBuilder

from .constants import (
    _COMMON_ELEMENTS_TITLE_STOPWORDS,
    _COMMON_ELEMENTS_TOKEN_RE,
    _TITLE_LAW_NO_SUFFIX_RE,
)
from .query_rules import (
    _is_common_elements_query,
    _is_interpretation_sections_common_elements_query,
)

logger = logging.getLogger(__name__)


def common_elements_ref_tokens(pipeline: RAGPipelineBuilder, text: str) -> tuple[str, ...]:
    normalized = _TITLE_LAW_NO_SUFFIX_RE.sub("", text or "")
    normalized = re.sub(r"\b(19|20)\d{2}\b", " ", normalized)
    tokens = [
        token
        for token in _COMMON_ELEMENTS_TOKEN_RE.findall(normalized.lower())
        if token and token not in _COMMON_ELEMENTS_TITLE_STOPWORDS and len(token) > 2
    ]
    return tuple(dict.fromkeys(tokens))


def common_elements_title_match_score(
    pipeline: RAGPipelineBuilder, ref: str, chunk: RetrievedChunk | RankedChunk
) -> int:
    ref_tokens = pipeline.common_elements_ref_tokens(ref)
    if not ref_tokens:
        return 0

    haystack = " ".join(
        part
        for part in (
            str(getattr(chunk, "doc_title", "") or ""),
            str(getattr(chunk, "text", "") or "")[:1200],
        )
        if part
    )
    haystack_tokens = set(_COMMON_ELEMENTS_TOKEN_RE.findall(haystack.lower()))
    overlap = sum(1 for token in ref_tokens if token in haystack_tokens)
    if overlap <= 0:
        return 0
    if overlap == len(ref_tokens):
        return 100 + overlap
    if overlap >= max(1, len(ref_tokens) - 1):
        return 60 + overlap
    if overlap >= max(1, (len(ref_tokens) + 1) // 2):
        return 20 + overlap
    return 0


def common_elements_evidence_score(text: str, *, interpretation_sections: bool = False) -> int:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not normalized:
        return 0

    score = 0
    if interpretation_sections:
        if "rules of interpretation" in normalized:
            score += 18
        if "a statutory provision includes a reference" in normalized:
            score += 22
        if "reference to a person includes" in normalized:
            score += 20
        if "interpretation" in normalized:
            score += 6
        if "schedule 1" in normalized:
            score += 2
        if "interpretative provisions" in normalized:
            score += 1
        if (
            "defined terms" in normalized
            and "a statutory provision includes a reference" not in normalized
            and "reference to a person includes" not in normalized
        ):
            score -= 8
        return score

    if "schedule 1" in normalized:
        score += 5
    if "interpretation" in normalized:
        score += 4
    if "rules of interpretation" in normalized:
        score += 7
    if "interpretative provisions" in normalized:
        score += 4
    if "defined terms" in normalized:
        score += 2
    if "a statutory provision includes a reference" in normalized:
        score += 2
    return score


def ensure_common_elements_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    doc_refs: list[str],
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not reranked or not retrieved or not _is_common_elements_query(query):
        return reranked[: max(0, int(top_n))]

    refs = [ref for ref in doc_refs if str(ref).strip()] or pipeline.extract_title_refs_from_query(query)
    if len(refs) < 2:
        return reranked[: max(0, int(top_n))]

    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen_chunk_ids: set[str] = set()
    interpretation_sections_query = _is_interpretation_sections_common_elements_query(query)

    for ref in refs:
        best_anchor: RetrievedChunk | None = None
        best_anchor_key: tuple[int, int, float, int] | None = None
        for raw in retrieved:
            title_match = pipeline.common_elements_title_match_score(ref, raw)
            if title_match <= 0:
                continue
            evidence_score = pipeline.common_elements_evidence_score(
                str(getattr(raw, "text", "") or ""),
                interpretation_sections=interpretation_sections_query,
            )
            page_num = pipeline.section_page_num(str(getattr(raw, "section_path", "") or ""))
            candidate = (evidence_score, title_match, float(raw.score), page_num)
            if best_anchor_key is None or candidate > best_anchor_key:
                best_anchor_key = candidate
                best_anchor = raw

        if best_anchor is None:
            continue

        preferred_raw: list[RetrievedChunk] = [best_anchor]
        if interpretation_sections_query:
            best_clause: RetrievedChunk | None = None
            best_clause_key: tuple[int, float, int] | None = None
            for raw in retrieved:
                if raw.doc_id != best_anchor.doc_id:
                    continue
                evidence_score = pipeline.common_elements_evidence_score(
                    str(getattr(raw, "text", "") or ""),
                    interpretation_sections=True,
                )
                if evidence_score <= 0:
                    continue
                page_num = pipeline.section_page_num(str(getattr(raw, "section_path", "") or ""))
                candidate = (evidence_score, float(raw.score), page_num)
                if best_clause_key is None or candidate > best_clause_key:
                    best_clause_key = candidate
                    best_clause = raw
            if best_clause is not None:
                preferred_raw = [best_clause]

        for raw in preferred_raw:
            if raw.chunk_id in seen_chunk_ids:
                continue
            selected.append(reranked_by_id.get(raw.chunk_id) or pipeline.raw_to_ranked(raw))
            seen_chunk_ids.add(raw.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]

    for chunk in reranked:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        selected.append(chunk)
        seen_chunk_ids.add(chunk.chunk_id)
        if len(selected) >= top_n:
            break

    return selected[:top_n] if selected else reranked[: max(0, int(top_n))]
