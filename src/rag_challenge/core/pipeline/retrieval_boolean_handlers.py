# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed retrieval helpers for the pipeline hot path."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models import RankedChunk, RetrievedChunk

    from .builder import RAGPipelineBuilder

from .query_rules import (
    _is_account_effective_dates_query,
)

logger = logging.getLogger(__name__)

SKIP_ADMIN_ARTICLE_RE = re.compile(
    r"(?:under|in|of|per|pursuant to)\s+article\s+\d+",
    re.IGNORECASE,
)


def ensure_boolean_year_compare_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not retrieved:
        return reranked[: max(0, int(top_n))]

    refs = pipeline.paired_support_question_refs(query)
    if len(refs) < 2:
        return reranked[: max(0, int(top_n))]

    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen_chunk_ids: set[str] = set()
    matched_doc_ids: set[str] = set()

    for ref in refs:
        best_raw: RetrievedChunk | None = None
        best_score = 0
        for raw in retrieved:
            if raw.doc_id in matched_doc_ids:
                continue
            score = pipeline.boolean_year_seed_chunk_score(ref=ref, chunk=raw)
            if score > best_score:
                best_raw = raw
                best_score = score
        if best_raw is None:
            continue
        matched_doc_ids.add(best_raw.doc_id)
        if best_raw.chunk_id in seen_chunk_ids:
            continue
        selected.append(reranked_by_id.get(best_raw.chunk_id) or pipeline.raw_to_ranked(best_raw))
        seen_chunk_ids.add(best_raw.chunk_id)
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
def ensure_boolean_admin_compare_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not retrieved:
        return reranked[: max(0, int(top_n))]

    refs = pipeline.paired_support_question_refs(query)
    if len(refs) < 2:
        return reranked[: max(0, int(top_n))]

    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen_chunk_ids: set[str] = set()
    matched_doc_ids: set[str] = set()

    for ref in refs:
        best_raw = pipeline.best_named_administration_chunk(
            ref=ref,
            chunks=retrieved,
            excluded_doc_ids=tuple(matched_doc_ids),
        )
        if best_raw is None:
            continue
        doc_id = str(best_raw.doc_id or "").strip()
        if doc_id:
            matched_doc_ids.add(doc_id)
        if best_raw.chunk_id in seen_chunk_ids:
            continue
        selected.append(reranked_by_id.get(best_raw.chunk_id) or pipeline.raw_to_ranked(best_raw))
        seen_chunk_ids.add(best_raw.chunk_id)
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
def ensure_boolean_judge_compare_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not retrieved:
        return reranked[: max(0, int(top_n))]

    refs = pipeline.paired_support_question_refs(query)
    if len(refs) < 2:
        return reranked[: max(0, int(top_n))]

    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen_chunk_ids: set[str] = set()
    matched_doc_ids: set[str] = set()

    for ref in refs:
        best_raw: RetrievedChunk | None = None
        best_score = 0
        for raw in retrieved:
            doc_id = str(raw.doc_id or "").strip()
            if doc_id in matched_doc_ids:
                continue
            identity_score = pipeline.case_ref_identity_score(ref=ref, chunk=raw)
            if identity_score <= 0:
                continue
            score = identity_score + pipeline.case_judge_seed_chunk_score(chunk=raw)
            if score > best_score:
                best_raw = raw
                best_score = score
        if best_raw is None:
            continue
        doc_id = str(best_raw.doc_id or "").strip()
        if doc_id:
            matched_doc_ids.add(doc_id)
        if best_raw.chunk_id in seen_chunk_ids:
            continue
        selected.append(reranked_by_id.get(best_raw.chunk_id) or pipeline.raw_to_ranked(best_raw))
        seen_chunk_ids.add(best_raw.chunk_id)
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
def ensure_notice_doc_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not retrieved or not pipeline.is_notice_focus_query(query):
        return reranked[: max(0, int(top_n))]

    desired_docs = 2 if "precise calendar date" in pipeline.normalize_support_text(query).casefold() else 1
    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen_chunk_ids: set[str] = set()
    seen_doc_ids: set[str] = set()

    scored: list[tuple[int, float, int, RetrievedChunk]] = []
    for raw in retrieved:
        score = pipeline.notice_doc_score(query=query, raw=raw)
        if score <= 0:
            continue
        page_num = pipeline.page_num(str(getattr(raw, "section_path", "") or ""))
        scored.append((score, float(raw.score), -page_num, raw))

    scored.sort(reverse=True)
    for _score, _retrieval_score, _page_rank, raw in scored:
        doc_id = str(raw.doc_id or "").strip()
        if doc_id and doc_id in seen_doc_ids:
            continue
        selected.append(reranked_by_id.get(raw.chunk_id) or pipeline.raw_to_ranked(raw))
        seen_chunk_ids.add(raw.chunk_id)
        if doc_id:
            seen_doc_ids.add(doc_id)
        if len(seen_doc_ids) >= desired_docs or len(selected) >= top_n:
            break

    for chunk in reranked:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        selected.append(chunk)
        seen_chunk_ids.add(chunk.chunk_id)
        if len(selected) >= top_n:
            break

    return selected[:top_n] if selected else reranked[: max(0, int(top_n))]
def ensure_account_effective_dates_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    doc_refs: list[str],
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not retrieved or not _is_account_effective_dates_query(query):
        return reranked[: max(0, int(top_n))]

    refs = [ref for ref in doc_refs if str(ref).strip()] or pipeline.support_question_refs(query)
    if not refs:
        return reranked[: max(0, int(top_n))]

    ref = refs[0]
    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen_chunk_ids: set[str] = set()

    best_anchor: RetrievedChunk | None = None
    best_anchor_score = 0
    for raw in retrieved:
        score = pipeline.named_commencement_title_match_score(ref, raw)
        if score > best_anchor_score:
            best_anchor = raw
            best_anchor_score = score

    if best_anchor is not None:
        best_effective: RetrievedChunk | None = None
        best_effective_score = 0
        for raw in retrieved:
            if raw.doc_id != best_anchor.doc_id:
                continue
            score = pipeline.account_effective_clause_score(text=str(getattr(raw, "text", "") or ""))
            if score > best_effective_score:
                best_effective = raw
                best_effective_score = score
        for raw in [best_anchor, best_effective] if best_effective is not None else [best_anchor]:
            if raw.chunk_id in seen_chunk_ids:
                continue
            selected.append(reranked_by_id.get(raw.chunk_id) or pipeline.raw_to_ranked(raw))
            seen_chunk_ids.add(raw.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]

    best_enactment: RetrievedChunk | None = None
    best_enactment_score = 0
    for raw in retrieved:
        score = pipeline.account_enactment_clause_score(ref=ref, raw=raw)
        if score > best_enactment_score:
            best_enactment = raw
            best_enactment_score = score
    if best_enactment is not None and best_enactment.chunk_id not in seen_chunk_ids:
        selected.append(reranked_by_id.get(best_enactment.chunk_id) or pipeline.raw_to_ranked(best_enactment))
        seen_chunk_ids.add(best_enactment.chunk_id)
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
def account_effective_support_family_seed_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    ref: str,
    retrieved: Sequence[RetrievedChunk],
) -> list[str]:
    best_effective: RetrievedChunk | None = None
    best_effective_score = 0
    best_enactment: RetrievedChunk | None = None
    best_enactment_score = 0

    for raw in retrieved:
        effective_score = pipeline.account_effective_clause_score(text=str(getattr(raw, "text", "") or ""))
        if effective_score > best_effective_score:
            best_effective = raw
            best_effective_score = effective_score

        enactment_score = pipeline.account_enactment_clause_score(ref=ref, raw=raw)
        if enactment_score > best_enactment_score:
            best_enactment = raw
            best_enactment_score = enactment_score

    seeds: list[str] = []
    if best_effective is not None and best_effective_score > 0:
        seeds.append(best_effective.chunk_id)
    if best_enactment is not None and best_enactment_score > 0:
        seeds.append(best_enactment.chunk_id)
    return pipeline.dedupe_chunk_ids(seeds)
def prune_boolean_context_for_single_doc_article(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    answer_type: str,
    doc_refs: list[str] | tuple[str, ...] | None,
    context_chunks: list[RankedChunk],
) -> list[RankedChunk]:
    """For boolean + single doc_ref + explicit article queries, restrict context
    to chunks from the referenced document to prevent cross-doc contamination."""
    if (answer_type or "").strip().lower() != "boolean":
        return list(context_chunks)
    refs = [str(r).strip() for r in (doc_refs or []) if str(r).strip()]
    if len(refs) != 1:
        return list(context_chunks)
    if not SKIP_ADMIN_ARTICLE_RE.search(query or ""):
        return list(context_chunks)

    ref_lower = refs[0].casefold()
    ref_tokens = set(ref_lower.split())
    matching_doc_ids: set[str] = set()
    for chunk in context_chunks:
        title_lower = (chunk.doc_title or "").casefold()
        if ref_lower in title_lower or title_lower in ref_lower:
            matching_doc_ids.add(chunk.doc_id)
            continue
        title_tokens = set(title_lower.split())
        if ref_tokens and len(ref_tokens & title_tokens) >= max(1, len(ref_tokens) - 1):
            matching_doc_ids.add(chunk.doc_id)

    if not matching_doc_ids:
        return list(context_chunks)

    pruned = [c for c in context_chunks if c.doc_id in matching_doc_ids]
    if len(pruned) >= 2:
        return pruned
    return list(context_chunks)
def administration_support_family_seed_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    ref: str,
    retrieved: Sequence[RetrievedChunk],
) -> list[str]:
    best_chunk = pipeline.best_named_administration_chunk(ref=ref, chunks=retrieved)
    chunk_id = str(getattr(best_chunk, "chunk_id", "") or "").strip() if best_chunk is not None else ""
    return [chunk_id] if chunk_id else []
def remuneration_recordkeeping_clause_score(pipeline: RAGPipelineBuilder, raw: RetrievedChunk) -> int:
    normalized = re.sub(r"\s+", " ", str(getattr(raw, "text", "") or "").strip()).casefold()
    if not normalized:
        return 0
    score = 0
    if "article 16" in normalized or "16. payroll records" in normalized or "payroll records" in normalized:
        score += 80
    if "remuneration" in normalized:
        score += 80
    if "pay period" in normalized:
        score += 80
    if "gross and net" in normalized:
        score += 40
    return score
