# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed retrieval helpers for the pipeline hot path."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rag_challenge.core.classifier import QueryClassifier
from rag_challenge.models import RankedChunk

from .constants import _DIFC_CASE_ID_RE
from .query_rules import (
    _is_broad_enumeration_query,
    _is_case_issue_date_name_compare_query,
    _is_common_judge_compare_query,
    _is_named_commencement_query,
    _is_named_multi_title_lookup_query,
)
from .retrieval_seed_selection import extract_title_refs_from_query
from .support_query_primitives import page_num

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models import RetrievedChunk

    from .builder import RAGPipelineBuilder

logger = logging.getLogger(__name__)


def detect_coverage_gaps(
    query: str,
    context_chunks: list[object],
    doc_refs: list[str] | None = None,
) -> str:
    """Detect entities mentioned in query but absent from context chunks.

    Returns a prompt hint string warning the LLM about missing entities,
    or empty string if all entities are covered.
    """
    # Gather all entity references from the query.
    title_refs = extract_title_refs_from_query(query)
    all_refs = list(doc_refs or []) + title_refs
    if len(all_refs) < 2:
        # Only flag gaps for multi-entity queries.
        return ""

    # Build a searchable text from all context chunk titles and texts.
    chunks_text_parts: list[str] = []
    for chunk in context_chunks:
        doc_title = getattr(chunk, "doc_title", "") or ""
        text = getattr(chunk, "text", "") or ""
        chunks_text_parts.append(f"{doc_title} {text}".lower())
    context_blob = " ".join(chunks_text_parts)

    # Check each entity for presence in context.
    missing: list[str] = []
    seen: set[str] = set()
    for ref in all_refs:
        ref_clean = ref.strip()
        if not ref_clean:
            continue
        key = ref_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        # Check if ANY significant words from the ref appear together in context.
        ref_words = [w for w in key.split() if len(w) > 2 and w not in {"the", "of", "and", "in", "for", "law", "no."}]
        if not ref_words:
            continue
        # Require at least 60% of distinctive words to appear.
        found_count = sum(1 for w in ref_words if w in context_blob)
        if len(ref_words) > 0 and found_count / len(ref_words) < 0.6:
            missing.append(ref_clean)

    if not missing:
        return ""

    missing_list = ", ".join(missing)
    return (
        f"IMPORTANT: The retrieved sources do NOT contain information about: {missing_list}. "
        f"Do NOT guess or fabricate information about these items. "
        f'For any part of the question about these items, state that information is not available for [item name].'
    )


def build_entity_scope(context_chunks: Sequence[object]) -> str:
    """Build an entity scope constraint from context chunk doc_titles.

    Returns a prompt hint listing the exact documents available in context,
    preventing the LLM from referencing laws/entities from parametric memory.
    """
    doc_titles: set[str] = set()
    for chunk in context_chunks:
        title = (getattr(chunk, "doc_title", "") or "").strip()
        if title:
            doc_titles.add(title)

    if len(doc_titles) < 2:
        return ""  # Not useful for single-doc queries.

    titles_str = "; ".join(sorted(doc_titles))
    return (
        f"ENTITY SCOPE: Your retrieved sources cover ONLY these documents: [{titles_str}]. "
        f"When listing specific laws or documents in your answer, reference ONLY those named above "
        f"or entities EXPLICITLY mentioned by exact name within the source text you were given. "
        f"Do NOT add any laws, regulations, or documents from your own knowledge."
    )


def ensure_must_include_context(
    *,
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    must_include_chunk_ids: list[str],
    top_n: int,
) -> list[RankedChunk]:
    if not must_include_chunk_ids or top_n <= 0:
        return reranked[: max(0, int(top_n))]

    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    retrieved_by_id = {chunk.chunk_id: chunk for chunk in retrieved}

    selected: list[RankedChunk] = []
    seen: set[str] = set()

    for chunk_id in must_include_chunk_ids:
        if chunk_id in seen:
            continue
        chunk = reranked_by_id.get(chunk_id)
        if chunk is None:
            raw = retrieved_by_id.get(chunk_id)
            if raw is None:
                continue
            chunk = RankedChunk(
                chunk_id=raw.chunk_id,
                doc_id=raw.doc_id,
                doc_title=raw.doc_title,
                doc_type=raw.doc_type,
                section_path=raw.section_path,
                text=raw.text,
                retrieval_score=float(raw.score),
                # These injected chunks didn't go through the reranker; use retrieval score as a stable proxy.
                rerank_score=float(raw.score),
                doc_summary=raw.doc_summary,
                page_family=getattr(raw, "page_family", ""),
                doc_family=getattr(raw, "doc_family", ""),
                chunk_type=getattr(raw, "chunk_type", ""),
                amount_roles=list(getattr(raw, "amount_roles", []) or []),
            )
        selected.append(chunk)
        seen.add(chunk.chunk_id)
        if len(selected) >= top_n:
            return selected[:top_n]

    for chunk in reranked:
        if chunk.chunk_id in seen:
            continue
        selected.append(chunk)
        seen.add(chunk.chunk_id)
        if len(selected) >= top_n:
            break

    return selected[:top_n]
def ensure_page_one_context(
    *,
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not reranked or not retrieved:
        return reranked[: max(0, int(top_n))]

    page_one_by_doc: dict[str, RetrievedChunk] = {}
    for chunk in retrieved:
        doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
        section_path = str(getattr(chunk, "section_path", "") or "")
        if not doc_id or page_num(section_path) != 1:
            continue
        current = page_one_by_doc.get(doc_id)
        if current is None or float(chunk.score) > float(current.score):
            page_one_by_doc[doc_id] = chunk

    selected: list[RankedChunk] = []
    seen: set[str] = set()

    for chunk in reranked:
        page_one = page_one_by_doc.get(chunk.doc_id)
        if page_one is not None and page_one.chunk_id not in seen:
            selected.append(
                RankedChunk(
                    chunk_id=page_one.chunk_id,
                    doc_id=page_one.doc_id,
                    doc_title=page_one.doc_title,
                    doc_type=page_one.doc_type,
                    section_path=page_one.section_path,
                    text=page_one.text,
                    retrieval_score=float(page_one.score),
                    rerank_score=float(page_one.score),
                    doc_summary=page_one.doc_summary,
                    page_family=getattr(page_one, "page_family", ""),
                    doc_family=getattr(page_one, "doc_family", ""),
                    chunk_type=getattr(page_one, "chunk_type", ""),
                    amount_roles=list(getattr(page_one, "amount_roles", []) or []),
                )
            )
            seen.add(page_one.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]
        if chunk.chunk_id in seen:
            continue
        selected.append(chunk)
        seen.add(chunk.chunk_id)
        if len(selected) >= top_n:
            break

    return selected[:top_n]
def doc_family_collapse_candidate_score(pipeline: RAGPipelineBuilder, *, query: str, chunk: RetrievedChunk | RankedChunk) -> tuple[int, int, float]:
    normalized_query_refs = [ref.casefold() for ref in pipeline.support_question_refs(query)[:4]]
    haystack = re.sub(
        r"\s+",
        " ",
        " ".join(
            part
            for part in (
                str(getattr(chunk, "doc_title", "") or ""),
                str(getattr(chunk, "text", "") or "")[:500],
            )
            if part
        ),
    ).strip().casefold()
    ref_bonus = 0
    if haystack and any(ref in haystack for ref in normalized_query_refs):
        ref_bonus = 2
    page_num = pipeline.page_num(str(getattr(chunk, "section_path", "") or ""))
    if page_num <= 2:
        page_bonus = 2
    elif page_num <= 4:
        page_bonus = 1
    else:
        page_bonus = 0
    retrieval_score = float(
        getattr(
            chunk,
            "score",
            getattr(chunk, "retrieval_score", 0.0),
        )
    )
    return ref_bonus, page_bonus, retrieval_score
def collapse_doc_family_crowding_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    answer_type: str,
    doc_ref_count: int,
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    must_include_chunk_ids: Sequence[str],
    top_n: int,
) -> list[RankedChunk]:
    bounded = reranked[: max(0, int(top_n))]
    if top_n <= 1 or len(bounded) <= 1 or doc_ref_count < 2 or not retrieved:
        return bounded
    if QueryClassifier.extract_explicit_page_reference(query) is not None:
        return bounded
    if _is_broad_enumeration_query(query):
        return bounded

    q_lower = re.sub(r"\s+", " ", query).strip().lower()
    normalized_answer_type = answer_type.strip().lower()
    compare_like = normalized_answer_type in {"boolean", "name", "names", "date", "number"} and (
        len(_DIFC_CASE_ID_RE.findall(query or "")) >= 2
        or len(pipeline.support_question_refs(query)) >= 2
        or _is_case_issue_date_name_compare_query(query, answer_type=answer_type)
        or _is_common_judge_compare_query(query)
        or "same year" in q_lower
        or "same party" in q_lower
        or "appeared in both" in q_lower
        or "administ" in q_lower
    )
    metadata_like = (
        pipeline.is_named_metadata_support_query(query)
        or _is_named_multi_title_lookup_query(query)
        or _is_named_commencement_query(query)
    )
    if not (compare_like or metadata_like):
        return bounded

    reranked_doc_ids = [
        str(getattr(chunk, "doc_id", "") or "").strip()
        for chunk in bounded
        if str(getattr(chunk, "doc_id", "") or "").strip()
    ]
    if not reranked_doc_ids:
        return bounded
    distinct_reranked_doc_ids = list(dict.fromkeys(reranked_doc_ids))
    target_doc_count = min(2, int(top_n))
    if len(distinct_reranked_doc_ids) >= target_doc_count:
        return bounded

    dominant_doc_id = distinct_reranked_doc_ids[0]
    alternative_by_doc: dict[str, RetrievedChunk] = {}
    for raw in retrieved:
        doc_id = str(getattr(raw, "doc_id", "") or "").strip()
        if not doc_id or doc_id == dominant_doc_id:
            continue
        current = alternative_by_doc.get(doc_id)
        if current is None or pipeline.doc_family_collapse_candidate_score(query=query, chunk=raw) > pipeline.doc_family_collapse_candidate_score(
            query=query,
            chunk=current,
        ):
            alternative_by_doc[doc_id] = raw
    if not alternative_by_doc:
        return bounded

    replacement_index: int | None = None
    must_include_set = {str(chunk_id).strip() for chunk_id in must_include_chunk_ids if str(chunk_id).strip()}
    for idx in range(len(bounded) - 1, 0, -1):
        chunk = bounded[idx]
        if chunk.chunk_id in must_include_set:
            continue
        if str(getattr(chunk, "doc_id", "") or "").strip() == dominant_doc_id:
            replacement_index = idx
            break
    if replacement_index is None:
        return bounded

    alternative = max(
        alternative_by_doc.values(),
        key=lambda raw: pipeline.doc_family_collapse_candidate_score(query=query, chunk=raw),
    )
    alternative_ranked = pipeline.raw_to_ranked(alternative)
    if any(chunk.chunk_id == alternative_ranked.chunk_id for chunk in bounded):
        return bounded

    collapsed = list(bounded)
    collapsed[replacement_index] = alternative_ranked
    return collapsed[: max(0, int(top_n))]
def raw_to_ranked(chunk: RetrievedChunk) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        doc_type=chunk.doc_type,
        section_path=chunk.section_path,
        text=chunk.text,
        retrieval_score=float(chunk.score),
        rerank_score=float(chunk.score),
        doc_summary=chunk.doc_summary,
        page_family=getattr(chunk, "page_family", ""),
        doc_family=getattr(chunk, "doc_family", ""),
        chunk_type=getattr(chunk, "chunk_type", ""),
        amount_roles=list(getattr(chunk, "amount_roles", []) or []),
    )
def augment_strict_context_chunks(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    answer_type: str,
    context_chunks: Sequence[RankedChunk],
    retrieved: Sequence[RetrievedChunk],
) -> tuple[list[RankedChunk], bool]:
    if answer_type.strip().lower() != "name":
        return list(context_chunks), False

    explicit_ref = QueryClassifier.extract_explicit_page_reference(query)
    if explicit_ref is None or explicit_ref.requested_page is None or explicit_ref.requested_page <= 0:
        return list(context_chunks), False

    query_lower = re.sub(r"\s+", " ", (query or "").strip()).casefold()
    if "claim number" not in query_lower and "claim no" not in query_lower:
        return list(context_chunks), False
    if not any(token in query_lower for token in ("originate", "originated", "arose", "arisen")):
        return list(context_chunks), False

    requested_page = explicit_ref.requested_page

    def _rescue_score(raw: RetrievedChunk) -> tuple[int, float]:
        text = re.sub(r"\s+", " ", str(raw.text or "")).strip().casefold()
        score = 0
        if pipeline.page_num(str(raw.section_path or "")) == requested_page:
            score += 500
        if "claim no." in text or "claim no " in text:
            score += 160
        if "appeal against" in text or "urgent application" in text:
            score += 140
        if "origin" in text:
            score += 40
        if "/2" in text:
            score += 80
        return score, float(raw.score)

    rescue_candidates = [
        raw for raw in retrieved if pipeline.page_num(str(raw.section_path or "")) == requested_page
    ]
    if not rescue_candidates and requested_page == 2:
        rescue_candidates = [
            raw for raw in retrieved if pipeline.page_num(str(raw.section_path or "")) in {1, 2}
        ]
    if not rescue_candidates:
        return list(context_chunks), False

    augmented = list(context_chunks)
    seen_chunk_ids = {chunk.chunk_id for chunk in augmented}
    added = False
    for raw in sorted(rescue_candidates, key=_rescue_score, reverse=True)[:2]:
        if raw.chunk_id in seen_chunk_ids:
            continue
        augmented.append(pipeline.raw_to_ranked(raw))
        seen_chunk_ids.add(raw.chunk_id)
        added = True
    return augmented, added
