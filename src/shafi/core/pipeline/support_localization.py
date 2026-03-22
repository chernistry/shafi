# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed support helpers for the pipeline hot path."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from shafi.models import Citation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shafi.models import RankedChunk

    from .builder import RAGPipelineBuilder

from .constants import (
    _CITE_RE,
)
from .query_rules import (
    _is_broad_enumeration_query,
    _is_named_multi_title_lookup_query,
)


def citations_from_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    chunk_ids: Sequence[str],
    context_chunks: Sequence[RankedChunk],
) -> list[Citation]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
    citations: list[Citation] = []
    for chunk_id in chunk_ids:
        chunk = chunks_by_id.get(str(chunk_id).strip())
        if chunk is None:
            continue
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                doc_title=str(chunk.doc_title or ""),
                section_path=str(chunk.section_path or "") or None,
            )
        )
    return citations


def localize_strict_support_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    answer_type: str,
    answer: str,
    query: str,
    context_chunks: Sequence[RankedChunk],
) -> list[str]:
    if not context_chunks:
        return []

    kind = answer_type.strip().lower()
    if kind == "names":
        fragments = pipeline.split_names(_CITE_RE.sub("", answer))
    elif kind == "boolean":
        return pipeline.localize_boolean_support_chunk_ids(
            answer=answer,
            query=query,
            context_chunks=context_chunks,
        )
    else:
        fragments = [answer]

    localized: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        chunk_id = pipeline.best_support_chunk_id(
            answer_type=kind,
            query=query,
            fragment=fragment,
            context_chunks=context_chunks,
            allow_first_chunk_fallback=False,
        )
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            localized.append(chunk_id)

    return localized


def localize_boolean_support_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    answer: str,
    query: str,
    context_chunks: Sequence[RankedChunk],
) -> list[str]:
    if not context_chunks:
        return []

    compare_localized = pipeline.localize_boolean_compare_support_chunk_ids(
        query=query,
        context_chunks=context_chunks,
    )
    if compare_localized:
        return compare_localized

    polarity = pipeline.normalize_support_text(answer).casefold()
    query_terms = pipeline.support_terms(query)
    query_lower = pipeline.normalize_support_text(query).casefold()
    exception_query = any(
        marker in query_lower
        for marker in (
            " if ",
            " unless ",
            " except ",
            " provided ",
            " notwithstanding ",
            " bad faith",
            " good faith",
            " liable",
            " liability",
        )
    )

    ranked: list[tuple[int, int, str, set[str], bool]] = []
    for idx, chunk in enumerate(context_chunks):
        base_score = pipeline.chunk_support_score(
            answer_type="boolean",
            query=query,
            fragment=query or answer,
            chunk=chunk,
        )
        blob = pipeline.chunk_support_blob(chunk)
        blob_lower = blob.casefold()
        matched_terms = query_terms.intersection(pipeline.support_terms(blob))
        has_exception_clause = bool(
            re.search(
                r"\b(?:except|unless|provided\s+that|notwithstanding|bad\s+faith|good\s+faith|liable|liability|"
                r"does\s+not\s+apply|nothing\s+in)\b",
                blob_lower,
            )
        )
        if polarity.startswith("yes") and has_exception_clause:
            base_score += 18
        if polarity.startswith("no") and bool(
            re.search(
                r"\b(?:not\s+liable|no\s+liability|shall\s+not|may\s+not|is\s+not\s+liable|immune)\b",
                blob_lower,
            )
        ):
            base_score += 18
        if base_score <= 0:
            continue
        ranked.append((base_score, -idx, chunk.chunk_id, matched_terms, has_exception_clause))

    if not ranked:
        return []

    ranked.sort(reverse=True)
    primary_chunk_id = ranked[0][2]
    truncated_ranked = ranked[: min(len(ranked), 6)]
    max_term_overlap = max(
        len(matched_terms) for _score, _order, _chunk_id, matched_terms, _has_exception in truncated_ranked
    )
    exception_available = any(
        has_exception for _score, _order, _chunk_id, _matched_terms, has_exception in truncated_ranked
    )

    def _candidate_score(indices: tuple[int, ...]) -> tuple[int, int, int]:
        selected = [truncated_ranked[idx] for idx in indices]
        total_score = sum(score for score, _order, _chunk_id, _matched_terms, _has_exception in selected)
        covered: set[str] = set()
        for _score, _order, _chunk_id, matched_terms, _has_exception in selected:
            covered.update(matched_terms)
        exception_covered = any(has_exception for _score, _order, _chunk_id, _matched_terms, has_exception in selected)
        completeness_penalty = 0
        if exception_query and exception_available and not exception_covered:
            completeness_penalty -= 10_000
        if exception_query and len(covered) < max_term_overlap:
            completeness_penalty -= (max_term_overlap - len(covered)) * 40
        return (completeness_penalty + total_score + (len(covered) * 12), -len(indices), -indices[0])

    best_indices = (0,)
    best_tuple = _candidate_score(best_indices)

    for idx in range(len(truncated_ranked)):
        candidate = (idx,)
        score_tuple = _candidate_score(candidate)
        if score_tuple > best_tuple:
            best_tuple = score_tuple
            best_indices = candidate

    for left in range(len(truncated_ranked)):
        for right in range(left + 1, len(truncated_ranked)):
            candidate = (left, right)
            score_tuple = _candidate_score(candidate)
            if score_tuple > best_tuple:
                best_tuple = score_tuple
                best_indices = candidate

    localized = [truncated_ranked[idx][2] for idx in best_indices]
    if not localized:
        return [primary_chunk_id]
    return localized


def localize_boolean_compare_support_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    context_chunks: Sequence[RankedChunk],
) -> list[str]:
    refs = pipeline.paired_support_question_refs(query)
    if len(refs) < 2 or not context_chunks:
        return []

    query_lower = pipeline.normalize_support_text(query).casefold()
    if "same year" in query_lower:

        def scorer(ref: str, chunk: RankedChunk) -> int:
            return pipeline.boolean_year_seed_chunk_score(ref=ref, chunk=chunk)
    elif "administ" in query_lower:

        def scorer(ref: str, chunk: RankedChunk) -> int:
            clause_score = pipeline.named_administration_clause_score(
                ref=ref,
                text=str(getattr(chunk, "text", "") or ""),
            )
            if clause_score <= 0:
                return 0
            return pipeline.boolean_admin_seed_chunk_score(ref=ref, chunk=chunk) + clause_score
    else:
        return []

    localized: list[str] = []
    seen_chunk_ids: set[str] = set()
    seen_doc_ids: set[str] = set()
    for ref in refs:
        best_chunk_id = ""
        best_doc_id = ""
        best_score = 0
        for chunk in context_chunks:
            doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
            if doc_id in seen_doc_ids:
                continue
            score = scorer(ref, chunk)
            if score > best_score:
                best_score = score
                best_chunk_id = chunk.chunk_id
                best_doc_id = doc_id
        if not best_chunk_id:
            continue
        if best_chunk_id not in seen_chunk_ids:
            localized.append(best_chunk_id)
            seen_chunk_ids.add(best_chunk_id)
        if best_doc_id:
            seen_doc_ids.add(best_doc_id)

    return localized if len(localized) >= 2 else []


def localize_free_text_support_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    answer: str,
    query: str,
    context_chunks: Sequence[RankedChunk],
) -> list[str]:
    if not context_chunks:
        return []

    items = pipeline.split_free_text_items(answer)
    if not items:
        return []

    localized: list[str] = []
    seen: set[str] = set()
    bounded_items = items[:8]
    for item_index, item in enumerate(bounded_items):
        item_chunks = pipeline.free_text_item_candidate_chunks(
            query=query,
            item=item,
            item_index=item_index,
            item_count=len(bounded_items),
            context_chunks=context_chunks,
        )
        item_slots = pipeline.extract_free_text_item_slots(query=query, item=item)
        primary_slot_ids: list[str] = []
        title_slot = pipeline.free_text_item_title_slot(pipeline.normalize_support_text(_CITE_RE.sub("", item)))
        if title_slot:
            chunk_id = pipeline.best_title_support_chunk_id(
                title=title_slot,
                context_chunks=item_chunks,
            )
            if chunk_id:
                primary_slot_ids.append(chunk_id)
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                localized.append(chunk_id)
        for slot in item_slots:
            chunk_id = pipeline.best_support_chunk_id(
                answer_type="free_text",
                query=query,
                fragment=slot,
                context_chunks=item_chunks,
                allow_first_chunk_fallback=False,
            )
            if chunk_id:
                primary_slot_ids.append(chunk_id)
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                localized.append(chunk_id)

        if pipeline.free_text_slot_full_context_priority(
            query=query,
            item_slots=item_slots,
            primary_slot_ids=primary_slot_ids,
        ):
            if title_slot:
                expanded_title_chunk_id = pipeline.best_title_support_chunk_id(
                    title=title_slot,
                    context_chunks=context_chunks,
                )
                if expanded_title_chunk_id and expanded_title_chunk_id not in seen:
                    seen.add(expanded_title_chunk_id)
                    localized.append(expanded_title_chunk_id)
            for slot in item_slots:
                chunk_id = pipeline.best_support_chunk_id(
                    answer_type="free_text",
                    query=query,
                    fragment=slot,
                    context_chunks=context_chunks,
                    allow_first_chunk_fallback=False,
                )
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    localized.append(chunk_id)

    if _is_named_multi_title_lookup_query(query):
        localized_doc_ids = {str(chunk.doc_id or chunk.chunk_id) for chunk in context_chunks if chunk.chunk_id in seen}
        for ref in pipeline.support_question_refs(query):
            ref_chunk_id = pipeline.best_title_support_chunk_id(
                title=ref,
                context_chunks=context_chunks,
            )
            if not ref_chunk_id:
                continue
            ref_chunk = next((chunk for chunk in context_chunks if chunk.chunk_id == ref_chunk_id), None)
            ref_doc_id = str(ref_chunk.doc_id or ref_chunk.chunk_id) if ref_chunk is not None else ""
            if ref_doc_id and ref_doc_id in localized_doc_ids:
                continue
            if ref_chunk_id not in seen:
                seen.add(ref_chunk_id)
                localized.append(ref_chunk_id)
            if ref_doc_id:
                localized_doc_ids.add(ref_doc_id)

    return localized


def suppress_named_administration_family_orphan_support_ids(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    cited_ids: Sequence[str],
    support_ids: Sequence[str],
    context_chunks: Sequence[RankedChunk],
) -> list[str]:
    normalized_query = re.sub(r"\s+", " ", (query or "").strip()).casefold()
    if not support_ids or not cited_ids or "administ" not in normalized_query or _is_broad_enumeration_query(query):
        return list(dict.fromkeys(str(chunk_id).strip() for chunk_id in support_ids if str(chunk_id).strip()))

    refs = pipeline.support_question_refs(query)
    if len(refs) < 2:
        return list(dict.fromkeys(str(chunk_id).strip() for chunk_id in support_ids if str(chunk_id).strip()))

    context_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}

    def _canonical_cited_for_ref(ref: str) -> bool:
        for raw_chunk_id in cited_ids:
            chunk = context_by_id.get(str(raw_chunk_id).strip())
            if chunk is None:
                continue
            if pipeline.boolean_admin_seed_chunk_score(ref=ref, chunk=chunk) <= 0:
                continue
            if not pipeline.is_consolidated_or_amended_family_chunk(chunk=chunk):
                return True
        return False

    canonical_refs = {ref for ref in refs if _canonical_cited_for_ref(ref)}
    if not canonical_refs:
        return list(dict.fromkeys(str(chunk_id).strip() for chunk_id in support_ids if str(chunk_id).strip()))

    filtered: list[str] = []
    seen: set[str] = set()
    for raw_chunk_id in support_ids:
        chunk_id = str(raw_chunk_id).strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        if chunk_id in cited_ids:
            filtered.append(chunk_id)
            continue

        chunk = context_by_id.get(chunk_id)
        if chunk is None or not pipeline.is_consolidated_or_amended_family_chunk(chunk=chunk):
            filtered.append(chunk_id)
            continue

        drop_surrogate = False
        for ref in canonical_refs:
            if pipeline.boolean_admin_seed_chunk_score(ref=ref, chunk=chunk) <= 0:
                continue
            drop_surrogate = True
            break
        if not drop_surrogate:
            filtered.append(chunk_id)

    return filtered
