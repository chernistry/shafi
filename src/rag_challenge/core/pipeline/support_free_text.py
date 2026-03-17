# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed support helpers for the pipeline hot path."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models import RankedChunk

    from .builder import RAGPipelineBuilder

from .constants import (
    _BULLET_ITEM_RE,
    _CITE_RE,
    _COMMENCEMENT_FIELD_RE,
    _ENACTED_ON_FIELD_RE,
    _LAST_UPDATED_FIELD_RE,
    _LAW_NO_REF_RE,
    _NUMBERED_ITEM_RE,
    _SUPPORT_STOPWORDS,
    _SUPPORT_TOKEN_RE,
    _TITLE_FIELD_RE,
    _TITLE_REF_RE,
)
from .query_rules import (
    _is_account_effective_dates_query,
    _is_named_amendment_query,
)


def split_names(answer: str) -> list[str]:
    raw_parts = [part.strip() for part in re.split(r"[,\n;]+", answer) if part.strip()]
    parts: list[str] = []
    for part in raw_parts:
        split_once = re.split(r"\s+\band\b\s+", part, maxsplit=1, flags=re.IGNORECASE)
        if len(split_once) == 2:
            parts.extend(item.strip() for item in split_once if item.strip())
        else:
            parts.append(part)
    return parts
def split_free_text_support_fragments(pipeline: RAGPipelineBuilder, answer: str) -> list[str]:
    cleaned = pipeline.normalize_support_text(_CITE_RE.sub("", answer))
    if not cleaned:
        return []

    numbered = [
        pipeline.normalize_support_text(match.group(1))
        for match in _NUMBERED_ITEM_RE.finditer(answer or "")
        if pipeline.normalize_support_text(match.group(1))
    ]
    if numbered:
        return numbered

    sentences = [
        pipeline.normalize_support_text(part)
        for part in re.split(r"(?<=[.!?])\s+", cleaned)
        if pipeline.normalize_support_text(part)
    ]
    if sentences:
        return sentences
    return [cleaned]
def split_free_text_items(pipeline: RAGPipelineBuilder, answer: str) -> list[str]:
    numbered = [
        pipeline.normalize_support_text(match.group(1))
        for match in _NUMBERED_ITEM_RE.finditer(answer or "")
        if pipeline.normalize_support_text(match.group(1))
    ]
    if numbered:
        return numbered

    bullet_items = [
        pipeline.normalize_support_text(match.group(1))
        for match in _BULLET_ITEM_RE.finditer(answer or "")
        if pipeline.normalize_support_text(match.group(1))
    ]
    if bullet_items:
        return bullet_items

    return pipeline.split_free_text_support_fragments(answer)
def free_text_item_title_slot(pipeline: RAGPipelineBuilder, item: str) -> str:
    title_field_match = _TITLE_FIELD_RE.search(item)
    if title_field_match is not None:
        return pipeline.normalize_support_text(title_field_match.group(1))

    prefix = re.split(r"\s+-\s+|:\s+", item, maxsplit=1)[0].strip(" ,.;:")
    if prefix and ("law" in prefix.casefold() or "regulation" in prefix.casefold()):
        return pipeline.normalize_support_text(prefix)
    return ""
def group_context_chunks_by_doc(
    pipeline: RAGPipelineBuilder,
    context_chunks: Sequence[RankedChunk],
) -> tuple[list[str], dict[str, list[RankedChunk]]]:
    chunks_by_doc: dict[str, list[RankedChunk]] = {}
    doc_order: list[str] = []
    for chunk in context_chunks:
        doc_key = str(chunk.doc_id or chunk.chunk_id)
        if doc_key not in chunks_by_doc:
            doc_order.append(doc_key)
        chunks_by_doc.setdefault(doc_key, []).append(chunk)
    return doc_order, chunks_by_doc
def free_text_doc_group_match_score(
    pipeline: RAGPipelineBuilder,
    *,
    ref: str,
    doc_chunks: Sequence[RankedChunk],
) -> int:
    normalized_ref = pipeline.normalize_support_text(ref).casefold()
    if not normalized_ref or not doc_chunks:
        return 0

    haystack = " ".join(
        part
        for part in (
            *(str(chunk.doc_title or "") for chunk in doc_chunks[:2]),
            *(str(chunk.doc_summary or "") for chunk in doc_chunks[:2]),
            *(str(chunk.text or "")[:1200] for chunk in doc_chunks[:2]),
        )
        if part
    )
    normalized_haystack = pipeline.normalize_support_text(haystack).casefold()
    if not normalized_haystack:
        return 0

    if normalized_ref in normalized_haystack:
        return 900 - min(normalized_haystack.find(normalized_ref), 600)

    ref_match = _LAW_NO_REF_RE.search(ref)
    if ref_match is not None:
        law_no_key = f"law no. {int(ref_match.group(1))} of {ref_match.group(2)}"
        if law_no_key in normalized_haystack:
            return 720

    ordered_ref_tokens = [
        token.casefold()
        for token in _SUPPORT_TOKEN_RE.findall(normalized_ref)
        if token.casefold() not in _SUPPORT_STOPWORDS and len(token) > 2
    ]
    if not ordered_ref_tokens:
        return 0

    haystack_tokens = {token.casefold() for token in _SUPPORT_TOKEN_RE.findall(normalized_haystack)}
    overlap = len(set(ordered_ref_tokens).intersection(haystack_tokens))
    if len(ordered_ref_tokens) >= 3 and overlap < len(set(ordered_ref_tokens)):
        ref_bigrams = [
            f"{ordered_ref_tokens[idx]} {ordered_ref_tokens[idx + 1]}"
            for idx in range(len(ordered_ref_tokens) - 1)
        ]
        bigram_overlap = sum(1 for bigram in ref_bigrams if bigram in normalized_haystack)
        if overlap >= max(1, len(set(ordered_ref_tokens)) - 1) and bigram_overlap < max(1, len(ref_bigrams) - 1):
            return 0

    if overlap == len(set(ordered_ref_tokens)):
        return 260 + overlap
    if overlap >= max(1, len(set(ordered_ref_tokens)) - 1):
        return 120 + overlap
    if overlap >= max(1, (len(set(ordered_ref_tokens)) + 1) // 2):
        return 50 + overlap
    return 0
def free_text_item_candidate_chunks(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    item: str,
    item_index: int,
    item_count: int,
    context_chunks: Sequence[RankedChunk],
) -> Sequence[RankedChunk]:
    if not context_chunks:
        return context_chunks

    refs: list[str] = []
    seen: set[str] = set()

    def _push(ref: str) -> None:
        normalized = pipeline.normalize_support_text(ref)
        if not normalized:
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        refs.append(normalized)

    query_refs = pipeline.support_question_refs(query)
    if item_count == len(query_refs) and item_index < len(query_refs):
        _push(query_refs[item_index])

    item_without_cites = pipeline.normalize_support_text(_CITE_RE.sub("", item))
    title_slot = pipeline.free_text_item_title_slot(item_without_cites)
    if title_slot:
        _push(title_slot)

    for title, year in _TITLE_REF_RE.findall(item_without_cites):
        _push(" ".join(part for part in (title.strip(), year.strip()) if part).strip(" ,.;:"))

    for match in _LAW_NO_REF_RE.finditer(item_without_cites):
        _push(f"Law No. {int(match.group(1))} of {match.group(2)}")

    if not refs:
        return context_chunks

    doc_order, chunks_by_doc = pipeline.group_context_chunks_by_doc(context_chunks)
    best_doc_id = ""
    best_score = 0
    for doc_id in doc_order:
        doc_chunks = chunks_by_doc.get(doc_id, [])
        score = max(
            (pipeline.free_text_doc_group_match_score(ref=ref, doc_chunks=doc_chunks) for ref in refs),
            default=0,
        )
        if score > best_score:
            best_score = score
            best_doc_id = doc_id

    if not best_doc_id or best_score <= 0:
        return context_chunks
    return chunks_by_doc.get(best_doc_id, context_chunks)
def free_text_slot_full_context_priority(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    item_slots: Sequence[str],
    primary_slot_ids: Sequence[str],
) -> bool:
    if _is_account_effective_dates_query(query) or _is_named_amendment_query(query):
        return True

    non_empty_slots = [slot for slot in item_slots if str(slot).strip()]
    if len(non_empty_slots) < 2:
        return False

    unique_primary_ids = {chunk_id for chunk_id in primary_slot_ids if str(chunk_id).strip()}
    return len(unique_primary_ids) < min(2, len(non_empty_slots))
def extract_free_text_item_slots(pipeline: RAGPipelineBuilder, *, query: str, item: str) -> list[str]:
    normalized_item = pipeline.normalize_support_text(_CITE_RE.sub("", item))
    if not normalized_item:
        return []

    query_lower = pipeline.normalize_support_text(query).casefold()
    slots: list[str] = []

    title_slot = pipeline.free_text_item_title_slot(normalized_item)
    if title_slot:
        slots.append(title_slot)

    if "updated" in query_lower:
        updated_match = _LAST_UPDATED_FIELD_RE.search(normalized_item)
        if updated_match is not None:
            slots.append(pipeline.normalize_support_text(updated_match.group(1)))

    if "enact" in query_lower:
        enacted_match = _ENACTED_ON_FIELD_RE.search(normalized_item)
        if enacted_match is not None:
            slots.append(pipeline.normalize_support_text(enacted_match.group(1)))

    if any(term in query_lower for term in ("commencement", "come into force", "effective date")):
        commencement_match = _COMMENCEMENT_FIELD_RE.search(normalized_item)
        if commencement_match is not None:
            slots.append(pipeline.normalize_support_text(commencement_match.group(1)))

    if "administ" in query_lower and ":" in normalized_item:
        remainder = re.split(r":\s+", normalized_item, maxsplit=1)[1].strip()
        if remainder:
            slots.append(remainder)

    bullet_lines = [
        pipeline.normalize_support_text(match.group(1))
        for match in _BULLET_ITEM_RE.finditer(item or "")
        if pipeline.normalize_support_text(match.group(1))
    ]
    for bullet in bullet_lines:
        bullet_title = pipeline.free_text_item_title_slot(bullet)
        slots.append(bullet_title or bullet)

    for title, year in _TITLE_REF_RE.findall(normalized_item):
        ref = " ".join(part for part in (title.strip(), year.strip()) if part).strip(" ,.;:")
        normalized_ref = pipeline.normalize_support_text(ref)
        if normalized_ref:
            slots.append(normalized_ref)

    if not slots:
        slots.append(normalized_item)

    deduped: list[str] = []
    seen: set[str] = set()
    for slot in slots:
        normalized_slot = pipeline.normalize_support_text(slot)
        if not normalized_slot:
            continue
        key = normalized_slot.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized_slot)
    return deduped
def best_title_support_chunk_id(
    pipeline: RAGPipelineBuilder,
    *,
    title: str,
    context_chunks: Sequence[RankedChunk],
) -> str:
    normalized_title = pipeline.normalize_support_text(title).casefold()
    if not normalized_title:
        return ""

    best_chunk_id = ""
    best_score = -1
    for idx, chunk in enumerate(context_chunks):
        doc_title = pipeline.normalize_support_text(str(getattr(chunk, "doc_title", "") or "")).casefold()
        doc_summary = pipeline.normalize_support_text(str(getattr(chunk, "doc_summary", "") or "")).casefold()
        text = pipeline.normalize_support_text(str(getattr(chunk, "text", "") or "")).casefold()
        score = 0
        if normalized_title and normalized_title in doc_title:
            score += 300
        if normalized_title and normalized_title in doc_summary:
            score += 120
        if normalized_title and normalized_title in text:
            score += 60
        text_raw = str(getattr(chunk, "text", "") or "")
        if "may be cited as" in text_raw.casefold() or "title:" in text_raw.casefold():
            score += 80
        if score > best_score or (score == best_score and idx == 0):
            best_score = score
            best_chunk_id = chunk.chunk_id

    if best_score <= 0:
        return ""
    return best_chunk_id
def doc_ids_for_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    chunk_ids: Sequence[str],
    context_chunks: Sequence[RankedChunk],
) -> set[str]:
    context_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
    doc_ids: set[str] = set()
    for raw_chunk_id in chunk_ids:
        chunk = context_by_id.get(str(raw_chunk_id).strip())
        if chunk is None:
            continue
        doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids
def context_family_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    doc_ids: set[str],
    context_chunks: Sequence[RankedChunk],
) -> list[str]:
    if not doc_ids:
        return []
    ordered: list[str] = []
    seen_pages: set[tuple[str, str]] = set()
    for chunk in context_chunks:
        doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
        if doc_id not in doc_ids:
            continue
        page_key = (doc_id, str(getattr(chunk, "section_path", "") or "").strip())
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        ordered.append(chunk.chunk_id)
    return ordered
def best_support_chunk_id_for_doc_page(
    pipeline: RAGPipelineBuilder,
    *,
    doc_id: str | None,
    page_num: int,
    context_chunks: Sequence[RankedChunk],
) -> str:
    if page_num <= 0:
        return ""

    target_doc_id = str(doc_id or "").strip()
    best_chunk_id = ""
    best_key: tuple[int, float, float, int] | None = None
    for idx, chunk in enumerate(context_chunks):
        chunk_doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
        if target_doc_id and chunk_doc_id != target_doc_id:
            continue
        if pipeline.page_num(str(getattr(chunk, "section_path", "") or "")) != page_num:
            continue
        text = pipeline.normalize_support_text(str(getattr(chunk, "text", "") or "")).casefold()
        score = 0
        if page_num == 1:
            score += 80
            if "may be cited as" in text or "judgment" in text or "claimant" in text or "respondent" in text:
                score += 40
        candidate = (
            score,
            float(getattr(chunk, "rerank_score", 0.0) or 0.0),
            float(getattr(chunk, "retrieval_score", 0.0) or 0.0),
            -idx,
        )
        if best_key is None or candidate > best_key:
            best_key = candidate
            best_chunk_id = chunk.chunk_id
    return best_chunk_id
