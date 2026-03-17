# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed retrieval helpers for the pipeline hot path."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    from rag_challenge.models import RankedChunk, RetrievedChunk

    from .builder import RAGPipelineBuilder

from .constants import (
    _COMMON_ELEMENTS_TITLE_STOPWORDS,
    _COMMON_ELEMENTS_TOKEN_RE,
    _GENERIC_SELF_ADMIN_RE,
    _ISO_DATE_RE,
    _LAW_NO_REF_RE,
    _REGISTRAR_SELF_ADMIN_RE,
    _SLASH_DATE_RE,
    _SUPPORT_STOPWORDS,
    _TEXTUAL_DATE_RE,
)
from .query_rules import (
    _extract_question_title_refs,
    _is_named_amendment_query,
    _is_named_commencement_query,
    _is_named_multi_title_lookup_query,
)

logger = logging.getLogger(__name__)


def named_commencement_title_match_score(pipeline: RAGPipelineBuilder, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
    normalized_ref = re.sub(r"\s+", " ", ref).strip().casefold()
    if not normalized_ref:
        return 0

    haystack = " ".join(
        part
        for part in (
            str(getattr(chunk, "doc_title", "") or ""),
            str(getattr(chunk, "text", "") or "")[:1600],
        )
        if part
    )
    normalized_haystack = re.sub(r"\s+", " ", haystack).strip().casefold()
    if not normalized_haystack:
        return 0

    position = normalized_haystack.find(normalized_ref)
    if position >= 0:
        return 1200 - min(position, 600)

    law_ref_match = _LAW_NO_REF_RE.search(ref)
    if law_ref_match is not None:
        law_no_key = f"law no. {int(law_ref_match.group(1))} of {law_ref_match.group(2)}"
        position = normalized_haystack.find(law_no_key)
        if position >= 0:
            return 1000 - min(position, 600)
        return 0

    ref_tokens = [
        token
        for token in _COMMON_ELEMENTS_TOKEN_RE.findall(normalized_ref)
        if token and token not in _COMMON_ELEMENTS_TITLE_STOPWORDS and len(token) > 2
    ]
    if not ref_tokens:
        return 0
    haystack_tokens = set(_COMMON_ELEMENTS_TOKEN_RE.findall(normalized_haystack))
    overlap = sum(1 for token in ref_tokens if token in haystack_tokens)
    if overlap == len(ref_tokens):
        return 400 + overlap
    if overlap >= max(1, len(ref_tokens) - 1):
        return 240 + overlap
    if overlap >= max(1, (len(ref_tokens) + 1) // 2):
        return 120 + overlap
    return 0
def named_commencement_clause_score(text: str) -> int:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not normalized:
        return 0

    score = 0
    if "commencement" in normalized:
        score += 6
    if "comes into force" in normalized or "shall come into force" in normalized:
        score += 8
    if "enactment notice" in normalized:
        score += 4
    if "90" in normalized and "days following" in normalized:
        score += 4
    if re.search(r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b", normalized):
        score += 3
    return score
def named_multi_title_clause_score(pipeline: RAGPipelineBuilder, *, query: str, text: str) -> int:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not normalized:
        return 0

    query_lower = re.sub(r"\s+", " ", (query or "").strip()).lower()
    score = 0
    if "citation title" in query_lower or "title of" in query_lower or "titles of" in query_lower:
        if "may be cited as" in normalized:
            score += 20
        if "the title is" in normalized or "citation title" in normalized:
            score += 8
    if "administ" in query_lower:
        if pipeline.chunk_has_self_registrar_clause(text=text):
            score += 24
        elif "registrar" in normalized:
            score += 6
    if _is_named_commencement_query(query):
        score += pipeline.named_commencement_clause_score(text)
    if "updated" in query_lower:
        if "updated" in normalized or "amended" in normalized or "effective from" in normalized:
            score += 10
        if _ISO_DATE_RE.search(normalized) or _SLASH_DATE_RE.search(normalized) or _TEXTUAL_DATE_RE.search(normalized):
            score += 6
    return score
def named_amendment_clause_score(pipeline: RAGPipelineBuilder, *, query: str, ref: str, text: str) -> int:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not normalized:
        return 0

    score = 0
    if "amended by" in normalized:
        score += 18
    if "as amended by" in normalized:
        score += 12
    if "enacted on" in normalized or "hereby enact" in normalized:
        score += 8

    ref_terms = {
        token
        for token in pipeline.support_terms(ref)
        if token not in _SUPPORT_STOPWORDS and len(token) > 2
    }
    if ref_terms:
        score += len(ref_terms.intersection(pipeline.support_terms(normalized))) * 8

    query_terms = {
        token
        for token in pipeline.support_terms(query)
        if token not in _SUPPORT_STOPWORDS and token not in ref_terms and len(token) > 2
    }
    if query_terms:
        score += len(query_terms.intersection(pipeline.support_terms(normalized))) * 2

    return score
def named_penalty_clause_score(pipeline: RAGPipelineBuilder, *, query: str, ref: str, text: str) -> int:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not normalized:
        return 0

    amount_match = re.search(
        r"\b(?:usd|us\\$)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,})(?:\.\d+)?\b",
        normalized,
    )
    if amount_match is None:
        return 0

    score = 0
    if "penalt" in normalized:
        score += 10
    if "offence" in normalized or "offense" in normalized:
        score += 6
    if "illegal" in normalized:
        score += 12
    score += 10

    ref_terms = {
        token
        for token in pipeline.support_terms(ref)
        if token not in _SUPPORT_STOPWORDS and len(token) > 2 and token not in {"law", "regulations", "regulation"}
    }
    if ref_terms:
        score += len(ref_terms.intersection(pipeline.support_terms(normalized))) * 8

    query_terms = {
        token
        for token in pipeline.support_terms(query)
        if token not in _SUPPORT_STOPWORDS and token not in ref_terms and len(token) > 2
    }
    if query_terms:
        score += len(query_terms.intersection(pipeline.support_terms(normalized))) * 3

    return score
def chunk_has_named_administration_clause(pipeline: RAGPipelineBuilder, *, text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return False
    return _GENERIC_SELF_ADMIN_RE.search(normalized) is not None
def named_administration_clause_score(pipeline: RAGPipelineBuilder, *, ref: str, text: str) -> int:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not normalized or not pipeline.chunk_has_named_administration_clause(text=text):
        return 0

    score = 18
    if "administered by" in normalized or "shall administer this law" in normalized:
        score += 8
    if "difca" in normalized or "registrar" in normalized:
        score += 4

    ref_terms = {
        token
        for token in pipeline.support_terms(ref)
        if token not in _SUPPORT_STOPWORDS and len(token) > 2 and token not in {"law", "regulations", "regulation"}
    }
    if ref_terms:
        score += len(ref_terms.intersection(pipeline.support_terms(normalized))) * 8
    return score
def chunk_has_self_registrar_clause(pipeline: RAGPipelineBuilder, *, text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return False
    return _REGISTRAR_SELF_ADMIN_RE.search(normalized) is not None
def ensure_self_registrar_context(
    pipeline: RAGPipelineBuilder,
    *,
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not reranked or not retrieved:
        return reranked[: max(0, int(top_n))]

    evidence_by_doc: dict[str, RetrievedChunk] = {}
    page_one_by_doc: dict[str, RetrievedChunk] = {}
    best_by_doc: dict[str, RetrievedChunk] = {}
    for chunk in retrieved:
        doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
        if not doc_id:
            continue
        current_best = best_by_doc.get(doc_id)
        if current_best is None or float(chunk.score) > float(current_best.score):
            best_by_doc[doc_id] = chunk
        section_path = str(getattr(chunk, "section_path", "") or "").lower()
        if "page:1" in section_path:
            current_page_one = page_one_by_doc.get(doc_id)
            if current_page_one is None or float(chunk.score) > float(current_page_one.score):
                page_one_by_doc[doc_id] = chunk
        if not pipeline.chunk_has_self_registrar_clause(text=str(getattr(chunk, "text", "") or "")):
            continue
        current_evidence = evidence_by_doc.get(doc_id)
        if current_evidence is None or float(chunk.score) > float(current_evidence.score):
            evidence_by_doc[doc_id] = chunk

    if not evidence_by_doc:
        return reranked[: max(0, int(top_n))]

    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen: set[str] = set()

    matched_doc_ids: list[str] = []
    for chunk in reranked:
        if chunk.doc_id in evidence_by_doc and chunk.doc_id not in matched_doc_ids:
            matched_doc_ids.append(chunk.doc_id)
    for doc_id in evidence_by_doc:
        if doc_id not in matched_doc_ids:
            matched_doc_ids.append(doc_id)

    for doc_id in matched_doc_ids:
        preferred = [page_one_by_doc.get(doc_id) or best_by_doc.get(doc_id), evidence_by_doc.get(doc_id)]
        for raw in preferred:
            if raw is None or raw.chunk_id in seen:
                continue
            ranked = reranked_by_id.get(raw.chunk_id) or pipeline.raw_to_ranked(raw)
            selected.append(ranked)
            seen.add(raw.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]

    for chunk in reranked:
        if chunk.doc_id not in evidence_by_doc or chunk.chunk_id in seen:
            continue
        selected.append(chunk)
        seen.add(chunk.chunk_id)
        if len(selected) >= top_n:
            break

    return selected[:top_n] if selected else reranked[: max(0, int(top_n))]
def ensure_named_commencement_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    doc_refs: list[str],
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not reranked or not retrieved or not _is_named_commencement_query(query):
        return reranked[: max(0, int(top_n))]

    refs = [ref for ref in doc_refs if str(ref).strip()] or _extract_question_title_refs(query)
    if len(refs) < 2:
        return reranked[: max(0, int(top_n))]

    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen_chunk_ids: set[str] = set()
    matched_doc_ids: set[str] = set()

    for ref in refs:
        best_anchor: RetrievedChunk | None = None
        best_anchor_score = 0
        for raw in retrieved:
            score = pipeline.named_commencement_title_match_score(ref, raw)
            if score > best_anchor_score:
                best_anchor = raw
                best_anchor_score = score
        if best_anchor is None or best_anchor.doc_id in matched_doc_ids:
            continue

        matched_doc_ids.add(best_anchor.doc_id)
        preferred_raw: list[RetrievedChunk] = [best_anchor]
        best_clause: RetrievedChunk | None = None
        best_clause_score = 0
        for raw in retrieved:
            if raw.doc_id != best_anchor.doc_id:
                continue
            score = pipeline.named_commencement_clause_score(str(getattr(raw, "text", "") or ""))
            if score > best_clause_score:
                best_clause = raw
                best_clause_score = score
        if best_clause is not None and best_clause.chunk_id != best_anchor.chunk_id:
            preferred_raw.append(best_clause)

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
def ensure_named_penalty_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    doc_refs: list[str],
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not reranked or not retrieved:
        return reranked[: max(0, int(top_n))]

    refs = [ref for ref in doc_refs if str(ref).strip()] or pipeline.support_question_refs(query)
    if len(refs) < 2:
        return reranked[: max(0, int(top_n))]

    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen_chunk_ids: set[str] = set()
    matched_doc_ids: set[str] = set()

    for ref in refs:
        best_anchor: RetrievedChunk | None = None
        best_anchor_score = 0
        for raw in retrieved:
            score = pipeline.named_commencement_title_match_score(ref, raw)
            if score > best_anchor_score:
                best_anchor = raw
                best_anchor_score = score
        if best_anchor is None or best_anchor.doc_id in matched_doc_ids:
            continue

        matched_doc_ids.add(best_anchor.doc_id)
        preferred_raw: list[RetrievedChunk] = [best_anchor]
        best_clause: RetrievedChunk | None = None
        best_clause_score = 0
        for raw in retrieved:
            if raw.doc_id != best_anchor.doc_id:
                continue
            score = pipeline.named_penalty_clause_score(
                query=query,
                ref=ref,
                text=str(getattr(raw, "text", "") or ""),
            )
            if score > best_clause_score:
                best_clause = raw
                best_clause_score = score
        if best_clause is not None and best_clause.chunk_id != best_anchor.chunk_id:
            preferred_raw.append(best_clause)

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
def ensure_named_amendment_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    doc_refs: list[str],
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not reranked or not retrieved or not _is_named_amendment_query(query):
        return reranked[: max(0, int(top_n))]

    refs = [ref for ref in doc_refs if str(ref).strip()] or pipeline.support_question_refs(query)
    if not refs:
        return reranked[: max(0, int(top_n))]

    amendment_ref = refs[0]
    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen_chunk_ids: set[str] = set()

    best_anchor: RetrievedChunk | None = None
    best_anchor_score = 0
    for raw in retrieved:
        score = pipeline.named_commencement_title_match_score(amendment_ref, raw) + pipeline.named_amendment_clause_score(
            query=query,
            ref=amendment_ref,
            text=str(getattr(raw, "text", "") or ""),
        )
        normalized = re.sub(r"\s+", " ", str(getattr(raw, "text", "") or "").strip()).casefold()
        if "hereby enact" in normalized or "enacted on" in normalized:
            score += 40
        if score > best_anchor_score:
            best_anchor = raw
            best_anchor_score = score

    amender_doc_id = str(best_anchor.doc_id or "").strip() if best_anchor is not None else ""
    if best_anchor is not None:
        preferred_amender: list[RetrievedChunk] = [best_anchor]
        best_clause: RetrievedChunk | None = None
        best_clause_score = 0
        for raw in retrieved:
            if str(raw.doc_id or "").strip() != amender_doc_id:
                continue
            score = pipeline.named_amendment_clause_score(
                query=query,
                ref=amendment_ref,
                text=str(getattr(raw, "text", "") or ""),
            )
            if score > best_clause_score:
                best_clause = raw
                best_clause_score = score
        if best_clause is not None and best_clause.chunk_id != best_anchor.chunk_id:
            preferred_amender.append(best_clause)
        for raw in preferred_amender:
            if raw.chunk_id in seen_chunk_ids:
                continue
            selected.append(reranked_by_id.get(raw.chunk_id) or pipeline.raw_to_ranked(raw))
            seen_chunk_ids.add(raw.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]

    doc_best_clause: dict[str, RetrievedChunk] = {}
    doc_best_score: dict[str, int] = {}
    for raw in retrieved:
        doc_id = str(raw.doc_id or "").strip()
        if not doc_id or doc_id == amender_doc_id:
            continue
        score = pipeline.named_amendment_clause_score(
            query=query,
            ref=amendment_ref,
            text=str(getattr(raw, "text", "") or ""),
        )
        if score <= 0:
            continue
        if score > doc_best_score.get(doc_id, 0):
            doc_best_score[doc_id] = score
            doc_best_clause[doc_id] = raw

    for raw in sorted(
        doc_best_clause.values(),
        key=lambda chunk: (doc_best_score.get(str(chunk.doc_id or "").strip(), 0), float(chunk.score)),
        reverse=True,
    ):
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
def ensure_named_administration_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    doc_refs: list[str],
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not reranked or not retrieved:
        return reranked[: max(0, int(top_n))]

    refs = [ref for ref in doc_refs if str(ref).strip()] or pipeline.support_question_refs(query)
    if len(refs) < 2:
        return reranked[: max(0, int(top_n))]

    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen_chunk_ids: set[str] = set()
    matched_doc_ids: set[str] = set()

    for ref in refs:
        best_anchor = pipeline.best_named_administration_chunk(
            ref=ref,
            chunks=retrieved,
            excluded_doc_ids=tuple(matched_doc_ids),
        )
        best_doc_id = str(best_anchor.doc_id or "").strip() if best_anchor is not None else ""
        if best_anchor is None or best_doc_id in matched_doc_ids:
            continue

        matched_doc_ids.add(best_doc_id)
        preferred_raw: list[RetrievedChunk] = [best_anchor]
        best_clause: RetrievedChunk | None = None
        best_clause_score = 0
        for raw in retrieved:
            if str(raw.doc_id or "").strip() != best_doc_id:
                continue
            score = pipeline.named_administration_clause_score(ref=ref, text=str(getattr(raw, "text", "") or ""))
            if score > best_clause_score:
                best_clause = raw
                best_clause_score = score
        if best_clause is not None and best_clause.chunk_id != best_anchor.chunk_id:
            preferred_raw.append(best_clause)

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
def ensure_named_multi_title_context(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    doc_refs: list[str],
    reranked: list[RankedChunk],
    retrieved: list[RetrievedChunk],
    top_n: int,
) -> list[RankedChunk]:
    if top_n <= 0 or not reranked or not retrieved or not _is_named_multi_title_lookup_query(query):
        return reranked[: max(0, int(top_n))]

    refs = [ref for ref in doc_refs if str(ref).strip()] or pipeline.support_question_refs(query)
    if len(refs) < 2:
        return reranked[: max(0, int(top_n))]

    reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
    selected: list[RankedChunk] = []
    seen_chunk_ids: set[str] = set()
    matched_doc_ids: set[str] = set()

    for ref in refs:
        best_anchor: RetrievedChunk | None = None
        best_anchor_score = 0
        for raw in retrieved:
            score = pipeline.named_commencement_title_match_score(ref, raw)
            if score > best_anchor_score:
                best_anchor = raw
                best_anchor_score = score
        if best_anchor is None or best_anchor.doc_id in matched_doc_ids:
            continue

        matched_doc_ids.add(best_anchor.doc_id)
        preferred_raw: list[RetrievedChunk] = [best_anchor]
        best_clause: RetrievedChunk | None = None
        best_clause_score = 0
        for raw in retrieved:
            if raw.doc_id != best_anchor.doc_id:
                continue
            score = pipeline.named_multi_title_clause_score(
                query=query,
                text=str(getattr(raw, "text", "") or ""),
            )
            if score > best_clause_score:
                best_clause = raw
                best_clause_score = score
        if best_clause is not None and best_clause.chunk_id != best_anchor.chunk_id:
            preferred_raw.append(best_clause)

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
