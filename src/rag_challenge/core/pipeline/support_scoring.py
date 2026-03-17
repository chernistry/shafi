# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed support helpers for the pipeline hot path."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rag_challenge.models import RankedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models import RetrievedChunk

    from .builder import RAGPipelineBuilder

from .constants import (
    _CITE_RE,
    _LAW_NO_REF_RE,
    _SUPPORT_STOPWORDS,
    _YEAR_RE,
)
from .query_rules import (
    _is_account_effective_dates_query,
    _is_named_commencement_query,
)


def boolean_year_compare_chunk_score(pipeline: RAGPipelineBuilder, *, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
    base = pipeline.named_commencement_title_match_score(ref, chunk)
    if base <= 0:
        return 0

    text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
    page_num = pipeline.page_num(str(getattr(chunk, "section_path", "") or ""))
    score = base
    if page_num <= 4:
        score += 240
    elif page_num <= 8:
        score += 80
    if _LAW_NO_REF_RE.search(text):
        score += 180
    if _YEAR_RE.search(text):
        score += 60
    if "title" in text:
        score += 60
    if "enact" in text or "legislative authority" in text:
        score += 40
    return score
def account_effective_clause_score(pipeline: RAGPipelineBuilder, *, text: str) -> int:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not normalized:
        return 0

    score = 0
    if "pre-existing accounts" in normalized:
        score += 14
    if "new accounts" in normalized:
        score += 14
    if "effective date" in normalized:
        score += 12
    if "31 december" in normalized or "1 january" in normalized:
        score += 10
    return score
def account_enactment_clause_score(pipeline: RAGPipelineBuilder, *, ref: str, raw: RetrievedChunk) -> int:
    normalized = re.sub(r"\s+", " ", str(getattr(raw, "text", "") or "")).strip().casefold()
    if not normalized:
        return 0
    if not any(marker in normalized for marker in ("hereby enact", "enactment notice", "enacted on")):
        return 0

    doc_title = re.sub(r"\s+", " ", str(getattr(raw, "doc_title", "") or "")).strip().casefold()
    explicit_notice_doc = "enactment notice" in doc_title or normalized.startswith("enactment notice")
    generic_notice_reference = any(
        phrase in normalized
        for phrase in (
            "date specified in the enactment notice",
            "comes into force on the date specified in the enactment notice",
        )
    )
    explicit_enactment_date = bool(re.search(r"\bhereby enact\s+on\s+(?:this\s+)?[0-9]{1,2}", normalized))

    score = pipeline.named_commencement_title_match_score(ref, raw)
    if score <= 0:
        ref_terms = {
            token
            for token in pipeline.support_terms(ref)
            if token not in _SUPPORT_STOPWORDS and len(token) > 2
        }
        overlap = len(ref_terms.intersection(pipeline.support_terms(normalized)))
        if overlap >= max(2, len(ref_terms) - 1):
            score += 180 + (overlap * 18)
    if "hereby enact" in normalized:
        score += 220
    if "enactment notice" in normalized:
        score += 120
    if "enacted on" in normalized:
        score += 100
    if explicit_enactment_date:
        score += 260
    if explicit_notice_doc:
        score += 320
    if generic_notice_reference and not explicit_notice_doc and not explicit_enactment_date:
        score -= 760
    if _YEAR_RE.search(normalized):
        score += 30
    if pipeline.page_num(str(getattr(raw, "section_path", "") or "")) == 1:
        score += 50
    return score
def restriction_effectiveness_clause_score(
    pipeline: RAGPipelineBuilder,
    *,
    ref: str,
    chunk: RetrievedChunk | RankedChunk,
) -> int:
    normalized = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
    if not normalized or "restriction" not in normalized or "actual knowledge" not in normalized:
        return 0

    score = pipeline.named_commencement_title_match_score(ref, chunk)
    if score <= 0:
        ref_terms = {
            token
            for token in pipeline.support_terms(ref)
            if token not in _SUPPORT_STOPWORDS and len(token) > 2
        }
        overlap = len(ref_terms.intersection(pipeline.support_terms(normalized)))
        if overlap >= max(2, len(ref_terms) - 1):
            score += 180 + (overlap * 18)
    if "ineffective against any person other than a person who had actual knowledge" in normalized:
        score += 460
    if "restriction on transfer" in normalized:
        score += 180
    if "actual knowledge" in normalized:
        score += 140
    if "uncertificated" in normalized:
        score += 90
    if "notified" in normalized:
        score += 70
    if "article 23" in normalized:
        score += 80
    return score
def doc_shortlist_score(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    ref: str,
    doc_chunks: Sequence[RetrievedChunk],
) -> int:
    if not doc_chunks:
        return 0

    normalized_ref = pipeline.normalize_support_text(ref).casefold()
    title_score = max(pipeline.named_commencement_title_match_score(ref, chunk) for chunk in doc_chunks)
    identity_blob = pipeline.normalize_support_text(
        " ".join(
            part
            for part in (
                str(doc_chunks[0].doc_title or ""),
                str(doc_chunks[0].doc_summary or ""),
            )
            if part
        )
    ).casefold()
    identity_score = 0
    if normalized_ref and normalized_ref in identity_blob:
        identity_score += 900
    law_ref_match = _LAW_NO_REF_RE.search(ref)
    if law_ref_match is not None:
        law_key = f"law no. {int(law_ref_match.group(1))} of {law_ref_match.group(2)}"
        if law_key in identity_blob:
            identity_score += 700
    ref_terms = pipeline.support_terms(ref)
    identity_terms = pipeline.support_terms(identity_blob)
    if ref_terms:
        overlap = len(ref_terms.intersection(identity_terms))
        if overlap >= min(2, len(ref_terms)):
            identity_score += overlap * 90

    query_lower = pipeline.normalize_support_text(query).casefold()
    surrogate_enabled = _is_named_commencement_query(query) or _is_account_effective_dates_query(query)
    enactment_surrogate = 0
    if surrogate_enabled:
        enactment_surrogate = max(
            (pipeline.account_enactment_clause_score(ref=ref, raw=chunk) for chunk in doc_chunks[:4]),
            default=0,
        )
    administration_surrogate = 0
    if "administ" in query_lower:
        administration_surrogate = max(
            (
                pipeline.named_administration_clause_score(
                    ref=ref,
                    text=str(getattr(chunk, "text", "") or ""),
                )
                + (140 if pipeline.page_num(str(getattr(chunk, "section_path", "") or "")) <= 5 else 0)
                + (
                    40
                    if "may be cited as"
                    in re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
                    else 0
                )
            )
            for chunk in doc_chunks[:6]
        )
    if title_score <= 0:
        if enactment_surrogate <= 0 and administration_surrogate <= 0:
            return 0
        title_score = min(320, max(enactment_surrogate, administration_surrogate))
    if identity_score <= 0 and max(enactment_surrogate, administration_surrogate) > 0:
        identity_score = min(450, max(enactment_surrogate, administration_surrogate))
    if identity_score <= 0:
        return 0
    if administration_surrogate > 0:
        identity_score += min(620, administration_surrogate * 12)
    identity_score += pipeline.ref_doc_family_consistency_adjustment(ref=ref, chunk=doc_chunks[0])

    query_terms = pipeline.support_terms(query)
    best_overlap = 0
    best_retrieval_score = 0.0
    for chunk in doc_chunks[:4]:
        blob = pipeline.chunk_support_blob(
            RankedChunk(
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
        )
        overlap = len(query_terms.intersection(pipeline.support_terms(blob)))
        if ref_terms and ref_terms.issubset(pipeline.support_terms(blob)):
            overlap += 4
        best_overlap = max(best_overlap, overlap)
        best_retrieval_score = max(best_retrieval_score, float(chunk.score))

    return identity_score + title_score + (best_overlap * 10) + int(best_retrieval_score * 100)
def apply_doc_shortlist_gating(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    doc_refs: Sequence[str],
    retrieved: Sequence[RetrievedChunk],
    must_keep_chunk_ids: Sequence[str] = (),
) -> list[RetrievedChunk]:
    if not retrieved:
        return []

    refs = pipeline.combined_named_refs(query=query, doc_refs=doc_refs)
    if not refs:
        return list(retrieved)

    chunks_by_doc: dict[str, list[RetrievedChunk]] = {}
    ordered_docs: list[str] = []
    for chunk in retrieved:
        doc_id = str(chunk.doc_id or "").strip()
        if not doc_id:
            continue
        if doc_id not in chunks_by_doc:
            ordered_docs.append(doc_id)
        chunks_by_doc.setdefault(doc_id, []).append(chunk)

    selected_doc_ids: set[str] = set()
    for ref in refs[:4]:
        scored_docs: list[tuple[int, float, str]] = []
        for doc_id in ordered_docs:
            doc_chunks = chunks_by_doc.get(doc_id, [])
            score = pipeline.doc_shortlist_score(query=query, ref=ref, doc_chunks=doc_chunks)
            if score <= 0:
                continue
            best_score = max(float(chunk.score) for chunk in doc_chunks) if doc_chunks else 0.0
            scored_docs.append((score, best_score, doc_id))
        scored_docs.sort(reverse=True)
        for _score, _best_score, doc_id in scored_docs[:2]:
            selected_doc_ids.add(doc_id)

    if _is_account_effective_dates_query(query):
        best_notice_doc: tuple[int, float, str] | None = None
        primary_ref = refs[0]
        for doc_id in ordered_docs:
            doc_chunks = chunks_by_doc.get(doc_id, [])
            surrogate = max(
                (pipeline.account_enactment_clause_score(ref=primary_ref, raw=chunk) for chunk in doc_chunks[:4]),
                default=0,
            )
            if surrogate <= 0:
                continue
            best_score = max(float(chunk.score) for chunk in doc_chunks) if doc_chunks else 0.0
            candidate = (surrogate, best_score, doc_id)
            if best_notice_doc is None or candidate > best_notice_doc:
                best_notice_doc = candidate
        if best_notice_doc is not None:
            selected_doc_ids.add(best_notice_doc[2])

    must_keep_ids = {chunk_id for chunk_id in must_keep_chunk_ids if str(chunk_id).strip()}
    if must_keep_ids:
        for chunk in retrieved:
            if chunk.chunk_id not in must_keep_ids:
                continue
            doc_id = str(chunk.doc_id or "").strip()
            if doc_id:
                selected_doc_ids.add(doc_id)

    if not selected_doc_ids:
        return list(retrieved)
    return [chunk for chunk in retrieved if str(chunk.doc_id or "").strip() in selected_doc_ids]
def normalize_numeric_text(text: str) -> str:
    return re.sub(r"[,\s]", "", (text or "").strip())
def chunk_support_blob(pipeline: RAGPipelineBuilder, chunk: RankedChunk) -> str:
    return pipeline.normalize_support_text(
        " ".join(
            part
            for part in (
                str(chunk.doc_title or ""),
                str(chunk.doc_summary or ""),
                str(chunk.text or ""),
            )
            if part
        )
    )
def chunk_support_score(
    pipeline: RAGPipelineBuilder,
    *,
    answer_type: str,
    query: str,
    fragment: str,
    chunk: RankedChunk,
) -> int:
    blob = pipeline.chunk_support_blob(chunk)
    if not blob:
        return 0

    blob_lower = blob.casefold()
    fragment_clean = pipeline.normalize_support_text(_CITE_RE.sub("", fragment))
    fragment_lower = fragment_clean.casefold()
    query_lower = pipeline.normalize_support_text(query).casefold()
    score = 0

    if fragment_lower:
        if len(fragment_lower) >= 8 and fragment_lower in blob_lower:
            score += 80
        fragment_terms = pipeline.support_terms(fragment_clean)
        if fragment_terms:
            blob_terms = pipeline.support_terms(blob)
            score += len(fragment_terms.intersection(blob_terms)) * 8

    query_terms = pipeline.support_terms(query)
    if query_terms:
        score += len(query_terms.intersection(pipeline.support_terms(blob))) * 3

    for ref in pipeline.support_question_refs(query):
        normalized_ref = pipeline.normalize_support_text(ref).casefold()
        if normalized_ref and normalized_ref in blob_lower:
            score += 30

    kind = answer_type.strip().lower()
    if kind == "number":
        numeric_answer = pipeline.normalize_numeric_text(fragment_clean)
        if not numeric_answer or numeric_answer not in pipeline.normalize_numeric_text(blob):
            return 0
        score += 120
    elif kind == "date":
        date_variants = pipeline.date_fragment_variants(fragment_clean)
        if not date_variants or not any(variant in blob_lower for variant in date_variants):
            return 0
        score += 120
    elif kind in {"name", "names"}:
        if fragment_lower and fragment_lower in blob_lower:
            score += 100
        else:
            title_score = pipeline.named_commencement_title_match_score(fragment_clean, chunk)
            if title_score <= 0:
                return 0
            score += max(80, min(title_score, 140))
    elif kind == "boolean":
        polarity_answer = fragment_clean.strip().lower()
        positive_hits = sum(
            marker in blob_lower
            for marker in (" may ", " can ", " shall ", " entitled ", " includes ", " must ", " effective ")
        )
        negative_hits = sum(
            marker in blob_lower
            for marker in (" not ", " no ", " may not ", " shall not ", " ineffective ", " prohibited ")
        )
        if polarity_answer.startswith("yes"):
            score += positive_hits * 4
        elif polarity_answer.startswith("no"):
            score += negative_hits * 4
        if query_lower and query_lower in blob_lower:
            score += 20

    if pipeline.is_notice_focus_query(query):
        explicit_notice_doc = "enactment notice" in blob_lower or "hereby enact" in blob_lower
        generic_notice_reference = "date specified in the enactment notice" in blob_lower
        if explicit_notice_doc:
            score += 140
        elif generic_notice_reference:
            score -= 120

    return score
def best_support_chunk_id(
    pipeline: RAGPipelineBuilder,
    *,
    answer_type: str,
    query: str,
    fragment: str,
    context_chunks: Sequence[RankedChunk],
    allow_first_chunk_fallback: bool,
) -> str:
    best_chunk_id = ""
    best_score = -1
    for idx, chunk in enumerate(context_chunks):
        score = pipeline.chunk_support_score(
            answer_type=answer_type,
            query=query,
            fragment=fragment,
            chunk=chunk,
        )
        if score > best_score:
            best_score = score
            best_chunk_id = chunk.chunk_id
        if score == best_score and best_chunk_id and idx == 0:
            best_chunk_id = chunk.chunk_id

    if best_score > 0 and best_chunk_id:
        return best_chunk_id
    if allow_first_chunk_fallback and context_chunks:
        return context_chunks[0].chunk_id
    return ""
