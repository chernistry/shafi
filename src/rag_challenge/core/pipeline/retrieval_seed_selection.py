# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed retrieval helpers for the pipeline hot path."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rag_challenge.models import RankedChunk, RetrievedChunk

    from .builder import RAGPipelineBuilder

from .constants import (
    _AMENDMENT_TITLE_RE,
    _LAW_NO_REF_RE,
    _SUPPORT_STOPWORDS,
    _SUPPORT_TOKEN_RE,
    _TITLE_CONTEXT_BAD_LEAD_RE,
    _TITLE_GENERIC_QUESTION_LEAD_RE,
    _TITLE_LEADING_CONNECTOR_RE,
    _TITLE_PREPOSITION_BAD_LEAD_RE,
    _TITLE_QUERY_BAD_LEAD_RE,
    _TITLE_REF_BAD_LEAD_RE,
    _TITLE_REF_RE,
)

logger = logging.getLogger(__name__)


def select_seed_chunk_id(pipeline: RAGPipelineBuilder, chunks: list[RetrievedChunk], seed_terms: list[str]) -> str | None:
    if not chunks or not seed_terms:
        return None

    best: tuple[int, int, float, str] | None = None  # (score, -page, retrieval_score, chunk_id)
    for chunk in chunks[: max(1, min(12, len(chunks)))]:
        text = (chunk.text or "").lower()
        if not text:
            continue
        score = sum(1 for term in seed_terms if term and term in text)
        if score <= 0:
            continue
        page = pipeline.section_page_num(getattr(chunk, "section_path", "") or "")
        candidate = (score, -page, float(chunk.score), chunk.chunk_id)
        if best is None or candidate > best:
            best = candidate
    return best[3] if best is not None else None
def boolean_year_seed_chunk_score(pipeline: RAGPipelineBuilder, *, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
    score = pipeline.boolean_year_compare_chunk_score(ref=ref, chunk=chunk)
    if score <= 0:
        return 0

    text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
    score += pipeline.ref_doc_family_consistency_adjustment(ref=ref, chunk=chunk)
    if "may be cited as" in text:
        score += 220
    if "repeal" in text or "replaced" in text or "replaces" in text or "as amended by" in text:
        score -= 260
    if "consolidated version" in text or "last updated" in text:
        score -= 120
    return score
def boolean_admin_seed_chunk_score(pipeline: RAGPipelineBuilder, *, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
    base = pipeline.named_commencement_title_match_score(ref, chunk)
    if base <= 0:
        return 0

    text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
    score = base + pipeline.ref_doc_family_consistency_adjustment(ref=ref, chunk=chunk)
    if pipeline.page_num(str(getattr(chunk, "section_path", "") or "")) <= 4:
        score += 200
    if "administ" in text or "administration" in text:
        score += 220
    if any(marker in text for marker in ("relevant authority", "registrar", "difca", "difc authority")):
        score += 80
    if "may be cited as" in text or "title" in text:
        score += 40
    return score
def case_judge_seed_chunk_score(pipeline: RAGPipelineBuilder, *, chunk: RetrievedChunk | RankedChunk) -> int:
    text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
    if not text:
        return 0

    score = 0
    page_num = pipeline.page_num(str(getattr(chunk, "section_path", "") or ""))
    if page_num == 1:
        score += 320
    elif page_num == 2:
        score += 160
    elif page_num > 2:
        score -= min(180, (page_num - 2) * 28)
    if (
        "order with reasons" in text
        or "judgment of" in text
        or "reasons of" in text
        or "hearing held before" in text
        or "before h.e." in text
        or "judgment of the court of appeal" in text
    ):
        score += 260
    if any(marker in text for marker in ("chief justice", "justice ", "assistant registrar", "registrar", "sct judge")):
        score += 260
    if "claim no." in text or "case no:" in text:
        score += 40
    if any(marker in text for marker in ("issued by:", "introduction", "background", "discussion and determination")):
        score -= 40
    return score
def case_ref_identity_score(pipeline: RAGPipelineBuilder, *, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
    normalized_ref = pipeline.normalize_support_text(ref).casefold()
    if not normalized_ref:
        return 0

    haystack = pipeline.normalize_support_text(
        " ".join(
            part
            for part in (
                str(getattr(chunk, "doc_title", "") or ""),
                str(getattr(chunk, "doc_summary", "") or ""),
                str(getattr(chunk, "text", "") or "")[:900],
            )
            if part
        )
    ).casefold()
    if not haystack:
        return 0

    if normalized_ref in haystack:
        return 1000 - min(haystack.find(normalized_ref), 600)

    ordered_ref_tokens = [
        token.casefold()
        for token in _SUPPORT_TOKEN_RE.findall(normalized_ref)
        if token.casefold() not in _SUPPORT_STOPWORDS and len(token) > 2
    ]
    if not ordered_ref_tokens:
        return 0

    overlap = 0
    cursor = 0
    for token in ordered_ref_tokens:
        idx = haystack.find(token, cursor)
        if idx >= 0:
            overlap += 1
            cursor = idx + len(token)
        elif token in haystack:
            overlap += 1
    if overlap < min(2, len(ordered_ref_tokens)):
        return 0
    return overlap * 120
def select_case_judge_seed_chunk_id(pipeline: RAGPipelineBuilder, chunks: Sequence[RetrievedChunk]) -> str | None:
    best_chunk_id = ""
    best_key: tuple[int, int, float] | None = None
    for chunk in chunks:
        score = pipeline.case_judge_seed_chunk_score(chunk=chunk)
        if score <= 0:
            continue
        page_num = pipeline.page_num(str(getattr(chunk, "section_path", "") or ""))
        candidate = (score, -max(page_num, 0), float(chunk.score))
        if best_key is None or candidate > best_key:
            best_key = candidate
            best_chunk_id = chunk.chunk_id
    return best_chunk_id or None
def case_issue_date_seed_chunk_score(pipeline: RAGPipelineBuilder, *, chunk: RetrievedChunk | RankedChunk) -> int:
    text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
    if not text:
        return 0

    score = 0
    page_num = pipeline.page_num(str(getattr(chunk, "section_path", "") or ""))
    if page_num <= 2:
        score += 220
    if "date of issue" in text:
        score += 320
    if "issued by" in text or "at:" in text:
        score += 60
    if "decision date" in text or "judgment" in text or "judgement" in text:
        score -= 80
    if "claim no." in text:
        score += 20
    return score
def select_case_issue_date_seed_chunk_id(pipeline: RAGPipelineBuilder, chunks: Sequence[RetrievedChunk]) -> str | None:
    best_chunk_id = ""
    best_score = 0
    for chunk in chunks:
        score = pipeline.case_issue_date_seed_chunk_score(chunk=chunk)
        if score > best_score:
            best_score = score
            best_chunk_id = chunk.chunk_id
    return best_chunk_id or None
def case_outcome_seed_chunk_score(pipeline: RAGPipelineBuilder, *, chunk: RetrievedChunk | RankedChunk) -> int:
    text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
    if not text:
        return 0

    score = 0
    page_num = pipeline.page_num(str(getattr(chunk, "section_path", "") or ""))
    if page_num == 1:
        score += 260
    elif page_num == 2:
        score += 120
    if "it is hereby ordered that" in text:
        score += 320
    if "order with reasons" in text:
        score += 180
    if "application is refused" in text or "application was dismissed" in text:
        score += 220
    if "no order as to costs" in text or "costs" in text:
        score += 40
    return score
def select_case_outcome_seed_chunk_id(pipeline: RAGPipelineBuilder, chunks: Sequence[RetrievedChunk]) -> str | None:
    best_chunk_id = ""
    best_score = 0
    for chunk in chunks:
        score = pipeline.case_outcome_seed_chunk_score(chunk=chunk)
        if score > best_score:
            best_score = score
            best_chunk_id = chunk.chunk_id
    return best_chunk_id or None
def ref_doc_family_consistency_adjustment(pipeline: RAGPipelineBuilder, *, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
    law_ref_match = _LAW_NO_REF_RE.search(ref)
    if law_ref_match is None:
        return 0

    target_pair = (int(law_ref_match.group(1)), law_ref_match.group(2))
    identity_blob = pipeline.normalize_support_text(
        " ".join(
            part
            for part in (
                str(getattr(chunk, "doc_title", "") or ""),
                str(getattr(chunk, "doc_summary", "") or ""),
            )
            if part
        )
    ).casefold()
    if not identity_blob:
        return 0

    score = 0
    law_pairs = {
        (int(match.group(1)), match.group(2))
        for match in _LAW_NO_REF_RE.finditer(identity_blob)
    }
    if target_pair in law_pairs:
        score += 140

    foreign_pairs = {
        pair for pair in law_pairs
        if pair != target_pair and pair[1] != target_pair[1]
    }
    if foreign_pairs:
        score -= min(260, len(foreign_pairs) * 90)

    if any(marker in identity_blob for marker in ("consolidated version", "amendments up to", "as amended by")):
        if foreign_pairs:
            score -= 120
        else:
            score -= 40

    return score
def is_notice_focus_query(pipeline: RAGPipelineBuilder, query: str) -> bool:
    normalized = re.sub(r"\s+", " ", (query or "").strip()).casefold()
    if not normalized:
        return False
    return (
        "enactment notice" in normalized
        or "enacted law" in normalized
        or ("come into force" in normalized and "precise calendar date" in normalized)
    )
def notice_doc_score(pipeline: RAGPipelineBuilder, *, query: str, raw: RetrievedChunk) -> int:
    normalized = re.sub(r"\s+", " ", str(getattr(raw, "text", "") or "")).strip().casefold()
    if not normalized:
        return 0

    doc_title = re.sub(r"\s+", " ", str(getattr(raw, "doc_title", "") or "")).strip().casefold()
    explicit_notice_doc = "enactment notice" in doc_title or normalized.startswith("enactment notice")
    if not explicit_notice_doc and "hereby enact" not in normalized:
        return 0

    score = 0
    if explicit_notice_doc:
        score += 320
    if "hereby enact" in normalized:
        score += 220
    if "shall come into force" in normalized or "comes into force" in normalized:
        score += 140
    if re.search(r"\b(?:on\s+this\s+)?\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[a-z]+\s+\d{4}\b", normalized):
        score += 200
    if re.search(r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b", normalized):
        score += 80
    if "date specified in the enactment notice" in normalized and not explicit_notice_doc:
        score -= 260

    query_lower = pipeline.normalize_support_text(query).casefold()
    if "full title" in query_lower and "in the form now attached" in normalized:
        score += 120
    if "precise calendar date" in query_lower and re.search(r"\b\d{4}\b", normalized):
        score += 60
    return score
def is_consolidated_or_amended_family_chunk(pipeline: RAGPipelineBuilder, *, chunk: RetrievedChunk | RankedChunk) -> bool:
    family_blob = pipeline.normalize_support_text(
        " ".join(
            part
            for part in (
                str(getattr(chunk, "doc_title", "") or ""),
                str(getattr(chunk, "doc_summary", "") or ""),
            )
            if part
        )
    ).casefold()
    return any(marker in family_blob for marker in ("consolidated version", "amendments up to", "as amended by"))
def is_canonical_ref_family_chunk(
    pipeline: RAGPipelineBuilder,
    *,
    ref: str,
    chunk: RetrievedChunk | RankedChunk,
) -> bool:
    law_ref_match = _LAW_NO_REF_RE.search(ref)
    if law_ref_match is None:
        return False
    if pipeline.is_consolidated_or_amended_family_chunk(chunk=chunk):
        return False

    target_key = f"law no. {int(law_ref_match.group(1))} of {law_ref_match.group(2)}"
    combined_blob = pipeline.normalize_support_text(
        " ".join(
            part
            for part in (
                str(getattr(chunk, "doc_title", "") or ""),
                str(getattr(chunk, "doc_summary", "") or ""),
                str(getattr(chunk, "text", "") or ""),
            )
            if part
        )
    ).casefold()
    if target_key not in combined_blob:
        return False
    return pipeline.named_commencement_title_match_score(ref, chunk) > 0
def best_named_administration_chunk(
    pipeline: RAGPipelineBuilder,
    *,
    ref: str,
    chunks: Sequence[RetrievedChunk],
    excluded_doc_ids: Sequence[str] = (),
) -> RetrievedChunk | None:
    excluded = {str(doc_id).strip() for doc_id in excluded_doc_ids if str(doc_id).strip()}
    best_canonical_clause_chunk: RetrievedChunk | None = None
    best_canonical_clause_tuple: tuple[int, int, float] | None = None
    best_clause_chunk: RetrievedChunk | None = None
    best_clause_tuple: tuple[int, int, float] | None = None
    best_anchor_chunk: RetrievedChunk | None = None
    best_anchor_tuple: tuple[int, int, float] | None = None

    for chunk in chunks:
        doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
        if doc_id and doc_id in excluded:
            continue

        anchor_score = pipeline.boolean_admin_seed_chunk_score(ref=ref, chunk=chunk)
        clause_score = pipeline.named_administration_clause_score(ref=ref, text=str(getattr(chunk, "text", "") or ""))
        if anchor_score <= 0 and clause_score <= 0:
            continue

        page_bonus = 1_000 - min(pipeline.page_num(str(getattr(chunk, "section_path", "") or "")), 999)
        retrieval_score = float(getattr(chunk, "score", getattr(chunk, "rerank_score", 0.0)) or 0.0)
        anchor_tuple = (anchor_score, page_bonus, retrieval_score)
        if best_anchor_tuple is None or anchor_tuple > best_anchor_tuple:
            best_anchor_tuple = anchor_tuple
            best_anchor_chunk = chunk

        if clause_score > 0:
            family_adjustment = pipeline.ref_doc_family_consistency_adjustment(ref=ref, chunk=chunk)
            clause_tuple = (clause_score, family_adjustment + anchor_score + page_bonus, retrieval_score)
            if pipeline.is_canonical_ref_family_chunk(ref=ref, chunk=chunk) and (
                best_canonical_clause_tuple is None or clause_tuple > best_canonical_clause_tuple
            ):
                best_canonical_clause_tuple = clause_tuple
                best_canonical_clause_chunk = chunk
            if best_clause_tuple is None or clause_tuple > best_clause_tuple:
                best_clause_tuple = clause_tuple
                best_clause_chunk = chunk

    return best_canonical_clause_chunk or best_clause_chunk or best_anchor_chunk
def select_targeted_title_seed_chunk_id(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    answer_type: str,
    ref: str,
    chunks: Sequence[RetrievedChunk],
    seed_terms: Sequence[str],
) -> str | None:
    normalized_query = re.sub(r"\s+", " ", query).strip().casefold()
    if not chunks:
        return None

    scorer: Callable[[RetrievedChunk], int] | None = None
    if answer_type == "boolean" and "same year" in normalized_query:
        def _score_year_seed(chunk: RetrievedChunk) -> int:
            return pipeline.boolean_year_seed_chunk_score(ref=ref, chunk=chunk)

        scorer = _score_year_seed
    elif answer_type == "boolean" and "administ" in normalized_query:
        def _score_admin_seed(chunk: RetrievedChunk) -> int:
            return pipeline.boolean_admin_seed_chunk_score(ref=ref, chunk=chunk)

        scorer = _score_admin_seed

    if scorer is not None:
        best_chunk_id = ""
        best_score = 0
        for chunk in chunks:
            score = scorer(chunk)
            if score > best_score:
                best_score = score
                best_chunk_id = chunk.chunk_id
        if best_chunk_id:
            return best_chunk_id

    return pipeline.select_seed_chunk_id(list(chunks), list(seed_terms))
def extract_title_refs_from_query(query: str) -> list[str]:
    raw = (query or "").strip()
    if not raw:
        return []
    found: list[str] = []
    for match in _AMENDMENT_TITLE_RE.finditer(raw):
        ref = re.sub(r"\s+", " ", match.group(1).strip())
        ref = _TITLE_REF_BAD_LEAD_RE.sub("", ref)
        ref = _TITLE_QUERY_BAD_LEAD_RE.sub("", ref)
        ref = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", ref)
        ref = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", ref)
        ref = _TITLE_LEADING_CONNECTOR_RE.sub("", ref).strip(" ,.;:")
        if ref:
            found.append(ref)
    for match in _TITLE_REF_RE.finditer(raw):
        title = re.sub(r"\s+", " ", match.group(1).strip())
        title = _TITLE_REF_BAD_LEAD_RE.sub("", title).strip()
        title = _TITLE_CONTEXT_BAD_LEAD_RE.sub("", title).strip()
        title = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", title).strip()
        title = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", title).strip()
        title = _TITLE_LEADING_CONNECTOR_RE.sub("", title).strip(" ,.;:")
        year = match.group(2).strip() if match.lastindex and match.lastindex >= 2 and match.group(2) else ""
        if not title:
            continue
        # Normalize suffix casing and pluralization for matching ingestion citations.
        words = title.split(" ")
        if words:
            last = words[-1].lower()
            if last == "law":
                words[-1] = "Law"
            elif last in {"regulation", "regulations"}:
                words[-1] = "Regulations"
        normalized = " ".join(words).strip()
        if year:
            normalized = f"{normalized} {year}"
        found.append(normalized)

    seen: set[str] = set()
    out: list[str] = []
    for item in found:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    pruned: list[str] = []
    lowered_out = [item.casefold() for item in out]
    for idx, item in enumerate(out):
        lowered = lowered_out[idx]
        if any(
            idx != other_idx
            and lowered != other_lowered
            and re.search(rf"\b{re.escape(lowered)}\b", other_lowered)
            for other_idx, other_lowered in enumerate(lowered_out)
        ):
            continue
        pruned.append(item)
    return pruned
def extract_title_ref_from_chunk_text(chunk: RetrievedChunk) -> str:
    text = str(getattr(chunk, "text", "") or "")
    for match in _TITLE_REF_RE.finditer(text):
        title = re.sub(r"\s+", " ", match.group(1).strip())
        year = match.group(2).strip() if match.lastindex and match.lastindex >= 2 and match.group(2) else ""
        if title:
            return f"{title} {year}".strip()
    return re.sub(r"\s+", " ", str(getattr(chunk, "doc_title", "") or "").strip())
