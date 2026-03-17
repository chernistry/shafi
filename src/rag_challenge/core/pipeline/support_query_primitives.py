# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed support helpers for the pipeline hot path."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models import RankedChunk, RetrievedChunk

    from .builder import RAGPipelineBuilder

from .constants import (
    _LAW_NO_REF_RE,
    _MONTH_NAME_TO_NUMBER,
    _MONTH_NUMBER_TO_NAME,
    _SUPPORT_STOPWORDS,
    _SUPPORT_TOKEN_RE,
)
from .query_rules import (
    _extract_question_title_refs,
    _is_account_effective_dates_query,
    _is_broad_enumeration_query,
    _is_citation_title_query,
    _is_common_elements_query,
    _is_interpretation_sections_common_elements_query,
    _is_named_commencement_query,
    _is_restriction_effectiveness_query,
)


def normalize_support_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())
def support_terms(pipeline: RAGPipelineBuilder, text: str) -> set[str]:
    return {
        token.lower()
        for token in _SUPPORT_TOKEN_RE.findall(text or "")
        if len(token) > 2 and token.lower() not in _SUPPORT_STOPWORDS
    }
def support_question_refs(pipeline: RAGPipelineBuilder, query: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for ref in _extract_question_title_refs(query):
        normalized = pipeline.normalize_support_text(ref)
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            refs.append(normalized)
    for match in _LAW_NO_REF_RE.finditer(query or ""):
        ref = f"Law No. {int(match.group(1))} of {match.group(2)}"
        normalized = pipeline.normalize_support_text(ref)
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            refs.append(normalized)
    return refs
def paired_support_question_refs(pipeline: RAGPipelineBuilder, query: str) -> list[str]:
    title_refs = [
        pipeline.normalize_support_text(ref)
        for ref in _extract_question_title_refs(query)
        if pipeline.normalize_support_text(ref)
    ]
    law_refs = [
        pipeline.normalize_support_text(f"Law No. {int(match.group(1))} of {match.group(2)}")
        for match in _LAW_NO_REF_RE.finditer(query or "")
    ]
    if len(title_refs) < 2 or len(title_refs) != len(law_refs):
        return pipeline.support_question_refs(query)

    paired_refs: list[str] = []
    seen: set[str] = set()
    for title_ref, law_ref in zip(title_refs, law_refs, strict=False):
        law_suffix = law_ref[4:] if law_ref.startswith("Law ") else law_ref
        combined = pipeline.normalize_support_text(f"{title_ref} {law_suffix}")
        key = combined.casefold()
        if not combined or key in seen:
            continue
        seen.add(key)
        paired_refs.append(combined)

    return paired_refs or pipeline.support_question_refs(query)
def combined_named_refs(pipeline: RAGPipelineBuilder, *, query: str, doc_refs: Sequence[str]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for ref in [*doc_refs, *pipeline.support_question_refs(query)]:
        normalized = pipeline.normalize_support_text(str(ref))
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        refs.append(normalized)
    return refs
def ordinal_suffix(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
def date_fragment_variants(pipeline: RAGPipelineBuilder, fragment: str) -> set[str]:
    normalized = pipeline.normalize_support_text(fragment).casefold().replace(",", "")
    if not normalized:
        return set()

    year = month = day = 0
    iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", normalized)
    if iso_match is not None:
        year = int(iso_match.group(1))
        month = int(iso_match.group(2))
        day = int(iso_match.group(3))
    else:
        slash_match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", normalized)
        if slash_match is not None:
            first = int(slash_match.group(1))
            second = int(slash_match.group(2))
            year = int(slash_match.group(3))
            day, month = (first, second) if first > 12 else (second, first)
        else:
            textual_match = re.search(
                r"\b(\d{1,2})(?:st|nd|rd|th)?(?:\s+day\s+of)?\s+([a-z]+)\s+(\d{4})\b",
                normalized,
            )
            if textual_match is None:
                return set()
            month = _MONTH_NAME_TO_NUMBER.get(textual_match.group(2), 0)
            if month <= 0:
                return set()
            day = int(textual_match.group(1))
            year = int(textual_match.group(3))

    if not (1 <= month <= 12 and 1 <= day <= 31 and year > 0):
        return set()

    month_name = _MONTH_NUMBER_TO_NAME[month]
    ordinal = pipeline.ordinal_suffix(day)
    variants = {
        f"{year:04d}-{month:02d}-{day:02d}",
        f"{day}/{month}/{year}",
        f"{day:02d}/{month:02d}/{year}",
        f"{day} {month_name} {year}",
        f"{day}{ordinal} {month_name} {year}",
        f"{day} day of {month_name} {year}",
        f"{day}{ordinal} day of {month_name} {year}",
    }
    return {variant.casefold() for variant in variants if variant}
def matched_doc_chunks_for_ref(
    pipeline: RAGPipelineBuilder,
    *,
    ref: str,
    retrieved: Sequence[RetrievedChunk],
) -> list[RetrievedChunk]:
    best_anchor_doc_id = ""
    best_anchor_score = 0
    for raw in retrieved:
        score = pipeline.named_commencement_title_match_score(ref, raw)
        if score > best_anchor_score:
            best_anchor_doc_id = raw.doc_id
            best_anchor_score = score
    if not best_anchor_doc_id or best_anchor_score <= 0:
        return []
    return [raw for raw in retrieved if raw.doc_id == best_anchor_doc_id]
def ref_has_criterion_support(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    ref: str,
    ref_chunks: Sequence[RetrievedChunk],
) -> bool:
    if not ref_chunks:
        return False
    query_lower = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if _is_common_elements_query(query):
        interpretation_sections = _is_interpretation_sections_common_elements_query(query)
        return any(
            pipeline.common_elements_evidence_score(str(getattr(chunk, "text", "") or ""), interpretation_sections=interpretation_sections) > 0
            for chunk in ref_chunks
        )
    if "penalt" in query_lower:
        return any(
            pipeline.named_penalty_clause_score(
                query=query,
                ref=ref,
                text=str(getattr(chunk, "text", "") or ""),
            )
            > 0
            for chunk in ref_chunks
        )
    if "administ" in query_lower:
        return any(
            pipeline.named_administration_clause_score(
                ref=ref,
                text=str(getattr(chunk, "text", "") or ""),
            )
            > 0
            for chunk in ref_chunks
        )
    return any(
        pipeline.named_multi_title_clause_score(query=query, text=str(getattr(chunk, "text", "") or "")) > 0
        for chunk in ref_chunks
    )
def missing_named_ref_targets(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    doc_refs: Sequence[str],
    retrieved: Sequence[RetrievedChunk],
) -> list[str]:
    refs = [ref for ref in doc_refs if str(ref).strip()] or pipeline.support_question_refs(query)
    if len(refs) < 2:
        return []
    missing: list[str] = []
    for ref in refs:
        ref_chunks = pipeline.matched_doc_chunks_for_ref(ref=ref, retrieved=retrieved)
        if not pipeline.ref_has_criterion_support(query=query, ref=ref, ref_chunks=ref_chunks):
            missing.append(ref)
    return missing
def targeted_named_ref_query(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    ref: str,
    refs: Sequence[str],
) -> str:
    query_lower = re.sub(r"\s+", " ", (query or "").strip()).lower()
    base_query = query or ""
    for other_ref in refs:
        other_clean = str(other_ref).strip()
        if not other_clean or other_clean.casefold() == ref.casefold():
            continue
        base_query = re.sub(re.escape(other_clean), " ", base_query, flags=re.IGNORECASE)
    base_query = re.sub(r"\s+", " ", base_query).strip()

    if _is_common_elements_query(query):
        if _is_interpretation_sections_common_elements_query(query):
            return (
                f"{ref} schedule 1 interpretation rules of interpretation "
                "a statutory provision includes a reference reference to a person includes"
            )
        return f"{ref} schedule 1 interpretative provisions defined terms"
    if _is_account_effective_dates_query(query):
        return (
            f"{ref} pre-existing accounts new accounts effective date enactment notice "
            "hereby enact enacted on date specified in the enactment notice"
        )
    if _is_restriction_effectiveness_query(query):
        return (
            f"{ref} article 23 restriction on transfer security actual knowledge "
            "ineffective against any person other than a person who had actual knowledge "
            "uncertificated security registered owner notified of the restriction"
        )
    if "same year" in query_lower and "enact" in query_lower:
        return f"{ref} title law no year enacted enactment"
    if _is_named_commencement_query(query):
        return f"{ref} commencement effective date enactment notice come into force"
    if "penalt" in query_lower:
        return f"{ref} penalty offences illegal penalties appendix penalty for offences"
    if "administ" in query_lower:
        if "registrar" in query_lower:
            return (
                f"{ref} may be cited as administration administered by the registrar "
                "this law is administered by this law shall be administered by "
                "shall administer this law administration of this law"
            )
        return (
            f"{ref} may be cited as administration administered by "
            "this law is administered by this law shall be administered by "
            "shall administer this law administration of this law"
        )
    if _is_citation_title_query(query):
        return f'{ref} citation title may be cited as "'
    if "updated" in query_lower:
        return f"{ref} updated amended effective from"
    return f"{ref} {base_query}".strip() if base_query else ref
def should_apply_doc_shortlist_gating(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    answer_type: str,
    doc_refs: Sequence[str],
) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not q or _is_broad_enumeration_query(query):
        return False
    refs = [ref for ref in doc_refs if str(ref).strip()] or pipeline.support_question_refs(query)
    if not refs:
        return False
    if answer_type in {"boolean", "number", "date", "name", "names"} and refs:
        return any(
            term in q
            for term in (
                "title",
                "full title",
                "law number",
                "updated",
                "citation title",
                "commencement",
                "effective date",
                "enact",
                "administ",
            )
        )
    return any(
        term in q
        for term in (
            "title of",
            "titles of",
            "last updated",
            "citation title",
            "citation titles",
            "commencement",
            "effective date",
            "enact",
            "administ",
            "amend",
        )
    )
def page_num(section_path: str | None) -> int:
    match = re.search(r"page:(\d+)", section_path or "", flags=re.IGNORECASE)
    if match is None:
        return 10_000
    try:
        return int(match.group(1))
    except ValueError:
        return 10_000
def build_chunk_snippet(chunk: RetrievedChunk | RankedChunk, *, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = f"{text[: max_chars - 3].rstrip()}..."
    section_path = str(getattr(chunk, "section_path", "") or "").strip()
    if section_path:
        return f"{section_path} | {text}"
    return text
def build_chunk_snippet_map(pipeline: RAGPipelineBuilder, chunks: Sequence[RetrievedChunk | RankedChunk]) -> dict[str, str]:
    snippets: dict[str, str] = {}
    for chunk in chunks:
        chunk_id = str(getattr(chunk, "chunk_id", "") or "").strip()
        if not chunk_id or chunk_id in snippets:
            continue
        snippet = pipeline.build_chunk_snippet(chunk)
        if snippet:
            snippets[chunk_id] = snippet
    return snippets
def build_chunk_page_hint_map(pipeline: RAGPipelineBuilder, chunks: Sequence[RetrievedChunk | RankedChunk]) -> dict[str, str]:
    page_hints: dict[str, str] = {}
    for chunk in chunks:
        chunk_id = str(getattr(chunk, "chunk_id", "") or "").strip()
        doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
        page_num = pipeline.page_num(str(getattr(chunk, "section_path", "") or ""))
        if not chunk_id or not doc_id or page_num <= 0 or chunk_id in page_hints:
            continue
        page_hints[chunk_id] = f"{doc_id}_{page_num}"
    return page_hints
def page_text_looks_like_continuation_tail(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if len(normalized) < 48:
        return False
    if normalized.endswith(("...", "…")):
        return True
    last = normalized[-1]
    if last in {",", ";", ":", "-"}:
        return True
    if last in {".", "!", "?", '"', "'", "]", "}"}:
        return False
    return last.isalnum() or last == ")"
def page_text_looks_like_continuation_head(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if len(normalized) < 24:
        return False
    if normalized[:1].islower():
        return True
    lowered = normalized.casefold()
    return lowered.startswith(
        (
            "and ",
            "or ",
            "but ",
            "if ",
            "unless ",
            "provided ",
            "where ",
            "which ",
            "that ",
            "including ",
            "in addition ",
            "continued ",
            "continuation ",
        )
    )
def page_text_looks_like_new_section(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return False
    prefix = normalized[:96]
    if re.match(r"^(?:article|section|schedule|part|chapter)\b", prefix, re.IGNORECASE):
        return True
    return bool(re.match(r"^[A-Z0-9\s'\"()/-]{10,}$", prefix) and len(prefix.split()) <= 12)
def expand_page_spanning_support_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    chunk_ids: Sequence[str],
    context_chunks: Sequence[RankedChunk],
) -> list[str]:
    ordered_ids = list(dict.fromkeys(str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id).strip()))
    if not ordered_ids or not context_chunks:
        return ordered_ids

    context_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
    chunks_by_doc_page: dict[str, dict[int, list[RankedChunk]]] = {}
    for chunk in context_chunks:
        doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
        if not doc_id:
            continue
        page_num = pipeline.page_num(str(getattr(chunk, "section_path", "") or ""))
        if page_num == 10_000:
            continue
        chunks_by_doc_page.setdefault(doc_id, {}).setdefault(page_num, []).append(chunk)

    expanded: list[str] = []
    seen: set[str] = set()

    def _append(chunk_id: str) -> None:
        normalized = str(chunk_id).strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        expanded.append(normalized)

    for chunk_id in ordered_ids:
        _append(chunk_id)
        chunk = context_by_id.get(chunk_id)
        if chunk is None:
            continue

        doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
        page_num = pipeline.page_num(str(getattr(chunk, "section_path", "") or ""))
        if not doc_id or page_num == 10_000:
            continue

        current_text = str(getattr(chunk, "text", "") or "")
        doc_pages = chunks_by_doc_page.get(doc_id, {})
        previous_page_chunks = doc_pages.get(page_num - 1, [])
        next_page_chunks = doc_pages.get(page_num + 1, [])

        if previous_page_chunks:
            previous_chunk = previous_page_chunks[0]
            previous_text = str(getattr(previous_chunk, "text", "") or "")
            if (
                pipeline.page_text_looks_like_continuation_tail(previous_text)
                or (
                    pipeline.page_text_looks_like_continuation_head(current_text)
                    and not pipeline.page_text_looks_like_new_section(current_text)
                )
            ):
                _append(previous_chunk.chunk_id)

        if next_page_chunks:
            next_chunk = next_page_chunks[0]
            next_text = str(getattr(next_chunk, "text", "") or "")
            if (
                pipeline.page_text_looks_like_continuation_tail(current_text)
                or (
                    pipeline.page_text_looks_like_continuation_head(next_text)
                    and not pipeline.page_text_looks_like_new_section(next_text)
                )
            ):
                _append(next_chunk.chunk_id)

    return expanded
