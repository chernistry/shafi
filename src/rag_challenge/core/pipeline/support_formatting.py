# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed support helpers for the pipeline hot path."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rag_challenge.models import RankedChunk

if TYPE_CHECKING:
    from rag_challenge.models import RetrievedChunk

    from .builder import RAGPipelineBuilder

from .constants import (
    _CASE_REF_PREFIX_RE,
    _CITE_RE,
    _DIFC_CASE_ID_RE,
    _ISO_DATE_RE,
    _NUMBER_RE,
    _SLASH_DATE_RE,
    _TEXTUAL_DATE_RE,
    _UNANSWERABLE_FREE_TEXT,
    _UNANSWERABLE_STRICT,
)


def raw_ranked(chunks: list[RetrievedChunk], *, top_n: int) -> list[RankedChunk]:
    if not chunks:
        return []
    sorted_chunks = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
    return [
        RankedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            doc_title=chunk.doc_title,
            doc_type=chunk.doc_type,
            section_path=chunk.section_path,
            text=chunk.text,
            retrieval_score=chunk.score,
            rerank_score=chunk.score,
            doc_summary=chunk.doc_summary,
            page_family=getattr(chunk, "page_family", ""),
            doc_family=getattr(chunk, "doc_family", ""),
            chunk_type=getattr(chunk, "chunk_type", ""),
            amount_roles=list(getattr(chunk, "amount_roles", []) or []),
        )
        for chunk in sorted_chunks[: max(0, int(top_n))]
    ]
def citation_suffix(cited_ids: list[str] | tuple[str, ...], *, enabled: bool) -> str:
    if not enabled:
        return ""
    ids = [chunk_id.strip() for chunk_id in cited_ids if str(chunk_id).strip()]
    if not ids:
        return ""
    keep = ids[:3]
    return f" (cite: {', '.join(keep)})"
def strict_type_citation_suffix(pipeline: RAGPipelineBuilder, cited_ids: list[str] | tuple[str, ...]) -> str:
    return pipeline.citation_suffix(
        cited_ids,
        enabled=bool(getattr(pipeline._settings.pipeline, "strict_types_append_citations", False)),
    )
def is_unanswerable_strict_answer(answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    return normalized in {"null", "none", ""}
def is_unanswerable_free_text_answer(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", (answer or "").strip().lower())
    return normalized.startswith("there is no information on this question") or "insufficient sources retrieved" in normalized
def strict_type_fallback(pipeline: RAGPipelineBuilder, answer_type: str, cited_ids: list[str] | tuple[str, ...]) -> str:
    kind = answer_type.strip().lower()
    if kind in {"boolean", "number", "date", "name", "names"}:
        return _UNANSWERABLE_STRICT
    return pipeline.insufficient_sources_answer(cited_ids)
def insufficient_sources_answer(pipeline: RAGPipelineBuilder, cited_ids: list[str] | tuple[str, ...]) -> str:
    _ = cited_ids
    return _UNANSWERABLE_FREE_TEXT
def coerce_strict_type_format(
    pipeline: RAGPipelineBuilder,
    answer: str,
    answer_type: str,
    cited_ids: list[str] | tuple[str, ...],
) -> tuple[str, bool]:
    kind = answer_type.strip().lower()
    text = answer.strip()
    if not text:
        return (pipeline.strict_type_fallback(kind, cited_ids), False)
    normalized = text.lower()
    if (
        "insufficient sources" in normalized
        or "there is no information on this question" in normalized
        or normalized.strip() in {"null", "none"}
    ):
        return (pipeline.strict_type_fallback(kind, cited_ids), False)

    stripped_text = _CITE_RE.sub("", text).strip()
    stripped_text = re.sub(r"\s+", " ", stripped_text).strip()
    suffix = pipeline.strict_type_citation_suffix(cited_ids)

    if kind == "boolean":
        lowered = stripped_text.lower().lstrip()
        if lowered.startswith("yes"):
            return (f"Yes{suffix}".strip(), True)
        if lowered.startswith("no"):
            return (f"No{suffix}".strip(), True)
        if "yes" in lowered and "no" not in lowered:
            return (f"Yes{suffix}".strip(), True)
        if "no" in lowered and "yes" not in lowered:
            return (f"No{suffix}".strip(), True)
        return (pipeline.strict_type_fallback(kind, cited_ids), False)

    if kind == "number":
        for match in _NUMBER_RE.finditer(stripped_text):
            start, end = match.span()
            before = stripped_text[max(0, start - 24) : start]
            after = stripped_text[end : min(len(stripped_text), end + 10)]
            if after.lstrip().startswith("/") and re.match(r"\s*/\s*\d{2,4}", after):
                continue
            if re.search(r"(?:CA|CFI|ARB|SCT|TCD|ENF|DEC)\s*$", before, re.IGNORECASE):
                continue
            return (f"{match.group(0)}{suffix}".strip(), True)
        return (pipeline.strict_type_fallback(kind, cited_ids), False)

    if kind == "date":
        match = _ISO_DATE_RE.search(stripped_text) or _SLASH_DATE_RE.search(stripped_text) or _TEXTUAL_DATE_RE.search(stripped_text)
        if match is None:
            return (pipeline.strict_type_fallback(kind, cited_ids), False)
        return (f"{match.group(0)}{suffix}".strip(), True)

    if kind == "name":
        # If the model included a DIFC case ID, prefer returning just that normalized ID.
        case_match = _DIFC_CASE_ID_RE.search(stripped_text)
        if case_match is not None:
            prefix = case_match.group(1).upper()
            num = int(case_match.group(2))
            year = case_match.group(3)
            return (f"{prefix} {num:03d}/{year}{suffix}".strip(), True)

        # Prefer full DIFC law titles that include the law number, e.g. "Strata Title Law, DIFC Law No. 5 of 2007".
        law_title_match = re.search(
            r"([A-Z][^\n]{0,180}?\b(?:DIFC\s+)?Law\s+No\.?\s*\d+\s+of\s+\d{4})",
            stripped_text,
        )
        if law_title_match is not None and law_title_match.group(1).strip():
            candidate = re.sub(r"\s+", " ", law_title_match.group(1).strip())
            candidate = re.sub(r"\bNo\.\s*", "No ", candidate)
            candidate = candidate.rstrip(" .;")
            return (f"{candidate}{suffix}".strip(), True)

        stripped = stripped_text
        for pattern in (
            r"(?:is|called|known as|referred to as|named)\s+[\"']?([A-Z][^\"'!?\n]{1,80})[\"']?",
            r"term\s+[\"']([^\"']+)[\"']",
        ):
            m = re.search(pattern, stripped, re.IGNORECASE)
            if m and m.group(1).strip():
                stripped = m.group(1).strip()
                break
        # Tighten "name" outputs aggressively: evaluators expect a short entity/title, not a clause.
        stripped = re.sub(r"[.!?]", "", stripped).strip()
        lowered = stripped.lower()
        for marker in (
            " subject to ",
            " provided that ",
            " pursuant to ",
            " in accordance with ",
            " as per ",
            " as provided ",
            " under ",
        ):
            idx = lowered.find(marker)
            if idx != -1:
                stripped = stripped[:idx].strip()
                break
        # Prefer the first phrase if the model returned a longer explanatory fragment.
        for sep in (" — ", " - ", ";", ":", ","):
            if sep in stripped:
                stripped = stripped.split(sep, 1)[0].strip()
        words = stripped.split()
        if len(words) > 12:
            stripped = " ".join(words[:12]).strip()
        if not stripped:
            return (pipeline.strict_type_fallback(kind, cited_ids), False)
        return (f"{stripped}{suffix}".strip(), True)

    if kind == "names":
        stripped = re.sub(
            r"^(?:the\s+)?(?:names?|parties|individuals?)\s+(?:are|is|include[s]?)\s*:?\s*",
            "",
            stripped_text,
            flags=re.IGNORECASE,
        ).strip().rstrip(".")
        stripped = _CASE_REF_PREFIX_RE.sub("", stripped).strip()
        if not stripped:
            return (pipeline.strict_type_fallback(kind, cited_ids), False)
        return (f"{stripped}{suffix}".strip(), True)

    return (stripped_text, True)
