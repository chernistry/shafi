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
    _TEXTUAL_DATE_MONTH_FIRST_RE,
    _TEXTUAL_DATE_RE,
    _UNANSWERABLE_FREE_TEXT,
    _UNANSWERABLE_STRICT,
)

_NUMBERED_LEGAL_TITLE_RE = re.compile(
    r"([A-Z][^\n]{0,180}?\b(?:DIFC\s+)?(?:Law|Regulations|Rules|Order|Notice|Practice Direction)\s+No\.?\s*\d+\s+of\s+\d{4})",
)
_DIFC_CASE_REF_RE = re.compile(
    r"\b(CFI|CA|ARB|SCT|TCD|ENF|DEC)\s*[/\-]?\s*(\d{1,3})\s*[/\-]\s*(\d{4})\b",
    re.IGNORECASE,
)


def _normalize_numbered_legal_title(candidate: str) -> str:
    """Normalize UAE/DIFC numbered legal titles for strict outputs."""

    normalized = re.sub(r"\s+", " ", candidate.strip())
    normalized = re.sub(r"\bNo\b\.?\s*", "No. ", normalized)
    return normalized.rstrip(" .;")


def _normalize_difc_case_refs(text: str) -> str:
    """Normalize DIFC-style case references inside strict multi-name answers."""

    def _replace(match: re.Match[str]) -> str:
        prefix = match.group(1).upper()
        number = int(match.group(2))
        year = match.group(3)
        return f"{prefix} {number:03d}/{year}"

    return _DIFC_CASE_REF_RE.sub(_replace, text)


def raw_ranked(chunks: list[RetrievedChunk], *, top_n: int) -> list[RankedChunk]:
    if not chunks:
        return []
    sorted_chunks = sorted(chunks, key=lambda chunk: (-chunk.score, chunk.doc_id, chunk.chunk_id))
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
            normalized_refs=list(getattr(chunk, "normalized_refs", []) or []),
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
    if normalized.startswith("there is no information on this question"):
        return True
    if "insufficient sources retrieved" in normalized:
        return True
    # LLM sometimes deviates from the exact sentinel and produces
    # "There is no information on X in the provided sources/documents."
    # Catch these so they get empty refs in the submission.
    _in_sources = "provided sources" in normalized or "provided documents" in normalized
    if _in_sources and (
        normalized.startswith("there is no information on ")
        or normalized.startswith("there is no information about ")
        or normalized.startswith("there is no information in the provided")
    ):
        return True
    return False
def strict_type_fallback(pipeline: RAGPipelineBuilder, answer_type: str, cited_ids: list[str] | tuple[str, ...]) -> str:
    kind = answer_type.strip().lower()
    if kind in {"boolean", "number", "date", "name", "names"}:
        return _UNANSWERABLE_STRICT
    return pipeline.insufficient_sources_answer(cited_ids)
def insufficient_sources_answer(pipeline: RAGPipelineBuilder, cited_ids: list[str] | tuple[str, ...]) -> str:
    _ = cited_ids
    return _UNANSWERABLE_FREE_TEXT
_TIME_KEYWORDS_RE = re.compile(
    r"\b(?:year|years|period|term|duration|months?|days?|weeks?|time)\b",
    re.IGNORECASE,
)
_MONEY_KEYWORDS_RE = re.compile(
    r"\b(?:amount|fine|penalty|damages?|dirhams?|aed|usd|value|cost|fee|price|compensation|claim\s+value|claim\s+amount)\b",
    re.IGNORECASE,
)


def coerce_strict_type_format(
    pipeline: RAGPipelineBuilder,
    answer: str,
    answer_type: str,
    cited_ids: list[str] | tuple[str, ...],
    question: str = "",
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
        # Priority 1: answer starts with yes/no (clearest signal)
        if lowered.startswith("yes"):
            return (f"Yes{suffix}".strip(), True)
        if re.match(r"no(?:\b|$)", lowered):
            return (f"No{suffix}".strip(), True)
        # Priority 2: yes/no after a clear delimiter (colon, period, dash, newline)
        after_delim = re.search(r"(?:[:.\-\n])\s*(yes|no)\b", lowered)
        if after_delim:
            return (f"{'Yes' if after_delim.group(1) == 'yes' else 'No'}{suffix}".strip(), True)
        # Priority 3: explicit boolean synonyms anywhere
        if re.search(r"\b(?:true|correct|affirmative)\b", lowered):
            return (f"Yes{suffix}".strip(), True)
        if re.search(r"\b(?:false|incorrect|negative)\b", lowered):
            return (f"No{suffix}".strip(), True)
        # Priority 4: standalone yes/no as a word boundary (not substring)
        if re.search(r"\byes\b", lowered) and not re.search(r"\bno\b", lowered):
            return (f"Yes{suffix}".strip(), True)
        if re.search(r"\bno\b", lowered) and not re.search(r"\byes\b", lowered):
            return (f"No{suffix}".strip(), True)
        return (pipeline.strict_type_fallback(kind, cited_ids), False)

    if kind == "number":
        q_wants_time = bool(_TIME_KEYWORDS_RE.search(question)) if question else False
        q_wants_money = bool(_MONEY_KEYWORDS_RE.search(question)) if question else False
        candidates: list[tuple[str, str]] = []  # (number_str, context_category)
        for match in _NUMBER_RE.finditer(stripped_text):
            start, end = match.span()
            before = stripped_text[max(0, start - 24) : start]
            after = stripped_text[end : min(len(stripped_text), end + 10)]
            if after.lstrip().startswith("/") and re.match(r"\s*/\s*\d{2,4}", after):
                continue
            if re.search(r"(?:CA|CFI|ARB|SCT|TCD|ENF|DEC)\s*$", before, re.IGNORECASE):
                continue
            # Skip Article/Section/Schedule/etc. numbers — they are structural, not answer values.
            if re.search(
                r"(?:Article|Section|Schedule|Regulation|Rule|Part|Chapter|Clause)\s*$",
                before,
                re.IGNORECASE,
            ):
                continue
            # Categorize by surrounding context
            context = (before + " " + after).lower()
            if _TIME_KEYWORDS_RE.search(context):
                cat = "time"
            elif _MONEY_KEYWORDS_RE.search(context):
                cat = "money"
            else:
                cat = "unknown"
            candidates.append((match.group(0), cat))

        if not candidates:
            return (pipeline.strict_type_fallback(kind, cited_ids), False)

        # If question specifies a category, prefer matching candidates
        if (q_wants_time or q_wants_money) and len(candidates) > 1:
            preferred_cat = "time" if q_wants_time else "money"
            preferred = [c for c in candidates if c[1] == preferred_cat]
            if preferred:
                return (f"{preferred[0][0]}{suffix}".strip(), True)

        # Default: return first valid candidate (original behavior)
        return (f"{candidates[0][0]}{suffix}".strip(), True)

    if kind == "date":
        match = (
            _ISO_DATE_RE.search(stripped_text)
            or _SLASH_DATE_RE.search(stripped_text)
            or _TEXTUAL_DATE_RE.search(stripped_text)
            or _TEXTUAL_DATE_MONTH_FIRST_RE.search(stripped_text)
        )
        if match is None:
            return (pipeline.strict_type_fallback(kind, cited_ids), False)
        return (f"{match.group(0)}{suffix}".strip(), True)

    if kind == "name":
        search_text = re.sub(
            r"^(?:this\s+law\s+may\s+be\s+cited\s+as\s+(?:the\s+)?|the\s+applicable\s+instrument\s+is\s+|case\s+reference\s*:\s*)",
            "",
            stripped_text,
            flags=re.IGNORECASE,
        ).strip()
        # If the model included a DIFC case ID, prefer returning just that normalized ID.
        case_match = _DIFC_CASE_REF_RE.search(search_text) or _DIFC_CASE_ID_RE.search(search_text)
        if case_match is not None:
            prefix = case_match.group(1).upper()
            num = int(case_match.group(2))
            year = case_match.group(3)
            return (f"{prefix} {num:03d}/{year}{suffix}".strip(), True)

        # Prefer full DIFC law titles that include the law number, e.g. "Strata Title Law, DIFC Law No. 5 of 2007".
        law_title_match = _NUMBERED_LEGAL_TITLE_RE.search(search_text)
        if law_title_match is not None and law_title_match.group(1).strip():
            candidate = _normalize_numbered_legal_title(law_title_match.group(1))
            return (f"{candidate}{suffix}".strip(), True)

        stripped = search_text
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
        # Before truncating at prepositions, check if the entire phrase is a
        # recognized legal title pattern (e.g., "Trust Law of the Emirate of Dubai").
        # If it matches a title pattern, skip aggressive truncation.
        _is_full_legal_title = bool(re.match(
            r"^[A-Z][A-Za-z\s]+(?:Law|Regulations?|Rules?|Order|Act|Code|Statute|Decree)\s+of\s+",
            stripped,
        ))
        lowered = stripped.lower()
        if not _is_full_legal_title:
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
        # But skip comma splitting if the text looks like "X Law of Y, DIFC Law No. Z" (legal title).
        for sep in (" — ", " - ", ";", ":"):
            if sep in stripped:
                stripped = stripped.split(sep, 1)[0].strip()
        # Only split on comma if result doesn't look like a truncated title
        if "," in stripped:
            before_comma = stripped.split(",", 1)[0].strip()
            after_comma = stripped.split(",", 1)[1].strip()
            # Keep comma if after-comma part is a law number reference
            if not re.match(r"(?:DIFC\s+)?(?:Law|Regulations?|Rules?)\s+No", after_comma, re.IGNORECASE):
                stripped = before_comma
        words = stripped.split()
        if len(words) > 20:
            stripped = " ".join(words[:20]).strip()
        # Strip leading indefinite articles: "a Confirmation Statement" → "Confirmation Statement".
        # EQA or LLM may prefix answers with "a/an" which evaluators reject.
        # Do NOT strip "the" — proper names may include it (e.g. "the Owner", "the Claimant").
        stripped = re.sub(r"^(?:a|an)\s+", "", stripped, flags=re.IGNORECASE).strip()
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
        # Normalize all DIFC case references (e.g. "CFI/7/2024" → "CFI 007/2024").
        def _normalize_case_ref(m: re.Match[str]) -> str:
            prefix = m.group(1).upper()
            num = int(m.group(2))
            year = m.group(3)
            return f"{prefix} {num:03d}/{year}"

        stripped = _DIFC_CASE_ID_RE.sub(_normalize_case_ref, stripped)
        if not stripped:
            return (pipeline.strict_type_fallback(kind, cited_ids), False)
        stripped = _normalize_difc_case_refs(stripped)
        return (f"{stripped}{suffix}".strip(), True)

    return (stripped_text, True)
