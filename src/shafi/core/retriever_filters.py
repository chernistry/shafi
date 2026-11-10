"""Pure helper functions for retriever query shaping and Qdrant filters."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

from qdrant_client import models

from shafi.core.classifier import QueryClassifier

if TYPE_CHECKING:
    from shafi.models import DocType

_DIFC_CASE_RE = re.compile(r"^(CFI|CA|SCT|ENF|DEC|TCD|ARB)\s+0*(\d{1,4})[/\-](\d{4})$", re.IGNORECASE)
_LAW_NO_RE = re.compile(r"^Law\s+No\.?\s*(\d+)\s+of\s+(\d{4})$", re.IGNORECASE)
_NUMBERED_INSTRUMENT_RE = re.compile(
    r"^(?:(?P<jurisdiction>DIFC|DFSA|UAE)\s+)?(?P<family>Law|Regulations?|Rules?)\s+No\.?\s*(?P<number>\d+)\s+of\s+(?P<year>\d{4})$",
    re.IGNORECASE,
)
_TITLE_WITH_YEAR_RE = re.compile(r"^(?P<title>.+?)\s+(?P<year>19\d{2}|20\d{2})$", re.IGNORECASE)


def build_sparse_query(*, query: str, extracted_refs: list[str] | tuple[str, ...]) -> str:
    """Build the sparse retrieval query with legal-ref boosting when safe.

    Args:
        query: Raw user question text.
        extracted_refs: Explicit document references already extracted from the query.

    Returns:
        Sparse query text, optionally boosted with repeated exact legal references.
    """
    base_query = str(query or "").strip()
    if not base_query:
        return ""
    if any(_DIFC_CASE_RE.match(str(ref).strip()) is not None for ref in extracted_refs):
        return base_query

    exact_refs = QueryClassifier.extract_exact_legal_refs(base_query)
    if not exact_refs:
        return base_query

    capped_refs = exact_refs[:4]
    boosted_tail = " ".join([*capped_refs, *capped_refs]).strip()
    if not boosted_tail:
        return base_query
    return f"{base_query}\n{boosted_tail}".strip()


def coerce_int(value: object) -> int | None:
    """Coerce mixed payload values into integers.

    Args:
        value: Arbitrary payload value from Qdrant metadata.

    Returns:
        Parsed integer when conversion is safe, otherwise ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def coerce_float(value: object) -> float | None:
    """Coerce mixed payload values into floats.

    Args:
        value: Arbitrary payload value from Qdrant metadata.

    Returns:
        Parsed float when conversion is safe, otherwise ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def coerce_str_list(value: object) -> list[str]:
    """Coerce payload arrays into clean string lists.

    Args:
        value: Arbitrary payload value from Qdrant metadata.

    Returns:
        Non-empty stripped strings when the payload is list-like, otherwise ``[]``.
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in cast("list[object]", value):
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def build_filter(
    *,
    doc_type_filter: DocType | None,
    jurisdiction_filter: str | None,
    doc_refs: list[str] | tuple[str, ...] | None = None,
    chunk_types: list[str] | tuple[str, ...] | None = None,
    must_not_doc_ids: list[str] | tuple[str, ...] | None = None,
) -> models.Filter | None:
    """Build the Qdrant filter used by retriever prefetch/search calls.

    Args:
        doc_type_filter: Optional document type constraint.
        jurisdiction_filter: Optional jurisdiction constraint.
        doc_refs: Optional explicit document references.
        chunk_types: Optional chunk-type restriction for anchor retrieval.
        must_not_doc_ids: Optional document IDs to exclude (for diversity expansion).

    Returns:
        Qdrant filter object when any restriction is active, otherwise ``None``.
    """
    conditions: list[models.Condition] = []
    if doc_type_filter is not None:
        conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=doc_type_filter.value),
            )
        )
    if jurisdiction_filter:
        conditions.append(
            models.FieldCondition(
                key="jurisdiction",
                match=models.MatchValue(value=jurisdiction_filter),
            )
        )
    refs = [ref.strip() for ref in (list(doc_refs) if doc_refs is not None else []) if str(ref).strip()]
    chunk_type_values = [
        chunk_type.strip()
        for chunk_type in (list(chunk_types) if chunk_types is not None else [])
        if str(chunk_type).strip()
    ]
    if chunk_type_values:
        conditions.append(
            models.FieldCondition(
                key="chunk_type",
                match=models.MatchAny(any=chunk_type_values),
            )
        )
    title_refs = doc_title_filter_variants(refs)
    must_not: list[models.Condition] = []
    exclude_ids = [
        did.strip() for did in (list(must_not_doc_ids) if must_not_doc_ids is not None else []) if str(did).strip()
    ]
    if exclude_ids:
        must_not.append(
            models.FieldCondition(
                key="doc_id",
                match=models.MatchAny(any=exclude_ids),
            )
        )
    if refs:
        ref_conditions: list[models.Condition] = [
            models.FieldCondition(
                key="citations",
                match=models.MatchAny(any=refs),
            )
        ]
        if title_refs:
            ref_conditions.append(
                models.FieldCondition(
                    key="doc_title",
                    match=models.MatchAny(any=title_refs),
                )
            )
        # For case refs, also match doc_title via full-text to catch chunks that
        # don't carry the case ref in their citations field (e.g. cover-page chunks).
        # MatchText uses Qdrant's word tokenizer where "/" IS a word boundary, so
        # "CFI 057/2025" tokenises to ["CFI","057","2025"] and correctly matches both
        # "CFI 057/2025 ..." (slash-year) and "[2025] DIFC CFI 057 ..." (bracket-year)
        # titles — both contain the same three tokens.  Using only the canonical form
        # avoids year-collision false-positives (e.g. "TCD 001" matching TCD 001/2024
        # when querying for TCD 001/2023).
        case_text_seen: set[str] = set()
        for ref in refs:
            m = _DIFC_CASE_RE.match(ref)
            if m:
                canonical = f"{m.group(1).upper()} {int(m.group(2)):03d}/{m.group(3)}"
                if canonical not in case_text_seen:
                    case_text_seen.add(canonical)
                    ref_conditions.append(
                        models.FieldCondition(
                            key="doc_title",
                            match=models.MatchText(text=canonical),
                        )
                    )
                    # NOTE: Do NOT add a short form (e.g. "SCT 415") — it matches all
                    # "SCT 415/YYYY" variants and causes BM25 to rank the wrong document.
                    # The canonical "SCT 415/2024" already covers bracket-year titles because
                    # "/" is a word boundary so the year token is always present.
        # Direct metadata filter on case_numbers payload field.  This catches chunks
        # whose case_numbers list contains the canonical form even when the citation/
        # title fields do not carry it (e.g. interior pages of a judgment).
        case_number_match_values: list[str] = []
        for ref in refs:
            m = _DIFC_CASE_RE.match(ref)
            if m:
                prefix = m.group(1).upper()
                num_raw = int(m.group(2))
                year = m.group(3)
                case_number_match_values.append(f"{prefix} {num_raw:03d}/{year}")
                # Unpadded variant for bracket-format ingested chunks.
                if num_raw < 100:
                    case_number_match_values.append(f"{prefix} {num_raw}/{year}")
        if case_number_match_values:
            ref_conditions.append(
                models.FieldCondition(
                    key="case_numbers",
                    match=models.MatchAny(any=sorted(set(case_number_match_values))),
                )
            )
        return models.Filter(
            must=[*conditions, models.Filter(should=ref_conditions)],
            must_not=must_not or None,
        )
    if conditions or must_not:
        return models.Filter(must=conditions or None, must_not=must_not or None)
    return None


def expand_doc_ref_variants(refs: list[str] | tuple[str, ...]) -> list[str]:
    """Expand explicit doc refs into citation/title match variants.

    Args:
        refs: Raw extracted document references.

    Returns:
        Deduplicated reference variants for citation matching.
    """
    variants: list[str] = []
    for raw in refs:
        ref = raw.strip()
        if not ref:
            continue
        variants.append(ref)

        case_match = _DIFC_CASE_RE.match(ref)
        if case_match is not None:
            prefix = case_match.group(1).upper()
            num_raw = int(case_match.group(2))
            year = case_match.group(3)
            variants.append(f"{prefix} {num_raw}/{year}")
            variants.append(f"{prefix} {num_raw:03d}/{year}")
            variants.append(f"{prefix}{num_raw:03d}/{year}")
            variants.append(f"{prefix}{num_raw}/{year}")
            # OCR/PDF variants: dash instead of slash, space around slash
            variants.append(f"{prefix} {num_raw:03d}-{year}")
            variants.append(f"{prefix}-{num_raw:03d}/{year}")
            variants.append(f"Case {prefix} {num_raw:03d}/{year}")

        law_match = _LAW_NO_RE.match(ref)
        if law_match is not None:
            num = int(law_match.group(1))
            year = law_match.group(2)
            variants.append(f"Law No. {num} of {year}")
            variants.append(f"Law No {num} of {year}")
            variants.append(f"DIFC Law No. {num} of {year}")
            variants.append(f"DIFC Law No {num} of {year}")
            # Common corpus forms: "Law No.X of YYYY", zero-padded
            variants.append(f"Law No.{num} of {year}")
            continue

        instrument_match = _NUMBERED_INSTRUMENT_RE.match(ref)
        if instrument_match is not None:
            jurisdiction = (instrument_match.group("jurisdiction") or "").upper().strip()
            family_raw = instrument_match.group("family").strip()
            family = (
                "Law"
                if family_raw.casefold() == "law"
                else "Regulations"
                if family_raw.casefold().startswith("regulation")
                else "Rules"
            )
            num = int(instrument_match.group("number"))
            year = instrument_match.group("year")
            variants.append(f"{family} No. {num} of {year}")
            variants.append(f"{family} No {num} of {year}")
            variants.append(f"{family} No.{num} of {year}")
            if jurisdiction:
                variants.append(f"{jurisdiction} {family} No. {num} of {year}")
                variants.append(f"{jurisdiction} {family} No {num} of {year}")
            if not jurisdiction:
                # Try DIFC prefix since most laws are DIFC
                variants.append(f"DIFC {family} No. {num} of {year}")
                variants.append(f"DIFC {family} No {num} of {year}")
            continue

        title_with_year_match = _TITLE_WITH_YEAR_RE.match(ref)
        if title_with_year_match is not None:
            title_only = re.sub(r"\s+", " ", title_with_year_match.group("title")).strip(" ,.;:")
            if title_only:
                variants.append(title_only)
                variants.append(title_only.upper())

    seen: set[str] = set()
    out: list[str] = []
    for candidate in variants:
        key = candidate.strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        out.append(key)
    return out


def doc_title_filter_variants(refs: list[str] | tuple[str, ...]) -> list[str]:
    """Build doc-title variants suitable for Qdrant ``doc_title`` matching.

    Args:
        refs: Raw extracted document references.

    Returns:
        Deduplicated title-oriented variants, excluding law-number and case refs.
    """
    variants: list[str] = []
    for raw in refs:
        ref = re.sub(r"\s+", " ", str(raw).strip())
        if not ref or _LAW_NO_RE.match(ref) is not None or _DIFC_CASE_RE.match(ref) is not None:
            continue
        variants.append(ref)
        variants.append(ref.upper())  # ALL_CAPS variant — many regulation titles stored uppercase in Qdrant
        title_with_year_match = _TITLE_WITH_YEAR_RE.match(ref)
        if title_with_year_match is not None:
            title_only = re.sub(r"\s+", " ", title_with_year_match.group("title")).strip(" ,.;:")
            if title_only:
                variants.append(title_only)
                variants.append(title_only.upper())

    seen: set[str] = set()
    out: list[str] = []
    for candidate in variants:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out
