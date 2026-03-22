"""Offline LLM document-interrogation schemas and prompt helpers."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


def _empty_page_inputs() -> list[DocumentInterrogationPageInput]:
    return []


def _empty_str_list() -> list[str]:
    return []


def _empty_page_signals() -> list[InterrogatedPageSignal]:
    return []


_LEADING_LABEL_RE = re.compile(
    r"^(?:the\s+)?(?:document\s+title|law\s+title|title|issuing\s+authority|authority|"
    r"amendment\s+relationship|relationship)\s*[:\-]\s*",
    re.IGNORECASE,
)
_RELATIONSHIP_HINT_RE = re.compile(
    r"^(?P<relation>amends|amended by|as amended by|amendment to|repeals|repealed by|"
    r"replaces|replaced by|supersedes|superseded by|modifies|modified by|updates|updated by|"
    r"re-enacts|re-enacted by|consolidated with amendments up to)"
    r"(?:\s+the)?\s*(?P<target>.+)$",
    re.IGNORECASE,
)
_RELATIONSHIP_RELATIONS = {
    "amends",
    "amended by",
    "as amended by",
    "amendment to",
    "repeals",
    "repealed by",
    "replaces",
    "replaced by",
    "supersedes",
    "superseded by",
    "modifies",
    "modified by",
    "updates",
    "updated by",
    "re-enacts",
    "re-enacted by",
    "consolidated with amendments up to",
}
_RELATIONSHIP_CANONICAL_FORMS = {
    "as amended by": "amended by",
}


class DocumentInterrogationPageInput(BaseModel):
    """Small page/window input passed into offline interrogation.

    Args:
        page_num: 1-based page number.
        text: Compact page text or top-window text.
    """

    page_num: int
    text: str


class DocumentInterrogationInput(BaseModel):
    """Minimal document input for offline interrogation.

    Args:
        doc_id: Stable document identifier.
        doc_title: Human-readable document title.
        pages: Small ordered page/window inputs.
    """

    doc_id: str
    doc_title: str
    pages: list[DocumentInterrogationPageInput] = Field(default_factory=_empty_page_inputs)


class InterrogatedPageSignal(BaseModel):
    """Structured page-level semantic signal produced by offline interrogation.

    Args:
        page_num: 1-based page number.
        heading_summary: Compact heading-like summary for the page.
        page_template_family: Deterministic-like page family predicted by the LLM.
        field_labels_present: Canonical field labels present on the page.
        primary_evidence: Whether the page is primary evidence rather than secondary reference.
    """

    page_num: int
    heading_summary: str = ""
    page_template_family: str = ""
    field_labels_present: list[str] = Field(default_factory=_empty_str_list)
    primary_evidence: bool = False


class DocumentInterrogationRecord(BaseModel):
    """Strict offline enrichment record for one document.

    Args:
        doc_id: Stable document identifier.
        doc_title: Human-readable document title.
        document_type: Predicted document type.
        issuing_authority: Canonical issuing authority when present.
        law_title: Canonical law title when present.
        law_number: Law number or notice number when present.
        law_year: Canonical law year when present.
        key_parties: Key parties/caption entities.
        amendment_relationships: Parent/base/amendment relationships.
        canonical_law_title: Clean law-like title intended for downstream lookup.
        canonical_issuing_authority: Normalized issuing authority name.
        normalized_amendment_relationships: Canonical amendment relationship hints.
        amendment_targets: Canonical amendment target titles extracted from relationship hints.
        authoritative_sections: Sections/pages likely to hold authoritative evidence.
        likely_answer_page_families: Likely page families for common fact questions.
        page_signals: Page-level semantic outputs.
        compact_shadow_text: Compact retrieval/search text derived from structured outputs.
    """

    doc_id: str
    doc_title: str
    document_type: str = ""
    issuing_authority: str = ""
    law_title: str = ""
    law_number: str = ""
    law_year: str = ""
    key_parties: list[str] = Field(default_factory=_empty_str_list)
    amendment_relationships: list[str] = Field(default_factory=_empty_str_list)
    canonical_law_title: str = ""
    canonical_issuing_authority: str = ""
    normalized_amendment_relationships: list[str] = Field(default_factory=_empty_str_list)
    amendment_targets: list[str] = Field(default_factory=_empty_str_list)
    authoritative_sections: list[str] = Field(default_factory=_empty_str_list)
    likely_answer_page_families: list[str] = Field(default_factory=_empty_str_list)
    page_signals: list[InterrogatedPageSignal] = Field(default_factory=_empty_page_signals)
    compact_shadow_text: str = ""


def build_document_interrogation_system_prompt() -> str:
    """Build the stable system prompt for offline legal document interrogation.

    Returns:
        str: Strict system prompt requesting JSON-only structured output.
    """

    return (
        "You are enriching legal and regulatory documents for offline indexing.\n"
        "Return JSON only. Do not explain.\n"
        "Use conservative extraction. Prefer empty strings/lists over guessing.\n"
        "Classify pages into compact families such as title_cover, caption_header, "
        "issued_by_authority, article_body, schedule_table, operative_order, "
        "appendix_reference, duplicate_or_reference_like.\n"
        "Normalize law titles and authorities conservatively when they are explicit.\n"
        "Extract amendment relationships as canonical hints such as 'amends X', 'replaced by X', "
        "'repeals Y', or 'consolidated with amendments up to Z'.\n"
        "Mark primary_evidence true only when the page itself is a likely source page."
    )


def build_document_interrogation_user_prompt(doc: DocumentInterrogationInput) -> str:
    """Build the user prompt for one document interrogation request.

    Args:
        doc: Minimal document payload to interrogate.

    Returns:
        str: JSON-oriented user prompt with embedded schema and page snippets.
    """

    page_lines: list[dict[str, object]] = [
        {
            "page_num": page.page_num,
            "text": page.text,
        }
        for page in doc.pages
    ]
    schema: dict[str, object] = {
        "doc_id": doc.doc_id,
        "doc_title": doc.doc_title,
        "document_type": "",
        "issuing_authority": "",
        "law_title": "",
        "law_number": "",
        "law_year": "",
        "key_parties": [],
        "amendment_relationships": [],
        "canonical_law_title": "",
        "canonical_issuing_authority": "",
        "normalized_amendment_relationships": [],
        "amendment_targets": [],
        "authoritative_sections": [],
        "likely_answer_page_families": [],
        "page_signals": [
            {
                "page_num": 1,
                "heading_summary": "",
                "page_template_family": "",
                "field_labels_present": [],
                "primary_evidence": False,
            }
        ],
    }
    return (
        "Produce one JSON object matching this schema exactly:\n"
        f"{json.dumps(schema, ensure_ascii=True, indent=2)}\n\n"
        "Document payload:\n"
        f"{json.dumps({'doc_id': doc.doc_id, 'doc_title': doc.doc_title, 'pages': page_lines}, ensure_ascii=True, indent=2)}"
    )


def build_compact_shadow_text(record: DocumentInterrogationRecord) -> str:
    """Build compact search text from structured interrogation output.

    Args:
        record: Parsed interrogation record.

    Returns:
        str: Compact text suitable for storage in offline metadata/search surfaces.
    """

    parts = [
        record.doc_title,
        record.canonical_law_title,
        record.document_type,
        record.issuing_authority,
        record.canonical_issuing_authority,
        record.law_title,
        record.law_number,
        record.law_year,
        *record.key_parties,
        *record.normalized_amendment_relationships,
        *record.amendment_targets,
        *record.authoritative_sections,
        *record.likely_answer_page_families,
    ]
    for page in record.page_signals[:8]:
        parts.extend([page.heading_summary, page.page_template_family, *page.field_labels_present])
    compact = " | ".join(part.strip() for part in parts if part and part.strip())
    return " ".join(compact.split())


def parse_document_interrogation_json(raw_json: str) -> DocumentInterrogationRecord:
    """Parse and normalize LLM JSON into a stable interrogation record.

    Args:
        raw_json: Raw JSON string returned by the offline LLM call.

    Returns:
        DocumentInterrogationRecord: Validated normalized record.
    """

    record = DocumentInterrogationRecord.model_validate_json(raw_json)
    normalized_record = record.model_copy(
        update={
            "canonical_law_title": _normalize_title_like_text(record.law_title or record.doc_title),
            "canonical_issuing_authority": _normalize_authority_like_text(
                record.issuing_authority,
            ),
            "normalized_amendment_relationships": _normalize_amendment_relationships(
                record.amendment_relationships,
            ),
            "amendment_targets": _extract_amendment_targets(record.amendment_relationships),
        }
    )
    return normalized_record.model_copy(update={"compact_shadow_text": build_compact_shadow_text(normalized_record)})


def _normalize_title_like_text(text: str) -> str:
    """Normalize a law-like title conservatively for downstream indexing.

    Args:
        text: Raw title-like value.

    Returns:
        str: Cleaned title-like text.
    """

    cleaned = " ".join((text or "").split()).strip()
    cleaned = _LEADING_LABEL_RE.sub("", cleaned)
    cleaned = re.sub(r"(?i)^the\s+", "", cleaned)
    return cleaned.rstrip(" .;,")


def _normalize_authority_like_text(text: str) -> str:
    """Normalize a legal authority name conservatively.

    Args:
        text: Raw authority-like value.

    Returns:
        str: Cleaned authority text.
    """

    cleaned = " ".join((text or "").split()).strip()
    cleaned = _LEADING_LABEL_RE.sub("", cleaned)
    cleaned = re.sub(r"(?i)^issued\s+by\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^administered\s+by\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^made\s+by\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^promulgated\s+by\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^approved\s+by\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^signed\s+by\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^the\s+", "", cleaned)
    return cleaned.rstrip(" .;,")


def _normalize_amendment_relationships(relationships: list[str]) -> list[str]:
    """Normalize amendment relationships into canonical lookup hints.

    Args:
        relationships: Raw relationship strings from interrogation output.

    Returns:
        list[str]: Normalized relationship hints.
    """

    normalized: list[str] = []
    for relationship in relationships:
        cleaned = _normalize_title_like_text(relationship)
        if not cleaned:
            continue
        hint = _extract_relationship_hint(cleaned)
        if hint is None:
            normalized.append(cleaned)
            continue
        relation, target = hint
        normalized.append(f"{relation} {target}".strip())
    return _unique_preserving_order(normalized)


def _extract_amendment_targets(relationships: list[str]) -> list[str]:
    """Extract amendment targets from raw relationship strings.

    Args:
        relationships: Raw relationship strings from interrogation output.

    Returns:
        list[str]: Canonical target titles useful for downstream joins.
    """

    targets: list[str] = []
    for relationship in relationships:
        cleaned = _normalize_title_like_text(relationship)
        if not cleaned:
            continue
        hint = _extract_relationship_hint(cleaned)
        if hint is None:
            continue
        _, target = hint
        targets.append(target)
    return _unique_preserving_order(targets)


def _extract_relationship_hint(text: str) -> tuple[str, str] | None:
    """Split a relationship string into canonical relation and target text.

    Args:
        text: Relationship string.

    Returns:
        tuple[str, str] | None: Canonical relation and target title when recognized.
    """

    match = _RELATIONSHIP_HINT_RE.match(text)
    if match is None:
        return None
    relation = " ".join((match.group("relation") or "").split()).casefold()
    if relation not in _RELATIONSHIP_RELATIONS:
        return None
    relation = _RELATIONSHIP_CANONICAL_FORMS.get(relation, relation)
    target = _normalize_title_like_text(match.group("target") or "")
    if not target:
        return None
    return relation, target


def _unique_preserving_order(values: list[str]) -> list[str]:
    """Deduplicate a list while preserving first-seen order.

    Args:
        values: Input sequence.

    Returns:
        list[str]: Deduplicated values in order.
    """

    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def load_document_interrogation_inputs(path: Path) -> list[DocumentInterrogationInput]:
    """Load JSON or JSONL interrogation inputs from disk.

    Args:
        path: Path to JSON or JSONL input file.

    Returns:
        list[DocumentInterrogationInput]: Parsed document inputs.
    """

    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [DocumentInterrogationInput.model_validate_json(line) for line in raw_text.splitlines() if line.strip()]

    data = json.loads(raw_text)
    if isinstance(data, list):
        return [DocumentInterrogationInput.model_validate(item) for item in cast("list[object]", data)]
    return [DocumentInterrogationInput.model_validate(data)]
