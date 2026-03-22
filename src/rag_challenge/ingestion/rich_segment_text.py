"""Rich and plain segment text composition for offline shadow ablations."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from rag_challenge.ingestion.external_segment_payload import ExternalSegmentRecord


class SegmentTextMode(StrEnum):
    """Supported shadow composition modes."""

    PLAIN = "plain"
    RICH = "rich"


class SegmentNoiseAnalysis(BaseModel):
    """Title/header duplication analysis for one composed segment."""

    title_repeated: bool = False
    hierarchy_repeated: bool = False
    duplicate_line_count: int = 0


_STRUCTURAL_LINE_RE = re.compile(
    r"^(?P<label>article|section|schedule|part|chapter)\s+(?P<number>[A-Za-z0-9().-]+)(?:\s*[-:]\s*(?P<title>.+))?$",
    re.IGNORECASE,
)
_CAPTION_RE = re.compile(r"\b(?:v\.?|vs\.?|versus)\b", re.IGNORECASE)
_ENACTMENT_NOTICE_RE = re.compile(r"\b(?:enactment notice|commencement|comes into force|effective date)\b", re.IGNORECASE)
_ISSUED_BY_RE = re.compile(r"\bissued by\b", re.IGNORECASE)
_DATE_PANEL_RE = re.compile(r"\b(?:date of issue|date of enactment|date)\b", re.IGNORECASE)
_LAW_NUMBER_RE = re.compile(r"\b(?:law|regulations?|rules?)\s+no\.?\s*\d+\s+of\s+\d{4}\b", re.IGNORECASE)


def compose_segment_text(
    segment: ExternalSegmentRecord,
    *,
    mode: SegmentTextMode,
    context_char_limit: int = 240,
) -> str:
    """Compose retrieval text for one segment.

    Args:
        segment: Source segment payload record.
        mode: Requested composition mode.
        context_char_limit: Max context characters used in rich mode.

    Returns:
        Composed retrieval text.
    """
    if mode is SegmentTextMode.PLAIN:
        return _plain_text(segment)
    return _rich_text(segment, context_char_limit=context_char_limit)


def analyze_segment_noise(segment: ExternalSegmentRecord, *, mode: SegmentTextMode) -> SegmentNoiseAnalysis:
    """Measure title/header duplication in composed text.

    Args:
        segment: Source segment payload record.
        mode: Requested composition mode.

    Returns:
        Per-segment duplication analysis.
    """
    composed = compose_segment_text(segment, mode=mode)
    lines = [line.strip() for line in composed.splitlines() if line.strip()]
    normalized_lines = [line.casefold() for line in lines]
    title = segment.title.strip().casefold()
    hierarchy = " > ".join(part.strip() for part in segment.hierarchy if part.strip()).casefold()
    duplicate_line_count = len(normalized_lines) - len(set(normalized_lines))
    return SegmentNoiseAnalysis(
        title_repeated=bool(title) and normalized_lines.count(title) > 1,
        hierarchy_repeated=bool(hierarchy) and normalized_lines.count(hierarchy) > 1,
        duplicate_line_count=max(0, duplicate_line_count),
    )


def _plain_text(segment: ExternalSegmentRecord) -> str:
    return segment.text.strip() or segment.embedding_text.strip() or segment.context_text.strip()


def _rich_text(segment: ExternalSegmentRecord, *, context_char_limit: int) -> str:
    descriptor = segment.metadata.document_descriptor.strip()
    hierarchy = " > ".join(part.strip() for part in segment.hierarchy if part.strip())
    structural_markers = _structural_markers(segment)
    local_text = segment.text.strip()
    context_text = _bounded_context(segment.context_text.strip(), local_text=local_text, limit=context_char_limit)
    parts = [
        segment.title.strip(),
        descriptor,
        "\n".join(structural_markers),
        hierarchy,
        local_text,
        context_text,
    ]
    deduped_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = part.casefold()
        if not part or normalized in seen:
            continue
        seen.add(normalized)
        deduped_parts.append(part)
    return "\n".join(deduped_parts)


def _structural_markers(segment: ExternalSegmentRecord) -> list[str]:
    """Extract domain structural markers for richer legal embedding text.

    Args:
        segment: Source segment payload record.

    Returns:
        list[str]: Normalized marker lines derived from segment metadata.
    """

    markers: list[str] = []
    if segment.structure_type.strip():
        markers.append(f"structure_type: {segment.structure_type.strip()}")

    for part in segment.hierarchy:
        normalized = part.strip()
        if not normalized:
            continue
        if _STRUCTURAL_LINE_RE.match(normalized):
            markers.append(f"heading: {normalized}")
        elif _CAPTION_RE.search(normalized):
            markers.append(f"caption: {normalized}")

    blob = " \n".join(
        [
            segment.title,
            segment.context_text,
            segment.embedding_text,
            segment.metadata.document_descriptor,
        ]
    )
    if _ENACTMENT_NOTICE_RE.search(blob):
        markers.append("notice: enactment / commencement")
    if _ISSUED_BY_RE.search(blob):
        markers.append("panel: issued by")
    if _DATE_PANEL_RE.search(blob):
        markers.append("panel: date")
    if _LAW_NUMBER_RE.search(blob):
        markers.append("panel: law number")
    if segment.metadata.case_refs:
        markers.append(f"case_refs: {'; '.join(segment.metadata.case_refs)}")
    if segment.metadata.law_refs:
        markers.append(f"law_refs: {'; '.join(segment.metadata.law_refs)}")

    deduped: list[str] = []
    seen: set[str] = set()
    for marker in markers:
        normalized = marker.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(marker)
    return deduped


def _bounded_context(context_text: str, *, local_text: str, limit: int) -> str:
    if not context_text or limit <= 0:
        return ""
    if local_text and context_text.casefold().startswith(local_text.casefold()):
        tail = context_text[len(local_text) :].strip()
        return tail[:limit].strip()
    return context_text[:limit].strip()
