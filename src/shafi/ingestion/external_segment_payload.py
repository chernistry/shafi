"""Typed loader for the external structured segment payload."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


type JsonObject = dict[str, object]


def _segment_list_factory() -> list[ExternalSegmentRecord]:
    """Return a typed empty segment list for Pydantic defaults."""

    return []


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class ExternalSegmentMetadata(BaseModel):
    """Structured metadata attached to one external segment.

    Args:
        case_refs: Case-reference strings found in the segment.
        law_refs: Law-reference strings found in the segment.
        token_count: Source token count emitted by the payload generator.
        document_descriptor: Compact document descriptor from the payload.
    """

    case_refs: list[str] = Field(default_factory=list)
    law_refs: list[str] = Field(default_factory=list)
    token_count: int = 0
    document_descriptor: str = ""


class ExternalSegmentRecord(BaseModel):
    """Normalized external segment record.

    Args:
        segment_id: Stable external segment identifier.
        doc_id: Parent document identifier.
        page_number: 1-based page number.
        text: Raw segment text.
        title: Document title field from the payload.
        structure_type: Segment structure type.
        hierarchy: Hierarchy path emitted by the payload.
        context_text: Expanded context text for the segment.
        embedding_text: Embedding-oriented text from the payload.
        metadata: Structured metadata bundle.
    """

    segment_id: str
    doc_id: str
    page_number: int
    text: str = ""
    title: str = ""
    structure_type: str = ""
    hierarchy: list[str] = Field(default_factory=list)
    context_text: str = ""
    embedding_text: str = ""
    metadata: ExternalSegmentMetadata = Field(default_factory=ExternalSegmentMetadata)

    @property
    def page_id(self) -> str:
        """Return the platform-style page identifier for the segment."""
        return f"{self.doc_id}_{self.page_number}"

    @property
    def search_blob(self) -> str:
        """Return the normalized text blob used by offline lexical scoring."""
        parts = [
            self.text,
            self.context_text,
            self.embedding_text,
            " ".join(self.hierarchy),
            " ".join(self.metadata.case_refs),
            " ".join(self.metadata.law_refs),
            self.metadata.document_descriptor,
        ]
        return "\n".join(part for part in parts if part)


class ExternalSegmentPayload(BaseModel):
    """Top-level external payload bundle."""

    embedding_model: str = ""
    segments_path: str = ""
    output_cache_name: str = ""
    segments: list[ExternalSegmentRecord] = Field(default_factory=_segment_list_factory)


def load_external_segment_payload(path: Path) -> ExternalSegmentPayload:
    """Load the external segment payload from JSON.

    Args:
        path: Payload JSON path.

    Returns:
        Parsed payload with typed segment rows.

    Raises:
        ValueError: If the payload root is not a JSON object.
    """

    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object payload at {path}")
    return ExternalSegmentPayload.model_validate(cast("JsonObject", raw))


def normalize_shadow_text(text: str) -> str:
    """Normalize a text string for lightweight lexical matching.

    Args:
        text: Raw text.

    Returns:
        Lowercased alphanumeric token string joined by single spaces.
    """

    return " ".join(token.lower() for token in _TOKEN_RE.findall(text))


def tokenize_shadow_text(text: str) -> list[str]:
    """Tokenize text for lexical shadow retrieval.

    Args:
        text: Raw text.

    Returns:
        Ordered normalized tokens.
    """

    return normalize_shadow_text(text).split()
