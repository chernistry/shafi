"""Deterministic instruction builders for instruction-conditioned reranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_ARTICLE_RE = re.compile(r"\b(article|section|schedule)\s+\d+[a-z]?\b", re.IGNORECASE)
_PARTY_RE = re.compile(r"\b(claimant|party|parties|defendant|appellant|respondent|caption)\b", re.IGNORECASE)
_TITLE_RE = re.compile(r"\b(title|law title|short title|cited as)\b", re.IGNORECASE)
_AUTHORITY_RE = re.compile(r"\b(issued by|authority|judge|registrar|justice)\b", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(date of issue|issue date|effective date|commencement date"
    r"|enactment date|enactment|enact"
    r"|come into force|came into force|enter into force|entered into force"
    r"|take effect|took effect|in force from|in force on)\b",
    re.IGNORECASE,
)
_LAW_NUMBER_RE = re.compile(r"\b(law no|law number|enactment notice)\b", re.IGNORECASE)
_CASE_REF_RE = re.compile(r"\b[A-Z]{2,4}\s+\d{3}/\d{4}\b")


@dataclass(frozen=True, slots=True)
class RerankInstruction:
    """Instruction metadata for a rerank request.

    Args:
        family: Short deterministic family label for telemetry and tests.
        instruction: Provider-facing instruction text.
    """

    family: str
    instruction: str


def build_rerank_instruction(
    query: str,
    answer_type: str,
    *,
    doc_refs: Sequence[str] = (),
) -> RerankInstruction | None:
    """Build a deterministic rerank instruction from query signals.

    Args:
        query: Raw user query.
        answer_type: Normalized answer type.
        doc_refs: Extracted document references from the pipeline state.

    Returns:
        Matching rerank instruction or ``None`` when the default rerank path
        should remain plain query-plus-documents.
    """

    query_text = str(query or "").strip()
    case_ref_count = len(_CASE_REF_RE.findall(query_text))
    doc_ref_count = len([ref for ref in doc_refs if str(ref).strip()])

    if case_ref_count >= 2 or doc_ref_count >= 2:
        return RerankInstruction(
            family="compare_authoritative_pair",
            instruction=(
                "Prefer one authoritative page per referenced side. Favor caption, title, or operative pages that "
                "name the party, claimant, judge, or direct outcome over incidental mentions."
            ),
        )
    if _ARTICLE_RE.search(query_text):
        return RerankInstruction(
            family="exact_provision",
            instruction=(
                "Prefer the exact article, section, or schedule page that directly states the provision. "
                "Penalize commentary, summaries, duplicates, and nearby incidental mentions."
            ),
        )
    if _AUTHORITY_RE.search(query_text) or _DATE_RE.search(query_text) or _LAW_NUMBER_RE.search(query_text):
        return RerankInstruction(
            family="authority_metadata",
            instruction=(
                "Prefer official title, issuance, enactment, or authority blocks that explicitly state the date, "
                "issuing authority, law number, or enactment notice."
            ),
        )
    if _PARTY_RE.search(query_text) or _TITLE_RE.search(query_text):
        return RerankInstruction(
            family="title_caption_party",
            instruction=(
                "Prefer caption, title, and official heading pages that explicitly identify the party, claimant, "
                "case title, or short title. Penalize body references and secondary mentions."
            ),
        )
    if str(answer_type or "").casefold() in {"date", "name", "names"}:
        return RerankInstruction(
            family="strict_field_lookup",
            instruction=(
                "Prefer the most authoritative page that explicitly states the requested field value and avoid "
                "broad contextual pages that only mention related material."
            ),
        )
    return None


def compose_instruction_conditioned_query(query: str, instruction: str | None) -> str:
    """Compose the provider-facing query text for instruction-aware reranking.

    Args:
        query: Raw user query.
        instruction: Optional rerank instruction.

    Returns:
        Query text with a deterministic instruction prefix when supplied.
    """

    query_text = str(query or "").strip()
    if not instruction:
        return query_text
    return f"Instruction: {instruction}\nQuestion: {query_text}"
