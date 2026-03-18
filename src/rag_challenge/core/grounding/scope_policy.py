"""Doc-scope helpers for grounding sidecar multi-document selection."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_challenge.models.schemas import QueryScopePrediction, ScopeMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models import RankedChunk

_CASE_REF_RE = re.compile(r"\b[A-Z]{2,4}\s+\d{3}/\d{4}\b", re.IGNORECASE)


@dataclass(frozen=True)
class DocScopeCandidate:
    """Context-derived candidate document for sidecar scope selection."""

    doc_id: str
    doc_title: str
    evidence_text: str
    hit_count: int


def extract_case_refs(query: str) -> list[str]:
    """Extract ordered unique case references from a query.

    Args:
        query: Raw user question text.

    Returns:
        Ordered unique case-reference strings as they appear in the query.
    """
    seen: set[str] = set()
    refs: list[str] = []
    for match in _CASE_REF_RE.finditer(query or ""):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        refs.append(value)
    return refs


def select_sidecar_doc_scope(
    *,
    query: str,
    scope: QueryScopePrediction,
    context_chunks: Sequence[RankedChunk],
) -> list[str]:
    """Select the minimal doc scope for sidecar page retrieval.

    Args:
        query: Raw user question text.
        scope: Query scope prediction for the sidecar.
        context_chunks: Ranked answer-path context chunks.

    Returns:
        Ordered document IDs that should be searched for grounding pages.
    """
    candidates = _build_doc_scope_candidates(context_chunks)
    if not candidates:
        return []

    case_refs = extract_case_refs(query)
    if scope.scope_mode is ScopeMode.COMPARE_PAIR:
        return _select_compare_doc_scope(candidates, case_refs, page_budget=scope.page_budget)
    if scope.scope_mode is ScopeMode.FULL_CASE_FILES and scope.requires_all_docs_in_case:
        return _select_full_case_doc_scope(candidates, case_refs)
    return [candidate.doc_id for candidate in candidates]


def _build_doc_scope_candidates(context_chunks: Sequence[RankedChunk]) -> list[DocScopeCandidate]:
    """Collapse context chunks into scored document candidates.

    Args:
        context_chunks: Ranked answer-path context chunks.

    Returns:
        Ordered doc candidates scored by context footprint strength.
    """
    hit_counts: Counter[str] = Counter()
    doc_titles: dict[str, str] = {}
    evidence_texts: dict[str, str] = {}

    for chunk in context_chunks:
        doc_id = str(chunk.doc_id or "").strip()
        if not doc_id:
            continue
        hit_counts[doc_id] += 1
        if doc_id not in doc_titles:
            doc_titles[doc_id] = str(chunk.doc_title or "")
        if doc_id not in evidence_texts and chunk.text:
            evidence_texts[doc_id] = str(chunk.text)

    ordered_doc_ids = sorted(
        hit_counts,
        key=lambda doc_id: (
            -hit_counts[doc_id],
            doc_titles.get(doc_id, "").casefold(),
            doc_id,
        ),
    )
    return [
        DocScopeCandidate(
            doc_id=doc_id,
            doc_title=doc_titles.get(doc_id, ""),
            evidence_text=evidence_texts.get(doc_id, ""),
            hit_count=hit_counts[doc_id],
        )
        for doc_id in ordered_doc_ids
    ]


def _select_compare_doc_scope(
    candidates: Sequence[DocScopeCandidate],
    case_refs: Sequence[str],
    *,
    page_budget: int,
) -> list[str]:
    """Select one relevant document per requested case in compare scope.

    Args:
        candidates: Ordered document candidates from context.
        case_refs: Case references extracted from the query.
        page_budget: Requested compare-page budget from the scope classifier.

    Returns:
        Ordered doc IDs for compare-side page retrieval.
    """
    if case_refs:
        selected: list[str] = []
        for case_ref in case_refs:
            match = next((candidate for candidate in candidates if _matches_case_ref(candidate, case_ref)), None)
            if match is None or match.doc_id in selected:
                continue
            selected.append(match.doc_id)
        if selected:
            return selected[: max(2, page_budget)]

    return [candidate.doc_id for candidate in candidates[: max(2, page_budget)]]


def _select_full_case_doc_scope(
    candidates: Sequence[DocScopeCandidate],
    case_refs: Sequence[str],
) -> list[str]:
    """Select all documents that belong to the requested case scope.

    Args:
        candidates: Ordered document candidates from context.
        case_refs: Case references extracted from the query.

    Returns:
        Ordered doc IDs that belong to the requested full-case scope.
    """
    if case_refs:
        matched = [
            candidate.doc_id
            for candidate in candidates
            if any(_matches_case_ref(candidate, case_ref) for case_ref in case_refs)
        ]
        if matched:
            return matched
    return [candidate.doc_id for candidate in candidates]


def _matches_case_ref(candidate: DocScopeCandidate, case_ref: str) -> bool:
    """Return whether a context-derived doc candidate matches a case reference.

    Args:
        candidate: Context-derived doc candidate.
        case_ref: Case reference string extracted from the query.

    Returns:
        True when the candidate text or title matches the case reference.
    """
    needle = case_ref.casefold()
    return needle in candidate.doc_title.casefold() or needle in candidate.evidence_text.casefold()
