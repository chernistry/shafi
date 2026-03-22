"""Doc-scope helpers for grounding sidecar multi-document selection."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_challenge.core.grounding.law_family_graph import (
    LawFamilyBundle,
    build_candidate_law_family_bundle,
    build_query_law_family_bundle,
    law_family_match_score,
)
from rag_challenge.models.schemas import QueryScopePrediction, ScopeMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models import RankedChunk

# Matches standard "CFI 069/2024", fully-hyphenated "ENF-022-2023",
# mixed-separator "ENF 084-2023" / "CFI 084-2024", and slash-prefix
# "ARB/031/2025" variants seen in private corpus questions.
_CASE_REF_RE = re.compile(
    r"\b[A-Z]{2,4}(?:\s+\d{3}[-/]\d{4}|-\d{3}-\d{4}|/\d{3}/\d{4})\b",
    re.IGNORECASE,
)
# Fully-hyphenated: "ENF-022-2023" → groups (prefix, number, year)
_CASE_REF_HYPH_RE = re.compile(r"^([A-Za-z]{2,4})-(\d{3})-(\d{4})$")
# Mixed: "ENF 084-2023" → groups (prefix, number, year)
_CASE_REF_MIXED_RE = re.compile(r"^([A-Za-z]{2,4})\s+(\d{3})-(\d{4})$")
# Slash-prefix: "ARB/031/2025" → groups (prefix, number, year)
_CASE_REF_SLASH_RE = re.compile(r"^([A-Za-z]{2,4})/(\d{3})/(\d{4})$")


@dataclass(frozen=True)
class DocScopeCandidate:
    """Context-derived candidate document for sidecar scope selection."""

    doc_id: str
    doc_title: str
    evidence_text: str
    hit_count: int
    law_family_bundle: LawFamilyBundle
    all_case_numbers: frozenset[str] = frozenset()


def extract_case_refs(query: str) -> list[str]:
    """Extract ordered unique case references from a query.

    Normalises both standard "CFI 069/2024" and hyphenated "ENF-022-2023"
    variants to the canonical space+slash form "ENF 022/2023".

    Args:
        query: Raw user question text.

    Returns:
        Ordered unique case-reference strings as they appear in the query.
    """
    seen: set[str] = set()
    refs: list[str] = []
    for match in _CASE_REF_RE.finditer(query or ""):
        raw = match.group(0).strip()
        # Normalise all separator variants to canonical "ENF 022/2023" form.
        hyph = _CASE_REF_HYPH_RE.match(raw)
        mixed = _CASE_REF_MIXED_RE.match(raw)
        slash = _CASE_REF_SLASH_RE.match(raw)
        if hyph:
            value = f"{hyph.group(1).upper()} {hyph.group(2)}/{hyph.group(3)}"
        elif mixed:
            value = f"{mixed.group(1).upper()} {mixed.group(2)}/{mixed.group(3)}"
        elif slash:
            value = f"{slash.group(1).upper()} {slash.group(2)}/{slash.group(3)}"
        else:
            value = re.sub(r"\s+", " ", raw)
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
    query_law_bundle = build_query_law_family_bundle(query)
    if scope.scope_mode is ScopeMode.COMPARE_PAIR:
        return _select_compare_doc_scope(candidates, case_refs, page_budget=scope.page_budget)
    if scope.scope_mode is ScopeMode.FULL_CASE_FILES and scope.requires_all_docs_in_case:
        return _select_full_case_doc_scope(candidates, case_refs)
    if query_law_bundle.exact_keys:
        law_scope = _select_law_family_doc_scope(candidates, query_law_bundle)
        if law_scope:
            # Safety: always include the top-ranked context candidate.
            # The law family matcher can pick a secondary doc and miss
            # the primary one when law abbreviations are ambiguous.
            top_doc = candidates[0].doc_id
            if top_doc not in law_scope:
                law_scope.insert(0, top_doc)
            return law_scope
    return [candidate.doc_id for candidate in candidates]


def _build_doc_scope_candidates(context_chunks: Sequence[RankedChunk]) -> list[DocScopeCandidate]:
    """Collapse context chunks into scored document candidates.

    Args:
        context_chunks: Ranked answer-path context chunks.

    Returns:
        Ordered doc candidates scored by context footprint strength.
    """
    hit_counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    doc_titles: dict[str, str] = {}
    evidence_texts: dict[str, str] = {}
    law_family_bundles: dict[str, LawFamilyBundle] = {}
    case_numbers_by_doc: dict[str, set[str]] = {}

    for idx, chunk in enumerate(context_chunks):
        doc_id = str(chunk.doc_id or "").strip()
        if not doc_id:
            continue
        hit_counts[doc_id] += 1
        if doc_id not in first_seen:
            first_seen[doc_id] = idx
        if doc_id not in doc_titles:
            doc_titles[doc_id] = str(chunk.doc_title or "")
        if doc_id not in evidence_texts and chunk.text:
            evidence_texts[doc_id] = str(chunk.text)
        if doc_id not in law_family_bundles:
            law_family_bundles[doc_id] = build_candidate_law_family_bundle(
                str(chunk.doc_title or ""),
                tuple(str(title) for title in getattr(chunk, "law_titles", []) if str(title).strip()),
            )
        # Collect case_numbers from all chunks — needed for docs whose title is
        # party names only (e.g. "Tr88house v Bond", no "CA 006/2024" in title).
        for cn in getattr(chunk, "case_numbers", []) or []:
            cleaned = str(cn).strip()
            if cleaned:
                case_numbers_by_doc.setdefault(doc_id, set()).add(cleaned)

    ordered_doc_ids = sorted(
        hit_counts,
        key=lambda doc_id: (
            -hit_counts[doc_id],
            first_seen.get(doc_id, 0),  # preserve retrieval order for equal hit counts
        ),
    )
    return [
        DocScopeCandidate(
            doc_id=doc_id,
            doc_title=doc_titles.get(doc_id, ""),
            evidence_text=evidence_texts.get(doc_id, ""),
            hit_count=hit_counts[doc_id],
            law_family_bundle=law_family_bundles.get(doc_id, LawFamilyBundle(exact_keys=(), related_keys=())),
            all_case_numbers=frozenset(case_numbers_by_doc.get(doc_id, set())),
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


def _select_law_family_doc_scope(
    candidates: Sequence[DocScopeCandidate],
    query_law_bundle: LawFamilyBundle,
) -> list[str]:
    """Select documents whose normalized law families match the query.

    Args:
        candidates: Ordered document candidates from context.
        query_law_bundle: Canonical query law-family bundle.

    Returns:
        list[str]: Matching document IDs in ranked order.
    """
    scored_matches = [
        (candidate.doc_id, law_family_match_score(query_law_bundle, candidate.law_family_bundle), candidate.hit_count)
        for candidate in candidates
    ]
    matched = [
        doc_id
        for doc_id, score, _ in sorted(scored_matches, key=lambda item: (-item[1], -item[2], item[0]))
        if score > 0.0
    ]
    return matched


def _matches_case_ref(candidate: DocScopeCandidate, case_ref: str) -> bool:
    """Return whether a context-derived doc candidate matches a case reference.

    Args:
        candidate: Context-derived doc candidate.
        case_ref: Case reference string extracted from the query.

    Returns:
        True when the candidate text, title, or case_numbers metadata matches the case ref.
    """
    needle = case_ref.casefold()
    if needle in candidate.doc_title.casefold() or needle in candidate.evidence_text.casefold():
        return True
    # Fallback: check case_numbers extracted during ingest.  Critical for docs whose
    # title is party names only (e.g. "Tr88house v Bond") and whose OCR text is empty —
    # neither title nor evidence_text carries the case ref, but the payload field does.
    return any(needle == cn.casefold() for cn in candidate.all_case_numbers)
