"""Deterministic page-semantic helpers for grounding sidecar ranking.

This module keeps richer page-understanding rules out of ``evidence_selector``
so they remain testable, auditable, and easy to roll back independently.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from shafi.core.grounding.authority_priors import (
    authority_signal_score as _authority_signal_score,
)
from shafi.core.grounding.authority_priors import (
    matches_field_family as _authority_matches_field_family,
)
from shafi.models.schemas import RetrievedPage, ScopeMode

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "article",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "page",
    "the",
    "to",
    "what",
    "which",
    "who",
}
_DATE_QUERY_RE = re.compile(r"\b(date|effective|issued)\b", re.IGNORECASE)
_AUTHORITY_QUERY_RE = re.compile(r"\b(judge|registrar|authority|issued by|issued-by)\b", re.IGNORECASE)
_CLAIM_QUERY_RE = re.compile(r"\bclaim no|claim number\b", re.IGNORECASE)
_LAW_NUMBER_RE = re.compile(r"\blaw no|law number\b", re.IGNORECASE)
_SCHEDULE_QUERY_RE = re.compile(r"\b(schedule|table|penalty|fine|rate|fee)\b", re.IGNORECASE)
_TITLE_ARTICLE_QUERY_RE = re.compile(r"\b(article|section|schedule)\b", re.IGNORECASE)
_PARTY_QUERY_RE = re.compile(r"\b(claimant|party|parties|judge|title|caption)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SemanticPageSetDecision:
    """Bounded page-set decision produced by semantic selection rules.

    Args:
        page_ids: Ordered selected page IDs.
        activation_family: Short label describing why the pair/neighbor rule fired.
    """

    page_ids: tuple[str, ...]
    activation_family: str


def build_shadow_text(page: RetrievedPage) -> str:
    """Build a compact shadow representation from page-top semantics.

    Args:
        page: Retrieved page payload with semantic metadata.

    Returns:
        str: Compact representation emphasizing heading, field, and template cues.
    """

    parts = [
        *page.heading_lines[:4],
        *page.top_lines[:6],
        *page.field_labels_present,
        page.page_template_family,
        page.document_template_family,
        page.canonical_law_family,
        *page.article_refs,
    ]
    compact = " | ".join(part.strip() for part in parts if part and part.strip())
    return re.sub(r"\s+", " ", compact).strip()


def rerank_pages_with_shadow_signal(query: str, pages: list[RetrievedPage]) -> list[RetrievedPage]:
    """Reorder pages using compact heading/field shadow signals.

    Args:
        query: Raw grounding query.
        pages: Retrieved pages from the page collection.

    Returns:
        list[RetrievedPage]: Pages sorted by original score plus shadow-signal boost.
    """

    rescored = [page.model_copy(update={"score": page.score + shadow_signal_score(query, page)}) for page in pages]
    return sorted(
        rescored,
        key=lambda page: (
            -page.score,
            -page.officialness_score,
            page.page_num,
            page.page_id,
        ),
    )


def shadow_signal_score(query: str, page: RetrievedPage) -> float:
    """Score compact heading/field shadow match for a page.

    Args:
        query: Raw grounding query.
        page: Candidate page.

    Returns:
        float: Additive score boost from shadow-text match.
    """

    query_tokens = _content_tokens(query)
    if not query_tokens:
        return 0.0

    heading_tokens = _content_tokens(" ".join(page.heading_lines))
    top_tokens = _content_tokens(" ".join(page.top_lines))
    field_tokens = _content_tokens(" ".join(page.field_labels_present))

    score = 0.0
    if heading_tokens:
        score += 0.35 * _overlap_ratio(query_tokens, heading_tokens)
    if top_tokens:
        score += 0.2 * _overlap_ratio(query_tokens, top_tokens)
    if field_tokens:
        score += 0.25 * _overlap_ratio(query_tokens, field_tokens)

    if _DATE_QUERY_RE.search(query) and (page.has_date_of_issue_pattern or _matches_field_family(page, "date")):
        score += 0.35
    if _AUTHORITY_QUERY_RE.search(query) and (
        page.has_issued_by_pattern
        or page.page_template_family in {"caption_header", "issued_by_authority"}
        or _matches_field_family(page, "authority")
    ):
        score += 0.35
    if _CLAIM_QUERY_RE.search(query) and (page.has_claim_number_pattern or _matches_field_family(page, "claim")):
        score += 0.3
    if _LAW_NUMBER_RE.search(query) and (page.has_law_number_pattern or _matches_field_family(page, "law_number")):
        score += 0.3
    return score


def authority_signal_score(
    query: str,
    page: RetrievedPage,
    *,
    peer_pages: list[RetrievedPage],
) -> float:
    """Score page authority while suppressing reference/incidental pages.

    Args:
        query: Raw grounding query.
        page: Candidate page to score.
        peer_pages: Other retrieved pages from the same retrieval pass.

    Returns:
        float: Additive authority bonus or suppression penalty.
    """
    return _authority_signal_score(query, page, peer_pages=peer_pages) + shadow_signal_score(query, page)


def select_semantic_page_set(
    *,
    query: str,
    scope_mode: ScopeMode,
    ordered_page_ids: list[str],
    page_candidates: list[RetrievedPage],
    page_budget: int,
) -> SemanticPageSetDecision | None:
    """Select a bounded semantic page pair or neighborhood when justified.

    Args:
        query: Raw grounding query.
        scope_mode: Query scope mode controlling bounded behavior.
        ordered_page_ids: Current heuristic ordering.
        page_candidates: Retrieved page candidates.
        page_budget: Current scope page budget.

    Returns:
        SemanticPageSetDecision | None: Explicit bounded page-set choice or ``None``.
    """

    if scope_mode not in {ScopeMode.COMPARE_PAIR, ScopeMode.FULL_CASE_FILES, ScopeMode.SINGLE_FIELD_SINGLE_DOC}:
        return None
    if scope_mode in {ScopeMode.COMPARE_PAIR, ScopeMode.FULL_CASE_FILES} and page_budget < 2:
        return None
    if len(page_candidates) < 2:
        return None
    if scope_mode is ScopeMode.SINGLE_FIELD_SINGLE_DOC and not _supports_single_doc_pair(query):
        return None

    pages_by_id = {page.page_id: page for page in page_candidates if page.page_id}
    ordered_pages = [pages_by_id[page_id] for page_id in ordered_page_ids if page_id in pages_by_id]
    if len(ordered_pages) < 2:
        return None

    if _SCHEDULE_QUERY_RE.search(query):
        decision = _select_schedule_pair(ordered_pages)
        if decision is not None:
            return decision

    if _TITLE_ARTICLE_QUERY_RE.search(query):
        decision = _select_title_article_pair(ordered_pages)
        if decision is not None:
            return decision
        decision = _select_adjacent_article_pair(ordered_pages)
        if decision is not None:
            return decision

    if _PARTY_QUERY_RE.search(query):
        decision = _select_caption_operative_pair(ordered_pages)
        if decision is not None:
            return decision

    return None


def _select_schedule_pair(ordered_pages: list[RetrievedPage]) -> SemanticPageSetDecision | None:
    by_doc = _pages_by_doc(ordered_pages)
    for doc_pages in by_doc.values():
        schedule = next((page for page in doc_pages if page.page_template_family == "schedule_table"), None)
        header = next(
            (
                page
                for page in doc_pages
                if page.page_template_family in {"title_cover", "official_primary", "issued_by_authority"}
                and page.page_id != getattr(schedule, "page_id", "")
            ),
            None,
        )
        if schedule and header:
            return SemanticPageSetDecision((header.page_id, schedule.page_id), "schedule_pair")
    return None


def _select_title_article_pair(ordered_pages: list[RetrievedPage]) -> SemanticPageSetDecision | None:
    by_doc = _pages_by_doc(ordered_pages)
    for doc_pages in by_doc.values():
        article = next((page for page in doc_pages if page.page_template_family == "article_body"), None)
        structural = next(
            (
                page
                for page in doc_pages
                if page.page_template_family in {"title_cover", "issued_by_authority", "official_primary"}
                and page.page_id != getattr(article, "page_id", "")
            ),
            None,
        )
        if article and structural:
            return SemanticPageSetDecision((structural.page_id, article.page_id), "title_article_pair")
    return None


def _select_adjacent_article_pair(ordered_pages: list[RetrievedPage]) -> SemanticPageSetDecision | None:
    by_doc = _pages_by_doc(ordered_pages)
    for doc_pages in by_doc.values():
        article_pages = [page for page in doc_pages if page.page_template_family == "article_body"]
        for first in article_pages:
            for second in article_pages:
                if first.page_id == second.page_id:
                    continue
                if abs(first.page_num - second.page_num) == 1:
                    return SemanticPageSetDecision((first.page_id, second.page_id), "adjacent_article_pair")
    return None


def _select_caption_operative_pair(ordered_pages: list[RetrievedPage]) -> SemanticPageSetDecision | None:
    by_doc = _pages_by_doc(ordered_pages)
    for doc_pages in by_doc.values():
        caption = next((page for page in doc_pages if page.page_template_family == "caption_header"), None)
        operative = next((page for page in doc_pages if page.page_template_family == "operative_order"), None)
        if caption and operative:
            return SemanticPageSetDecision((caption.page_id, operative.page_id), "caption_operative_pair")
    return None


def _pages_by_doc(pages: list[RetrievedPage]) -> dict[str, list[RetrievedPage]]:
    grouped: dict[str, list[RetrievedPage]] = defaultdict(list)
    for page in pages:
        if page.doc_id and page.page_id:
            grouped[page.doc_id].append(page)
    return grouped


def _matches_field_family(page: RetrievedPage, family: str) -> bool:
    return _authority_matches_field_family(page, family)


def _supports_single_doc_pair(query: str) -> bool:
    """Return whether single-doc pair mode is justified for the query.

    Args:
        query: Raw grounding query.

    Returns:
        ``True`` when a narrow two-page family is requested.
    """

    return bool(
        _SCHEDULE_QUERY_RE.search(query) or _TITLE_ARTICLE_QUERY_RE.search(query) or _PARTY_QUERY_RE.search(query)
    )


def _content_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall((text or "").casefold()) if token not in _STOPWORDS}


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))
