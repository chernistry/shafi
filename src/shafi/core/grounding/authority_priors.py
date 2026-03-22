"""Deterministic authority-prior helpers for grounding sidecar decisions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from shafi.models.schemas import QueryScopePrediction, RetrievedPage, ScopeMode

if TYPE_CHECKING:
    from collections.abc import Sequence

_DATE_QUERY_RE = re.compile(r"\b(date|effective|issued)\b", re.IGNORECASE)
_AUTHORITY_QUERY_RE = re.compile(r"\b(judge|registrar|authority|issued by|issued-by)\b", re.IGNORECASE)
_CLAIM_QUERY_RE = re.compile(r"\bclaim no|claim number\b", re.IGNORECASE)
_LAW_NUMBER_RE = re.compile(r"\blaw no|law number|enactment notice\b", re.IGNORECASE)
_SCHEDULE_QUERY_RE = re.compile(r"\b(schedule|table|penalty|fine|rate|fee)\b", re.IGNORECASE)
_TITLE_ARTICLE_QUERY_RE = re.compile(r"\b(article|section|schedule)\b", re.IGNORECASE)
_PARTY_QUERY_RE = re.compile(r"\b(claimant|party|parties|judge|title|caption)\b", re.IGNORECASE)
_OUTCOME_QUERY_RE = re.compile(r"\b(costs|cost awarded|final ruling|outcome|ordered)\b", re.IGNORECASE)
_QUERY_FIELD_LABELS = {
    "date": {"date of issue", "effective date", "date"},
    "authority": {"issued by", "judge", "registrar", "authority"},
    "claim": {"claim no", "claim number"},
    "law_number": {"law no", "law number", "enactment notice"},
}
_REFERENCE_TEMPLATE_FAMILIES = {"appendix_reference", "duplicate_or_reference_like"}
_TITLE_PRIMARY_FAMILIES = {"title_cover", "caption_header", "issued_by_authority", "official_primary"}
_OPERATIVE_PRIMARY_FAMILIES = {"operative_order", "official_primary"}


def supports_single_doc_authority_scope(query: str, scope: QueryScopePrediction) -> bool:
    """Return whether a single-doc scope should activate authority-first sidecar.

    Args:
        query: Raw grounding query.
        scope: Query-scope prediction from the sidecar classifier.

    Returns:
        ``True`` when the query is a narrow authoritative-page family that can
        safely use the sidecar for single-document grounding.
    """

    if scope.scope_mode is not ScopeMode.SINGLE_FIELD_SINGLE_DOC:
        return False
    if scope.hard_anchor_strings:
        return True
    query_text = str(query or "")
    if any(
        pattern.search(query_text)
        for pattern in (
            _DATE_QUERY_RE,
            _AUTHORITY_QUERY_RE,
            _CLAIM_QUERY_RE,
            _LAW_NUMBER_RE,
            _SCHEDULE_QUERY_RE,
            _TITLE_ARTICLE_QUERY_RE,
            _PARTY_QUERY_RE,
            _OUTCOME_QUERY_RE,
        )
    ):
        return True
    target_roles = {role.strip().casefold() for role in scope.target_page_roles if role.strip()}
    return bool(
        target_roles
        & {
            "issued_by_block",
            "issued_by_authority",
            "article_clause",
            "schedule_table",
            "operative_order",
            "costs_block",
        }
    )


def authority_signal_score(
    query: str,
    page: RetrievedPage,
    *,
    peer_pages: Sequence[RetrievedPage],
) -> float:
    """Score page authority while suppressing reference/incidental pages.

    Args:
        query: Raw grounding query.
        page: Candidate page to score.
        peer_pages: Other candidate pages considered in the same selection pass.

    Returns:
        Additive authority bonus or penalty.
    """

    if not has_semantic_authority_signals(page):
        return 0.0

    score = 0.5 * page.officialness_score + 0.35 * page.source_vs_reference_prior
    score += template_precedence_bonus(query, page)

    if page.page_template_family in _REFERENCE_TEMPLATE_FAMILIES:
        score -= 0.7
    if page.source_vs_reference_prior < 0.35:
        score -= 0.25

    strongest_same_doc = strongest_page_by_doc(peer_pages).get(page.doc_id)
    if strongest_same_doc is not None and strongest_same_doc.page_id != page.page_id:
        officialness_gap = strongest_same_doc.officialness_score - page.officialness_score
        if officialness_gap >= 0.25 and page.page_template_family in _REFERENCE_TEMPLATE_FAMILIES:
            score -= 0.5
        if (
            is_title_like_query(query)
            and page.page_template_family not in _TITLE_PRIMARY_FAMILIES
            and strongest_same_doc.page_template_family in _TITLE_PRIMARY_FAMILIES
        ):
            score -= 0.25
        if (
            _OUTCOME_QUERY_RE.search(query)
            and page.page_template_family not in _OPERATIVE_PRIMARY_FAMILIES
            and strongest_same_doc.page_template_family in _OPERATIVE_PRIMARY_FAMILIES
        ):
            score -= 0.25
    return score


def select_authoritative_single_page(query: str, ordered_pages: Sequence[RetrievedPage]) -> RetrievedPage | None:
    """Choose the best single authoritative page from ordered candidates.

    Args:
        query: Raw grounding query.
        ordered_pages: Candidate pages in current score order.

    Returns:
        The best authoritative single page or ``None`` when no candidates exist.
    """

    if not ordered_pages:
        return None

    peer_pages = list(ordered_pages)
    best_page: RetrievedPage | None = None
    best_key: tuple[float, float, float, float, str] | None = None
    for index, page in enumerate(ordered_pages):
        key = (
            authority_signal_score(query, page, peer_pages=peer_pages),
            template_precedence_bonus(query, page),
            page.officialness_score,
            page.source_vs_reference_prior - (0.02 * index),
            page.page_id,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_page = page
    return best_page


def template_precedence_bonus(query: str, page: RetrievedPage) -> float:
    """Return a query-aware template precedence bonus for a page.

    Args:
        query: Raw grounding query.
        page: Candidate page.

    Returns:
        Additive template bonus.
    """

    score = 0.0
    if _DATE_QUERY_RE.search(query) and (
        page.has_date_of_issue_pattern
        or page.page_template_family in {"issued_by_authority", "title_cover", "official_primary"}
        or matches_field_family(page, "date")
    ):
        score += 0.5
    if _AUTHORITY_QUERY_RE.search(query) and (
        page.has_issued_by_pattern
        or page.page_template_family in {"caption_header", "issued_by_authority", "title_cover"}
        or matches_field_family(page, "authority")
    ):
        score += 0.45
    if _CLAIM_QUERY_RE.search(query) and (page.has_claim_number_pattern or matches_field_family(page, "claim")):
        score += 0.35
    if _LAW_NUMBER_RE.search(query) and (page.has_law_number_pattern or matches_field_family(page, "law_number")):
        score += 0.35
    if _PARTY_QUERY_RE.search(query) and page.page_template_family in {
        "caption_header",
        "title_cover",
        "official_primary",
    }:
        score += 0.4
    if _TITLE_ARTICLE_QUERY_RE.search(query) and page.page_template_family in {"article_body", "schedule_table"}:
        score += 0.45
    if _OUTCOME_QUERY_RE.search(query) and page.page_template_family in _OPERATIVE_PRIMARY_FAMILIES:
        score += 0.45
    return score


def matches_field_family(page: RetrievedPage, family: str) -> bool:
    """Return whether page field labels match the requested family.

    Args:
        page: Candidate page.
        family: Supported field family label.

    Returns:
        ``True`` when the page contains a matching normalized field label.
    """

    labels = {label.casefold() for label in page.field_labels_present}
    return bool(labels & _QUERY_FIELD_LABELS[family])


def has_semantic_authority_signals(page: RetrievedPage) -> bool:
    """Return whether a page carries enough semantic signals for authority logic.

    Args:
        page: Candidate page.

    Returns:
        ``True`` when semantic authority features are present.
    """

    return bool(
        page.page_template_family
        or page.document_template_family
        or page.officialness_score > 0.0
        or page.source_vs_reference_prior > 0.0
        or page.field_labels_present
        or page.heading_lines
        or page.has_date_of_issue_pattern
        or page.has_issued_by_pattern
        or page.has_claim_number_pattern
        or page.has_law_number_pattern
    )


def strongest_page_by_doc(pages: Sequence[RetrievedPage]) -> dict[str, RetrievedPage]:
    """Return the strongest page per document by authority-oriented ordering.

    Args:
        pages: Candidate pages.

    Returns:
        Mapping of document ID to strongest authority-bearing page.
    """

    strongest: dict[str, RetrievedPage] = {}
    for page in pages:
        current = strongest.get(page.doc_id)
        if current is None or page_rank_key(page) > page_rank_key(current):
            strongest[page.doc_id] = page
    return strongest


def page_rank_key(page: RetrievedPage) -> tuple[float, float, float, str]:
    """Build a deterministic authority-oriented rank key for a page.

    Args:
        page: Candidate page.

    Returns:
        Rank key used for strongest-page comparisons.
    """

    return (
        page.officialness_score,
        page.source_vs_reference_prior,
        page.score,
        page.page_id,
    )


def is_title_like_query(query: str) -> bool:
    """Return whether the query targets title/caption/authority-style evidence.

    Args:
        query: Raw grounding query.

    Returns:
        ``True`` when title-like authority families are requested.
    """

    return bool(_DATE_QUERY_RE.search(query) or _AUTHORITY_QUERY_RE.search(query) or _PARTY_QUERY_RE.search(query))
