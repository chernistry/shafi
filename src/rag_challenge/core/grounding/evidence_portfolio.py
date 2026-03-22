"""Deterministic evidence-portfolio selection for grounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_challenge.core.grounding.authority_priors import authority_signal_score
from rag_challenge.core.grounding.condition_audit import audit_candidate_pages
from rag_challenge.core.grounding.law_family_graph import (
    LawFamilyBundle,
    build_query_law_family_bundle,
    law_family_match_score,
)
from rag_challenge.models.schemas import ScopeMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models.schemas import QueryScopePrediction, RetrievedPage


@dataclass(frozen=True, slots=True)
class PortfolioCandidate:
    """One candidate evidence set considered by the portfolio selector.

    Args:
        name: Stable candidate identifier.
        page_ids: Ordered candidate page IDs.
        activation_family: Why the candidate exists.
        reasons: Human-readable source notes for telemetry/debugging.
    """

    name: str
    page_ids: tuple[str, ...]
    activation_family: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidencePortfolio:
    """Portfolio of page-set candidates for one grounding request.

    Args:
        query: Raw user query.
        scope_mode: Query scope for grounding.
        candidates: Ordered candidate evidence sets.
    """

    query: str
    scope_mode: ScopeMode
    candidates: tuple[PortfolioCandidate, ...]


@dataclass(frozen=True, slots=True)
class PortfolioSelectionResult:
    """Outcome of deterministic portfolio scoring.

    Args:
        selected_candidate: Winning candidate.
        ranked_candidates: Candidate ranking from best to worst.
        candidate_scores: Deterministic portfolio score by candidate name.
        decision_reasons: Short notes explaining the winning choice.
    """

    selected_candidate: PortfolioCandidate
    ranked_candidates: tuple[PortfolioCandidate, ...]
    candidate_scores: dict[str, float]
    decision_reasons: tuple[str, ...]


def select_best_page_set(
    *,
    query: str,
    answer_type: str,
    scope: QueryScopePrediction,
    page_lookup: dict[str, RetrievedPage],
    portfolio: EvidencePortfolio,
) -> PortfolioSelectionResult:
    """Choose the best evidence-set candidate from a deterministic portfolio.

    Args:
        query: Raw user question.
        answer_type: Normalized answer type.
        scope: Query-scope prediction.
        page_lookup: Candidate pages keyed by page ID.
        portfolio: Candidate evidence sets.

    Returns:
        PortfolioSelectionResult: Winning candidate plus deterministic ranking.
    """

    scored_candidates: list[tuple[float, int, str, PortfolioCandidate, tuple[str, ...]]] = []
    score_by_name: dict[str, float] = {}
    for candidate in portfolio.candidates:
        pages = [page_lookup[page_id] for page_id in candidate.page_ids if page_id in page_lookup]
        audit = audit_candidate_pages(
            query=query,
            answer_type=answer_type,
            scope_mode=scope.scope_mode,
            pages=pages,
        )
        score = _portfolio_score(
            query=query,
            scope=scope,
            pages=pages,
            audit_coverage=audit.coverage_ratio,
            activation_family=candidate.activation_family,
        )
        score_by_name[candidate.name] = round(score, 6)
        reasons = (
            f"coverage={audit.coverage_ratio:.2f}",
            f"pages={len(candidate.page_ids)}",
            f"family={candidate.activation_family}",
        )
        scored_candidates.append((score, -len(candidate.page_ids), candidate.name, candidate, reasons))

    scored_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    best_score, _, _, best_candidate, best_reasons = scored_candidates[0]
    ranked_candidates = tuple(item[3] for item in scored_candidates)
    return PortfolioSelectionResult(
        selected_candidate=best_candidate,
        ranked_candidates=ranked_candidates,
        candidate_scores=score_by_name,
        decision_reasons=(
            f"selected={best_candidate.name}",
            f"score={best_score:.3f}",
            *best_reasons,
        ),
    )


def build_law_bundle_candidate(
    *,
    query: str,
    ordered_pages: Sequence[RetrievedPage],
) -> PortfolioCandidate | None:
    """Build a narrow law-family candidate when the query names a law bundle.

    Args:
        query: Raw user question.
        ordered_pages: Candidate pages in current order.

    Returns:
        PortfolioCandidate | None: Law-family candidate or ``None``.
    """

    query_bundle = build_query_law_family_bundle(query)
    if not query_bundle.exact_keys:
        return None

    matching_pages = [
        page
        for page in ordered_pages
        if _page_law_bundle_match_score(query_bundle, page) > 0.0 and page.page_id
    ]
    if not matching_pages:
        return None

    matching_pages.sort(
        key=lambda page: (
            -_page_law_bundle_match_score(query_bundle, page),
            -page.officialness_score,
            page.page_num,
            page.page_id,
        ),
    )
    return PortfolioCandidate(
        name="law_bundle",
        page_ids=(matching_pages[0].page_id,),
        activation_family="law_bundle",
        reasons=(f"matched_family={matching_pages[0].canonical_law_family}",),
    )


def _portfolio_score(
    *,
    query: str,
    scope: QueryScopePrediction,
    pages: Sequence[RetrievedPage],
    audit_coverage: float,
    activation_family: str,
) -> float:
    """Compute a deterministic portfolio score from page-set properties.

    Args:
        query: Raw user question.
        scope: Query-scope prediction.
        pages: Candidate pages.
        audit_coverage: Typed-slot coverage ratio for the pages.
        activation_family: Candidate activation family label.

    Returns:
        float: Deterministic portfolio score.
    """

    if not pages:
        return -5.0
    authority_total = sum(
        authority_signal_score(query, page, peer_pages=pages)
        + (0.2 * page.officialness_score)
        + (0.1 * page.source_vs_reference_prior)
        for page in pages
    )
    distinct_docs = len({page.doc_id for page in pages if page.doc_id})
    # F-beta 2.5: recall 6.25x > precision. Reduced from 1.1 to 0.2 — mild
    # preference for focused sets without penalizing recall-boosting page sets.
    budget_penalty = max(0, len(pages) - max(1, scope.page_budget)) * 0.2
    reference_penalty = 0.35 * len(
        [
            page
            for page in pages
            if page.page_template_family == "duplicate_or_reference_like" or page.source_vs_reference_prior < 0.3
        ]
    )
    compare_bonus = 0.0
    if scope.scope_mode is ScopeMode.COMPARE_PAIR:
        compare_bonus = 0.8 if distinct_docs >= 2 else -1.0
    family_bonus = 0.2 if activation_family in {"typed_compare_panel", "law_bundle"} else 0.0
    return authority_total + (1.75 * audit_coverage) + compare_bonus + family_bonus - budget_penalty - reference_penalty


def _page_law_bundle_match_score(query_bundle: LawFamilyBundle, page: RetrievedPage) -> float:
    """Score how well one page matches the query law-family bundle.

    Args:
        query_bundle: Query law-family bundle.
        page: Candidate page.

    Returns:
        float: Narrow law-family match score.
    """

    candidate_bundle = LawFamilyBundle(
        exact_keys=tuple(value for value in [page.canonical_law_family] if value),
        related_keys=tuple(
            value
            for value in (
                page.canonical_law_family,
                *page.related_law_families,
                *page.law_title_aliases,
            )
            if value
        ),
    )
    return law_family_match_score(query_bundle, candidate_bundle)
