"""Deterministic escalation gate for bounded grounding-sidecar inquiry.

This module ports the useful LRAS idea of recognizing when the first-pass
selection is likely insufficient, without introducing freeform search loops,
RL, or answer-path mutations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_challenge.core.grounding.authority_priors import authority_signal_score
from rag_challenge.core.grounding.law_family_graph import build_query_law_family_bundle
from rag_challenge.core.rerank_instructions import build_rerank_instruction
from rag_challenge.models.schemas import ScopeMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.core.grounding.condition_audit import ConditionAuditResult
    from rag_challenge.models import RankedChunk
    from rag_challenge.models.schemas import QueryScopePrediction, RetrievedPage


@dataclass(frozen=True, slots=True)
class SearchEscalationDecision:
    """Deterministic decision about whether grounding should escalate.

    Args:
        should_escalate: Whether bounded escalation is justified.
        reasons: Ordered deterministic reasons for escalation.
        allowed_family: Supported family label for rewrite/verifier logic.
        allow_shadow_rewrite: Whether deterministic shadow retrieval is allowed.
        allow_relevance_verifier: Whether bounded LLM verification is allowed.
    """

    should_escalate: bool
    reasons: tuple[str, ...]
    allowed_family: str
    allow_shadow_rewrite: bool
    allow_relevance_verifier: bool


def decide_search_escalation(
    *,
    query: str,
    answer_type: str,
    scope: QueryScopePrediction,
    context_chunks: Sequence[RankedChunk],
    ordered_pages: Sequence[RetrievedPage],
    selected_pages: Sequence[RetrievedPage],
    audit: ConditionAuditResult,
    rerank_margin_threshold: float = 0.06,
    page_margin_threshold: float = 0.2,
    authority_strength_threshold: float = 0.95,
) -> SearchEscalationDecision:
    """Return whether the bounded grounding lane should escalate.

    Args:
        query: Raw user query.
        answer_type: Normalized answer type.
        scope: Current query-scope prediction.
        context_chunks: Answer-path reranked chunks.
        ordered_pages: Candidate pages in current score order.
        selected_pages: Current deterministic selected pages.
        audit: Typed support audit for the selected pages.
        rerank_margin_threshold: Margin below which rerank confidence is weak.
        page_margin_threshold: Margin below which page ordering is ambiguous.
        authority_strength_threshold: Minimum authority score treated as strong.

    Returns:
        SearchEscalationDecision: Ordered, bounded escalation contract.
    """

    allowed_family = _supported_family(query=query, answer_type=answer_type)
    if not allowed_family:
        return SearchEscalationDecision(
            should_escalate=False,
            reasons=(),
            allowed_family="",
            allow_shadow_rewrite=False,
            allow_relevance_verifier=False,
        )

    if audit.success and (
        len(ordered_pages) <= max(len(selected_pages), 1)
        or _selected_pages_are_strong(
            query=query,
            selected_pages=selected_pages,
            authority_strength_threshold=authority_strength_threshold,
        )
    ):
        return SearchEscalationDecision(
            should_escalate=False,
            reasons=(),
            allowed_family=allowed_family,
            allow_shadow_rewrite=True,
            allow_relevance_verifier=True,
        )

    reasons: list[str] = []
    if _low_rerank_margin(context_chunks, rerank_margin_threshold):
        reasons.append("low_rerank_confidence")
    if audit.failed_slots:
        reasons.append("typed_slots_missing")
    if scope.scope_mode is ScopeMode.COMPARE_PAIR and "compare_docs" in audit.failed_slots:
        reasons.append("compare_missing_side")
    if _law_family_is_ambiguous(query=query, ordered_pages=ordered_pages):
        reasons.append("law_family_ambiguous")
    if _top_pages_are_close_and_inconclusive(
        query=query,
        ordered_pages=ordered_pages,
        page_margin_threshold=page_margin_threshold,
        authority_strength_threshold=authority_strength_threshold,
    ):
        reasons.append("page_margin_close")
    if not _has_strong_authoritative_hit(
        query=query,
        ordered_pages=ordered_pages,
        authority_strength_threshold=authority_strength_threshold,
    ):
        reasons.append("no_strong_authoritative_hit")

    ordered_reasons = tuple(dict.fromkeys(reasons))
    return SearchEscalationDecision(
        should_escalate=bool(ordered_reasons),
        reasons=ordered_reasons,
        allowed_family=allowed_family,
        allow_shadow_rewrite=True,
        allow_relevance_verifier=True,
    )


def _supported_family(*, query: str, answer_type: str) -> str:
    """Return the bounded-family label supported by the new challenger.

    Args:
        query: Raw user query.
        answer_type: Normalized answer type.

    Returns:
        str: Supported family label or an empty string.
    """

    instruction = build_rerank_instruction(query, answer_type, doc_refs=())
    return instruction.family if instruction is not None else ""


def _low_rerank_margin(context_chunks: Sequence[RankedChunk], threshold: float) -> bool:
    """Return whether the answer-path rerank margin is too small.

    Args:
        context_chunks: Answer-path reranked chunks.
        threshold: Margin threshold.

    Returns:
        bool: True when the first-pass rerank confidence is weak.
    """

    if len(context_chunks) < 2:
        return False
    ranked_scores = sorted(
        [float(getattr(chunk, "rerank_score", 0.0) or 0.0) for chunk in context_chunks],
        reverse=True,
    )
    return (ranked_scores[0] - ranked_scores[1]) < threshold


def _law_family_is_ambiguous(*, query: str, ordered_pages: Sequence[RetrievedPage]) -> bool:
    """Return whether top pages imply conflicting law-family interpretations.

    Args:
        query: Raw user query.
        ordered_pages: Candidate pages in current order.

    Returns:
        bool: True when multiple law families compete near the top.
    """

    query_bundle = build_query_law_family_bundle(query)
    if not query_bundle.exact_keys:
        return False
    families = {
        page.canonical_law_family
        for page in ordered_pages[:3]
        if str(page.canonical_law_family).strip()
    }
    return len(families) >= 2


def _top_pages_are_close_and_inconclusive(
    *,
    query: str,
    ordered_pages: Sequence[RetrievedPage],
    page_margin_threshold: float,
    authority_strength_threshold: float,
) -> bool:
    """Return whether the top page ordering is close and authority-weak.

    Args:
        query: Raw user query.
        ordered_pages: Candidate pages in current order.
        page_margin_threshold: Minimum score gap treated as clear.
        authority_strength_threshold: Strong authority threshold.

    Returns:
        bool: True when ordering is close and no page is strongly authoritative.
    """

    if len(ordered_pages) < 2:
        return False
    top_page = ordered_pages[0]
    second_page = ordered_pages[1]
    score_margin = float(top_page.score) - float(second_page.score)
    if score_margin >= page_margin_threshold:
        return False
    top_authority = authority_signal_score(query, top_page, peer_pages=list(ordered_pages[:3]))
    second_authority = authority_signal_score(query, second_page, peer_pages=list(ordered_pages[:3]))
    return max(top_authority, second_authority) < authority_strength_threshold


def _has_strong_authoritative_hit(
    *,
    query: str,
    ordered_pages: Sequence[RetrievedPage],
    authority_strength_threshold: float,
) -> bool:
    """Return whether top candidates contain a strong authoritative hit.

    Args:
        query: Raw user query.
        ordered_pages: Candidate pages in current order.
        authority_strength_threshold: Minimum authority threshold.

    Returns:
        bool: True when at least one top page looks strongly authoritative.
    """

    for page in ordered_pages[:3]:
        if authority_signal_score(query, page, peer_pages=list(ordered_pages[:3])) >= authority_strength_threshold:
            return True
    return False


def _selected_pages_are_strong(
    *,
    query: str,
    selected_pages: Sequence[RetrievedPage],
    authority_strength_threshold: float,
) -> bool:
    """Return whether already-selected pages are strong enough to trust.

    Args:
        query: Raw user query.
        selected_pages: Current selected pages.
        authority_strength_threshold: Minimum authority threshold.

    Returns:
        bool: True when selected pages are already strong.
    """

    if not selected_pages:
        return False
    return all(
        authority_signal_score(query, page, peer_pages=list(selected_pages)) >= authority_strength_threshold
        for page in selected_pages
    )
