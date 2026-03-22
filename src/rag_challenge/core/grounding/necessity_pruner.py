"""Counterfactual necessity pruning for grounding page sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_challenge.core.grounding.condition_audit import audit_candidate_pages
from rag_challenge.models.schemas import ScopeMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models.schemas import RetrievedPage


def prune_redundant_pages(
    *,
    query: str,
    answer_type: str,
    scope_mode: ScopeMode,
    ordered_pages: Sequence[RetrievedPage],
    page_budget: int = 2,
) -> tuple[str, ...]:
    """Prune pages that are not necessary for typed support coverage.

    Args:
        query: Raw user question.
        answer_type: Normalized answer type.
        scope_mode: Sidecar scope mode for the query.
        ordered_pages: Candidate pages in current preferred order.

    Returns:
        tuple[str, ...]: Minimal ordered page IDs that preserve typed coverage.
    """

    if len(ordered_pages) <= 1:
        return tuple(page.page_id for page in ordered_pages if page.page_id)

    kept_pages = [page for page in ordered_pages if page.page_id]
    baseline_audit = audit_candidate_pages(
        query=query,
        answer_type=answer_type,
        scope_mode=scope_mode,
        pages=kept_pages,
    )
    if not baseline_audit.success:
        return tuple(page.page_id for page in kept_pages)

    # F-beta 2.5 recall optimization: recall weighted 6.25x over precision.
    # Missing 1 gold page costs ~46% G; adding 1 wrong page costs ~6.5%.
    # Pruning correct pages is catastrophic for F2.5. Protect ALL selected pages.
    # Organizers confirmed: "indicate the most appropriate and COMPLETE sources."
    protected_count = max(page_budget, len(kept_pages))
    for index in range(len(kept_pages) - 1, -1, -1):
        if len(kept_pages) <= protected_count:
            break
        candidate_pages = kept_pages[:index] + kept_pages[index + 1 :]
        candidate_audit = audit_candidate_pages(
            query=query,
            answer_type=answer_type,
            scope_mode=scope_mode,
            pages=candidate_pages,
        )
        if candidate_audit.success:
            kept_pages = candidate_pages

    return tuple(page.page_id for page in kept_pages if page.page_id)
