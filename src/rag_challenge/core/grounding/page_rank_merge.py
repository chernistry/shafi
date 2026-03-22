"""Helpers for merging trained page rankings into heuristic page order."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence


def ordered_rankable_candidate_page_ids(
    *,
    heuristic_order: Sequence[str],
    candidate_page_ids: Collection[str],
) -> list[str]:
    """Return candidate page IDs in heuristic order.

    Args:
        heuristic_order: Page IDs ordered by the heuristic selector.
        candidate_page_ids: Candidate page IDs that the trained scorer may reorder.

    Returns:
        Candidate page IDs that appear in the heuristic ordering.
    """
    candidate_set = set(candidate_page_ids)
    return [page_id for page_id in heuristic_order if page_id in candidate_set]


def merge_ranked_candidate_subset(
    *,
    heuristic_order: Sequence[str],
    ranked_candidate_page_ids: Sequence[str],
    candidate_page_ids: Collection[str],
) -> list[str] | None:
    """Merge a ranked candidate subset back into heuristic order.

    The trained scorer is only allowed to reorder the candidate-page slots already
    present in the heuristic ordering. Any non-candidate pages keep their original
    relative positions so the runtime path stays fail-closed.

    Args:
        heuristic_order: Full heuristic page ordering.
        ranked_candidate_page_ids: Candidate pages ranked by the trained scorer.
        candidate_page_ids: Candidate pages that may be replaced in the heuristic order.

    Returns:
        Full page ordering with candidate slots replaced by the ranked subset, or
        ``None`` when the ranked subset is invalid.
    """
    rankable_candidate_ids = ordered_rankable_candidate_page_ids(
        heuristic_order=heuristic_order,
        candidate_page_ids=candidate_page_ids,
    )
    if len(rankable_candidate_ids) != len(ranked_candidate_page_ids):
        return None
    if len(set(ranked_candidate_page_ids)) != len(ranked_candidate_page_ids):
        return None
    if set(rankable_candidate_ids) != set(ranked_candidate_page_ids):
        return None

    candidate_set = set(rankable_candidate_ids)
    ranked_iter = iter(ranked_candidate_page_ids)
    merged: list[str] = []
    for page_id in heuristic_order:
        if page_id in candidate_set:
            merged.append(next(ranked_iter))
            continue
        merged.append(page_id)

    try:
        next(ranked_iter)
    except StopIteration:
        return merged
    return None
