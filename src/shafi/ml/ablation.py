"""Offline ablation helpers for grounding-model evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shafi.ml.grounding_dataset import GroundingMlRow
    from shafi.ml.training_scaffold import PageTrainingExample


@dataclass(frozen=True)
class SelectedPageMetrics:
    """Aggregated metrics for one offline page-selection lane.

    Args:
        supervised_question_count: Questions with usable positive-page supervision.
        overall_selected_hit_rate: Fraction of supervised questions selecting any positive page.
        weighted_selected_hit_rate: Weighted hit rate using supervision sample weights.
        soft_label_hit_rate: Hit rate on soft-AI-gold supervised questions.
        reviewed_hit_rate: Hit rate on reviewed supervised questions.
        compare_full_case_hit_rate: Hit rate on compare/full-case supervised questions.
        negative_unanswerable_zero_pages_rate: Fraction of negative/unanswerable rows selecting zero pages.
        average_selected_pages: Mean selected-page count over all rows.
        average_positive_recall: Mean fraction of positive pages covered on supervised rows.
    """

    supervised_question_count: int
    overall_selected_hit_rate: float
    weighted_selected_hit_rate: float
    soft_label_hit_rate: float
    reviewed_hit_rate: float
    compare_full_case_hit_rate: float
    negative_unanswerable_zero_pages_rate: float
    average_selected_pages: float
    average_positive_recall: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly metric payload.

        Returns:
            Serialized metric dictionary.
        """
        return asdict(self)


def compute_selected_page_metrics(
    rows: Sequence[GroundingMlRow],
    *,
    grouped_examples: dict[str, list[PageTrainingExample]],
    selected_pages_by_question: dict[str, list[str]],
) -> SelectedPageMetrics:
    """Compute offline lane metrics from selected page IDs.

    Args:
        rows: Exported grounding rows.
        grouped_examples: Supervised page examples grouped by question ID.
        selected_pages_by_question: Candidate lane outputs keyed by question ID.

    Returns:
        Aggregated lane metrics.
    """
    supervised_question_count = 0
    overall_hits = 0
    weighted_hits = 0.0
    weighted_total = 0.0
    soft_hits = 0
    soft_total = 0
    reviewed_hits = 0
    reviewed_total = 0
    compare_hits = 0
    compare_total = 0
    positive_recall_total = 0.0
    average_selected_pages_total = 0
    negative_zero_hits = 0
    negative_zero_total = 0

    for row in rows:
        selected_pages = selected_pages_by_question.get(row.question_id, [])
        average_selected_pages_total += len(selected_pages)

        if row.scope_mode == "negative_unanswerable":
            negative_zero_total += 1
            if not selected_pages:
                negative_zero_hits += 1

        group = grouped_examples.get(row.question_id, [])
        if not group:
            continue

        supervised_question_count += 1
        positive_pages = {example.page_id for example in group if example.label == 1}
        if not positive_pages:
            continue

        selected_set = set(selected_pages)
        is_hit = bool(selected_set & positive_pages)
        if is_hit:
            overall_hits += 1
        positive_recall_total += len(selected_set & positive_pages) / float(len(positive_pages))

        supervision_source = group[0].supervision_source
        sample_weight = group[0].sample_weight
        weighted_total += sample_weight
        if is_hit:
            weighted_hits += sample_weight

        if supervision_source == "soft_ai_gold":
            soft_total += 1
            if is_hit:
                soft_hits += 1
        if supervision_source == "reviewed":
            reviewed_total += 1
            if is_hit:
                reviewed_hits += 1
        if row.scope_mode in {"compare_pair", "full_case_files"}:
            compare_total += 1
            if is_hit:
                compare_hits += 1

    total_rows = len(rows)
    return SelectedPageMetrics(
        supervised_question_count=supervised_question_count,
        overall_selected_hit_rate=overall_hits / supervised_question_count if supervised_question_count else 0.0,
        weighted_selected_hit_rate=weighted_hits / weighted_total if weighted_total else 0.0,
        soft_label_hit_rate=soft_hits / soft_total if soft_total else 0.0,
        reviewed_hit_rate=reviewed_hits / reviewed_total if reviewed_total else 0.0,
        compare_full_case_hit_rate=compare_hits / compare_total if compare_total else 0.0,
        negative_unanswerable_zero_pages_rate=negative_zero_hits / negative_zero_total if negative_zero_total else 0.0,
        average_selected_pages=average_selected_pages_total / total_rows if total_rows else 0.0,
        average_positive_recall=positive_recall_total / supervised_question_count if supervised_question_count else 0.0,
    )


def filter_rows_by_question_ids(
    rows: Sequence[GroundingMlRow],
    *,
    question_ids: set[str],
) -> list[GroundingMlRow]:
    """Filter exported rows to a reviewed question-ID slice.

    Args:
        rows: Candidate export rows.
        question_ids: Allowed question IDs for the target slice.

    Returns:
        Stable filtered row list.
    """
    if not question_ids:
        return []
    return [row for row in rows if row.question_id in question_ids]
