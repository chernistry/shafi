from __future__ import annotations

from scripts.run_grounding_sidecar_ablation import (
    _current_sidecar_budget_map,
    _pad_role_prediction_rows,
    _unseen_role_labels,
)

from rag_challenge.ml.ablation import compute_selected_page_metrics, filter_rows_by_question_ids
from rag_challenge.ml.grounding_dataset import (
    DocCandidateRecord,
    GroundingMlRow,
    PageCandidateRecord,
    PageRetrievalFeatureRecord,
    SupportFactFeatureRecord,
)
from rag_challenge.ml.training_scaffold import PageTrainingExample


def _row(question_id: str, *, scope_mode: str = "single_field_single_doc") -> GroundingMlRow:
    return GroundingMlRow(
        question_id=question_id,
        question=f"Question {question_id}",
        answer_type="name",
        golden_answer="answer",
        label_page_ids=[],
        label_source="suspect_ai_gold",
        label_trust_tier="",
        scope_mode=scope_mode,
        target_page_roles=["article_clause"],
        hard_anchor_strings=["Article 1"],
        doc_candidates=[
            DocCandidateRecord(
                doc_id="doc",
                page_candidate_count=2,
                candidate_sources=["legacy_context"],
            )
        ],
        page_candidates=[
            PageCandidateRecord(page_id=f"{question_id}_1", doc_id="doc", page_num=1),
            PageCandidateRecord(page_id=f"{question_id}_2", doc_id="doc", page_num=2),
        ],
        legacy_selected_pages=[],
        sidecar_selected_pages=[],
        support_fact_features=SupportFactFeatureRecord(),
        page_retrieval_features=PageRetrievalFeatureRecord(),
        label_is_suspect=True,
        source_paths={},
    )


def _example(question_id: str, page_id: str, label: int, *, source: str, weight: float) -> PageTrainingExample:
    return PageTrainingExample(
        question_id=question_id,
        page_id=page_id,
        features={},
        label=label,
        sample_weight=weight,
        supervision_source=source,
    )


def test_compute_selected_page_metrics_reports_hits_and_zero_negative_pages() -> None:
    rows = [
        _row("q1"),
        _row("q2", scope_mode="compare_pair"),
        _row("q3", scope_mode="negative_unanswerable"),
    ]
    grouped_examples = {
        "q1": [
            _example("q1", "q1_1", 1, source="soft_ai_gold", weight=1.5),
            _example("q1", "q1_2", 0, source="soft_ai_gold", weight=1.5),
        ],
        "q2": [
            _example("q2", "q2_1", 0, source="reviewed", weight=3.0),
            _example("q2", "q2_2", 1, source="reviewed", weight=3.0),
        ],
    }
    selected_pages = {
        "q1": ["q1_1"],
        "q2": ["q2_2"],
        "q3": [],
    }

    metrics = compute_selected_page_metrics(
        rows,
        grouped_examples=grouped_examples,
        selected_pages_by_question=selected_pages,
    )

    assert metrics.supervised_question_count == 2
    assert metrics.overall_selected_hit_rate == 1.0
    assert metrics.weighted_selected_hit_rate == 1.0
    assert metrics.soft_label_hit_rate == 1.0
    assert metrics.reviewed_hit_rate == 1.0
    assert metrics.compare_full_case_hit_rate == 1.0
    assert metrics.negative_unanswerable_zero_pages_rate == 1.0
    assert metrics.average_selected_pages == 2 / 3
    assert metrics.average_positive_recall == 1.0


def test_filter_rows_by_question_ids_keeps_stable_order() -> None:
    rows = [_row("q1"), _row("q2"), _row("q3")]

    filtered = filter_rows_by_question_ids(rows, question_ids={"q3", "q1"})

    assert [row.question_id for row in filtered] == ["q1", "q3"]


def test_ablation_role_helpers_preserve_unseen_eval_labels() -> None:
    unseen = _unseen_role_labels(
        role_targets=[["article_clause", "costs_block"], ["operative_order"]],
        trained_role_labels=["article_clause"],
    )
    padded = _pad_role_prediction_rows(
        rows=[[1], [0]],
        row_count=2,
        unseen_label_count=len(unseen),
    )

    assert unseen == ["costs_block", "operative_order"]
    assert padded == [[1, 0, 0], [0, 0, 0]]


def test_current_sidecar_budget_map_tracks_existing_page_budget() -> None:
    rows = [
        _row("q1"),
        _row("q2"),
        _row("q3", scope_mode="negative_unanswerable"),
    ]
    rows[0].sidecar_selected_pages = ["q1_1", "q1_2"]
    rows[1].legacy_selected_pages = ["q2_2"]

    budgets = _current_sidecar_budget_map(rows)

    assert budgets == {"q1": 2, "q2": 1, "q3": 0}
