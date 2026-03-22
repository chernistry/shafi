from __future__ import annotations

from scripts.train_grounding_router import _build_role_prediction_matrix, _unseen_dev_role_labels

from rag_challenge.ml.external_grounding_data import NormalizedExternalRow
from rag_challenge.ml.grounding_dataset import (
    DocCandidateRecord,
    GroundingMlRow,
    PageCandidateRecord,
    PageRetrievalFeatureRecord,
    SupportFactFeatureRecord,
)
from rag_challenge.ml.router_training import (
    RouterTrainingExample,
    build_external_router_examples,
    build_internal_router_examples,
)


def _internal_row(
    *,
    question_id: str,
    confidence: str,
    label_weight: float,
) -> GroundingMlRow:
    return GroundingMlRow(
        question_id=question_id,
        question=f"Question {question_id}",
        answer_type="name",
        golden_answer="answer",
        label_page_ids=["doc_1"],
        label_source="reviewed",
        label_confidence=confidence,
        label_weight=label_weight,
        scope_mode="single_field_single_doc",
        target_page_roles=["article_clause"],
        hard_anchor_strings=["Article 1"],
        doc_candidates=[DocCandidateRecord(doc_id="doc", page_candidate_count=1, candidate_sources=["legacy_used"])],
        page_candidates=[PageCandidateRecord(page_id="doc_1", doc_id="doc", page_num=1)],
        legacy_selected_pages=["doc_1"],
        sidecar_selected_pages=["doc_1"],
        support_fact_features=SupportFactFeatureRecord(),
        page_retrieval_features=PageRetrievalFeatureRecord(),
        label_is_suspect=False,
        source_paths={},
    )


def test_build_external_router_examples_maps_obliqa_to_article_clause() -> None:
    rows = [
        NormalizedExternalRow(
            source_dataset="obliqa",
            sample_id="ob-1",
            text="A firm must keep records.",
            question="What must the firm do?",
            label_type="support_scope",
            role_label="",
            scope_label="single_field_single_doc",
            support_label="supported",
            metadata_json='{"passage_count": 1}',
        )
    ]

    examples = build_external_router_examples(rows, sample_weight=0.25)

    assert len(examples) == 1
    assert examples[0].scope_target == "single_field_single_doc"
    assert examples[0].page_budget_target == 1
    assert examples[0].role_targets == ["article_clause"]
    assert examples[0].sample_weight == 0.25


def test_build_external_router_examples_maps_ledgar_costs_to_costs_block() -> None:
    rows = [
        NormalizedExternalRow(
            source_dataset="ledgar",
            sample_id="led-1",
            text="The borrower shall pay all fees and expenses.",
            question="Which clause role best matches this provision?",
            label_type="role_label",
            role_label="fees_and_expenses",
            scope_label="single_field_single_doc",
            support_label="role_supervision",
            metadata_json='{"split": "train"}',
        )
    ]

    examples = build_external_router_examples(rows, sample_weight=0.25)

    assert examples[0].page_budget_target == 1
    assert "costs_block" in examples[0].role_targets


def test_build_external_router_examples_skips_unsupported_scope_only_rows() -> None:
    rows = [
        NormalizedExternalRow(
            source_dataset="obliqa",
            sample_id="ob-2",
            text="A firm must do A and B.",
            question="What must the firm do when two obligations apply?",
            label_type="support_scope",
            role_label="",
            scope_label="multi_passage_support",
            support_label="supported",
            metadata_json='{"passage_count": 2}',
        )
    ]

    examples = build_external_router_examples(rows, sample_weight=0.25)

    assert len(examples) == 1
    assert examples[0].scope_target is None
    assert examples[0].page_budget_target == 2


def test_build_internal_router_examples_filters_to_high_confidence_when_requested() -> None:
    rows = [
        _internal_row(question_id="high", confidence="high", label_weight=1.0),
        _internal_row(question_id="medium", confidence="medium", label_weight=0.5),
    ]

    examples = build_internal_router_examples(rows, label_mode="reviewed_high_confidence")

    assert [example.sample_id for example in examples] == ["high"]
    assert examples[0].sample_weight == 1.0


def test_build_internal_router_examples_preserve_reviewed_weights() -> None:
    rows = [
        _internal_row(question_id="high", confidence="high", label_weight=1.0),
        _internal_row(question_id="medium", confidence="medium", label_weight=0.5),
    ]

    examples = build_internal_router_examples(rows, label_mode="reviewed_weighted")

    assert [example.sample_weight for example in examples] == [1.0, 0.5]


def test_router_eval_helpers_pad_unseen_dev_roles_with_zero_predictions() -> None:
    dev_examples = [
        RouterTrainingExample(
            sample_id="high",
            text="question: costs",
            scope_target="single_field_single_doc",
            page_budget_target=1,
            role_targets=["article_clause", "costs_block"],
            sample_weight=1.0,
            source="internal",
        )
    ]

    unseen = _unseen_dev_role_labels(dev_examples=dev_examples, train_role_labels=["article_clause"])
    prediction_rows = _build_role_prediction_matrix(
        prediction_columns=[[1]],
        row_count=1,
        unseen_label_count=len(unseen),
    )

    assert unseen == ["costs_block"]
    assert prediction_rows == [[1, 0]]
