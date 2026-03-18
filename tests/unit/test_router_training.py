from __future__ import annotations

from rag_challenge.ml.external_grounding_data import NormalizedExternalRow
from rag_challenge.ml.router_training import build_external_router_examples


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
