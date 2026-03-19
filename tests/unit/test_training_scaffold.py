from __future__ import annotations

from rag_challenge.ml.grounding_dataset import (
    DocCandidateRecord,
    GroundingMlRow,
    PageCandidateRecord,
    PageRetrievalFeatureRecord,
    SupportFactFeatureRecord,
)
from rag_challenge.ml.training_scaffold import (
    build_page_training_examples,
    build_router_dataset,
    deterministic_subset,
    internal_row_sample_weight,
)


def _build_row(
    *,
    question_id: str,
    answer_type: str = "name",
    scope_mode: str = "single_field_single_doc",
    label_source: str = "soft_ai_gold",
    label_confidence: str = "",
    label_weight: float = 0.0,
    label_page_ids: list[str] | None = None,
    legacy_selected_pages: list[str] | None = None,
    sidecar_selected_pages: list[str] | None = None,
) -> GroundingMlRow:
    return GroundingMlRow(
        question_id=question_id,
        question=f"Question {question_id}",
        answer_type=answer_type,
        golden_answer="annual return",
        label_page_ids=list(label_page_ids or []),
        label_source=label_source,
        label_confidence=label_confidence,
        label_weight=label_weight,
        scope_mode=scope_mode,
        target_page_roles=["article_clause"],
        hard_anchor_strings=["Article 16"],
        doc_candidates=[
            DocCandidateRecord(
                doc_id="law",
                page_candidate_count=2,
                candidate_sources=["legacy_context"],
                legacy_selected=True,
                sidecar_selected=True,
            )
        ],
        page_candidates=[
            PageCandidateRecord(
                page_id="law_16",
                doc_id="law",
                page_num=16,
                candidate_sources=["legacy_used", "sidecar_used"],
                legacy_context_rank=1,
                sidecar_context_rank=1,
                anchor_hits=["Article 16"],
                snippet_excerpt="Article 16 requires the annual return.",
            ),
            PageCandidateRecord(
                page_id="law_17",
                doc_id="law",
                page_num=17,
                candidate_sources=["legacy_context"],
                legacy_context_rank=2,
                snippet_excerpt="Article 17 covers another obligation.",
            ),
        ],
        legacy_selected_pages=list(legacy_selected_pages or ["law_16"]),
        sidecar_selected_pages=list(sidecar_selected_pages or []),
        support_fact_features=SupportFactFeatureRecord(doc_ref_count=1, target_page_roles_count=1, explicit_anchor_count=1),
        page_retrieval_features=PageRetrievalFeatureRecord(
            legacy_retrieved_page_count=2,
            legacy_context_page_count=2,
            legacy_sidecar_used_overlap_count=1,
        ),
        label_is_suspect=label_source == "suspect_ai_gold",
        source_paths={},
    )


def test_deterministic_subset_is_stable() -> None:
    rows = [_build_row(question_id=f"qid-{index}") for index in range(8)]

    subset_a = deterministic_subset(rows, limit=3, seed=610)
    subset_b = deterministic_subset(rows, limit=3, seed=610)

    assert [row.question_id for row in subset_a] == [row.question_id for row in subset_b]
    assert len(subset_a) == 3


def test_build_router_dataset_derives_zero_budget_for_negative_queries() -> None:
    rows = [
        _build_row(question_id="q-positive", sidecar_selected_pages=["law_16"]),
        _build_row(
            question_id="q-negative",
            answer_type="boolean",
            scope_mode="negative_unanswerable",
            label_source="suspect_ai_gold",
            label_page_ids=[],
            legacy_selected_pages=[],
            sidecar_selected_pages=[],
        ),
    ]

    dataset = build_router_dataset(rows)

    assert dataset.scope_targets == ["single_field_single_doc", "negative_unanswerable"]
    assert dataset.page_budget_targets == [1, 0]
    assert dataset.role_targets[0] == ["article_clause"]


def test_page_training_examples_prefer_reviewed_labels() -> None:
    row = _build_row(
        question_id="qid-reviewed",
        label_source="reviewed",
        label_confidence="high",
        label_weight=1.0,
        label_page_ids=["law_16"],
        legacy_selected_pages=["law_17"],
        sidecar_selected_pages=["law_17"],
    )

    examples = build_page_training_examples([row], label_mode="all")

    assert len(examples) == 2
    positive = next(example for example in examples if example.page_id == "law_16")
    negative = next(example for example in examples if example.page_id == "law_17")
    assert positive.label == 1
    assert positive.sample_weight == 3.0
    assert positive.supervision_source == "reviewed"
    assert positive.features["from_legacy_used"] is True
    assert positive.features["doc_selected_by_legacy"] is True
    assert positive.features["explicit_anchor_count"] == 1
    assert negative.label == 0


def test_page_training_examples_skip_unreviewed_rows_in_reviewed_only_mode() -> None:
    row = _build_row(
        question_id="qid-suspect",
        label_source="suspect_ai_gold",
        label_page_ids=[],
        legacy_selected_pages=["law_16"],
        sidecar_selected_pages=["law_16"],
    )

    examples = build_page_training_examples([row], label_mode="reviewed_only")

    assert examples == []


def test_page_training_examples_weight_reviewed_medium_rows() -> None:
    row = _build_row(
        question_id="qid-medium",
        label_source="reviewed",
        label_confidence="medium",
        label_weight=0.5,
        label_page_ids=["law_16"],
    )

    examples = build_page_training_examples([row], label_mode="reviewed_weighted")

    positive = next(example for example in examples if example.page_id == "law_16")
    assert positive.sample_weight == 1.5
    assert positive.supervision_source == "reviewed"


def test_page_training_examples_drop_medium_rows_in_high_confidence_mode() -> None:
    row = _build_row(
        question_id="qid-medium",
        label_source="reviewed",
        label_confidence="medium",
        label_weight=0.5,
        label_page_ids=["law_16"],
    )

    examples = build_page_training_examples([row], label_mode="reviewed_high_confidence")

    assert examples == []


def test_internal_row_sample_weight_uses_reviewed_confidence_modes() -> None:
    high = _build_row(
        question_id="qid-high",
        label_source="reviewed",
        label_confidence="high",
        label_weight=1.0,
    )
    medium = _build_row(
        question_id="qid-medium",
        label_source="reviewed",
        label_confidence="medium",
        label_weight=0.5,
    )
    low = _build_row(
        question_id="qid-low",
        label_source="reviewed",
        label_confidence="low",
        label_weight=0.0,
    )

    assert internal_row_sample_weight(high, label_mode="reviewed_high_confidence") == 1.0
    assert internal_row_sample_weight(medium, label_mode="reviewed_high_confidence") == 0.0
    assert internal_row_sample_weight(medium, label_mode="reviewed_weighted") == 0.5
    assert internal_row_sample_weight(low, label_mode="reviewed_weighted") == 0.0
