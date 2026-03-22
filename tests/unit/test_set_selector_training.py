from __future__ import annotations

from rag_challenge.ml.grounding_dataset import (
    DocCandidateRecord,
    GroundingMlRow,
    PageCandidateRecord,
    PageRetrievalFeatureRecord,
    SupportFactFeatureRecord,
)
from rag_challenge.ml.set_selector_training import (
    SetSelectorConfig,
    SetSelectorTrainer,
    TrainingTriple,
)


def _triple() -> TrainingTriple:
    return TrainingTriple(
        query="Who issued Employment Law?",
        positive_text="Employment Law 2020 was issued by DIFC Authority.",
        positive_page_id="law-a_1",
        negative_text="Alice Smith v Registrar. Justice Patel.",
        negative_page_id="case-a_1",
        source="compiled:law",
        difficulty="easy",
        strategy="cross_doc",
    )


def _row() -> GroundingMlRow:
    return GroundingMlRow(
        question_id="q-1",
        question="Who issued Employment Law?",
        answer_type="name",
        golden_answer="DIFC Authority",
        label_page_ids=["law-a_1"],
        label_source="reviewed",
        label_confidence="high",
        label_weight=1.0,
        scope_mode="single_field_single_doc",
        target_page_roles=["title_page"],
        hard_anchor_strings=["Employment Law"],
        doc_candidates=[DocCandidateRecord(doc_id="law-a", page_candidate_count=2)],
        page_candidates=[
            PageCandidateRecord(
                page_id="law-a_1",
                doc_id="law-a",
                page_num=1,
                snippet_excerpt="Employment Law 2020 was issued by DIFC Authority.",
            ),
            PageCandidateRecord(
                page_id="law-b_1",
                doc_id="law-b",
                page_num=1,
                snippet_excerpt="Employment Amendment Law 2021 transitional obligations.",
            ),
            PageCandidateRecord(
                page_id="case-a_1",
                doc_id="case-a",
                page_num=1,
                snippet_excerpt="Alice Smith v Registrar. Justice Patel.",
            ),
        ],
        legacy_selected_pages=["law-a_1"],
        sidecar_selected_pages=["law-a_1"],
        support_fact_features=SupportFactFeatureRecord(doc_ref_count=1, explicit_anchor_count=1, target_page_roles_count=1),
        page_retrieval_features=PageRetrievalFeatureRecord(legacy_context_page_count=3),
        source_paths={},
    )


def test_prepare_pairwise_data_creates_positive_and_negative_rows() -> None:
    trainer = SetSelectorTrainer()
    labels = trainer.prepare_pairwise_data([_triple()])

    assert len(labels) == 2
    assert labels[0].label == 1
    assert labels[1].label == 0


def test_prepare_set_utility_data_uses_grounding_rows() -> None:
    trainer = SetSelectorTrainer()
    labels = trainer.prepare_set_utility_data([_row()])

    assert len(labels) == 1
    assert labels[0].selected_pages == ["law-a_1"]
    assert labels[0].answer_coverage == 1.0


def test_train_and_select_compact_evidence_set(tmp_path) -> None:
    trainer = SetSelectorTrainer()
    pairwise = trainer.prepare_pairwise_data([_triple()])
    artifact_path = trainer.train_cross_encoder(
        base_model="deterministic-cross-encoder-r1",
        pairwise_data=pairwise,
        config=SetSelectorConfig(min_set_size=1, max_set_size=2),
        output_dir=tmp_path / "selector",
    )
    selected = trainer.select_compact_evidence_set(
        query="Who issued Employment Law?",
        candidate_page_texts={
            "law-a_1": "Employment Law 2020 was issued by DIFC Authority.",
            "law-b_1": "Employment Amendment Law 2021 transitional obligations.",
            "case-a_1": "Alice Smith v Registrar. Justice Patel.",
        },
        artifact_path=artifact_path,
    )

    assert selected[0] == "law-a_1"
    assert len(selected) <= 2


def test_evaluate_selector_computes_metrics(tmp_path) -> None:
    trainer = SetSelectorTrainer()
    pairwise = trainer.prepare_pairwise_data([_triple()])
    artifact_path = trainer.train_cross_encoder(
        base_model="deterministic-cross-encoder-r1",
        pairwise_data=pairwise,
        config=SetSelectorConfig(min_set_size=1, max_set_size=2),
        output_dir=tmp_path / "selector",
    )
    labels = trainer.prepare_set_utility_data([_row()])
    metrics = trainer.evaluate_selector(artifact_path=artifact_path, eval_labels=labels)

    assert metrics.page_recall == 1.0
    assert metrics.set_utility_rate == 1.0
