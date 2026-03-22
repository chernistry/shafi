from __future__ import annotations

import json

from rag_challenge.ml.grounding_dataset import (
    DocCandidateRecord,
    GroundingMlRow,
    PageCandidateRecord,
    PageRetrievalFeatureRecord,
    SupportFactFeatureRecord,
)
from rag_challenge.ml.hard_negative_miner import HardNegativeMiner
from rag_challenge.ml.synthetic_qa_factory import BridgeFactRecord, QuestionFamily, SyntheticQAExample
from rag_challenge.ml.teacher_labels import (
    TeacherLabelBuilder,
    build_page_texts_from_registry,
    write_pairwise_labels_jsonl,
    write_training_triples_jsonl,
)
from rag_challenge.models import CaseObject, CaseParty, CorpusRegistry, LawObject, LegalDocType


def _grounding_row() -> GroundingMlRow:
    return GroundingMlRow(
        question_id="q-1",
        question="What does Article 5 of Employment Law provide?",
        answer_type="free_text",
        golden_answer="notice before termination",
        label_page_ids=["law-a_2"],
        label_source="reviewed",
        label_confidence="high",
        label_weight=1.0,
        scope_mode="single_field_single_doc",
        target_page_roles=["article_clause"],
        hard_anchor_strings=["Article 5"],
        doc_candidates=[DocCandidateRecord(doc_id="law-a", page_candidate_count=2)],
        page_candidates=[
            PageCandidateRecord(
                page_id="law-a_2",
                doc_id="law-a",
                page_num=2,
                snippet_excerpt="Article 5 requires notice before termination.",
            ),
            PageCandidateRecord(
                page_id="law-a_1",
                doc_id="law-a",
                page_num=1,
                snippet_excerpt="Employment Law 2020 issued by DIFC Authority.",
            ),
        ],
        legacy_selected_pages=["law-a_2"],
        sidecar_selected_pages=[],
        support_fact_features=SupportFactFeatureRecord(doc_ref_count=1, explicit_anchor_count=1, target_page_roles_count=1),
        page_retrieval_features=PageRetrievalFeatureRecord(legacy_context_page_count=2),
        source_paths={},
    )


def _registry() -> CorpusRegistry:
    return CorpusRegistry(
        laws={
            "law-a": LawObject(
                object_id="law:law-a",
                doc_id="law-a",
                title="Employment Law 2020",
                short_title="Employment Law",
                legal_doc_type=LegalDocType.LAW,
                issuing_authority="DIFC Authority",
                commencement_date="1 January 2020",
                page_ids=["law-a_1", "law-a_2"],
                page_texts={
                    "law-a_1": "Employment Law 2020. Issued by DIFC Authority.",
                    "law-a_2": "Article 5 requires notice before termination.",
                },
            ),
            "law-b": LawObject(
                object_id="law:law-b",
                doc_id="law-b",
                title="Employment Amendment Law 2021",
                short_title="Employment Amendment Law",
                legal_doc_type=LegalDocType.LAW,
                page_ids=["law-b_1"],
                page_texts={"law-b_1": "Employment Amendment Law 2021 transitional obligations."},
            ),
        },
        cases={
            "case-a": CaseObject(
                object_id="case:case-a",
                doc_id="case-a",
                title="Alice Smith v Registrar",
                legal_doc_type=LegalDocType.CASE,
                case_number="CFI 002/2021",
                judges=["Justice Patel"],
                parties=[CaseParty(name="Alice Smith", role="claimant")],
                page_ids=["case-a_1"],
                page_texts={"case-a_1": "Alice Smith v Registrar. Justice Patel."},
            )
        },
    )


def _builder() -> TeacherLabelBuilder:
    page_texts, page_doc_ids, aliases_by_doc_id = build_page_texts_from_registry(_registry())
    miner = HardNegativeMiner(page_texts=page_texts, page_doc_ids=page_doc_ids, aliases_by_doc_id=aliases_by_doc_id)
    return TeacherLabelBuilder(miner=miner, page_texts=page_texts, page_doc_ids=page_doc_ids)


def test_build_from_grounding_rows_uses_reviewed_positive_pages() -> None:
    triples = _builder().build_from_grounding_rows([_grounding_row()])

    assert triples
    assert triples[0].positive_page_id == "law-a_2"
    assert triples[0].source == "grounding:reviewed"


def test_build_from_synthetic_and_bridge_fact_sources() -> None:
    synthetic = SyntheticQAExample(
        question_id="syn-1",
        question="Who issued Employment Law?",
        answer="DIFC Authority",
        answer_type="name",
        gold_page_ids=["law-a_1"],
        gold_doc_ids=["law-a"],
        question_family=QuestionFamily.FACTOID_AUTHORITY,
        difficulty="easy",
        hard_negative_page_ids=["law-b_1"],
        source_object_ids=["law:law-a"],
        generation_method="law_authority",
    )
    bridge = BridgeFactRecord(
        fact_id="bf-1",
        question="Which law mentions transitional obligations?",
        answer="Employment Amendment Law 2021",
        page_ids=["law-b_1"],
        doc_ids=["law-b"],
    )

    builder = _builder()
    synthetic_triples = builder.build_from_synthetic_qa([synthetic])
    bridge_triples = builder.build_from_bridge_facts([bridge])

    assert synthetic_triples[0].negative_page_id == "law-b_1"
    assert bridge_triples[0].source == "bridge_fact"


def test_build_from_compiled_registry_and_denoise() -> None:
    builder = _builder()
    triples = builder.build_from_compiled_registry(_registry())
    noisy = [
        triples[0],
        triples[0].__class__(
            query=triples[0].query,
            positive_text="Employment Law 2020",
            positive_page_id="law-a_1",
            negative_text="Employment Law 2020 issued by DIFC Authority",
            negative_page_id="law-a_1-copy",
            source="compiled:law",
            difficulty="easy",
            strategy="lexical_near_miss",
        ),
    ]

    denoised = builder.denoise_false_negatives(noisy)
    assert len(denoised) == 1


def test_statistics_and_exports(tmp_path) -> None:
    builder = _builder()
    triples = builder.combine_and_deduplicate(
        [
            builder.build_from_grounding_rows([_grounding_row()]),
            builder.build_from_compiled_registry(_registry()),
        ]
    )
    stats = builder.statistics(triples)
    triples_path = tmp_path / "triples.jsonl"
    pairwise_path = tmp_path / "pairwise.jsonl"
    write_training_triples_jsonl(triples_path, triples)
    write_pairwise_labels_jsonl(pairwise_path, triples)

    assert stats.total_triples == len(triples)
    assert json.loads(triples_path.read_text(encoding="utf-8").splitlines()[0])["query"]
    pairwise_rows = pairwise_path.read_text(encoding="utf-8").splitlines()
    assert len(pairwise_rows) == len(triples) * 2
