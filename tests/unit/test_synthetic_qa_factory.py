from __future__ import annotations

import json

from shafi.ml.synthetic_qa_factory import (
    QuestionFamily,
    SyntheticQAFactory,
    build_manifest,
    load_bridge_fact_records,
    mine_hard_negative_page_ids,
)
from shafi.models import (
    ApplicabilityEdge,
    ApplicabilityEdgeType,
    ApplicabilityGraph,
    CaseObject,
    CaseParty,
    CorpusRegistry,
    DocType,
    LawObject,
    LegalDocType,
    LegalSegment,
    SegmentType,
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
                    "law-a_1": "Employment Law 2020. Issued by DIFC Authority. Comes into force on 1 January 2020.",
                    "law-a_2": "Article 5. Employees must receive notice before termination.",
                },
            ),
            "law-b": LawObject(
                object_id="law:law-b",
                doc_id="law-b",
                title="Employment Amendment Law 2021",
                short_title="Employment Amendment Law",
                legal_doc_type=LegalDocType.LAW,
                issuing_authority="DIFC Authority",
                commencement_date="1 January 2021",
                page_ids=["law-b_1"],
                page_texts={"law-b_1": "Employment Amendment Law 2021. Issued by DIFC Authority."},
            ),
        },
        cases={
            "case-a": CaseObject(
                object_id="case:case-a",
                doc_id="case-a",
                title="Smith v Jones",
                legal_doc_type=LegalDocType.CASE,
                case_number="CFI 001/2020",
                judges=["Justice Smith"],
                parties=[
                    CaseParty(name="Alice Smith", role="claimant"),
                    CaseParty(name="Bob Jones", role="respondent"),
                ],
                page_ids=["case-a_1"],
                page_texts={"case-a_1": "CFI 001/2020. Alice Smith v Bob Jones. Justice Smith."},
            ),
            "case-b": CaseObject(
                object_id="case:case-b",
                doc_id="case-b",
                title="Alice Smith v Registrar",
                legal_doc_type=LegalDocType.CASE,
                case_number="CFI 002/2021",
                judges=["Justice Patel"],
                parties=[CaseParty(name="Alice Smith", role="claimant")],
                page_ids=["case-b_1"],
                page_texts={"case-b_1": "CFI 002/2021. Alice Smith v Registrar. Justice Patel."},
            ),
        },
    )


def _segments() -> list[LegalSegment]:
    return [
        LegalSegment(
            segment_id="law-a:article-5",
            segment_type=SegmentType.ARTICLE,
            doc_id="law-a",
            doc_title="Employment Law 2020",
            doc_type=DocType.STATUTE,
            legal_path="Article 5",
            text="Article 5. Employees must receive notice before termination.",
            page_ids=["law-a_2"],
            start_page=2,
            end_page=2,
        ),
        LegalSegment(
            segment_id="law-b:schedule-1",
            segment_type=SegmentType.SCHEDULE,
            doc_id="law-b",
            doc_title="Employment Amendment Law 2021",
            doc_type=DocType.STATUTE,
            legal_path="Schedule 1",
            text="Schedule 1. Transitional employment obligations apply from 2021.",
            page_ids=["law-b_1"],
            start_page=1,
            end_page=1,
        ),
    ]


def _graph() -> ApplicabilityGraph:
    return ApplicabilityGraph(
        nodes=["law-a", "law-b"],
        edges=[
            ApplicabilityEdge(
                source_doc_id="law-b",
                target_doc_id="law-a",
                edge_type=ApplicabilityEdgeType.AMENDS,
                effective_date="1 January 2021",
                evidence_page_id="law-b_1",
            ),
            ApplicabilityEdge(
                source_doc_id="law-b",
                target_doc_id="law-a",
                edge_type=ApplicabilityEdgeType.REPLACES,
                effective_date="1 January 2021",
                evidence_page_id="law-b_1",
            ),
        ],
        laws={"law-a": _registry().laws["law-a"], "law-b": _registry().laws["law-b"]},
    )


def test_generate_all_covers_core_question_families() -> None:
    factory = SyntheticQAFactory()
    examples = factory.generate_all(registry=_registry(), segments=_segments(), graph=_graph())

    families = {example.question_family for example in examples}
    assert QuestionFamily.FACTOID_TITLE in families
    assert QuestionFamily.PROVISION_ARTICLE in families
    assert QuestionFamily.COMPARE_PARTY in families
    assert QuestionFamily.TEMPORAL_AMENDMENT in families
    assert QuestionFamily.COUNTERFACTUAL_UNSUPPORTED in families
    assert QuestionFamily.ADVERSARIAL_ALIAS in families


def test_bridge_fact_loading_and_manifest_counts(tmp_path) -> None:
    payload = [
        {
            "fact_id": "bf-1",
            "question": "Which law mentions transitional employment obligations?",
            "answer": "Employment Amendment Law 2021",
            "page_ids": ["law-b_1"],
            "doc_ids": ["law-b"],
        }
    ]
    path = tmp_path / "bridge_facts.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    bridge_facts = load_bridge_fact_records(path)
    examples = SyntheticQAFactory().generate_all(
        registry=_registry(),
        segments=_segments(),
        graph=_graph(),
        bridge_facts=bridge_facts,
    )
    manifest = build_manifest(
        examples=examples,
        bridge_fact_count=len(bridge_facts),
        llm_paraphrase_enabled=False,
        source_paths={"bridge_facts": str(path)},
    )

    assert bridge_facts[0].fact_id == "bf-1"
    assert manifest.bridge_fact_count == 1
    assert manifest.total_examples == len(examples)


def test_mine_hard_negative_page_ids_excludes_gold_pages() -> None:
    negatives = mine_hard_negative_page_ids(
        question="What does Article 5 of Employment Law 2020 provide?",
        gold_page_ids=["law-a_2"],
        gold_doc_ids=["law-a"],
        page_texts={
            "law-a_1": "Employment Law 2020 issued by DIFC Authority",
            "law-a_2": "Article 5. Employees must receive notice before termination.",
            "law-b_1": "Employment Amendment Law 2021 transitional employment obligations",
        },
        page_doc_ids={"law-a_1": "law-a", "law-a_2": "law-a", "law-b_1": "law-b"},
        limit=3,
    )

    assert "law-a_2" not in negatives
    assert negatives[0] in {"law-a_1", "law-b_1"}


def test_validate_example_against_gold_pages_handles_supported_and_counterfactual() -> None:
    factory = SyntheticQAFactory()
    examples = factory.generate_all(registry=_registry(), segments=_segments(), graph=_graph())
    page_texts = {
        "law-a_1": "Employment Law 2020. Issued by DIFC Authority. Comes into force on 1 January 2020.",
        "law-a_2": "Article 5. Employees must receive notice before termination.",
        "law-b_1": "Employment Amendment Law 2021. Issued by DIFC Authority.",
        "case-a_1": "CFI 001/2020. Alice Smith v Bob Jones. Justice Smith.",
        "case-b_1": "CFI 002/2021. Alice Smith v Registrar. Justice Patel.",
    }
    supported = next(example for example in examples if example.question_family is QuestionFamily.FACTOID_AUTHORITY)
    unsupported = next(
        example for example in examples if example.question_family is QuestionFamily.COUNTERFACTUAL_UNSUPPORTED
    )

    assert factory.validate_example_against_gold_pages(supported, page_texts=page_texts) is True
    assert factory.validate_example_against_gold_pages(unsupported, page_texts=page_texts) is True


def test_generate_all_respects_family_cap() -> None:
    factory = SyntheticQAFactory()
    examples = factory.generate_all(registry=_registry(), segments=_segments(), graph=_graph(), max_per_family=1)

    seen: dict[QuestionFamily, int] = {}
    for example in examples:
        seen[example.question_family] = seen.get(example.question_family, 0) + 1
    assert all(count == 1 for count in seen.values())
