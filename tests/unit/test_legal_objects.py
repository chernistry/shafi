from __future__ import annotations

from shafi.models.legal_objects import (
    ArticleNode,
    CorpusRegistry,
    LawObject,
    LegalDocType,
)


def test_law_object_round_trip_serializes_article_tree() -> None:
    law = LawObject(
        object_id="law:limitation_act",
        doc_id="limitation_act",
        title="Limitation Act 2020",
        legal_doc_type=LegalDocType.LAW,
        short_title="Limitation Act 2020",
        year="2020",
        article_tree=[
            ArticleNode(
                article_id="limitation_act:1",
                label="Section 1",
                title="Short Title",
                page_ids=["limitation_act_1"],
            )
        ],
    )

    round_tripped = LawObject.model_validate_json(law.model_dump_json())

    assert round_tripped.title == "Limitation Act 2020"
    assert round_tripped.article_tree[0].label == "Section 1"


def test_corpus_registry_serializes_nested_objects() -> None:
    registry = CorpusRegistry(
        source_doc_count=1,
        laws={
            "limitation_act": LawObject(
                object_id="law:limitation_act",
                doc_id="limitation_act",
                title="Limitation Act 2020",
                legal_doc_type=LegalDocType.LAW,
            )
        },
    )

    payload = CorpusRegistry.model_validate_json(registry.model_dump_json())

    assert payload.source_doc_count == 1
    assert "limitation_act" in payload.laws
