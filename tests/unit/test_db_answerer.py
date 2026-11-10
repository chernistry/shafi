from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shafi.core.db_answerer import DatabaseAnswerer
from shafi.core.entity_registry import EntityRegistry
from shafi.core.field_lookup import FieldLookupTable
from shafi.core.query_contract import ExecutionEngine, PredicateType, QueryContractCompiler
from shafi.ingestion.canonical_entities import EntityAliasResolver
from shafi.models import QueryComplexity
from shafi.models.legal_objects import (
    CaseObject,
    CaseParty,
    CorpusRegistry,
    LawObject,
    LegalDocType,
    OrderObject,
)
from shafi.telemetry import TelemetryCollector


def _registry() -> CorpusRegistry:
    return CorpusRegistry(
        source_doc_count=2,
        laws={
            "companies_law": LawObject(
                object_id="law:companies_law",
                doc_id="companies_law",
                title="DIFC Companies Law No. 3 of 2004",
                source_path="companies_law.pdf",
                page_ids=["companies_law_1", "companies_law_2"],
                source_text="Issued by DIFC Authority. Commencement date 1 January 2005.",
                page_texts={
                    "companies_law_1": "DIFC Companies Law No. 3 of 2004. Issued by DIFC Authority.",
                    "companies_law_2": "This law comes into force on 1 January 2005.",
                },
                field_page_ids={
                    "title": ["companies_law_1"],
                    "law_number": ["companies_law_1"],
                    "issued_by": ["companies_law_1"],
                    "authority": ["companies_law_1"],
                    "commencement_date": ["companies_law_2"],
                    "date": ["companies_law_2"],
                },
                legal_doc_type=LegalDocType.LAW,
                short_title="Companies Law",
                law_number="3",
                year="2004",
                issuing_authority="DIFC Authority",
                commencement_date="1 January 2005",
            )
        },
        orders={
            "data_protection_notice": OrderObject(
                object_id="enactment_notice:data_protection_notice",
                doc_id="data_protection_notice",
                title="_______________________________________________",
                source_path="notice.pdf",
                page_ids=["data_protection_notice_1"],
                source_text=(
                    "ENACTMENT NOTICE\n"
                    "We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact\n"
                    "on this 01 day of March 2024\n"
                    "the\n"
                    "DATA PROTECTION LAW\n"
                    "DIFC LAW NO. 5 OF 2020\n"
                ),
                page_texts={"data_protection_notice_1": "DATA PROTECTION LAW\nDIFC LAW NO. 5 OF 2020"},
                field_page_ids={
                    "title": ["data_protection_notice_1"],
                    "date": ["data_protection_notice_1"],
                    "authority": ["data_protection_notice_1"],
                    "issued_by": ["data_protection_notice_1"],
                },
                legal_doc_type=LegalDocType.ENACTMENT_NOTICE,
                issued_by="the",
                effective_date="",
            )
        },
        cases={
            "cfi_001_2024": CaseObject(
                object_id="case:cfi_001_2024",
                doc_id="cfi_001_2024",
                title="Alpha Ltd v Beta Ltd",
                source_path="cfi_001_2024.pdf",
                page_ids=["cfi_001_2024_1", "cfi_001_2024_2"],
                source_text="Claimant: Alpha Ltd. Respondent: Beta Ltd. Judge Jane Smith.",
                page_texts={
                    "cfi_001_2024_1": "Alpha Ltd v Beta Ltd. Claimant: Alpha Ltd. Respondent: Beta Ltd.",
                    "cfi_001_2024_2": "Before Justice Jane Smith. The claim was dismissed.",
                },
                field_page_ids={
                    "title": ["cfi_001_2024_1"],
                    "case_number": ["cfi_001_2024_1"],
                    "claimant": ["cfi_001_2024_1"],
                    "respondent": ["cfi_001_2024_1"],
                    "party": ["cfi_001_2024_1"],
                    "judge": ["cfi_001_2024_2"],
                    "outcome": ["cfi_001_2024_2"],
                },
                legal_doc_type=LegalDocType.CASE,
                case_number="CFI 001/2024",
                judges=["Justice Jane Smith"],
                parties=[
                    CaseParty(name="Alpha Ltd", role="claimant"),
                    CaseParty(name="Beta Ltd", role="respondent"),
                ],
                outcome_summary="The claim was dismissed.",
            )
        },
    )


def _write_runtime_artifacts(tmp_path) -> tuple[str, str]:
    registry = _registry()
    registry_path = tmp_path / "corpus_registry.json"
    registry_path.write_text(registry.model_dump_json(indent=2), encoding="utf-8")
    resolver = EntityAliasResolver.build_from_registry(registry)
    alias_path = resolver.export(tmp_path / "canonical_entity_registry.json")
    return str(registry_path), str(alias_path)


def test_database_answerer_answers_lookup_field_from_registry(tmp_path) -> None:
    registry_path, alias_path = _write_runtime_artifacts(tmp_path)
    compiler = QueryContractCompiler.from_alias_registry_path(alias_path)
    contract = compiler.compile(
        "Who issued DIFC Companies Law No. 3 of 2004?",
        answer_type="name",
    )
    answerer = DatabaseAnswerer.from_paths(
        lookup_table_path=registry_path,
        alias_registry_path=alias_path,
        confidence_threshold=0.85,
    )

    answer = answerer.answer(contract)

    assert contract.execution_plan == [ExecutionEngine.FIELD_LOOKUP]
    assert answer is not None
    assert answer.value == "DIFC Authority"
    assert answer.source_page_ids == ["companies_law_1"]
    assert answerer.format_answer(answer, "name") == "DIFC Authority"


@pytest.mark.asyncio
async def test_pipeline_short_circuits_to_database_answerer(tmp_path) -> None:
    from shafi.core.pipeline import RAGPipelineBuilder

    registry_path, alias_path = _write_runtime_artifacts(tmp_path)
    settings = SimpleNamespace(
        embedding=SimpleNamespace(model="kanon-2-embedder"),
        reranker=SimpleNamespace(primary_model="zerank-2", top_n=6, rerank_candidates=80),
        verifier=SimpleNamespace(enabled=False),
        ingestion=SimpleNamespace(corpus_registry_path=registry_path),
        pipeline=SimpleNamespace(
            confidence_threshold=0.3,
            retry_query_max_anchors=3,
            rerank_max_candidates_strict_types=20,
            boolean_rerank_candidates_cap=12,
            strict_doc_ref_top_k=16,
            strict_multi_ref_top_k_per_ref=12,
            strict_prefetch_dense=24,
            strict_prefetch_sparse=24,
            free_text_targeted_multi_ref_top_k=12,
            enable_multi_hop=False,
            canonical_entity_registry_path=alias_path,
            db_answerer_enabled=True,
            db_answerer_confidence_threshold=0.85,
        ),
    )
    retriever = MagicMock()
    retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
    retriever.retrieve = AsyncMock()
    retriever.retrieve_with_retry = AsyncMock()

    reranker = MagicMock()
    reranker.rerank = AsyncMock()

    generator = MagicMock()
    generator.generate = AsyncMock()
    generator.generate_stream = AsyncMock()

    classifier = MagicMock()
    classifier.normalize_query.side_effect = lambda q: q.strip()
    classifier.classify.return_value = QueryComplexity.SIMPLE
    classifier.select_model.return_value = "gpt-4o-mini"
    classifier.select_max_tokens.return_value = 64

    with patch("shafi.core.pipeline.get_settings", return_value=settings):
        builder = RAGPipelineBuilder(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            classifier=classifier,
            verifier=None,
        )
        app = builder.compile()
        collector = TelemetryCollector(request_id="db-answer", answer_type="name")
        events: list[dict[str, object]] = []
        with patch("shafi.core.pipeline.get_stream_writer", return_value=lambda event: events.append(event)):
            result = await app.ainvoke(
                {
                    "query": "Who issued DIFC Companies Law No. 3 of 2004?",
                    "request_id": "db-answer",
                    "answer_type": "name",
                    "collector": collector,
                }
            )

    assert result["answer"] == "DIFC Authority"
    assert result["db_answer"].source_page_ids == ["companies_law_1"]
    assert result["telemetry"].used_page_ids == ["companies_law_1"]
    assert retriever.retrieve.await_count == 0
    assert reranker.rerank.await_count == 0
    assert generator.generate.await_count == 0
    assert any(event.get("type") == "answer_final" and event.get("text") == "DIFC Authority" for event in events)


def test_database_answerer_falls_back_when_confidence_threshold_is_too_high(tmp_path) -> None:
    registry_path, alias_path = _write_runtime_artifacts(tmp_path)
    compiler = QueryContractCompiler.from_alias_registry_path(alias_path)
    contract = compiler.compile(
        "Who issued DIFC Companies Law No. 3 of 2004?",
        answer_type="name",
    )
    answerer = DatabaseAnswerer.from_paths(
        lookup_table_path=registry_path,
        alias_registry_path=alias_path,
        confidence_threshold=0.99,
    )

    assert answerer.answer(contract) is None
    assert FieldLookupTable.build_from_registry(_registry()).lookup("law:companies_law", "issued_by") is not None


def test_database_answerer_uses_enactment_notice_for_law_queries(tmp_path) -> None:
    registry_path, alias_path = _write_runtime_artifacts(tmp_path)
    compiler = QueryContractCompiler.from_alias_registry_path(alias_path)
    contract = compiler.compile(
        "What is the law number of the Data Protection Law?",
        answer_type="number",
    )
    answerer = DatabaseAnswerer.from_paths(
        lookup_table_path=registry_path,
        alias_registry_path=alias_path,
        confidence_threshold=0.85,
    )

    answer = answerer.answer(contract)

    assert answer is not None
    assert answer.value == "5"
    assert answer.source_doc_id == "data_protection_notice"
    assert answerer.format_answer(answer, "number") == "5"


def test_database_answerer_answers_case_lookup_without_alias_resolver() -> None:
    compiler = QueryContractCompiler()
    contract = compiler.compile(
        "Who were the claimants in case CFI 001/2024?",
        answer_type="names",
    )
    answerer = DatabaseAnswerer(
        lookup_table=FieldLookupTable.build_from_registry(_registry()),
        entity_registry=EntityRegistry(),
        confidence_threshold=0.85,
    )

    answer = answerer.answer(contract)

    assert contract.predicate is PredicateType.LOOKUP_FIELD
    assert contract.primary_entities
    assert contract.primary_entities[0].canonical_id == "case_number:cfi0012024"
    assert answer is not None
    assert answer.value == "Alpha Ltd"
    assert answer.source_doc_id == "cfi_001_2024"
    assert answer.source_page_ids == ["cfi_001_2024_1"]
