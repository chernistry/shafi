from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_challenge.core.applicability_engine import ApplicabilityEngine, TemporalQueryType
from rag_challenge.core.query_contract import ExecutionEngine, QueryContractCompiler
from rag_challenge.ingestion.canonical_entities import EntityAliasResolver
from rag_challenge.models import QueryComplexity
from rag_challenge.models.legal_objects import CorpusRegistry, LawObject, LegalDocType
from rag_challenge.telemetry import TelemetryCollector


def _registry() -> CorpusRegistry:
    return CorpusRegistry(
        source_doc_count=3,
        laws={
            "old_law_2018": LawObject(
                object_id="law:old_law_2018",
                doc_id="old_law_2018",
                title="Old Law 2018",
                short_title="Old Law",
                law_number="1",
                year="2018",
                legal_doc_type=LegalDocType.LAW,
                page_ids=["old_law_2018_1", "old_law_2018_2"],
                page_texts={
                    "old_law_2018_1": "Old Law 2018 comes into force on 1 January 2018.",
                    "old_law_2018_2": "This law remains in force until replaced.",
                },
            ),
            "new_law_2022": LawObject(
                object_id="law:new_law_2022",
                doc_id="new_law_2022",
                title="New Law 2022",
                short_title="New Law",
                law_number="2",
                year="2022",
                legal_doc_type=LegalDocType.LAW,
                page_ids=["new_law_2022_1"],
                page_texts={
                    "new_law_2022_1": "New Law 2022 replaces Old Law 2018 and comes into force on 1 January 2022.",
                },
            ),
            "amendment_law_2023": LawObject(
                object_id="law:amendment_law_2023",
                doc_id="amendment_law_2023",
                title="Amendment Law 2023",
                short_title="Amendment Law",
                law_number="3",
                year="2023",
                legal_doc_type=LegalDocType.AMENDMENT,
                page_ids=["amendment_law_2023_1"],
                page_texts={
                    "amendment_law_2023_1": "This Amendment Law amends New Law 2022 and comes into force on 1 February 2023.",
                },
            ),
        },
    )


def _write_runtime_artifacts(tmp_path) -> tuple[str, str]:
    registry = _registry()
    registry_path = tmp_path / "corpus_registry.json"
    registry_path.write_text(registry.model_dump_json(indent=2), encoding="utf-8")
    resolver = EntityAliasResolver.build_from_registry(registry)
    alias_path = resolver.export(tmp_path / "canonical_entity_registry.json")
    return str(registry_path), str(alias_path)


def test_applicability_engine_answers_commencement(tmp_path) -> None:
    registry_path, alias_path = _write_runtime_artifacts(tmp_path)
    compiler = QueryContractCompiler.from_alias_registry_path(alias_path)
    contract = compiler.compile("When did Old Law 2018 come into force?", answer_type="date")
    engine = ApplicabilityEngine.from_registry_path(registry_path)

    result = engine.answer(contract)

    assert contract.execution_plan == [ExecutionEngine.TEMPORAL_QUERY]
    assert result is not None
    assert result.query_type is TemporalQueryType.COMMENCEMENT
    assert result.answer_formatted == "1 January 2018"


def test_applicability_engine_answers_supersession(tmp_path) -> None:
    registry_path, alias_path = _write_runtime_artifacts(tmp_path)
    compiler = QueryContractCompiler.from_alias_registry_path(alias_path)
    contract = compiler.compile("Which law replaced Old Law 2018?", answer_type="name")
    engine = ApplicabilityEngine.from_registry_path(registry_path)

    result = engine.answer(contract)

    assert result is not None
    assert result.query_type is TemporalQueryType.SUPERSESSION
    assert result.answer_formatted == "New Law 2022"


def test_applicability_engine_answers_temporal_status_on_specific_date(tmp_path) -> None:
    registry_path, alias_path = _write_runtime_artifacts(tmp_path)
    compiler = QueryContractCompiler.from_alias_registry_path(alias_path)
    contract = compiler.compile("Was Old Law 2018 in force on 1 June 2021?", answer_type="boolean")
    engine = ApplicabilityEngine.from_registry_path(registry_path)

    result = engine.answer(contract)

    assert result is not None
    assert result.query_type is TemporalQueryType.TEMPORAL_STATUS
    assert result.answer_formatted == "Yes"


def test_applicability_engine_answers_amendment_list(tmp_path) -> None:
    registry_path, alias_path = _write_runtime_artifacts(tmp_path)
    compiler = QueryContractCompiler.from_alias_registry_path(alias_path)
    contract = compiler.compile("What amendments apply to New Law 2022?", answer_type="free_text")
    engine = ApplicabilityEngine.from_registry_path(registry_path)

    result = engine.answer(contract)

    assert result is not None
    assert result.query_type is TemporalQueryType.AMENDMENT_LIST
    assert "Amendment Law 2023" in result.answer_formatted


@pytest.mark.asyncio
async def test_pipeline_short_circuits_to_temporal_engine(tmp_path) -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

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
            db_answerer_enabled=False,
            db_answerer_confidence_threshold=0.85,
            compare_engine_enabled=False,
            temporal_engine_enabled=True,
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

    with patch("rag_challenge.core.pipeline.get_settings", return_value=settings):
        builder = RAGPipelineBuilder(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            classifier=classifier,
            verifier=None,
        )
        app = builder.compile()
        collector = TelemetryCollector(request_id="temporal-answer", answer_type="date")
        events: list[dict[str, object]] = []
        with patch("rag_challenge.core.pipeline.get_stream_writer", return_value=lambda event: events.append(event)):
            result = await app.ainvoke(
                {
                    "query": "When did Old Law 2018 come into force?",
                    "request_id": "temporal-answer",
                    "answer_type": "date",
                    "collector": collector,
                }
            )

    assert result["answer"] == "1 January 2018"
    assert result["telemetry"].used_page_ids == ["old_law_2018_1"]
    assert retriever.retrieve.await_count == 0
    assert reranker.rerank.await_count == 0
    assert generator.generate.await_count == 0
    assert any(event.get("type") == "answer_final" and event.get("text") == "1 January 2018" for event in events)
