from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_challenge.core.compare_engine import CompareEngine, CompareType
from rag_challenge.core.query_contract import ExecutionEngine, QueryContractCompiler
from rag_challenge.ingestion.canonical_entities import EntityAliasResolver
from rag_challenge.models import QueryComplexity
from rag_challenge.models.legal_objects import (
    CaseObject,
    CaseParty,
    CorpusRegistry,
    LegalDocType,
)
from rag_challenge.telemetry import TelemetryCollector


def _registry() -> CorpusRegistry:
    return CorpusRegistry(
        source_doc_count=2,
        cases={
            "cfi_001_2024": CaseObject(
                object_id="case:cfi_001_2024",
                doc_id="cfi_001_2024",
                title="Alpha Ltd v Beta Ltd",
                case_number="CFI 001/2024",
                judges=["Justice Jane Smith"],
                parties=[
                    CaseParty(name="Alpha Ltd", role="claimant"),
                    CaseParty(name="Beta Ltd", role="respondent"),
                ],
                date="1 January 2024",
                legal_doc_type=LegalDocType.CASE,
                page_ids=["cfi_001_2024_1", "cfi_001_2024_2"],
                field_page_ids={
                    "case_number": ["cfi_001_2024_1"],
                    "party": ["cfi_001_2024_1"],
                    "judge": ["cfi_001_2024_2"],
                    "date": ["cfi_001_2024_2"],
                },
            ),
            "cfi_002_2024": CaseObject(
                object_id="case:cfi_002_2024",
                doc_id="cfi_002_2024",
                title="Gamma Ltd v Beta Ltd",
                case_number="CFI 002/2024",
                judges=["Justice Jane Smith"],
                parties=[
                    CaseParty(name="Gamma Ltd", role="claimant"),
                    CaseParty(name="Beta Ltd", role="respondent"),
                ],
                date="4 January 2024",
                legal_doc_type=LegalDocType.CASE,
                page_ids=["cfi_002_2024_1", "cfi_002_2024_2"],
                field_page_ids={
                    "case_number": ["cfi_002_2024_1"],
                    "party": ["cfi_002_2024_1"],
                    "judge": ["cfi_002_2024_2"],
                    "date": ["cfi_002_2024_2"],
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


def test_compare_engine_answers_common_judge_boolean(tmp_path) -> None:
    registry_path, alias_path = _write_runtime_artifacts(tmp_path)
    compiler = QueryContractCompiler.from_alias_registry_path(alias_path)
    contract = compiler.compile(
        "Did CFI 001/2024 and CFI 002/2024 share a judge?",
        answer_type="boolean",
    )
    engine = CompareEngine.from_path(registry_path)

    result = engine.execute(contract)

    assert contract.execution_plan == [ExecutionEngine.COMPARE_JOIN]
    assert result is not None
    assert result.result_type is CompareType.COMMON_JUDGE
    assert result.formatted_answer == "Yes"
    assert result.entities == ["Justice Jane Smith"]
    assert result.source_page_ids == ["cfi_001_2024_2", "cfi_002_2024_2"]


def test_compare_engine_answers_common_party_name(tmp_path) -> None:
    registry_path, alias_path = _write_runtime_artifacts(tmp_path)
    compiler = QueryContractCompiler.from_alias_registry_path(alias_path)
    contract = compiler.compile(
        "Which party was common to CFI 001/2024 and CFI 002/2024?",
        answer_type="name",
    )
    engine = CompareEngine.from_path(registry_path)

    result = engine.execute(contract)

    assert result is not None
    assert result.result_type is CompareType.COMMON_PARTY
    assert result.formatted_answer == "Beta Ltd"


def test_compare_engine_answers_earlier_date_compare(tmp_path) -> None:
    registry_path, alias_path = _write_runtime_artifacts(tmp_path)
    compiler = QueryContractCompiler.from_alias_registry_path(alias_path)
    contract = compiler.compile(
        "Which case has an earlier decision date: CFI 001/2024 or CFI 002/2024?",
        answer_type="name",
    )
    engine = CompareEngine.from_path(registry_path)

    result = engine.execute(contract)

    assert result is not None
    assert result.result_type is CompareType.ATTRIBUTE_COMPARE
    assert result.formatted_answer == "CFI 001/2024"


@pytest.mark.asyncio
async def test_pipeline_short_circuits_to_compare_engine(tmp_path) -> None:
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
            compare_engine_enabled=True,
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
        collector = TelemetryCollector(request_id="compare-answer", answer_type="boolean")
        events: list[dict[str, object]] = []
        with patch("rag_challenge.core.pipeline.get_stream_writer", return_value=lambda event: events.append(event)):
            result = await app.ainvoke(
                {
                    "query": "Did CFI 001/2024 and CFI 002/2024 share a judge?",
                    "request_id": "compare-answer",
                    "answer_type": "boolean",
                    "collector": collector,
                }
            )

    assert result["answer"] == "Yes"
    assert result["compare_result"].source_page_ids == ["cfi_001_2024_2", "cfi_002_2024_2"]
    assert result["telemetry"].used_page_ids == ["cfi_001_2024_2", "cfi_002_2024_2"]
    assert retriever.retrieve.await_count == 0
    assert reranker.rerank.await_count == 0
    assert generator.generate.await_count == 0
    assert any(event.get("type") == "answer_final" and event.get("text") == "Yes" for event in events)
