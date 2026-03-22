from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shafi.core.pipeline import RAGPipelineBuilder
from shafi.core.proof_answerer import ProofAnswer
from shafi.models import Citation, DocType, QueryComplexity, RankedChunk
from shafi.telemetry import TelemetryCollector


def _chunk(*, chunk_id: str, text: str) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        doc_title="Test Law",
        doc_type=DocType.STATUTE,
        section_path="page:1",
        text=text,
        retrieval_score=0.9,
        rerank_score=0.9,
        doc_summary="",
    )


def _build_builder() -> RAGPipelineBuilder:
    retriever = MagicMock()
    retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
    retriever.retrieve = AsyncMock(
        return_value=[
            _chunk(chunk_id="c0", text="The law applies."),
            _chunk(chunk_id="c1", text="Payment is due within 30 days."),
        ]
    )
    retriever.retrieve_with_retry = AsyncMock(
        return_value=[
            _chunk(chunk_id="c0", text="The law applies."),
            _chunk(chunk_id="c1", text="Payment is due within 30 days."),
        ]
    )

    reranker = MagicMock()
    reranker.rerank = AsyncMock(
        return_value=[
            RankedChunk(
                chunk_id="c0",
                doc_id="doc1",
                doc_title="Test Law",
                doc_type=DocType.STATUTE,
                section_path="page:1",
                text="The law applies.",
                retrieval_score=0.9,
                rerank_score=0.9,
                doc_summary="",
            ),
            RankedChunk(
                chunk_id="c1",
                doc_id="doc1",
                doc_title="Test Law",
                doc_type=DocType.STATUTE,
                section_path="page:1",
                text="Payment is due within 30 days.",
                retrieval_score=0.8,
                rerank_score=0.8,
                doc_summary="",
            ),
        ]
    )
    reranker.get_last_used_model = MagicMock(return_value="zerank-2")

    generator = MagicMock()
    generator.generate = AsyncMock(return_value=("The law applies. Payment is due within 30 days.", []))

    async def _gen_stream_proof(*args, **kwargs):
        yield "The law applies. Payment is due within 30 days."

    generator.generate_stream = _gen_stream_proof
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Test Law")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)

    classifier = MagicMock()
    classifier.normalize_query.side_effect = lambda q: q.strip()
    classifier.classify.return_value = QueryComplexity.SIMPLE
    classifier.select_model.return_value = "gpt-4o-mini"
    classifier.select_max_tokens.return_value = 300

    settings = SimpleNamespace(
        embedding=SimpleNamespace(model="kanon-2-embedder"),
        reranker=SimpleNamespace(primary_model="zerank-2", top_n=6, rerank_candidates=80),
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
            canonical_entity_registry_path="",
            claim_graph_enabled=True,
            proof_compiler_enabled=True,
            proof_min_coverage=0.5,
            proof_allow_partial_answers=True,
            proof_fluency_pass_enabled=True,
        ),
        verifier=SimpleNamespace(enabled=True),
    )

    with patch("shafi.core.pipeline.get_settings", return_value=settings):
        return RAGPipelineBuilder(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            classifier=classifier,
        )


@pytest.mark.asyncio
async def test_pipeline_proof_compiler_opt_in() -> None:
    builder = _build_builder()
    app = builder.compile()
    collector = TelemetryCollector(request_id="proof-test")

    with patch("shafi.core.pipeline.get_stream_writer", return_value=lambda _event: None):
        result = await app.ainvoke(
            {
                "query": "What is the answer?",
                "request_id": "proof-test",
                "collector": collector,
                "answer_type": "free_text",
            }
        )

    assert "proof_answer" in result
    assert result["proof_answer"].answer_text
    assert result["telemetry"].claim_graph_enabled is True
    assert result["telemetry"].proof_compiler_enabled is True


@pytest.mark.asyncio
async def test_pipeline_proof_compiler_skips_non_free_text_answers() -> None:
    builder = _build_builder()
    app = builder.compile()
    collector = TelemetryCollector(request_id="proof-date-test")

    with patch("shafi.core.pipeline.get_stream_writer", return_value=lambda _event: None):
        result = await app.ainvoke(
            {
                "query": "When did the law come into force?",
                "request_id": "proof-date-test",
                "collector": collector,
                "answer_type": "date",
            }
        )

    assert result["telemetry"].proof_compiler_enabled is True
    assert result["telemetry"].proof_compiler_used is False
    assert result["telemetry"].proof_compiler_fallback_reason == "non_free_text_answer_type"


@pytest.mark.asyncio
async def test_pipeline_proof_compiler_respects_runtime_coverage_gate() -> None:
    builder = _build_builder()
    app = builder.compile()
    collector = TelemetryCollector(request_id="proof-coverage-test")

    with (
        patch("shafi.core.pipeline.get_stream_writer", return_value=lambda _event: None),
        patch(
            "shafi.core.pipeline.generation_logic.ProofCarryingCompiler.compile",
            return_value=ProofAnswer(
                answer_text="Proof answer override.",
                support_coverage=0.2,
                fallback_reason="",
            ),
        ),
    ):
        result = await app.ainvoke(
            {
                "query": "What is the answer?",
                "request_id": "proof-coverage-test",
                "collector": collector,
                "answer_type": "free_text",
            }
        )

    assert result["answer"] == "The law applies. Payment is due within 30 days."
    assert result["telemetry"].proof_compiler_enabled is True
    assert result["telemetry"].proof_compiler_used is False
    assert result["telemetry"].proof_compiler_fallback_reason == "coverage_below_runtime_threshold"
