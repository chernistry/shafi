from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_challenge.models import DocType, QueryComplexity, RetrievedChunk
from rag_challenge.telemetry import TelemetryCollector


def _retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c0",
        doc_id="d1",
        doc_title="Doc",
        doc_type=DocType.STATUTE,
        text="Some legal text.",
        score=0.9,
    )


@pytest.mark.asyncio
async def test_boolean_retrieve_uses_prefetch_overrides_without_doc_refs() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    settings = SimpleNamespace(
        llm=SimpleNamespace(simple_model="gpt-4o-mini", strict_max_tokens=150),
        embedding=SimpleNamespace(model="kanon-2-embedder"),
        reranker=SimpleNamespace(primary_model="zerank-2", top_n=6),
        pipeline=SimpleNamespace(
            confidence_threshold=0.3,
            retry_query_max_anchors=3,
            enable_multi_hop=False,
            doc_ref_sparse_only=True,
            boolean_prefetch_dense=40,
            boolean_prefetch_sparse=40,
        ),
        verifier=SimpleNamespace(enabled=True),
    )

    with patch("rag_challenge.core.pipeline.get_settings", return_value=settings):
        retriever = MagicMock()
        retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
        # First call: parallel sparse-first returns empty (forces hybrid fallback).
        # Second call: hybrid fallback with boolean prefetch overrides.
        retriever.retrieve = AsyncMock(side_effect=[[], [_retrieved_chunk()]])

        reranker = MagicMock()
        generator = MagicMock()
        classifier = MagicMock()
        classifier.normalize_query.side_effect = lambda q: q.strip()
        classifier.classify.return_value = QueryComplexity.SIMPLE
        classifier.select_model.return_value = "gpt-4o-mini"
        classifier.select_max_tokens.return_value = 300

        builder = RAGPipelineBuilder(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            classifier=classifier,
        )

        await builder._retrieve(
            {
                "query": "Is this valid?",
                "request_id": "req-1",
                "question_id": "q-1",
                "answer_type": "boolean",
                "collector": TelemetryCollector(request_id="req-1"),
                "doc_refs": [],
                "sub_queries": [],
            }
        )

    # The hybrid fallback call (second call) should use boolean prefetch overrides
    hybrid_call = retriever.retrieve.call_args_list[1]
    assert hybrid_call.kwargs["prefetch_dense"] == 40
    assert hybrid_call.kwargs["prefetch_sparse"] == 40
