from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag_challenge.models import QueryComplexity
from rag_challenge.telemetry import TelemetryCollector


@pytest.mark.asyncio
async def test_boolean_classification_caps_max_tokens() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    settings = SimpleNamespace(
        llm=SimpleNamespace(simple_model="gpt-4o-mini", strict_max_tokens=150),
        embedding=SimpleNamespace(model="kanon-2-embedder"),
        reranker=SimpleNamespace(primary_model="zerank-2", top_n=6),
        pipeline=SimpleNamespace(
            confidence_threshold=0.3,
            retry_query_max_anchors=3,
            strict_types_force_simple_model=True,
            boolean_max_tokens=96,
        ),
        verifier=SimpleNamespace(enabled=True),
    )
    with patch("rag_challenge.core.pipeline.get_settings", return_value=settings):
        retriever = MagicMock()
        reranker = MagicMock()
        generator = MagicMock()
        classifier = MagicMock()
        classifier.normalize_query.side_effect = lambda q: q.strip()
        classifier.classify.return_value = QueryComplexity.COMPLEX
        classifier.select_model.return_value = "openai/gpt-4o"
        classifier.select_max_tokens.return_value = 500
        classifier.extract_doc_refs.return_value = []

        builder = RAGPipelineBuilder(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            classifier=classifier,
        )
        update = await builder._classify(
            {
                "query": "Is this allowed under Article 11?",
                "request_id": "req-1",
                "question_id": "q-1",
                "answer_type": "boolean",
                "collector": TelemetryCollector(request_id="req-1"),
            }
        )

    assert update["max_tokens"] == 96
    assert update["model"] == "gpt-4o-mini"
