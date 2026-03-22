from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_challenge.models import DocType, QueryComplexity, RankedChunk, RetrievedChunk
from rag_challenge.telemetry import TelemetryCollector


def _retrieved_chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc:0:0:abc",
        doc_id="doc",
        doc_title="General Partnership Law",
        doc_type=DocType.STATUTE,
        section_path="Article 17",
        text=text,
        score=0.9,
    )


def _ranked_chunk(text: str) -> RankedChunk:
    return RankedChunk(
        chunk_id="doc:0:0:abc",
        doc_id="doc",
        doc_title="General Partnership Law",
        doc_type=DocType.STATUTE,
        section_path="Article 17",
        text=text,
        retrieval_score=0.9,
        rerank_score=0.9,
    )


@pytest.mark.asyncio
async def test_boolean_format_remains_compliant() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    text = "A person cannot become a Partner without consent of all existing Partners unless otherwise agreed."
    settings = SimpleNamespace(
        llm=SimpleNamespace(simple_model="gpt-4o-mini", strict_max_tokens=150),
        embedding=SimpleNamespace(model="kanon-2-embedder"),
        reranker=SimpleNamespace(primary_model="zerank-2", top_n=6, rerank_candidates=80),
        pipeline=SimpleNamespace(
            confidence_threshold=0.3,
            retry_query_max_anchors=3,
            strict_types_force_simple_model=True,
            strict_types_extraction_enabled=True,
            strict_types_extraction_max_chunks=4,
            rerank_enabled_strict_types=True,
            rerank_enabled_boolean=False,
            strict_types_context_top_n=3,
            boolean_context_top_n=2,
            enable_multi_hop=False,
            doc_ref_sparse_only=True,
        ),
        verifier=SimpleNamespace(enabled=False),
    )

    with patch("rag_challenge.core.pipeline.get_settings", return_value=settings):
        retriever = MagicMock()
        retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
        retriever.retrieve = AsyncMock(return_value=[_retrieved_chunk(text)])
        retriever.retrieve_with_retry = AsyncMock(return_value=[_retrieved_chunk(text)])

        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=[_ranked_chunk(text)])
        reranker.get_last_used_model = MagicMock(return_value="zerank-2")

        generator = MagicMock()
        generator.generate = AsyncMock(return_value=("No (cite: doc:0:0:abc)", []))
        generator.generate_stream = AsyncMock()
        generator.extract_cited_chunk_ids = MagicMock(return_value=["doc:0:0:abc"])
        generator.extract_citations = MagicMock(return_value=[])
        generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)

        classifier = MagicMock()
        classifier.normalize_query.side_effect = lambda q: q.strip()
        classifier.classify.return_value = QueryComplexity.SIMPLE
        classifier.select_model.return_value = "gpt-4o-mini"
        classifier.select_max_tokens.return_value = 300
        classifier.extract_doc_refs.return_value = []

        builder = RAGPipelineBuilder(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            classifier=classifier,
        )
        app = builder.compile()
        collector = TelemetryCollector(request_id="strict-1")
        result = await app.ainvoke(
            {
                "query": "Can a person become a Partner without consent of all Partners?",
                "request_id": "strict-1",
                "question_id": "strict-1",
                "answer_type": "boolean",
                "collector": collector,
            }
        )

    assert str(result["answer"]).strip().lower().startswith(("yes", "no"))
