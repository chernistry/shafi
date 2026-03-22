from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_challenge.models import Citation, DocType, QueryComplexity, RankedChunk, RetrievedChunk
from rag_challenge.telemetry import TelemetryCollector


def _retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc:0:0:abc",
        doc_id="doc",
        doc_title="ENF Case",
        doc_type=DocType.CASE_LAW,
        section_path="Section 1",
        text="The DIFC Court granted enforcement. No mention of jury.",
        score=0.9,
    )


def _ranked_chunk() -> RankedChunk:
    return RankedChunk(
        chunk_id="doc:0:0:abc",
        doc_id="doc",
        doc_title="ENF Case",
        doc_type=DocType.CASE_LAW,
        section_path="Section 1",
        text="The DIFC Court granted enforcement. No mention of jury.",
        retrieval_score=0.9,
        rerank_score=0.9,
    )


@pytest.mark.asyncio
async def test_trick_question_returns_insufficient_sources_with_guard() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    settings = SimpleNamespace(
        llm=SimpleNamespace(simple_model="gpt-4o-mini", strict_max_tokens=150),
        embedding=SimpleNamespace(model="kanon-2-embedder"),
        reranker=SimpleNamespace(primary_model="zerank-2", top_n=6, rerank_candidates=80),
        pipeline=SimpleNamespace(
            confidence_threshold=0.3,
            retry_query_max_anchors=3,
            premise_guard_enabled=True,
            premise_guard_terms=["jury", "miranda", "parole", "plea bargain", "plea"],
            enable_multi_hop=False,
            doc_ref_sparse_only=True,
            strict_types_extraction_enabled=True,
            strict_types_extraction_max_chunks=4,
            rerank_enabled_strict_types=True,
            rerank_enabled_boolean=False,
            strict_types_context_top_n=3,
            boolean_context_top_n=2,
        ),
        verifier=SimpleNamespace(enabled=False),
    )

    with patch("rag_challenge.core.pipeline.get_settings", return_value=settings):
        retriever = MagicMock()
        retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
        retriever.retrieve = AsyncMock(return_value=[_retrieved_chunk()])
        retriever.retrieve_with_retry = AsyncMock(return_value=[_retrieved_chunk()])

        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=[_ranked_chunk()])
        reranker.get_last_used_model = MagicMock(return_value="zerank-2")

        generator = MagicMock()
        generator.generate_stream = AsyncMock()
        generator.generate = AsyncMock(return_value=("The jury decided ...", []))
        generator.extract_cited_chunk_ids = MagicMock(side_effect=lambda text: ["doc:0:0:abc"] if "cite:" in text else [])
        generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="doc:0:0:abc", doc_title="ENF Case")])
        generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)

        classifier = MagicMock()
        classifier.normalize_query.side_effect = lambda q: q.strip()
        classifier.classify.return_value = QueryComplexity.SIMPLE
        classifier.select_model.return_value = "gpt-4o-mini"
        classifier.select_max_tokens.return_value = 300
        classifier.extract_doc_refs.return_value = ["ENF 053/2025"]

        builder = RAGPipelineBuilder(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            classifier=classifier,
        )
        app = builder.compile()
        collector = TelemetryCollector(request_id="trick-1")

        result = await app.ainvoke(
            {
                "query": "What did the jury decide in case ENF 053/2025?",
                "request_id": "trick-1",
                "question_id": "trick-1",
                "answer_type": "free_text",
                "collector": collector,
            }
        )

    assert str(result["answer"]) == "There is no information on this question."
    generator.generate_stream.assert_not_called()
