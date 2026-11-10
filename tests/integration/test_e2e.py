from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from shafi.models import Citation, DocType, QueryComplexity, RankedChunk, RetrievedChunk
from shafi.telemetry import TelemetryCollector

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "docs"


def _chunker_settings_stub() -> SimpleNamespace:
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            chunk_size_tokens=450,
            chunk_overlap_tokens=45,
        )
    )


def _make_ranked_from_retrieved(chunks: list[RetrievedChunk], top_n: int) -> list[RankedChunk]:
    ranked: list[RankedChunk] = []
    for idx, chunk in enumerate(chunks[:top_n]):
        ranked.append(
            RankedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                doc_type=chunk.doc_type,
                section_path=chunk.section_path,
                text=chunk.text,
                retrieval_score=chunk.score,
                rerank_score=1.0 - (idx * 0.05),
                doc_summary=chunk.doc_summary,
            )
        )
    return ranked


def _load_fixture_retrieved_chunks() -> list[RetrievedChunk]:
    from shafi.ingestion.chunker import LegalChunker
    from shafi.ingestion.parser import DocumentParser

    parser = DocumentParser()
    with patch("shafi.ingestion.chunker.get_settings", return_value=_chunker_settings_stub()):
        chunker = LegalChunker()
    docs = parser.parse_directory(FIXTURES_DIR)

    retrieved: list[RetrievedChunk] = []
    for doc in docs:
        for idx, chunk in enumerate(chunker.chunk_document(doc)):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    doc_type=chunk.doc_type,
                    section_path=chunk.section_path,
                    text=chunk.chunk_text,
                    score=max(0.0, 0.99 - (idx * 0.03)),
                    doc_summary=chunk.doc_summary,
                )
            )
    return retrieved


@pytest.mark.asyncio
async def test_fixture_docs_parse_and_chunk_with_real_components() -> None:
    from shafi.ingestion.chunker import LegalChunker
    from shafi.ingestion.parser import DocumentParser

    parser = DocumentParser()
    docs = parser.parse_directory(FIXTURES_DIR)

    assert FIXTURES_DIR.exists()
    assert len(docs) == 3

    with patch("shafi.ingestion.chunker.get_settings", return_value=_chunker_settings_stub()):
        chunker = LegalChunker()
        chunks = [chunk for doc in docs for chunk in chunker.chunk_document(doc)]
    assert len(chunks) >= 5

    # Deterministic IDs: parsing/chunking same fixtures twice produces same chunk IDs.
    docs_again = parser.parse_directory(FIXTURES_DIR)
    with patch("shafi.ingestion.chunker.get_settings", return_value=_chunker_settings_stub()):
        chunker_again = LegalChunker()
        chunks_again = [chunk for doc in docs_again for chunk in chunker_again.chunk_document(doc)]
    assert [chunk.chunk_id for chunk in chunks] == [chunk.chunk_id for chunk in chunks_again]
    assert any(chunk.doc_type == DocType.STATUTE for chunk in chunks)
    assert any(chunk.doc_type == DocType.CONTRACT for chunk in chunks)


@pytest.mark.asyncio
async def test_ingestion_pipeline_with_real_fixtures_and_mocked_external_services() -> None:
    from shafi.ingestion.chunker import LegalChunker
    from shafi.ingestion.parser import DocumentParser
    from shafi.ingestion.pipeline import IngestionPipeline

    parser = DocumentParser()
    with patch("shafi.ingestion.chunker.get_settings", return_value=_chunker_settings_stub()):
        chunker = LegalChunker()

    sac = MagicMock()
    sac.generate_doc_summary = AsyncMock(return_value="Fixture summary.")
    sac.augment_chunks.side_effect = lambda chunks, summary: [
        chunk.model_copy(
            update={
                "chunk_text_for_embedding": f"[DOC_SUMMARY]\n{summary}\n\n[CHUNK]\n{chunk.chunk_text}",
                "doc_summary": summary,
            }
        )
        for chunk in chunks
    ]

    embedder = AsyncMock()
    embedder.embed_documents = AsyncMock(side_effect=lambda texts: [[0.1] * 8 for _ in texts])
    embedder.close = AsyncMock()

    store = AsyncMock()
    store.ensure_collection = AsyncMock()
    store.ensure_payload_indexes = AsyncMock()
    store.upsert_chunks = AsyncMock(side_effect=lambda chunks, vectors: len(chunks))
    store.delete_stale_doc_versions = AsyncMock()
    store.close = AsyncMock()

    with TemporaryDirectory() as manifest_tmp_dir:
        settings = SimpleNamespace(
            ingestion=SimpleNamespace(
                ingest_version="itest-v1",
                manifest_filename=".shafi_ingestion_manifest.json",
                manifest_dir=manifest_tmp_dir,
                manifest_hash_chunk_size_bytes=1024 * 1024,
                manifest_schema_version=1,
            )
        )
        with (
            patch("shafi.ingestion.pipeline.get_settings", return_value=settings),
            patch("shafi.ingestion.chunker.get_settings", return_value=_chunker_settings_stub()),
        ):
            pipeline = IngestionPipeline(parser=parser, chunker=chunker, sac=sac, embedder=embedder, store=store)
            stats = await pipeline.run(FIXTURES_DIR)
            await pipeline.close()

    assert stats.docs_parsed == 3
    assert stats.docs_failed == 0
    assert stats.chunks_created > 0
    assert stats.chunks_embedded == stats.chunks_created
    assert stats.chunks_upserted == stats.chunks_created
    assert stats.sac_summaries_generated == 3
    assert stats.errors == []

    total_upserted_chunks = 0
    total_upserted_vectors = 0
    for call in store.upsert_chunks.await_args_list:
        upserted_chunks, upserted_vectors = call.args
        total_upserted_chunks += len(upserted_chunks)
        total_upserted_vectors += len(upserted_vectors)
        assert all(chunk.doc_summary == "Fixture summary." for chunk in upserted_chunks)
        assert all(chunk.chunk_text_for_embedding.startswith("[DOC_SUMMARY]") for chunk in upserted_chunks)
    assert total_upserted_chunks == total_upserted_vectors == stats.chunks_created
    assert store.delete_stale_doc_versions.await_count == 3


@pytest.mark.asyncio
async def test_rag_pipeline_builder_direct_end_to_end_chunk_id_chain() -> None:
    from shafi.core.pipeline import RAGPipelineBuilder

    retrieved = _load_fixture_retrieved_chunks()
    assert retrieved, "fixture chunk retrieval set should not be empty"

    settings = SimpleNamespace(
        embedding=SimpleNamespace(model="kanon-2-embedder"),
        reranker=SimpleNamespace(primary_model="zerank-2", top_n=4),
        pipeline=SimpleNamespace(confidence_threshold=0.3),
    )
    with patch("shafi.core.pipeline.get_settings", return_value=settings):
        retriever = MagicMock()
        retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
        retriever.retrieve = AsyncMock(return_value=retrieved[:8])
        retriever.retrieve_with_retry = AsyncMock(return_value=retrieved[:8])

        reranker = MagicMock()
        reranked = _make_ranked_from_retrieved(retrieved[:8], top_n=4)
        reranker.rerank = AsyncMock(return_value=reranked)

        generator = MagicMock()
        top_cid = reranked[0].chunk_id
        second_cid = reranked[1].chunk_id if len(reranked) > 1 else top_cid

        async def _gen_stream(*args: object, **kwargs: object):
            del args, kwargs
            yield "The limitation period is six years "
            yield f"(cite: {top_cid}, {second_cid})."

        generator.generate_stream = _gen_stream
        generator.extract_cited_chunk_ids = MagicMock(return_value=[top_cid, second_cid])
        generator.extract_citations = MagicMock(
            return_value=[
                Citation(chunk_id=top_cid, doc_title=reranked[0].doc_title, section_path=reranked[0].section_path),
                Citation(chunk_id=second_cid, doc_title=reranked[1].doc_title, section_path=reranked[1].section_path),
            ]
        )

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
        app = builder.compile()
        collector = TelemetryCollector(request_id="e2e-direct")
        events: list[dict[str, object]] = []

        with patch("shafi.core.pipeline.get_stream_writer", return_value=lambda event: events.append(event)):
            result = await app.ainvoke(
                {
                    "query": "What is the limitation period for contract claims?",
                    "request_id": "e2e-direct",
                    "collector": collector,
                }
            )

    assert isinstance(result.get("answer"), str)
    assert result["answer"].strip()
    assert "telemetry" in result
    assert result.get("citations")

    telemetry = result["telemetry"]
    assert telemetry.request_id == "e2e-direct"
    assert telemetry.model_embed == "kanon-2-embedder"
    assert telemetry.model_rerank == "zerank-2"
    assert telemetry.model_llm == "gpt-4o-mini"
    assert telemetry.total_ms >= 0
    assert telemetry.ttft_ms >= 0
    assert telemetry.total_ms >= telemetry.ttft_ms
    assert telemetry.retrieved_chunk_ids
    assert telemetry.context_chunk_ids
    assert telemetry.cited_chunk_ids
    assert set(telemetry.cited_chunk_ids).issubset(set(telemetry.context_chunk_ids))
    assert set(telemetry.context_chunk_ids).issubset(set(telemetry.retrieved_chunk_ids))

    citation_ids = {citation.chunk_id for citation in result["citations"]}
    assert citation_ids.issubset(set(telemetry.context_chunk_ids))
    assert any(event.get("type") == "token" for event in events)
    assert any(event.get("type") == "telemetry" for event in events)


@pytest.mark.asyncio
async def test_qdrant_smoke_optional_skips_if_unavailable() -> None:
    from shafi.core.qdrant import QdrantStore

    collection_name = f"itest_{uuid4().hex[:8]}"
    qdrant_settings = SimpleNamespace(
        url="http://localhost:6333",
        api_key="",
        collection=collection_name,
        pool_size=4,
        timeout_s=2.0,
        prefetch_dense=10,
        prefetch_sparse=10,
        use_cloud_inference=True,
        fusion_method="RRF",
    )
    embedding_settings = SimpleNamespace(dimensions=8)
    ingestion_settings = SimpleNamespace(upsert_batch_size=10, ingest_version="itest-v1")
    settings = SimpleNamespace(
        qdrant=qdrant_settings,
        embedding=embedding_settings,
        ingestion=ingestion_settings,
    )

    with patch("shafi.core.qdrant.get_settings", return_value=settings):
        store = QdrantStore()
        try:
            if not await store.health_check():
                pytest.skip("Local Qdrant not available at http://localhost:6333")
            await store.ensure_collection()
            await store.ensure_payload_indexes()
            count = await store.count_points()
            assert count >= 0
        finally:
            with suppress(Exception):
                await store.client.delete_collection(collection_name=collection_name)
            await store.close()
