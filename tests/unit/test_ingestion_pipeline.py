from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_challenge.models import Chunk, DocType, ParsedDocument


@pytest.fixture
def mock_settings():
    settings = SimpleNamespace(
        ingestion=SimpleNamespace(
            ingest_version="v1",
            manifest_dir="",
            manifest_filename=".rag_challenge_ingestion_manifest.json",
            manifest_hash_chunk_size_bytes=1024 * 1024,
            manifest_schema_version=1,
            sac_concurrency=4,
        )
    )
    with patch("rag_challenge.ingestion.pipeline.get_settings", return_value=settings):
        yield settings


@pytest.fixture
def doc_dir(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "doc1.txt").write_text("doc1 content", encoding="utf-8")
    (root / "doc2.txt").write_text("doc2 content", encoding="utf-8")
    return root


def _make_chunk(doc: ParsedDocument, idx: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}:0:{idx}:abc12345",
        doc_id=doc.doc_id,
        doc_title=doc.title,
        doc_type=doc.doc_type,
        jurisdiction=doc.jurisdiction,
        section_path=f"Section {idx + 1}",
        chunk_text=f"chunk {idx} text",
        chunk_text_for_embedding=f"chunk {idx} text",
        doc_summary="",
        token_count=10,
    )


@pytest.fixture
def mock_deps(doc_dir: Path):
    parser = MagicMock()
    chunker = MagicMock()
    sac = MagicMock()
    embedder = AsyncMock()
    store = AsyncMock()

    docs = [
        ParsedDocument(doc_id="d1", title="Doc 1", doc_type=DocType.STATUTE, jurisdiction="US", full_text="text1"),
        ParsedDocument(doc_id="d2", title="Doc 2", doc_type=DocType.CONTRACT, jurisdiction="US", full_text="text2"),
    ]
    path_to_doc = {
        doc_dir / "doc1.txt": docs[0].model_copy(update={"source_path": str(doc_dir / "doc1.txt")}),
        doc_dir / "doc2.txt": docs[1].model_copy(update={"source_path": str(doc_dir / "doc2.txt")}),
    }
    parser.list_supported_files.return_value = list(path_to_doc)
    parser.parse_file.side_effect = lambda file_path: path_to_doc[Path(file_path)]
    parser.parse_directory.return_value = list(path_to_doc.values())

    def chunk_doc(doc: ParsedDocument) -> list[Chunk]:
        return [_make_chunk(doc, 0), _make_chunk(doc, 1), _make_chunk(doc, 2)]

    chunker.chunk_document.side_effect = chunk_doc

    sac.generate_doc_summary = AsyncMock(return_value="A legal document summary.")
    sac.augment_chunks.side_effect = lambda chunks, summary: [
        chunk.model_copy(
            update={
                "chunk_text_for_embedding": f"[DOC_SUMMARY]\n{summary}\n\n[CHUNK]\n{chunk.chunk_text}",
                "doc_summary": summary,
            }
        )
        for chunk in chunks
    ]

    embedder.embed_documents = AsyncMock(side_effect=lambda texts: [[0.1] * 8 for _ in texts])
    embedder.close = AsyncMock()

    store.ensure_collection = AsyncMock()
    store.ensure_payload_indexes = AsyncMock()
    store.upsert_chunks = AsyncMock(side_effect=lambda chunks, vecs, sparse_vectors=None: len(chunks))
    store.delete_stale_doc_versions = AsyncMock()
    store.close = AsyncMock()

    return parser, chunker, sac, embedder, store


@pytest.mark.asyncio
async def test_pipeline_full_run(mock_settings, mock_deps, doc_dir: Path):
    from rag_challenge.ingestion.pipeline import IngestionPipeline

    parser, chunker, sac, embedder, store = mock_deps
    pipeline = IngestionPipeline(parser=parser, chunker=chunker, sac=sac, embedder=embedder, store=store)

    stats = await pipeline.run(doc_dir)

    assert stats.docs_parsed == 2
    assert stats.docs_failed == 0
    assert stats.docs_skipped_unchanged == 0
    assert stats.docs_deleted == 0
    assert stats.chunks_created == 6
    assert stats.chunks_embedded == 6
    assert stats.chunks_upserted == 6
    assert stats.sac_summaries_generated == 2
    assert stats.elapsed_s > 0
    assert stats.errors == []

    store.ensure_collection.assert_awaited_once()
    store.ensure_payload_indexes.assert_awaited_once()
    assert embedder.embed_documents.await_count == 2
    assert store.upsert_chunks.await_count == 2
    assert store.delete_stale_doc_versions.await_count == 2


@pytest.mark.asyncio
async def test_pipeline_sparse_encoder_uses_retrieval_oriented_text(mock_settings, mock_deps, doc_dir: Path):
    from rag_challenge.ingestion.pipeline import IngestionPipeline

    parser, chunker, sac, embedder, store = mock_deps
    pipeline = IngestionPipeline(parser=parser, chunker=chunker, sac=sac, embedder=embedder, store=store)
    sparse_encoder = MagicMock()
    sparse_encoder.encode_documents.return_value = [[0.1], [0.2], [0.3]]
    pipeline._sparse_encoder = sparse_encoder

    stats = await pipeline.run(doc_dir)

    assert stats.chunks_upserted == 6
    sparse_encoder.encode_documents.assert_called()
    sparse_texts = sparse_encoder.encode_documents.call_args.args[0]
    assert all(text.startswith("[DOC_SUMMARY]") for text in sparse_texts)
    upsert_kwargs = store.upsert_chunks.await_args.kwargs
    assert upsert_kwargs["sparse_vectors"] == [[0.1], [0.2], [0.3]]


@pytest.mark.asyncio
async def test_pipeline_handles_per_doc_failure_and_continues(mock_settings, mock_deps, doc_dir: Path):
    from rag_challenge.ingestion.pipeline import IngestionPipeline

    parser, chunker, sac, embedder, store = mock_deps
    call_count = 0

    def failing_chunk(doc: ParsedDocument) -> list[Chunk]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ValueError("chunking failed")
        return [_make_chunk(doc, 0)]

    chunker.chunk_document.side_effect = failing_chunk

    pipeline = IngestionPipeline(parser=parser, chunker=chunker, sac=sac, embedder=embedder, store=store)
    stats = await pipeline.run(doc_dir)

    assert stats.docs_parsed == 2
    assert stats.docs_failed == 1
    assert stats.chunks_created == 1
    assert stats.chunks_embedded == 1
    assert stats.chunks_upserted == 1
    assert len(stats.errors) == 1
    assert "chunking failed" in stats.errors[0]
    assert store.delete_stale_doc_versions.await_count == 1


@pytest.mark.asyncio
async def test_pipeline_returns_early_when_no_docs(mock_settings, mock_deps, doc_dir: Path):
    from rag_challenge.ingestion.pipeline import IngestionPipeline

    parser, chunker, sac, embedder, store = mock_deps
    parser.parse_directory.return_value = []
    parser.list_supported_files.return_value = []
    pipeline = IngestionPipeline(parser=parser, chunker=chunker, sac=sac, embedder=embedder, store=store)

    stats = await pipeline.run(doc_dir)

    assert stats.docs_parsed == 0
    assert stats.docs_skipped_unchanged == 0
    assert stats.chunks_created == 0
    embedder.embed_documents.assert_not_awaited()
    store.upsert_chunks.assert_not_awaited()


@pytest.mark.asyncio
async def test_pipeline_close_closes_owned_resources(mock_settings, mock_deps):
    from rag_challenge.ingestion.pipeline import IngestionPipeline

    parser, chunker, sac, embedder, store = mock_deps
    pipeline = IngestionPipeline(parser=parser, chunker=chunker, sac=sac, embedder=embedder, store=store)

    await pipeline.close()

    embedder.close.assert_awaited_once()
    store.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_skips_unchanged_docs_on_second_run(mock_settings, mock_deps, doc_dir: Path):
    from rag_challenge.ingestion.pipeline import IngestionPipeline

    parser, chunker, sac, embedder, store = mock_deps
    pipeline = IngestionPipeline(parser=parser, chunker=chunker, sac=sac, embedder=embedder, store=store)

    first_stats = await pipeline.run(doc_dir)
    assert first_stats.docs_parsed == 2
    assert first_stats.docs_skipped_unchanged == 0

    parser.parse_file.reset_mock()
    chunker.chunk_document.reset_mock()
    sac.generate_doc_summary.reset_mock()
    sac.augment_chunks.reset_mock()
    embedder.embed_documents.reset_mock()
    store.upsert_chunks.reset_mock()
    store.delete_stale_doc_versions.reset_mock()

    second_stats = await pipeline.run(doc_dir)

    assert second_stats.docs_parsed == 0
    assert second_stats.docs_skipped_unchanged == 2
    assert second_stats.chunks_created == 0
    assert second_stats.chunks_embedded == 0
    assert second_stats.chunks_upserted == 0
    parser.parse_file.assert_not_called()
    chunker.chunk_document.assert_not_called()
    sac.generate_doc_summary.assert_not_awaited()
    sac.augment_chunks.assert_not_called()
    embedder.embed_documents.assert_not_awaited()
    store.upsert_chunks.assert_not_awaited()
    store.delete_stale_doc_versions.assert_not_awaited()
