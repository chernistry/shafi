from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Headers
from qdrant_client.http.exceptions import UnexpectedResponse

from shafi.models import Chunk, DocType


@pytest.fixture
def mock_settings():
    settings = SimpleNamespace(
        qdrant=SimpleNamespace(
            url="http://localhost:6333",
            api_key="",
            collection="test_collection",
            shadow_collection="test_collection_shadow",
            page_collection="test_pages",
            pool_size=5,
            timeout_s=10.0,
            use_cloud_inference=True,
            enable_sparse_bm25=True,
            sparse_model="Qdrant/bm25",
            fastembed_cache_dir="",
            check_compatibility=False,
        ),
        embedding=SimpleNamespace(dimensions=1024),
        ingestion=SimpleNamespace(upsert_batch_size=100, ingest_version="v1"),
    )
    with patch("shafi.core.qdrant.get_settings", return_value=settings):
        yield settings


@pytest.fixture
def mock_qdrant_client():
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.upsert = AsyncMock()
    client.set_sparse_model = MagicMock()
    client.get_collections = AsyncMock()
    client.close = AsyncMock()
    return client


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        doc_title="Test Contract",
        doc_type=DocType.CONTRACT,
        jurisdiction="US",
        section_path="Article 1 > Section 2",
        chunk_text="This is the original chunk text.",
        chunk_text_for_embedding="[DOC_SUMMARY]\nSummary\n\n[CHUNK]\nThis is the original chunk text.",
        doc_summary="Summary",
        token_count=50,
    )


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_not_exists(mock_settings, mock_qdrant_client):
    from shafi.core.qdrant import QdrantStore

    store = QdrantStore(client=mock_qdrant_client)
    await store.ensure_collection()
    mock_qdrant_client.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_collection_skips_when_exists(mock_settings, mock_qdrant_client):
    from shafi.core.qdrant import QdrantStore

    mock_qdrant_client.collection_exists.return_value = True
    store = QdrantStore(client=mock_qdrant_client)
    await store.ensure_collection()
    mock_qdrant_client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_chunks(mock_settings, mock_qdrant_client):
    from shafi.core.qdrant import QdrantStore

    store = QdrantStore(client=mock_qdrant_client)
    chunks = [_make_chunk(f"c{i}") for i in range(5)]
    vectors = [[0.1] * 1024 for _ in range(5)]
    count = await store.upsert_chunks(chunks, vectors)
    assert count == 5
    mock_qdrant_client.upsert.assert_called_once()
    payload = mock_qdrant_client.upsert.await_args.kwargs["points"][0].payload
    assert payload["doc_family"] == ""
    assert payload["shadow_search_text"] == ""


@pytest.mark.asyncio
async def test_upsert_shadow_chunks_uses_shadow_collection(mock_settings, mock_qdrant_client):
    from shafi.core.qdrant import QdrantStore

    store = QdrantStore(client=mock_qdrant_client)
    chunk = _make_chunk("c-shadow").model_copy(
        update={
            "shadow_search_text": "Operating Law 2018\nArticle 16\nfiled document",
            "law_titles": ["Operating Law 2018"],
            "article_refs": ["Article 16"],
            "cross_refs": ["Article 16"],
        }
    )

    count = await store.upsert_shadow_chunks([chunk], [[0.1] * 1024])

    assert count == 1
    kwargs = mock_qdrant_client.upsert.await_args.kwargs
    assert kwargs["collection_name"] == "test_collection_shadow"
    payload = kwargs["points"][0].payload
    assert payload["shadow_search_text"].startswith("Operating Law 2018")
    assert payload["article_refs"] == ["Article 16"]


@pytest.mark.asyncio
async def test_upsert_validates_length_mismatch(mock_settings, mock_qdrant_client):
    from shafi.core.qdrant import QdrantStore

    store = QdrantStore(client=mock_qdrant_client)
    chunks = [_make_chunk("c1")]
    vectors = [[0.1] * 1024, [0.2] * 1024]
    with pytest.raises(ValueError, match="equal length"):
        await store.upsert_chunks(chunks, vectors)


@pytest.mark.asyncio
async def test_upsert_falls_back_to_dense_only_when_qdrant_inference_unavailable(mock_settings, mock_qdrant_client):
    from shafi.core.qdrant import QdrantStore

    mock_qdrant_client.upsert = AsyncMock(
        side_effect=[
            UnexpectedResponse(
                status_code=500,
                reason_phrase="Internal Server Error",
                content=b'{"status":{"error":"InferenceService is not initialized"}}',
                headers=Headers(),
            ),
            None,
        ]
    )
    store = QdrantStore(client=mock_qdrant_client)
    chunks = [_make_chunk(f"c{i}") for i in range(2)]
    vectors = [[0.1] * 1024 for _ in range(2)]

    count = await store.upsert_chunks(chunks, vectors)

    assert count == 2
    assert mock_qdrant_client.upsert.await_count == 2
    retry_kwargs = mock_qdrant_client.upsert.await_args_list[1].kwargs
    points = retry_kwargs["points"]
    assert isinstance(points, list)
    assert all(isinstance(point.vector, dict) and "dense" in point.vector for point in points)


@pytest.mark.asyncio
async def test_health_check_success(mock_settings, mock_qdrant_client):
    from shafi.core.qdrant import QdrantStore

    store = QdrantStore(client=mock_qdrant_client)
    assert await store.health_check() is True


@pytest.mark.asyncio
async def test_health_check_failure(mock_settings, mock_qdrant_client):
    from shafi.core.qdrant import QdrantStore

    mock_qdrant_client.get_collections.side_effect = RuntimeError("connection refused")
    store = QdrantStore(client=mock_qdrant_client)
    assert await store.health_check() is False
