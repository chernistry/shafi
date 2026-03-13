import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from pydantic import SecretStr

from rag_challenge.core.embedding import EmbeddingClient, EmbeddingError


@pytest.fixture
def mock_settings():
    """Patch get_settings to return test config."""
    embedding = SimpleNamespace(
        provider="isaacus",
        model="kanon-2-embedder",
        api_url="https://api.isaacus.com/v1/embeddings",
        ollama_base_url="http://localhost:11434",
        api_key=SecretStr("test-key"),
        dimensions=1024,
        batch_size=128,
        concurrency=4,
        timeout_s=30.0,
        connect_timeout_s=10.0,
        ollama_batch_size=32,
        ollama_concurrency=2,
        ollama_timeout_s=90.0,
        retry_attempts=6,
        retry_base_delay_s=0.5,
        retry_max_delay_s=16.0,
        retry_jitter_s=2.0,
        circuit_failure_threshold=3,
        circuit_reset_timeout_s=60.0,
    )
    settings = SimpleNamespace(embedding=embedding)
    with patch("rag_challenge.core.embedding.get_settings", return_value=settings):
        yield settings


def _make_embed_response(count: int, dims: int = 1024) -> dict[str, object]:
    return {
        "embeddings": [[0.1] * dims for _ in range(count)],
        "usage": {"total_tokens": count * 10},
    }


async def _no_sleep(_: float) -> None:
    return None


@pytest.mark.asyncio
async def test_embed_query_single(mock_settings):
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_embed_response(1)))
    async with httpx.AsyncClient(transport=transport) as client:
        ec = EmbeddingClient(client=client)
        result = await ec.embed_query("What is contract law?")
        assert len(result) == 1024
        assert isinstance(result[0], float)


@pytest.mark.asyncio
async def test_embed_documents_batching(mock_settings):
    """Test that 200 texts are split into 2 batches (128 + 72)."""
    call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(req.content)
        count = len(body["texts"])
        assert body["task"] == "retrieval/document"
        return httpx.Response(200, json=_make_embed_response(count))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        ec = EmbeddingClient(client=client)
        texts = [f"chunk {i}" for i in range(200)]
        result = await ec.embed_documents(texts)
        assert len(result) == 200
        assert call_count == 2  # ceil(200/128)


@pytest.mark.asyncio
async def test_embed_empty_list(mock_settings):
    async with httpx.AsyncClient() as client:
        ec = EmbeddingClient(client=client)
        result = await ec.embed_documents([])
        assert result == []


@pytest.mark.asyncio
async def test_embed_retry_on_429(mock_settings):
    attempt = 0

    def handler(req: httpx.Request) -> httpx.Response:
        del req
        nonlocal attempt
        attempt += 1
        if attempt <= 2:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=_make_embed_response(1))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        ec = EmbeddingClient(client=client, sleep_func=_no_sleep)
        result = await ec.embed_query("test")
        assert len(result) == 1024
        assert attempt == 3


@pytest.mark.asyncio
async def test_embed_raises_on_persistent_failure(mock_settings):
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="Internal Server Error"))
    async with httpx.AsyncClient(transport=transport) as client:
        ec = EmbeddingClient(client=client, sleep_func=_no_sleep)
        with pytest.raises(EmbeddingError, match="failed after"):
            await ec.embed_query("test")


@pytest.mark.asyncio
async def test_embed_raises_on_client_error(mock_settings):
    transport = httpx.MockTransport(lambda req: httpx.Response(400, text="Bad request"))
    async with httpx.AsyncClient(transport=transport) as client:
        ec = EmbeddingClient(client=client)
        with pytest.raises(EmbeddingError, match="400"):
            await ec.embed_query("test")


@pytest.mark.asyncio
async def test_embed_query_uses_query_task(mock_settings):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["task"] == "retrieval/query"
        return httpx.Response(200, json=_make_embed_response(1))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        ec = EmbeddingClient(client=client)
        await ec.embed_query("test query")


@pytest.mark.asyncio
async def test_embed_documents_uses_document_task(mock_settings):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["task"] == "retrieval/document"
        return httpx.Response(200, json=_make_embed_response(len(body["texts"])))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        ec = EmbeddingClient(client=client)
        await ec.embed_documents(["doc1", "doc2"])


@pytest.mark.asyncio
async def test_embed_parses_object_rows_with_embedding_field(mock_settings):
    response = {
        "embeddings": [
            {"embedding": [0.1] * 4},
            {"embedding": [0.2] * 4},
        ]
    }
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=response))
    async with httpx.AsyncClient(transport=transport) as client:
        ec = EmbeddingClient(client=client)
        vectors = await ec.embed_documents(["a", "b"])
    assert vectors == [[0.1] * 4, [0.2] * 4]


@pytest.mark.asyncio
async def test_ollama_embed_batch_via_embed_endpoint(mock_settings):
    mock_settings.embedding.provider = "ollama"

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/embed"
        body = json.loads(req.content)
        assert body["input"] == ["a", "b"]
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as client:
        ec = EmbeddingClient(client=client)
        vectors = await ec.embed_documents(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_ollama_falls_back_to_legacy_embeddings_endpoint(mock_settings):
    mock_settings.embedding.provider = "ollama"
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/api/embed":
            return httpx.Response(404, text="not found")
        assert req.url.path == "/api/embeddings"
        body = json.loads(req.content)
        prompt = body["prompt"]
        if prompt == "a":
            return httpx.Response(200, json={"embedding": [0.1, 0.2]})
        return httpx.Response(200, json={"embedding": [0.3, 0.4]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as client:
        ec = EmbeddingClient(client=client)
        vectors = await ec.embed_documents(["a", "b"])
    assert calls[0] == "/api/embed"
    assert calls.count("/api/embeddings") == 2
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_ollama_embed_documents_uses_provider_batch_size(mock_settings):
    mock_settings.embedding.provider = "ollama"
    mock_settings.embedding.batch_size = 128
    mock_settings.embedding.ollama_batch_size = 2
    call_sizes: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        values = body["input"]
        count = 1 if isinstance(values, str) else len(values)
        call_sizes.append(count)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * count})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as client:
        ec = EmbeddingClient(client=client)
        vectors = await ec.embed_documents(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert call_sizes == [2, 2, 1]
