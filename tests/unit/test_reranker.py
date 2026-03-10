import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import SecretStr

from rag_challenge.models import DocType, RetrievedChunk


def _make_chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            doc_id=f"d{i // 5}",
            doc_title="Test Doc",
            doc_type=DocType.STATUTE,
            section_path=f"sec-{i}",
            text=f"Legal text chunk {i}",
            score=1.0 - (i * 0.01),
        )
        for i in range(n)
    ]


def _zerank_response(count: int, *, base_score: float = 1.0) -> dict[str, object]:
    return {
        "results": [{"index": i, "score": base_score - (i * 0.01)} for i in range(count)],
    }


def _cohere_result(index: int, score: float) -> SimpleNamespace:
    return SimpleNamespace(index=index, relevance_score=score)


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
def mock_settings():
    settings = SimpleNamespace(
        reranker=SimpleNamespace(
            primary_model="zerank-2",
            primary_api_url="https://api.zeroentropy.dev/v1/models/rerank",
            primary_api_key=SecretStr("ze-key"),
            primary_batch_size=50,
            primary_latency_mode="fast",
            primary_timeout_s=5.0,
            primary_max_connections=20,
            primary_connect_timeout_s=10.0,
            fallback_model="rerank-v4.0-fast",
            fallback_api_key=SecretStr("cohere-key"),
            fallback_timeout_s=5.0,
            top_n=6,
            rerank_candidates=80,
            retry_attempts=4,
            retry_base_delay_s=0.5,
            retry_max_delay_s=8.0,
            retry_jitter_s=1.0,
            circuit_failure_threshold=3,
            circuit_reset_timeout_s=60.0,
        )
    )
    with patch("rag_challenge.core.reranker.get_settings", return_value=settings):
        yield settings


@pytest.mark.asyncio
async def test_rerank_empty_returns_empty(mock_settings):
    from rag_challenge.core.reranker import RerankerClient

    cohere_client = SimpleNamespace(rerank=AsyncMock())
    async with httpx.AsyncClient() as client:
        rc = RerankerClient(client=client, cohere_client=cohere_client)
        result = await rc.rerank("query", [], top_n=3)
    assert result == []
    cohere_client.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_zerank_single_batch_success_sorted_and_trimmed(mock_settings):
    from rag_challenge.core.reranker import RerankerClient

    chunks = _make_chunks(10)

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["model"] == "zerank-2"
        assert body["latency"] == "fast"
        assert len(body["documents"]) == 10
        return httpx.Response(200, json=_zerank_response(10))

    transport = httpx.MockTransport(handler)
    cohere_client = SimpleNamespace(rerank=AsyncMock())
    async with httpx.AsyncClient(transport=transport) as client:
        rc = RerankerClient(client=client, cohere_client=cohere_client)
        result = await rc.rerank("What is contract law?", chunks, top_n=4)

    assert len(result) == 4
    assert [chunk.chunk_id for chunk in result] == ["c0", "c1", "c2", "c3"]
    assert result[0].rerank_score >= result[-1].rerank_score
    assert rc.get_last_used_model() == "zerank-2"
    cohere_client.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_zerank_splits_80_docs_into_50_and_30(mock_settings):
    from rag_challenge.core.reranker import RerankerClient

    chunks = _make_chunks(80)
    batch_sizes: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        n = len(body["documents"])
        batch_sizes.append(n)
        return httpx.Response(200, json=_zerank_response(n, base_score=0.9))

    transport = httpx.MockTransport(handler)
    cohere_client = SimpleNamespace(rerank=AsyncMock())
    async with httpx.AsyncClient(transport=transport) as client:
        rc = RerankerClient(client=client, cohere_client=cohere_client)
        result = await rc.rerank("query", chunks, top_n=6)

    assert len(result) == 6
    assert batch_sizes == [50, 30]
    cohere_client.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_falls_back_to_cohere_after_zerank_failure(mock_settings):
    from rag_challenge.core.reranker import RerankerClient

    chunks = _make_chunks(5)
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    cohere_client = SimpleNamespace(
        rerank=AsyncMock(
            return_value=SimpleNamespace(
                results=[_cohere_result(0, 0.1), _cohere_result(1, 0.8), _cohere_result(2, 0.5), _cohere_result(3, 0.2), _cohere_result(4, 0.4)]
            )
        )
    )

    async with httpx.AsyncClient(transport=transport) as client:
        rc = RerankerClient(client=client, cohere_client=cohere_client, sleep_func=_no_sleep)
        result = await rc.rerank("query", chunks, top_n=3)

    assert attempts == 4  # retries exhausted before fallback
    assert [chunk.chunk_id for chunk in result] == ["c1", "c2", "c4"]
    assert rc.get_last_used_model() == "rerank-v4.0-fast"
    cohere_client.rerank.assert_awaited_once()


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_three_failures_and_skips_zerank(mock_settings):
    from rag_challenge.core.reranker import CircuitState, RerankerClient

    chunks = _make_chunks(2)
    zerank_calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal zerank_calls
        zerank_calls += 1
        return httpx.Response(500, text="temporary failure")

    transport = httpx.MockTransport(handler)
    cohere_client = SimpleNamespace(
        rerank=AsyncMock(return_value=SimpleNamespace(results=[_cohere_result(0, 0.9), _cohere_result(1, 0.8)]))
    )

    async with httpx.AsyncClient(transport=transport) as client:
        rc = RerankerClient(client=client, cohere_client=cohere_client, sleep_func=_no_sleep)
        for _ in range(3):
            await rc.rerank("query", chunks, top_n=1)

        assert rc._circuit.state == CircuitState.OPEN
        before = zerank_calls
        await rc.rerank("query", chunks, top_n=1)

    assert zerank_calls == before
    assert cohere_client.rerank.await_count == 4


@pytest.mark.asyncio
async def test_circuit_half_open_recovers_on_success(mock_settings):
    from rag_challenge.core.reranker import CircuitState, RerankerClient

    chunks = _make_chunks(3)

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        return httpx.Response(200, json=_zerank_response(len(body["documents"])))

    transport = httpx.MockTransport(handler)
    cohere_client = SimpleNamespace(rerank=AsyncMock())
    async with httpx.AsyncClient(transport=transport) as client:
        rc = RerankerClient(client=client, cohere_client=cohere_client)
        rc._circuit.state = CircuitState.OPEN
        rc._circuit.failure_count = 3
        rc._circuit.last_failure_time = time.monotonic() - 61.0

        result = await rc.rerank("query", chunks, top_n=2)

    assert len(result) == 2
    assert rc._circuit.state == CircuitState.CLOSED
    assert rc._circuit.failure_count == 0
    cohere_client.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_retry_after_header_is_respected_on_429(mock_settings):
    from rag_challenge.core.reranker import RerankerClient

    chunks = _make_chunks(1)
    attempts = 0
    sleeps: list[float] = []

    async def capture_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=_zerank_response(1))

    transport = httpx.MockTransport(handler)
    cohere_client = SimpleNamespace(rerank=AsyncMock())
    async with httpx.AsyncClient(transport=transport) as client:
        rc = RerankerClient(client=client, cohere_client=cohere_client, sleep_func=capture_sleep)
        result = await rc.rerank("query", chunks, top_n=1)

    assert attempts == 2
    assert len(result) == 1
    assert sleeps and sleeps[0] == 0.0


@pytest.mark.asyncio
async def test_prefer_fast_uses_cohere_first(mock_settings):
    from rag_challenge.core.reranker import RerankerClient

    chunks = _make_chunks(3)
    zerank_calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal zerank_calls
        zerank_calls += 1
        return httpx.Response(500, text="should not be called")

    transport = httpx.MockTransport(handler)
    cohere_client = SimpleNamespace(
        rerank=AsyncMock(return_value=SimpleNamespace(results=[_cohere_result(0, 0.9), _cohere_result(1, 0.8), _cohere_result(2, 0.7)]))
    )
    async with httpx.AsyncClient(transport=transport) as client:
        rc = RerankerClient(client=client, cohere_client=cohere_client)
        result = await rc.rerank("query", chunks, top_n=2, prefer_fast=True)

    assert len(result) == 2
    assert zerank_calls == 0
    cohere_client.rerank.assert_awaited_once()


@pytest.mark.asyncio
async def test_degrades_to_raw_scores_when_both_providers_fail(mock_settings):
    from rag_challenge.core.reranker import RerankerClient

    chunks = _make_chunks(2)
    transport = httpx.MockTransport(lambda _: httpx.Response(500, text="ze down"))
    cohere_client = SimpleNamespace(rerank=AsyncMock(side_effect=RuntimeError("cohere down")))

    async with httpx.AsyncClient(transport=transport) as client:
        rc = RerankerClient(client=client, cohere_client=cohere_client, sleep_func=_no_sleep)
        result = await rc.rerank("query", chunks, top_n=1)

    assert len(result) == 1
    assert result[0].chunk_id == "c0"
    assert result[0].rerank_score == pytest.approx(result[0].retrieval_score)
    assert rc.get_last_used_model() == "raw_retrieval_fallback"
