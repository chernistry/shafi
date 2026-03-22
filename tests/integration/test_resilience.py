from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import SecretStr

from shafi.models import DocType, RetrievedChunk


def _make_retrieved_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_title="Doc",
            doc_type=DocType.STATUTE,
            section_path="S1",
            text="Low score but still valid",
            score=0.2,
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="d1",
            doc_title="Doc",
            doc_type=DocType.STATUTE,
            section_path="S2",
            text="High score candidate",
            score=0.9,
        ),
    ]


async def _no_sleep(_: float) -> None:
    return None


@pytest.mark.asyncio
async def test_reranker_fallback_to_raw_scores() -> None:
    settings = SimpleNamespace(
        reranker=SimpleNamespace(
            primary_model="zerank-2",
            primary_api_url="https://api.zeroentropy.dev/v1/models/rerank",
            primary_api_key=SecretStr("ze-key"),
            primary_batch_size=50,
            primary_latency_mode="fast",
            primary_timeout_s=5.0,
            primary_max_connections=20,
            primary_concurrency_limit=1,
            primary_min_interval_s=0.25,
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

    with patch("shafi.core.reranker.get_settings", return_value=settings):
        from shafi.core.reranker import RerankerClient

        transport = httpx.MockTransport(lambda _: httpx.Response(500, text="zerank down"))
        cohere_client = SimpleNamespace(rerank=AsyncMock(side_effect=RuntimeError("cohere down")))
        async with httpx.AsyncClient(transport=transport) as client:
            reranker = RerankerClient(client=client, cohere_client=cohere_client, sleep_func=_no_sleep)
            ranked = await reranker.rerank("query", _make_retrieved_chunks(), top_n=2)

    assert [chunk.chunk_id for chunk in ranked] == ["c2", "c1"]
    assert [chunk.rerank_score for chunk in ranked] == [pytest.approx(0.9), pytest.approx(0.2)]


@pytest.mark.asyncio
async def test_llm_cascade_on_primary_failure() -> None:
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            provider="openai_compatible",
            base_url="https://api.openai.com/v1",
            api_key=SecretStr(""),
            openai_api_key=SecretStr("openai-test"),
            openrouter_api_key=SecretStr(""),
            openrouter_referer="",
            openrouter_title="",
            anthropic_api_key=SecretStr("anthropic-test"),
            simple_model="gpt-4o-mini",
            complex_model="gpt-4o",
            fallback_model="claude-3-5-sonnet-latest",
            summary_model="gpt-4o-mini",
            simple_max_tokens=300,
            complex_max_tokens=500,
            temperature=0.0,
            timeout_s=60.0,
            connect_timeout_s=10.0,
            max_context_tokens=2500,
            stream_include_usage=True,
            circuit_failure_threshold=3,
            circuit_reset_timeout_s=60.0,
        )
    )

    calls: list[str] = []

    async def fake_create(**kwargs: object) -> object:
        model = str(kwargs["model"])
        calls.append(model)
        if model == "gpt-4o-mini":
            raise RuntimeError("mini down")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="fallback answer"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    openai_client = AsyncMock()
    openai_client.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=fake_create)))
    openai_client.close = AsyncMock()

    with patch("shafi.llm.provider.get_settings", return_value=settings):
        from shafi.llm.provider import LLMProvider

        provider = LLMProvider(openai_client=openai_client)  # type: ignore[arg-type]
        result = await provider.generate_with_cascade(
            system_prompt="sys",
            user_prompt="usr",
            models=["gpt-4o-mini", "gpt-4o"],
        )

    assert result.text == "fallback answer"
    assert calls == ["gpt-4o-mini", "gpt-4o"]


def test_circuit_breaker_opens_and_recovers() -> None:
    from shafi.core.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(name="test", failure_threshold=3, reset_timeout_s=0.1)
    for _ in range(3):
        cb.record_failure()

    assert cb.allow_request() is False
    time.sleep(0.15)
    assert cb.allow_request() is True
    cb.record_success()
    assert cb.state.value == "closed"
