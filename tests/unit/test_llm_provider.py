from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr


def _usage(prompt: int, completion: int, total: int) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _resp(text: str, *, prompt_tokens: int = 10, completion_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=_usage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens),
    )


def _stream_chunk(text: str | None = None, *, usage: SimpleNamespace | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
        usage=usage,
    )


class FakeAsyncStream:
    def __init__(self, items: list[object]) -> None:
        self._items = items

    def __aiter__(self) -> "FakeAsyncStream":
        self._iter = iter(self._items)
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@pytest.fixture
def mock_settings():
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
    with patch("rag_challenge.llm.provider.get_settings", return_value=settings):
        yield settings


def _mock_openai_with_create(side_effect: object) -> AsyncMock:
    client = AsyncMock()
    create_mock = AsyncMock()
    if callable(side_effect):
        create_mock.side_effect = side_effect
    else:
        create_mock.return_value = side_effect
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    client.close = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_generate_returns_llm_result(mock_settings):
    from rag_challenge.llm.provider import LLMProvider

    mock_openai = _mock_openai_with_create(_resp("The answer is 42.", prompt_tokens=100, completion_tokens=20))
    provider = LLMProvider(openai_client=mock_openai)  # type: ignore[arg-type]

    result = await provider.generate(system_prompt="You are helpful.", user_prompt="What is the answer?")

    assert result.text == "The answer is 42."
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20
    assert result.total_tokens == 120
    assert result.model == "gpt-4o-mini"
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_generate_with_cascade_falls_back(mock_settings):
    from rag_challenge.llm.provider import LLMProvider

    calls: list[str] = []

    async def fake_create(**kwargs: object) -> object:
        model = kwargs["model"]
        assert isinstance(model, str)
        calls.append(model)
        if len(calls) < 3:
            raise RuntimeError("temporary failure")
        return _resp("Fallback answer", prompt_tokens=50, completion_tokens=10)

    mock_openai = _mock_openai_with_create(fake_create)
    provider = LLMProvider(openai_client=mock_openai)  # type: ignore[arg-type]

    result = await provider.generate_with_cascade(system_prompt="test", user_prompt="test")

    assert result.text == "Fallback answer"
    assert calls == ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-latest"]
    assert result.model == "claude-3-5-sonnet-latest"


@pytest.mark.asyncio
async def test_stream_generate_yields_tokens_and_captures_usage(mock_settings):
    from rag_challenge.llm.provider import LLMProvider

    stream = FakeAsyncStream(
        [
            _stream_chunk("Hello"),
            _stream_chunk(" "),
            _stream_chunk("world"),
            _stream_chunk(None, usage=_usage(80, 12, 92)),
        ]
    )
    mock_openai = _mock_openai_with_create(stream)
    provider = LLMProvider(openai_client=mock_openai)  # type: ignore[arg-type]

    tokens = [token async for token in provider.stream_generate(system_prompt="sys", user_prompt="usr")]

    assert "".join(tokens) == "Hello world"
    assert provider.get_last_stream_usage() == {
        "prompt_tokens": 80,
        "completion_tokens": 12,
        "total_tokens": 92,
    }


@pytest.mark.asyncio
async def test_stream_generate_with_cascade_retries_before_tokens(mock_settings):
    from rag_challenge.llm.provider import LLMProvider

    calls: list[str] = []

    async def fake_create(**kwargs: object) -> object:
        model = kwargs["model"]
        assert isinstance(model, str)
        calls.append(model)
        if len(calls) == 1:
            raise RuntimeError("connection failed")
        return FakeAsyncStream([_stream_chunk("ok"), _stream_chunk(None, usage=_usage(1, 1, 2))])

    mock_openai = _mock_openai_with_create(fake_create)
    provider = LLMProvider(openai_client=mock_openai)  # type: ignore[arg-type]

    tokens = [
        token
        async for token in provider.stream_generate_with_cascade(
            system_prompt="sys",
            user_prompt="usr",
            models=["gpt-4o-mini", "gpt-4o"],
        )
    ]

    assert "".join(tokens) == "ok"
    assert calls == ["gpt-4o-mini", "gpt-4o"]


@pytest.mark.asyncio
async def test_stream_cascade_does_not_fallback_after_tokens_started(mock_settings):
    from rag_challenge.llm.provider import LLMProvider, LLMProviderError

    class FailingStream:
        def __aiter__(self) -> "FailingStream":
            self._step = 0
            return self

        async def __anext__(self) -> object:
            if self._step == 0:
                self._step += 1
                return _stream_chunk("partial")
            raise RuntimeError("mid-stream failure")

    mock_openai = _mock_openai_with_create(FailingStream())
    provider = LLMProvider(openai_client=mock_openai)  # type: ignore[arg-type]

    collected: list[str] = []
    with pytest.raises(LLMProviderError, match="stream failed"):
        async for token in provider.stream_generate_with_cascade(
            system_prompt="sys",
            user_prompt="usr",
            models=["gpt-4o-mini", "gpt-4o"],
        ):
            collected.append(token)

    assert collected == ["partial"]


@pytest.mark.asyncio
async def test_generate_raises_provider_error_when_all_models_fail(mock_settings):
    from rag_challenge.llm.provider import LLMProvider, LLMProviderError

    async def fail_create(**_: object) -> object:
        raise RuntimeError("down")

    mock_openai = _mock_openai_with_create(fail_create)
    provider = LLMProvider(openai_client=mock_openai)  # type: ignore[arg-type]

    with pytest.raises(LLMProviderError, match="All models in cascade failed"):
        await provider.generate_with_cascade(
            system_prompt="sys",
            user_prompt="usr",
            models=["gpt-4o-mini", "gpt-4o"],
        )


def test_provider_initializes_openai_compatible_client_with_base_url_and_headers(mock_settings):
    from rag_challenge.llm.provider import LLMProvider

    mock_settings.llm.base_url = "https://openrouter.ai/api/v1"
    mock_settings.llm.openrouter_api_key = SecretStr("sk-or-test")
    mock_settings.llm.openai_api_key = SecretStr("")
    mock_settings.llm.openrouter_referer = "http://localhost:8000"
    mock_settings.llm.openrouter_title = "rag-challenge"

    with patch("rag_challenge.llm.provider.AsyncOpenAI") as mock_client_cls:
        client = AsyncMock()
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
        client.close = AsyncMock()
        mock_client_cls.return_value = client

        LLMProvider()

    kwargs = mock_client_cls.call_args.kwargs
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert kwargs["api_key"] == "sk-or-test"
    assert kwargs["default_headers"]["HTTP-Referer"] == "http://localhost:8000"
    assert kwargs["default_headers"]["X-Title"] == "rag-challenge"
