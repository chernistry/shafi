from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI

from shafi.config import get_settings
from shafi.core.circuit_breaker import CircuitBreaker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    latency_ms: float
    provider: str = ""
    finish_reason: str = ""
    cached_tokens: int = 0


class LLMProviderError(RuntimeError):
    pass


class LLMProvider:
    """Async LLM provider with streaming and model cascade support."""

    def __init__(self, openai_client: AsyncOpenAI | None = None) -> None:
        self._settings = get_settings().llm
        self._external_client = openai_client is not None
        resolved_api_key = self._resolve_api_key_value()
        self._http_client: httpx.AsyncClient | None = None
        if openai_client is not None:
            self._openai = openai_client
        else:
            self._http_client = httpx.AsyncClient(
                http2=bool(getattr(self._settings, "http2", True)),
                limits=httpx.Limits(
                    max_connections=int(getattr(self._settings, "max_connections", 50)),
                    max_keepalive_connections=int(getattr(self._settings, "max_keepalive_connections", 20)),
                ),
                timeout=httpx.Timeout(
                    float(self._settings.timeout_s),
                    connect=float(getattr(self._settings, "connect_timeout_s", 10.0)),
                ),
            )
            self._openai = AsyncOpenAI(
                api_key=resolved_api_key,
                base_url=self._settings.base_url,
                default_headers=self._resolve_default_headers(),
                http_client=self._http_client,
            )
        self._circuits: dict[str, CircuitBreaker] = {}
        self._last_stream_model: str = ""
        self._last_stream_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
        self._last_stream_provider: str = ""
        self._last_stream_finish_reason: str = ""

    async def close(self) -> None:
        if not self._external_client and hasattr(self._openai, "close"):
            await self._openai.close()
        if self._http_client is not None:
            await self._http_client.aclose()

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prediction: dict[str, str] | None = None,
    ) -> LLMResult:
        chosen_model = model or self._settings.simple_model
        chosen_max_tokens = max_tokens if max_tokens is not None else self._settings.simple_max_tokens
        chosen_temperature = temperature if temperature is not None else self._settings.temperature
        messages = self._build_messages(system_prompt=system_prompt, user_prompt=user_prompt)
        circuit = self._get_circuit(chosen_model)
        if not circuit.allow_request():
            raise LLMProviderError(f"LLM circuit open ({chosen_model})")

        t0 = time.perf_counter()
        create = cast("Any", self._openai.chat.completions.create)
        extra_body = self._openrouter_provider_hints()
        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "temperature": chosen_temperature,
            "stream": False,
            "extra_body": extra_body,
        }
        if prediction is not None:
            kwargs["prediction"] = prediction
        if "gpt-5" in chosen_model or "o1" in chosen_model or "o3" in chosen_model:
            kwargs["max_completion_tokens"] = chosen_max_tokens
        else:
            kwargs["max_tokens"] = chosen_max_tokens

        try:
            resp: Any
            try:
                resp = await create(**kwargs)
            except Exception as exc:
                if extra_body is not None and self._is_openrouter_provider_hint_rejected(exc):
                    logger.warning("openrouter_provider_hint_rejected", exc_info=True)
                    kwargs.pop("extra_body", None)
                    resp = await create(**kwargs)
                else:
                    raise
        except Exception as exc:
            circuit.record_failure()
            raise LLMProviderError(f"LLM call failed ({chosen_model}): {exc}") from exc
        circuit.record_success()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = self._extract_non_stream_text(resp)
        usage = getattr(resp, "usage", None)
        prompt_tokens = self._usage_int(usage, "prompt_tokens")
        completion_tokens = self._usage_int(usage, "completion_tokens")
        total_tokens = self._usage_int(usage, "total_tokens")
        cached_tokens = self._extract_cached_tokens(usage)
        provider = self._extract_provider_name(resp) or self._transport_provider_name()
        finish_reason = self._extract_finish_reason(resp)

        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            model=chosen_model,
            provider=provider,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )

    async def stream_generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prediction: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        chosen_model = model or self._settings.simple_model
        chosen_max_tokens = max_tokens if max_tokens is not None else self._settings.simple_max_tokens
        chosen_temperature = temperature if temperature is not None else self._settings.temperature
        messages = self._build_messages(system_prompt=system_prompt, user_prompt=user_prompt)
        self._last_stream_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
        self._last_stream_model = ""
        self._last_stream_provider = ""
        self._last_stream_finish_reason = ""
        circuit = self._get_circuit(chosen_model)
        if not circuit.allow_request():
            raise LLMProviderError(f"LLM circuit open ({chosen_model})")

        create = cast("Any", self._openai.chat.completions.create)
        extra_body = self._openrouter_provider_hints()
        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "temperature": chosen_temperature,
            "stream": True,
            "stream_options": {"include_usage": self._settings.stream_include_usage},
            "extra_body": extra_body,
        }
        if prediction is not None:
            kwargs["prediction"] = prediction
        if "gpt-5" in chosen_model or "o1" in chosen_model or "o3" in chosen_model:
            kwargs["max_completion_tokens"] = chosen_max_tokens
        else:
            kwargs["max_tokens"] = chosen_max_tokens

        try:
            stream: Any
            try:
                stream = await create(**kwargs)
            except Exception as exc:
                if extra_body is not None and self._is_openrouter_provider_hint_rejected(exc):
                    logger.warning("openrouter_provider_hint_rejected", exc_info=True)
                    kwargs.pop("extra_body", None)
                    stream = await create(**kwargs)
                else:
                    raise
        except Exception as exc:
            circuit.record_failure()
            raise LLMProviderError(f"LLM stream failed ({chosen_model}): {exc}") from exc

        self._last_stream_model = chosen_model
        self._last_stream_provider = self._transport_provider_name()
        try:
            async for chunk in stream:
                chunk_model = getattr(chunk, "model", None)
                if isinstance(chunk_model, str) and chunk_model.strip():
                    self._last_stream_model = chunk_model.strip()
                provider = self._extract_provider_name(chunk)
                if provider:
                    self._last_stream_provider = provider
                finish_reason = self._extract_finish_reason(chunk)
                if finish_reason:
                    self._last_stream_finish_reason = finish_reason
                delta_text = self._extract_stream_delta_text(chunk)
                if delta_text:
                    yield delta_text

                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    self._last_stream_usage = {
                        "prompt_tokens": self._usage_int(usage, "prompt_tokens"),
                        "completion_tokens": self._usage_int(usage, "completion_tokens"),
                        "total_tokens": self._usage_int(usage, "total_tokens"),
                        "cached_tokens": self._extract_cached_tokens(usage),
                    }
        except Exception as exc:
            circuit.record_failure()
            raise LLMProviderError(f"LLM stream failed ({chosen_model}): {exc}") from exc
        circuit.record_success()

    async def stream_generate_with_cascade(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        models: Sequence[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prediction: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        cascade_models = list(models) if models is not None else self._default_cascade_models()
        last_error: Exception | None = None

        for model in cascade_models:
            started = False
            try:
                async for token in self.stream_generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    prediction=prediction,
                ):
                    started = True
                    yield token
                return
            except LLMProviderError as exc:
                if started:
                    raise
                logger.warning("Streaming model %s failed to start; trying next: %s", model, exc)
                last_error = exc

        raise LLMProviderError(f"All models in stream cascade failed. Last error: {last_error}")

    async def generate_with_cascade(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        models: Sequence[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prediction: dict[str, str] | None = None,
    ) -> LLMResult:
        cascade_models = list(models) if models is not None else self._default_cascade_models()
        last_error: Exception | None = None

        for model in cascade_models:
            try:
                return await self.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    prediction=prediction,
                )
            except LLMProviderError as exc:
                logger.warning("Model %s failed; trying next: %s", model, exc)
                last_error = exc

        raise LLMProviderError(f"All models in cascade failed. Last error: {last_error}")

    def get_last_stream_usage(self) -> dict[str, int]:
        return dict(self._last_stream_usage)

    def get_last_stream_model(self) -> str:
        return str(self._last_stream_model or "")

    def get_last_stream_provider(self) -> str:
        return str(self._last_stream_provider or "")

    def get_last_stream_finish_reason(self) -> str:
        return str(self._last_stream_finish_reason or "")

    def get_last_stream_cached_tokens(self) -> int:
        return int(self._last_stream_usage.get("cached_tokens", 0))

    def _default_cascade_models(self) -> list[str]:
        return [
            self._settings.simple_model,
            self._settings.complex_model,
            self._settings.fallback_model,
        ]

    def _get_circuit(self, model: str) -> CircuitBreaker:
        circuit = self._circuits.get(model)
        if circuit is None:
            circuit = CircuitBreaker(
                name=f"llm:{model}",
                failure_threshold=int(self._settings.circuit_failure_threshold),
                reset_timeout_s=float(self._settings.circuit_reset_timeout_s),
            )
            self._circuits[model] = circuit
        return circuit

    def _resolve_default_headers(self) -> dict[str, str] | None:
        headers: dict[str, str] = {}
        referer = str(getattr(self._settings, "openrouter_referer", "")).strip()
        title = str(getattr(self._settings, "openrouter_title", "")).strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        return headers or None

    def _openrouter_provider_hints(self) -> dict[str, object] | None:
        base_url = str(getattr(self._settings, "base_url", "")).lower()
        if "openrouter.ai" not in base_url:
            return None

        order_obj = getattr(self._settings, "openrouter_provider_order", [])
        order_items = cast("list[object]", order_obj) if isinstance(order_obj, list) else []
        order = [str(item).strip() for item in order_items]
        order = [item for item in order if item]
        allow_fallbacks = bool(getattr(self._settings, "openrouter_allow_fallbacks", True))

        provider: dict[str, object] = {"allow_fallbacks": allow_fallbacks}
        if order:
            provider["order"] = order
        return {"provider": provider} if provider else None

    @staticmethod
    def _is_openrouter_provider_hint_rejected(exc: Exception) -> bool:
        # Avoid importing openai exception types; detect via status_code/message.
        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code != 400:
            return False
        text = str(exc).lower()
        return "provider" in text or "allow_fallbacks" in text or "order" in text

    def _resolve_api_key_value(self) -> str:
        resolver = getattr(self._settings, "resolved_api_key", None)
        if callable(resolver):
            secret = resolver()
            getter = getattr(secret, "get_secret_value", None)
            if callable(getter):
                value = getter()
                if isinstance(value, str):
                    return value

        for attr in ("api_key", "openrouter_api_key", "openai_api_key"):
            secret = getattr(self._settings, attr, None)
            getter = getattr(secret, "get_secret_value", None)
            if callable(getter):
                value = getter()
                if isinstance(value, str) and value.strip():
                    return value
        return ""

    @staticmethod
    def _build_messages(*, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_non_stream_text(resp: Any) -> str:
        choices_obj: object = getattr(resp, "choices", None)
        if not isinstance(choices_obj, list) or not choices_obj:
            raise LLMProviderError("LLM response missing choices")
        choices = cast("list[object]", choices_obj)
        first_choice: object = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # Some SDK variants can return structured segments; keep MVP simple but safe.
        return str(content)

    @staticmethod
    def _extract_stream_delta_text(chunk: Any) -> str:
        choices_obj: object = getattr(chunk, "choices", None)
        if not isinstance(choices_obj, list) or not choices_obj:
            return ""
        choices = cast("list[object]", choices_obj)
        first_choice: object = choices[0]
        delta = getattr(first_choice, "delta", None)
        content = getattr(delta, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    @staticmethod
    def _extract_finish_reason(resp_or_chunk: Any) -> str:
        choices_obj: object = getattr(resp_or_chunk, "choices", None)
        if not isinstance(choices_obj, list) or not choices_obj:
            return ""
        first_choice: object = cast("list[object]", choices_obj)[0]
        finish_reason = getattr(first_choice, "finish_reason", None)
        return finish_reason.strip() if isinstance(finish_reason, str) and finish_reason.strip() else ""

    def _extract_provider_name(self, resp_or_chunk: Any) -> str:
        for attr in ("provider", "openrouter_provider", "service_tier"):
            value = getattr(resp_or_chunk, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _transport_provider_name(self) -> str:
        base_url = str(getattr(self._settings, "base_url", "")).strip()
        if not base_url:
            return ""
        lowered = base_url.lower()
        if "openrouter.ai" in lowered:
            return "openrouter"
        if "api.openai.com" in lowered:
            return "openai"
        host = urlparse(base_url).hostname or ""
        return host.strip()

    @staticmethod
    def _extract_cached_tokens(usage: Any) -> int:
        """Extract cached_tokens from OpenAI usage.prompt_tokens_details."""
        details = getattr(usage, "prompt_tokens_details", None)
        if details is None:
            return 0
        value = getattr(details, "cached_tokens", 0)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return 0

    @staticmethod
    def _usage_int(usage: Any, field: str) -> int:
        value = getattr(usage, field, 0)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return 0
