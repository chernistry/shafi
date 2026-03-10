from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from rag_challenge.config import get_settings
from rag_challenge.core.circuit_breaker import CircuitBreaker
from rag_challenge.prompts.loader import load_prompt

logger = logging.getLogger(__name__)


def _grounding_evidence_factory() -> list[GroundingEvidence]:
    return []


class JudgeScores(BaseModel):
    accuracy: int = Field(ge=0, le=5)
    grounding: int = Field(ge=0, le=5)
    clarity: int = Field(ge=0, le=5)
    uncertainty_handling: int = Field(ge=0, le=5)


class GroundingEvidence(BaseModel):
    claim: str
    support_excerpt: str


class JudgeResult(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    scores: JudgeScores
    format_issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    grounding_evidence: list[GroundingEvidence] = Field(default_factory=_grounding_evidence_factory)
    recommended_fix: str = ""


@dataclass(frozen=True)
class JudgeOutcome:
    """Wrapper used by the harness to track judge failures separately."""

    result: JudgeResult | None
    model: str
    failure: str = ""


_JUDGE_SYSTEM_PROMPT = load_prompt("eval/judge_system")
_JUDGE_SYSTEM_STRICT_JSON_SUFFIX = load_prompt("eval/judge_system_strict_json_suffix")
_JUDGE_USER_PROMPT_TEMPLATE = load_prompt("eval/judge_user")


def parse_judge_result(raw_text: str) -> JudgeResult:
    """Parse judge output into `JudgeResult`, handling common non-JSON wrappers."""
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty judge response")

    candidate = _strip_code_fences(text)
    obj = _extract_first_json_object(candidate)
    return JudgeResult.model_validate(obj)


class JudgeClient:
    def __init__(self) -> None:
        settings = get_settings().judge
        self._settings = settings

        self._http_client = httpx.AsyncClient(
            http2=bool(getattr(settings, "http2", True)),
            limits=httpx.Limits(
                max_connections=int(getattr(settings, "max_connections", 20)),
                max_keepalive_connections=int(getattr(settings, "max_keepalive_connections", 10)),
            ),
            timeout=httpx.Timeout(
                float(getattr(settings, "timeout_s", 60.0)),
                connect=float(getattr(settings, "connect_timeout_s", 10.0)),
            ),
        )

        self._client = AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=str(settings.base_url),
            default_headers=self._default_headers(),
            http_client=self._http_client,
        )

        self._circuit = CircuitBreaker(
            name="judge",
            failure_threshold=3,
            reset_timeout_s=60.0,
        )

    async def close(self) -> None:
        await self._client.close()
        await self._http_client.aclose()

    async def evaluate(
        self,
        *,
        question: str,
        answer_type: str,
        answer: str,
        used_pages: list[str],
        sources_text: str,
    ) -> JudgeOutcome:
        if not self._circuit.allow_request():
            return JudgeOutcome(result=None, model=str(self._settings.model), failure="judge_circuit_open")

        user_prompt = _build_user_prompt(
            question=question,
            answer_type=answer_type,
            answer=answer,
            used_pages=used_pages,
            sources_text=sources_text,
        )

        extra_body = self._openrouter_provider_hints()

        # Attempt 1: normal judge prompt.
        outcome = await self._call_once(
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            extra_body=extra_body,
        )
        if outcome.result is not None:
            self._circuit.record_success()
            return outcome

        # Attempt 2: stricter instruction for models that wrap JSON.
        strict_system = f"{_JUDGE_SYSTEM_PROMPT}\n{_JUDGE_SYSTEM_STRICT_JSON_SUFFIX}"
        outcome_2 = await self._call_once(
            system_prompt=strict_system,
            user_prompt=user_prompt,
            extra_body=extra_body,
        )
        if outcome_2.result is not None:
            self._circuit.record_success()
            return outcome_2

        self._circuit.record_failure()
        return outcome_2

    async def _call_once(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        extra_body: dict[str, object] | None,
    ) -> JudgeOutcome:
        import asyncio
        model = str(self._settings.model)
        create = self._client.chat.completions.create  # type: ignore[reportUnknownMemberType]

        for attempt in range(5):
            try:
                resp: Any
                try:
                    resp = await create(
                        model=model,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        max_tokens=int(self._settings.max_tokens),
                        temperature=float(self._settings.temperature),
                        stream=False,
                        extra_body=extra_body,
                    )
                except Exception as exc:
                    if extra_body is not None and _is_openrouter_provider_hint_rejected(exc):
                        logger.warning("openrouter_provider_hint_rejected", exc_info=True)
                        resp = await create(
                            model=model,
                            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                            max_tokens=int(self._settings.max_tokens),
                            temperature=float(self._settings.temperature),
                            stream=False,
                        )
                    else:
                        raise

                content = _extract_non_stream_text(resp)
                try:
                    parsed = parse_judge_result(content)
                except (ValueError, json.JSONDecodeError, ValidationError) as parse_exc:
                    return JudgeOutcome(result=None, model=model, failure=f"judge_parse_error: {parse_exc}")
                return JudgeOutcome(result=parsed, model=model)

            except Exception as exc:
                exc_str = str(exc).lower()
                if attempt < 4 and ("429" in exc_str or "rate limit" in exc_str or "timeout" in exc_str.lower() or "502" in exc_str or "503" in exc_str):
                    wait_time = 2 ** attempt * 2  # 2, 4, 8, 16 seconds
                    logger.warning(f"Judge API error on attempt {attempt + 1}, retrying in {wait_time}s: {exc}")
                    await asyncio.sleep(wait_time)
                else:
                    return JudgeOutcome(result=None, model=model, failure=f"judge_call_failed: {exc}")
        return JudgeOutcome(result=None, model=model, failure="judge_call_failed: max retries reached")

    def _default_headers(self) -> dict[str, str] | None:
        # OpenRouter uses these headers for attribution; harmless for other providers.
        headers: dict[str, str] = {}
        base_url = str(self._settings.base_url).lower()
        if "openrouter.ai" in base_url:
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = "rag-challenge-judge"
        return headers or None

    def _openrouter_provider_hints(self) -> dict[str, object] | None:
        base_url = str(self._settings.base_url).lower()
        if "openrouter.ai" not in base_url:
            return None

        order = [item.strip() for item in list(getattr(self._settings, "openrouter_provider_order", [])) if item.strip()]
        allow_fallbacks = bool(getattr(self._settings, "openrouter_allow_fallbacks", True))
        provider: dict[str, object] = {"allow_fallbacks": allow_fallbacks}
        if order:
            provider["order"] = order
        return {"provider": provider} if provider else None


def _build_user_prompt(
    *,
    question: str,
    answer_type: str,
    answer: str,
    used_pages: list[str],
    sources_text: str,
) -> str:
    used_pages_json = json.dumps(list(used_pages), ensure_ascii=False)
    return _JUDGE_USER_PROMPT_TEMPLATE.format(
        question=question,
        answer_type=answer_type,
        answer=answer,
        used_pages=used_pages_json,
        sources_text=sources_text,
    )


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    # Prefer the first fenced block that contains JSON.
    import re

    m = re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if m is not None:
        return m.group(1).strip()
    return stripped.replace("```json", "").replace("```", "").strip()


def _extract_first_json_object(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        if isinstance(obj, dict):
            return cast("dict[str, object]", obj)
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _extract_non_stream_text(resp: Any) -> str:
    choices_obj: object = getattr(resp, "choices", None)
    if not isinstance(choices_obj, list) or not choices_obj:
        return ""
    first_choice = cast("list[object]", choices_obj)[0]
    message = getattr(first_choice, "message", None)
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def _is_openrouter_provider_hint_rejected(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code != 400:
        return False
    text = str(exc).lower()
    return "provider" in text or "allow_fallbacks" in text or "order" in text
