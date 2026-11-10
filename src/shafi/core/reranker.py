from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from random import uniform
from typing import TYPE_CHECKING, Any, cast

import httpx
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt

from shafi.config import get_settings
from shafi.core.circuit_breaker import CircuitBreaker, CircuitState
from shafi.core.local_cross_encoder_reranker import LocalCrossEncoderReranker
from shafi.core.local_flashrank_reranker import LocalFlashRankReranker
from shafi.core.request_limiter import ClockFunc, SleepFunc, get_shared_request_limiter
from shafi.core.rerank_instructions import compose_instruction_conditioned_query
from shafi.models import RankedChunk, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = ["CircuitBreaker", "CircuitState", "RerankerClient", "RerankerError"]


class RerankerError(RuntimeError):
    pass


def _ranked_chunk_sort_key(chunk: RankedChunk) -> tuple[float, float, str, str, str]:
    """Return a deterministic sort key for ranked chunks.

    Args:
        chunk: Ranked chunk candidate to order.

    Returns:
        tuple[float, float, str, str, str]: Sort key that preserves score
        priority while breaking ties deterministically on stable metadata.
    """

    return (
        -float(chunk.rerank_score),
        -float(chunk.retrieval_score),
        str(chunk.doc_id),
        str(chunk.section_path),
        str(chunk.chunk_id),
    )


class _RetryableRerankerError(RerankerError):
    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


@dataclass(frozen=True, slots=True)
class _RerankProviderResult:
    """Provider rerank output with optional confidence metadata."""

    scores: list[float]
    confidence: float | None = None


class RerankerClient:
    """Zerank 2 primary reranker with Cohere fallback."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        cohere_client: Any | None = None,
        sleep_func: SleepFunc = asyncio.sleep,
        clock_func: ClockFunc = time.monotonic,
    ) -> None:
        settings = get_settings()
        self._settings = settings.reranker
        self._pipeline_settings = getattr(settings, "pipeline", None)
        self._external_client = client is not None
        self._client = client or self._make_client()
        self._cohere = cohere_client if cohere_client is not None else self._make_cohere_client()
        self._sleep = sleep_func
        self._clock = clock_func
        self._local_reranker: LocalCrossEncoderReranker | None = None
        self._flashrank_reranker: LocalFlashRankReranker | None = None
        self._shadow_selective_icr_reranker: Any | None = None
        self._last_used_model = self._settings.primary_model
        self._last_rerank_confidence = 0.0
        self._last_instruction_family = ""
        self._last_shadow_used_model = ""
        self._last_shadow_latency_ms = 0
        self._last_shadow_candidate_count = 0
        self._last_shadow_chunk_ids: list[str] = []
        self._last_shadow_page_ids: list[str] = []
        self._circuit = CircuitBreaker(
            name="zerank-reranker",
            failure_threshold=int(self._settings.circuit_failure_threshold),
            reset_timeout_s=float(self._settings.circuit_reset_timeout_s),
        )
        self._primary_request_limiter = get_shared_request_limiter(
            "zerank-reranker-primary",
            concurrency_limit=int(self._settings.primary_concurrency_limit),
            min_interval_s=float(self._settings.primary_min_interval_s),
            sleep_func=self._sleep,
            clock=self._clock,
        )

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=int(self._settings.primary_max_connections),
                max_keepalive_connections=int(self._settings.primary_max_connections),
            ),
            timeout=httpx.Timeout(self._settings.primary_timeout_s, connect=self._settings.primary_connect_timeout_s),
        )

    def _make_cohere_client(self) -> Any:
        try:
            import cohere
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency exists in normal env
            raise RerankerError("cohere package is required for fallback reranker support") from exc

        return cohere.AsyncClientV2(
            api_key=self._settings.fallback_api_key.get_secret_value(),
            timeout=self._settings.fallback_timeout_s,
        )

    async def close(self) -> None:
        if not self._external_client:
            await self._client.aclose()

    async def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_n: int | None = None,
        *,
        prefer_fast: bool = False,
        instruction: str | None = None,
        instruction_family: str = "",
    ) -> list[RankedChunk]:
        """Rerank retrieved chunks using Zerank with optional instruction control.

        Args:
            query: Raw query text.
            chunks: Retrieved chunks to rerank.
            top_n: Output cap.
            prefer_fast: Whether to try the fast fallback reranker first.
            instruction: Optional deterministic rerank instruction.
            instruction_family: Short label for the active instruction family.

        Returns:
            Ranked chunk list trimmed to ``top_n``.
        """
        if top_n is None:
            top_n = self._settings.top_n

        if not chunks:
            self._last_instruction_family = instruction_family
            self._last_rerank_confidence = 0.0
            return []

        max_chars = int(getattr(self._pipeline_settings, "rerank_doc_max_chars", 1500))
        documents = [chunk.text[:max_chars] for chunk in chunks] if max_chars > 0 else [chunk.text for chunk in chunks]
        conditioned_query = compose_instruction_conditioned_query(query, instruction)
        self._last_instruction_family = instruction_family

        if self._use_local_model():
            try:
                result = await self._local_cross_encoder_rerank(conditioned_query, documents)
                self._last_used_model = str(getattr(self._settings, "local_model_path", "") or "local_cross_encoder")
                self._last_rerank_confidence = self._resolve_confidence(result.scores, result.confidence)
                return self._build_ranked(chunks, result.scores, top_n)
            except Exception as exc:
                logger.warning("Local cross-encoder reranker failed; degrading to raw retrieval scores: %s", exc)
                self._last_used_model = "raw_retrieval_fallback"
                self._last_rerank_confidence = 0.0
                return self._build_raw_score_fallback(chunks, top_n)

        if self._use_flashrank_model():
            try:
                result = await self._flashrank_rerank(conditioned_query, documents)
                model_name = str(getattr(self._settings, "local_model_path", "") or "ms-marco-MiniLM-L-12-v2")
                self._last_used_model = f"flashrank:{model_name}"
                self._last_rerank_confidence = self._resolve_confidence(result.scores, result.confidence)
                return self._build_ranked(chunks, result.scores, top_n)
            except Exception as exc:
                logger.warning("FlashRank reranker failed; degrading to raw retrieval scores: %s", exc)
                self._last_used_model = "raw_retrieval_fallback"
                self._last_rerank_confidence = 0.0
                return self._build_raw_score_fallback(chunks, top_n)

        if self._use_isaacus_model():
            try:
                result = await self._isaacus_rerank(conditioned_query, documents)
                self._last_used_model = self._settings.isaacus_model
                self._last_rerank_confidence = self._resolve_confidence(result.scores, result.confidence)
                return self._build_ranked(chunks, result.scores, top_n)
            except Exception as exc:
                logger.warning("Isaacus reranker failed; falling back to Cohere: %s", exc)
            try:
                result = await self._cohere_rerank(conditioned_query, documents)
                self._last_used_model = self._settings.fallback_model
                self._last_rerank_confidence = self._resolve_confidence(result.scores, result.confidence)
                return self._build_ranked(chunks, result.scores, top_n)
            except Exception as exc2:
                logger.error("Isaacus + Cohere both failed; degrading to raw retrieval scores: %s", exc2)
                self._last_used_model = "raw_retrieval_fallback"
                self._last_rerank_confidence = 0.0
                return self._build_raw_score_fallback(chunks, top_n)

        if prefer_fast:
            try:
                result = await self._cohere_rerank(conditioned_query, documents)
                self._last_used_model = self._settings.fallback_model
                self._last_rerank_confidence = self._resolve_confidence(result.scores, result.confidence)
                return self._build_ranked(chunks, result.scores, top_n)
            except Exception as exc:
                logger.warning("Fast reranker path failed; falling back to primary: %s", exc)

        if self._circuit.allow_request():
            try:
                result = await self._zerank_rerank(conditioned_query, documents)
                self._circuit.record_success()
                self._last_used_model = self._settings.primary_model
                self._last_rerank_confidence = self._resolve_confidence(result.scores, result.confidence)
                return self._build_ranked(chunks, result.scores, top_n)
            except Exception as exc:
                logger.warning("Zerank reranker failed, using Cohere fallback: %s", exc)
                self._circuit.record_failure()
        else:
            logger.info("Zerank reranker circuit open; skipping to Cohere fallback")

        try:
            result = await self._cohere_rerank(conditioned_query, documents)
            self._last_used_model = self._settings.fallback_model
            self._last_rerank_confidence = self._resolve_confidence(result.scores, result.confidence)
            return self._build_ranked(chunks, result.scores, top_n)
        except Exception as exc:
            logger.error("Both rerankers failed; degrading to raw retrieval scores: %s", exc)
            self._last_used_model = "raw_retrieval_fallback"
            self._last_rerank_confidence = 0.0
            return self._build_raw_score_fallback(chunks, top_n)

    def _use_local_model(self) -> bool:
        return str(getattr(self._settings, "provider_mode", "api")).casefold() == "local"

    def _use_isaacus_model(self) -> bool:
        return str(getattr(self._settings, "provider_mode", "api")).casefold() == "isaacus"

    def _use_flashrank_model(self) -> bool:
        return str(getattr(self._settings, "provider_mode", "api")).casefold() == "flashrank"

    async def _local_cross_encoder_rerank(self, query: str, documents: list[str]) -> _RerankProviderResult:
        if self._local_reranker is None:
            model_path = str(getattr(self._settings, "local_model_path", "")).strip()
            if not model_path:
                raise RerankerError("RERANK_LOCAL_MODEL_PATH is required when RERANK_PROVIDER_MODE=local")
            self._local_reranker = LocalCrossEncoderReranker(model_path=model_path)
        scores = await asyncio.to_thread(self._local_reranker.score_documents, query=query, documents=documents)
        return _RerankProviderResult(scores=scores, confidence=None)

    async def _flashrank_rerank(self, query: str, documents: list[str]) -> _RerankProviderResult:
        """Score documents using local FlashRank ONNX model (~20ms, no API call).

        Args:
            query: Query (may include instruction prefix from compose_instruction_conditioned_query).
            documents: Document texts to score, up to rerank_candidates length.

        Returns:
            _RerankProviderResult with scores in the same order as ``documents``.
        """
        if self._flashrank_reranker is None:
            model_name = str(getattr(self._settings, "local_model_path", "") or "ms-marco-MiniLM-L-12-v2").strip()
            cache_dir = str(getattr(self._settings, "flashrank_cache_dir", "/tmp/flashrank_models")).strip()
            self._flashrank_reranker = LocalFlashRankReranker(model_name=model_name, cache_dir=cache_dir)
        scores = await asyncio.to_thread(self._flashrank_reranker.score_documents, query=query, documents=documents)
        return _RerankProviderResult(scores=scores, confidence=None)

    def get_last_used_model(self) -> str:
        return self._last_used_model

    def get_last_rerank_confidence(self) -> float:
        """Return the last rerank confidence estimate in ``[0.0, 1.0]``."""

        return self._last_rerank_confidence

    def get_last_instruction_family(self) -> str:
        """Return the last rerank instruction family label."""

        return self._last_instruction_family

    def get_last_shadow_used_model(self) -> str:
        """Return the last selective ICR shadow model label."""

        return self._last_shadow_used_model

    def get_last_shadow_latency_ms(self) -> int:
        """Return the last selective ICR shadow latency in milliseconds."""

        return self._last_shadow_latency_ms

    def get_last_shadow_candidate_count(self) -> int:
        """Return the last selective ICR shadow candidate count."""

        return self._last_shadow_candidate_count

    def get_last_shadow_chunk_ids(self) -> list[str]:
        """Return the last selective ICR shadow chunk IDs."""

        return list(self._last_shadow_chunk_ids)

    def get_last_shadow_page_ids(self) -> list[str]:
        """Return the last selective ICR shadow page IDs."""

        return list(self._last_shadow_page_ids)

    async def shadow_rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_n: int | None = None,
        *,
        instruction: str | None = None,
        instruction_family: str = "",
    ) -> list[RankedChunk]:
        """Run the selective ICR shadow lane without affecting the main path.

        Args:
            query: Raw query text.
            chunks: Candidate chunks to score.
            top_n: Output cap.
            instruction: Optional deterministic rerank instruction.
            instruction_family: Short label for the active instruction family.

        Returns:
            Shadow-ranked chunks with the same output shape as the main reranker.
        """

        if top_n is None:
            top_n = self._settings.top_n
        if not chunks:
            self._last_shadow_used_model = ""
            self._last_shadow_latency_ms = 0
            self._last_shadow_candidate_count = 0
            self._last_shadow_chunk_ids = []
            self._last_shadow_page_ids = []
            return []

        from shafi.core.selective_icr_reranker import SelectiveICRConfig, SelectiveICRReranker

        if self._shadow_selective_icr_reranker is None:
            self._shadow_selective_icr_reranker = SelectiveICRReranker(
                config=SelectiveICRConfig(
                    model_path=str(getattr(self._settings, "shadow_selective_icr_model_path", "") or ""),
                    max_chars=int(getattr(self._settings, "shadow_selective_icr_max_chars", 1800)),
                    normalize_scores=bool(getattr(self._settings, "shadow_selective_icr_normalize_scores", True)),
                    provider_exit=bool(getattr(self._settings, "shadow_selective_icr_provider_exit", False)),
                )
            )

        conditioned_query = compose_instruction_conditioned_query(query, instruction)
        t0 = self._clock()
        scored = self._shadow_selective_icr_reranker.rank(conditioned_query, chunks, top_n=top_n)
        latency_ms = max(0, int((self._clock() - t0) * 1000.0))
        self._last_shadow_used_model = self._shadow_selective_icr_reranker.model_name
        self._last_shadow_latency_ms = latency_ms
        self._last_shadow_candidate_count = len(chunks)
        self._last_shadow_chunk_ids = [item.chunk_id for item in scored]
        self._last_shadow_page_ids = [item.page_id for item in scored]
        self._last_instruction_family = instruction_family

        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        ranked: list[RankedChunk] = []
        for item in scored:
            chunk = chunk_by_id.get(item.chunk_id, chunks[0])
            ranked.append(
                RankedChunk(
                    chunk_id=item.chunk_id,
                    doc_id=item.doc_id,
                    doc_title=chunk.doc_title,
                    doc_type=chunk.doc_type,
                    section_path=item.section_path,
                    text=item.text,
                    retrieval_score=item.retrieval_score,
                    rerank_score=item.rerank_score,
                    doc_summary=chunk.doc_summary,
                    page_family=item.page_family,
                    doc_family=item.doc_family,
                    chunk_type=item.chunk_type,
                    amount_roles=list(item.amount_roles),
                    normalized_refs=list(chunk.normalized_refs),
                    law_titles=list(chunk.law_titles),
                    article_refs=list(chunk.article_refs),
                    case_numbers=list(chunk.case_numbers),
                )
            )
        return ranked

    async def _zerank_rerank(self, query: str, documents: list[str]) -> _RerankProviderResult:
        scores: list[float] = [0.0] * len(documents)
        confidences: list[float] = []
        batch_size = max(1, int(self._settings.primary_batch_size))

        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            response = await self._zerank_post(query, batch)
            confidence = self._extract_response_confidence(response)
            if confidence is not None:
                confidences.extend([confidence] * len(batch))
            results_obj = response.get("results")
            if not isinstance(results_obj, list):
                raise RerankerError("Zerank response missing 'results' list")

            for item_obj in cast("list[object]", results_obj):
                if not isinstance(item_obj, dict):
                    raise RerankerError("Zerank response contains non-object result")
                item = cast("dict[str, object]", item_obj)
                index = item.get("index")
                score_obj = item.get("score", item.get("relevance_score"))
                if not isinstance(index, int) or not (0 <= index < len(batch)):
                    raise RerankerError("Zerank result index is invalid")
                if not isinstance(score_obj, (int, float, str)):
                    raise RerankerError("Zerank result score is invalid")
                try:
                    score = float(score_obj)
                except (TypeError, ValueError) as exc:
                    raise RerankerError("Zerank result score is invalid") from exc
                scores[start + index] = score

        provider_confidence = sum(confidences) / len(confidences) if confidences else None
        return _RerankProviderResult(scores=scores, confidence=provider_confidence)

    async def _zerank_post(self, query: str, documents: list[str]) -> dict[str, object]:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_RetryableRerankerError),
            stop=stop_after_attempt(max(1, int(self._settings.retry_attempts))),
            wait=self._retry_wait_seconds,
            reraise=True,
            sleep=self._sleep,
        ):
            with attempt:
                return await self._zerank_post_once(query=query, documents=documents)
        raise AssertionError("unreachable")

    async def _zerank_post_once(self, *, query: str, documents: list[str]) -> dict[str, object]:
        payload = {
            "model": self._settings.primary_model,
            "query": query,
            "documents": documents,
            "latency": self._settings.primary_latency_mode,
        }
        headers = {"Authorization": f"Bearer {self._settings.primary_api_key.get_secret_value()}"}

        async with self._primary_request_limiter.acquire():
            try:
                resp = await self._client.post(self._settings.primary_api_url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                raise _RetryableRerankerError("Zerank request timed out") from exc
            except httpx.TransportError as exc:
                raise _RetryableRerankerError("Zerank transport error") from exc

        if resp.status_code < 400:
            data = resp.json()
            if not isinstance(data, dict):
                raise RerankerError("Zerank returned non-object JSON")
            return cast("dict[str, object]", data)

        if resp.status_code in (429, 500, 502, 503, 504):
            raise _RetryableRerankerError(
                f"Zerank transient error {resp.status_code}",
                retry_after_s=self._parse_retry_after_seconds(resp),
            )

        raise RerankerError(f"Zerank error {resp.status_code}: {resp.text[:500]}")

    async def _isaacus_rerank(self, query: str, documents: list[str]) -> _RerankProviderResult:
        """Rerank using Isaacus kanon-2-reranker — legal-specific, same API key as embeddings."""
        scores: list[float] = [0.0] * len(documents)
        batch_size = max(1, int(self._settings.primary_batch_size))

        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            response = await self._isaacus_post(query, batch)
            results_obj = response.get("results")
            if not isinstance(results_obj, list):
                raise RerankerError("Isaacus response missing 'results' list")

            for item_obj in cast("list[object]", results_obj):
                if not isinstance(item_obj, dict):
                    raise RerankerError("Isaacus response contains non-object result")
                item = cast("dict[str, object]", item_obj)
                index = item.get("index")
                score_obj = item.get("score")
                if not isinstance(index, int) or not (0 <= index < len(batch)):
                    raise RerankerError("Isaacus result index is invalid")
                if not isinstance(score_obj, (int, float, str)):
                    raise RerankerError("Isaacus result score is invalid")
                try:
                    score = float(score_obj)
                except (TypeError, ValueError) as exc:
                    raise RerankerError("Isaacus result score is invalid") from exc
                scores[start + index] = score

        return _RerankProviderResult(scores=scores, confidence=None)

    async def _isaacus_post(self, query: str, documents: list[str]) -> dict[str, object]:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_RetryableRerankerError),
            stop=stop_after_attempt(max(1, int(self._settings.retry_attempts))),
            wait=self._retry_wait_seconds,
            reraise=True,
            sleep=self._sleep,
        ):
            with attempt:
                return await self._isaacus_post_once(query=query, documents=documents)
        raise AssertionError("unreachable")

    async def _isaacus_post_once(self, *, query: str, documents: list[str]) -> dict[str, object]:
        api_key_secret = self._settings.isaacus_api_key
        api_key = api_key_secret.get_secret_value()
        if not api_key:
            raise RerankerError("ISAACUS_API_KEY is required when RERANK_PROVIDER_MODE=isaacus")

        payload: dict[str, object] = {
            "model": self._settings.isaacus_model,
            "query": query,
            "texts": documents,
            "top_n": len(documents),
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        async with self._primary_request_limiter.acquire():
            try:
                resp = await self._client.post(self._settings.isaacus_api_url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                raise _RetryableRerankerError("Isaacus request timed out") from exc
            except httpx.TransportError as exc:
                raise _RetryableRerankerError("Isaacus transport error") from exc

        if resp.status_code < 400:
            data = resp.json()
            if not isinstance(data, dict):
                raise RerankerError("Isaacus returned non-object JSON")
            return cast("dict[str, object]", data)

        if resp.status_code in (429, 500, 502, 503, 504):
            raise _RetryableRerankerError(
                f"Isaacus transient error {resp.status_code}",
                retry_after_s=self._parse_retry_after_seconds(resp),
            )

        raise RerankerError(f"Isaacus error {resp.status_code}: {resp.text[:500]}")

    async def _cohere_rerank(self, query: str, documents: list[str]) -> _RerankProviderResult:
        response = await self._cohere.rerank(
            model=self._settings.fallback_model,
            query=query,
            documents=documents,
            top_n=len(documents),
        )

        results_obj = getattr(response, "results", None)
        if not isinstance(results_obj, list):
            raise RerankerError("Cohere response missing 'results'")

        scores: list[float] = [0.0] * len(documents)
        for item in cast("list[object]", results_obj):
            index_obj = getattr(item, "index", None)
            score_obj = getattr(item, "relevance_score", None)
            if not isinstance(index_obj, int) or not (0 <= index_obj < len(documents)):
                raise RerankerError("Cohere result index is invalid")
            if not isinstance(score_obj, (int, float, str)):
                raise RerankerError("Cohere result score is invalid")
            try:
                scores[index_obj] = float(score_obj)
            except (TypeError, ValueError) as exc:
                raise RerankerError("Cohere result score is invalid") from exc

        return _RerankProviderResult(scores=scores, confidence=None)

    def _retry_wait_seconds(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
        if isinstance(exc, _RetryableRerankerError) and exc.retry_after_s is not None:
            return exc.retry_after_s

        attempt_number = max(1, retry_state.attempt_number)
        base = min(
            float(self._settings.retry_max_delay_s),
            float(self._settings.retry_base_delay_s) * (2 ** (attempt_number - 1)),
        )
        return base + uniform(0.0, float(self._settings.retry_jitter_s))

    @staticmethod
    def _extract_response_confidence(response: dict[str, object]) -> float | None:
        """Extract provider confidence from a rerank response when available.

        Args:
            response: Provider response payload.

        Returns:
            Confidence in ``[0.0, 1.0]`` or ``None`` when unavailable.
        """

        for key in ("confidence",):
            value = response.get(key)
            if isinstance(value, (int, float)):
                return max(0.0, min(1.0, float(value)))
        for key in ("meta", "metadata"):
            value = response.get(key)
            if isinstance(value, dict):
                value_dict = cast("dict[str, object]", value)
                nested = value_dict.get("confidence")
                if isinstance(nested, (int, float)):
                    return max(0.0, min(1.0, float(nested)))
        return None

    @staticmethod
    def _resolve_confidence(scores: Sequence[float], provider_confidence: float | None) -> float:
        """Resolve rerank confidence from provider or score-margin heuristics.

        Args:
            scores: Rerank scores in candidate order.
            provider_confidence: Optional provider-emitted confidence.

        Returns:
            Confidence estimate in ``[0.0, 1.0]``.
        """

        if provider_confidence is not None:
            return max(0.0, min(1.0, float(provider_confidence)))
        if not scores:
            return 0.0
        ordered = sorted((float(score) for score in scores), reverse=True)
        top_score = ordered[0]
        second_score = ordered[1] if len(ordered) > 1 else 0.0
        margin = max(0.0, top_score - second_score)
        max_abs = max(1.0, abs(top_score), abs(second_score))
        return max(0.0, min(1.0, margin / max_abs))

    @staticmethod
    def _parse_retry_after_seconds(resp: httpx.Response) -> float | None:
        raw = resp.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return max(0.0, value)

    @staticmethod
    def _build_ranked(
        chunks: Sequence[RetrievedChunk],
        scores: Sequence[float],
        top_n: int,
    ) -> list[RankedChunk]:
        ranked = [
            RankedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                doc_type=chunk.doc_type,
                section_path=chunk.section_path,
                text=chunk.text,
                retrieval_score=chunk.score,
                rerank_score=float(score),
                doc_summary=chunk.doc_summary,
                page_family=getattr(chunk, "page_family", ""),
                doc_family=getattr(chunk, "doc_family", ""),
                chunk_type=getattr(chunk, "chunk_type", ""),
                amount_roles=list(getattr(chunk, "amount_roles", []) or []),
                normalized_refs=list(getattr(chunk, "normalized_refs", []) or []),
            )
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        ranked.sort(key=_ranked_chunk_sort_key)
        return ranked[: max(0, top_n)]

    @staticmethod
    def _build_raw_score_fallback(chunks: Sequence[RetrievedChunk], top_n: int) -> list[RankedChunk]:
        ranked = [
            RankedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                doc_type=chunk.doc_type,
                section_path=chunk.section_path,
                text=chunk.text,
                retrieval_score=chunk.score,
                rerank_score=chunk.score,
                doc_summary=chunk.doc_summary,
                page_family=getattr(chunk, "page_family", ""),
                doc_family=getattr(chunk, "doc_family", ""),
                chunk_type=getattr(chunk, "chunk_type", ""),
                amount_roles=list(getattr(chunk, "amount_roles", []) or []),
                normalized_refs=list(getattr(chunk, "normalized_refs", []) or []),
            )
            for chunk in chunks
        ]
        ranked.sort(key=_ranked_chunk_sort_key)
        return ranked[: max(0, top_n)]
