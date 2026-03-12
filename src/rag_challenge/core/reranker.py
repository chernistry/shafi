from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from random import uniform
from typing import Any, cast

import httpx
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt

from rag_challenge.config import get_settings
from rag_challenge.core.circuit_breaker import CircuitBreaker, CircuitState
from rag_challenge.models import RankedChunk, RetrievedChunk

logger = logging.getLogger(__name__)

__all__ = ["CircuitBreaker", "CircuitState", "RerankerClient", "RerankerError"]


class RerankerError(RuntimeError):
    pass


class _RetryableRerankerError(RerankerError):
    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


SleepFunc = Callable[[float], Awaitable[None]]


class RerankerClient:
    """Zerank 2 primary reranker with Cohere fallback."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        cohere_client: Any | None = None,
        sleep_func: SleepFunc = asyncio.sleep,
    ) -> None:
        settings = get_settings()
        self._settings = settings.reranker
        self._pipeline_settings = getattr(settings, "pipeline", None)
        self._external_client = client is not None
        self._client = client or self._make_client()
        self._cohere = cohere_client if cohere_client is not None else self._make_cohere_client()
        self._sleep = sleep_func
        self._last_used_model = self._settings.primary_model
        self._circuit = CircuitBreaker(
            name="zerank-reranker",
            failure_threshold=int(self._settings.circuit_failure_threshold),
            reset_timeout_s=float(self._settings.circuit_reset_timeout_s),
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
    ) -> list[RankedChunk]:
        if top_n is None:
            top_n = self._settings.top_n

        if not chunks:
            return []

        max_chars = int(getattr(self._pipeline_settings, "rerank_doc_max_chars", 1500))
        documents = [chunk.text[:max_chars] for chunk in chunks] if max_chars > 0 else [chunk.text for chunk in chunks]

        if prefer_fast:
            try:
                scores = await self._cohere_rerank(query, documents)
                self._last_used_model = self._settings.fallback_model
                return self._build_ranked(chunks, scores, top_n)
            except Exception as exc:
                logger.warning("Fast reranker path failed; falling back to primary: %s", exc)

        if self._circuit.allow_request():
            try:
                scores = await self._zerank_rerank(query, documents)
                self._circuit.record_success()
                self._last_used_model = self._settings.primary_model
                return self._build_ranked(chunks, scores, top_n)
            except Exception as exc:
                logger.warning("Zerank reranker failed, using Cohere fallback: %s", exc)
                self._circuit.record_failure()
        else:
            logger.info("Zerank reranker circuit open; skipping to Cohere fallback")

        try:
            scores = await self._cohere_rerank(query, documents)
            self._last_used_model = self._settings.fallback_model
            return self._build_ranked(chunks, scores, top_n)
        except Exception as exc:
            logger.error("Both rerankers failed; degrading to raw retrieval scores: %s", exc)
            self._last_used_model = "raw_retrieval_fallback"
            return self._build_raw_score_fallback(chunks, top_n)

    def get_last_used_model(self) -> str:
        return self._last_used_model

    async def _zerank_rerank(self, query: str, documents: list[str]) -> list[float]:
        scores: list[float] = [0.0] * len(documents)
        batch_size = max(1, int(self._settings.primary_batch_size))

        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            response = await self._zerank_post(query, batch)
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

        return scores

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

    async def _cohere_rerank(self, query: str, documents: list[str]) -> list[float]:
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

        return scores

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
                page_number=chunk.page_number,
                page_type=chunk.page_type,
                heading_text=chunk.heading_text,
                doc_refs=list(chunk.doc_refs),
                law_no=chunk.law_no,
                law_year=chunk.law_year,
                article_refs=list(chunk.article_refs),
                has_caption_terms=chunk.has_caption_terms,
                has_order_terms=chunk.has_order_terms,
                doc_summary=chunk.doc_summary,
            )
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        ranked.sort(key=lambda c: c.rerank_score, reverse=True)
        return ranked[: max(0, top_n)]

    @staticmethod
    def _build_raw_score_fallback(chunks: Sequence[RetrievedChunk], top_n: int) -> list[RankedChunk]:
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        return [
            RankedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                doc_type=chunk.doc_type,
                section_path=chunk.section_path,
                text=chunk.text,
                retrieval_score=chunk.score,
                rerank_score=chunk.score,
                page_number=chunk.page_number,
                page_type=chunk.page_type,
                heading_text=chunk.heading_text,
                doc_refs=list(chunk.doc_refs),
                law_no=chunk.law_no,
                law_year=chunk.law_year,
                article_refs=list(chunk.article_refs),
                has_caption_terms=chunk.has_caption_terms,
                has_order_terms=chunk.has_order_terms,
                doc_summary=chunk.doc_summary,
            )
            for chunk in sorted_chunks[: max(0, top_n)]
        ]
