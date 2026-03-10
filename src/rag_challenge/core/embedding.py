from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from random import uniform
from typing import cast

import httpx
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt

from rag_challenge.config import get_settings
from rag_challenge.core.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class _RetryableEmbeddingError(EmbeddingError):
    """Raised on 429/5xx so retry logic can distinguish transient vs fatal errors."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


SleepFunc = Callable[[float], Awaitable[None]]


class EmbeddingClient:
    """Async client for Isaacus Kanon 2 Embedder API."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        sleep_func: SleepFunc = asyncio.sleep,
    ) -> None:
        self._settings = get_settings().embedding
        self._external_client = client is not None
        self._client = client or self._make_client()
        self._sleep = sleep_func
        self._circuit = CircuitBreaker(
            name="embedding",
            failure_threshold=int(self._settings.circuit_failure_threshold),
            reset_timeout_s=float(self._settings.circuit_reset_timeout_s),
        )
        # LRU cache for query embeddings (query text -> vector)
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache_max = 1000

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._settings.api_key.get_secret_value()}"},
            limits=httpx.Limits(
                max_connections=self._settings.concurrency,
                max_keepalive_connections=self._settings.concurrency,
            ),
            timeout=httpx.Timeout(self._settings.timeout_s, connect=self._settings.connect_timeout_s),
        )

    async def close(self) -> None:
        if not self._external_client:
            await self._client.aclose()

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text using task=retrieval/query (with LRU cache)."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._query_cache:
            self._query_cache.move_to_end(cache_key)
            return self._query_cache[cache_key]
        if not self._circuit.allow_request():
            raise EmbeddingError("Embedding circuit is open")
        try:
            results = await self._embed_batch([text], task="retrieval/query")
            self._circuit.record_success()
        except _RetryableEmbeddingError as exc:
            self._circuit.record_failure()
            raise EmbeddingError(f"Kanon 2 embedding failed after retries: {exc}") from exc
        except Exception:
            self._circuit.record_failure()
            raise
        vec = results[0]
        if len(self._query_cache) >= self._query_cache_max:
            self._query_cache.popitem(last=False)
        self._query_cache[cache_key] = vec
        return vec

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunks using task=retrieval/document with batching and concurrency."""
        if not texts:
            return []

        sem = asyncio.Semaphore(self._settings.concurrency)
        batches = list(self._chunk_list(list(texts), self._settings.batch_size))

        async def _process_batch(batch: list[str]) -> list[list[float]]:
            async with sem:
                return await self._embed_batch(batch, task="retrieval/document")

        if not self._circuit.allow_request():
            raise EmbeddingError("Embedding circuit is open")
        try:
            results = await asyncio.gather(*[_process_batch(batch) for batch in batches])
            self._circuit.record_success()
        except _RetryableEmbeddingError as exc:
            self._circuit.record_failure()
            raise EmbeddingError(f"Kanon 2 embedding failed after retries: {exc}") from exc
        except Exception:
            self._circuit.record_failure()
            raise

        return [vec for batch_result in results for vec in batch_result]

    async def _embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
        """Send a single embedding request (up to batch_size texts) with retries."""
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_RetryableEmbeddingError),
            stop=stop_after_attempt(max(1, int(self._settings.retry_attempts))),
            wait=self._retry_wait_seconds,
            reraise=True,
            sleep=self._sleep,
        ):
            with attempt:
                return await self._embed_batch_once(texts=texts, task=task)

        raise AssertionError("unreachable")

    async def _embed_batch_once(self, texts: list[str], task: str) -> list[list[float]]:
        payload = {
            "model": self._settings.model,
            "texts": texts,
            "dimensions": self._settings.dimensions,
            "task": task,
        }

        resp = await self._client.post(self._settings.api_url, json=payload)

        if resp.status_code < 400:
            return self._parse_embeddings(resp, expected_count=len(texts))

        if resp.status_code in (429, 500, 502, 503, 504):
            retry_after_s = self._parse_retry_after_seconds(resp)
            raise _RetryableEmbeddingError(
                f"Kanon 2 API transient error {resp.status_code} (will retry)",
                retry_after_s=retry_after_s,
            )

        raise EmbeddingError(f"Kanon 2 API error {resp.status_code}: {resp.text[:500]}")

    def _retry_wait_seconds(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
        if isinstance(exc, _RetryableEmbeddingError) and exc.retry_after_s is not None:
            return exc.retry_after_s

        attempt_number = max(1, retry_state.attempt_number)
        base = min(
            float(self._settings.retry_max_delay_s),
            float(self._settings.retry_base_delay_s) * (2 ** (attempt_number - 1)),
        )
        return base + uniform(0.0, float(self._settings.retry_jitter_s))

    @staticmethod
    def _parse_retry_after_seconds(resp: httpx.Response) -> float | None:
        header = resp.headers.get("Retry-After")
        if not header:
            return None

        try:
            seconds = float(header)
        except ValueError:
            return None

        return max(0.0, seconds)

    @staticmethod
    def _parse_embeddings(resp: httpx.Response, *, expected_count: int) -> list[list[float]]:
        data: object = resp.json()
        if not isinstance(data, dict):
            raise EmbeddingError("Kanon 2 API returned non-object JSON")

        data_dict = cast("dict[str, object]", data)
        embeddings_obj = data_dict.get("embeddings")
        if not isinstance(embeddings_obj, list):
            raise EmbeddingError("Kanon 2 API response missing 'embeddings' list")

        rows = cast("list[object]", embeddings_obj)
        embeddings: list[list[float]] = []
        for row_obj in rows:
            row: object = row_obj
            if isinstance(row, dict):
                row_dict = cast("dict[str, object]", row)
                candidate = row_dict.get("embedding", row_dict.get("vector"))
                if candidate is not None:
                    row = candidate
            if not isinstance(row, list):
                raise EmbeddingError("Kanon 2 API response contains non-list embedding row")
            try:
                vector = [float(value) for value in cast("list[int | float | str]", row)]
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("Kanon 2 API response contains non-numeric embedding values") from exc
            embeddings.append(vector)

        if len(embeddings) != expected_count:
            raise EmbeddingError(
                f"Kanon 2 API returned {len(embeddings)} embeddings for {expected_count} texts"
            )

        return embeddings

    @staticmethod
    def _chunk_list(items: list[str], size: int) -> list[list[str]]:
        return [items[i : i + size] for i in range(0, len(items), size)]
