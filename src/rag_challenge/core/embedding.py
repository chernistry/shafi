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
        provider = str(getattr(self._settings, "provider", "isaacus")).strip().lower()
        headers: dict[str, str] = {}
        base_url = self._settings.api_url
        max_connections = self._settings.concurrency
        timeout_s = self._settings.timeout_s
        if provider == "isaacus":
            headers = {"Authorization": f"Bearer {self._settings.api_key.get_secret_value()}"}
        elif provider == "ollama":
            base_url = self._settings.ollama_base_url
            max_connections = max(1, int(getattr(self._settings, "ollama_concurrency", self._settings.concurrency)))
            timeout_s = float(getattr(self._settings, "ollama_timeout_s", self._settings.timeout_s))
        return httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            timeout=httpx.Timeout(timeout_s, connect=self._settings.connect_timeout_s),
        )

    def _effective_batch_size(self) -> int:
        provider = str(getattr(self._settings, "provider", "isaacus")).strip().lower()
        if provider == "ollama":
            return max(1, int(getattr(self._settings, "ollama_batch_size", self._settings.batch_size)))
        return max(1, int(self._settings.batch_size))

    def _effective_concurrency(self) -> int:
        provider = str(getattr(self._settings, "provider", "isaacus")).strip().lower()
        if provider == "ollama":
            return max(1, int(getattr(self._settings, "ollama_concurrency", self._settings.concurrency)))
        return max(1, int(self._settings.concurrency))

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

        sem = asyncio.Semaphore(self._effective_concurrency())
        batches = list(self._chunk_list(list(texts), self._effective_batch_size()))

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
        provider = str(getattr(self._settings, "provider", "isaacus")).strip().lower()
        if provider == "ollama":
            return await self._embed_batch_once_ollama(texts)

        return await self._embed_batch_once_isaacus(texts=texts, task=task)

    async def _embed_batch_once_isaacus(self, *, texts: list[str], task: str) -> list[list[float]]:
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

    async def _embed_batch_once_ollama(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self._settings.model,
            "input": texts if len(texts) > 1 else texts[0],
        }
        resp = await self._client.post("/api/embed", json=payload)
        if resp.status_code == 404:
            return await self._embed_batch_legacy_ollama(texts)
        if resp.status_code < 400:
            return self._parse_ollama_embeddings(resp, expected_count=len(texts))
        if resp.status_code in (429, 500, 502, 503, 504):
            retry_after_s = self._parse_retry_after_seconds(resp)
            raise _RetryableEmbeddingError(
                f"Ollama transient error {resp.status_code} (will retry)",
                retry_after_s=retry_after_s,
            )
        raise EmbeddingError(f"Ollama API error {resp.status_code}: {resp.text[:500]}")

    async def _embed_batch_legacy_ollama(self, texts: list[str]) -> list[list[float]]:
        async def _embed_single(text: str) -> list[float]:
            resp = await self._client.post(
                "/api/embeddings",
                json={"model": self._settings.model, "prompt": text},
            )
            if resp.status_code < 400:
                parsed = self._parse_ollama_embeddings(resp, expected_count=1)
                return parsed[0]
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after_s = self._parse_retry_after_seconds(resp)
                raise _RetryableEmbeddingError(
                    f"Ollama legacy transient error {resp.status_code} (will retry)",
                    retry_after_s=retry_after_s,
                )
            raise EmbeddingError(f"Ollama legacy API error {resp.status_code}: {resp.text[:500]}")

        return await asyncio.gather(*[_embed_single(text) for text in texts])

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
    def _parse_ollama_embeddings(resp: httpx.Response, *, expected_count: int) -> list[list[float]]:
        data: object = resp.json()
        if not isinstance(data, dict):
            raise EmbeddingError("Ollama API returned non-object JSON")

        data_dict = cast("dict[str, object]", data)
        embeddings_obj = data_dict.get("embeddings")
        if isinstance(embeddings_obj, list):
            rows = cast("list[object]", embeddings_obj)
            parsed: list[list[float]] = []
            for row in rows:
                if not isinstance(row, list):
                    raise EmbeddingError("Ollama embeddings response contains non-list row")
                try:
                    parsed.append([float(value) for value in cast("list[int | float | str]", row)])
                except (TypeError, ValueError) as exc:
                    raise EmbeddingError("Ollama embeddings response contains non-numeric values") from exc
            if len(parsed) != expected_count:
                raise EmbeddingError(
                    f"Ollama API returned {len(parsed)} embeddings for {expected_count} texts"
                )
            return parsed

        embedding_obj = data_dict.get("embedding")
        if isinstance(embedding_obj, list):
            try:
                vector = [float(value) for value in cast("list[int | float | str]", embedding_obj)]
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("Ollama embedding response contains non-numeric values") from exc
            if expected_count != 1:
                raise EmbeddingError(
                    f"Ollama API returned a single embedding for {expected_count} texts"
                )
            return [vector]

        data_rows = data_dict.get("data")
        if isinstance(data_rows, list):
            parsed_rows: list[list[float]] = []
            for row in cast("list[object]", data_rows):
                if not isinstance(row, dict):
                    raise EmbeddingError("Ollama API response contains non-object data row")
                embedding_candidate = cast("dict[str, object]", row).get("embedding")
                if not isinstance(embedding_candidate, list):
                    raise EmbeddingError("Ollama API response data row missing embedding list")
                try:
                    parsed_rows.append([float(value) for value in cast("list[int | float | str]", embedding_candidate)])
                except (TypeError, ValueError) as exc:
                    raise EmbeddingError("Ollama API response data row contains non-numeric values") from exc
            if len(parsed_rows) != expected_count:
                raise EmbeddingError(
                    f"Ollama API returned {len(parsed_rows)} embeddings for {expected_count} texts"
                )
            return parsed_rows

        raise EmbeddingError("Ollama API response missing embeddings")

    @staticmethod
    def _chunk_list(items: list[str], size: int) -> list[list[str]]:
        return [items[i : i + size] for i in range(0, len(items), size)]
