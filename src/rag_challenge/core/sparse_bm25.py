from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence


class SparseEncoderError(RuntimeError):
    pass


class BM25SparseEncoder:
    """Client-side sparse BM25 encoder (fastembed) for Qdrant sparse vectors.

    This avoids relying on Qdrant's server-side InferenceService for BM25.
    """

    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = self._resolve_cache_dir(cache_dir)
        self._threads = threads
        self._model = self._create_model()

        # Small LRU cache for query vectors: (normalized query) -> SparseVector
        self._query_cache: OrderedDict[str, SparseVector] = OrderedDict()
        self._query_cache_max = 512

    def _create_model(self) -> Any:
        try:
            from fastembed.sparse import SparseTextEmbedding
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise SparseEncoderError("fastembed is required for client-side BM25 sparse vectors") from exc

        return SparseTextEmbedding(
            self._model_name,
            cache_dir=self._cache_dir,
            threads=self._threads,
            lazy_load=True,
        )

    @staticmethod
    def _resolve_cache_dir(cache_dir: str | None) -> str:
        raw = (cache_dir or "").strip()
        # Keep fastembed cache inside the workspace by default so local runs stay writable
        # and deterministic instead of falling back to container-specific homes like /home/appuser.
        preferred = Path(raw).expanduser() if raw else BM25SparseEncoder._workspace_cache_dir()
        try:
            return BM25SparseEncoder._prepare_cache_dir(preferred)
        except OSError as exc:
            fallback = BM25SparseEncoder._workspace_cache_dir()
            if fallback == preferred:
                raise SparseEncoderError(f"Failed creating BM25 cache dir at {preferred}: {exc}") from exc
            logger.warning(
                "Preferred BM25 cache dir %s is unavailable; falling back to workspace cache %s",
                preferred,
                fallback,
            )
            try:
                return BM25SparseEncoder._prepare_cache_dir(fallback)
            except OSError as fallback_exc:  # pragma: no cover - genuine filesystem failure
                raise SparseEncoderError(f"Failed creating BM25 fallback cache dir at {fallback}: {fallback_exc}") from fallback_exc

    @staticmethod
    def _workspace_cache_dir() -> Path:
        return Path.cwd() / ".cache" / "fastembed"

    @staticmethod
    def _prepare_cache_dir(path: Path) -> str:
        resolved = path.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return str(resolved)

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        if not texts:
            return []

        try:
            embeddings = list(self._model.embed(list(texts)))
        except Exception as exc:
            raise SparseEncoderError(f"BM25 sparse encoding failed: {exc}") from exc

        return [self._to_sparse_vector(obj) for obj in embeddings]

    def encode_query(self, text: str) -> SparseVector:
        normalized = text.strip()
        if not normalized:
            return SparseVector(indices=[], values=[])

        cache_key = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._query_cache.move_to_end(cache_key)
            return cached

        vectors = self.encode_documents([normalized])
        vector = vectors[0] if vectors else SparseVector(indices=[], values=[])

        if len(self._query_cache) >= self._query_cache_max:
            self._query_cache.popitem(last=False)
        self._query_cache[cache_key] = vector
        return vector

    @staticmethod
    def _to_sparse_vector(obj: object) -> SparseVector:
        indices_obj = getattr(obj, "indices", None)
        values_obj = getattr(obj, "values", None)
        if indices_obj is None or values_obj is None:
            raise SparseEncoderError("Sparse embedding missing indices/values")

        try:
            indices = [int(v) for v in cast("Sequence[Any]", indices_obj)]
            values = [float(v) for v in cast("Sequence[Any]", values_obj)]
        except Exception as exc:
            raise SparseEncoderError("Failed converting sparse embedding indices/values") from exc

        if len(indices) != len(values):
            logger.warning("Sparse embedding indices/values length mismatch: %d != %d", len(indices), len(values))
            length = min(len(indices), len(values))
            indices = indices[:length]
            values = values[:length]

        return SparseVector(indices=indices, values=values)
