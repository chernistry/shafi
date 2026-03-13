from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from rag_challenge.core.local_page_reranker import PageRerankScore

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ["LocalLateInteractionReranker"]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return matrix / safe_norms


def _maxsim_mean(query_embedding: np.ndarray, page_embedding: np.ndarray) -> float:
    if query_embedding.ndim != 2 or page_embedding.ndim != 2:
        raise ValueError("Late-interaction embeddings must be 2D arrays")
    if query_embedding.shape[1] != page_embedding.shape[1]:
        raise ValueError("Query/page embeddings must have matching hidden size")
    if query_embedding.shape[0] == 0 or page_embedding.shape[0] == 0:
        return float("-inf")
    query_norm = _l2_normalize(query_embedding.astype(np.float32, copy=False))
    page_norm = _l2_normalize(page_embedding.astype(np.float32, copy=False))
    similarity = query_norm @ page_norm.T
    return float(similarity.max(axis=1).mean())


class LocalLateInteractionReranker:
    def __init__(
        self,
        *,
        model_name: str = "answerdotai/answerai-colbert-small-v1",
        max_chars: int = 4000,
        max_query_chars: int = 1200,
        model_obj: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._max_chars = max(0, int(max_chars))
        self._max_query_chars = max(0, int(max_query_chars))
        self._model = model_obj if model_obj is not None else self._load_model()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self) -> Any:
        try:
            fastembed = importlib.import_module("fastembed")
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via integration only
            raise RuntimeError("fastembed is required for late-interaction local reranking") from exc
        embedding_cls = fastembed.LateInteractionTextEmbedding
        return embedding_cls(model_name=self._model_name)

    def _truncate_text(self, text: str, *, query: bool) -> str:
        limit = self._max_query_chars if query else self._max_chars
        if limit <= 0:
            return text
        return text[:limit]

    def score_pages(self, *, query: str, pages: Sequence[tuple[str, str]]) -> list[PageRerankScore]:
        if not pages:
            return []
        truncated_query = self._truncate_text(query, query=True)
        query_embeddings = list(cast("Iterable[np.ndarray]", self._model.query_embed([truncated_query])))
        if len(query_embeddings) != 1:
            raise RuntimeError(f"Expected exactly one query embedding, got {len(query_embeddings)}")
        query_embedding = query_embeddings[0]
        page_texts = [self._truncate_text(text, query=False) for _, text in pages]
        page_embeddings = list(cast("Iterable[np.ndarray]", self._model.passage_embed(page_texts)))
        if len(page_embeddings) != len(pages):
            raise RuntimeError(
                f"Expected {len(pages)} page embeddings from {self._model_name}, got {len(page_embeddings)}"
            )
        ranked = [
            PageRerankScore(page_id=page_id, score=_maxsim_mean(query_embedding, page_embedding))
            for (page_id, _), page_embedding in zip(pages, page_embeddings, strict=True)
        ]
        return sorted(ranked, key=lambda item: (-item.score, item.page_id))
