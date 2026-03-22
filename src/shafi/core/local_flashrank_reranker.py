"""Local FlashRank reranker adapter — fast offline cross-encoder via ONNX.

FlashRank uses ONNX-quantized MS-MARCO cross-encoders and runs entirely locally
with no API latency.  Typical throughput: 20-30ms for 12 candidates vs 700-800ms
for zerank-2 API.

Supported models (downloaded on first use to FLASHRANK_CACHE_DIR):
  ms-marco-TinyBERT-L-2-v2   (fastest, ~5ms,  lowest quality)
  ms-marco-MiniLM-L-12-v2    (balanced, ~22ms, good quality)
  ms-marco-MultiBERT-L-12     (slower,  ~40ms, multilingual)
  rank-T5-flan                (slow,    ~80ms, T5-based)

Config (via env / profile):
  RERANK_PROVIDER_MODE=flashrank
  RERANK_LOCAL_MODEL_PATH=ms-marco-MiniLM-L-12-v2  (or full path)
  RERANK_FLASHRANK_CACHE_DIR=/tmp/flashrank_models  (optional)
"""

from __future__ import annotations

import importlib
from typing import Any


class LocalFlashRankReranker:
    """Thin adapter around FlashRank for cross-encoder reranking.

    Args:
        model_name: FlashRank model name or local path.
        cache_dir: Directory for downloaded model files.
        model_obj: Optional pre-loaded Ranker for testing.
    """

    def __init__(
        self,
        *,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        cache_dir: str = "/tmp/flashrank_models",
        model_obj: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._ranker: Any = model_obj if model_obj is not None else self._load_ranker()

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    def score_documents(self, *, query: str, documents: list[str]) -> list[float]:
        """Score documents for a query using FlashRank.

        Returns scores in the same order as ``documents`` (not sorted).

        Args:
            query: Query string.
            documents: Document strings to score.

        Returns:
            Float relevance scores in the same order as ``documents``.
        """
        if not documents:
            return []
        flashrank = importlib.import_module("flashrank")
        passages = [{"id": i, "text": text} for i, text in enumerate(documents)]
        req = flashrank.RerankRequest(query=query, passages=passages)
        results: list[dict[str, object]] = self._ranker.rerank(req)
        # FlashRank returns results sorted by score descending; restore original order.
        id_to_score: dict[int, float] = {}
        for r in results:
            doc_id = int(r["id"])  # type: ignore[arg-type]
            score = float(r.get("score", 0.0))  # type: ignore[arg-type]
            id_to_score[doc_id] = score
        return [id_to_score.get(i, 0.0) for i in range(len(documents))]

    def _load_ranker(self) -> Any:
        try:
            flashrank = importlib.import_module("flashrank")
        except ModuleNotFoundError as exc:
            raise RuntimeError("flashrank is required; install with: pip install flashrank") from exc
        return flashrank.Ranker(model_name=self._model_name, cache_dir=self._cache_dir)
