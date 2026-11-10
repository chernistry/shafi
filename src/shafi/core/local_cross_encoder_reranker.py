"""Local cross-encoder reranker adapter for optional offline promotion."""

from __future__ import annotations

import importlib
from typing import Any, cast


class LocalCrossEncoderReranker:
    """Thin wrapper around a sentence-transformers style cross encoder."""

    def __init__(self, *, model_path: str, model_obj: Any | None = None) -> None:
        """Initialize the reranker.

        Args:
            model_path: Local model path or model name.
            model_obj: Optional preloaded model for tests.
        """
        self._model_path = model_path
        self._model = model_obj if model_obj is not None else self._load_model()

    @property
    def model_path(self) -> str:
        """Return the configured model path."""
        return self._model_path

    def score_documents(self, *, query: str, documents: list[str]) -> list[float]:
        """Score documents for a query."""
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        scores = self._model.predict(pairs)
        if isinstance(scores, list):
            score_values = cast("list[object]", scores)
            return _coerce_scores(score_values)
        if hasattr(scores, "tolist"):
            score_values = cast("list[object]", scores.tolist())
            return _coerce_scores(score_values)
        raise RuntimeError("Local cross encoder returned unsupported score type")

    def _load_model(self) -> Any:
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError("sentence-transformers is required for local reranking") from exc
        return sentence_transformers.CrossEncoder(self._model_path)


def _coerce_scores(scores: list[object]) -> list[float]:
    """Coerce model outputs into float scores."""
    coerced: list[float] = []
    for score in scores:
        if not isinstance(score, (int, float, str)):
            raise RuntimeError("Local cross encoder returned non-numeric scores")
        coerced.append(float(score))
    return coerced
