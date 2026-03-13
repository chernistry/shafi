from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["LocalPageReranker", "PageRerankScore"]


@dataclass(frozen=True)
class PageRerankScore:
    page_id: str
    score: float


def _auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalPageReranker:
    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        batch_size: int = 8,
        max_chars: int = 4000,
        model_obj: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = _auto_device() if device == "auto" else device
        self._batch_size = max(1, int(batch_size))
        self._max_chars = max(0, int(max_chars))
        self._model = model_obj if model_obj is not None else self._load_model()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def _load_model(self) -> Any:
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError:
            return _TransformersCrossEncoderLike(
                model_name=self._model_name,
                device=self._device,
                batch_size=self._batch_size,
            )

        cross_encoder_cls = sentence_transformers.CrossEncoder
        try:
            return cross_encoder_cls(self._model_name, device=self._device)
        except Exception:
            if self._device == "cpu":
                return _TransformersCrossEncoderLike(
                    model_name=self._model_name,
                    device="cpu",
                    batch_size=self._batch_size,
                )
            self._device = "cpu"
            try:
                return cross_encoder_cls(self._model_name, device="cpu")
            except Exception:
                return _TransformersCrossEncoderLike(
                    model_name=self._model_name,
                    device="cpu",
                    batch_size=self._batch_size,
                )

    def score_pages(self, *, query: str, pages: Sequence[tuple[str, str]]) -> list[PageRerankScore]:
        if not pages:
            return []
        pairs = [(query, text[: self._max_chars] if self._max_chars > 0 else text) for _, text in pages]
        scores_obj = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        scores = [float(score) for score in scores_obj]
        ranked = [PageRerankScore(page_id=page_id, score=score) for (page_id, _), score in zip(pages, scores, strict=True)]
        return sorted(ranked, key=lambda item: (-item.score, item.page_id))


class _TransformersCrossEncoderLike:
    def __init__(self, *, model_name: str, device: str, batch_size: int) -> None:
        transformers = importlib.import_module("transformers")
        tokenizer_cls = transformers.AutoTokenizer
        model_cls = transformers.AutoModelForSequenceClassification
        self._tokenizer = tokenizer_cls.from_pretrained(model_name)
        self._model = model_cls.from_pretrained(model_name)
        if getattr(self._tokenizer, "pad_token", None) is None and getattr(self._tokenizer, "eos_token", None) is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if getattr(self._tokenizer, "pad_token_id", None) is None and getattr(self._tokenizer, "eos_token_id", None) is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        if getattr(self._model, "config", None) is not None and getattr(self._model.config, "pad_token_id", None) is None:
            self._model.config.pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        self._model.eval()
        self._batch_size = max(1, int(batch_size))
        self._torch_device = _torch_device(device)
        self._model.to(self._torch_device)
        max_length = getattr(self._tokenizer, "model_max_length", 512)
        self._max_length = 512 if not isinstance(max_length, int) or max_length <= 0 or max_length > 4096 else max_length

    def predict(self, pairs: Sequence[tuple[str, str]], *, batch_size: int, show_progress_bar: bool) -> list[float]:
        del show_progress_bar
        effective_batch_size = max(1, int(batch_size))
        scores: list[float] = []
        for start in range(0, len(pairs), effective_batch_size):
            batch = pairs[start : start + effective_batch_size]
            queries = [query for query, _ in batch]
            texts = [text for _, text in batch]
            encoded = self._tokenizer(
                queries,
                texts,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._torch_device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            logits = outputs.logits
            batch_scores = logits[:, -1] if logits.ndim == 2 and logits.shape[1] > 1 else logits.reshape(-1)
            scores.extend(float(score) for score in batch_scores.detach().cpu().tolist())
        return scores


def _torch_device(device: str) -> torch.device:
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
