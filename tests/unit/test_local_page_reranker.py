from __future__ import annotations

import types

import rag_challenge.core.local_page_reranker as local_page_reranker
from rag_challenge.core.local_page_reranker import LocalPageReranker


class _FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int, show_progress_bar: bool) -> list[float]:
        assert batch_size == 2
        assert show_progress_bar is False
        assert pairs[0][1] == "abcde"
        return list(self._scores)


def test_local_page_reranker_sorts_scores_descending() -> None:
    reranker = LocalPageReranker(batch_size=2, max_chars=5, model_obj=_FakeModel([0.2, 0.9, 0.4]))
    ranked = reranker.score_pages(
        query="query",
        pages=[
            ("doc_1", "abcdefgh"),
            ("doc_2", "abcdefgh"),
            ("doc_3", "abcdefgh"),
        ],
    )
    assert [row.page_id for row in ranked] == ["doc_2", "doc_3", "doc_1"]
    assert [row.score for row in ranked] == [0.9, 0.4, 0.2]


def test_transformers_cross_encoder_like_assigns_pad_token_from_eos(monkeypatch) -> None:
    class _FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token = None
            self.pad_token_id = None
            self.eos_token = "</s>"
            self.eos_token_id = 7
            self.model_max_length = 512

    class _FakeModel:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(pad_token_id=None)

        def eval(self) -> None:
            return None

        def to(self, device) -> _FakeModel:
            return self

    fake_tokenizer = _FakeTokenizer()
    fake_model = _FakeModel()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str) -> _FakeTokenizer:
            return fake_tokenizer

    class _FakeAutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(model_name: str) -> _FakeModel:
            return fake_model

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeAutoTokenizer,
        AutoModelForSequenceClassification=_FakeAutoModelForSequenceClassification,
    )

    monkeypatch.setattr(local_page_reranker.importlib, "import_module", lambda name: fake_transformers)

    scorer = local_page_reranker._TransformersCrossEncoderLike(
        model_name="Qwen/Qwen3-Reranker-0.6B",
        device="cpu",
        batch_size=2,
    )

    assert scorer._tokenizer.pad_token == "</s>"
    assert scorer._tokenizer.pad_token_id == 7
    assert scorer._model.config.pad_token_id == 7
