from __future__ import annotations

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
