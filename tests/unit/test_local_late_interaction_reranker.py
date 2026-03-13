from __future__ import annotations

import numpy as np

from rag_challenge.core.local_late_interaction_reranker import LocalLateInteractionReranker


class _FakeLateInteractionModel:
    def query_embed(self, queries: list[str]) -> list[np.ndarray]:
        assert queries == ["query words"]
        return [
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
        ]

    def passage_embed(self, texts: list[str]) -> list[np.ndarray]:
        assert texts == ["alpha text", "beta text", "gamma text"]
        return [
            np.array([[1.0, 0.0], [0.3, 0.0]], dtype=np.float32),
            np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ]


def test_local_late_interaction_reranker_sorts_scores_descending() -> None:
    reranker = LocalLateInteractionReranker(model_obj=_FakeLateInteractionModel(), max_chars=20, max_query_chars=20)
    ranked = reranker.score_pages(
        query="query words",
        pages=[
            ("doc_1", "alpha text"),
            ("doc_2", "beta text"),
            ("doc_3", "gamma text"),
        ],
    )
    assert [row.page_id for row in ranked] == ["doc_3", "doc_1", "doc_2"]
    assert ranked[0].score > ranked[1].score
    assert ranked[1].score >= ranked[2].score


class _TruncatingModel:
    def __init__(self) -> None:
        self.seen_query: list[str] = []
        self.seen_passages: list[str] = []

    def query_embed(self, queries: list[str]) -> list[np.ndarray]:
        self.seen_query = queries
        return [np.ones((1, 2), dtype=np.float32)]

    def passage_embed(self, texts: list[str]) -> list[np.ndarray]:
        self.seen_passages = texts
        return [np.ones((1, 2), dtype=np.float32) for _ in texts]


def test_local_late_interaction_reranker_truncates_query_and_pages() -> None:
    model = _TruncatingModel()
    reranker = LocalLateInteractionReranker(model_obj=model, max_chars=5, max_query_chars=4)
    reranker.score_pages(query="abcdefgh", pages=[("doc_1", "123456789")])
    assert model.seen_query == ["abcd"]
    assert model.seen_passages == ["12345"]
