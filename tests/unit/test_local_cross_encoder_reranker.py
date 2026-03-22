from __future__ import annotations

from rag_challenge.core.local_cross_encoder_reranker import LocalCrossEncoderReranker


class _FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        assert pairs[0][0] == "employment query"
        return [0.9 if "Employment Law" in document else 0.2 for _query, document in pairs]


def test_local_cross_encoder_reranker_scores_documents() -> None:
    reranker = LocalCrossEncoderReranker(model_path="local-model", model_obj=_FakeCrossEncoder())

    scores = reranker.score_documents(
        query="employment query",
        documents=[
            "Employment Law 2020 issued by DIFC Authority.",
            "Alice Smith v Registrar.",
        ],
    )

    assert scores == [0.9, 0.2]
