from __future__ import annotations

from typing import TYPE_CHECKING

import joblib

from rag_challenge.core.grounding.query_scope_classifier import classify_query_scope
from rag_challenge.ml.page_scorer_runtime import RuntimePageScorer, RuntimePageScoringRequest
from rag_challenge.ml.training_scaffold import PAGE_SCORER_FEATURE_POLICY
from rag_challenge.models import RetrievedPage

if TYPE_CHECKING:
    from pathlib import Path


class _DummyVectorizer:
    def transform(self, features: list[dict[str, object]]) -> list[dict[str, object]]:
        return features


class _DummyProbabilityModel:
    def predict_proba(self, features: list[dict[str, object]]) -> list[list[float]]:
        rows: list[list[float]] = []
        for item in features:
            score = float(item.get("page_num", 0)) / 10.0
            rows.append([1.0 - score, score])
        return rows


def _write_bundle(path: Path, *, feature_policy: str = PAGE_SCORER_FEATURE_POLICY) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": _DummyVectorizer(),
            "model": _DummyProbabilityModel(),
            "feature_policy": feature_policy,
            "label_mode": "reviewed_high_confidence",
        },
        path,
    )
    return path


def test_runtime_page_scorer_ranks_candidates_with_runtime_safe_bundle(tmp_path: Path) -> None:
    model_path = _write_bundle(tmp_path / "page_scorer.joblib")
    scorer = RuntimePageScorer(model_path=str(model_path))

    request = RuntimePageScoringRequest(
        query="Who was the judge in CFI 001/2024 and CFI 002/2024?",
        normalized_answer="Justice A",
        answer_type="name",
        scope=classify_query_scope("Who was the judge in CFI 001/2024 and CFI 002/2024?", "names"),
        doc_ids=["case-a", "case-b"],
        page_candidates=[
            RetrievedPage(page_id="case-a_1", doc_id="case-a", page_num=1, score=0.9, page_text="Judge A"),
            RetrievedPage(page_id="case-a_2", doc_id="case-a", page_num=2, score=0.8, page_text="Judge A"),
            RetrievedPage(page_id="case-b_1", doc_id="case-b", page_num=1, score=0.85, page_text="Judge B"),
        ],
        context_page_ids=["case-a_1", "case-b_1"],
        legacy_used_page_ids=["case-a_1"],
        heuristic_scores={"case-a_1": 2.0, "case-a_2": 1.5, "case-b_1": 1.9},
    )

    result = scorer.rank_pages(request)

    assert result.used is True
    assert result.model_path == str(model_path)
    assert result.ranked_page_ids == ["case-a_2", "case-a_1", "case-b_1"]
    assert result.fallback_reason == ""


def test_runtime_page_scorer_fails_closed_for_missing_bundle(tmp_path: Path) -> None:
    scorer = RuntimePageScorer(model_path=str(tmp_path / "missing.joblib"))

    result = scorer.rank_pages(
        RuntimePageScoringRequest(
            query="Who was the judge in CFI 001/2024 and CFI 002/2024?",
            normalized_answer="Justice A",
            answer_type="name",
            scope=classify_query_scope("Who was the judge in CFI 001/2024 and CFI 002/2024?", "names"),
            doc_ids=["case-a", "case-b"],
            page_candidates=[RetrievedPage(page_id="case-a_1", doc_id="case-a", page_num=1, score=0.9)],
            context_page_ids=[],
            legacy_used_page_ids=[],
            heuristic_scores={"case-a_1": 2.0},
        )
    )

    assert result.used is False
    assert result.fallback_reason == "model_bundle_missing_or_invalid"


def test_runtime_page_scorer_rejects_unsupported_feature_policy(tmp_path: Path) -> None:
    model_path = _write_bundle(tmp_path / "page_scorer.joblib", feature_policy="legacy_leaky")
    scorer = RuntimePageScorer(model_path=str(model_path))

    result = scorer.rank_pages(
        RuntimePageScoringRequest(
            query="Who was the judge in CFI 001/2024 and CFI 002/2024?",
            normalized_answer="Justice A",
            answer_type="name",
            scope=classify_query_scope("Who was the judge in CFI 001/2024 and CFI 002/2024?", "names"),
            doc_ids=["case-a", "case-b"],
            page_candidates=[
                RetrievedPage(page_id="case-a_1", doc_id="case-a", page_num=1, score=0.9),
                RetrievedPage(page_id="case-a_2", doc_id="case-a", page_num=2, score=0.8),
            ],
            context_page_ids=[],
            legacy_used_page_ids=[],
            heuristic_scores={"case-a_1": 2.0, "case-a_2": 1.5},
        )
    )

    assert result.used is False
    assert result.fallback_reason == "unsupported_feature_policy"
