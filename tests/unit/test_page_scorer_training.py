from __future__ import annotations

from scripts.train_page_scorer import _build_label_quality_note
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from rag_challenge.ml.page_scorer_training import (
    build_score_lookup,
    compute_heuristic_ranking_metrics,
    compute_ranking_metrics,
    count_question_sources,
    rank_group_with_scores,
    top_feature_weights,
)
from rag_challenge.ml.training_scaffold import PageTrainingExample


def _example(
    *,
    question_id: str,
    page_id: str,
    label: int,
    supervision_source: str = "sidecar_selected",
    **features: object,
) -> PageTrainingExample:
    return PageTrainingExample(
        question_id=question_id,
        page_id=page_id,
        features={"page_num": 1, **features},
        label=label,
        sample_weight=1.0,
        supervision_source=supervision_source,
    )


def test_compute_ranking_metrics_reports_hit_at_k_and_mrr() -> None:
    examples = [
        _example(question_id="q1", page_id="a", label=1),
        _example(question_id="q1", page_id="b", label=0),
        _example(question_id="q2", page_id="c", label=0),
        _example(question_id="q2", page_id="d", label=1),
    ]

    metrics = compute_ranking_metrics(examples, [0.9, 0.1, 0.8, 0.6])

    assert metrics.question_count == 2
    assert metrics.hit_at_1 == 0.5
    assert metrics.hit_at_2 == 1.0
    assert metrics.mean_reciprocal_rank == 0.75


def test_compute_heuristic_ranking_metrics_prefers_used_pages() -> None:
    examples = [
        _example(question_id="q1", page_id="neg", label=0, from_legacy_context=True, legacy_context_rank=1),
        _example(
            question_id="q1",
            page_id="pos",
            label=1,
            from_sidecar_used=True,
            sidecar_context_rank=2,
            anchor_hit_count=1,
        ),
    ]

    metrics = compute_heuristic_ranking_metrics(examples)

    assert metrics.hit_at_1 == 1.0
    assert metrics.hit_at_2 == 1.0
    assert metrics.mean_reciprocal_rank == 1.0


def test_rank_group_with_scores_orders_best_page_first() -> None:
    group = [
        _example(question_id="q1", page_id="neg", label=0),
        _example(question_id="q1", page_id="pos", label=1),
    ]

    score_lookup = build_score_lookup(group, [0.1, 0.9])
    ranked = rank_group_with_scores(group, score_lookup)

    assert ranked[0].page_id == "pos"


def test_count_question_sources_is_grouped_by_question() -> None:
    examples = [
        _example(question_id="q1", page_id="a", label=1, supervision_source="reviewed"),
        _example(question_id="q1", page_id="b", label=0, supervision_source="reviewed"),
        _example(question_id="q2", page_id="c", label=1, supervision_source="legacy_selected"),
    ]

    counts = count_question_sources(examples)

    assert counts == {"legacy_selected": 1, "reviewed": 1}


def test_top_feature_weights_returns_positive_and_negative_features() -> None:
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(
        [
            {"feature_a": 1.0},
            {"feature_b": 1.0},
            {"feature_a": 1.0},
            {"feature_b": 1.0},
        ]
    )
    model = LogisticRegression(max_iter=200, random_state=617)
    model.fit(x_train, [1, 0, 1, 0])

    summary = top_feature_weights(model, vectorizer, top_n=1)

    assert summary["positive"][0]["feature"] == "feature_a"
    assert summary["negative"][0]["feature"] == "feature_b"


def test_build_label_quality_note_describes_reviewed_weighted_mode() -> None:
    note = _build_label_quality_note("reviewed_weighted")

    assert "confidence-aware weighting" in note
