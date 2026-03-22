"""Helpers for offline page-scorer training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol, cast

from shafi.ml.training_scaffold import PageTrainingExample, group_page_examples

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

type PageRankKey = tuple[int, int, int, int, int, int, int, int, str]
type ModelRankKey = tuple[float, str]


class _FeatureNameVectorizer(Protocol):
    """Minimal vectorizer protocol for feature-name extraction."""

    def get_feature_names_out(self) -> Sequence[str]:
        """Return fitted feature names."""
        ...


class _LinearCoefModel(Protocol):
    """Minimal linear-model protocol for coefficient extraction."""

    coef_: Sequence[Sequence[float]]


@dataclass(frozen=True)
class RankingMetrics:
    """Grouped ranking metrics for page-scoring evaluation.

    Args:
        question_count: Number of evaluated grouped questions.
        hit_at_1: Grouped hit@1.
        hit_at_2: Grouped hit@2.
        mean_reciprocal_rank: Mean reciprocal rank over grouped questions.
    """

    question_count: int
    hit_at_1: float
    hit_at_2: float
    mean_reciprocal_rank: float

    def to_dict(self) -> dict[str, float | int]:
        """Return the metric payload as a JSON-friendly mapping.

        Returns:
            Serialized metric dictionary.
        """
        return asdict(self)


def compute_ranking_metrics(
    examples: Sequence[PageTrainingExample],
    scores: Sequence[float],
) -> RankingMetrics:
    """Compute grouped page-ranking metrics from model scores.

    Args:
        examples: Flat page examples.
        scores: Predicted relevance scores aligned with `examples`.

    Returns:
        Grouped ranking metrics.
    """
    if not examples:
        return RankingMetrics(question_count=0, hit_at_1=0.0, hit_at_2=0.0, mean_reciprocal_rank=0.0)

    score_lookup = build_score_lookup(examples, scores)
    return _metrics_from_ranked_groups(
        grouped_examples=group_page_examples(examples),
        ranker=_build_model_ranker(score_lookup),
    )


def compute_heuristic_ranking_metrics(examples: Sequence[PageTrainingExample]) -> RankingMetrics:
    """Compute grouped metrics for the heuristic page-order baseline.

    Args:
        examples: Flat page examples.

    Returns:
        Grouped heuristic ranking metrics.
    """
    if not examples:
        return RankingMetrics(question_count=0, hit_at_1=0.0, hit_at_2=0.0, mean_reciprocal_rank=0.0)

    return _metrics_from_ranked_groups(
        grouped_examples=group_page_examples(examples),
        ranker=_heuristic_rank_key,
    )


def count_question_sources(examples: Sequence[PageTrainingExample]) -> dict[str, int]:
    """Count grouped supervision sources by question.

    Args:
        examples: Flat page examples.

    Returns:
        Question-level counts by supervision source.
    """
    counts: dict[str, int] = {}
    for group in group_page_examples(examples).values():
        source = group[0].supervision_source if group else "unknown"
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def build_score_lookup(
    examples: Sequence[PageTrainingExample],
    scores: Sequence[float],
) -> dict[str, float]:
    """Build a question/page score lookup for grouped ranking.

    Args:
        examples: Flat page examples.
        scores: Predicted scores aligned with `examples`.

    Returns:
        Mapping keyed by `question_id:page_id`.
    """
    return {
        f"{example.question_id}:{example.page_id}": float(score)
        for example, score in zip(examples, scores, strict=False)
    }


def rank_group_with_scores(
    group: Sequence[PageTrainingExample],
    score_lookup: dict[str, float],
) -> list[PageTrainingExample]:
    """Rank one grouped question using trained model scores.

    Args:
        group: One grouped question worth of page examples.
        score_lookup: Trained page-score lookup.

    Returns:
        Ranked page examples, best first.
    """
    return sorted(group, key=_build_model_ranker(score_lookup))


def rank_group_with_heuristic(group: Sequence[PageTrainingExample]) -> list[PageTrainingExample]:
    """Rank one grouped question using the heuristic baseline.

    Args:
        group: One grouped question worth of page examples.

    Returns:
        Ranked page examples, best first.
    """
    return sorted(group, key=_heuristic_rank_key)


def top_feature_weights(
    model: object,
    vectorizer: object,
    *,
    top_n: int,
) -> dict[str, list[dict[str, float | str]]]:
    """Extract top positive and negative feature weights.

    Supports linear models (coef_) and tree-based models (feature_importances_).

    Args:
        model: Fitted model (LogisticRegression, LightGBM, XGBoost, etc.).
        vectorizer: Fitted dictionary vectorizer.
        top_n: Number of features to keep per polarity (or top importance).

    Returns:
        Positive and negative weighted feature summaries. For tree models,
        "positive" contains top-importance features and "negative" is empty.
    """
    if not hasattr(vectorizer, "get_feature_names_out"):
        raise TypeError(f"Unsupported page scorer vectorizer: {type(vectorizer)!r}")

    typed_vectorizer = cast("_FeatureNameVectorizer", vectorizer)
    feature_names = [str(value) for value in typed_vectorizer.get_feature_names_out()]

    # Tree-based models: use feature_importances_ (all non-negative)
    if hasattr(model, "feature_importances_"):
        importances = [float(value) for value in model.feature_importances_]  # type: ignore[union-attr]
        top_indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:top_n]
        return {
            "positive": [{"feature": str(feature_names[i]), "weight": float(importances[i])} for i in top_indices],
            "negative": [],
        }

    # Linear models: use coef_
    if not hasattr(model, "coef_"):
        raise TypeError(f"Unsupported page scorer model: {type(model)!r}")

    typed_model = cast("_LinearCoefModel", model)
    coefficients = [float(value) for value in typed_model.coef_[0]]
    positive_indices = sorted(range(len(coefficients)), key=lambda index: coefficients[index], reverse=True)[:top_n]
    negative_indices = sorted(range(len(coefficients)), key=lambda index: coefficients[index])[:top_n]
    return {
        "positive": [
            {"feature": str(feature_names[index]), "weight": float(coefficients[index])} for index in positive_indices
        ],
        "negative": [
            {"feature": str(feature_names[index]), "weight": float(coefficients[index])} for index in negative_indices
        ],
    }


def _metrics_from_ranked_groups(
    *,
    grouped_examples: dict[str, list[PageTrainingExample]],
    ranker: Callable[[PageTrainingExample], ModelRankKey | PageRankKey],
) -> RankingMetrics:
    """Compute grouped ranking metrics from a deterministic ranker.

    Args:
        grouped_examples: Question-grouped page examples.
        ranker: Sort key callable.

    Returns:
        Grouped ranking metrics.
    """
    question_count = 0
    hit_at_1 = 0
    hit_at_2 = 0
    reciprocal_rank_total = 0.0
    for group in grouped_examples.values():
        if not group:
            continue
        question_count += 1
        ranked = sorted(group, key=ranker)
        if any(example.label == 1 for example in ranked[:1]):
            hit_at_1 += 1
        if any(example.label == 1 for example in ranked[:2]):
            hit_at_2 += 1
        reciprocal_rank_total += _reciprocal_rank(ranked)
    if question_count == 0:
        return RankingMetrics(question_count=0, hit_at_1=0.0, hit_at_2=0.0, mean_reciprocal_rank=0.0)
    return RankingMetrics(
        question_count=question_count,
        hit_at_1=hit_at_1 / question_count,
        hit_at_2=hit_at_2 / question_count,
        mean_reciprocal_rank=reciprocal_rank_total / question_count,
    )


def _reciprocal_rank(group: Sequence[PageTrainingExample]) -> float:
    """Return reciprocal rank of the first positive page in a ranked group.

    Args:
        group: Ranked page examples.

    Returns:
        Reciprocal rank, or `0.0` when no positive exists.
    """
    for index, example in enumerate(group, start=1):
        if example.label == 1:
            return 1.0 / float(index)
    return 0.0


def _heuristic_rank_key(example: PageTrainingExample) -> tuple[int, int, int, int, int, int, int, int, str]:
    """Build a deterministic sort key for the heuristic page baseline.

    Args:
        example: Candidate page example.

    Returns:
        Sort key where lower is better.
    """
    features = example.features
    used = int(bool(features.get("from_sidecar_used")) or bool(features.get("from_legacy_used")))
    cited = int(bool(features.get("from_sidecar_cited")) or bool(features.get("from_legacy_cited")))
    context = int(bool(features.get("from_sidecar_context")) or bool(features.get("from_legacy_context")))
    retrieved = int(bool(features.get("from_sidecar_retrieved")) or bool(features.get("from_legacy_retrieved")))
    anchor_hits = int(features.get("anchor_hit_count", 0))
    answer_in_snippet = int(bool(features.get("answer_in_snippet")))
    best_context_rank = _min_positive_rank(
        int(features.get("sidecar_context_rank", 0)),
        int(features.get("legacy_context_rank", 0)),
    )
    best_retrieved_rank = _min_positive_rank(
        int(features.get("sidecar_retrieved_rank", 0)),
        int(features.get("legacy_retrieved_rank", 0)),
    )
    page_num = int(features.get("page_num", 0))
    return (
        -used,
        -cited,
        -context,
        -retrieved,
        -anchor_hits,
        -answer_in_snippet,
        best_context_rank,
        best_retrieved_rank,
        f"{page_num:06d}:{example.page_id}",
    )


def _build_model_ranker(score_lookup: dict[str, float]) -> Callable[[PageTrainingExample], ModelRankKey]:
    """Build a deterministic rank key from model score lookup.

    Args:
        score_lookup: Page-score mapping keyed by `question_id:page_id`.

    Returns:
        Callable rank key for grouped sorting.
    """

    def _rank_key(example: PageTrainingExample) -> ModelRankKey:
        score = score_lookup.get(f"{example.question_id}:{example.page_id}", float("-inf"))
        return (-score, example.page_id)

    return _rank_key


def _min_positive_rank(*values: int) -> int:
    """Return the best positive rank from a small rank family.

    Args:
        *values: Rank candidates where `0` means unavailable.

    Returns:
        Lowest positive rank, or a large sentinel when none are positive.
    """
    positive = [value for value in values if value > 0]
    if not positive:
        return 10_000_000
    return min(positive)
