# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportCallIssue=false
"""Offline training and evaluation helpers for retrieval utility prediction."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

from shafi.core.retrieval_utility import BundleSnapshot, RetrievalUtilityPredictor, UtilityPredictorArtifact


@dataclass(frozen=True, slots=True)
class UtilityTrainingExample:
    """One labeled bundle-sufficiency example."""

    question_id: str
    snapshot: BundleSnapshot
    label: int


@dataclass(frozen=True, slots=True)
class UtilityMetrics:
    """Summary metrics for a utility predictor benchmark."""

    sample_count: int
    positive_count: int
    accuracy: float
    f1: float
    auroc: float
    heuristic_accuracy: float
    heuristic_f1: float
    accuracy_delta: float
    f1_delta: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly metrics dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FittedUtilityPredictor:
    """Fitted vectorizer/model bundle for retrieval utility prediction."""

    artifact: UtilityPredictorArtifact
    feature_policy: str


class UtilityPredictorTrainer:
    """Train and evaluate an offline retrieval utility predictor."""

    feature_policy = "bundle_utility_r1"

    def build_training_examples(
        self,
        raw_results: Sequence[Mapping[str, Any]],
        reviewed_gold: Mapping[str, Mapping[str, Any]],
    ) -> list[UtilityTrainingExample]:
        """Build labeled utility examples from raw results and reviewed gold.

        Args:
            raw_results: Raw-results rows from a benchmark artifact.
            reviewed_gold: Reviewed gold rows keyed by question_id.

        Returns:
            Labeled bundle examples with missing-gold rows skipped.
        """
        examples: list[UtilityTrainingExample] = []
        for row in raw_results:
            snapshot = BundleSnapshot.from_raw_result(row)
            if not snapshot.question_id or snapshot.question_id not in reviewed_gold:
                continue
            gold_row = reviewed_gold[snapshot.question_id]
            label = _answer_is_correct(
                answer_type=str(gold_row.get("answer_type", snapshot.answer_type)),
                predicted_answer=str(row.get("answer_text", "")),
                golden_answer=gold_row.get("golden_answer"),
            )
            examples.append(
                UtilityTrainingExample(
                    question_id=snapshot.question_id,
                    snapshot=snapshot,
                    label=label,
                )
            )
        return examples

    def fit(self, examples: Sequence[UtilityTrainingExample], *, threshold: float = 0.5) -> FittedUtilityPredictor:
        """Fit a logistic model on bundle-sufficiency examples.

        Args:
            examples: Labeled training examples.
            threshold: Sufficiency threshold for routing.

        Returns:
            Fitted model artifact.
        """
        vectorizer = DictVectorizer(sparse=True)
        features = [RetrievalUtilityPredictor().extract_features(example.snapshot) for example in examples]
        x_train = vectorizer.fit_transform(features)
        model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)
        model.fit(x_train, [example.label for example in examples])
        artifact = UtilityPredictorArtifact(
            vectorizer=vectorizer,
            model=model,
            threshold=threshold,
            feature_policy=self.feature_policy,
        )
        return FittedUtilityPredictor(artifact=artifact, feature_policy=self.feature_policy)

    def evaluate(
        self,
        fitted: FittedUtilityPredictor,
        examples: Sequence[UtilityTrainingExample],
    ) -> UtilityMetrics:
        """Evaluate a fitted predictor against a labeled set.

        Args:
            fitted: Fitted model artifact.
            examples: Evaluation examples.

        Returns:
            Aggregate metrics, including a trivial heuristic baseline.
        """
        predictor = RetrievalUtilityPredictor(artifact=fitted.artifact, threshold=fitted.artifact.threshold)
        y_true = [example.label for example in examples]
        probabilities = [float(predictor.predict(example.snapshot).bundle_sufficiency) for example in examples]
        y_pred = [int(prob >= fitted.artifact.threshold) for prob in probabilities]
        heuristic_pred = [int(_heuristic_bundle_sufficient(example.snapshot)) for example in examples]
        return UtilityMetrics(
            sample_count=len(examples),
            positive_count=sum(y_true),
            accuracy=_safe_accuracy(y_true, y_pred),
            f1=_safe_f1(y_true, y_pred),
            auroc=_safe_auroc(y_true, probabilities),
            heuristic_accuracy=_safe_accuracy(y_true, heuristic_pred),
            heuristic_f1=_safe_f1(y_true, heuristic_pred),
            accuracy_delta=_safe_accuracy(y_true, y_pred) - _safe_accuracy(y_true, heuristic_pred),
            f1_delta=_safe_f1(y_true, y_pred) - _safe_f1(y_true, heuristic_pred),
        )

    def cross_validate(self, examples: Sequence[UtilityTrainingExample], *, folds: int = 5) -> UtilityMetrics:
        """Run a deterministic stratified cross-validation benchmark.

        Args:
            examples: Labeled examples.
            folds: Number of folds.

        Returns:
            Mean cross-validation metrics.
        """
        if len(examples) < folds:
            return self.evaluate(self.fit(examples), examples)

        vectorizer = DictVectorizer(sparse=True)
        x = vectorizer.fit_transform(
            [RetrievalUtilityPredictor().extract_features(example.snapshot) for example in examples]
        )
        y = [example.label for example in examples]
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        accuracies: list[float] = []
        f1_scores: list[float] = []
        auc_scores: list[float] = []
        heuristic_accuracies: list[float] = []
        heuristic_f1_scores: list[float] = []
        for train_idx, test_idx in skf.split(x, y):
            model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)
            model.fit(x[train_idx], [y[index] for index in train_idx])
            proba = [float(prob) for prob in model.predict_proba(x[test_idx])[:, 1]]
            pred = [int(prob >= 0.5) for prob in proba]
            y_test = [y[index] for index in test_idx]
            accuracies.append(_safe_accuracy(y_test, pred))
            f1_scores.append(_safe_f1(y_test, pred))
            auc_scores.append(_safe_auroc(y_test, proba))
            heuristic_pred = [int(_heuristic_bundle_sufficient(examples[index].snapshot)) for index in test_idx]
            heuristic_accuracies.append(_safe_accuracy(y_test, heuristic_pred))
            heuristic_f1_scores.append(_safe_f1(y_test, heuristic_pred))

        mean_accuracy = sum(accuracies) / len(accuracies)
        mean_f1 = sum(f1_scores) / len(f1_scores)
        mean_auc = sum(auc_scores) / len(auc_scores)
        mean_heuristic_accuracy = sum(heuristic_accuracies) / len(heuristic_accuracies)
        mean_heuristic_f1 = sum(heuristic_f1_scores) / len(heuristic_f1_scores)
        return UtilityMetrics(
            sample_count=len(examples),
            positive_count=sum(y),
            accuracy=mean_accuracy,
            f1=mean_f1,
            auroc=mean_auc,
            heuristic_accuracy=mean_heuristic_accuracy,
            heuristic_f1=mean_heuristic_f1,
            accuracy_delta=mean_accuracy - mean_heuristic_accuracy,
            f1_delta=mean_f1 - mean_heuristic_f1,
        )

    def save(self, fitted: FittedUtilityPredictor, output_dir: Path) -> Path:
        """Persist a fitted predictor artifact to disk.

        Args:
            fitted: Fitted model bundle.
            output_dir: Target directory.

        Returns:
            Path to the serialized artifact.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / "retrieval_utility_predictor.joblib"
        joblib.dump(
            {
                "vectorizer": fitted.artifact.vectorizer,
                "model": fitted.artifact.model,
                "threshold": fitted.artifact.threshold,
                "feature_policy": fitted.artifact.feature_policy,
            },
            artifact_path,
        )
        return artifact_path

    def load(self, artifact_path: Path) -> RetrievalUtilityPredictor:
        """Load a saved predictor artifact from disk.

        Args:
            artifact_path: Serialized artifact path.

        Returns:
            Predictor ready for inference.
        """
        payload = cast("dict[str, Any]", joblib.load(artifact_path))
        fitted = UtilityPredictorArtifact(
            vectorizer=payload["vectorizer"],
            model=payload["model"],
            threshold=float(payload["threshold"]),
            feature_policy=str(payload["feature_policy"]),
        )
        return RetrievalUtilityPredictor(artifact=fitted, threshold=fitted.threshold)


def load_reviewed_gold(path: Path) -> dict[str, dict[str, Any]]:
    """Load reviewed gold rows keyed by question id.

    Args:
        path: Path to reviewed gold JSON.

    Returns:
        Mapping from question id to gold row.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["question_id"]): cast("dict[str, Any]", row) for row in rows}


def load_raw_results(path: Path) -> list[dict[str, Any]]:
    """Load a raw-results artifact as JSON.

    Args:
        path: Path to a raw-results JSON file.

    Returns:
        Parsed raw-result rows.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [cast("dict[str, Any]", row) for row in rows]


def build_feature_importance(
    fitted: FittedUtilityPredictor, *, top_n: int = 12
) -> dict[str, list[dict[str, float | str]]]:
    """Extract top positive and negative feature weights.

    Args:
        fitted: Fitted predictor artifact.
        top_n: Number of features to keep per polarity.

    Returns:
        Feature weight summary.
    """
    vectorizer = fitted.artifact.vectorizer
    model = fitted.artifact.model
    feature_names = [str(name) for name in vectorizer.get_feature_names_out()]
    coefficients = [float(value) for value in model.coef_[0]]
    positive_indices = sorted(range(len(coefficients)), key=lambda index: coefficients[index], reverse=True)[:top_n]
    negative_indices = sorted(range(len(coefficients)), key=lambda index: coefficients[index])[:top_n]
    return {
        "positive": [{"feature": feature_names[index], "weight": coefficients[index]} for index in positive_indices],
        "negative": [{"feature": feature_names[index], "weight": coefficients[index]} for index in negative_indices],
    }


def _answer_is_correct(*, answer_type: str, predicted_answer: str, golden_answer: object) -> int:
    """Return whether the predicted answer matches the gold answer.

    Args:
        answer_type: Competition answer type.
        predicted_answer: Model output text.
        golden_answer: Reference answer from reviewed gold.

    Returns:
        1 when the answer is judged correct, otherwise 0.
    """
    del answer_type
    if golden_answer is None:
        return 0
    if isinstance(golden_answer, bool):
        return int(_normalize(predicted_answer) == ("yes" if golden_answer else "no"))
    if isinstance(golden_answer, (int, float)):
        return int(_normalize(predicted_answer) == _normalize(str(golden_answer)))
    golden = _normalize(str(golden_answer))
    predicted = _normalize(predicted_answer)
    return int(predicted == golden or golden in predicted or predicted in golden)


def _heuristic_bundle_sufficient(snapshot: BundleSnapshot) -> bool:
    """Simple heuristic baseline for bundle sufficiency.

    Args:
        snapshot: Input bundle snapshot.

    Returns:
        Heuristic yes/no sufficiency prediction.
    """
    return snapshot.cited_page_count > 0 or snapshot.context_page_count >= 3


def _normalize(text: str) -> str:
    """Normalize answer text for rough exact-match scoring."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _safe_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return accuracy with empty-set protection."""
    if not y_true:
        return 0.0
    return float(accuracy_score(y_true, y_pred))


def _safe_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return F1 with empty-set protection."""
    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred))


def _safe_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Return AUROC with single-class protection."""
    if not y_true or len(set(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))
