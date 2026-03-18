"""Train an offline grounding router from internal and auxiliary external data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from rag_challenge.ml.router_training import (
    RouterTrainingExample,
    build_external_router_examples,
    build_internal_router_examples,
    load_external_normalized_rows,
)
from rag_challenge.ml.training_scaffold import deterministic_subset, load_grounding_rows

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for router training.

    Returns:
        Configured argparse parser.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=repo_root / "data" / "derived" / "grounding_ml" / "v1" / "train.jsonl")
    parser.add_argument("--dev-jsonl", type=Path, default=repo_root / "data" / "derived" / "grounding_ml" / "v1" / "dev.jsonl")
    parser.add_argument(
        "--external-jsonl",
        type=Path,
        default=repo_root / "data" / "external" / "normalized" / "v1" / "normalized_rows.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=616)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-dev-rows", type=int, default=0)
    parser.add_argument("--max-external-rows", type=int, default=0)
    parser.add_argument("--external-sample-weight", type=float, default=0.25)
    parser.add_argument("--top-feature-count", type=int, default=12)
    parser.add_argument("--disable-external-augmentation", action="store_true")
    return parser


def main() -> int:
    """Run offline grounding-router training and emit artifacts.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()

    internal_train_rows = deterministic_subset(
        load_grounding_rows(args.train_jsonl),
        limit=args.max_train_rows or None,
        seed=args.seed,
    )
    internal_dev_rows = deterministic_subset(
        load_grounding_rows(args.dev_jsonl),
        limit=args.max_dev_rows or None,
        seed=args.seed,
    )

    internal_train_examples = build_internal_router_examples(internal_train_rows)
    internal_dev_examples = build_internal_router_examples(internal_dev_rows)

    external_examples: list[RouterTrainingExample] = []
    external_source_counts: dict[str, int] = {}
    if not args.disable_external_augmentation and args.external_jsonl.exists():
        external_rows = _deterministic_external_subset(
            load_external_normalized_rows(args.external_jsonl),
            limit=args.max_external_rows or None,
            seed=args.seed,
        )
        external_examples = build_external_router_examples(
            external_rows,
            sample_weight=args.external_sample_weight,
        )
        external_source_counts = dict(Counter(example.source for example in external_examples))

    combined_train_examples = [*internal_train_examples, *external_examples]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vectorizer.fit([example.text for example in combined_train_examples])

    scope_model, scope_metrics, scope_confusion = _train_single_label_model(
        train_examples=[example for example in internal_train_examples if example.scope_target is not None],
        dev_examples=[example for example in internal_dev_examples if example.scope_target is not None],
        vectorizer=vectorizer,
        target_getter=lambda example: example.scope_target or "",
        majority_target=_majority_label(
            [example.scope_target or "" for example in internal_train_examples if example.scope_target is not None]
        ),
        top_feature_count=args.top_feature_count,
    )

    budget_model, budget_metrics, budget_confusion = _train_single_label_model(
        train_examples=[example for example in internal_train_examples if example.page_budget_target is not None],
        dev_examples=[example for example in internal_dev_examples if example.page_budget_target is not None],
        vectorizer=vectorizer,
        target_getter=lambda example: str(example.page_budget_target),
        majority_target=_majority_label(
            [
                str(example.page_budget_target)
                for example in internal_train_examples
                if example.page_budget_target is not None
            ]
        ),
        top_feature_count=args.top_feature_count,
    )

    roles_model, roles_metrics = _train_roles_model(
        train_examples=[example for example in combined_train_examples if example.role_targets],
        dev_examples=internal_dev_examples,
        vectorizer=vectorizer,
        top_feature_count=args.top_feature_count,
    )

    metrics = {
        "internal_train_count": len(internal_train_examples),
        "internal_dev_count": len(internal_dev_examples),
        "external_train_count": len(external_examples),
        "external_source_counts": external_source_counts,
        "scope_accuracy": scope_metrics["accuracy"],
        "scope_majority_accuracy": scope_metrics["majority_accuracy"],
        "scope_rule_reference_accuracy": 1.0,
        "budget_accuracy": budget_metrics["accuracy"],
        "budget_majority_accuracy": budget_metrics["majority_accuracy"],
        "budget_rule_reference_accuracy": 1.0,
        "roles_micro_f1": roles_metrics["micro_f1"],
        "roles_majority_micro_f1": roles_metrics["majority_micro_f1"],
        "roles_rule_reference_micro_f1": 1.0,
        "router_reference_note": (
            "Internal dev targets are generated by the current rule-based router. "
            "The rule router is therefore a ceiling/reference, not a fair competitor."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "scope_model": scope_model,
            "budget_model": budget_model,
            "roles_model": roles_model,
            "role_labels": roles_model["role_labels"],
            "seed": args.seed,
        },
        args.output_dir / "router.joblib",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "train_jsonl": str(args.train_jsonl),
                "dev_jsonl": str(args.dev_jsonl),
                "external_jsonl": str(args.external_jsonl),
                "seed": args.seed,
                "max_train_rows": args.max_train_rows,
                "max_dev_rows": args.max_dev_rows,
                "max_external_rows": args.max_external_rows,
                "external_sample_weight": args.external_sample_weight,
                "disable_external_augmentation": args.disable_external_augmentation,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "confusion_matrix.json").write_text(
        json.dumps(
            {
                "scope": scope_confusion,
                "budget": budget_confusion,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "top_features.json").write_text(
        json.dumps(
            {
                "scope": scope_metrics["top_features"],
                "budget": budget_metrics["top_features"],
                "roles": roles_metrics["top_features"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)
    print(json.dumps(metrics, indent=2))
    return 0


def _train_single_label_model(
    *,
    train_examples: Sequence[RouterTrainingExample],
    dev_examples: Sequence[RouterTrainingExample],
    vectorizer: TfidfVectorizer,
    target_getter,
    majority_target: str,
    top_feature_count: int,
) -> tuple[LogisticRegression, dict[str, object], dict[str, object]]:
    """Train and evaluate one single-label router head.

    Args:
        train_examples: Training examples with valid targets.
        dev_examples: Internal dev examples with valid targets.
        vectorizer: Shared TF-IDF vectorizer.
        target_getter: Callable extracting the target string.
        majority_target: Simple baseline label.
        top_feature_count: Number of top features to emit per class.

    Returns:
        Tuple of fitted model, metrics dictionary, and confusion-matrix payload.
    """
    x_train = vectorizer.transform([example.text for example in train_examples])
    y_train = [target_getter(example) for example in train_examples]
    weights = [example.sample_weight for example in train_examples]
    x_dev = vectorizer.transform([example.text for example in dev_examples])
    y_dev = [target_getter(example) for example in dev_examples]

    model = LogisticRegression(max_iter=400, class_weight="balanced", random_state=616)
    model.fit(x_train, y_train, sample_weight=weights)
    predictions = list(model.predict(x_dev)) if dev_examples else []
    majority_predictions = [majority_target] * len(y_dev)

    labels = [str(label) for label in model.classes_]
    confusion = {
        "labels": labels,
        "matrix": confusion_matrix(y_dev, predictions, labels=labels).tolist() if y_dev else [],
    }
    metrics = {
        "accuracy": float(accuracy_score(y_dev, predictions)) if y_dev else 0.0,
        "majority_accuracy": float(accuracy_score(y_dev, majority_predictions)) if y_dev else 0.0,
        "top_features": _top_features_for_logistic(model, vectorizer, top_n=top_feature_count),
    }
    return model, metrics, confusion


def _train_roles_model(
    *,
    train_examples: Sequence[RouterTrainingExample],
    dev_examples: Sequence[RouterTrainingExample],
    vectorizer: TfidfVectorizer,
    top_feature_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train and evaluate the multi-label page-role head.

    Args:
        train_examples: Training examples with non-empty role targets.
        dev_examples: Internal dev examples.
        vectorizer: Shared TF-IDF vectorizer.
        top_feature_count: Number of top features per role.

    Returns:
        Tuple of model bundle and metrics.
    """
    x_train = vectorizer.transform([example.text for example in train_examples])
    y_train_labels = [example.role_targets for example in train_examples]
    weights = [example.sample_weight for example in train_examples]

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_labels)
    estimators: list[LogisticRegression] = []
    for column_index in range(y_train.shape[1]):
        estimator = LogisticRegression(max_iter=400, class_weight="balanced", random_state=616)
        estimator.fit(x_train, y_train[:, column_index], sample_weight=weights)
        estimators.append(estimator)

    x_dev = vectorizer.transform([example.text for example in dev_examples])
    y_dev = mlb.transform([example.role_targets for example in dev_examples])
    prediction_columns = [estimator.predict(x_dev).tolist() for estimator in estimators] if dev_examples else []
    predictions = [list(values) for values in zip(*prediction_columns, strict=False)] if prediction_columns else []

    majority_roles = _majority_role_set([example.role_targets for example in train_examples])
    majority_matrix = mlb.transform([majority_roles] * len(dev_examples)) if dev_examples else []

    metrics = {
        "micro_f1": float(f1_score(y_dev, predictions, average="micro", zero_division=0)) if dev_examples else 0.0,
        "majority_micro_f1": (
            float(f1_score(y_dev, majority_matrix, average="micro", zero_division=0)) if dev_examples else 0.0
        ),
        "top_features": _top_features_for_ovr(estimators, vectorizer, list(mlb.classes_), top_n=top_feature_count),
    }
    model_bundle = {
        "model_type": "independent_logreg_roles",
        "estimators": estimators,
        "role_labels": list(mlb.classes_),
    }
    return model_bundle, metrics


def _deterministic_external_subset(
    rows,
    *,
    limit: int | None,
    seed: int,
):
    """Return a deterministic subset of external rows.

    Args:
        rows: External normalized rows.
        limit: Optional maximum row count.
        seed: Deterministic selection seed.

    Returns:
        Ordered subset of rows.
    """
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row.source_dataset}:{row.sample_id}".encode()).hexdigest(),
            row.source_dataset,
            row.sample_id,
        ),
    )
    if limit is None or limit <= 0 or limit >= len(ordered):
        return list(ordered)
    return list(ordered[:limit])


def _majority_label(values: Sequence[str]) -> str:
    """Return the most common class label.

    Args:
        values: Candidate label values.

    Returns:
        Most common label, or an empty string.
    """
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def _majority_role_set(role_lists: Sequence[Sequence[str]]) -> list[str]:
    """Build a simple constant multi-label baseline role set.

    Args:
        role_lists: Training role targets.

    Returns:
        Constant role prediction set for the baseline.
    """
    counts: Counter[str] = Counter()
    for roles in role_lists:
        counts.update(roles)
    if not counts:
        return []
    threshold = max(1, round(len(role_lists) * 0.35))
    selected = [role for role, count in counts.items() if count >= threshold]
    if selected:
        return sorted(selected)
    return [counts.most_common(1)[0][0]]


def _top_features_for_logistic(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    *,
    top_n: int,
) -> dict[str, list[str]]:
    """Extract top weighted features for a logistic-regression model.

    Args:
        model: Fitted logistic-regression model.
        vectorizer: Fitted TF-IDF vectorizer.
        top_n: Number of features per class.

    Returns:
        Mapping from class label to feature list.
    """
    feature_names = vectorizer.get_feature_names_out()
    if len(model.classes_) == 2 and model.coef_.shape[0] == 1:
        coef = model.coef_[0]
        negative = coef.argsort()[:top_n]
        positive = coef.argsort()[-top_n:][::-1]
        return {
            str(model.classes_[0]): [str(feature_names[index]) for index in negative],
            str(model.classes_[1]): [str(feature_names[index]) for index in positive],
        }

    top_features: dict[str, list[str]] = {}
    for index, label in enumerate(model.classes_):
        top_indices = model.coef_[index].argsort()[-top_n:][::-1]
        top_features[str(label)] = [str(feature_names[position]) for position in top_indices]
    return top_features


def _top_features_for_ovr(
    estimators: Sequence[LogisticRegression],
    vectorizer: TfidfVectorizer,
    labels: Sequence[str],
    *,
    top_n: int,
) -> dict[str, list[str]]:
    """Extract top weighted features for each one-vs-rest role estimator.

    Args:
        estimators: Fitted per-role logistic regressions.
        vectorizer: Fitted TF-IDF vectorizer.
        labels: Role label order.
        top_n: Number of features per role.

    Returns:
        Mapping from role label to top feature list.
    """
    feature_names = vectorizer.get_feature_names_out()
    top_features: dict[str, list[str]] = {}
    for label, estimator in zip(labels, estimators, strict=False):
        top_indices = estimator.coef_[0].argsort()[-top_n:][::-1]
        top_features[str(label)] = [str(feature_names[position]) for position in top_indices]
    return top_features


if __name__ == "__main__":
    raise SystemExit(main())
