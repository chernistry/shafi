# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Train a lightweight offline page scorer from grounding export candidates.

Supports logistic regression (baseline), LightGBM, and XGBoost model types.
Reports ranking metrics plus F-beta 2.5 (matching competition grounding metric).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold

from shafi.ml.page_scorer_training import (
    compute_heuristic_ranking_metrics,
    compute_ranking_metrics,
    count_question_sources,
    top_feature_weights,
)
from shafi.ml.training_scaffold import (
    PAGE_SCORER_FEATURE_POLICY,
    PageTrainingExample,
    build_page_training_examples,
    deterministic_subset,
    load_grounding_rows,
)

ModelType = str  # "logistic" | "lgbm" | "xgb"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for page-scorer training.

    Returns:
        Configured argparse parser.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=repo_root / "data" / "derived" / "grounding_ml" / "v2_reviewed" / "train.jsonl",
    )
    parser.add_argument(
        "--dev-jsonl",
        type=Path,
        default=repo_root / "data" / "derived" / "grounding_ml" / "v2_reviewed" / "dev.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=610)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-dev-rows", type=int, default=0)
    parser.add_argument(
        "--label-mode",
        choices=["reviewed_high_confidence", "reviewed_weighted", "reviewed_only", "soft_and_reviewed", "all"],
        default="reviewed_weighted",
    )
    parser.add_argument("--top-feature-count", type=int, default=12)
    parser.add_argument(
        "--augment-jsonl",
        type=Path,
        default=None,
        help="Optional augmentation JSONL (e.g. ObliQA) merged into training with lower weight.",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "lgbm", "xgb"],
        default="logistic",
        help="Model type: logistic regression (baseline), LightGBM, or XGBoost.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds (0 to disable).",
    )
    return parser


def _build_logistic_model(seed: int) -> LogisticRegression:
    """Build a logistic regression model with recall-favoring class weights.

    Args:
        seed: Random state seed.

    Returns:
        Configured LogisticRegression instance.
    """
    return LogisticRegression(
        max_iter=2000,
        class_weight={0: 1, 1: 6.25},
        random_state=seed,
        solver="liblinear",
    )


def _build_lgbm_model(seed: int) -> Any:
    """Build a LightGBM classifier configured for recall-dominated grounding.

    Args:
        seed: Random state seed.

    Returns:
        Configured LGBMClassifier instance.

    Raises:
        ImportError: If lightgbm is not installed.
    """
    from lightgbm import LGBMClassifier  # type: ignore[import-untyped]

    return LGBMClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        num_leaves=8,
        min_child_samples=15,
        scale_pos_weight=6.25,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=seed,
        verbose=-1,
    )


def _build_xgb_model(seed: int) -> Any:
    """Build an XGBoost classifier configured for recall-dominated grounding.

    Args:
        seed: Random state seed.

    Returns:
        Configured XGBClassifier instance.

    Raises:
        ImportError: If xgboost is not installed.
    """
    from xgboost import XGBClassifier  # type: ignore[import-untyped]

    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=6.25,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )


def _build_model(model_type: ModelType, seed: int) -> Any:
    """Build a model instance by type.

    Args:
        model_type: One of "logistic", "lgbm", "xgb".
        seed: Random state seed.

    Returns:
        Configured model instance.
    """
    if model_type == "lgbm":
        return _build_lgbm_model(seed)
    if model_type == "xgb":
        return _build_xgb_model(seed)
    return _build_logistic_model(seed)


def _predict_proba_positive(model: Any, x: Any) -> Any:
    """Extract positive-class probabilities from a model.

    Args:
        model: Fitted classifier.
        x: Feature matrix.

    Returns:
        Array of positive-class probabilities.
    """
    return model.predict_proba(x)[:, 1]  # type: ignore[no-any-return]


def _compute_fbeta_25(
    examples: list[PageTrainingExample],
    scores: Any,
    *,
    threshold: float = 0.5,
) -> float:
    """Compute F-beta 2.5 score (matching competition grounding metric).

    Args:
        examples: Training examples with labels.
        scores: Predicted positive-class probabilities.
        threshold: Classification threshold.

    Returns:
        F-beta 2.5 score.
    """
    if not examples:
        return 0.0
    y_true = np.array([ex.label for ex in examples])
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    return float(fbeta_score(y_true, y_pred, beta=2.5, zero_division="warn"))


def _cross_validate(
    examples: list[PageTrainingExample],
    *,
    model_type: ModelType,
    n_folds: int,
    seed: int,
) -> dict[str, float]:
    """Run stratified k-fold cross-validation and report mean metrics.

    Stratification is by label to ensure positive/negative balance in each fold.

    Args:
        examples: All training examples.
        model_type: Model type to train.
        n_folds: Number of CV folds.
        seed: Random state seed.

    Returns:
        Dict with mean CV metrics.
    """
    if n_folds < 2 or len(examples) < n_folds:
        return {}

    labels = np.array([ex.label for ex in examples])
    n_positive = int(labels.sum())
    if n_positive < n_folds or (len(labels) - n_positive) < n_folds:
        return {}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics: list[dict[str, float]] = []

    for train_idx, val_idx in skf.split(labels, labels):
        fold_train = [examples[i] for i in train_idx]
        fold_val = [examples[i] for i in val_idx]

        vec = DictVectorizer(sparse=True)
        x_tr = vec.fit_transform([ex.features for ex in fold_train])
        x_val = vec.transform([ex.features for ex in fold_val])
        y_tr = [ex.label for ex in fold_train]
        sw_tr = [ex.sample_weight for ex in fold_train]

        fold_model = _build_model(model_type, seed)
        fold_model.fit(x_tr, y_tr, sample_weight=sw_tr)

        val_scores = _predict_proba_positive(fold_model, x_val)
        ranking = compute_ranking_metrics(fold_val, val_scores)
        fbeta = _compute_fbeta_25(fold_val, val_scores)

        fold_metrics.append({
            "hit_at_1": ranking.hit_at_1,
            "hit_at_2": ranking.hit_at_2,
            "mrr": ranking.mean_reciprocal_rank,
            "fbeta_2_5": fbeta,
        })

    means: dict[str, float] = {}
    for key in fold_metrics[0]:
        values = [fm[key] for fm in fold_metrics]
        means[f"cv_mean_{key}"] = float(np.mean(values))
        means[f"cv_std_{key}"] = float(np.std(values))
    return means


def main() -> int:
    """Run page-scorer training and emit model artifacts.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    train_rows = deterministic_subset(
        load_grounding_rows(args.train_jsonl), limit=args.max_train_rows or None, seed=args.seed
    )
    dev_rows = deterministic_subset(
        load_grounding_rows(args.dev_jsonl), limit=args.max_dev_rows or None, seed=args.seed
    )
    train_examples = build_page_training_examples(train_rows, label_mode=args.label_mode)
    dev_examples = build_page_training_examples(dev_rows, label_mode=args.label_mode)

    # Optional augmentation data (e.g. ObliQA) — merged with lower weight.
    if args.augment_jsonl and args.augment_jsonl.is_file():
        aug_rows = load_grounding_rows(args.augment_jsonl)
        aug_examples = build_page_training_examples(aug_rows, label_mode="all")
        for ex in aug_examples:
            object.__setattr__(ex, "sample_weight", ex.sample_weight * 0.15)
        print(f"Augmentation: {len(aug_examples)} examples from {args.augment_jsonl}")
        train_examples = train_examples + aug_examples

    # --- Vectorize ---
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([ex.features for ex in train_examples])
    y_train = [ex.label for ex in train_examples]
    sample_weight = [ex.sample_weight for ex in train_examples]

    # --- Train model ---
    model_type: ModelType = args.model_type
    print(f"Training {model_type} model...")
    model = _build_model(model_type, args.seed)
    model.fit(x_train, y_train, sample_weight=sample_weight)

    # --- Evaluate ---
    train_scores = _predict_proba_positive(model, vectorizer.transform([ex.features for ex in train_examples]))
    dev_scores = (
        _predict_proba_positive(model, vectorizer.transform([ex.features for ex in dev_examples]))
        if dev_examples
        else np.array([])
    )
    train_metrics = compute_ranking_metrics(train_examples, list(train_scores))
    dev_metrics = compute_ranking_metrics(dev_examples, list(dev_scores))
    heuristic_metrics = compute_heuristic_ranking_metrics(dev_examples)
    feature_importance = top_feature_weights(model, vectorizer, top_n=args.top_feature_count)

    train_fbeta = _compute_fbeta_25(train_examples, train_scores)
    dev_fbeta = _compute_fbeta_25(dev_examples, dev_scores)

    # --- Cross-validate ---
    cv_metrics: dict[str, float] = {}
    if args.cv_folds >= 2:
        print(f"Running {args.cv_folds}-fold cross-validation...")
        cv_metrics = _cross_validate(
            train_examples,
            model_type=model_type,
            n_folds=args.cv_folds,
            seed=args.seed,
        )

    metrics: dict[str, Any] = {
        "model_type": model_type,
        "train_example_count": len(train_examples),
        "dev_example_count": len(dev_examples),
        "train_question_count": len({ex.question_id for ex in train_examples}),
        "dev_question_count": len({ex.question_id for ex in dev_examples}),
        "train_hit_at_1": train_metrics.hit_at_1,
        "train_hit_at_2": train_metrics.hit_at_2,
        "train_mrr": train_metrics.mean_reciprocal_rank,
        "train_fbeta_2_5": train_fbeta,
        "dev_hit_at_1": dev_metrics.hit_at_1,
        "dev_hit_at_2": dev_metrics.hit_at_2,
        "dev_mrr": dev_metrics.mean_reciprocal_rank,
        "dev_fbeta_2_5": dev_fbeta,
        "heuristic_dev_hit_at_1": heuristic_metrics.hit_at_1,
        "heuristic_dev_hit_at_2": heuristic_metrics.hit_at_2,
        "heuristic_dev_mrr": heuristic_metrics.mean_reciprocal_rank,
        "train_supervision_question_counts": count_question_sources(train_examples),
        "dev_supervision_question_counts": count_question_sources(dev_examples),
        "label_mode": args.label_mode,
        "feature_policy": PAGE_SCORER_FEATURE_POLICY,
        "label_quality_note": _build_label_quality_note(args.label_mode),
        **cv_metrics,
    }

    # --- Save artifacts ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "model_type": model_type,
            "seed": args.seed,
            "label_mode": args.label_mode,
            "feature_policy": PAGE_SCORER_FEATURE_POLICY,
        },
        args.output_dir / "page_scorer.joblib",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "train_jsonl": str(args.train_jsonl),
                "dev_jsonl": str(args.dev_jsonl),
                "seed": args.seed,
                "model_type": model_type,
                "label_mode": args.label_mode,
                "feature_policy": PAGE_SCORER_FEATURE_POLICY,
                "max_train_rows": args.max_train_rows,
                "max_dev_rows": args.max_dev_rows,
                "top_feature_count": args.top_feature_count,
                "cv_folds": args.cv_folds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "feature_importance.json").write_text(
        json.dumps(feature_importance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)
    print(json.dumps(metrics, indent=2))
    return 0


def _build_label_quality_note(label_mode: str) -> str:
    """Describe the supervision policy used for the current training run.

    Args:
        label_mode: CLI label-mode string.

    Returns:
        Short human-readable supervision note for metrics artifacts.
    """
    if label_mode == "reviewed_high_confidence":
        return "Only reviewed high-confidence page labels are used for direct supervision."
    if label_mode == "reviewed_weighted":
        return "Reviewed page labels are used with confidence-aware weighting; low-confidence rows are excluded."
    if label_mode == "reviewed_only":
        return "Only reviewed page labels are used for direct supervision."
    if label_mode == "soft_and_reviewed":
        return "Reviewed labels are primary and soft AI-gold rows are kept only as low-value diagnostics."
    return (
        "suspect_ai_gold page labels are excluded from direct supervision; "
        "the scorer trains on reviewed/soft labels plus low-weight heuristic fallback pages."
    )


if __name__ == "__main__":
    raise SystemExit(main())
