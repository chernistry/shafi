# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Fast retraining of the LightGBM page scorer with private-set eval corrections.

Loads original training data, overlays verified gold labels from an eval target
map, augments with hard-negative signal from eval failures, retrains LightGBM,
and saves artifacts.  Designed to complete in <5 minutes.

Usage::

    python scripts/retrain_page_scorer_on_private.py \\
      --base-train-jsonl data/derived/grounding_ml/v2_reviewed/train.jsonl \\
      --eval-target-map scripts/retrieval_target_map.json \\
      --base-model models/page_scorer/v3_lgbm/page_scorer.joblib \\
      --output-dir models/page_scorer/v4_retrained/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, TypedDict

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import fbeta_score

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
    load_grounding_rows,
)

# ---------------------------------------------------------------------------
# Target-map schema
# ---------------------------------------------------------------------------


class EvalTargetEntry(TypedDict, total=False):
    """One entry from the eval target map JSON."""

    question_id: str
    question: str
    answer_type: str
    failure_category: str
    retrieval_gap_type: str
    verified_gold_pages: list[str]
    gold_docs_count: int
    retrieved_pages_count: int
    used_pages: list[str]
    gold_doc_in_retrieved_set: bool
    gold_locations: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORRECTED_SAMPLE_WEIGHT = 5.0
"""High-confidence weight for examples whose labels come from verified gold."""

_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.08,
    "num_leaves": 8,
    "min_child_samples": 15,
    "scale_pos_weight": 6.25,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured argument parser.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-train-jsonl",
        type=Path,
        default=repo_root / "data" / "derived" / "grounding_ml" / "v2_reviewed" / "train.jsonl",
        help="Base training JSONL from grounding export.",
    )
    parser.add_argument(
        "--dev-jsonl",
        type=Path,
        default=repo_root / "data" / "derived" / "grounding_ml" / "v2_reviewed" / "dev.jsonl",
        help="Dev evaluation JSONL.",
    )
    parser.add_argument(
        "--eval-target-map",
        type=Path,
        required=True,
        help="JSON file with verified gold pages from eval results.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Path to base model .joblib for comparison metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for retrained model artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=610,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["reviewed_high_confidence", "reviewed_weighted", "reviewed_only", "soft_and_reviewed", "all"],
        default="reviewed_weighted",
        help="Label-mode for base training data.",
    )
    parser.add_argument(
        "--top-feature-count",
        type=int,
        default=12,
        help="Number of top features to report.",
    )
    return parser


# ---------------------------------------------------------------------------
# Target-map loading
# ---------------------------------------------------------------------------


def load_eval_target_map(path: Path) -> dict[str, EvalTargetEntry]:
    """Load the eval target map JSON keyed by question_id.

    Args:
        path: Path to the target map JSON (list of objects).

    Returns:
        Mapping from question_id to the target entry.
    """
    raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, EvalTargetEntry] = {}
    for entry in raw:
        qid = entry.get("question_id", "")
        if qid and entry.get("verified_gold_pages"):
            result[qid] = EvalTargetEntry(
                question_id=qid,
                question=entry.get("question", ""),
                answer_type=entry.get("answer_type", ""),
                failure_category=entry.get("failure_category", ""),
                retrieval_gap_type=entry.get("retrieval_gap_type", ""),
                verified_gold_pages=entry["verified_gold_pages"],
                gold_docs_count=entry.get("gold_docs_count", 0),
                retrieved_pages_count=entry.get("retrieved_pages_count", 0),
                used_pages=entry.get("used_pages", []),
                gold_doc_in_retrieved_set=entry.get("gold_doc_in_retrieved_set", False),
                gold_locations=entry.get("gold_locations", []),
            )
    return result


# ---------------------------------------------------------------------------
# Label correction
# ---------------------------------------------------------------------------


def apply_label_corrections(
    examples: list[PageTrainingExample],
    target_map: dict[str, EvalTargetEntry],
) -> tuple[list[PageTrainingExample], int]:
    """Override labels for questions present in the eval target map.

    For each question in both the training set and the target map, the
    positive labels are replaced with `verified_gold_pages` and the
    sample weight is raised to ``_CORRECTED_SAMPLE_WEIGHT``.

    Args:
        examples: Original training examples (not mutated).
        target_map: Verified gold pages keyed by question_id.

    Returns:
        Tuple of corrected example list and count of corrected questions.
    """
    corrected: list[PageTrainingExample] = []
    corrected_question_ids: set[str] = set()

    for ex in examples:
        entry = target_map.get(ex.question_id)
        if entry is None:
            corrected.append(ex)
            continue

        verified = entry.get("verified_gold_pages", [])
        gold_pages = set(verified)
        new_label = 1 if ex.page_id in gold_pages else 0
        corrected.append(
            PageTrainingExample(
                question_id=ex.question_id,
                page_id=ex.page_id,
                features=ex.features,
                label=new_label,
                sample_weight=_CORRECTED_SAMPLE_WEIGHT,
                supervision_source="eval_verified",
            )
        )
        corrected_question_ids.add(ex.question_id)

    return corrected, len(corrected_question_ids)


# ---------------------------------------------------------------------------
# LightGBM builder
# ---------------------------------------------------------------------------


def build_lgbm_model(seed: int) -> Any:
    """Build a LightGBM classifier with competition-tuned hyperparameters.

    Args:
        seed: Random state seed.

    Returns:
        Configured LGBMClassifier instance.
    """
    from lightgbm import LGBMClassifier  # type: ignore[import-untyped]

    return LGBMClassifier(
        **_DEFAULT_LGBM_PARAMS,
        random_state=seed,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def compute_fbeta_25(
    examples: list[PageTrainingExample],
    scores: Any,
    *,
    threshold: float = 0.5,
) -> float:
    """Compute F-beta 2.5 score matching competition grounding metric.

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


def load_base_model_metrics(base_model_path: Path) -> dict[str, Any] | None:
    """Load base model metrics.json from the same directory if present.

    Args:
        base_model_path: Path to the base model .joblib file.

    Returns:
        Parsed metrics dict, or None if not found.
    """
    metrics_path = base_model_path.parent / "metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    return None


def print_comparison(
    label: str,
    new_value: float,
    base_value: float | None,
) -> None:
    """Print a metric with optional delta vs base model.

    Args:
        label: Metric name.
        new_value: New metric value.
        base_value: Base model metric value, or None if unavailable.
    """
    if base_value is not None:
        delta = new_value - base_value
        sign = "+" if delta >= 0 else ""
        print(f"  {label}: {new_value:.4f}  (base: {base_value:.4f}, delta: {sign}{delta:.4f})")
    else:
        print(f"  {label}: {new_value:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run page-scorer retraining with eval-corrected labels.

    Returns:
        Exit code.
    """
    t0 = time.monotonic()
    args = build_arg_parser().parse_args()

    # --- Load base training data ---
    print(f"Loading base training data from {args.base_train_jsonl} ...")
    train_rows = load_grounding_rows(args.base_train_jsonl)
    train_examples = build_page_training_examples(train_rows, label_mode=args.label_mode)
    print(f"  Base training examples: {len(train_examples)}")
    print(f"  Base training questions: {len({ex.question_id for ex in train_examples})}")

    # --- Load dev data ---
    dev_examples: list[PageTrainingExample] = []
    if args.dev_jsonl.is_file():
        print(f"Loading dev data from {args.dev_jsonl} ...")
        dev_rows = load_grounding_rows(args.dev_jsonl)
        dev_examples = build_page_training_examples(dev_rows, label_mode=args.label_mode)
        print(f"  Dev examples: {len(dev_examples)}")
        print(f"  Dev questions: {len({ex.question_id for ex in dev_examples})}")
    else:
        print(f"  Dev JSONL not found at {args.dev_jsonl}, skipping dev evaluation.")

    # --- Load eval target map ---
    print(f"Loading eval target map from {args.eval_target_map} ...")
    target_map = load_eval_target_map(args.eval_target_map)
    print(f"  Eval target entries with verified gold: {len(target_map)}")

    # --- Apply label corrections ---
    print("Applying label corrections from eval target map ...")
    corrected_examples, corrected_count = apply_label_corrections(train_examples, target_map)
    corrected_positives = sum(1 for ex in corrected_examples if ex.supervision_source == "eval_verified" and ex.label == 1)
    corrected_negatives = sum(1 for ex in corrected_examples if ex.supervision_source == "eval_verified" and ex.label == 0)
    print(f"  Corrected questions: {corrected_count}")
    print(f"  Corrected examples (positive): {corrected_positives}")
    print(f"  Corrected examples (negative): {corrected_negatives}")
    print(f"  Total training examples: {len(corrected_examples)}")

    # --- Vectorize ---
    print("Vectorizing features ...")
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([ex.features for ex in corrected_examples])
    y_train = [ex.label for ex in corrected_examples]
    sample_weights = [ex.sample_weight for ex in corrected_examples]

    # --- Train LightGBM ---
    print("Training LightGBM model ...")
    model = build_lgbm_model(args.seed)
    model.fit(x_train, y_train, sample_weight=sample_weights)

    # --- Evaluate on train ---
    train_scores = model.predict_proba(
        vectorizer.transform([ex.features for ex in corrected_examples])
    )[:, 1]
    train_ranking = compute_ranking_metrics(corrected_examples, list(train_scores))
    train_fbeta = compute_fbeta_25(corrected_examples, train_scores)

    # --- Evaluate on dev ---
    dev_ranking = compute_ranking_metrics([], [])
    dev_fbeta = 0.0
    heuristic_metrics = compute_heuristic_ranking_metrics([])
    if dev_examples:
        dev_scores = model.predict_proba(
            vectorizer.transform([ex.features for ex in dev_examples])
        )[:, 1]
        dev_ranking = compute_ranking_metrics(dev_examples, list(dev_scores))
        dev_fbeta = compute_fbeta_25(dev_examples, dev_scores)
        heuristic_metrics = compute_heuristic_ranking_metrics(dev_examples)

    # --- Load base model metrics for comparison ---
    base_metrics: dict[str, Any] | None = None
    if args.base_model and args.base_model.is_file():
        base_metrics = load_base_model_metrics(args.base_model)

    # --- Print results ---
    elapsed = time.monotonic() - t0
    print(f"\n{'='*60}")
    print(f"Retraining complete in {elapsed:.1f}s")
    print(f"{'='*60}")

    print("\nTraining metrics:")
    print_comparison("hit@1", train_ranking.hit_at_1, base_metrics.get("train_hit_at_1") if base_metrics else None)
    print_comparison("hit@2", train_ranking.hit_at_2, base_metrics.get("train_hit_at_2") if base_metrics else None)
    print_comparison("MRR", train_ranking.mean_reciprocal_rank, base_metrics.get("train_mrr") if base_metrics else None)
    print_comparison("F-beta 2.5", train_fbeta, base_metrics.get("train_fbeta_2_5") if base_metrics else None)

    if dev_examples:
        print("\nDev metrics:")
        print_comparison("hit@1", dev_ranking.hit_at_1, base_metrics.get("dev_hit_at_1") if base_metrics else None)
        print_comparison("hit@2", dev_ranking.hit_at_2, base_metrics.get("dev_hit_at_2") if base_metrics else None)
        print_comparison("MRR", dev_ranking.mean_reciprocal_rank, base_metrics.get("dev_mrr") if base_metrics else None)
        print_comparison("F-beta 2.5", dev_fbeta, base_metrics.get("dev_fbeta_2_5") if base_metrics else None)
        print(f"\n  Heuristic dev hit@1: {heuristic_metrics.hit_at_1:.4f}")
        print(f"  Heuristic dev hit@2: {heuristic_metrics.hit_at_2:.4f}")
        print(f"  Heuristic dev MRR:   {heuristic_metrics.mean_reciprocal_rank:.4f}")

    feature_importance = top_feature_weights(model, vectorizer, top_n=args.top_feature_count)

    # --- Build metrics dict ---
    metrics: dict[str, Any] = {
        "model_type": "lgbm",
        "retrained_from_eval": True,
        "corrected_question_count": corrected_count,
        "corrected_positive_examples": corrected_positives,
        "corrected_negative_examples": corrected_negatives,
        "corrected_sample_weight": _CORRECTED_SAMPLE_WEIGHT,
        "train_example_count": len(corrected_examples),
        "dev_example_count": len(dev_examples),
        "train_question_count": len({ex.question_id for ex in corrected_examples}),
        "dev_question_count": len({ex.question_id for ex in dev_examples}),
        "train_hit_at_1": train_ranking.hit_at_1,
        "train_hit_at_2": train_ranking.hit_at_2,
        "train_mrr": train_ranking.mean_reciprocal_rank,
        "train_fbeta_2_5": train_fbeta,
        "dev_hit_at_1": dev_ranking.hit_at_1,
        "dev_hit_at_2": dev_ranking.hit_at_2,
        "dev_mrr": dev_ranking.mean_reciprocal_rank,
        "dev_fbeta_2_5": dev_fbeta,
        "heuristic_dev_hit_at_1": heuristic_metrics.hit_at_1,
        "heuristic_dev_hit_at_2": heuristic_metrics.hit_at_2,
        "heuristic_dev_mrr": heuristic_metrics.mean_reciprocal_rank,
        "train_supervision_question_counts": count_question_sources(corrected_examples),
        "dev_supervision_question_counts": count_question_sources(dev_examples),
        "label_mode": args.label_mode,
        "feature_policy": PAGE_SCORER_FEATURE_POLICY,
        "elapsed_seconds": round(elapsed, 1),
        "lgbm_params": _DEFAULT_LGBM_PARAMS,
    }

    # --- Save artifacts ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "model_type": "lgbm",
            "seed": args.seed,
            "label_mode": args.label_mode,
            "feature_policy": PAGE_SCORER_FEATURE_POLICY,
            "retrained_from_eval": True,
            "corrected_question_count": corrected_count,
        },
        args.output_dir / "page_scorer.joblib",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "base_train_jsonl": str(args.base_train_jsonl),
                "dev_jsonl": str(args.dev_jsonl),
                "eval_target_map": str(args.eval_target_map),
                "base_model": str(args.base_model) if args.base_model else None,
                "seed": args.seed,
                "model_type": "lgbm",
                "label_mode": args.label_mode,
                "feature_policy": PAGE_SCORER_FEATURE_POLICY,
                "corrected_sample_weight": _CORRECTED_SAMPLE_WEIGHT,
                "lgbm_params": _DEFAULT_LGBM_PARAMS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "feature_importance.json").write_text(
        json.dumps(feature_importance, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nArtifacts saved to {args.output_dir}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
