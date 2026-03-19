"""Train a lightweight offline page scorer from grounding export candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from rag_challenge.ml.page_scorer_training import (
    compute_heuristic_ranking_metrics,
    compute_ranking_metrics,
    count_question_sources,
    top_feature_weights,
)
from rag_challenge.ml.training_scaffold import (
    build_page_training_examples,
    deterministic_subset,
    load_grounding_rows,
)


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
    return parser


def main() -> int:
    """Run page-scorer training and emit model artifacts.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    train_rows = deterministic_subset(load_grounding_rows(args.train_jsonl), limit=args.max_train_rows or None, seed=args.seed)
    dev_rows = deterministic_subset(load_grounding_rows(args.dev_jsonl), limit=args.max_dev_rows or None, seed=args.seed)
    train_examples = build_page_training_examples(train_rows, label_mode=args.label_mode)
    dev_examples = build_page_training_examples(dev_rows, label_mode=args.label_mode)

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([example.features for example in train_examples])
    y_train = [example.label for example in train_examples]
    sample_weight = [example.sample_weight for example in train_examples]

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
        solver="liblinear",
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)

    train_scores = model.predict_proba(vectorizer.transform([example.features for example in train_examples]))[:, 1]
    dev_scores = model.predict_proba(vectorizer.transform([example.features for example in dev_examples]))[:, 1] if dev_examples else []
    train_metrics = compute_ranking_metrics(train_examples, train_scores)
    dev_metrics = compute_ranking_metrics(dev_examples, dev_scores)
    heuristic_metrics = compute_heuristic_ranking_metrics(dev_examples)
    feature_importance = top_feature_weights(model, vectorizer, top_n=args.top_feature_count)

    metrics = {
        "train_example_count": len(train_examples),
        "dev_example_count": len(dev_examples),
        "train_question_count": len({example.question_id for example in train_examples}),
        "dev_question_count": len({example.question_id for example in dev_examples}),
        "train_hit_at_1": train_metrics.hit_at_1,
        "train_hit_at_2": train_metrics.hit_at_2,
        "train_mrr": train_metrics.mean_reciprocal_rank,
        "dev_hit_at_1": dev_metrics.hit_at_1,
        "dev_hit_at_2": dev_metrics.hit_at_2,
        "dev_mrr": dev_metrics.mean_reciprocal_rank,
        "heuristic_dev_hit_at_1": heuristic_metrics.hit_at_1,
        "heuristic_dev_hit_at_2": heuristic_metrics.hit_at_2,
        "heuristic_dev_mrr": heuristic_metrics.mean_reciprocal_rank,
        "train_supervision_question_counts": count_question_sources(train_examples),
        "dev_supervision_question_counts": count_question_sources(dev_examples),
        "label_mode": args.label_mode,
        "label_quality_note": _build_label_quality_note(args.label_mode),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "seed": args.seed,
            "label_mode": args.label_mode,
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
                "label_mode": args.label_mode,
                "max_train_rows": args.max_train_rows,
                "max_dev_rows": args.max_dev_rows,
                "top_feature_count": args.top_feature_count,
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
