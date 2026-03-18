"""Train a lightweight offline page scorer from grounding export candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from rag_challenge.ml.training_scaffold import (
    build_page_training_examples,
    deterministic_subset,
    group_page_examples,
    load_grounding_rows,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for page-scorer training.

    Returns:
        Configured argparse parser.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=repo_root / "data" / "derived" / "grounding_ml" / "v1" / "train.jsonl")
    parser.add_argument("--dev-jsonl", type=Path, default=repo_root / "data" / "derived" / "grounding_ml" / "v1" / "dev.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=610)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-dev-rows", type=int, default=0)
    parser.add_argument("--label-mode", choices=["reviewed_only", "soft_and_reviewed", "all"], default="all")
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

    model = LogisticRegression(max_iter=400, class_weight="balanced", random_state=args.seed)
    model.fit(x_train, y_train, sample_weight=sample_weight)

    train_hit_at_1 = _group_hit_rate(train_examples, model.predict_proba(vectorizer.transform([e.features for e in train_examples]))[:, 1])
    dev_scores = model.predict_proba(vectorizer.transform([e.features for e in dev_examples]))[:, 1] if dev_examples else []
    dev_hit_at_1 = _group_hit_rate(dev_examples, dev_scores)
    heuristic_hit_at_1 = _heuristic_hit_rate(dev_examples)

    metrics = {
        "train_example_count": len(train_examples),
        "dev_example_count": len(dev_examples),
        "train_question_count": len({example.question_id for example in train_examples}),
        "dev_question_count": len({example.question_id for example in dev_examples}),
        "train_hit_at_1": train_hit_at_1,
        "dev_hit_at_1": dev_hit_at_1,
        "heuristic_dev_hit_at_1": heuristic_hit_at_1,
        "label_mode": args.label_mode,
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
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)
    print(json.dumps(metrics, indent=2))
    return 0


def _group_hit_rate(examples: list, scores) -> float:
    """Compute grouped hit@1 for page examples."""
    if not examples:
        return 0.0
    grouped = group_page_examples(examples)
    score_map = {f"{example.question_id}:{example.page_id}": float(score) for example, score in zip(examples, scores, strict=False)}
    hits = 0
    total = 0
    for question_id, group in grouped.items():
        total += 1
        best = max(group, key=lambda example: score_map.get(f"{question_id}:{example.page_id}", 0.0))
        if best.label == 1:
            hits += 1
    return hits / total if total else 0.0


def _heuristic_hit_rate(examples: list) -> float:
    """Compute heuristic hit@1 using sidecar/legacy-used source markers."""
    if not examples:
        return 0.0
    grouped = group_page_examples(examples)
    hits = 0
    total = 0
    for group in grouped.values():
        total += 1
        preferred = next(
            (
                example
                for example in group
                if bool(example.features.get("from_sidecar_used")) or bool(example.features.get("from_legacy_used"))
            ),
            group[0],
        )
        if preferred.label == 1:
            hits += 1
    return hits / total if total else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
