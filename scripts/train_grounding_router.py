"""Train a lightweight offline router for grounding-sidecar scope decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from rag_challenge.ml.training_scaffold import build_router_dataset, deterministic_subset, load_grounding_rows


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for router training.

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
    return parser


def main() -> int:
    """Run router training and emit model artifacts.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    train_rows = deterministic_subset(load_grounding_rows(args.train_jsonl), limit=args.max_train_rows or None, seed=args.seed)
    dev_rows = deterministic_subset(load_grounding_rows(args.dev_jsonl), limit=args.max_dev_rows or None, seed=args.seed)

    train_ds = build_router_dataset(train_rows)
    dev_ds = build_router_dataset(dev_rows)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train = vectorizer.fit_transform(train_ds.texts)
    x_dev = vectorizer.transform(dev_ds.texts)

    scope_model = LogisticRegression(max_iter=400, class_weight="balanced", random_state=args.seed)
    scope_model.fit(x_train, train_ds.scope_targets)
    scope_pred = scope_model.predict(x_dev) if dev_ds.texts else []

    budget_model = LogisticRegression(max_iter=400, class_weight="balanced", random_state=args.seed)
    budget_model.fit(x_train, train_ds.page_budget_targets)
    budget_pred = budget_model.predict(x_dev) if dev_ds.texts else []

    mlb = MultiLabelBinarizer()
    y_roles_train = mlb.fit_transform(train_ds.role_targets)
    roles_model = OneVsRestClassifier(LogisticRegression(max_iter=400, class_weight="balanced", random_state=args.seed))
    roles_model.fit(x_train, y_roles_train)
    role_pred = roles_model.predict(x_dev) if dev_ds.texts else []
    y_roles_dev = mlb.transform(dev_ds.role_targets) if dev_ds.texts else []

    metrics = {
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "scope_accuracy": float(accuracy_score(dev_ds.scope_targets, scope_pred)) if dev_ds.texts else 0.0,
        "budget_accuracy": float(accuracy_score(dev_ds.page_budget_targets, budget_pred)) if dev_ds.texts else 0.0,
        "roles_micro_f1": float(f1_score(y_roles_dev, role_pred, average="micro", zero_division=0)) if dev_ds.texts else 0.0,
        "router_reference_note": "Targets are current heuristic scope labels; this scaffold measures imitation readiness, not benchmark superiority.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "scope_model": scope_model,
            "budget_model": budget_model,
            "roles_model": roles_model,
            "role_labels": list(mlb.classes_),
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
                "seed": args.seed,
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


if __name__ == "__main__":
    raise SystemExit(main())
