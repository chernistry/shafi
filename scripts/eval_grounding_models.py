"""Evaluate offline grounding router and page-scorer artifacts on exported rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from rag_challenge.ml.training_scaffold import (
    build_page_training_examples,
    build_router_dataset,
    deterministic_subset,
    group_page_examples,
    load_grounding_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for offline model evaluation.

    Returns:
        Configured argparse parser.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-jsonl", type=Path, default=repo_root / "data" / "derived" / "grounding_ml" / "v1" / "dev.jsonl")
    parser.add_argument("--router-model", type=Path, required=True)
    parser.add_argument("--page-scorer-model", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=610)
    parser.add_argument("--max-dev-rows", type=int, default=0)
    parser.add_argument("--label-mode", choices=["reviewed_only", "soft_and_reviewed", "all"], default="all")
    return parser


def main() -> int:
    """Run offline model evaluation.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    rows = deterministic_subset(load_grounding_rows(args.dev_jsonl), limit=args.max_dev_rows or None, seed=args.seed)

    router_bundle = joblib.load(args.router_model)
    router_ds = build_router_dataset(rows)
    x_dev = router_bundle["vectorizer"].transform(router_ds.texts)
    scope_pred = router_bundle["scope_model"].predict(x_dev)
    budget_pred = router_bundle["budget_model"].predict(x_dev)
    budget_targets = [str(value) for value in router_ds.page_budget_targets]
    mlb = MultiLabelBinarizer(classes=router_bundle["role_labels"])
    y_roles = mlb.fit_transform(router_ds.role_targets)
    role_pred = _predict_role_matrix(router_bundle["roles_model"], x_dev)

    page_bundle = joblib.load(args.page_scorer_model)
    page_examples = build_page_training_examples(rows, label_mode=args.label_mode)
    page_scores = (
        page_bundle["model"].predict_proba(
            page_bundle["vectorizer"].transform([example.features for example in page_examples])
        )[:, 1]
        if page_examples
        else []
    )

    summary = {
        "router": {
            "scope_accuracy": float(accuracy_score(router_ds.scope_targets, scope_pred)) if router_ds.texts else 0.0,
            "budget_accuracy": float(accuracy_score(budget_targets, budget_pred)) if router_ds.texts else 0.0,
            "roles_micro_f1": float(f1_score(y_roles, role_pred, average="micro", zero_division=0)) if router_ds.texts else 0.0,
            "heuristic_reference_accuracy": 1.0 if router_ds.texts else 0.0,
        },
        "page_scorer": {
            "trained_hit_at_1": _group_hit_rate(page_examples, page_scores),
            "heuristic_hit_at_1": _heuristic_hit_rate(page_examples),
            "question_count": len(group_page_examples(page_examples)),
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output_path)
    print(json.dumps(summary, indent=2))
    return 0


def _group_hit_rate(examples: list, scores) -> float:
    """Compute grouped hit@1 from predicted page scores."""
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
    """Compute heuristic hit@1 from exported source markers."""
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

def _predict_role_matrix(roles_model: object, x_dev) -> list[list[int]]:
    """Predict a multi-label role matrix from either router artifact shape.

    Args:
        roles_model: Stored role model artifact.
        x_dev: Dev feature matrix.

    Returns:
        Role prediction matrix.
    """
    if hasattr(roles_model, "predict"):
        predictions = roles_model.predict(x_dev)
        return predictions.tolist() if hasattr(predictions, "tolist") else predictions
    if isinstance(roles_model, dict):
        estimators = roles_model.get("estimators")
        if isinstance(estimators, list):
            prediction_columns = _predict_columns(estimators, x_dev)
            return [list(values) for values in zip(*prediction_columns, strict=False)] if prediction_columns else []
    raise TypeError(f"Unsupported roles model artifact: {type(roles_model)!r}")


def _predict_columns(estimators: Sequence[object], x_dev) -> list[list[int]]:
    """Predict one binary column per estimator.

    Args:
        estimators: Per-role estimators.
        x_dev: Dev feature matrix.

    Returns:
        Per-role prediction columns.
    """
    columns: list[list[int]] = []
    for estimator in estimators:
        if not hasattr(estimator, "predict"):
            raise TypeError(f"Unsupported role estimator: {type(estimator)!r}")
        prediction = estimator.predict(x_dev)
        columns.append(prediction.tolist() if hasattr(prediction, "tolist") else list(prediction))
    return columns


if __name__ == "__main__":
    raise SystemExit(main())
