"""Run an offline ablation for heuristic and trained grounding-sidecar lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from rag_challenge.ml.ablation import compute_selected_page_metrics
from rag_challenge.ml.page_scorer_training import (
    build_score_lookup,
    rank_group_with_heuristic,
    rank_group_with_scores,
)
from rag_challenge.ml.training_scaffold import (
    build_page_training_examples,
    build_router_dataset,
    deterministic_subset,
    group_page_examples,
    load_grounding_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.ml.grounding_dataset import GroundingMlRow
    from rag_challenge.ml.training_scaffold import PageTrainingExample


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the offline ablation harness.

    Returns:
        Configured argparse parser.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-jsonl", type=Path, default=repo_root / "data" / "derived" / "grounding_ml" / "v1" / "dev.jsonl")
    parser.add_argument(
        "--router-model",
        type=Path,
        default=repo_root
        / ".sdd"
        / "researches"
        / "616_train_grounding_router_v1_2026-03-18"
        / "artifacts"
        / "models"
        / "grounding_router"
        / "v1_full_internal_only"
        / "router.joblib",
    )
    parser.add_argument(
        "--page-scorer-model",
        type=Path,
        default=repo_root
        / ".sdd"
        / "researches"
        / "617_train_page_scorer_v1_2026-03-19"
        / "artifacts"
        / "models"
        / "page_scorer"
        / "v1_full"
        / "page_scorer.joblib",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=610)
    parser.add_argument("--max-dev-rows", type=int, default=0)
    parser.add_argument("--label-mode", choices=["reviewed_only", "soft_and_reviewed", "all"], default="all")
    return parser


def main() -> int:
    """Run the offline ablation and write a JSON summary.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    rows = deterministic_subset(load_grounding_rows(args.dev_jsonl), limit=args.max_dev_rows or None, seed=args.seed)

    router_bundle = joblib.load(args.router_model)
    router_metrics, budget_predictions = _evaluate_router(rows, router_bundle)

    page_bundle = joblib.load(args.page_scorer_model)
    page_examples = build_page_training_examples(rows, label_mode=args.label_mode)
    grouped_examples = group_page_examples(page_examples)
    page_scores = (
        page_bundle["model"].predict_proba(
            page_bundle["vectorizer"].transform([example.features for example in page_examples])
        )[:, 1]
        if page_examples
        else []
    )
    score_lookup = build_score_lookup(page_examples, page_scores)

    legacy_selected = {row.question_id: list(row.legacy_selected_pages) for row in rows}
    sidecar_selected = {
        row.question_id: list(row.sidecar_selected_pages or row.legacy_selected_pages)
        for row in rows
    }
    trained_router_only = _select_with_budget(
        rows,
        grouped_examples=grouped_examples,
        budgets=budget_predictions,
        selector=lambda group: rank_group_with_heuristic(group),
    )
    trained_router_plus_page_scorer = _select_with_budget(
        rows,
        grouped_examples=grouped_examples,
        budgets=budget_predictions,
        selector=lambda group: rank_group_with_scores(group, score_lookup),
    )

    lane_metrics = {
        "repaired_legacy": compute_selected_page_metrics(
            rows,
            grouped_examples=grouped_examples,
            selected_pages_by_question=legacy_selected,
        ).to_dict(),
        "heuristic_sidecar": compute_selected_page_metrics(
            rows,
            grouped_examples=grouped_examples,
            selected_pages_by_question=sidecar_selected,
        ).to_dict(),
        "trained_router_only": compute_selected_page_metrics(
            rows,
            grouped_examples=grouped_examples,
            selected_pages_by_question=trained_router_only,
        ).to_dict(),
        "trained_router_plus_page_scorer": compute_selected_page_metrics(
            rows,
            grouped_examples=grouped_examples,
            selected_pages_by_question=trained_router_plus_page_scorer,
        ).to_dict(),
    }

    output = {
        "dataset": {
            "row_count": len(rows),
            "label_mode": args.label_mode,
            "page_supervised_question_count": len(grouped_examples),
            "soft_label_question_count": sum(
                1 for group in grouped_examples.values() if group and group[0].supervision_source == "soft_ai_gold"
            ),
            "reviewed_question_count": sum(
                1 for group in grouped_examples.values() if group and group[0].supervision_source == "reviewed"
            ),
            "compare_full_case_question_count": sum(
                1 for row in rows if row.scope_mode in {"compare_pair", "full_case_files"}
            ),
            "negative_unanswerable_question_count": sum(
                1 for row in rows if row.scope_mode == "negative_unanswerable"
            ),
            "answer_drift": 0,
            "answer_drift_note": "offline ablation only; no answer-path mutation occurred",
        },
        "router": router_metrics,
        "lanes": lane_metrics,
        "pairwise": {
            "trained_router_only_vs_heuristic_sidecar": _pairwise_delta(
                baseline=lane_metrics["heuristic_sidecar"],
                candidate=lane_metrics["trained_router_only"],
            ),
            "trained_router_plus_page_scorer_vs_heuristic_sidecar": _pairwise_delta(
                baseline=lane_metrics["heuristic_sidecar"],
                candidate=lane_metrics["trained_router_plus_page_scorer"],
            ),
        },
        "note": (
            "This ablation measures offline page-selection proxies only. "
            "Current AI-generated page gold is noisy, so these numbers are not submission-grade grounding truth."
        ),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output_path)
    print(json.dumps(output, indent=2))
    return 0


def _evaluate_router(rows: Sequence[GroundingMlRow], router_bundle: dict[str, object]) -> tuple[dict[str, float], dict[str, int]]:
    """Evaluate the trained router and return predicted budgets.

    Args:
        rows: Exported dev rows.
        router_bundle: Loaded router artifact.

    Returns:
        Tuple of router metric payload and predicted budgets by question ID.
    """
    router_ds = build_router_dataset(rows)
    x_dev = router_bundle["vectorizer"].transform(router_ds.texts)
    scope_predictions = router_bundle["scope_model"].predict(x_dev)
    budget_predictions = router_bundle["budget_model"].predict(x_dev)
    budget_targets = [str(value) for value in router_ds.page_budget_targets]
    mlb = MultiLabelBinarizer(classes=router_bundle["role_labels"])
    y_roles = mlb.fit_transform(router_ds.role_targets)
    role_predictions = _predict_role_matrix(router_bundle["roles_model"], x_dev)
    budget_map = {
        question_id: max(0, int(str(prediction)))
        for question_id, prediction in zip(router_ds.question_ids, budget_predictions, strict=False)
    }
    return (
        {
            "scope_accuracy": float(accuracy_score(router_ds.scope_targets, scope_predictions)) if router_ds.texts else 0.0,
            "budget_accuracy": float(accuracy_score(budget_targets, budget_predictions)) if router_ds.texts else 0.0,
            "roles_micro_f1": float(f1_score(y_roles, role_predictions, average="micro", zero_division=0))
            if router_ds.texts
            else 0.0,
            "heuristic_reference_accuracy": 1.0 if router_ds.texts else 0.0,
        },
        budget_map,
    )


def _select_with_budget(
    rows: Sequence[GroundingMlRow],
    *,
    grouped_examples: dict[str, list[PageTrainingExample]],
    budgets: dict[str, int],
    selector,
) -> dict[str, list[str]]:
    """Select top pages per question using a budget and ranking function.

    Args:
        rows: Exported rows.
        grouped_examples: Page examples grouped by question ID.
        budgets: Predicted page budgets by question ID.
        selector: Ranking function returning grouped pages best-first.

    Returns:
        Selected page IDs keyed by question ID.
    """
    selected: dict[str, list[str]] = {}
    for row in rows:
        budget = max(0, budgets.get(row.question_id, 1))
        group = grouped_examples.get(row.question_id, [])
        if budget == 0 or not group:
            selected[row.question_id] = []
            continue
        ranked = selector(group)
        selected[row.question_id] = [example.page_id for example in ranked[:budget]]
    return selected


def _pairwise_delta(
    *,
    baseline: dict[str, float | int],
    candidate: dict[str, float | int],
) -> dict[str, float]:
    """Compute candidate-minus-baseline deltas for shared numeric metrics.

    Args:
        baseline: Baseline lane metrics.
        candidate: Candidate lane metrics.

    Returns:
        Numeric metric deltas.
    """
    deltas: dict[str, float] = {}
    for key, baseline_value in baseline.items():
        candidate_value = candidate.get(key)
        if isinstance(baseline_value, bool) or isinstance(candidate_value, bool):
            continue
        if isinstance(baseline_value, (int, float)) and isinstance(candidate_value, (int, float)):
            deltas[key] = float(candidate_value) - float(baseline_value)
    return deltas


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
