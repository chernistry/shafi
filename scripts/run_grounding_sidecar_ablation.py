"""Run an offline ablation for heuristic and trained grounding-sidecar lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from rag_challenge.ml.ablation import compute_selected_page_metrics, filter_rows_by_question_ids
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
    parser.add_argument(
        "--router-model",
        type=Path,
        default=repo_root
        / ".sdd"
        / "researches"
        / "622_reviewed_aware_router_and_page_scorer_retune_r1_2026-03-19"
        / "artifacts"
        / "models"
        / "grounding_router"
        / "v2_reviewed_internal_only"
        / "router.joblib",
    )
    parser.add_argument(
        "--page-scorer-model",
        type=Path,
        default=repo_root
        / ".sdd"
        / "researches"
        / "622_reviewed_aware_router_and_page_scorer_retune_r1_2026-03-19"
        / "artifacts"
        / "models"
        / "page_scorer"
        / "v2_reviewed_high_only"
        / "page_scorer.joblib",
    )
    parser.add_argument(
        "--hard-slice-json",
        type=Path,
        default=repo_root / ".sdd" / "golden" / "reviewed" / "reviewed_high_confidence_81.json",
    )
    parser.add_argument(
        "--soft-slice-json",
        type=Path,
        default=repo_root / ".sdd" / "golden" / "reviewed" / "reviewed_all_100.json",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=610)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--hard-label-mode",
        choices=["reviewed_high_confidence", "reviewed_weighted", "reviewed_only", "soft_and_reviewed", "all"],
        default="reviewed_high_confidence",
    )
    parser.add_argument(
        "--soft-label-mode",
        choices=["reviewed_high_confidence", "reviewed_weighted", "reviewed_only", "soft_and_reviewed", "all"],
        default="reviewed_weighted",
    )
    return parser


def main() -> int:
    """Run the offline ablation and write a JSON summary.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    rows = deterministic_subset(
        _load_combined_rows(train_jsonl=args.train_jsonl, dev_jsonl=args.dev_jsonl),
        limit=args.max_rows or None,
        seed=args.seed,
    )
    hard_slice_ids = _load_slice_question_ids(args.hard_slice_json)
    soft_slice_ids = _load_slice_question_ids(args.soft_slice_json)
    hard_rows = filter_rows_by_question_ids(rows, question_ids=hard_slice_ids)
    soft_rows = filter_rows_by_question_ids(rows, question_ids=soft_slice_ids)

    router_bundle = joblib.load(args.router_model)
    router_metrics, budget_predictions = _evaluate_router(rows, router_bundle)

    page_bundle = joblib.load(args.page_scorer_model)
    candidate_examples = build_page_training_examples(rows, label_mode="all")
    candidate_groups = group_page_examples(candidate_examples)
    hard_examples = build_page_training_examples(hard_rows, label_mode=args.hard_label_mode)
    soft_examples = build_page_training_examples(soft_rows, label_mode=args.soft_label_mode)
    hard_grouped_examples = group_page_examples(hard_examples)
    soft_grouped_examples = group_page_examples(soft_examples)
    page_scores = (
        page_bundle["model"].predict_proba(
            page_bundle["vectorizer"].transform([example.features for example in candidate_examples])
        )[:, 1]
        if candidate_examples
        else []
    )
    score_lookup = build_score_lookup(candidate_examples, page_scores)

    legacy_selected = {row.question_id: list(row.legacy_selected_pages) for row in rows}
    sidecar_selected = {
        row.question_id: list(row.sidecar_selected_pages or row.legacy_selected_pages)
        for row in rows
    }
    trained_router_only = _select_with_budget(
        rows,
        grouped_examples=candidate_groups,
        budgets=budget_predictions,
        selector=lambda group: rank_group_with_heuristic(group),
    )
    trained_router_plus_page_scorer = _select_with_budget(
        rows,
        grouped_examples=candidate_groups,
        budgets=budget_predictions,
        selector=lambda group: rank_group_with_scores(group, score_lookup),
    )

    lane_metrics = {
        "hard_gate": _compute_slice_lane_metrics(
            rows=hard_rows,
            grouped_examples=hard_grouped_examples,
            legacy_selected=legacy_selected,
            sidecar_selected=sidecar_selected,
            trained_router_only=trained_router_only,
            trained_router_plus_page_scorer=trained_router_plus_page_scorer,
        ),
        "soft_diagnostic": _compute_slice_lane_metrics(
            rows=soft_rows,
            grouped_examples=soft_grouped_examples,
            legacy_selected=legacy_selected,
            sidecar_selected=sidecar_selected,
            trained_router_only=trained_router_only,
            trained_router_plus_page_scorer=trained_router_plus_page_scorer,
        ),
    }

    output = {
        "dataset": {
            "row_count": len(rows),
            "hard_slice_row_count": len(hard_rows),
            "soft_slice_row_count": len(soft_rows),
            "hard_label_mode": args.hard_label_mode,
            "soft_label_mode": args.soft_label_mode,
            "candidate_question_count": len(candidate_groups),
            "hard_supervised_question_count": len(hard_grouped_examples),
            "soft_supervised_question_count": len(soft_grouped_examples),
            "hard_slice_question_ids_path": str(args.hard_slice_json),
            "soft_slice_question_ids_path": str(args.soft_slice_json),
            "compare_full_case_question_count": sum(1 for row in rows if row.scope_mode in {"compare_pair", "full_case_files"}),
            "negative_unanswerable_question_count": sum(1 for row in rows if row.scope_mode == "negative_unanswerable"),
            "answer_drift": 0,
            "answer_drift_note": "offline ablation only; no answer-path mutation occurred",
        },
        "router": router_metrics,
        "lanes": lane_metrics,
        "pairwise": {
            "hard_gate": {
                "trained_router_only_vs_heuristic_sidecar": _pairwise_delta(
                    baseline=lane_metrics["hard_gate"]["heuristic_sidecar"],
                    candidate=lane_metrics["hard_gate"]["trained_router_only"],
                ),
                "trained_router_plus_page_scorer_vs_heuristic_sidecar": _pairwise_delta(
                    baseline=lane_metrics["hard_gate"]["heuristic_sidecar"],
                    candidate=lane_metrics["hard_gate"]["trained_router_plus_page_scorer"],
                ),
            },
            "soft_diagnostic": {
                "trained_router_only_vs_heuristic_sidecar": _pairwise_delta(
                    baseline=lane_metrics["soft_diagnostic"]["heuristic_sidecar"],
                    candidate=lane_metrics["soft_diagnostic"]["trained_router_only"],
                ),
                "trained_router_plus_page_scorer_vs_heuristic_sidecar": _pairwise_delta(
                    baseline=lane_metrics["soft_diagnostic"]["heuristic_sidecar"],
                    candidate=lane_metrics["soft_diagnostic"]["trained_router_plus_page_scorer"],
                ),
            },
        },
        "verdict": _build_verdict(lane_metrics),
        "note": "This ablation measures reviewed offline page-selection truth only; no answer-path mutation occurred.",
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output_path)
    print(json.dumps(output, indent=2))
    return 0


def _compute_slice_lane_metrics(
    *,
    rows: Sequence[GroundingMlRow],
    grouped_examples: dict[str, list[PageTrainingExample]],
    legacy_selected: dict[str, list[str]],
    sidecar_selected: dict[str, list[str]],
    trained_router_only: dict[str, list[str]],
    trained_router_plus_page_scorer: dict[str, list[str]],
) -> dict[str, dict[str, float | int]]:
    """Compute all lane metrics for one reviewed slice.

    Args:
        rows: Slice rows.
        grouped_examples: Supervised grouped examples for the slice.
        legacy_selected: Legacy selected pages keyed by question ID.
        sidecar_selected: Heuristic sidecar selected pages keyed by question ID.
        trained_router_only: Router-only selected pages keyed by question ID.
        trained_router_plus_page_scorer: Router+page scorer selected pages keyed by question ID.

    Returns:
        Lane metrics keyed by lane name.
    """
    return {
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


def _build_verdict(lane_metrics: dict[str, dict[str, dict[str, float | int]]]) -> dict[str, object]:
    """Build a simple freeze/promote verdict from reviewed slice metrics.

    Args:
        lane_metrics: Slice metrics keyed by slice and lane.

    Returns:
        JSON-friendly verdict payload.
    """
    hard_heuristic = lane_metrics["hard_gate"]["heuristic_sidecar"]
    soft_heuristic = lane_metrics["soft_diagnostic"]["heuristic_sidecar"]
    candidate_names = ["trained_router_only", "trained_router_plus_page_scorer"]
    best_candidate = max(
        candidate_names,
        key=lambda name: float(lane_metrics["hard_gate"][name]["weighted_selected_hit_rate"]),
    )
    best_hard = lane_metrics["hard_gate"][best_candidate]
    best_soft = lane_metrics["soft_diagnostic"][best_candidate]
    hard_gain = float(best_hard["weighted_selected_hit_rate"]) - float(hard_heuristic["weighted_selected_hit_rate"])
    soft_gain = float(best_soft["overall_selected_hit_rate"]) - float(soft_heuristic["overall_selected_hit_rate"])
    page_delta = float(best_hard["average_selected_pages"]) - float(hard_heuristic["average_selected_pages"])
    keep_frozen = hard_gain < 0.0 or soft_gain < 0.0 or (page_delta > 0.0 and hard_gain <= 0.0)
    return {
        "best_trained_lane": best_candidate,
        "hard_gate_weighted_gain_vs_heuristic": hard_gain,
        "soft_overall_gain_vs_heuristic": soft_gain,
        "hard_average_selected_pages_delta_vs_heuristic": page_delta,
        "decision": "freeze_trained_lane" if keep_frozen else "candidate_survives_offline_gate",
    }


def _load_combined_rows(*, train_jsonl: Path, dev_jsonl: Path) -> list[GroundingMlRow]:
    """Load and de-duplicate reviewed export rows across train and dev files.

    Args:
        train_jsonl: Training JSONL path.
        dev_jsonl: Development JSONL path.

    Returns:
        Stable de-duplicated row list.
    """
    combined: dict[str, GroundingMlRow] = {}
    for path in (train_jsonl, dev_jsonl):
        for row in load_grounding_rows(path):
            combined[row.question_id] = row
    return [combined[question_id] for question_id in sorted(combined)]


def _load_slice_question_ids(path: Path) -> set[str]:
    """Load question IDs from a reviewed slice JSON file.

    Args:
        path: Reviewed slice path.

    Returns:
        Question ID set.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("cases", [])
    question_ids = set()
    for item in items:
        if isinstance(item, dict):
            question_id = item.get("question_id")
            if isinstance(question_id, str) and question_id.strip():
                question_ids.add(question_id.strip())
    return question_ids


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
    trained_role_labels = [str(label) for label in router_bundle["role_labels"]]
    unseen_role_labels = _unseen_role_labels(
        role_targets=router_ds.role_targets,
        trained_role_labels=trained_role_labels,
    )
    eval_labels = [*trained_role_labels, *unseen_role_labels]
    mlb = MultiLabelBinarizer(classes=eval_labels)
    mlb.fit([[]])
    y_roles = mlb.transform(router_ds.role_targets)
    role_predictions = _predict_role_matrix(
        router_bundle["roles_model"],
        x_dev,
        row_count=len(router_ds.texts),
        unseen_label_count=len(unseen_role_labels),
    )
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
            "held_out_unseen_role_labels": unseen_role_labels,
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


def _predict_role_matrix(
    roles_model: object,
    x_dev,
    *,
    row_count: int,
    unseen_label_count: int,
) -> list[list[int]]:
    """Predict a multi-label role matrix from either router artifact shape.

    Args:
        roles_model: Stored role model artifact.
        x_dev: Dev feature matrix.
        row_count: Number of evaluation rows.
        unseen_label_count: Count of evaluation-only role labels.

    Returns:
        Role prediction matrix.
    """
    if hasattr(roles_model, "predict"):
        predictions = roles_model.predict(x_dev)
        rows = predictions.tolist() if hasattr(predictions, "tolist") else predictions
        return _pad_role_prediction_rows(
            rows=rows,
            row_count=row_count,
            unseen_label_count=unseen_label_count,
        )
    if isinstance(roles_model, dict):
        estimators = roles_model.get("estimators")
        if isinstance(estimators, list):
            prediction_columns = _predict_columns(estimators, x_dev)
            rows = [list(values) for values in zip(*prediction_columns, strict=False)] if prediction_columns else []
            return _pad_role_prediction_rows(
                rows=rows,
                row_count=row_count,
                unseen_label_count=unseen_label_count,
            )
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


def _pad_role_prediction_rows(
    *,
    rows: Sequence[Sequence[int]],
    row_count: int,
    unseen_label_count: int,
) -> list[list[int]]:
    """Pad predicted role rows to cover evaluation-only unseen labels.

    Args:
        rows: Predicted role rows aligned to trained role labels.
        row_count: Number of evaluation rows.
        unseen_label_count: Count of evaluation-only labels.

    Returns:
        Dense prediction matrix aligned to the evaluation label order.
    """
    base_rows = [list(row) for row in rows] if rows else [[] for _ in range(row_count)]
    if unseen_label_count <= 0:
        return base_rows
    return [row + [0] * unseen_label_count for row in base_rows]


def _unseen_role_labels(
    *,
    role_targets: Sequence[Sequence[str]],
    trained_role_labels: Sequence[str],
) -> list[str]:
    """Return sorted evaluation role labels missing from trained artifacts.

    Args:
        role_targets: Evaluation role-target lists.
        trained_role_labels: Labels seen by the trained router role head.

    Returns:
        Sorted list of evaluation-only role labels.
    """
    trained_label_set = set(trained_role_labels)
    unseen = {
        role
        for targets in role_targets
        for role in targets
        if role not in trained_label_set
    }
    return sorted(unseen)


if __name__ == "__main__":
    raise SystemExit(main())
