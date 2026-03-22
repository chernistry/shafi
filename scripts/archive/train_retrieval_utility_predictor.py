"""Train and benchmark the retrieval-utility predictor offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_challenge.ml.utility_predictor import (
    UtilityPredictorTrainer,
    build_feature_importance,
    load_raw_results,
    load_reviewed_gold,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for retrieval-utility training.

    Returns:
        Configured argument parser.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-results",
        type=Path,
        default=repo_root
        / ".sdd"
        / "researches"
        / "639_grounding_resume_after_devops_baseline_r1_2026-03-19"
        / "raw_results_reviewed_public100_sidecar_current.json",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=repo_root / ".sdd" / "golden" / "reviewed" / "reviewed_all_100.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> int:
    """Train the predictor and emit metrics/artifacts.

    Returns:
        Process exit code.
    """
    args = build_arg_parser().parse_args()
    trainer = UtilityPredictorTrainer()
    reviewed_gold = load_reviewed_gold(args.gold)
    raw_results = load_raw_results(args.raw_results)
    examples = trainer.build_training_examples(raw_results, reviewed_gold)
    metrics = trainer.cross_validate(examples, folds=args.folds)
    fitted = trainer.fit(examples, threshold=args.threshold)
    artifact_path = trainer.save(fitted, args.output_dir)
    feature_importance = build_feature_importance(fitted)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "raw_results": str(args.raw_results),
                "gold": str(args.gold),
                "folds": args.folds,
                "threshold": args.threshold,
                "sample_count": len(examples),
                "positive_count": metrics.positive_count,
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
    print(artifact_path)
    print(json.dumps(metrics.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
