"""Evaluate a dense retriever artifact against retrieval gold cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_challenge.ml.dense_retriever_training import DenseRetrieverTrainer, load_artifact, load_eval_cases


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--compare-baseline", action="store_true")
    return parser


def main() -> int:
    """Run dense retriever evaluation.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    trainer = DenseRetrieverTrainer()
    artifact = load_artifact(args.model)
    eval_cases = load_eval_cases(args.eval_jsonl)
    metrics = trainer.evaluate_against_gold(artifact=artifact, eval_cases=eval_cases)
    payload: dict[str, object] = {"metrics": as_json(metrics)}
    if args.compare_baseline:
        payload["baseline_delta"] = trainer.compare_to_baseline(artifact=artifact, eval_cases=eval_cases)
    rendered = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


def as_json(metrics) -> dict[str, object]:
    """Convert metrics into a JSON-safe payload."""
    return {
        "mrr_at_10": metrics.mrr_at_10,
        "recall_at_10": metrics.recall_at_10,
        "recall_at_20": metrics.recall_at_20,
        "purity": metrics.purity,
        "per_family_metrics": metrics.per_family_metrics,
    }


if __name__ == "__main__":
    raise SystemExit(main())
