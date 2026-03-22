"""Evaluate the deterministic compact set selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_challenge.ml.set_selector_training import SetSelectorTrainer, load_set_utility_labels


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--set-utility-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main() -> int:
    """Run set-selector evaluation.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    trainer = SetSelectorTrainer()
    labels = load_set_utility_labels(args.set_utility_data)
    metrics = trainer.evaluate_selector(artifact_path=args.model, eval_labels=labels)
    payload = {
        "page_precision": metrics.page_precision,
        "page_recall": metrics.page_recall,
        "set_utility_rate": metrics.set_utility_rate,
        "avg_set_size": metrics.avg_set_size,
        "downstream_correctness": metrics.downstream_correctness,
    }
    rendered = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
