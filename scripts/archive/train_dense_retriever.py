"""Train the deterministic corpus-tuned dense retriever artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from shafi.ml.dense_retriever_training import DenseRetrieverConfig, DenseRetrieverTrainer


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="deterministic-hash-r1")
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    parser.add_argument("--min-token-weight", type=float, default=0.0)
    return parser


def main() -> int:
    """Run dense retriever training.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    trainer = DenseRetrieverTrainer()
    dataset = trainer.prepare_dataset(args.training_data)
    artifact_path = trainer.train(
        base_model=args.base_model,
        dataset=dataset,
        config=DenseRetrieverConfig(
            dimensions=args.dimensions,
            negative_weight=args.negative_weight,
            min_token_weight=args.min_token_weight,
        ),
        output_dir=args.output_dir,
    )
    print(artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
