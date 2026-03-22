"""Train the deterministic corpus-tuned reranker and set-selector artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_challenge.ml.grounding_dataset import GroundingMlRow
from rag_challenge.ml.set_selector_training import (
    SetSelectorConfig,
    SetSelectorTrainer,
    load_pairwise_triples,
    write_set_utility_jsonl,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="deterministic-cross-encoder-r1")
    parser.add_argument("--pairwise-data", type=Path, required=True)
    parser.add_argument("--grounding-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--set-utility-out", type=Path, default=None)
    parser.add_argument("--min-set-size", type=int, default=3)
    parser.add_argument("--max-set-size", type=int, default=8)
    return parser


def main() -> int:
    """Run set-selector training.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    trainer = SetSelectorTrainer()
    triples = load_pairwise_triples(args.pairwise_data)
    rows = [
        GroundingMlRow.model_validate_json(line)
        for line in args.grounding_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = SetSelectorConfig(min_set_size=args.min_set_size, max_set_size=args.max_set_size)
    pairwise_data = trainer.prepare_pairwise_data(triples)
    set_utility_data = trainer.prepare_set_utility_data(rows)
    cross_encoder_path = trainer.train_cross_encoder(
        base_model=args.base_model,
        pairwise_data=pairwise_data,
        config=config,
        output_dir=args.output_dir,
    )
    selector_path = trainer.train_set_selector(
        cross_encoder_path=cross_encoder_path,
        set_utility_data=set_utility_data,
        config=config,
        output_dir=args.output_dir,
    )
    if args.set_utility_out is not None:
        args.set_utility_out.parent.mkdir(parents=True, exist_ok=True)
        write_set_utility_jsonl(args.set_utility_out, set_utility_data)
    print(selector_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
