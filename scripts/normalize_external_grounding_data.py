"""Normalize external legal datasets into a shared grounding schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_challenge.ml.external_grounding_data import export_normalized_external_grounding_data


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for external-data normalization.

    Returns:
        Configured argument parser.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obliqa-root",
        type=Path,
        default=repo_root / "data" / "external" / "obliqa" / "raw" / "hf_snapshot",
    )
    parser.add_argument(
        "--cuad-root",
        type=Path,
        default=repo_root / "data" / "external" / "cuad" / "raw" / "hf_snapshot",
    )
    parser.add_argument(
        "--contractnli-root",
        type=Path,
        default=repo_root / "data" / "external" / "contractnli" / "raw" / "hf_snapshot",
    )
    parser.add_argument(
        "--ledgar-root",
        type=Path,
        default=repo_root / "data" / "external" / "ledgar" / "raw" / "hf_snapshot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "external" / "normalized" / "v1",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=0,
        help="Optional per-dataset cap. Zero means no cap.",
    )
    return parser


def main() -> int:
    """Run the external-data normalization export.

    Returns:
        Process exit code.
    """
    args = build_arg_parser().parse_args()
    manifest = export_normalized_external_grounding_data(
        obliqa_root=args.obliqa_root,
        cuad_root=args.cuad_root,
        contractnli_root=args.contractnli_root,
        ledgar_root=args.ledgar_root,
        output_dir=args.output_dir,
        max_rows_per_dataset=args.max_rows_per_dataset,
    )
    print(args.output_dir)
    print(json.dumps(manifest.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
