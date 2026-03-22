"""Generate an offline adversarial grounding corpus for future training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shafi.ml.adversarial_grounding_rows import (
    build_adversarial_grounding_rows,
    build_adversarial_manifest,
    write_adversarial_grounding_rows,
)
from shafi.ml.training_scaffold import load_grounding_rows, split_rows_by_holdout_doc_family


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True, help="Grounding export JSONL input")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--legal-ner-dir",
        type=Path,
        default=Path("/Users/sasha/IdeaProjects/personal_projects/shafi/data/external/Legal NER Dataset"),
        help="Optional Legal NER dataset directory for weak taxonomy priors",
    )
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1062)
    return parser


def main() -> int:
    """Run the export CLI."""

    args = build_arg_parser().parse_args()
    rows_by_id = {}
    for path in args.input_jsonl:
        for row in load_grounding_rows(path):
            rows_by_id[row.question_id] = row
    combined_rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    train_source_rows, dev_source_rows = split_rows_by_holdout_doc_family(
        combined_rows,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    legal_ner_path = args.legal_ner_dir if args.legal_ner_dir.exists() else None
    train_rows = build_adversarial_grounding_rows(train_source_rows, legal_ner_taxonomy_path=legal_ner_path)
    dev_rows = build_adversarial_grounding_rows(dev_source_rows, legal_ner_taxonomy_path=legal_ner_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_adversarial_grounding_rows(args.output_dir / "train.jsonl", train_rows)
    write_adversarial_grounding_rows(args.output_dir / "dev.jsonl", dev_rows)

    manifest = build_adversarial_manifest(
        train_rows=train_rows,
        dev_rows=dev_rows,
        legal_ner_taxonomy_path=legal_ner_path,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "feature_inventory.md").write_text(
        _build_feature_inventory(train_rows=train_rows, dev_rows=dev_rows),
        encoding="utf-8",
    )
    print(args.output_dir)
    print(manifest.model_dump_json(indent=2))
    return 0


def _build_feature_inventory(*, train_rows: list[object], dev_rows: list[object]) -> str:
    """Render a short markdown summary for the export."""

    total_rows = len(train_rows) + len(dev_rows)
    lines = [
        "# Adversarial Grounding Corpus",
        "",
        f"- rows: `{total_rows}`",
        f"- train rows: `{len(train_rows)}`",
        f"- dev rows: `{len(dev_rows)}`",
        "- negative families: same-doc nearby pages, same-family wrong-law, title/authority confusers, contradiction/unsupported, compact set-level negatives",
        "- split policy: holdout by `holdout_doc_family_key`",
        "- Legal NER labels are used only as optional weak taxonomy priors",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
