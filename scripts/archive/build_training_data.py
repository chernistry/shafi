"""Build retrieval/reranker training triples from offline supervision sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_challenge.ml.grounding_dataset import GroundingMlRow
from rag_challenge.ml.hard_negative_miner import HardNegativeMiner
from rag_challenge.ml.synthetic_qa_factory import (
    SyntheticQAExample,
    load_bridge_fact_records,
    load_corpus_registry,
)
from rag_challenge.ml.teacher_labels import (
    TeacherLabelBuilder,
    build_page_texts_from_registry,
    write_pairwise_labels_jsonl,
    write_training_triples_jsonl,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounding-jsonl", type=Path, required=True)
    parser.add_argument("--synthetic-jsonl", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--bridge-facts", type=Path, default=None)
    parser.add_argument("--out-triples", type=Path, required=True)
    parser.add_argument("--out-pairwise", type=Path, required=True)
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument("--denoise", action="store_true")
    return parser


def main() -> int:
    """Run the teacher-label export.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    grounding_rows = [
        GroundingMlRow.model_validate_json(line)
        for line in args.grounding_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    synthetic_examples = [
        SyntheticQAExample.model_validate_json(line)
        for line in args.synthetic_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry = load_corpus_registry(args.registry)
    bridge_facts = load_bridge_fact_records(args.bridge_facts) if args.bridge_facts is not None else []

    page_texts, page_doc_ids, aliases_by_doc_id = build_page_texts_from_registry(registry)
    miner = HardNegativeMiner(page_texts=page_texts, page_doc_ids=page_doc_ids, aliases_by_doc_id=aliases_by_doc_id)
    builder = TeacherLabelBuilder(miner=miner, page_texts=page_texts, page_doc_ids=page_doc_ids)
    triples = builder.combine_and_deduplicate(
        [
            builder.build_from_grounding_rows(grounding_rows),
            builder.build_from_synthetic_qa(synthetic_examples),
            builder.build_from_compiled_registry(registry),
            builder.build_from_bridge_facts(bridge_facts),
        ]
    )
    if args.denoise:
        triples = builder.denoise_false_negatives(triples)
    stats = builder.statistics(triples)

    args.out_triples.parent.mkdir(parents=True, exist_ok=True)
    args.out_pairwise.parent.mkdir(parents=True, exist_ok=True)
    write_training_triples_jsonl(args.out_triples, triples)
    write_pairwise_labels_jsonl(args.out_pairwise, triples)
    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats.__dict__, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(stats.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
