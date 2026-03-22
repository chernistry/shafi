"""Build a deterministic synthetic QA corpus from compiled offline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_challenge.ml.synthetic_qa_factory import (
    SyntheticQAFactory,
    build_manifest,
    load_applicability_graph,
    load_bridge_fact_records,
    load_corpus_registry,
    load_legal_segments,
    write_synthetic_qa_jsonl,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--bridge-facts", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-family", type=int, default=0)
    parser.add_argument("--llm-paraphrase", action="store_true")
    parser.add_argument("--stats", action="store_true")
    return parser


def main() -> int:
    """Run the synthetic QA builder.

    Returns:
        Exit code.
    """
    args = build_arg_parser().parse_args()
    registry = load_corpus_registry(args.registry)
    segments = load_legal_segments(args.segments)
    graph = load_applicability_graph(args.graph)
    bridge_facts = load_bridge_fact_records(args.bridge_facts) if args.bridge_facts is not None else []

    factory = SyntheticQAFactory()
    examples = factory.generate_all(
        registry=registry,
        segments=segments,
        graph=graph,
        bridge_facts=bridge_facts,
        max_per_family=args.max_per_family,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_synthetic_qa_jsonl(args.out, examples)

    manifest = build_manifest(
        examples=examples,
        bridge_fact_count=len(bridge_facts),
        llm_paraphrase_enabled=bool(args.llm_paraphrase),
        source_paths={
            "registry": str(args.registry),
            "segments": str(args.segments),
            "graph": str(args.graph),
            "bridge_facts": str(args.bridge_facts) if args.bridge_facts is not None else "",
        },
    )
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    if args.stats:
        print(json.dumps(manifest.model_dump(mode="json"), indent=2))
    else:
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
