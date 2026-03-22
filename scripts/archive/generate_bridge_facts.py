"""Generate bridge facts from the compiled legal corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rag_challenge.ingestion.bridge_facts import BridgeFactGenerator
from rag_challenge.ingestion.canonical_entities import EntityAliasResolver
from rag_challenge.ingestion.corpus_compiler import CorpusCompiler


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for bridge-fact generation.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-dir", type=Path, required=True, help="Directory containing source documents.")
    parser.add_argument("--entities", type=Path, default=None, help="Optional canonical entity registry JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Output path for generated bridge facts.")
    parser.add_argument("--stats", action="store_true", help="Print fact generation statistics.")
    return parser


def main() -> None:
    """Generate bridge facts and write them to disk."""

    parser = build_argument_parser()
    args = parser.parse_args()

    registry = CorpusCompiler().compile_corpus(args.doc_dir)
    resolver = (
        EntityAliasResolver.load(args.entities)
        if args.entities is not None and args.entities.exists()
        else EntityAliasResolver.build_from_registry(registry)
    )
    facts = BridgeFactGenerator().generate_all(corpus_registry=registry, entity_resolver=resolver)
    BridgeFactGenerator.write_facts(facts, args.out)

    if args.stats:
        counts = Counter(fact.fact_type.value for fact in facts)
        print(json.dumps({"fact_count": len(facts), "fact_count_by_type": dict(sorted(counts.items()))}, indent=2))


if __name__ == "__main__":
    main()
