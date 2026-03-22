"""CLI for compiling parsed legal documents into a corpus registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_challenge.ingestion.corpus_compiler import CorpusCompiler


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser for the corpus compiler CLI.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-dir", type=Path, required=True, help="Directory containing source documents.")
    parser.add_argument("--out", type=Path, required=True, help="Path to the output registry JSON file.")
    parser.add_argument(
        "--format",
        choices=("json",),
        default="json",
        help="Serialization format for the compiled registry.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print compiler summary after writing output.")
    return parser


def main() -> None:
    """Compile a corpus registry and write it to disk.

    Raises:
        ValueError: If an unsupported output format is requested.
    """
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.format != "json":
        raise ValueError(f"Unsupported output format: {args.format}")
    compiler = CorpusCompiler()
    registry = compiler.compile_corpus(args.doc_dir)
    compiler.write_registry(registry, args.out)
    if args.verbose:
        print(
            {
                "source_doc_count": registry.source_doc_count,
                "laws": len(registry.laws),
                "cases": len(registry.cases),
                "orders": len(registry.orders),
                "practice_directions": len(registry.practice_directions),
                "amendments": len(registry.amendments),
                "other_documents": len(registry.other_documents),
                "entities": len(registry.entities),
                "links": len(registry.links),
            }
        )


if __name__ == "__main__":
    main()
