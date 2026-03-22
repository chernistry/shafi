"""Build an amendment and temporal applicability graph from a corpus registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_challenge.ingestion.applicability_graph import (
    build_applicability_graph,
    validate_graph,
    write_applicability_graph,
)
from rag_challenge.models.legal_objects import CorpusRegistry


def main() -> int:
    """Run the CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True, help="Input corpus registry JSON")
    parser.add_argument("--out", type=Path, required=True, help="Output graph JSON")
    parser.add_argument("--validate", action="store_true", help="Run validation and print warning counts")
    parser.add_argument("--visualize", type=Path, default=None, help="Optional Mermaid output path")
    args = parser.parse_args()

    registry = CorpusRegistry.model_validate_json(args.registry.read_text(encoding="utf-8"))
    graph = build_applicability_graph(registry)
    write_applicability_graph(graph=graph, output_path=args.out)
    if args.validate:
        warnings = validate_graph(graph)
        print(f"validation_warnings={len(warnings)}")
    if args.visualize is not None:
        args.visualize.write_text(_render_mermaid(graph), encoding="utf-8")
    return 0


def _render_mermaid(graph) -> str:
    lines = ["graph TD"]
    for edge in graph.edges:
        lines.append(f"  {edge.source_doc_id}[\"{edge.source_doc_id}\"] -->|{edge.edge_type.value}| {edge.target_doc_id}[\"{edge.target_doc_id}\"]")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
