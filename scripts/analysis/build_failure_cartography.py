#!/usr/bin/env python3
"""Build a closed-world failure cartography ledger from historical artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from rag_challenge.eval.failure_cartography import (
    build_failure_ledger,
    discover_run_artifacts,
    load_reviewed_golden,
    load_run_observations,
    render_summary_markdown,
)

if TYPE_CHECKING:
    from rag_challenge.eval.failure_cartography_models import RunObservation


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed CLI args.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--reviewed-golden",
        type=Path,
        default=Path(".sdd/golden/reviewed/reviewed_all_100.json"),
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Build and write the failure cartography outputs.

    Returns:
        int: Process exit code.
    """

    args = parse_args()
    reviewed = load_reviewed_golden(args.reviewed_golden.resolve())
    observations: list[RunObservation] = []
    seen_paths: set[Path] = set()
    for runs_dir in [path.resolve() for path in args.runs_dir]:
        for path in discover_run_artifacts(runs_dir):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            observations.extend(load_run_observations(resolved, reviewed))
    ledger = build_failure_ledger(reviewed=reviewed, observations=observations)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(ledger.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(render_summary_markdown(ledger), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
