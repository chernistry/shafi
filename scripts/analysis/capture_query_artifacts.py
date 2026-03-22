"""Capture raw-results artifacts for an arbitrary local question set."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--raw-results-out", type=Path, required=True)
    parser.add_argument("--submission-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--truth-audit-out", type=Path, default=None)
    parser.add_argument("--truth-audit-workbook-out", type=Path, default=None)
    parser.add_argument("--ingest-doc-dir", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    """Run the capture CLI.

    Returns:
        Process exit code.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from shafi.submission.query_capture import capture_query_artifacts

    args = build_arg_parser().parse_args()
    summary = asyncio.run(
        capture_query_artifacts(
            questions_path=args.questions,
            raw_results_path=args.raw_results_out,
            submission_path=args.submission_out,
            summary_path=args.summary_out,
            docs_dir=args.docs_dir,
            truth_audit_path=args.truth_audit_out,
            truth_audit_workbook_path=args.truth_audit_workbook_out,
            concurrency=args.concurrency,
            fail_fast=args.fail_fast,
            ingest_doc_dir=args.ingest_doc_dir,
        )
    )
    print(args.raw_results_out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
