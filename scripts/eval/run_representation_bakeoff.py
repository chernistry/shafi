from __future__ import annotations

import argparse
from pathlib import Path

from shafi.eval.representation_bakeoff import (
    ExternalRepresentationRow,
    LocalRepresentationMetric,
    build_bakeoff_markdown,
    load_external_benchmark_csv,
    load_local_representation_metrics,
    summarize_representation_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local-first representation bakeoff summary.")
    parser.add_argument(
        "--external-csv",
        action="append",
        default=[],
        help="Optional external benchmark CSV exported from MTEB or similar.",
    )
    parser.add_argument(
        "--local-metrics",
        action="append",
        default=[],
        help="JSON or JSONL local metric file for one or more representation candidates.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for the markdown report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    external_paths = [Path(path) for path in args.external_csv]
    external_rows: list[ExternalRepresentationRow] = []
    for path in external_paths:
        external_rows.extend(load_external_benchmark_csv(path))

    local_paths = [Path(path) for path in args.local_metrics]
    local_metrics: list[LocalRepresentationMetric] = []
    for path in local_paths:
        local_metrics.extend(load_local_representation_metrics(path))

    summaries = summarize_representation_candidates(local_metrics, external_rows=external_rows)
    markdown = build_bakeoff_markdown(
        summaries,
        external_rows_used=external_paths,
        local_metric_files=local_paths,
    )
    report_path = output_dir / "representation_bakeoff.md"
    report_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote bakeoff report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
