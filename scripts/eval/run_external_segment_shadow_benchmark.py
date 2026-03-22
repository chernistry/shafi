"""Run the external segment payload shadow benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_challenge.eval.external_segment_shadow import (
    load_and_evaluate_external_segment_shadow,
    render_external_segment_shadow_markdown,
    run_external_segment_shadow_ablation,
)
from rag_challenge.ingestion.rich_segment_text import SegmentTextMode


def main() -> int:
    """Run the CLI entry point.

    Returns:
        Process exit code.
    """

    parser = argparse.ArgumentParser(description="Run external segment payload shadow benchmark")
    parser.add_argument("--payload", type=Path, required=True, help="External segment payload JSON")
    parser.add_argument("--benchmark", type=Path, required=True, help="Eval or raw-results benchmark JSON")
    parser.add_argument("--out-json", type=Path, required=True, help="Output JSON summary path")
    parser.add_argument("--out-md", type=Path, required=True, help="Output markdown summary path")
    parser.add_argument("--projected-top-k", type=int, default=3, help="Max projected pages per case")
    parser.add_argument("--candidate-pool-size", type=int, default=24, help="Max ranked segments per case")
    parser.add_argument(
        "--composer-mode",
        choices=[SegmentTextMode.PLAIN.value, SegmentTextMode.RICH.value, "ablate"],
        default=SegmentTextMode.RICH.value,
        help="Segment text composition mode or full ablation",
    )
    args = parser.parse_args()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    if args.composer_mode == "ablate":
        summary = run_external_segment_shadow_ablation(
            payload_path=args.payload,
            benchmark_path=args.benchmark,
            projected_top_k=args.projected_top_k,
            candidate_pool_size=args.candidate_pool_size,
        )
        args.out_json.write_text(
            json.dumps(summary.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        args.out_md.write_text(
            "\n".join(
                [
                    "# External Segment Composer Ablation",
                    "",
                    "## Plain",
                    "",
                    render_external_segment_shadow_markdown(summary.plain_summary),
                    "",
                    "## Rich",
                    "",
                    render_external_segment_shadow_markdown(summary.rich_summary),
                    "",
                    "## Diagnostics",
                    "",
                    f"- title/header noise rate (plain): `{summary.title_header_noise_rate['plain']:.3f}`",
                    f"- title/header noise rate (rich): `{summary.title_header_noise_rate['rich']:.3f}`",
                    f"- average projected page count (plain): `{summary.avg_projected_page_count['plain']:.3f}`",
                    f"- average projected page count (rich): `{summary.avg_projected_page_count['rich']:.3f}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return 0

    summary = load_and_evaluate_external_segment_shadow(
        payload_path=args.payload,
        benchmark_path=args.benchmark,
        projected_top_k=args.projected_top_k,
        candidate_pool_size=args.candidate_pool_size,
        composer_mode=SegmentTextMode(args.composer_mode),
    )
    args.out_json.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(render_external_segment_shadow_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
