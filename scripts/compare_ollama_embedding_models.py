from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_ollama_embeddings import (  # noqa: E402
    BenchmarkSummary,
    benchmark_embeddings,
    load_texts,
)


def _rank_models(results: list[BenchmarkSummary]) -> list[BenchmarkSummary]:
    return sorted(
        results,
        key=lambda summary: (
            summary.latency_ms_p50,
            summary.latency_ms_p95,
            -summary.docs_per_second_mean,
            summary.model,
        ),
    )


def render_comparison(results: list[BenchmarkSummary]) -> str:
    ranked = _rank_models(results)
    lines = [
        "# Ollama Embedding Model Comparison",
        "",
        f"- compared_models: `{len(ranked)}`",
        f"- recommended_model: `{ranked[0].model}`",
        "",
        "| Rank | Model | Dim | p50 ms | p95 ms | Mean ms | Docs/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, summary in enumerate(ranked, start=1):
        lines.append(
            "| "
            f"{index} | `{summary.model}` | {summary.embedding_dimension} | "
            f"{summary.latency_ms_p50:.2f} | {summary.latency_ms_p95:.2f} | "
            f"{summary.latency_ms_mean:.2f} | {summary.docs_per_second_mean:.2f} |"
        )
    return "\n".join(lines)


async def _compare_models(args: argparse.Namespace) -> list[BenchmarkSummary]:
    texts = load_texts(args.text_file)
    results: list[BenchmarkSummary] = []
    for model in args.model:
        results.append(
            await benchmark_embeddings(
                model=model,
                texts=texts,
                rounds=args.rounds,
                warmup_rounds=args.warmup_rounds,
                base_url=args.base_url,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local Ollama embedding models.")
    parser.add_argument("--model", action="append", required=True, help="Embedding model name. Repeat for multiple models.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = asyncio.run(_compare_models(args))
    report = render_comparison(results)
    payload = {
        "recommended_model": _rank_models(results)[0].model,
        "results": [asdict(summary) for summary in _rank_models(results)],
    }

    if args.out is not None:
        args.out.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
