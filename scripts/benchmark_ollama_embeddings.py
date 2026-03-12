from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import httpx

EmbedFn = Callable[[str, list[str], str], Awaitable[list[list[float]]]]

_DEFAULT_TEXTS = [
    "According to page 2 of the judgment, from which specific claim number did the appeal originate?",
    "From the title page of the document, what is the official law number?",
    "Identify whether any person or company is a main party to both cases.",
    "Who are listed as the claimants in the case documents for CFI 067/2025?",
    "What was the outcome of the specific order or application described in case SCT 295/2025?",
]


@dataclass(frozen=True)
class BenchmarkSummary:
    model: str
    base_url: str
    sample_count: int
    rounds: int
    warmup_rounds: int
    embedding_dimension: int
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_mean: float
    docs_per_second_mean: float


async def _embed_via_ollama(base_url: str, texts: list[str], model: str) -> list[list[float]]:
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        response = await client.post("/api/embed", json={"model": model, "input": texts})
        response.raise_for_status()
        payload = response.json()
        embeddings_obj = payload.get("embeddings")
        if isinstance(embeddings_obj, list) and embeddings_obj and isinstance(embeddings_obj[0], list):
            return [cast("list[float]", item) for item in cast("list[object]", embeddings_obj)]
        embedding_obj = payload.get("embedding")
        if isinstance(embedding_obj, list) and embedding_obj and isinstance(embedding_obj[0], (int, float)):
            return [cast("list[float]", embedding_obj)]
        raise ValueError("Unexpected Ollama embed response shape")


async def benchmark_embeddings(
    *,
    model: str,
    texts: list[str],
    rounds: int,
    warmup_rounds: int,
    base_url: str,
    embed_fn: EmbedFn = _embed_via_ollama,
) -> BenchmarkSummary:
    if not texts:
        raise ValueError("At least one input text is required")
    for _ in range(warmup_rounds):
        await embed_fn(base_url, texts, model)
    latencies_ms: list[float] = []
    docs_per_second: list[float] = []
    dimension = 0
    for _ in range(rounds):
        started = time.perf_counter()
        embeddings = await embed_fn(base_url, texts, model)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not embeddings:
            raise ValueError("Embedding response was empty")
        dimension = len(embeddings[0])
        latencies_ms.append(elapsed_ms)
        docs_per_second.append(len(texts) / max(elapsed_ms / 1000.0, 1e-9))

    return BenchmarkSummary(
        model=model,
        base_url=base_url,
        sample_count=len(texts),
        rounds=rounds,
        warmup_rounds=warmup_rounds,
        embedding_dimension=dimension,
        latency_ms_p50=statistics.median(latencies_ms),
        latency_ms_p95=max(latencies_ms) if len(latencies_ms) < 2 else statistics.quantiles(latencies_ms, n=20)[18],
        latency_ms_mean=statistics.mean(latencies_ms),
        docs_per_second_mean=statistics.mean(docs_per_second),
    )


def load_texts(path: Path | None) -> list[str]:
    if path is None:
        return list(_DEFAULT_TEXTS)
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts:
        raise ValueError(f"No non-empty texts found in {path}")
    return texts


def render_report(summary: BenchmarkSummary) -> str:
    lines = [
        "# Ollama Embedding Benchmark",
        "",
        f"- model: `{summary.model}`",
        f"- base_url: `{summary.base_url}`",
        f"- sample_count: `{summary.sample_count}`",
        f"- rounds: `{summary.rounds}`",
        f"- warmup_rounds: `{summary.warmup_rounds}`",
        f"- embedding_dimension: `{summary.embedding_dimension}`",
        f"- latency_ms_p50: `{summary.latency_ms_p50:.2f}`",
        f"- latency_ms_p95: `{summary.latency_ms_p95:.2f}`",
        f"- latency_ms_mean: `{summary.latency_ms_mean:.2f}`",
        f"- docs_per_second_mean: `{summary.docs_per_second_mean:.2f}`",
    ]
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> BenchmarkSummary:
    return await benchmark_embeddings(
        model=args.model,
        texts=load_texts(args.text_file),
        rounds=args.rounds,
        warmup_rounds=args.warmup_rounds,
        base_url=args.base_url,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a local Ollama embedding model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    summary = asyncio.run(_main_async(args))
    report = render_report(summary)
    payload = asdict(summary)

    if args.out is not None:
        args.out.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
