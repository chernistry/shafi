#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx

DEFAULT_TEXTS = [
    "According to page 2 of the judgment, from which specific claim number did the appeal in CA 009/2024 originate?",
    "Identify the case with the higher monetary claim: SCT 169/2025 or SCT 295/2025?",
    "From the title pages of all documents in case CA 005/2025 and case CFI 067/2025, identify whether any individual or company is named as a main party in both cases.",
    "What document must every Registered Person file with the Registrar at the same time as applying for Licence renewal?",
]


@dataclass(frozen=True)
class LatencySummary:
    count: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * max(0.0, min(100.0, pct)) / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    frac = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def _summarize(samples: list[float]) -> LatencySummary:
    if not samples:
        return LatencySummary(count=0, p50_ms=0.0, p95_ms=0.0, mean_ms=0.0, min_ms=0.0, max_ms=0.0)
    return LatencySummary(
        count=len(samples),
        p50_ms=_percentile(samples, 50),
        p95_ms=_percentile(samples, 95),
        mean_ms=statistics.fmean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def _load_texts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_TEXTS)
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return texts or list(DEFAULT_TEXTS)


def _parse_vector(payload: dict[str, Any]) -> list[float]:
    vector = payload.get("embedding")
    if isinstance(vector, list):
        values = cast("list[object]", vector)
        if values and all(isinstance(v, int | float) for v in values):
            return [float(cast("int | float", v)) for v in values]

    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list):
        rows = cast("list[object]", embeddings)
        if rows:
            first = rows[0]
            if isinstance(first, list):
                first_values = cast("list[object]", first)
                if all(isinstance(v, int | float) for v in first_values):
                    return [float(cast("int | float", v)) for v in first_values]

    data = payload.get("data")
    if isinstance(data, list):
        rows = cast("list[object]", data)
        if not rows:
            raise ValueError("Unsupported embedding response shape")
        first = rows[0]
        if isinstance(first, dict):
            first_dict = cast("dict[str, object]", first)
            embedding = first_dict.get("embedding")
            if isinstance(embedding, list):
                embed_values = cast("list[object]", embedding)
                if all(isinstance(v, int | float) for v in embed_values):
                    return [float(cast("int | float", v)) for v in embed_values]

    raise ValueError("Unsupported embedding response shape")


class OllamaEmbeddingBench:
    def __init__(self, *, base_url: str, model: str, timeout_s: float, concurrency: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._concurrency = max(1, concurrency)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout_s,
            limits=httpx.Limits(max_keepalive_connections=self._concurrency * 2, max_connections=self._concurrency * 4),
        )
        self._sem = asyncio.Semaphore(self._concurrency)

    async def close(self) -> None:
        await self._client.aclose()

    async def _post_embed(self, text: str) -> list[float]:
        request_shapes = [
            ("/api/embed", {"model": self._model, "input": text}),
            ("/api/embeddings", {"model": self._model, "prompt": text}),
        ]
        last_exc: Exception | None = None
        for path, payload in request_shapes:
            try:
                async with self._sem:
                    response = await self._client.post(path, json=payload)
                response.raise_for_status()
                data = cast("dict[str, Any]", response.json())
                return _parse_vector(data)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"Embedding request failed for all supported Ollama endpoints: {last_exc}")

    async def warmup(self) -> int:
        vector = await self._post_embed("warmup")
        return len(vector)

    async def benchmark_single(self, *, texts: list[str], repeats: int) -> tuple[list[float], int]:
        latencies_ms: list[float] = []
        dim = 0
        for index in range(max(1, repeats)):
            text = texts[index % len(texts)]
            start = time.perf_counter()
            vector = await self._post_embed(text)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            dim = len(vector)
        return latencies_ms, dim

    async def benchmark_parallel(self, *, texts: list[str], repeats: int) -> tuple[list[float], int]:
        async def _one(text: str) -> tuple[float, int]:
            start = time.perf_counter()
            vector = await self._post_embed(text)
            return ((time.perf_counter() - start) * 1000.0, len(vector))

        tasks = [_one(texts[index % len(texts)]) for index in range(max(1, repeats))]
        results = await asyncio.gather(*tasks)
        latencies = [latency for latency, _dim in results]
        dim = results[0][1] if results else 0
        return latencies, dim


def _render_report(
    *,
    model: str,
    base_url: str,
    dimension: int,
    warmup_ms: float,
    single: LatencySummary,
    parallel: LatencySummary,
    parallel_repeats: int,
    concurrency: int,
) -> str:
    lines = [
        "# Local Embedding Benchmark",
        "",
        "- Provider: `ollama`",
        f"- Model: `{model}`",
        f"- Base URL: `{base_url}`",
        f"- Embedding dimension: `{dimension}`",
        f"- Warmup latency: `{warmup_ms:.2f} ms`",
        "",
        "## Single Request Latency",
        "",
        f"- count: `{single.count}`",
        f"- p50: `{single.p50_ms:.2f} ms`",
        f"- p95: `{single.p95_ms:.2f} ms`",
        f"- mean: `{single.mean_ms:.2f} ms`",
        f"- min/max: `{single.min_ms:.2f} / {single.max_ms:.2f} ms`",
        "",
        "## Parallel Request Latency",
        "",
        f"- concurrency: `{concurrency}`",
        f"- requests: `{parallel_repeats}`",
        f"- p50: `{parallel.p50_ms:.2f} ms`",
        f"- p95: `{parallel.p95_ms:.2f} ms`",
        f"- mean: `{parallel.mean_ms:.2f} ms`",
        f"- min/max: `{parallel.min_ms:.2f} / {parallel.max_ms:.2f} ms`",
    ]
    return "\n".join(lines) + "\n"


async def _run(args: argparse.Namespace) -> dict[str, object]:
    texts = _load_texts(args.texts)
    bench = OllamaEmbeddingBench(
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout_s,
        concurrency=args.concurrency,
    )
    try:
        warmup_start = time.perf_counter()
        dimension = await bench.warmup()
        warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
        single_latencies, _ = await bench.benchmark_single(texts=texts, repeats=args.single_repeats)
        parallel_latencies, _ = await bench.benchmark_parallel(texts=texts, repeats=args.parallel_repeats)
    finally:
        await bench.close()

    single_summary = _summarize(single_latencies)
    parallel_summary = _summarize(parallel_latencies)
    markdown = _render_report(
        model=args.model,
        base_url=args.base_url,
        dimension=dimension,
        warmup_ms=warmup_ms,
        single=single_summary,
        parallel=parallel_summary,
        parallel_repeats=args.parallel_repeats,
        concurrency=args.concurrency,
    )
    json_payload: dict[str, object] = {
        "provider": "ollama",
        "model": args.model,
        "base_url": args.base_url,
        "dimension": dimension,
        "warmup_ms": warmup_ms,
        "single": single_summary.__dict__,
        "parallel": parallel_summary.__dict__,
        "parallel_repeats": args.parallel_repeats,
        "concurrency": args.concurrency,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    return {"markdown": markdown, "json": json_payload}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local Ollama embedding latency and throughput.")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--texts", type=Path, default=None)
    parser.add_argument("--single-repeats", type=int, default=12)
    parser.add_argument("--parallel-repeats", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = asyncio.run(_run(args))
    args.out.write_text(cast("str", payload["markdown"]), encoding="utf-8")
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload["json"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
