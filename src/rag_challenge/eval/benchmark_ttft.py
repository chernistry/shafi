from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import httpx

from rag_challenge.eval.golden import load_golden_dataset


def _float_list() -> list[float]:
    return []


def _str_list() -> list[str]:
    return []


def _stage_dict() -> dict[str, list[float]]:
    return {
        "classify_ms": [],
        "embed_ms": [],
        "qdrant_ms": [],
        "rerank_ms": [],
        "llm_ms": [],
        "verify_ms": [],
    }


@dataclass
class BenchmarkStats:
    ttft_values: list[float] = field(default_factory=_float_list)
    failures: list[str] = field(default_factory=_str_list)
    stage_values: dict[str, list[float]] = field(default_factory=_stage_dict)

    def summary(self) -> dict[str, object]:
        return {
            "count": len(self.ttft_values),
            "ttft_p50_ms": round(_percentile(self.ttft_values, 0.5), 1),
            "ttft_p95_ms": round(_percentile(self.ttft_values, 0.95), 1),
            "ttft_mean_ms": round(statistics.fmean(self.ttft_values), 1) if self.ttft_values else 0.0,
            "stage_p50_ms": {key: round(_percentile(values, 0.5), 1) for key, values in self.stage_values.items()},
            "stage_p95_ms": {key: round(_percentile(values, 0.95), 1) for key, values in self.stage_values.items()},
            "failures": len(self.failures),
        }


async def _run_case(
    *,
    endpoint: str,
    case_id: str,
    question: str,
    answer_type: str,
    client: httpx.AsyncClient,
    timeout_s: float,
) -> tuple[float | None, dict[str, float], str | None]:
    t0 = time.perf_counter()
    first_token_ms: float | None = None
    telemetry: dict[str, float] = {}
    payload = {
        "question": question,
        "request_id": case_id,
        "question_id": case_id,
        "answer_type": answer_type,
    }

    try:
        async with client.stream("POST", endpoint, json=payload, timeout=timeout_s) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if not raw:
                    continue
                event_obj: object = json.loads(raw)
                if not isinstance(event_obj, dict):
                    continue
                event = cast("dict[str, object]", event_obj)
                event_type = event.get("type")
                if event_type == "token" and first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t0) * 1000.0
                if event_type == "telemetry":
                    payload_obj = event.get("payload")
                    if isinstance(payload_obj, dict):
                        payload_data = cast("dict[str, object]", payload_obj)
                        telemetry = {
                            "classify_ms": _coerce_float(payload_data.get("classify_ms")),
                            "embed_ms": _coerce_float(payload_data.get("embed_ms")),
                            "qdrant_ms": _coerce_float(payload_data.get("qdrant_ms")),
                            "rerank_ms": _coerce_float(payload_data.get("rerank_ms")),
                            "llm_ms": _coerce_float(payload_data.get("llm_ms")),
                            "verify_ms": _coerce_float(payload_data.get("verify_ms")),
                        }
                if event_type == "done":
                    break
    except Exception as exc:
        return None, {}, str(exc)

    return first_token_ms, telemetry, None


async def run_benchmark(
    *,
    questions_path: str | Path,
    endpoint: str,
    concurrency: int,
    timeout_s: float,
) -> BenchmarkStats:
    cases = load_golden_dataset(questions_path)
    stats = BenchmarkStats()
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    lock = asyncio.Lock()

    async with httpx.AsyncClient() as client:
        async def _worker(case_index: int) -> None:
            case = cases[case_index]
            async with semaphore:
                ttft_ms, telemetry, failure = await _run_case(
                    endpoint=endpoint,
                    case_id=case.case_id,
                    question=case.question,
                    answer_type=case.answer_type,
                    client=client,
                    timeout_s=timeout_s,
                )
            async with lock:
                if failure is not None:
                    stats.failures.append(f"{case.case_id}: {failure}")
                    return
                if ttft_ms is not None:
                    stats.ttft_values.append(ttft_ms)
                for key, value in telemetry.items():
                    stats.stage_values[key].append(value)

        await asyncio.gather(*[_worker(i) for i in range(len(cases))])

    return stats


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * q)))
    return ordered[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TTFT for /query SSE endpoint.")
    parser.add_argument("--questions", default="dataset/public_dataset.json", help="Path to golden questions JSON.")
    parser.add_argument("--endpoint", default="http://localhost:8000/query", help="Query endpoint URL.")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests.")
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Per-request timeout in seconds.")
    parser.add_argument("--out", default="", help="Optional JSON output file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = asyncio.run(
        run_benchmark(
            questions_path=args.questions,
            endpoint=args.endpoint,
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
        )
    )
    summary = stats.summary()
    output = {"summary": summary, "failures": stats.failures}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
