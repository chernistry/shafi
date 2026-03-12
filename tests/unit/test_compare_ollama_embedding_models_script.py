from __future__ import annotations

from scripts.benchmark_ollama_embeddings import BenchmarkSummary
from scripts.compare_ollama_embedding_models import render_comparison


def test_render_comparison_ranks_fastest_model_first() -> None:
    slower = BenchmarkSummary(
        model="b-model",
        base_url="http://example.test",
        sample_count=5,
        rounds=5,
        warmup_rounds=1,
        embedding_dimension=1024,
        latency_ms_p50=210.0,
        latency_ms_p95=240.0,
        latency_ms_mean=220.0,
        docs_per_second_mean=22.0,
    )
    faster = BenchmarkSummary(
        model="a-model",
        base_url="http://example.test",
        sample_count=5,
        rounds=5,
        warmup_rounds=1,
        embedding_dimension=1024,
        latency_ms_p50=180.0,
        latency_ms_p95=195.0,
        latency_ms_mean=182.0,
        docs_per_second_mean=28.0,
    )

    report = render_comparison([slower, faster])

    assert "- recommended_model: `a-model`" in report
    assert "| 1 | `a-model` | 1024 | 180.00 | 195.00 | 182.00 | 28.00 |" in report
