from __future__ import annotations

from scripts.benchmark_local_embeddings import _parse_vector, _percentile, _summarize


def test_percentile_interpolates() -> None:
    assert _percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert _percentile([10.0, 20.0, 30.0, 40.0], 95) == 38.5


def test_summarize_returns_basic_latency_stats() -> None:
    summary = _summarize([10.0, 20.0, 40.0, 50.0])

    assert summary.count == 4
    assert summary.p50_ms == 30.0
    assert summary.min_ms == 10.0
    assert summary.max_ms == 50.0


def test_parse_vector_supports_embed_and_embeddings_shapes() -> None:
    assert _parse_vector({"embedding": [1, 2, 3]}) == [1.0, 2.0, 3.0]
    assert _parse_vector({"embeddings": [[4, 5]]}) == [4.0, 5.0]
    assert _parse_vector({"data": [{"embedding": [6, 7]}]}) == [6.0, 7.0]
