from __future__ import annotations

import pytest
from scripts.benchmark_ollama_embeddings import benchmark_embeddings, render_report


@pytest.mark.asyncio
async def test_benchmark_embeddings_uses_custom_embedder() -> None:
    calls: list[tuple[str, list[str], str]] = []

    async def fake_embed(base_url: str, texts: list[str], model: str) -> list[list[float]]:
        calls.append((base_url, list(texts), model))
        return [[0.1, 0.2, 0.3] for _ in texts]

    summary = await benchmark_embeddings(
        model="fake-model",
        texts=["a", "b"],
        rounds=2,
        warmup_rounds=1,
        base_url="http://example.test",
        embed_fn=fake_embed,
    )

    assert summary.model == "fake-model"
    assert summary.embedding_dimension == 3
    assert summary.sample_count == 2
    assert len(calls) == 3
    assert calls[0][0] == "http://example.test"
    assert calls[0][2] == "fake-model"
    report = render_report(summary)
    assert "- model: `fake-model`" in report
