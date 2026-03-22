from __future__ import annotations

from scripts.rank_local_embedding_branch_candidates import _build_rows, _score


def test_build_rows_joins_relevance_and_latency_by_model() -> None:
    rows = _build_rows(
        relevance_payload={
            "summaries": [
                {
                    "model": "embeddinggemma:latest",
                    "evaluated_cases": 45,
                    "skipped_cases": 28,
                    "gold_top1_rate": 0.711,
                    "gold_top3_rate": 0.956,
                    "mean_gold_margin": 0.0475,
                    "mean_best_gold_rank": 1.44,
                }
            ]
        },
        latency_payload={
            "results": [
                {
                    "model": "embeddinggemma:latest",
                    "latency_ms_p50": 146.1,
                    "latency_ms_p95": 288.7,
                    "docs_per_second_mean": 31.4,
                }
            ]
        },
    )

    assert rows == [
        {
            "model": "embeddinggemma:latest",
            "evaluated_cases": 45,
            "skipped_cases": 28,
            "gold_top1_rate": 0.711,
            "gold_top3_rate": 0.956,
            "mean_gold_margin": 0.0475,
            "mean_best_gold_rank": 1.44,
            "latency_ms_p50": 146.1,
            "latency_ms_p95": 288.7,
            "docs_per_second_mean": 31.4,
        }
    ]


def test_score_prefers_relevance_then_latency() -> None:
    faster = {
        "model": "faster",
        "gold_top1_rate": 0.65,
        "gold_top3_rate": 0.95,
        "mean_gold_margin": 0.02,
        "latency_ms_p50": 100.0,
        "docs_per_second_mean": 30.0,
    }
    better = {
        "model": "better",
        "gold_top1_rate": 0.71,
        "gold_top3_rate": 0.95,
        "mean_gold_margin": 0.02,
        "latency_ms_p50": 150.0,
        "docs_per_second_mean": 25.0,
    }

    assert _score(better) > _score(faster)
