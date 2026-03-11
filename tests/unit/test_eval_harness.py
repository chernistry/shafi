from __future__ import annotations

import httpx

from rag_challenge.eval.harness import EvalResult, _is_retryable_eval_exception


def test_eval_result_summary_includes_ttft_modifier_estimate_mean() -> None:
    result = EvalResult(
        ttft_values=[800.0, 1500.0, 2500.0, 4500.0],
        ttft_by_answer_type={
            "boolean": [800.0, 1500.0],
            "free_text": [2500.0, 4500.0],
        },
    )

    summary = result.summary()

    assert summary["ttft_modifier_estimate_mean"] == 0.9888
    assert summary["ttft_by_answer_type"] == {
        "boolean": {"p50_ms": 800.0, "p95_ms": 800.0, "ttft_modifier_estimate_mean": 1.035, "count": 2},
        "free_text": {"p50_ms": 2500.0, "p95_ms": 2500.0, "ttft_modifier_estimate_mean": 0.9425, "count": 2},
    }


def test_is_retryable_eval_exception_detects_transient_failures() -> None:
    assert _is_retryable_eval_exception(httpx.ConnectError("All connection attempts failed"))
    assert _is_retryable_eval_exception(httpx.RemoteProtocolError("peer closed connection"))
    assert _is_retryable_eval_exception(RuntimeError(""))
    assert not _is_retryable_eval_exception(ValueError("invalid answer format"))
