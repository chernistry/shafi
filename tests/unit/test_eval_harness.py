from __future__ import annotations

import httpx

from shafi.eval.harness import EvalResult, _classify_zero_support_case, _is_retryable_eval_exception


def test_eval_result_summary_includes_ttft_modifier_estimate_mean() -> None:
    result = EvalResult(
        ttft_values=[800.0, 1500.0, 2500.0, 4500.0],
        ttft_by_answer_type={
            "boolean": [800.0, 1500.0],
            "free_text": [2500.0, 4500.0],
        },
        support_page_checked_cases=4,
        zero_support_cases=2,
        justified_zero_support_cases=1,
        unjustified_zero_support_cases=1,
        zero_support_reason_counts={
            "no_information": 1,
            "missing_support_pages": 1,
        },
        unjustified_zero_support_examples=[
            {
                "case_id": "q-missing",
                "answer_type": "free_text",
                "reason": "missing_support_pages",
                "question": "Q?",
                "answer": "Answer",
            }
        ],
    )

    summary = result.summary()

    assert summary["ttft_modifier_estimate_mean"] == 0.9888
    assert summary["ttft_by_answer_type"] == {
        "boolean": {"p50_ms": 800.0, "p95_ms": 800.0, "ttft_modifier_estimate_mean": 1.035, "count": 2},
        "free_text": {"p50_ms": 2500.0, "p95_ms": 2500.0, "ttft_modifier_estimate_mean": 0.9425, "count": 2},
    }
    assert summary["support_page_guard"] == {
        "checked_cases": 4,
        "zero_support_cases": 2,
        "justified_zero_support_cases": 1,
        "unjustified_zero_support_cases": 1,
        "zero_support_reason_counts": {
            "missing_support_pages": 1,
            "no_information": 1,
        },
        "top_unjustified_zero_support_cases": [
            {
                "case_id": "q-missing",
                "answer_type": "free_text",
                "reason": "missing_support_pages",
                "question": "Q?",
                "answer": "Answer",
            }
        ],
    }


def test_classify_zero_support_case_distinguishes_legitimate_empty_support() -> None:
    assert _classify_zero_support_case(
        answer_text="There is no information on this question.",
        answer_type="free_text",
        failure=None,
        used_pages=[],
    ) == (True, "no_information")
    assert _classify_zero_support_case(
        answer_text="null",
        answer_type="boolean",
        failure=None,
        used_pages=[],
    ) == (True, "strict_unanswerable")
    assert _classify_zero_support_case(
        answer_text="Yes",
        answer_type="boolean",
        failure=None,
        used_pages=[],
    ) == (False, "missing_support_pages")
    assert _classify_zero_support_case(
        answer_text="Yes",
        answer_type="boolean",
        failure="transport failure",
        used_pages=[],
    ) == (False, "eval_failure")


def test_is_retryable_eval_exception_detects_transient_failures() -> None:
    assert _is_retryable_eval_exception(httpx.ConnectError("All connection attempts failed"))
    assert _is_retryable_eval_exception(httpx.RemoteProtocolError("peer closed connection"))
    assert _is_retryable_eval_exception(RuntimeError(""))
    assert not _is_retryable_eval_exception(ValueError("invalid answer format"))
