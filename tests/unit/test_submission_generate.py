from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from rag_challenge.submission.generate import (
    SubmissionCase,
    _coerce_answer_type,
    _normalize_free_text_answer,
    _run_case,
)


@dataclass
class _FakeResponse:
    text: str
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def post(self, endpoint: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        del endpoint, json, timeout
        return self._response


def _sse(*events: dict[str, object]) -> str:
    lines = [f"data: {json.dumps(evt)}\n" for evt in events]
    return "".join(lines)


@pytest.mark.asyncio
async def test_submission_prefers_used_pages_over_context_pages() -> None:
    telemetry = {
        "ttft_ms": 123,
        "time_per_output_token_ms": 45,
        "total_ms": 999,
        "used_page_ids": ["abc123_3"],
        "cited_page_ids": ["abc123_3"],
        "context_page_ids": ["abc123_3", "abc123_7"],
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "model_llm": "openai/gpt-4o-mini",
    }
    body = _sse(
        {"type": "token", "text": "Yes"},
        {"type": "telemetry", "payload": telemetry},
        {"type": "done"},
    )
    case = SubmissionCase(case_id="q-1", question="Q?", answer_type="boolean")
    result = await _run_case(case, _FakeClient(_FakeResponse(text=body)), "http://localhost:8000/query")

    assert result["retrieved_chunk_ids"] == ["abc123_3"]


@pytest.mark.asyncio
async def test_submission_falls_back_to_chunk_to_page_conversion() -> None:
    telemetry = {
        "ttft_ms": 123,
        "time_per_output_token_ms": 45,
        "total_ms": 999,
        "cited_chunk_ids": ["abc123:0:0:deadbeef"],
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "model_llm": "openai/gpt-4o-mini",
    }
    body = _sse(
        {"type": "token", "text": "Yes"},
        {"type": "telemetry", "payload": telemetry},
        {"type": "done"},
    )
    case = SubmissionCase(case_id="q-1", question="Q?", answer_type="boolean")
    result = await _run_case(case, _FakeClient(_FakeResponse(text=body)), "http://localhost:8000/query")

    assert result["retrieved_chunk_ids"] == ["abc123_1"]


@pytest.mark.asyncio
async def test_submission_does_not_fallback_to_context_pages() -> None:
    telemetry = {
        "ttft_ms": 123,
        "time_per_output_token_ms": 45,
        "total_ms": 999,
        "context_page_ids": ["abc123_3", "abc123_7"],
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "model_llm": "openai/gpt-4o-mini",
    }
    body = _sse(
        {"type": "token", "text": "Yes"},
        {"type": "telemetry", "payload": telemetry},
        {"type": "done"},
    )
    case = SubmissionCase(case_id="q-1", question="Q?", answer_type="boolean")
    result = await _run_case(case, _FakeClient(_FakeResponse(text=body)), "http://localhost:8000/query")

    assert result["retrieved_chunk_ids"] == []


def test_coerce_answer_type_parses_number_int_and_float() -> None:
    assert _coerce_answer_type("4", "number") == 4
    assert _coerce_answer_type("250,499.26", "number") == 250499.26
    assert _coerce_answer_type("405351504.0", "number") == 405351504.0


def test_coerce_answer_type_parses_boolean_json_values() -> None:
    assert _coerce_answer_type("Yes", "boolean") is True
    assert _coerce_answer_type("true", "boolean") is True
    assert _coerce_answer_type("No", "boolean") is False
    assert _coerce_answer_type("false", "boolean") is False


def test_coerce_answer_type_normalizes_textual_dates_to_iso() -> None:
    assert _coerce_answer_type("11 March 2025", "date") == "2025-03-11"
    assert _coerce_answer_type("March 11, 2025", "date") == "2025-03-11"
    assert _coerce_answer_type("01/03/2025", "date") == "2025-03-01"
    assert _coerce_answer_type("2025-03-11", "date") == "2025-03-11"


def test_coerce_answer_type_skips_case_number_fragments() -> None:
    value = _coerce_answer_type("In CA 005/2025 the claim value is 250,499.26 AED.", "number")
    assert value == 250499.26


def test_coerce_answer_type_keeps_unparseable_number_as_string() -> None:
    assert _coerce_answer_type("not a number", "number") == "not a number"


def test_coerce_answer_type_splits_names_and_dedupes() -> None:
    value = _coerce_answer_type("Alice, Bob and Carol; Alice\nDelta", "names")
    assert value == ["Alice", "Bob", "Carol", "Delta"]


def test_coerce_answer_type_names_does_not_alias_distinct_values() -> None:
    value = _coerce_answer_type("UK, United Kingdom", "names")
    assert value == ["UK", "United Kingdom"]


def test_normalize_free_text_answer_strips_citations_and_caps_length() -> None:
    answer = (
        "1. Foundations Law 2018 (cite: abc:0:0:1)\n"
        "2. Trust Law 2018 (cite: def:1:0:2)\n"
        "3. " + ("Very long explanation " * 30)
    )
    normalized = _normalize_free_text_answer(answer)

    assert "(cite:" not in normalized
    assert len(normalized) <= 280
    assert "Foundations Law 2018" in normalized
    assert "Trust Law 2018" in normalized


def test_normalize_free_text_answer_caps_to_three_sentences() -> None:
    answer = (
        "Alpha is supported. Beta is supported too. Gamma remains supported. "
        "Delta should not survive projection."
    )

    normalized = _normalize_free_text_answer(answer)

    assert normalized == "Alpha is supported. Beta is supported too. Gamma remains supported."


@pytest.mark.asyncio
async def test_submission_outputs_null_and_clears_refs_for_strict_unanswerable() -> None:
    telemetry = {
        "ttft_ms": 123,
        "time_per_output_token_ms": 45,
        "total_ms": 999,
        "context_page_ids": ["abc123_3"],
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "model_llm": "openai/gpt-4o-mini",
    }
    body = _sse(
        {"type": "token", "text": "null"},
        {"type": "telemetry", "payload": telemetry},
        {"type": "done"},
    )
    case = SubmissionCase(case_id="q-1", question="Q?", answer_type="number")
    result = await _run_case(case, _FakeClient(_FakeResponse(text=body)), "http://localhost:8000/query")

    assert result["answer"] is None
    assert result["retrieved_chunk_ids"] == []


@pytest.mark.asyncio
async def test_submission_outputs_boolean_as_json_bool() -> None:
    telemetry = {
        "ttft_ms": 123,
        "time_per_output_token_ms": 45,
        "total_ms": 999,
        "used_page_ids": ["abc123_3"],
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "model_llm": "openai/gpt-4o-mini",
    }
    body = _sse(
        {"type": "token", "text": "Yes"},
        {"type": "telemetry", "payload": telemetry},
        {"type": "done"},
    )
    case = SubmissionCase(case_id="q-1", question="Q?", answer_type="boolean")
    result = await _run_case(case, _FakeClient(_FakeResponse(text=body)), "http://localhost:8000/query")

    assert result["answer"] is True


@pytest.mark.asyncio
async def test_submission_normalizes_free_text_for_submission_format() -> None:
    telemetry = {
        "ttft_ms": 123,
        "time_per_output_token_ms": 45,
        "total_ms": 999,
        "used_page_ids": ["abc123_3"],
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "model_llm": "openai/gpt-4o-mini",
    }
    body = _sse(
        {
            "type": "answer_final",
            "text": "1. Alpha (cite: abc123:0:0:x)\n2. " + ("Beta gamma " * 40),
        },
        {"type": "telemetry", "payload": telemetry},
        {"type": "done"},
    )
    case = SubmissionCase(case_id="q-1", question="Q?", answer_type="free_text")
    result = await _run_case(case, _FakeClient(_FakeResponse(text=body)), "http://localhost:8000/query")

    assert isinstance(result["answer"], str)
    assert "(cite:" not in result["answer"]
    assert len(result["answer"]) <= 280
    assert result["answer"].count(".") <= 3
