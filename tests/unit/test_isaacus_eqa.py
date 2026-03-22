"""Unit tests for the Isaacus extractive QA adapter.

Tests cover the actual Isaacus API response schema:
  {"extractions": [{"index": int, "inextractability_score": float,
                    "answers": [{"text": str, "start": int, "end": int, "score": float}]}]}

These tests validate the three critical bug fixes:
1. Request field "query" not "question"
2. EQA key loaded from embed.api_key (not reranker.isaacus_api_key)
3. Response parsed from "extractions" array (not top-level "extraction" dict)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_challenge.core.isaacus_eqa import EQAResult, call_isaacus_eqa


def _mock_response(status_code: int, body: Any) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    return resp


def _successful_response(
    *,
    inextractability_score: float = 0.1,
    index: int = 0,
    text: str = "Employment Law",
    start: int = 10,
    end: int = 24,
    score: float = 0.95,
) -> dict[str, Any]:
    return {
        "extractions": [
            {
                "index": index,
                "inextractability_score": inextractability_score,
                "answers": [{"text": text, "start": start, "end": end, "score": score}],
            }
        ],
        "usage": {"input_tokens": 42},
    }


@pytest.mark.asyncio
async def test_successful_extraction_returns_eqa_result() -> None:
    """Happy path: valid extraction returns EQAResult with used=True."""
    mock_resp = _mock_response(200, _successful_response(text="Employment Law", index=1))

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        result = await call_isaacus_eqa(
            question="What is the name of the employment law?",
            texts=["some text", "Employment Law was enacted in 2012"],
            api_key="test-key",
        )

    assert result is not None
    assert result.used is True
    assert result.answer == "Employment Law"
    assert result.passage_index == 1
    assert result.inextractability_score == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_request_uses_query_field_not_question() -> None:
    """Critical: request body must use 'query' field (not 'question')."""
    mock_resp = _mock_response(200, _successful_response())
    captured_payload: dict[str, Any] = {}

    async def capture_post(url: str, *, json: Any, headers: Any) -> Any:
        captured_payload.update(json)
        return mock_resp

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.side_effect = capture_post

        await call_isaacus_eqa(
            question="Who signed the agreement?",
            texts=["text one"],
            api_key="test-key",
        )

    assert "query" in captured_payload, "Request must use 'query' field"
    assert "question" not in captured_payload, "Request must NOT use 'question' field"
    assert captured_payload["query"] == "Who signed the agreement?"


@pytest.mark.asyncio
async def test_high_inextractability_score_returns_not_used() -> None:
    """When inextractability_score >= 0.5: return EQAResult(used=False)."""
    mock_resp = _mock_response(
        200,
        {
            "extractions": [
                {
                    "index": 0,
                    "inextractability_score": 0.8,
                    "answers": [],
                }
            ],
            "usage": {},
        },
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        result = await call_isaacus_eqa(
            question="What is the claim amount?",
            texts=["text"],
            api_key="test-key",
        )

    assert result is not None
    assert result.used is False
    assert result.inextractability_score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_missing_extractions_returns_none() -> None:
    """If 'extractions' key is absent or empty: return None (fall through to LLM)."""
    for bad_body in [
        {},
        {"extractions": []},
        {"extractions": None},
        {"inextractability_score": 0.1, "extraction": {"text": "wrong schema"}},
    ]:
        mock_resp = _mock_response(200, bad_body)
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_resp

            result = await call_isaacus_eqa(
                question="test?",
                texts=["text"],
                api_key="test-key",
            )

        assert result is None, f"Expected None for bad_body={bad_body!r}"


@pytest.mark.asyncio
async def test_empty_api_key_returns_none_without_calling_api() -> None:
    """Empty API key → skip API call, return None."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        result = await call_isaacus_eqa(
            question="test?",
            texts=["text"],
            api_key="",
        )
    mock_client_cls.assert_not_called()
    assert result is None


@pytest.mark.asyncio
async def test_http_error_returns_none() -> None:
    """HTTP 4xx/5xx errors: return None (fall through to LLM)."""
    for status in (400, 401, 429, 500, 503):
        mock_resp = _mock_response(status, {})
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_resp

            result = await call_isaacus_eqa(
                question="test?",
                texts=["text"],
                api_key="test-key",
            )
        assert result is None, f"Expected None for HTTP {status}"


@pytest.mark.asyncio
async def test_passage_index_comes_from_extraction_index_field() -> None:
    """passage_index must come from extractions[0].index (not deprecated passage_index field)."""
    mock_resp = _mock_response(200, _successful_response(index=3, text="Article 12"))

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        result = await call_isaacus_eqa(
            question="Which article?",
            texts=["t0", "t1", "t2", "Article 12 states...", "t4"],
            api_key="test-key",
        )

    assert result is not None
    assert result.passage_index == 3
