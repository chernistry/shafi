"""Unit tests for post-hoc citation verifier."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_chunk(chunk_id: str, text: str) -> MagicMock:
    chunk = MagicMock()
    chunk.chunk_id = chunk_id
    chunk.text = text
    chunk.content = text
    return chunk


@pytest.mark.asyncio
async def test_verify_citations_keeps_supported() -> None:
    """YES response keeps the citation."""
    mock_choice = MagicMock()
    mock_choice.message.content = "YES"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("rag_challenge.core.citation_verifier.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        from rag_challenge.core.citation_verifier import verify_citations

        chunks = [_make_chunk("id1", "Article 5 states the penalty is USD 50,000.")]
        result = await verify_citations("The penalty is USD 50,000.", chunks, api_key="test")

    assert len(result) == 1
    assert result[0].chunk_id == "id1"


@pytest.mark.asyncio
async def test_verify_citations_drops_unsupported() -> None:
    """NO response removes the citation — but only if at least one remains."""
    responses = ["NO", "YES"]
    call_count = 0

    async def _mock_create(**kwargs: object) -> MagicMock:
        nonlocal call_count
        verdict = responses[call_count % len(responses)]
        call_count += 1
        mock_choice = MagicMock()
        mock_choice.message.content = verdict
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    with patch("rag_challenge.core.citation_verifier.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=_mock_create)
        mock_cls.return_value = mock_client

        from rag_challenge.core.citation_verifier import verify_citations

        chunks = [
            _make_chunk("bad_id", "Unrelated text about weather."),
            _make_chunk("good_id", "Article 5 states the penalty is USD 50,000."),
        ]
        result = await verify_citations("The penalty is USD 50,000.", chunks, api_key="test")

    assert len(result) == 1
    assert result[0].chunk_id == "good_id"


@pytest.mark.asyncio
async def test_verify_citations_safety_guard_keeps_all_if_all_dropped() -> None:
    """If ALL citations would be dropped, return original list (safety guard)."""
    mock_choice = MagicMock()
    mock_choice.message.content = "NO"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("rag_challenge.core.citation_verifier.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        from rag_challenge.core.citation_verifier import verify_citations

        chunks = [_make_chunk("id1", "Text A"), _make_chunk("id2", "Text B")]
        result = await verify_citations("Some answer.", chunks, api_key="test")

    # Safety guard: return original rather than empty list
    assert len(result) == 2


@pytest.mark.asyncio
async def test_verify_citations_returns_original_on_api_error() -> None:
    """API error on any chunk keeps that chunk (safe default)."""
    with patch("rag_challenge.core.citation_verifier.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))
        mock_cls.return_value = mock_client

        from rag_challenge.core.citation_verifier import verify_citations

        chunks = [_make_chunk("id1", "Text A")]
        result = await verify_citations("Some answer.", chunks, api_key="test")

    assert len(result) == 1
