"""Tests for step-back query rewriter graceful handling (shai-sprint-b2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_step_back_returns_original_on_api_failure() -> None:
    """On any API failure, returns original query unchanged."""
    with patch("rag_challenge.core.step_back_rewriter.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))
        mock_cls.return_value = mock_client

        from rag_challenge.core.step_back_rewriter import rewrite_step_back

        result = await rewrite_step_back("What does Article 5 say?", api_key="test-key")

    assert result == "What does Article 5 say?"


@pytest.mark.asyncio
async def test_step_back_returns_rewritten_on_success() -> None:
    """On success, returns the rewritten (non-empty) string."""
    mock_choice = MagicMock()
    mock_choice.message.content = "overtime pay obligations under employment law"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("rag_challenge.core.step_back_rewriter.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        from rag_challenge.core.step_back_rewriter import rewrite_step_back

        result = await rewrite_step_back(
            "What does Article 5(2)(a) say about overtime?", api_key="test-key"
        )

    assert result == "overtime pay obligations under employment law"


@pytest.mark.asyncio
async def test_step_back_returns_original_on_empty_response() -> None:
    """Returns original query if rewritten response is empty or too short."""
    mock_choice = MagicMock()
    mock_choice.message.content = "   "  # whitespace only — too short
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("rag_challenge.core.step_back_rewriter.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        from rag_challenge.core.step_back_rewriter import rewrite_step_back

        result = await rewrite_step_back("What does Article 5 say?", api_key="test-key")

    assert result == "What does Article 5 say?"
