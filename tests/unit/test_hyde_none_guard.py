"""Verify hyde.py handles None content from OpenAI API without crash."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_generate_hypothetical_document_handles_none_content() -> None:
    """If OpenAI returns content=None, function returns empty string, not AttributeError."""
    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_settings = MagicMock()
    mock_settings.llm.resolved_api_key.return_value.get_secret_value.return_value = "test-key"
    mock_settings.llm.base_url = None
    mock_settings.llm.timeout_s = 30

    with patch("shafi.core.hyde.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = mock_client

        from shafi.core.hyde import generate_hypothetical_document

        result = await generate_hypothetical_document("What is the penalty for late filing?", settings=mock_settings)

    assert result == "", f"Expected empty string for None content, got: {result!r}"


@pytest.mark.asyncio
async def test_generate_hypothetical_document_returns_stripped_string() -> None:
    """Normal content is stripped and returned."""
    mock_choice = MagicMock()
    mock_choice.message.content = "  Article 5 provides that penalties shall be applied.  "
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_settings = MagicMock()
    mock_settings.llm.resolved_api_key.return_value.get_secret_value.return_value = "test-key"
    mock_settings.llm.base_url = None
    mock_settings.llm.timeout_s = 30

    with patch("shafi.core.hyde.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = mock_client

        from shafi.core.hyde import generate_hypothetical_document

        result = await generate_hypothetical_document("What is the penalty?", settings=mock_settings)

    assert result == "Article 5 provides that penalties shall be applied."


@pytest.mark.asyncio
async def test_generate_hypothetical_document_returns_empty_on_exception() -> None:
    """Exceptions are caught and return empty string (graceful degradation)."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))

    mock_settings = MagicMock()
    mock_settings.llm.resolved_api_key.return_value.get_secret_value.return_value = "test-key"
    mock_settings.llm.base_url = None
    mock_settings.llm.timeout_s = 30

    with patch("shafi.core.hyde.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = mock_client

        from shafi.core.hyde import generate_hypothetical_document

        result = await generate_hypothetical_document("What is the penalty?", settings=mock_settings)

    assert result == "", f"Expected empty string on exception, got: {result!r}"
