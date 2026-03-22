from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shafi.llm.provider import LLMResult
from shafi.models import DocType, RankedChunk


def _chunk(chunk_id: str = "c1") -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        doc_title="Test Statute",
        doc_type=DocType.STATUTE,
        section_path="Section 1",
        text="A party must provide notice within 30 days.",
        retrieval_score=0.9,
        rerank_score=0.95,
    )


@pytest.fixture
def mock_settings():
    settings = SimpleNamespace(
        llm=SimpleNamespace(simple_model="gpt-4o-mini"),
        verifier=SimpleNamespace(max_tokens=500, temperature=0.0),
    )
    with patch("shafi.core.verifier.get_settings", return_value=settings):
        yield settings


def test_should_verify_no_citations(mock_settings) -> None:
    from shafi.core.verifier import AnswerVerifier

    verifier = AnswerVerifier(llm=MagicMock())
    assert verifier.should_verify("The limitation period is 6 years.", []) is True


def test_should_verify_with_citations(mock_settings) -> None:
    from shafi.core.verifier import AnswerVerifier

    verifier = AnswerVerifier(llm=MagicMock())
    assert verifier.should_verify("The limitation is 6 years (cite: c1).", ["c1"]) is False


def test_should_verify_strong_assertion_without_cite(mock_settings) -> None:
    from shafi.core.verifier import AnswerVerifier

    verifier = AnswerVerifier(llm=MagicMock())
    assert (
        verifier.should_verify(
            "The defendant must comply. The period is 6 years (cite: c1).",
            ["c1"],
        )
        is True
    )


def test_should_verify_all_assertions_cited(mock_settings) -> None:
    from shafi.core.verifier import AnswerVerifier

    verifier = AnswerVerifier(llm=MagicMock())
    assert (
        verifier.should_verify(
            "The party must comply (cite: c1). It is required under law (cite: c2).",
            ["c1", "c2"],
        )
        is False
    )


def test_should_verify_force_flag(mock_settings) -> None:
    from shafi.core.verifier import AnswerVerifier

    verifier = AnswerVerifier(llm=MagicMock())
    assert verifier.should_verify("Cited answer (cite: c1).", ["c1"], force=True) is True


@pytest.mark.asyncio
async def test_verify_grounded(mock_settings) -> None:
    from shafi.core.verifier import AnswerVerifier

    llm = MagicMock()
    llm.generate = AsyncMock(
        return_value=LLMResult(
            text='{"is_grounded": true, "unsupported_claims": [], "revised_answer": ""}',
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            model="gpt-4o-mini",
            latency_ms=12.0,
        )
    )

    verifier = AnswerVerifier(llm=llm)
    result = await verifier.verify("question", "answer (cite: c1)", [_chunk()])

    assert result.is_grounded is True
    assert result.unsupported_claims == []
    assert result.revised_answer == ""
    llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_verify_not_grounded_from_code_fence_json(mock_settings) -> None:
    from shafi.core.verifier import AnswerVerifier

    llm = MagicMock()
    llm.generate = AsyncMock(
        return_value=LLMResult(
            text=(
                "```json\n"
                '{"is_grounded": false, "unsupported_claims": ["invented fact"], '
                '"revised_answer": "corrected (cite: c1)"}\n'
                "```"
            ),
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            model="gpt-4o-mini",
            latency_ms=12.0,
        )
    )

    verifier = AnswerVerifier(llm=llm)
    result = await verifier.verify("question", "answer with lies", [_chunk()])

    assert result.is_grounded is False
    assert result.unsupported_claims == ["invented fact"]
    assert result.revised_answer == "corrected (cite: c1)"


@pytest.mark.asyncio
async def test_verify_handles_invalid_json(mock_settings) -> None:
    from shafi.core.verifier import AnswerVerifier

    llm = MagicMock()
    llm.generate = AsyncMock(
        return_value=LLMResult(
            text="not json",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            model="gpt-4o-mini",
            latency_ms=12.0,
        )
    )

    verifier = AnswerVerifier(llm=llm)
    result = await verifier.verify("q", "a", [_chunk()])

    assert result.is_grounded is True
    assert result.unsupported_claims == []
    assert result.revised_answer == ""
