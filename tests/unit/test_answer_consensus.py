"""Unit tests for the AnswerConsensus self-consistency voter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from shafi.core.pipeline.answer_consensus import (
    AnswerConsensus,
    _normalize_answer,
    _normalize_boolean,
    _normalize_number,
)


class TestNormalizeBoolean:
    def test_yes_variants(self) -> None:
        assert _normalize_boolean("Yes") == "Yes"
        assert _normalize_boolean("yes, because...") == "Yes"
        assert _normalize_boolean("True") == "Yes"
        assert _normalize_boolean("correct") == "Yes"
        assert _normalize_boolean("affirmative") == "Yes"

    def test_no_variants(self) -> None:
        assert _normalize_boolean("No") == "No"
        assert _normalize_boolean("no, the law...") == "No"
        assert _normalize_boolean("False") == "No"
        assert _normalize_boolean("incorrect") == "No"
        assert _normalize_boolean("negative") == "No"

    def test_ambiguous(self) -> None:
        assert _normalize_boolean("maybe") is None
        assert _normalize_boolean("") is None

    def test_yes_substring(self) -> None:
        assert _normalize_boolean("The answer is yes based on...") == "Yes"

    def test_no_word_boundary(self) -> None:
        assert _normalize_boolean("No person shall") == "No"


class TestNormalizeNumber:
    def test_simple_number(self) -> None:
        assert _normalize_number("5") == "5"

    def test_number_in_text(self) -> None:
        assert _normalize_number("The answer is 42 days") == "42"

    def test_number_with_commas(self) -> None:
        assert _normalize_number("10,000 dirhams") == "10,000"

    def test_no_number(self) -> None:
        assert _normalize_number("no number here") is None


class TestNormalizeAnswer:
    def test_boolean(self) -> None:
        assert _normalize_answer("Yes, because...", "boolean") == "Yes"

    def test_number(self) -> None:
        assert _normalize_answer("The fine is 5000", "number") == "5000"

    def test_name(self) -> None:
        assert _normalize_answer("Trust Law (cite: c0)", "name") == "Trust Law"

    def test_empty(self) -> None:
        assert _normalize_answer("", "boolean") is None


class TestAnswerConsensus:
    def _mock_generator(self, responses: list[str]) -> AsyncMock:
        gen = AsyncMock()
        gen.generate = AsyncMock(side_effect=[(r, None) for r in responses])
        return gen

    @pytest.mark.asyncio
    async def test_unanimous_high_confidence(self) -> None:
        gen = self._mock_generator(["Yes", "Yes, indeed", "Yes because..."])
        consensus = AnswerConsensus(gen)
        result = await consensus.vote(
            question="Is it required?",
            answer_type="boolean",
            chunks=["source text"],
        )
        assert result.confidence == "high"
        assert result.answer == "Yes"
        assert result.vote_count == 3

    @pytest.mark.asyncio
    async def test_majority_medium_confidence(self) -> None:
        gen = self._mock_generator(["Yes", "No", "Yes"])
        consensus = AnswerConsensus(gen)
        result = await consensus.vote(
            question="Is it required?",
            answer_type="boolean",
            chunks=["source text"],
        )
        assert result.confidence == "medium"
        assert result.answer == "Yes"
        assert result.vote_count == 2

    @pytest.mark.asyncio
    async def test_all_disagree_low_confidence(self) -> None:
        gen = self._mock_generator(["5", "10", "20"])
        consensus = AnswerConsensus(gen)
        result = await consensus.vote(
            question="How many?",
            answer_type="number",
            chunks=["source text"],
        )
        assert result.confidence == "low"

    @pytest.mark.asyncio
    async def test_generation_failure_handled(self) -> None:
        gen = AsyncMock()
        gen.generate = AsyncMock(side_effect=RuntimeError("LLM error"))
        consensus = AnswerConsensus(gen)
        result = await consensus.vote(
            question="test",
            answer_type="boolean",
            chunks=["text"],
        )
        assert result.confidence == "low"
        assert result.vote_count == 0

    @pytest.mark.asyncio
    async def test_number_consensus(self) -> None:
        gen = self._mock_generator(["42", "The answer is 42", "42 days"])
        consensus = AnswerConsensus(gen)
        result = await consensus.vote(
            question="How many days?",
            answer_type="number",
            chunks=["text"],
        )
        assert result.confidence == "high"
        assert result.answer == "42"
