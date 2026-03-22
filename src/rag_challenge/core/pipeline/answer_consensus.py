"""Self-consistency voting for ambiguous strict-type answers.

When the deterministic extractor (StrictAnswerer) is not confident, this
module generates the answer multiple times with varied prompts and returns
the majority vote.

Usage::

    consensus = AnswerConsensus(generator)
    result = await consensus.vote(
        question="Is a permit required?",
        answer_type="boolean",
        chunks=["chunk text ..."],
    )
    if result.confidence == "high":
        final_answer = result.answer

Integration point (for Agent 1):
    In generation_logic.py, after strict_answerer returns with ``.confident = False``
    AND ``answer_type in {"boolean", "number"}``, call::

        if pipeline_settings.enable_answer_consensus:
            consensus = AnswerConsensus(generator)
            result = await consensus.vote(
                question=state["query"],
                answer_type=answer_type,
                chunks=[c.text for c in context_chunks],
                model=state["model"],
                collector=collector,
            )
            if result.confidence in ("high", "medium"):
                answer = result.answer

    Gate behind ``PIPELINE_ENABLE_ANSWER_CONSENSUS`` setting (default False).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_challenge.llm.generator import RAGGenerator

logger = logging.getLogger(__name__)


_PROMPT_VARIANTS = [
    # Variant 0: Standard (no prefix)
    "",
    # Variant 1: Chain-of-thought
    "Think step by step before answering. ",
    # Variant 2: Evidence-grounded
    "Quote the exact text that supports your answer, then give the answer. ",
]


@dataclass
class ConsensusResult:
    """Outcome of a self-consistency vote."""

    answer: str
    """The majority-voted answer."""

    confidence: str
    """'high' (3/3), 'medium' (2/3), 'low' (all disagree)."""

    vote_count: int
    """Number of votes for the winning answer."""

    total_votes: int
    """Total number of votes cast."""

    all_answers: list[str] = field(default_factory=lambda: [])
    """Raw answers from each variant (for debugging)."""

    reason: str = ""
    """Human-readable explanation of the vote outcome."""


def _normalize_boolean(text: str) -> str | None:
    """Normalize a boolean-ish answer to 'Yes' or 'No'."""
    lowered = text.strip().lower()
    if lowered.startswith("yes") or lowered in ("true", "correct", "affirmative"):
        return "Yes"
    if re.match(r"no(?:\b|$)", lowered) or lowered in ("false", "incorrect", "negative"):
        return "No"
    if "yes" in lowered and "no" not in lowered:
        return "Yes"
    if "no" in lowered and "yes" not in lowered:
        return "No"
    return None


def _normalize_number(text: str) -> str | None:
    """Extract the first number from a text answer."""
    match = re.search(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return match.group(0) if match else None


def _normalize_answer(text: str, answer_type: str) -> str | None:
    """Normalize an answer for comparison based on type."""
    kind = answer_type.strip().lower()
    if kind == "boolean":
        return _normalize_boolean(text)
    if kind == "number":
        return _normalize_number(text)
    if kind in ("name", "date"):
        cleaned = text.strip()
        # Remove citation suffixes
        cleaned = re.sub(r"\s*\(cite:[^)]+\)", "", cleaned).strip()
        return cleaned if cleaned else None
    return text.strip() or None


class AnswerConsensus:
    """Generate multiple answers and vote on the majority.

    Args:
        generator: The RAG generator to use for answer generation.
    """

    def __init__(self, generator: RAGGenerator) -> None:
        self._generator = generator

    async def vote(
        self,
        question: str,
        answer_type: str,
        chunks: list[str] | list[Any],
        *,
        n_votes: int = 3,
        model: str = "",
        collector: Any = None,
    ) -> ConsensusResult:
        """Generate n answers with different prompts and vote.

        Args:
            question: The question to answer.
            answer_type: Expected answer type (boolean, number, etc.).
            chunks: Source chunks (text strings or RankedChunk objects).
            n_votes: Number of answer variants to generate (default 3).
            model: LLM model to use.
            collector: Optional telemetry collector.

        Returns:
            ConsensusResult with the majority answer and confidence.
        """
        raw_answers: list[str] = []
        normalized_answers: list[str] = []

        for i in range(min(n_votes, len(_PROMPT_VARIANTS))):
            prefix = _PROMPT_VARIANTS[i]
            prompt_hint = f"{prefix}Answer with ONLY the {answer_type}. No explanation." if prefix else ""

            try:
                text, _ = await self._generator.generate(
                    question,
                    chunks,  # type: ignore[arg-type]  # chunks may be str list at runtime
                    model=model,
                    max_tokens=64,
                    collector=collector,
                    answer_type=answer_type,
                    prompt_hint=prompt_hint,
                )
            except Exception:
                logger.warning("Consensus vote %d failed", i, exc_info=True)
                continue

            raw_answers.append(text.strip())
            norm = _normalize_answer(text, answer_type)
            if norm is not None:
                normalized_answers.append(norm)

        if not normalized_answers:
            return ConsensusResult(
                answer="",
                confidence="low",
                vote_count=0,
                total_votes=len(raw_answers),
                all_answers=raw_answers,
                reason="no valid answers generated",
            )

        # Count votes
        counts = Counter(normalized_answers)
        winner, winner_count = counts.most_common(1)[0]
        total = len(normalized_answers)

        if winner_count == total:
            confidence = "high"
            reason = f"all {total} votes agree"
        elif winner_count > total / 2:
            confidence = "medium"
            reason = f"{winner_count}/{total} votes agree"
        else:
            confidence = "low"
            reason = f"no majority — {dict(counts)}"

        logger.info(
            "Consensus vote: answer=%r confidence=%s votes=%d/%d raw=%r",
            winner, confidence, winner_count, total, raw_answers,
        )

        return ConsensusResult(
            answer=winner,
            confidence=confidence,
            vote_count=winner_count,
            total_votes=total,
            all_answers=raw_answers,
            reason=reason,
        )
