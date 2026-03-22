# pyright: reportPrivateUsage=false
"""Answer quality gate — single entry point for post-coercion validation.

Wraps the AnswerValidator (deterministic signal check) and AnswerConsensus
(multi-vote self-consistency) behind a clean interface that the pipeline can
call in a single line.

Integration point
-----------------
In ``generation_logic.py``, inside the ``if answer_type in strict_types:``
block, **after coercion + repair have produced a final ``answer``** (line ~448,
right before the ``if not answer:`` fallback on line 449).

Insert these lines:

.. code-block:: python

    # --- NOGA: Answer quality gate (2-line insertion) ---
    from .answer_quality_gate import run_answer_quality_gate

    answer, _quality_report = run_answer_quality_gate(
        question=state["query"],
        answer=answer,
        answer_type=answer_type,
        source_chunks=[c.text for c in context_chunks],
        settings=self._settings.pipeline,
        extracted=extracted,
    )

The function is safe to call unconditionally — it no-ops when both feature
flags are off (``enable_answer_validation`` and ``enable_answer_consensus``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shafi.config.settings import PipelineSettings

from .answer_validator import AnswerValidator, ValidationResult

logger = logging.getLogger(__name__)

_STRICT_TYPES = frozenset({"boolean", "number", "date", "name", "names"})


@dataclass
class QualityReport:
    """Diagnostic payload returned alongside the (possibly corrected) answer."""

    validation: ValidationResult | None = None
    consensus_used: bool = False
    consensus_answer: str | None = None
    consensus_confidence: str | None = None
    original_answer: str | None = None
    corrections: list[str] = field(default_factory=lambda: [])


def run_answer_quality_gate(
    *,
    question: str,
    answer: str,
    answer_type: str,
    source_chunks: list[str],
    settings: PipelineSettings,
    extracted: bool = False,
) -> tuple[str, QualityReport]:
    """Run post-coercion quality checks on a strict-type answer.

    Parameters
    ----------
    question:
        The original user question.
    answer:
        The coerced/repaired answer string.
    answer_type:
        One of boolean, number, date, name, names, free_text.
    source_chunks:
        Text bodies of the source chunks used for answering.
    settings:
        Pipeline settings (checked for feature flags).
    extracted:
        True if the strict_answerer already validated with confidence.
        When True, validation is lighter (skip signal-based boolean flip).

    Returns
    -------
    tuple of (final_answer, QualityReport)
        The answer may be corrected if validation detects an inconsistency.
    """
    kind = answer_type.strip().lower()
    report = QualityReport(original_answer=answer)

    # Gate: nothing to do for non-strict types
    if kind not in _STRICT_TYPES:
        return answer, report

    # Gate: skip for unanswerable/null answers
    normalized = answer.strip().lower()
    if normalized in {"null", "none", ""} or "insufficient sources" in normalized:
        return answer, report

    validation_enabled = bool(getattr(settings, "enable_answer_validation", False))
    consensus_enabled = bool(getattr(settings, "enable_answer_consensus", False))

    if not validation_enabled and not consensus_enabled:
        return answer, report

    # --- Step 1: Deterministic validation ---
    if validation_enabled:
        validator = AnswerValidator()
        result = validator.validate(
            question=question,
            answer=answer,
            answer_type=kind,
            source_chunks=source_chunks,
        )
        report.validation = result

        if not result.is_valid and result.suggested_answer:  # noqa: SIM102
            # Only apply correction when confidence is reasonable and the
            # strict_answerer didn't already validate this answer.
            if not extracted and result.confidence >= 0.6:
                logger.info(
                    "answer_quality_gate: correcting answer for qid via validation: %s -> %s (reason: %s)",
                    repr(answer[:60]),
                    repr(result.suggested_answer[:60]),
                    result.reason,
                )
                report.corrections.append(f"validator: {answer!r} -> {result.suggested_answer!r} ({result.reason})")
                answer = result.suggested_answer

    # --- Step 2: Consensus voting (async — skipped in synchronous gate) ---
    # NOTE: AnswerConsensus requires async LLM calls. This synchronous gate
    # cannot invoke it directly. Instead, we expose the decision logic here
    # so the pipeline can call consensus separately if needed.
    #
    # The pipeline should check:
    #   if report.should_run_consensus(extracted, settings):
    #       result = await consensus.vote(...)
    #       answer = result.answer if result.confidence != "low" else answer
    #
    # For now, consensus is documented but wired by the async pipeline layer.

    return answer, report


def should_run_consensus(
    *,
    answer_type: str,
    extracted: bool,
    settings: PipelineSettings,
    validation_result: ValidationResult | None = None,
) -> bool:
    """Decide whether consensus voting should be triggered.

    Returns True when:
    - ``enable_answer_consensus`` is on
    - The strict_answerer was NOT confident (``extracted is False``)
    - The answer type is boolean or number (highest ROI for voting)
    - Optionally: validation flagged an issue (boosts need for second opinion)
    """
    if not bool(getattr(settings, "enable_answer_consensus", False)):
        return False
    if extracted:
        return False
    kind = answer_type.strip().lower()
    if kind not in {"boolean", "number"}:
        return False
    # If validation already corrected, consensus is less critical
    return not (validation_result and validation_result.is_valid)
