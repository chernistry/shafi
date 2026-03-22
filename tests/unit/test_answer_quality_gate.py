"""Unit tests for the answer quality gate wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from shafi.core.pipeline.answer_quality_gate import (
    run_answer_quality_gate,
    should_run_consensus,
)


@dataclass
class _StubSettings:
    """Minimal duck-typed PipelineSettings for gate tests."""

    enable_answer_validation: bool = False
    enable_answer_consensus: bool = False


# ---------------------------------------------------------------------------
# run_answer_quality_gate
# ---------------------------------------------------------------------------


class TestRunAnswerQualityGate:
    def test_noop_when_both_flags_off(self) -> None:
        settings = _StubSettings()
        answer, report = run_answer_quality_gate(
            question="Is it required?",
            answer="Yes",
            answer_type="boolean",
            source_chunks=["No person shall be required."],
            settings=settings,  # type: ignore[arg-type]
        )
        assert answer == "Yes"
        assert report.validation is None

    def test_noop_for_free_text(self) -> None:
        settings = _StubSettings(enable_answer_validation=True)
        answer, report = run_answer_quality_gate(
            question="Explain the law.",
            answer="The law covers...",
            answer_type="free_text",
            source_chunks=["text"],
            settings=settings,  # type: ignore[arg-type]
        )
        assert answer == "The law covers..."
        assert report.validation is None

    def test_noop_for_null_answer(self) -> None:
        settings = _StubSettings(enable_answer_validation=True)
        answer, report = run_answer_quality_gate(
            question="What?",
            answer="null",
            answer_type="name",
            source_chunks=["text"],
            settings=settings,  # type: ignore[arg-type]
        )
        assert answer == "null"
        assert report.validation is None

    def test_validation_corrects_boolean(self) -> None:
        settings = _StubSettings(enable_answer_validation=True)
        answer, report = run_answer_quality_gate(
            question="Is registration required?",
            answer="No",
            answer_type="boolean",
            source_chunks=["Every entity is required to register with the authority."],
            settings=settings,  # type: ignore[arg-type]
        )
        assert answer == "Yes"
        assert report.validation is not None
        assert report.validation.is_valid is False
        assert len(report.corrections) == 1

    def test_validation_preserves_correct_answer(self) -> None:
        settings = _StubSettings(enable_answer_validation=True)
        answer, report = run_answer_quality_gate(
            question="Is registration required?",
            answer="Yes",
            answer_type="boolean",
            source_chunks=["Every entity is required to register with the authority."],
            settings=settings,  # type: ignore[arg-type]
        )
        assert answer == "Yes"
        assert report.validation is not None
        assert report.validation.is_valid is True
        assert len(report.corrections) == 0

    def test_skips_correction_when_extracted_confident(self) -> None:
        """When strict_answerer was confident, don't override its answer."""
        settings = _StubSettings(enable_answer_validation=True)
        answer, _report = run_answer_quality_gate(
            question="Is a permit required?",
            answer="Yes",
            answer_type="boolean",
            source_chunks=["Permits are not issued by the authority."],
            settings=settings,  # type: ignore[arg-type]
            extracted=True,
        )
        # Even if validation flags it, extracted=True prevents correction
        assert answer == "Yes"

    def test_report_tracks_original(self) -> None:
        settings = _StubSettings(enable_answer_validation=True)
        _, report = run_answer_quality_gate(
            question="Is registration required?",
            answer="No",
            answer_type="boolean",
            source_chunks=["Every entity is required to register with the authority."],
            settings=settings,  # type: ignore[arg-type]
        )
        assert report.original_answer == "No"

    def test_number_validation_runs(self) -> None:
        settings = _StubSettings(enable_answer_validation=True)
        _answer, report = run_answer_quality_gate(
            question="What is the term in years?",
            answer="10000",
            answer_type="number",
            source_chunks=["The penalty is 10,000 dirhams or 5 years."],
            settings=settings,  # type: ignore[arg-type]
        )
        assert report.validation is not None
        assert report.validation.is_valid is False

    def test_name_validation_runs(self) -> None:
        settings = _StubSettings(enable_answer_validation=True)
        _answer, report = run_answer_quality_gate(
            question="What is the name of the law?",
            answer="Trust Law",
            answer_type="name",
            source_chunks=["The Trust Law governs fiduciary duties."],
            settings=settings,  # type: ignore[arg-type]
        )
        assert report.validation is not None
        assert report.validation.is_valid is True


# ---------------------------------------------------------------------------
# should_run_consensus
# ---------------------------------------------------------------------------


class TestShouldRunConsensus:
    def test_false_when_flag_off(self) -> None:
        settings = _StubSettings(enable_answer_consensus=False)
        assert (
            should_run_consensus(
                answer_type="boolean",
                extracted=False,
                settings=settings,  # type: ignore[arg-type]
            )
            is False
        )

    def test_false_when_extracted(self) -> None:
        settings = _StubSettings(enable_answer_consensus=True)
        assert (
            should_run_consensus(
                answer_type="boolean",
                extracted=True,
                settings=settings,  # type: ignore[arg-type]
            )
            is False
        )

    def test_false_for_date_type(self) -> None:
        settings = _StubSettings(enable_answer_consensus=True)
        assert (
            should_run_consensus(
                answer_type="date",
                extracted=False,
                settings=settings,  # type: ignore[arg-type]
            )
            is False
        )

    def test_true_for_boolean_unextracted(self) -> None:
        settings = _StubSettings(enable_answer_consensus=True)
        assert (
            should_run_consensus(
                answer_type="boolean",
                extracted=False,
                settings=settings,  # type: ignore[arg-type]
            )
            is True
        )

    def test_true_for_number_unextracted(self) -> None:
        settings = _StubSettings(enable_answer_consensus=True)
        assert (
            should_run_consensus(
                answer_type="number",
                extracted=False,
                settings=settings,  # type: ignore[arg-type]
            )
            is True
        )
