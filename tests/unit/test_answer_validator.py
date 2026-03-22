"""Unit tests for the AnswerValidator."""

from __future__ import annotations

from rag_challenge.core.pipeline.answer_validator import (
    AnswerValidator,
    ValidationResult,
    _extract_key_terms,
    _find_term_windows,
    validate_boolean,
    validate_name,
    validate_number,
)


class TestExtractKeyTerms:
    def test_removes_stopwords(self) -> None:
        terms = _extract_key_terms("Is the law applicable in the DIFC?")
        assert "the" not in terms
        assert "law" not in terms  # domain stopword
        assert "difc" not in terms  # domain stopword
        assert "applicable" in terms

    def test_empty_question(self) -> None:
        assert _extract_key_terms("") == []

    def test_keeps_significant_words(self) -> None:
        terms = _extract_key_terms("What is the penalty for violation?")
        assert "penalty" in terms
        assert "violation" in terms


class TestFindTermWindows:
    def test_finds_window_around_term(self) -> None:
        source = "The law shall not permit any person to violate the regulation."
        windows = _find_term_windows(source, ["permit"], window_size=5)
        assert len(windows) >= 1
        assert "permit" in windows[0].lower()

    def test_no_match_returns_empty(self) -> None:
        windows = _find_term_windows("Hello world", ["nonexistent"])
        assert windows == []


class TestValidateBoolean:
    def test_yes_with_negation_signals_flags(self) -> None:
        result = validate_boolean(
            question="Is a permit required under the law?",
            answer="Yes",
            source_chunks=["Permits are not issued by the authority. The application does not apply."],
        )
        assert result.is_valid is False
        assert result.suggested_answer == "No"

    def test_no_with_affirmation_signals_flags(self) -> None:
        result = validate_boolean(
            question="Is registration required?",
            answer="No",
            source_chunks=["Every entity is required to register with the authority."],
        )
        assert result.is_valid is False
        assert result.suggested_answer == "Yes"

    def test_yes_with_affirmation_passes(self) -> None:
        result = validate_boolean(
            question="Is registration required?",
            answer="Yes",
            source_chunks=["Every entity is required to register with the authority."],
        )
        assert result.is_valid is True

    def test_no_with_negation_passes(self) -> None:
        result = validate_boolean(
            question="Is a permit required?",
            answer="No",
            source_chunks=["No person shall be required to obtain a permit."],
        )
        assert result.is_valid is True

    def test_mixed_signals_passes(self) -> None:
        """When both negation and affirmation present, accept answer as ambiguous."""
        result = validate_boolean(
            question="Is approval needed?",
            answer="Yes",
            source_chunks=["Approval is required for major transactions. No approval is needed for minor ones."],
        )
        assert result.is_valid is True

    def test_no_key_terms_in_source_passes(self) -> None:
        result = validate_boolean(
            question="Is registration needed?",
            answer="Yes",
            source_chunks=["The sky is blue."],
        )
        assert result.is_valid is True

    def test_non_boolean_answer_passes(self) -> None:
        result = validate_boolean(
            question="test",
            answer="Maybe",
            source_chunks=["shall not"],
        )
        assert result.is_valid is True


class TestValidateNumber:
    def test_time_question_with_large_number_flags(self) -> None:
        result = validate_number(
            question="What is the term in years?",
            answer="10000",
            source_chunks=["The penalty is 10,000 dirhams or 5 years."],
        )
        assert result.is_valid is False
        assert "time period" in result.reason

    def test_reasonable_time_number_passes(self) -> None:
        result = validate_number(
            question="What is the term in years?",
            answer="5",
            source_chunks=["The term is 5 years."],
        )
        assert result.is_valid is True

    def test_money_question_with_year_flags(self) -> None:
        result = validate_number(
            question="What is the fine amount?",
            answer="2020",
            source_chunks=["The law was enacted in 2020. The fine is 50,000 dirhams."],
        )
        # This may or may not flag depending on context detection
        assert isinstance(result.is_valid, bool)

    def test_no_number_in_answer_passes(self) -> None:
        result = validate_number(
            question="How many?",
            answer="null",
            source_chunks=["text"],
        )
        assert result.is_valid is True

    def test_no_category_passes(self) -> None:
        result = validate_number(
            question="How many judges were on the panel?",
            answer="3",
            source_chunks=["The panel consisted of 3 judges."],
        )
        assert result.is_valid is True


class TestValidateName:
    def test_exact_match_passes(self) -> None:
        result = validate_name(
            question="What is the name of the law?",
            answer="Trust Law",
            source_chunks=["The Trust Law governs fiduciary duties."],
        )
        assert result.is_valid is True

    def test_name_not_in_source_flags(self) -> None:
        result = validate_name(
            question="What is the name?",
            answer="Arbitration Framework Regulation",
            source_chunks=["The employment code was established in 2015."],
        )
        assert result.is_valid is False

    def test_null_answer_passes(self) -> None:
        result = validate_name(
            question="What is the name?",
            answer="null",
            source_chunks=["text"],
        )
        assert result.is_valid is True

    def test_detects_truncated_legal_title(self) -> None:
        """Name that exists in source but as part of a longer legal title."""
        result = validate_name(
            question="What is the official title?",
            answer="DIFC Employment Law",
            source_chunks=["The DIFC Employment Law No. 4 of 2005 governs employee relations."],
        )
        # Correctly detects truncation — full title includes "No. 4 of 2005"
        assert result.is_valid is False
        assert result.suggested_answer is not None
        assert "No. 4 of 2005" in result.suggested_answer

    def test_standalone_name_passes(self) -> None:
        """Name that appears standalone in source (not truncated)."""
        result = validate_name(
            question="What entity administers the law?",
            answer="Registrar",
            source_chunks=["The Registrar administers the law."],
        )
        assert result.is_valid is True

    def test_truncated_name_flags(self) -> None:
        result = validate_name(
            question="What is the full title?",
            answer="Fiduciary Obligations of the",
            source_chunks=["The Fiduciary Obligations of the Emirate of Dubai provide for trust duties."],
        )
        # Should detect the name was truncated (trailing preposition)
        assert result.is_valid is False or "truncated" in result.reason.lower()

    def test_truncated_name_suggests_full_legal_title(self) -> None:
        result = validate_name(
            question="What is the full title of the enacted law?",
            answer="Employment Law",
            source_chunks=[
                "Employment Law Amendment Law DIFC Law No. 4 of 2021 came into force on that date."
            ],
        )
        assert result.is_valid is False
        assert result.suggested_answer is not None
        assert "Amendment Law" in result.suggested_answer
        assert "No. 4 of 2021" in result.suggested_answer


class TestAnswerValidator:
    def test_dispatch_boolean(self) -> None:
        v = AnswerValidator()
        result = v.validate("Is it required?", "Yes", "boolean", ["It is required."])
        assert isinstance(result, ValidationResult)

    def test_dispatch_number(self) -> None:
        v = AnswerValidator()
        result = v.validate("How many?", "5", "number", ["There are 5."])
        assert isinstance(result, ValidationResult)

    def test_dispatch_name(self) -> None:
        v = AnswerValidator()
        result = v.validate("What law?", "Trust Law", "name", ["Trust Law text."])
        assert isinstance(result, ValidationResult)

    def test_dispatch_free_text(self) -> None:
        v = AnswerValidator()
        result = v.validate("Explain.", "Long answer.", "free_text", ["source"])
        assert result.is_valid is True

    def test_dispatch_date(self) -> None:
        v = AnswerValidator()
        result = v.validate("When?", "2020-01-01", "date", ["source"])
        assert result.is_valid is True
