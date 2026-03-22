"""Unit tests for UAE/DIFC strict support formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass

from rag_challenge.core.pipeline.support_formatting import coerce_strict_type_format


@dataclass
class _StubPipeline:
    """Minimal duck-typed pipeline for strict formatting tests."""

    def strict_type_fallback(self, answer_type: str, cited_ids: list[str] | tuple[str, ...]) -> str:
        _ = cited_ids
        return f"fallback:{answer_type}"

    def strict_type_citation_suffix(self, cited_ids: list[str] | tuple[str, ...]) -> str:
        _ = cited_ids
        return ""


def test_name_preserves_law_no_dot_for_numbered_titles() -> None:
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "This Law may be cited as the Trust Law No. 4 of 2018.",
        "name",
        [],
    )

    assert ok is True
    assert answer == "Trust Law No. 4 of 2018"


def test_name_accepts_numbered_regulations_titles() -> None:
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "The applicable instrument is Insolvency Regulations No 1 of 2022.",
        "name",
        [],
    )

    assert ok is True
    assert answer == "Insolvency Regulations No. 1 of 2022"


def test_name_normalizes_single_difc_case_reference() -> None:
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "Case reference: CFI/7/2024",
        "name",
        [],
    )

    assert ok is True
    assert answer == "CFI 007/2024"


def test_names_normalize_multiple_difc_case_references() -> None:
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "The names are: CFI/7/2024, CA 2/2023 and ENF-12/2022.",
        "names",
        [],
    )

    assert ok is True
    assert answer == "CFI 007/2024, CA 002/2023 and ENF 012/2022"


# ---------------------------------------------------------------------------
# Boolean hardening tests
# ---------------------------------------------------------------------------


def test_boolean_starts_with_yes() -> None:
    answer, ok = coerce_strict_type_format(_StubPipeline(), "Yes, the law requires it.", "boolean", [])
    assert ok is True
    assert answer == "Yes"


def test_boolean_starts_with_no() -> None:
    answer, ok = coerce_strict_type_format(_StubPipeline(), "No, it is not required.", "boolean", [])
    assert ok is True
    assert answer == "No"


def test_boolean_after_colon_yes() -> None:
    answer, ok = coerce_strict_type_format(_StubPipeline(), "The answer: yes", "boolean", [])
    assert ok is True
    assert answer == "Yes"


def test_boolean_after_colon_no() -> None:
    answer, ok = coerce_strict_type_format(_StubPipeline(), "Based on analysis: no", "boolean", [])
    assert ok is True
    assert answer == "No"


def test_boolean_true_synonym() -> None:
    answer, ok = coerce_strict_type_format(_StubPipeline(), "True, the requirement applies.", "boolean", [])
    assert ok is True
    assert answer == "Yes"


def test_boolean_false_synonym() -> None:
    answer, ok = coerce_strict_type_format(_StubPipeline(), "False, there is no such requirement.", "boolean", [])
    assert ok is True
    assert answer == "No"


def test_boolean_correct_synonym() -> None:
    answer, ok = coerce_strict_type_format(_StubPipeline(), "Correct, the law applies.", "boolean", [])
    assert ok is True
    assert answer == "Yes"


def test_boolean_incorrect_synonym() -> None:
    answer, ok = coerce_strict_type_format(_StubPipeline(), "Incorrect, that provision was repealed.", "boolean", [])
    assert ok is True
    assert answer == "No"


def test_boolean_word_boundary_yes() -> None:
    """'yes' as a word, not substring like 'yesterday'."""
    answer, ok = coerce_strict_type_format(_StubPipeline(), "It was resolved yesterday", "boolean", [])
    assert ok is False  # 'yesterday' should not match


def test_boolean_word_boundary_no() -> None:
    """'no' as a word, not substring like 'notable'."""
    answer, ok = coerce_strict_type_format(_StubPipeline(), "This is a notable achievement", "boolean", [])
    assert ok is False  # 'notable' should not match


# ---------------------------------------------------------------------------
# Name hardening tests
# ---------------------------------------------------------------------------


def test_name_preserves_legal_title_with_of() -> None:
    """'Trust Law of the Emirate of Dubai' should NOT be truncated at 'of'."""
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "Trust Law of the Emirate of Dubai",
        "name",
        [],
    )
    assert ok is True
    # Should preserve the full title, not truncate at "of"
    assert "Emirate" in answer or "Dubai" in answer


def test_name_preserves_regulation_of() -> None:
    """Legal titles with 'of' should be preserved."""
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "Employment Law of DIFC",
        "name",
        [],
    )
    assert ok is True
    assert "DIFC" in answer


def test_name_still_truncates_non_title_clauses() -> None:
    """Non-title text after 'subject to' should still be truncated."""
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "Ahmed Al Maktoum subject to the conditions of clause 5",
        "name",
        [],
    )
    assert ok is True
    assert "subject to" not in answer


def test_name_preserves_comma_with_law_number() -> None:
    """Don't split 'Trust Law, DIFC Law No. 4 of 2018' at the comma."""
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "Trust Law, DIFC Law No. 4 of 2018",
        "name",
        [],
    )
    assert ok is True
    # The numbered legal title regex should match the full title
    assert "No. 4 of 2018" in answer


# ---------------------------------------------------------------------------
# Number hardening tests
# ---------------------------------------------------------------------------


def test_number_prefers_time_when_question_asks_years() -> None:
    """When question asks about years, prefer time-context number."""
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "The penalty is 10,000 dirhams or 5 years imprisonment",
        "number",
        [],
        question="What is the prison term in years?",
    )
    assert ok is True
    assert answer == "5"


def test_number_prefers_money_when_question_asks_amount() -> None:
    """When question asks about fine/amount, prefer money-context number."""
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "The penalty is 10,000 dirhams or 5 years imprisonment",
        "number",
        [],
        question="What is the fine amount?",
    )
    assert ok is True
    assert answer == "10,000"


def test_number_default_first_match_when_no_question() -> None:
    """Without question context, return first valid number (backward compat)."""
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "The fee is 500 dirhams",
        "number",
        [],
    )
    assert ok is True
    assert answer == "500"


def test_number_skips_article_numbers() -> None:
    """Article/Section numbers should still be skipped."""
    answer, ok = coerce_strict_type_format(
        _StubPipeline(),
        "Article 5 prescribes a penalty of 1000 dirhams",
        "number",
        [],
    )
    assert ok is True
    assert answer == "1000"


def test_name_strips_leading_indefinite_article() -> None:
    """Regression 33060f26: EQA/LLM outputs 'a Confirmation Statement' — indefinite articles stripped."""
    for prefix in ("a ", "an ", "A ", "An "):
        answer, ok = coerce_strict_type_format(
            _StubPipeline(),
            f"{prefix}Confirmation Statement",
            "name",
            [],
        )
        assert ok is True
        assert answer == "Confirmation Statement", f"prefix={prefix!r} → {answer!r}"


def test_name_preserves_the_prefix() -> None:
    """Regression 5b78eff4: 'the Owner' must NOT be stripped — 'the' is part of proper name."""
    for name in ("the Owner", "the Claimant", "the Respondent", "The Company"):
        answer, ok = coerce_strict_type_format(_StubPipeline(), name, "name", [])
        assert ok is True
        assert answer == name, f"'the' was incorrectly stripped: {name!r} → {answer!r}"


def test_name_preserves_title_without_article() -> None:
    """Names without leading articles must not be affected by the strip."""
    for title in (
        "Employment Law Amendment Law DIFC Law No. 4 of 2021",
        "DIFC Non Profit Incorporated Organisations Law 2012",
        "CFI 010/2024",
    ):
        answer, ok = coerce_strict_type_format(_StubPipeline(), title, "name", [])
        assert ok is True
        assert answer == title, f"unexpected strip: {title!r} → {answer!r}"
