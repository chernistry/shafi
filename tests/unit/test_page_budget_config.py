"""Verify query_scope_classifier reads grounding_page_budget_default from settings."""

from unittest.mock import MagicMock

from shafi.core.grounding.query_scope_classifier import classify_query_scope


def _make_settings(budget: int) -> MagicMock:
    s = MagicMock()
    s.grounding_page_budget_default = budget
    return s


def test_single_field_uses_configured_budget_3() -> None:
    """A boolean/name question should use budget=3 when settings say so."""
    result = classify_query_scope("Who is the claimant?", "name", settings=_make_settings(3))
    assert result.page_budget == 3, f"Expected 3, got {result.page_budget}"


def test_single_field_uses_configured_budget_2() -> None:
    """Default budget=2 must be preserved when settings say 2."""
    result = classify_query_scope("What is the case number?", "number", settings=_make_settings(2))
    assert result.page_budget == 2


def test_no_settings_defaults_to_2() -> None:
    """Without settings, page_budget defaults to 2 (backward compat)."""
    result = classify_query_scope("What is the date?", "date")
    assert result.page_budget == 2


def test_anchor_query_uses_configured_budget() -> None:
    """Anchor-based SINGLE_FIELD_SINGLE_DOC should respect configured budget."""
    result = classify_query_scope("What does Article 5 say about overtime?", "free_text", settings=_make_settings(3))
    assert result.page_budget == 3


def test_costs_query_uses_configured_budget() -> None:
    """Costs/outcome queries should respect configured budget."""
    result = classify_query_scope("What costs were awarded?", "number", settings=_make_settings(3))
    assert result.page_budget == 3


def test_date_of_issue_query_uses_configured_budget() -> None:
    """Date-of-issue queries should respect configured budget."""
    result = classify_query_scope("What is the date of issue?", "date", settings=_make_settings(3))
    assert result.page_budget == 3


def test_explicit_page_always_budget_1() -> None:
    """Explicit page queries must always have budget=1 regardless of settings."""
    result = classify_query_scope("What is on page 5?", "free_text", settings=_make_settings(3))
    assert result.page_budget == 1


def test_full_case_always_budget_4() -> None:
    """Full-case queries must always have budget=4 regardless of settings."""
    result = classify_query_scope("Look through all documents for the outcome", "free_text", settings=_make_settings(3))
    assert result.page_budget == 4


def test_free_text_broad_always_budget_4() -> None:
    """Broad free_text without anchors always gets budget=4 regardless of settings."""
    result = classify_query_scope("Summarize the rights of employees", "free_text", settings=_make_settings(3))
    assert result.page_budget == 4
