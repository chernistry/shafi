"""Regression tests for generator answer cleanup functions."""

from __future__ import annotations

from rag_challenge.llm.generator_cleanup import (
    cleanup_final_answer,
    cleanup_truncated_answer,
    looks_like_truncated_tail,
)


# ── looks_like_truncated_tail ──


def test_truncated_tail_short_trailing_preposition() -> None:
    assert looks_like_truncated_tail("The court held that the") is True


def test_truncated_tail_very_short_of() -> None:
    assert looks_like_truncated_tail("provision of") is True


def test_truncated_tail_long_valid_text_ending_with_of() -> None:
    # Valid legal title ending with "of" — NOT truncated.
    assert looks_like_truncated_tail("Law No. 7 of 2019") is False


def test_truncated_tail_valid_enumerated_item_ending_with_of() -> None:
    assert looks_like_truncated_tail(
        "Application of Civil and Commercial Laws in the DIFC"
    ) is False


def test_truncated_tail_valid_text_ending_with_and() -> None:
    assert looks_like_truncated_tail(
        "Partners are jointly and severally liable"
    ) is False


def test_truncated_tail_valid_text_ending_with_the() -> None:
    assert looks_like_truncated_tail(
        "This law is administered by the Registrar of the"
    ) is False


def test_truncated_tail_text_with_proper_ending() -> None:
    assert looks_like_truncated_tail("The court ruled in favor.") is False


def test_truncated_tail_text_ending_with_citation() -> None:
    assert looks_like_truncated_tail("decided by DIFC Courts (cite: abc123)") is False


def test_truncated_tail_unmatched_open_paren() -> None:
    assert looks_like_truncated_tail("see Section 5(a") is True


def test_truncated_tail_law_no_standalone() -> None:
    # "Law No." ends with a period, so the regex treats it as terminated.
    # "Law No" (no period) is the truly orphaned form.
    assert looks_like_truncated_tail("Law No.") is False
    assert looks_like_truncated_tail("Law No") is False  # only 2 words, but no trailing prep


def test_truncated_tail_law_no_with_number() -> None:
    # "DIFC Law No. 5 of 2007" — complete reference, NOT truncated.
    assert looks_like_truncated_tail("DIFC Law No. 5 of 2007") is False


def test_truncated_tail_empty() -> None:
    assert looks_like_truncated_tail("") is False


def test_truncated_tail_medium_fragment_with_comma() -> None:
    # Medium text with internal punctuation ending with "the" — not truncated.
    assert looks_like_truncated_tail("According to Schedule 2, the") is False


# ── cleanup_truncated_answer ──


def test_cleanup_preserves_numbered_list_ending_with_preposition() -> None:
    answer = (
        "1. Law No. 7 of 2019\n"
        "2. Application of Civil and Commercial Laws in the DIFC\n"
        "3. Law No. 12 of 2004"
    )
    result = cleanup_truncated_answer(answer)
    assert "Law No. 7 of 2019" in result
    assert "Application of Civil and Commercial Laws in the DIFC" in result
    assert "Law No. 12 of 2004" in result


def test_cleanup_strips_orphaned_cite_from_numbered_list() -> None:
    answer = "1. The Trust Law 2018\n2. (cite: abc12"
    result = cleanup_truncated_answer(answer)
    assert "(cite:" not in result
    assert "Trust Law 2018" in result


def test_cleanup_truncated_genuinely_broken_text() -> None:
    # A single truncated sentence without a prior sentence boundary stays as-is —
    # cleanup only clips fragments AFTER a complete sentence.
    answer = "The court held that the"
    result = cleanup_truncated_answer(answer)
    assert result == answer


def test_cleanup_truncated_after_complete_sentence() -> None:
    # Fragment after a complete sentence IS cleaned.
    answer = "The court ruled in favor. The provisions of"
    result = cleanup_truncated_answer(answer)
    assert result.rstrip().endswith("favor.")


# ── cleanup_final_answer ──


def test_final_cleanup_preserves_valid_enumerated_answer() -> None:
    answer = (
        "1. Employment Law, DIFC Law No. 4 of 2005 (cite: abc123)\n"
        "2. Contract Law, DIFC Law No. 6 of 2004 (cite: def456)"
    )
    result = cleanup_final_answer(answer)
    assert "Employment Law" in result
    assert "Contract Law" in result
    assert "Law No. 4 of 2005" in result
    assert "Law No. 6 of 2004" in result


def test_final_cleanup_removes_empty_trailing_item() -> None:
    answer = "1. First item.\n2. Second item.\n3. "
    result = cleanup_final_answer(answer)
    assert "First item" in result
    assert result.rstrip().endswith(".")
