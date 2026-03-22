"""Unit tests for free-text answer quality improvements."""

from __future__ import annotations

from rag_challenge.core.pipeline.free_text_cleanup import (
    condense_free_text,
    strip_trailing_filler,
    strip_verbose_preamble,
)


class TestStripVerbosePreamble:
    def test_strips_according_to(self) -> None:
        text = "According to Article 5 of the Trust Law, the trustee must act in good faith."
        result = strip_verbose_preamble(text)
        assert result == "The trustee must act in good faith."

    def test_strips_based_on(self) -> None:
        text = "Based on the provided sources, the penalty is 10,000 dirhams."
        result = strip_verbose_preamble(text)
        assert result == "The penalty is 10,000 dirhams."

    def test_preserves_under_article(self) -> None:
        """Under [Article X], is evidence-first — must NOT be stripped."""
        text = "Under Article 8(1) of the Operating Law, companies must register."
        result = strip_verbose_preamble(text)
        assert result == text

    def test_strips_in_accordance_with(self) -> None:
        text = "In accordance with the provisions of DIFC Law No. 4, trusts are permitted."
        result = strip_verbose_preamble(text)
        assert result == "Trusts are permitted."

    def test_preserves_pursuant_to(self) -> None:
        """Pursuant to [X], is evidence-first — must NOT be stripped."""
        text = "Pursuant to Schedule 3, the fee is 500 dirhams."
        result = strip_verbose_preamble(text)
        assert result == text

    def test_strips_the_answer_is(self) -> None:
        text = "The answer is that the law requires registration."
        result = strip_verbose_preamble(text)
        assert result == "The law requires registration."

    def test_preserves_short_text(self) -> None:
        """Don't strip if remainder would be too short."""
        text = "According to the law, yes."
        result = strip_verbose_preamble(text)
        # "yes." is only 4 chars — too short, should preserve original
        assert result == text

    def test_preserves_non_preamble(self) -> None:
        text = "The trustee must act in good faith."
        result = strip_verbose_preamble(text)
        assert result == text

    def test_handles_empty(self) -> None:
        assert strip_verbose_preamble("") == ""

    def test_capitalizes_after_strip(self) -> None:
        text = "According to the law, the requirement applies to all entities."
        result = strip_verbose_preamble(text)
        assert result[0].isupper()


class TestStripTrailingFiller:
    def test_strips_as_per_sources(self) -> None:
        text = "The penalty is 10,000 dirhams as per the provided sources."
        result = strip_trailing_filler(text)
        assert "as per" not in result
        assert result == "The penalty is 10,000 dirhams"

    def test_strips_according_to_sources(self) -> None:
        text = "Registration is required, according to the available sources."
        result = strip_trailing_filler(text)
        assert "according to" not in result

    def test_strips_based_on_context(self) -> None:
        text = "The fee is 500 dirhams based on the provided context."
        result = strip_trailing_filler(text)
        assert "based on" not in result

    def test_preserves_clean_text(self) -> None:
        text = "The trustee must act in good faith."
        result = strip_trailing_filler(text)
        assert result == text

    def test_preserves_text_without_period(self) -> None:
        text = "The penalty is 10,000 dirhams"
        result = strip_trailing_filler(text)
        assert result == text


class TestCondenseFreeText:
    def test_full_pipeline(self) -> None:
        text = "According to Article 5, the trustee must act in good faith, as per the provided sources."
        result = condense_free_text(text)
        assert "According to" not in result
        assert "as per" not in result
        assert "trustee" in result

    def test_preserves_content(self) -> None:
        text = "The DIFC Employment Law No. 4 of 2005 governs employee relations."
        result = condense_free_text(text)
        assert result == text

    def test_normalizes_whitespace(self) -> None:
        text = "The  penalty   is  10,000  dirhams."
        result = condense_free_text(text)
        assert "  " not in result

    def test_empty(self) -> None:
        assert condense_free_text("") == ""
