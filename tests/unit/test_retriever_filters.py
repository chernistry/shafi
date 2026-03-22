"""Tests for deterministic retriever filter and citation variant helpers."""

from __future__ import annotations

from shafi.core.retriever_filters import expand_doc_ref_variants


def test_expand_doc_ref_variants_supports_numbered_instruments_across_jurisdictions() -> None:
    variants = expand_doc_ref_variants(("DIFC Regulations No. 1 of 2020", "DFSA Rules No. 3 of 2024"))

    assert "DIFC Regulations No. 1 of 2020" in variants
    assert "Regulations No. 1 of 2020" in variants
    assert "DIFC Regulations No 1 of 2020" in variants
    assert "DFSA Rules No. 3 of 2024" in variants
    assert "Rules No. 3 of 2024" in variants
    assert "DFSA Rules No 3 of 2024" in variants


def test_expand_doc_ref_variants_keeps_law_prefix_variants_intact() -> None:
    variants = expand_doc_ref_variants(("UAE Law No. 12 of 2021",))

    assert "Law No. 12 of 2021" in variants
    assert "UAE Law No. 12 of 2021" in variants
    assert "UAE Law No 12 of 2021" in variants


def test_build_filter_must_not_doc_ids_excludes_documents() -> None:
    from shafi.core.retriever_filters import build_filter

    result = build_filter(
        doc_type_filter=None,
        jurisdiction_filter=None,
        must_not_doc_ids=["doc_a", "doc_b"],
    )

    assert result is not None
    assert result.must_not is not None
    assert len(result.must_not) == 1
    condition = result.must_not[0]
    assert hasattr(condition, "key")
    assert condition.key == "doc_id"  # type: ignore[union-attr]


def test_build_filter_no_must_not_when_empty() -> None:
    from shafi.core.retriever_filters import build_filter

    result = build_filter(
        doc_type_filter=None,
        jurisdiction_filter=None,
        must_not_doc_ids=None,
    )

    assert result is None


def test_build_filter_case_ref_includes_doc_title_match_text() -> None:
    """Case ref doc_refs must include MatchText on doc_title for full-doc coverage.

    Chunks without the case ref in their citations field must still be
    retrievable via the doc_title substring match.
    """
    from qdrant_client import models

    from shafi.core.retriever_filters import build_filter

    result = build_filter(
        doc_type_filter=None,
        jurisdiction_filter=None,
        doc_refs=["CFI 057/2025"],
    )

    assert result is not None
    # Extract the should conditions from the must filter
    should_filter = next(
        f
        for f in result.must
        if isinstance(f, models.Filter)  # type: ignore[union-attr]
    )
    assert should_filter.should is not None
    # At least one MatchText condition on doc_title for CFI 057/2025
    match_text_conditions = [
        c
        for c in should_filter.should
        if hasattr(c, "key")
        and c.key == "doc_title"  # type: ignore[union-attr]
        and hasattr(c, "match")
        and isinstance(c.match, models.MatchText)  # type: ignore[union-attr]
        and c.match.text == "CFI 057/2025"  # type: ignore[union-attr]
    ]
    assert len(match_text_conditions) == 1, "Expected exactly one MatchText condition on doc_title for CFI 057/2025"


def test_build_filter_case_ref_canonical_only_match_text() -> None:
    """Case ref filter uses only the canonical slash-year MatchText form.

    "/" is a word boundary in Qdrant's text tokenizer, so MatchText("SCT 415/2024")
    tokenises to ["SCT","415","2024"] and matches both slash-year titles
    ("SCT 415/2024 ...") and bracket-year titles ("[2024] DIFC SCT 415 ...").
    The short form MatchText("SCT 415") was removed because it causes year-collision
    false-positives — e.g. querying SCT 160/2025 also matched SCT 160/2024 docs,
    returning the wrong document's chunks to BM25 and causing null answers.
    """
    from qdrant_client import models

    from shafi.core.retriever_filters import build_filter

    result = build_filter(
        doc_type_filter=None,
        jurisdiction_filter=None,
        doc_refs=["SCT 415/2024"],
    )

    assert result is not None
    should_filter = next(
        f
        for f in result.must
        if isinstance(f, models.Filter)  # type: ignore[union-attr]
    )
    assert should_filter.should is not None
    match_texts = {
        c.match.text  # type: ignore[union-attr]
        for c in should_filter.should
        if hasattr(c, "key")
        and c.key == "doc_title"  # type: ignore[union-attr]
        and hasattr(c, "match")
        and isinstance(c.match, models.MatchText)  # type: ignore[union-attr]
    }
    assert "SCT 415/2024" in match_texts, "Canonical slash-year form must be present"
    assert "SCT 415" not in match_texts, (
        "Short form without year must NOT be present — causes year-collision false-positives"
    )


def test_build_filter_case_ref_small_number_no_short_form_match_text() -> None:
    """Case ref filter must NOT include any short-form MatchText (with or without padding).

    Short-form MatchText ("SCT 042" or "SCT 42") caused year-collision false-positives —
    e.g. "SCT 042" matched both "[2024] DIFC SCT 042" and "[2025] DIFC SCT 042" docs.
    Only the canonical slash-year form is emitted; all other matching is via citations/
    case_numbers exact-match fields which are inherently year-specific.
    """
    from qdrant_client import models

    from shafi.core.retriever_filters import build_filter

    result = build_filter(
        doc_type_filter=None,
        jurisdiction_filter=None,
        doc_refs=["SCT 42/2024"],
    )

    assert result is not None
    should_filter = next(
        f
        for f in result.must
        if isinstance(f, models.Filter)  # type: ignore[union-attr]
    )
    assert should_filter.should is not None
    match_texts = {
        c.match.text  # type: ignore[union-attr]
        for c in should_filter.should
        if hasattr(c, "key")
        and c.key == "doc_title"  # type: ignore[union-attr]
        and hasattr(c, "match")
        and isinstance(c.match, models.MatchText)  # type: ignore[union-attr]
    }
    assert "SCT 042/2024" in match_texts, "Canonical slash-year form must be present"
    assert "SCT 042" not in match_texts, "Zero-padded short form must NOT be present (year-collision risk)"
    assert "SCT 42" not in match_texts, "Unpadded short form must NOT be present (prefix false-positives)"


def test_doc_title_filter_variants_includes_uppercase_for_plain_titles() -> None:
    """Plain regulation titles get an uppercase variant to match ALL_CAPS Qdrant titles.

    Many statute doc_titles are stored in ALL CAPS in the private collection
    (e.g. "OPERATING REGULATIONS", "FINANCIAL COLLATERAL REGULATIONS").
    doc_title_filter_variants must emit the uppercase form so MatchAny hits them.
    """
    from shafi.core.retriever_filters import doc_title_filter_variants

    variants = doc_title_filter_variants(["Operating Regulations"])
    assert "Operating Regulations" in variants, "Original mixed-case must be present"
    assert "OPERATING REGULATIONS" in variants, "ALL_CAPS variant must be present for Qdrant matching"

    variants2 = doc_title_filter_variants(["Financial Collateral Regulations"])
    assert "FINANCIAL COLLATERAL REGULATIONS" in variants2
