from __future__ import annotations

from types import SimpleNamespace

from shafi.core.pipeline.retrieval_primitives import (
    augment_query_for_sparse_retrieval,
    extract_provision_refs,
    seed_terms_for_query,
    targeted_provision_ref_query,
)


def test_extract_provision_refs_supports_nested_parens_and_multiple_kinds() -> None:
    refs = extract_provision_refs("See Article 11(1)(a), Section 4(2), Schedule 1 and Part IV.")

    assert refs == ["Article 11(1)(a)", "Section 4(2)", "Schedule 1", "Part IV"]


def test_augment_query_for_sparse_retrieval_adds_drafting_variants_for_provisions() -> None:
    boosted = augment_query_for_sparse_retrieval("Please apply Article 11(1)(a), Section 4(2), Schedule 1 and Part IV.")

    assert "11(1)(a)" in boosted
    assert "11 (1) (a)" in boosted
    assert "11." in boosted
    assert "4(2)" in boosted
    assert "4 (2)" in boosted
    assert "1." in boosted
    assert "IV" in boosted


def test_targeted_provision_ref_query_keeps_exact_ref_focus() -> None:
    pipeline = SimpleNamespace(extract_provision_refs=extract_provision_refs)
    targeted = targeted_provision_ref_query(
        pipeline,
        query="Article 11(1)(a) and Case B under DIFC Law",
        ref="Case A",
        refs=["Case B"],
    )

    assert targeted.startswith("Case A")
    assert "Article 11(1)(a)" in targeted
    assert "11 (1) (a)" in targeted
    assert "Case B" not in targeted
    assert "under DIFC Law" in targeted


def test_seed_terms_for_query_adds_provision_drafting_markers() -> None:
    terms = seed_terms_for_query("What does Section 4(2) say in Schedule 1?")

    assert "provision" in terms
    assert "subsection" in terms
    assert "subparagraph" in terms
    assert "paragraph" in terms
    assert "annex" in terms
    assert "appendix" in terms
    assert "section 4(2)" in terms
    assert "schedule 1" in terms
