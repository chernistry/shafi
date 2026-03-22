"""Tests for deterministic shadow query rewrites."""

from __future__ import annotations

from rag_challenge.core.retrieval.query_rewrite_shadow import build_shadow_rewrite_query
from rag_challenge.models.schemas import RetrievedPage, ScopeMode


def _make_page(page_id: str, *, article_refs: list[str] | None = None) -> RetrievedPage:
    doc_id, _, page_num = page_id.rpartition("_")
    return RetrievedPage(
        page_id=page_id,
        doc_id=doc_id,
        page_num=int(page_num),
        doc_title="Employment Law",
        doc_type="statute",
        page_text="Article 16 requires annual returns.",
        score=0.8,
        page_template_family="article_body",
        article_refs=article_refs or [],
        law_title_aliases=["Employment Law"],
    )


def _make_authority_page(page_id: str) -> RetrievedPage:
    doc_id, _, page_num = page_id.rpartition("_")
    return RetrievedPage(
        page_id=page_id,
        doc_id=doc_id,
        page_num=int(page_num),
        doc_title="DIFC Law No. 5 of 2020",
        doc_type="statute",
        page_text="This is an enactment notice for a DIFC law.",
        score=0.9,
        page_template_family="authority_page",
        article_refs=[],
        law_title_aliases=["DIFC Law No. 5 of 2020"],
        law_titles=["DIFC Law No. 5 of 2020"],
    )


def test_build_shadow_rewrite_query_exact_provision_keeps_anchor_and_official_terms() -> None:
    rewrite = build_shadow_rewrite_query(
        query="According to Article 16 of the law, what document must be filed?",
        family="exact_provision",
        hard_anchor_strings=("Article 16",),
        page_candidates=[_make_page("law_2", article_refs=["Article 16"])],
        scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
    )

    assert rewrite is not None
    assert "Article 16" in rewrite.rewritten_query
    assert "official text" in rewrite.rewritten_query
    assert rewrite.family == "exact_provision"


def test_build_shadow_rewrite_query_authority_metadata_adds_jurisdiction_terms() -> None:
    rewrite = build_shadow_rewrite_query(
        query="What is the DFSA official title and issue date?",
        family="authority_metadata",
        hard_anchor_strings=(),
        page_candidates=[_make_authority_page("law_2")],
        scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
    )

    assert rewrite is not None
    assert "Dubai Financial Services Authority" in rewrite.rewritten_query
    assert "Dubai International Financial Centre" in rewrite.rewritten_query
    assert "date of issue" in rewrite.rewritten_query
    assert rewrite.family == "authority_metadata"


def test_build_shadow_rewrite_query_returns_none_for_unsupported_family() -> None:
    rewrite = build_shadow_rewrite_query(
        query="Summarize the law in plain language.",
        family="",
        hard_anchor_strings=(),
        page_candidates=[],
        scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
    )

    assert rewrite is None
