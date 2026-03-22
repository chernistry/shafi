from __future__ import annotations

from shafi.core.grounding.page_semantic_lane import (
    authority_signal_score,
    build_shadow_text,
    rerank_pages_with_shadow_signal,
    select_semantic_page_set,
    shadow_signal_score,
)
from shafi.models.schemas import RetrievedPage, ScopeMode


def _make_page(
    page_id: str,
    *,
    score: float = 0.5,
    page_num: int | None = None,
    doc_id: str | None = None,
    heading_lines: list[str] | None = None,
    top_lines: list[str] | None = None,
    field_labels_present: list[str] | None = None,
    page_template_family: str = "",
    document_template_family: str = "",
    officialness_score: float = 0.0,
    source_vs_reference_prior: float = 0.0,
    has_date_of_issue_pattern: bool = False,
    has_issued_by_pattern: bool = False,
    has_claim_number_pattern: bool = False,
    has_law_number_pattern: bool = False,
) -> RetrievedPage:
    parsed_doc_id, _, parsed_page_num = page_id.rpartition("_")
    return RetrievedPage(
        page_id=page_id,
        doc_id=doc_id or parsed_doc_id,
        page_num=page_num if page_num is not None else int(parsed_page_num),
        doc_title="Doc",
        doc_type="statute",
        page_text="Body",
        score=score,
        heading_lines=heading_lines or [],
        top_lines=top_lines or [],
        field_labels_present=field_labels_present or [],
        page_template_family=page_template_family,
        document_template_family=document_template_family,
        officialness_score=officialness_score,
        source_vs_reference_prior=source_vs_reference_prior,
        has_date_of_issue_pattern=has_date_of_issue_pattern,
        has_issued_by_pattern=has_issued_by_pattern,
        has_claim_number_pattern=has_claim_number_pattern,
        has_law_number_pattern=has_law_number_pattern,
    )


def test_build_shadow_text_prefers_heading_and_field_signals() -> None:
    page = _make_page(
        "law_2",
        heading_lines=["Date of Issue"],
        top_lines=["DIFC Employment Law"],
        field_labels_present=["Issued by", "Date of Issue"],
        page_template_family="issued_by_authority",
        document_template_family="law_notice",
    )

    shadow = build_shadow_text(page)

    assert "Date of Issue" in shadow
    assert "DIFC Employment Law" in shadow
    assert "issued_by_authority" in shadow


def test_rerank_pages_with_shadow_signal_promotes_heading_match() -> None:
    pages = [
        _make_page("law_1", score=0.9, heading_lines=["General Provisions"]),
        _make_page(
            "law_2",
            score=0.6,
            heading_lines=["Date of Issue"],
            field_labels_present=["Date of Issue"],
            has_date_of_issue_pattern=True,
        ),
    ]

    reranked = rerank_pages_with_shadow_signal("What is the date of issue of the law?", pages)

    assert [page.page_id for page in reranked] == ["law_2", "law_1"]


def test_authority_signal_penalizes_reference_page_when_stronger_primary_exists() -> None:
    primary = _make_page(
        "case_1",
        page_template_family="caption_header",
        officialness_score=0.9,
        source_vs_reference_prior=0.9,
    )
    reference = _make_page(
        "case_2",
        page_template_family="duplicate_or_reference_like",
        officialness_score=0.2,
        source_vs_reference_prior=0.1,
    )

    assert authority_signal_score("Who were the claimants?", reference, peer_pages=[primary, reference]) < 0.0
    assert authority_signal_score("Who were the claimants?", primary, peer_pages=[primary, reference]) > 0.0


def test_select_semantic_page_set_returns_title_article_pair() -> None:
    pages = [
        _make_page(
            "law_1",
            page_template_family="title_cover",
            officialness_score=0.9,
            source_vs_reference_prior=0.9,
        ),
        _make_page(
            "law_5",
            page_template_family="article_body",
            officialness_score=0.8,
            source_vs_reference_prior=0.9,
        ),
    ]

    decision = select_semantic_page_set(
        query="According to Article 16 of the law, what does the title notice say?",
        scope_mode=ScopeMode.FULL_CASE_FILES,
        ordered_page_ids=["law_1", "law_5"],
        page_candidates=pages,
        page_budget=2,
    )

    assert decision is not None
    assert list(decision.page_ids) == ["law_1", "law_5"]


def test_select_semantic_page_set_returns_title_article_pair_for_single_doc_scope() -> None:
    pages = [
        _make_page(
            "law_1",
            page_template_family="title_cover",
            officialness_score=0.95,
            source_vs_reference_prior=0.9,
        ),
        _make_page(
            "law_5",
            page_template_family="article_body",
            officialness_score=0.8,
            source_vs_reference_prior=0.85,
        ),
    ]

    decision = select_semantic_page_set(
        query="According to Article 16 of the law, what document must be filed?",
        scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
        ordered_page_ids=["law_1", "law_5"],
        page_candidates=pages,
        page_budget=1,
    )

    assert decision is not None
    assert list(decision.page_ids) == ["law_1", "law_5"]


def test_select_semantic_page_set_stays_off_for_non_pair_budget() -> None:
    pages = [
        _make_page("law_1", page_template_family="title_cover"),
        _make_page("law_2", page_template_family="article_body"),
    ]

    decision = select_semantic_page_set(
        query="According to Article 16, what does the law say?",
        scope_mode=ScopeMode.FULL_CASE_FILES,
        ordered_page_ids=["law_1", "law_2"],
        page_candidates=pages,
        page_budget=1,
    )

    assert decision is None


def test_select_semantic_page_set_stays_off_for_generic_single_doc_query() -> None:
    pages = [
        _make_page("law_1", page_template_family="title_cover"),
        _make_page("law_2", page_template_family="article_body"),
    ]

    decision = select_semantic_page_set(
        query="What is the amount owed?",
        scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
        ordered_page_ids=["law_1", "law_2"],
        page_candidates=pages,
        page_budget=1,
    )

    assert decision is None


def test_shadow_signal_uses_claim_and_law_number_patterns() -> None:
    claim_page = _make_page(
        "law_7",
        field_labels_present=["Claim Number", "Law Number"],
        has_claim_number_pattern=True,
        has_law_number_pattern=True,
    )

    score = shadow_signal_score("What is the claim number and law number?", claim_page)

    assert score > 0.5
