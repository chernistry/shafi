# pyright: reportPrivateUsage=false
"""Tests for evidence-portfolio grounding helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_challenge.core.grounding.condition_audit import audit_candidate_pages
from rag_challenge.core.grounding.evidence_portfolio import (
    EvidencePortfolio,
    PortfolioCandidate,
    build_law_bundle_candidate,
    select_best_page_set,
)
from rag_challenge.core.grounding.evidence_selector import GroundingEvidenceSelector
from rag_challenge.core.grounding.necessity_pruner import prune_redundant_pages
from rag_challenge.core.grounding.typed_panel_extractor import build_typed_comparison_panel
from rag_challenge.models.schemas import QueryScopePrediction, RetrievedPage, ScopeMode
from rag_challenge.telemetry import TelemetryCollector


def _make_scope(mode: ScopeMode, *, page_budget: int = 2) -> QueryScopePrediction:
    """Create a deterministic scope for portfolio tests.

    Args:
        mode: Scope mode under test.
        page_budget: Grounding page budget.

    Returns:
        QueryScopePrediction: Minimal test scope.
    """

    return QueryScopePrediction(scope_mode=mode, page_budget=page_budget)


def _make_page(
    page_id: str,
    *,
    score: float,
    doc_title: str = "Doc",
    page_template_family: str = "",
    officialness_score: float = 0.0,
    source_vs_reference_prior: float = 0.0,
    has_caption_block: bool = False,
    canonical_law_family: str = "",
    related_law_families: list[str] | None = None,
    top_lines: list[str] | None = None,
    heading_lines: list[str] | None = None,
    article_refs: list[str] | None = None,
    has_law_number_pattern: bool = False,
    has_issued_by_pattern: bool = False,
) -> RetrievedPage:
    """Create one retrieved page payload for portfolio tests.

    Args:
        page_id: Stable page identifier.
        score: Retrieval score.
        doc_title: Document title.
        page_template_family: Semantic page template label.
        officialness_score: Deterministic officialness prior.
        source_vs_reference_prior: Source-vs-reference prior.
        has_caption_block: Whether the page looks like a caption/header page.
        canonical_law_family: Canonical law family for the page.
        related_law_families: Optional related family labels.
        top_lines: Optional top-line payload.
        heading_lines: Optional heading payload.
        article_refs: Optional article refs.
        has_law_number_pattern: Whether the page carries a law-number pattern.
        has_issued_by_pattern: Whether the page carries an issued-by pattern.

    Returns:
        RetrievedPage: Test page payload.
    """

    doc_id, _, page_num = page_id.rpartition("_")
    return RetrievedPage(
        page_id=page_id,
        doc_id=doc_id,
        page_num=int(page_num),
        doc_title=doc_title,
        doc_type="statute",
        page_text="Test page",
        score=score,
        page_template_family=page_template_family,
        officialness_score=officialness_score,
        source_vs_reference_prior=source_vs_reference_prior,
        has_caption_block=has_caption_block,
        canonical_law_family=canonical_law_family,
        related_law_families=list(related_law_families or []),
        top_lines=list(top_lines or []),
        heading_lines=list(heading_lines or []),
        article_refs=list(article_refs or []),
        has_law_number_pattern=has_law_number_pattern,
        has_issued_by_pattern=has_issued_by_pattern,
    )


def _make_selector() -> GroundingEvidenceSelector:
    """Build a selector with minimal mocked collaborators.

    Returns:
        GroundingEvidenceSelector: Selector for direct private-method tests.
    """

    settings = SimpleNamespace(
        enable_trained_page_scorer=False,
        trained_page_scorer_model_path="",
        grounding_support_fact_top_k=8,
        grounding_page_top_k=8,
    )
    return GroundingEvidenceSelector(
        retriever=MagicMock(),
        store=MagicMock(),
        embedder=MagicMock(),
        sparse_encoder=None,
        pipeline_settings=settings,
    )


def test_build_typed_comparison_panel_selects_one_authoritative_page_per_doc() -> None:
    pages = [
        _make_page(
            "doc-a_2",
            score=0.95,
            doc_title="CFI 001/2024 Alpha",
            page_template_family="duplicate_or_reference_like",
            officialness_score=0.1,
            source_vs_reference_prior=0.1,
        ),
        _make_page(
            "doc-a_1",
            score=0.7,
            doc_title="CFI 001/2024 Alpha",
            page_template_family="caption_header",
            officialness_score=0.9,
            source_vs_reference_prior=0.95,
            has_caption_block=True,
        ),
        _make_page(
            "doc-b_2",
            score=0.9,
            doc_title="CFI 002/2024 Beta",
            page_template_family="duplicate_or_reference_like",
            officialness_score=0.1,
            source_vs_reference_prior=0.1,
        ),
        _make_page(
            "doc-b_1",
            score=0.65,
            doc_title="CFI 002/2024 Beta",
            page_template_family="caption_header",
            officialness_score=0.92,
            source_vs_reference_prior=0.96,
            has_caption_block=True,
        ),
    ]

    panel = build_typed_comparison_panel(
        query="What common party appears in CFI 001/2024 and CFI 002/2024?",
        ordered_pages=pages,
    )

    assert tuple(item.page_id for item in panel) == ("doc-a_1", "doc-b_1")


def test_portfolio_selector_prefers_typed_compare_panel_on_party_query() -> None:
    pages = {
        page.page_id: page
        for page in [
            _make_page(
                "doc-a_2",
                score=0.95,
                doc_title="CFI 001/2024 Alpha",
                page_template_family="duplicate_or_reference_like",
                officialness_score=0.1,
                source_vs_reference_prior=0.1,
            ),
            _make_page(
                "doc-a_1",
                score=0.7,
                doc_title="CFI 001/2024 Alpha",
                page_template_family="caption_header",
                officialness_score=0.9,
                source_vs_reference_prior=0.95,
                has_caption_block=True,
            ),
            _make_page(
                "doc-b_2",
                score=0.9,
                doc_title="CFI 002/2024 Beta",
                page_template_family="duplicate_or_reference_like",
                officialness_score=0.1,
                source_vs_reference_prior=0.1,
            ),
            _make_page(
                "doc-b_1",
                score=0.65,
                doc_title="CFI 002/2024 Beta",
                page_template_family="caption_header",
                officialness_score=0.92,
                source_vs_reference_prior=0.96,
                has_caption_block=True,
            ),
        ]
    }
    scope = _make_scope(ScopeMode.COMPARE_PAIR, page_budget=2)
    portfolio = EvidencePortfolio(
        query="What common party appears in CFI 001/2024 and CFI 002/2024?",
        scope_mode=scope.scope_mode,
        candidates=(
            PortfolioCandidate(
                name="safe_sidecar",
                page_ids=("doc-a_2", "doc-b_2"),
                activation_family="safe_sidecar",
            ),
            PortfolioCandidate(
                name="typed_compare_panel",
                page_ids=("doc-a_1", "doc-b_1"),
                activation_family="typed_compare_panel",
            ),
        ),
    )

    result = select_best_page_set(
        query=portfolio.query,
        answer_type="name",
        scope=scope,
        page_lookup=pages,
        portfolio=portfolio,
    )

    assert result.selected_candidate.name == "typed_compare_panel"


def test_prune_redundant_pages_removes_extra_same_doc_page() -> None:
    pages = [
        _make_page(
            "doc-a_1",
            score=0.8,
            page_template_family="caption_header",
            officialness_score=0.9,
            source_vs_reference_prior=0.95,
            has_caption_block=True,
        ),
        _make_page(
            "doc-a_3",
            score=0.7,
            page_template_family="duplicate_or_reference_like",
            officialness_score=0.1,
            source_vs_reference_prior=0.1,
        ),
        _make_page(
            "doc-b_1",
            score=0.82,
            page_template_family="caption_header",
            officialness_score=0.9,
            source_vs_reference_prior=0.95,
            has_caption_block=True,
        ),
    ]

    result = prune_redundant_pages(
        query="What common party appears in both cases?",
        answer_type="name",
        scope_mode=ScopeMode.COMPARE_PAIR,
        ordered_pages=pages,
        page_budget=2,
    )

    # G-maximizer (chernistry, 2026-03-21): pruning is disabled to protect recall.
    # All candidate pages are kept even if redundant.
    assert result == ("doc-a_1", "doc-a_3", "doc-b_1")


def test_condition_audit_detects_missing_authority_slot() -> None:
    audit = audit_candidate_pages(
        query="Who issued the law?",
        answer_type="name",
        scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
        pages=[
            _make_page(
                "law_2",
                score=0.8,
                page_template_family="article_body",
                officialness_score=0.5,
                source_vs_reference_prior=0.5,
            )
        ],
    )

    assert audit.success is False
    assert "authority" in audit.failed_slots


def test_build_law_bundle_candidate_uses_matching_family() -> None:
    candidate = build_law_bundle_candidate(
        query="According to the Companies Law, what is the law number?",
        ordered_pages=[
            _make_page(
                "law_3",
                score=0.9,
                page_template_family="article_body",
                officialness_score=0.8,
                source_vs_reference_prior=0.8,
                canonical_law_family="companies law",
                related_law_families=["companies law enactment notice"],
                has_law_number_pattern=True,
            )
        ],
    )

    assert candidate is not None
    assert candidate.name == "law_bundle"
    assert candidate.page_ids == ("law_3",)


def test_selector_records_portfolio_diagnostics_and_uses_compare_candidate() -> None:
    selector = _make_selector()
    scope = _make_scope(ScopeMode.COMPARE_PAIR, page_budget=2)
    pages = [
        _make_page(
            "doc-a_2",
            score=0.95,
            doc_title="CFI 001/2024 Alpha",
            page_template_family="duplicate_or_reference_like",
            officialness_score=0.1,
            source_vs_reference_prior=0.1,
        ),
        _make_page(
            "doc-a_1",
            score=0.7,
            doc_title="CFI 001/2024 Alpha",
            page_template_family="caption_header",
            officialness_score=0.9,
            source_vs_reference_prior=0.95,
            has_caption_block=True,
        ),
        _make_page(
            "doc-b_2",
            score=0.9,
            doc_title="CFI 002/2024 Beta",
            page_template_family="duplicate_or_reference_like",
            officialness_score=0.1,
            source_vs_reference_prior=0.1,
        ),
        _make_page(
            "doc-b_1",
            score=0.65,
            doc_title="CFI 002/2024 Beta",
            page_template_family="caption_header",
            officialness_score=0.92,
            source_vs_reference_prior=0.96,
            has_caption_block=True,
        ),
    ]
    collector = TelemetryCollector("req-1")

    result = selector._select_minimal_pages(
        query="What common party appears in CFI 001/2024 and CFI 002/2024?",
        # Use "names" (plural) to isolate portfolio typed_compare_panel selection.
        # "name" (singular) triggers page_1 preference in safe_sidecar which
        # collapses safe_sidecar and typed_compare_panel to the same pages —
        # that behavior is tested in test_grounding_sidecar.py.
        answer_type="names",
        scope=scope,
        scored={
            "doc-a_2": 4.0,
            "doc-b_2": 3.9,
            "doc-a_1": 3.0,
            "doc-b_1": 2.9,
        },
        page_candidates=pages,
        collector=collector,
    )

    telemetry = collector.finalize()
    assert result == ["doc-a_1", "doc-b_1"]
    assert telemetry.grounding_portfolio_selected == "typed_compare_panel"
    assert "typed_compare_panel" in telemetry.grounding_portfolio_candidates
