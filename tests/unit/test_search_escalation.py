"""Tests for bounded grounding escalation decisions."""

from __future__ import annotations

from rag_challenge.core.grounding.condition_audit import ConditionAuditResult
from rag_challenge.core.grounding.search_escalation import decide_search_escalation
from rag_challenge.models import DocType, RankedChunk
from rag_challenge.models.schemas import QueryScopePrediction, RetrievedPage, ScopeMode


def _make_chunk(*, chunk_id: str, doc_id: str, rerank_score: float) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title="Doc",
        doc_type=DocType.STATUTE,
        text="text",
        retrieval_score=rerank_score,
        rerank_score=rerank_score,
    )


def _make_page(
    *,
    page_id: str,
    score: float,
    page_template_family: str = "",
    officialness_score: float = 0.0,
    source_vs_reference_prior: float = 0.0,
    article_refs: list[str] | None = None,
) -> RetrievedPage:
    doc_id, _, page_num = page_id.rpartition("_")
    return RetrievedPage(
        page_id=page_id,
        doc_id=doc_id,
        page_num=int(page_num),
        doc_title="Employment Law",
        doc_type="statute",
        page_text="Article 16 requires annual returns.",
        score=score,
        page_template_family=page_template_family,
        officialness_score=officialness_score,
        source_vs_reference_prior=source_vs_reference_prior,
        article_refs=article_refs or [],
    )


def test_decide_search_escalation_triggers_for_missing_article_slot_and_low_margin() -> None:
    decision = decide_search_escalation(
        query="According to Article 16 of the law, what document must be filed?",
        answer_type="name",
        scope=QueryScopePrediction(scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC, page_budget=1),
        context_chunks=[
            _make_chunk(chunk_id="law:0:0:a", doc_id="law", rerank_score=0.52),
            _make_chunk(chunk_id="law:1:0:b", doc_id="law", rerank_score=0.49),
        ],
        ordered_pages=[
            _make_page(page_id="law_1", score=0.81, page_template_family="duplicate_or_reference_like"),
            _make_page(
                page_id="law_2",
                score=0.79,
                page_template_family="article_body",
                officialness_score=0.7,
                source_vs_reference_prior=0.7,
                article_refs=["Article 16"],
            ),
        ],
        selected_pages=[_make_page(page_id="law_1", score=0.81, page_template_family="duplicate_or_reference_like")],
        audit=ConditionAuditResult(
            success=False,
            required_slots=("article_anchor",),
            covered_slots=(),
            failed_slots=("article_anchor",),
            reasons=("missing:article_anchor",),
            coverage_ratio=0.0,
        ),
    )

    assert decision.should_escalate is True
    assert decision.allowed_family == "exact_provision"
    assert "typed_slots_missing" in decision.reasons
    assert "low_rerank_confidence" in decision.reasons


def test_decide_search_escalation_skips_when_selected_page_is_already_strong() -> None:
    strong_page = _make_page(
        page_id="law_2",
        score=0.95,
        page_template_family="article_body",
        officialness_score=1.0,
        source_vs_reference_prior=1.0,
        article_refs=["Article 16"],
    )
    decision = decide_search_escalation(
        query="According to Article 16 of the law, what document must be filed?",
        answer_type="name",
        scope=QueryScopePrediction(scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC, page_budget=1),
        context_chunks=[_make_chunk(chunk_id="law:1:0:a", doc_id="law", rerank_score=0.9)],
        ordered_pages=[strong_page],
        selected_pages=[strong_page],
        audit=ConditionAuditResult(
            success=True,
            required_slots=("article_anchor",),
            covered_slots=("article_anchor",),
            failed_slots=(),
            reasons=("covered:article_anchor",),
            coverage_ratio=1.0,
        ),
        authority_strength_threshold=0.5,
    )

    assert decision.should_escalate is False
    assert decision.allowed_family == "exact_provision"
