"""Verify LightGBM and logistic page-scorer models produce different rankings."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_challenge.core.grounding.query_scope_classifier import classify_query_scope
from rag_challenge.ml.page_scorer_runtime import RuntimePageScorer, RuntimePageScoringRequest
from rag_challenge.models import RetrievedPage

REPO_ROOT = Path(__file__).resolve().parents[2]
LGBM_MODEL = REPO_ROOT / "models" / "page_scorer" / "v4_tamar_augmented" / "page_scorer.joblib"
LOGISTIC_MODEL = REPO_ROOT / "models" / "page_scorer" / "v3_logistic" / "page_scorer.joblib"


def _make_request() -> RuntimePageScoringRequest:
    """Build a realistic scoring request with 10 candidate pages."""
    return RuntimePageScoringRequest(
        query="According to Article 14(2)(b) of the General Partnership Law 2004, can a partner "
        "assign their interest in the partnership?",
        normalized_answer="Yes",
        answer_type="boolean",
        scope=classify_query_scope(
            "According to Article 14(2)(b) of the General Partnership Law 2004, can a partner "
            "assign their interest in the partnership?",
            "boolean",
        ),
        doc_ids=["doc-gp-law"],
        page_candidates=[
            RetrievedPage(page_id="doc-gp-law_1", doc_id="doc-gp-law", page_num=1, score=0.6, page_text="GENERAL PARTNERSHIP LAW Enacted by decree"),
            RetrievedPage(page_id="doc-gp-law_2", doc_id="doc-gp-law", page_num=2, score=0.65, page_text="Table of Contents Part I Preliminary"),
            RetrievedPage(page_id="doc-gp-law_3", doc_id="doc-gp-law", page_num=3, score=0.7, page_text="Article 1 definitions means"),
            RetrievedPage(page_id="doc-gp-law_4", doc_id="doc-gp-law", page_num=4, score=0.75, page_text="Article 14 Assignment of Partnership Interest (1) Subject to this Law (2)(b) A partner may assign"),
            RetrievedPage(page_id="doc-gp-law_5", doc_id="doc-gp-law", page_num=5, score=0.72, page_text="Article 15 Dissolution of Partnership shall must"),
            RetrievedPage(page_id="doc-gp-law_6", doc_id="doc-gp-law", page_num=6, score=0.68, page_text="Article 16 Distribution of assets pursuant to"),
            RetrievedPage(page_id="doc-gp-law_7", doc_id="doc-gp-law", page_num=7, score=0.64, page_text="Article 20 Liability of partners notwithstanding"),
            RetrievedPage(page_id="doc-gp-law_8", doc_id="doc-gp-law", page_num=8, score=0.62, page_text="Schedule 1 Interpretation provisions defined"),
            RetrievedPage(page_id="doc-gp-law_9", doc_id="doc-gp-law", page_num=9, score=0.58, page_text="Schedule 2 Forms and notices"),
            RetrievedPage(page_id="doc-gp-law_10", doc_id="doc-gp-law", page_num=10, score=0.55, page_text="Amendments and transitional provisions"),
        ],
        context_page_ids=["doc-gp-law_4", "doc-gp-law_5"],
        legacy_used_page_ids=["doc-gp-law_5"],
        heuristic_scores={f"doc-gp-law_{i}": 10.0 - i * 0.5 for i in range(1, 11)},
    )


@pytest.mark.skipif(not LGBM_MODEL.is_file(), reason="LightGBM model not trained yet")
def test_lgbm_model_loads_and_predicts() -> None:
    """LightGBM model loads and produces rankings."""
    scorer = RuntimePageScorer(model_path=str(LGBM_MODEL))
    result = scorer.rank_pages(_make_request())

    assert result.used is True
    assert len(result.ranked_page_ids) == 10
    assert result.fallback_reason == ""


@pytest.mark.skipif(not LOGISTIC_MODEL.is_file(), reason="Logistic model not trained yet")
def test_logistic_model_loads_and_predicts() -> None:
    """Logistic model loads and produces rankings."""
    scorer = RuntimePageScorer(model_path=str(LOGISTIC_MODEL))
    result = scorer.rank_pages(_make_request())

    assert result.used is True
    assert len(result.ranked_page_ids) == 10
    assert result.fallback_reason == ""


@pytest.mark.skipif(
    not (LGBM_MODEL.is_file() and LOGISTIC_MODEL.is_file()),
    reason="Both models needed for comparison",
)
def test_lgbm_produces_different_ranking_than_logistic() -> None:
    """LightGBM and logistic regression produce different page rankings."""
    request = _make_request()

    lgbm_scorer = RuntimePageScorer(model_path=str(LGBM_MODEL))
    logistic_scorer = RuntimePageScorer(model_path=str(LOGISTIC_MODEL))

    lgbm_result = lgbm_scorer.rank_pages(request)
    logistic_result = logistic_scorer.rank_pages(request)

    assert lgbm_result.used is True
    assert logistic_result.used is True

    # Rankings should differ — LightGBM captures feature interactions
    assert lgbm_result.ranked_page_ids != logistic_result.ranked_page_ids, (
        "LightGBM and logistic produced identical rankings — "
        "either models are equivalent or test data is degenerate"
    )
