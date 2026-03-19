from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_challenge.core.grounding.evidence_selector import GroundingEvidenceSelector
from rag_challenge.core.grounding.query_scope_classifier import (
    classify_query_scope,
    extract_explicit_page_numbers,
)
from rag_challenge.ml.page_scorer_runtime import RuntimePageScorerResult
from rag_challenge.models import DocType, RankedChunk, RetrievedPage, ScopeMode
from rag_challenge.telemetry import TelemetryCollector


def _make_ranked_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    section_path: str,
    text: str,
    doc_title: str = "Doc",
) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=doc_title,
        doc_type=DocType.STATUTE,
        section_path=section_path,
        text=text,
        retrieval_score=0.9,
        rerank_score=0.9,
        doc_summary="",
    )


def _make_selector(*, retrieved_pages: list[RetrievedPage]) -> tuple[GroundingEvidenceSelector, MagicMock]:
    retriever = MagicMock()
    retriever.retrieve_pages = AsyncMock(return_value=retrieved_pages)

    store = MagicMock()
    store.support_fact_collection_name = "legal_support_facts"
    store.client.collection_exists = AsyncMock(return_value=False)

    embedder = MagicMock()
    settings = SimpleNamespace(
        enable_trained_page_scorer=False,
        trained_page_scorer_model_path="",
        grounding_support_fact_top_k=8,
        grounding_page_top_k=5,
    )

    selector = GroundingEvidenceSelector(
        retriever=retriever,
        store=store,
        embedder=embedder,
        sparse_encoder=None,
        pipeline_settings=settings,
    )
    return selector, retriever


def test_extract_explicit_page_numbers_returns_unique_positive_page_numbers() -> None:
    assert extract_explicit_page_numbers("What is stated on page 2 and pages 2 and 5?") == [2, 5]


def test_classify_query_scope_uses_issued_by_role_for_compare_judge_queries() -> None:
    prediction = classify_query_scope(
        "Who was the judge in CFI 001/2024 and CFI 002/2024?",
        "name",
    )

    assert prediction.scope_mode is ScopeMode.COMPARE_PAIR
    assert "issued_by_block" in prediction.target_page_roles
    assert "title_cover" in prediction.target_page_roles


@pytest.mark.asyncio
async def test_grounding_sidecar_leaves_single_doc_date_scope_on_legacy_path() -> None:
    selector, retriever = _make_selector(
        retrieved_pages=[
            RetrievedPage(
                page_id="law_2",
                doc_id="law",
                page_num=2,
                doc_title="Law",
                doc_type="statute",
                page_text="Issued by: Registrar. Date of Issue: 2024-01-01.",
                score=0.95,
                page_role="issued_by_block",
            )
        ]
    )

    result = await selector.select_page_ids(
        query="What is the date of issue of the law?",
        answer="2024-01-01",
        answer_type="date",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="law:1:0:issued",
                doc_id="law",
                section_path="page:2",
                text="Issued by: Registrar. Date of Issue: 2024-01-01.",
            )
        ],
    )

    assert result is None
    retriever.retrieve_pages.assert_not_awaited()


@pytest.mark.asyncio
async def test_grounding_sidecar_does_not_activate_single_doc_scope_with_multiple_docs() -> None:
    selector, retriever = _make_selector(retrieved_pages=[])

    result = await selector.select_page_ids(
        query="What is the date of issue of the law?",
        answer="2024-01-01",
        answer_type="date",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="law-a:1:0:issued",
                doc_id="law-a",
                section_path="page:2",
                text="Issued by: Registrar. Date of Issue: 2024-01-01.",
            ),
            _make_ranked_chunk(
                chunk_id="law-b:0:0:title",
                doc_id="law-b",
                section_path="page:1",
                text="Title page.",
            ),
        ],
    )

    assert result is None
    retriever.retrieve_pages.assert_not_awaited()


@pytest.mark.asyncio
async def test_grounding_sidecar_passes_explicit_page_filter_for_single_doc_scope() -> None:
    selector, retriever = _make_selector(
        retrieved_pages=[
            RetrievedPage(
                page_id="law_2",
                doc_id="law",
                page_num=2,
                doc_title="Law",
                doc_type="statute",
                page_text="The required text is on page two.",
                score=0.88,
                page_role="other",
            )
        ]
    )

    result = await selector.select_page_ids(
        query="What is stated on page 2 of the law?",
        answer="The required text is on page two.",
        answer_type="free_text",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="law:0:0:title",
                doc_id="law",
                section_path="page:1",
                text="Title page.",
            )
        ],
    )

    assert result == ["law_2"]
    kwargs = retriever.retrieve_pages.await_args.kwargs
    assert kwargs["page_nums"] == [2]
    assert kwargs["doc_ids"] == ["law"]


@pytest.mark.asyncio
async def test_grounding_sidecar_leaves_single_doc_article_scope_on_legacy_path() -> None:
    selector, retriever = _make_selector(
        retrieved_pages=[
            RetrievedPage(
                page_id="law_16",
                doc_id="law",
                page_num=16,
                doc_title="Law",
                doc_type="statute",
                page_text="Article 16 requires the filing of the annual return.",
                score=0.91,
                page_role="article_clause",
                article_refs=["Article 16"],
            )
        ]
    )

    result = await selector.select_page_ids(
        query="According to Article 16 of the law, what document must be filed?",
        answer="annual return",
        answer_type="name",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="law:15:0:article",
                doc_id="law",
                section_path="page:16",
                text="Article 16 requires the filing of the annual return.",
            )
        ],
    )

    assert result is None
    retriever.retrieve_pages.assert_not_awaited()


@pytest.mark.asyncio
async def test_grounding_sidecar_leaves_broad_free_text_on_legacy_path() -> None:
    selector, retriever = _make_selector(retrieved_pages=[])

    result = await selector.select_page_ids(
        query="Summarize the law in plain language.",
        answer="Summary.",
        answer_type="free_text",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="law:0:0:title",
                doc_id="law",
                section_path="page:1",
                text="Title page.",
            )
        ],
    )

    assert result is None
    retriever.retrieve_pages.assert_not_awaited()


@pytest.mark.asyncio
async def test_grounding_sidecar_returns_empty_pages_for_negative_null_query() -> None:
    selector, retriever = _make_selector(retrieved_pages=[])

    result = await selector.select_page_ids(
        query="What was the jury finding in this DIFC matter?",
        answer="There is no information on this question.",
        answer_type="free_text",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="case-doc:0:0:title",
                doc_id="case-doc",
                doc_title="CFI 001/2024 Alpha Holdings",
                section_path="page:1",
                text="Case title page.",
            )
        ],
    )

    assert result == []
    retriever.retrieve_pages.assert_not_awaited()


@pytest.mark.asyncio
async def test_grounding_sidecar_compare_scope_limits_doc_scope_to_requested_cases() -> None:
    selector, retriever = _make_selector(
        retrieved_pages=[
            RetrievedPage(
                page_id="case-a-doc_2",
                doc_id="case-a-doc",
                page_num=2,
                doc_title="CFI 001/2024 Alpha Holdings",
                doc_type="case_law",
                page_text="Issued by: Justice A.",
                score=0.93,
                page_role="issued_by_block",
            ),
            RetrievedPage(
                page_id="case-b-doc_3",
                doc_id="case-b-doc",
                page_num=3,
                doc_title="CFI 002/2024 Beta Capital",
                doc_type="case_law",
                page_text="Issued by: Justice B.",
                score=0.91,
                page_role="issued_by_block",
            ),
        ]
    )

    result = await selector.select_page_ids(
        query="Who was the judge in CFI 001/2024 and CFI 002/2024?",
        answer="Justice A; Justice B",
        answer_type="names",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="case-a-doc:1:0:judge",
                doc_id="case-a-doc",
                doc_title="CFI 001/2024 Alpha Holdings",
                section_path="page:2",
                text="Issued by: Justice A.",
            ),
            _make_ranked_chunk(
                chunk_id="case-b-doc:2:0:judge",
                doc_id="case-b-doc",
                doc_title="CFI 002/2024 Beta Capital",
                section_path="page:3",
                text="Issued by: Justice B.",
            ),
            _make_ranked_chunk(
                chunk_id="noise-doc:0:0:title",
                doc_id="noise-doc",
                doc_title="CFI 003/2024 Noise Corp",
                section_path="page:1",
                text="Noise case title page.",
            ),
        ],
    )

    assert result == ["case-a-doc_2", "case-b-doc_3"]
    kwargs = retriever.retrieve_pages.await_args.kwargs
    assert kwargs["doc_ids"] == ["case-a-doc", "case-b-doc"]
    assert "issued_by_block" in kwargs["page_roles"]


@pytest.mark.asyncio
async def test_grounding_sidecar_full_case_scope_excludes_unrelated_docs() -> None:
    selector, retriever = _make_selector(
        retrieved_pages=[
            RetrievedPage(
                page_id="case-doc-judgment_5",
                doc_id="case-doc-judgment",
                page_num=5,
                doc_title="CFI 001/2024 Alpha Holdings Judgment",
                doc_type="case_law",
                page_text="Issued by: Justice A.",
                score=0.92,
                page_role="issued_by_block",
            ),
            RetrievedPage(
                page_id="case-doc-order_2",
                doc_id="case-doc-order",
                page_num=2,
                doc_title="CFI 001/2024 Alpha Holdings Order",
                doc_type="case_law",
                page_text="Issued by: Justice A.",
                score=0.89,
                page_role="issued_by_block",
            ),
        ]
    )

    result = await selector.select_page_ids(
        query="Look through all documents in CFI 001/2024 and tell me who the judge was.",
        answer="Justice A",
        answer_type="name",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="case-doc-judgment:4:0:judge",
                doc_id="case-doc-judgment",
                doc_title="CFI 001/2024 Alpha Holdings Judgment",
                section_path="page:5",
                text="Issued by: Justice A.",
            ),
            _make_ranked_chunk(
                chunk_id="case-doc-order:1:0:judge",
                doc_id="case-doc-order",
                doc_title="CFI 001/2024 Alpha Holdings Order",
                section_path="page:2",
                text="Issued by: Justice A.",
            ),
            _make_ranked_chunk(
                chunk_id="noise-doc:0:0:title",
                doc_id="noise-doc",
                doc_title="CFI 999/2024 Noise Corp",
                section_path="page:1",
                text="Noise case title page.",
            ),
        ],
    )

    assert result == ["case-doc-judgment_5", "case-doc-order_2"]
    kwargs = retriever.retrieve_pages.await_args.kwargs
    assert kwargs["doc_ids"] == ["case-doc-judgment", "case-doc-order"]
    assert "issued_by_block" in kwargs["page_roles"]


@pytest.mark.asyncio
async def test_grounding_sidecar_uses_trained_page_scorer_for_compare_scope() -> None:
    selector, _retriever = _make_selector(
        retrieved_pages=[
            RetrievedPage(
                page_id="case-a-doc_2",
                doc_id="case-a-doc",
                page_num=2,
                doc_title="CFI 001/2024 Alpha Holdings",
                doc_type="case_law",
                page_text="Issued by: Justice A.",
                score=0.95,
                page_role="issued_by_block",
            ),
            RetrievedPage(
                page_id="case-a-doc_3",
                doc_id="case-a-doc",
                page_num=3,
                doc_title="CFI 001/2024 Alpha Holdings",
                doc_type="case_law",
                page_text="Issued by: Justice A.",
                score=0.93,
                page_role="issued_by_block",
            ),
            RetrievedPage(
                page_id="case-b-doc_3",
                doc_id="case-b-doc",
                page_num=3,
                doc_title="CFI 002/2024 Beta Capital",
                doc_type="case_law",
                page_text="Issued by: Justice B.",
                score=0.91,
                page_role="issued_by_block",
            ),
        ]
    )
    selector._trained_page_scorer = SimpleNamespace(
        model_path="/tmp/runtime-safe-page-scorer.joblib",
        rank_pages=MagicMock(
            return_value=RuntimePageScorerResult(
                used=True,
                model_path="/tmp/runtime-safe-page-scorer.joblib",
                ranked_page_ids=["case-a-doc_3", "case-b-doc_3", "case-a-doc_2"],
            )
        ),
    )
    collector = TelemetryCollector(request_id="trained-page-scorer-compare")

    result = await selector.select_page_ids(
        query="Who was the judge in CFI 001/2024 and CFI 002/2024?",
        answer="Justice A; Justice B",
        answer_type="names",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="case-a-doc:1:0:judge",
                doc_id="case-a-doc",
                doc_title="CFI 001/2024 Alpha Holdings",
                section_path="page:2",
                text="Issued by: Justice A.",
            ),
            _make_ranked_chunk(
                chunk_id="case-b-doc:2:0:judge",
                doc_id="case-b-doc",
                doc_title="CFI 002/2024 Beta Capital",
                section_path="page:3",
                text="Issued by: Justice B.",
            ),
        ],
        current_used_ids=["case-a-doc:1:0:judge", "case-b-doc:2:0:judge"],
        collector=collector,
    )

    assert result == ["case-a-doc_3", "case-b-doc_3"]
    telemetry = collector.finalize()
    assert telemetry.trained_page_scorer_used is True
    assert telemetry.trained_page_scorer_model_path == "/tmp/runtime-safe-page-scorer.joblib"
    assert telemetry.trained_page_scorer_page_ids == ["case-a-doc_3", "case-b-doc_3", "case-a-doc_2"]
    assert telemetry.trained_page_scorer_fallback_reason == ""


@pytest.mark.asyncio
async def test_grounding_sidecar_reranks_candidate_subset_when_context_only_page_exists() -> None:
    selector, _retriever = _make_selector(
        retrieved_pages=[
            RetrievedPage(
                page_id="case-a-doc_2",
                doc_id="case-a-doc",
                page_num=2,
                doc_title="CFI 001/2024 Alpha Holdings",
                doc_type="case_law",
                page_text="Issued by: Justice A.",
                score=0.95,
                page_role="issued_by_block",
            ),
            RetrievedPage(
                page_id="case-b-doc_3",
                doc_id="case-b-doc",
                page_num=3,
                doc_title="CFI 002/2024 Beta Capital",
                doc_type="case_law",
                page_text="Issued by: Justice B.",
                score=0.91,
                page_role="issued_by_block",
            ),
        ]
    )
    fake_ranker = MagicMock(
        return_value=RuntimePageScorerResult(
            used=True,
            model_path="/tmp/runtime-safe-page-scorer.joblib",
            ranked_page_ids=["case-a-doc_2", "case-b-doc_3"],
        )
    )
    selector._trained_page_scorer = SimpleNamespace(
        model_path="/tmp/runtime-safe-page-scorer.joblib",
        rank_pages=fake_ranker,
    )
    collector = TelemetryCollector(request_id="trained-page-scorer-fallback")

    result = await selector.select_page_ids(
        query="Who was the judge in CFI 001/2024 and CFI 002/2024?",
        answer="Justice A; Justice B",
        answer_type="names",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="case-a-doc:3:0:judge",
                doc_id="case-a-doc",
                doc_title="CFI 001/2024 Alpha Holdings",
                section_path="page:4",
                text="Issued by: Justice A.",
            ),
            _make_ranked_chunk(
                chunk_id="case-b-doc:2:0:judge",
                doc_id="case-b-doc",
                doc_title="CFI 002/2024 Beta Capital",
                section_path="page:3",
                text="Issued by: Justice B.",
            ),
        ],
        current_used_ids=["case-a-doc:3:0:judge", "case-b-doc:2:0:judge"],
        collector=collector,
    )

    assert result == ["case-a-doc_2", "case-b-doc_3"]
    fake_ranker.assert_called_once()
    telemetry = collector.finalize()
    assert telemetry.trained_page_scorer_used is True
    assert telemetry.trained_page_scorer_page_ids == ["case-a-doc_2", "case-a-doc_4", "case-b-doc_3"]
    assert telemetry.trained_page_scorer_fallback_reason == ""


@pytest.mark.asyncio
async def test_grounding_sidecar_fails_closed_for_invalid_ranked_candidate_subset() -> None:
    selector, _retriever = _make_selector(
        retrieved_pages=[
            RetrievedPage(
                page_id="case-a-doc_2",
                doc_id="case-a-doc",
                page_num=2,
                doc_title="CFI 001/2024 Alpha Holdings",
                doc_type="case_law",
                page_text="Issued by: Justice A.",
                score=0.95,
                page_role="issued_by_block",
            ),
            RetrievedPage(
                page_id="case-b-doc_3",
                doc_id="case-b-doc",
                page_num=3,
                doc_title="CFI 002/2024 Beta Capital",
                doc_type="case_law",
                page_text="Issued by: Justice B.",
                score=0.91,
                page_role="issued_by_block",
            ),
        ]
    )
    selector._trained_page_scorer = SimpleNamespace(
        model_path="/tmp/runtime-safe-page-scorer.joblib",
        rank_pages=MagicMock(
            return_value=RuntimePageScorerResult(
                used=True,
                model_path="/tmp/runtime-safe-page-scorer.joblib",
                ranked_page_ids=["case-a-doc_2"],
            )
        ),
    )
    collector = TelemetryCollector(request_id="trained-page-scorer-invalid-subset")

    result = await selector.select_page_ids(
        query="Who was the judge in CFI 001/2024 and CFI 002/2024?",
        answer="Justice A; Justice B",
        answer_type="names",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="case-a-doc:3:0:judge",
                doc_id="case-a-doc",
                doc_title="CFI 001/2024 Alpha Holdings",
                section_path="page:4",
                text="Issued by: Justice A.",
            ),
            _make_ranked_chunk(
                chunk_id="case-b-doc:2:0:judge",
                doc_id="case-b-doc",
                doc_title="CFI 002/2024 Beta Capital",
                section_path="page:3",
                text="Issued by: Justice B.",
            ),
        ],
        current_used_ids=["case-a-doc:3:0:judge", "case-b-doc:2:0:judge"],
        collector=collector,
    )

    assert result == ["case-b-doc_3", "case-a-doc_4"]
    telemetry = collector.finalize()
    assert telemetry.trained_page_scorer_used is False
    assert telemetry.trained_page_scorer_fallback_reason == "ranked_subset_mismatch"
