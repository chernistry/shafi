from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from shafi.core.grounding.evidence_selector import GroundingEvidenceSelector
from shafi.core.grounding.query_scope_classifier import (
    classify_query_scope,
    extract_explicit_page_numbers,
)
from shafi.core.grounding.relevance_verifier import RelevanceVerificationResult
from shafi.ml.page_scorer_runtime import RuntimePageScorerResult
from shafi.models import DocType, QueryScopePrediction, RankedChunk, RetrievedPage, ScopeMode
from shafi.telemetry import TelemetryCollector


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


def _make_selector(
    *,
    retrieved_pages: list[RetrievedPage],
    llm_provider: object | None = None,
) -> tuple[GroundingEvidenceSelector, MagicMock]:
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
        grounding_escalation_enabled=True,
        grounding_low_rerank_margin_threshold=0.06,
        grounding_close_page_margin_threshold=0.2,
        grounding_authority_strength_threshold=0.95,
        grounding_relevance_verifier_enabled=True,
        grounding_relevance_verifier_min_confidence=0.7,
        grounding_relevance_verifier_max_candidates=3,
    )
    verifier_settings = SimpleNamespace(max_tokens=300, temperature=0.0)

    selector = GroundingEvidenceSelector(
        retriever=retriever,
        store=store,
        embedder=embedder,
        sparse_encoder=None,
        pipeline_settings=settings,
        verifier_settings=verifier_settings,
        llm_provider=llm_provider,
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


def test_multi_law_metadata_compare_uses_compare_pair_scope() -> None:
    """Multi-law date/commencement queries must use COMPARE_PAIR scope.

    SINGLE_FIELD_SINGLE_DOC could fill both page slots from the same law,
    leaving the second law unrepresented (root cause of false-null cases
    bb67fc19, b249b41b, d5bc7441).  COMPARE_PAIR.safe-sidecar selects
    one page per unique doc_id, ensuring both laws are covered.
    """
    for query in (
        "Were the IP Law and the Employment Law enacted in the same year?",
        "What is the commencement date of the Strata Title Law and the Financial Collateral Law?",
        "Which of the Leasing Law and the Real Property Law came into force earlier?",
    ):
        prediction = classify_query_scope(query, "boolean")
        assert prediction.scope_mode is ScopeMode.COMPARE_PAIR, (
            f"Expected COMPARE_PAIR for multi-law metadata query, got {prediction.scope_mode}: {query!r}"
        )
        assert "title_cover" in prediction.target_page_roles


@pytest.mark.asyncio
async def test_grounding_sidecar_activates_single_doc_date_scope_on_authority_path() -> None:
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

    assert result == ["law_2"]
    retriever.retrieve_pages.assert_awaited_once()


@pytest.mark.asyncio
async def test_grounding_sidecar_activates_single_doc_scope_with_two_docs() -> None:
    """Sidecar now activates for ≤3 doc SINGLE_FIELD queries (dual-case fix)."""
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

    # Sidecar activates for 2-doc queries and selects pages from context
    assert result is not None
    assert len(result) >= 1
    assert retriever.retrieve_pages.await_count >= 1


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
async def test_grounding_sidecar_activates_single_doc_article_scope_on_authority_path() -> None:
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

    assert result == ["law_16"]
    retriever.retrieve_pages.assert_awaited_once()


@pytest.mark.asyncio
async def test_grounding_sidecar_activates_broad_free_text_single_doc() -> None:
    """Sidecar activates for BROAD_FREE_TEXT with ≤3 docs (expanded gate)."""
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

    # Sidecar activates for single-doc BROAD_FREE_TEXT (expanded gate)
    assert result is not None
    assert len(result) >= 1
    assert retriever.retrieve_pages.await_count >= 1


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
async def test_grounding_sidecar_shadow_lane_promotes_heading_match() -> None:
    selector, _retriever = _make_selector(
        retrieved_pages=[
            RetrievedPage(
                page_id="case-doc_1",
                doc_id="case-doc",
                page_num=1,
                doc_title="CFI 001/2024 Alpha Holdings",
                doc_type="case_law",
                page_text="General material.",
                score=0.96,
                page_role="other",
                heading_lines=["General Provisions"],
            ),
            RetrievedPage(
                page_id="case-doc_2",
                doc_id="case-doc",
                page_num=2,
                doc_title="CFI 001/2024 Alpha Holdings",
                doc_type="case_law",
                page_text="Issued by: Justice A.",
                score=0.7,
                page_role="issued_by_block",
                heading_lines=["Issued by"],
                field_labels_present=["Issued by"],
                page_template_family="issued_by_authority",
                officialness_score=0.9,
                source_vs_reference_prior=0.9,
                has_issued_by_pattern=True,
            ),
            RetrievedPage(
                page_id="case-doc-2_3",
                doc_id="case-doc-2",
                page_num=3,
                doc_title="CFI 002/2024 Beta Capital",
                doc_type="case_law",
                page_text="Issued by: Justice B.",
                score=0.85,
                page_role="issued_by_block",
                heading_lines=["Issued by"],
                field_labels_present=["Issued by"],
                page_template_family="issued_by_authority",
                officialness_score=0.9,
                source_vs_reference_prior=0.9,
                has_issued_by_pattern=True,
            ),
        ]
    )

    result = await selector.select_page_ids(
        query="Who was the judge in CFI 001/2024 and CFI 002/2024?",
        answer="Justice A; Justice B",
        answer_type="names",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="case-doc:1:0:judge",
                doc_id="case-doc",
                doc_title="CFI 001/2024 Alpha Holdings",
                section_path="page:2",
                text="Issued by: Justice A.",
            ),
            _make_ranked_chunk(
                chunk_id="case-doc-2:2:0:judge",
                doc_id="case-doc-2",
                doc_title="CFI 002/2024 Beta Capital",
                section_path="page:3",
                text="Issued by: Justice B.",
            ),
        ],
    )

    assert "case-doc_2" in result
    assert "case-doc_1" not in result
    assert "case-doc-2_3" in result


@pytest.mark.asyncio
async def test_grounding_sidecar_full_case_schedule_pair_stays_bounded_to_two_pages() -> None:
    selector, _retriever = _make_selector(
        retrieved_pages=[
            RetrievedPage(
                page_id="schedule-doc_1",
                doc_id="schedule-doc",
                page_num=1,
                doc_title="Fee Schedule",
                doc_type="statute",
                page_text="Schedule title page.",
                score=0.9,
                page_role="title_cover",
                page_template_family="title_cover",
                officialness_score=0.9,
                source_vs_reference_prior=0.9,
            ),
            RetrievedPage(
                page_id="schedule-doc_4",
                doc_id="schedule-doc",
                page_num=4,
                doc_title="Fee Schedule",
                doc_type="statute",
                page_text="Schedule table with fees.",
                score=0.89,
                page_role="schedule_table",
                page_template_family="schedule_table",
                officialness_score=0.85,
                source_vs_reference_prior=0.9,
            ),
        ]
    )

    result = await selector.select_page_ids(
        query="Across all documents in CFI 001/2024, what schedule table lists the fee?",
        answer="Fee schedule.",
        answer_type="free_text",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="schedule-doc:0:0:title",
                doc_id="schedule-doc",
                doc_title="CFI 001/2024 Fee Schedule",
                section_path="page:1",
                text="Schedule title page.",
            ),
            _make_ranked_chunk(
                chunk_id="schedule-doc:3:0:table",
                doc_id="schedule-doc",
                doc_title="CFI 001/2024 Fee Schedule",
                section_path="page:4",
                text="Schedule table with fees.",
            ),
        ],
    )

    assert result == ["schedule-doc_1", "schedule-doc_4"]


@pytest.mark.asyncio
async def test_grounding_sidecar_relevance_verifier_fail_closed_keeps_deterministic_page() -> None:
    selector, retriever = _make_selector(
        retrieved_pages=[
            RetrievedPage(
                page_id="law_1",
                doc_id="law",
                page_num=1,
                doc_title="Employment Law",
                doc_type="statute",
                page_text="General introduction.",
                score=0.82,
                page_role="other",
                page_template_family="duplicate_or_reference_like",
                officialness_score=0.2,
                source_vs_reference_prior=0.1,
            ),
            RetrievedPage(
                page_id="law_2",
                doc_id="law",
                page_num=2,
                doc_title="Employment Law",
                doc_type="statute",
                page_text="Article 16 requires annual returns.",
                score=0.8,
                page_role="article_clause",
                page_template_family="article_body",
                article_refs=["Article 16"],
                officialness_score=0.85,
                source_vs_reference_prior=0.85,
            ),
        ]
    )
    selector._settings.grounding_authority_strength_threshold = 10.0

    class _FakeVerifier:
        async def verify(self, **_kwargs: object) -> RelevanceVerificationResult:
            return RelevanceVerificationResult(
                used=False,
                selected_page_ids=(),
                selection_mode="empty",
                confidence=0.0,
                candidate_assessments=(),
                reasons=("invalid_json",),
                fallback_reason="invalid_json",
            )

    selector._relevance_verifier = _FakeVerifier()  # pyright: ignore[reportAttributeAccessIssue]
    collector = TelemetryCollector("req-grounding-verifier-fallback")

    result = await selector.select_page_ids(
        query="According to Article 16 of the law, what document must be filed?",
        answer="annual return",
        answer_type="name",
        context_chunks=[
            _make_ranked_chunk(
                chunk_id="law:0:0:intro",
                doc_id="law",
                section_path="page:1",
                text="General introduction.",
            ),
            _make_ranked_chunk(
                chunk_id="law:1:0:article",
                doc_id="law",
                section_path="page:2",
                text="Article 16 requires annual returns.",
            ),
        ],
        collector=collector,
    )

    telemetry = collector.finalize()
    assert result == ["law_2"]
    assert retriever.retrieve_pages.await_count >= 1
    assert telemetry.grounding_relevance_verifier_used is False
    assert telemetry.grounding_relevance_verifier_fallback_reason == "invalid_json"


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

    assert result == ["case-a-doc_2", "case-b-doc_3"]
    telemetry = collector.finalize()
    assert telemetry.trained_page_scorer_used is False
    assert telemetry.trained_page_scorer_fallback_reason == "ranked_subset_mismatch"


def test_compare_pair_name_type_prefers_page_1_over_high_scored_content_page() -> None:
    """COMPARE_PAIR + name: sidecar must prefer doc_1 (case header) per doc.

    NOGA-24a root cause: 5 case-comparison name Qs cite doc_18/doc_7 instead
    of doc_1 (-3.5pp G). Reranker scores content pages higher, but judge/party
    names live on the case cover/header (page 1).
    """
    scope = QueryScopePrediction(
        scope_mode=ScopeMode.COMPARE_PAIR,
        target_page_roles=["issued_by_block", "title_cover"],
        page_budget=2,
    )
    # Simulate reranker-ordered list: content pages rank above header pages.
    ordered = [
        "case-a-doc_18",  # substantive ruling page — scored highest
        "case-a-doc_1",  # case header/cover — scored lower
        "case-b-doc_7",  # content page
        "case-b-doc_1",  # case header/cover
    ]
    result = GroundingEvidenceSelector._select_safe_sidecar_pages(
        query="Who was the judge in CFI 001/2024 and CFI 002/2024?",
        scope=scope,
        answer_type="name",
        ordered=ordered,
        ordered_pages=[],
        page_candidates=[],
    )
    # Must prefer page _1 for each doc over higher-scored content pages.
    assert result == ["case-a-doc_1", "case-b-doc_1"]


def test_compare_pair_boolean_type_prefers_page_1_for_enactment_date_questions() -> None:
    """COMPARE_PAIR + boolean: sidecar must prefer doc_1 (law cover with enactment date).

    Root cause of bb67fc19/d5bc7441 false nulls: multi-law boolean date-comparison
    questions (e.g. "Was IP Law enacted earlier than Employment Law?") need the law
    cover page (page _1) which has the enacted year. Without page_1, LLM returns null.
    """
    scope = QueryScopePrediction(
        scope_mode=ScopeMode.COMPARE_PAIR,
        target_page_roles=["title_cover"],
        page_budget=2,
    )
    ordered = [
        "law-a-doc_5",  # article content page — scored highest by reranker
        "law-a-doc_1",  # law cover with "DIFC LAW NO. 2 OF 2019, ENACTED 2019"
        "law-b-doc_3",  # content page
        "law-b-doc_1",  # law cover with enactment date
    ]
    result = GroundingEvidenceSelector._select_safe_sidecar_pages(
        query="Was the Intellectual Property Law enacted earlier in the year than the Employment Law?",
        scope=scope,
        answer_type="boolean",
        ordered=ordered,
        ordered_pages=[],
        page_candidates=[],
    )
    # boolean + COMPARE_PAIR: must prefer page _1 (law cover with enactment date).
    assert result == ["law-a-doc_1", "law-b-doc_1"]


def test_compare_pair_number_type_keeps_highest_scored_page() -> None:
    """COMPARE_PAIR for number/date/free_text must NOT prefer page_1 — keeps reranker order."""
    scope = QueryScopePrediction(
        scope_mode=ScopeMode.COMPARE_PAIR,
        target_page_roles=["title_cover"],
        page_budget=2,
    )
    ordered = [
        "case-a-doc_18",
        "case-a-doc_1",
        "case-b-doc_7",
        "case-b-doc_1",
    ]
    result = GroundingEvidenceSelector._select_safe_sidecar_pages(
        query="What was the claim amount in CFI 001/2024 vs CFI 002/2024?",
        scope=scope,
        answer_type="number",
        ordered=ordered,
        ordered_pages=[],
        page_candidates=[],
    )
    # number type: keep highest-scored page per doc (original per_doc_cmp behavior).
    assert result == ["case-a-doc_18", "case-b-doc_7"]
