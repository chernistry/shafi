from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Headers
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_challenge.ingestion.canonical_entities import EntityAliasResolver
from rag_challenge.models import DocType, RetrievedChunk
from rag_challenge.models.legal_objects import CaseObject, CaseParty, CorpusRegistry, LawObject, LegalDocType


@pytest.fixture
def mock_settings():
    settings = SimpleNamespace(
        qdrant=SimpleNamespace(
            prefetch_dense=60,
            prefetch_sparse=60,
            use_cloud_inference=True,
            enable_sparse_bm25=True,
            sparse_model="Qdrant/bm25",
            fusion_method="RRF",
            circuit_failure_threshold=3,
            circuit_reset_timeout_s=60.0,
            shadow_collection="legal_chunks_shadow",
            segment_collection="legal_segments",
            bridge_fact_collection="legal_bridge_facts",
        ),
        reranker=SimpleNamespace(rerank_candidates=80),
        pipeline=SimpleNamespace(
            retry_dense_bias=40,
            retry_sparse_bias=90,
            enable_shadow_search_text=False,
            enable_parallel_anchor_retrieval=False,
            enable_entity_boosts=False,
            enable_cross_ref_boosts=False,
            enable_segment_retrieval=False,
            segment_retrieval_top_k=4,
            segment_retrieval_page_budget=4,
            enable_bridge_fact_retrieval=False,
            bridge_fact_retrieval_top_k=6,
            bridge_fact_page_budget=4,
            shadow_retrieval_top_k=24,
            anchor_retrieval_top_k=16,
            enable_rag_fusion=False,
            rag_fusion_extra_prefetch=40,
            case_ref_metadata_first=False,
        ),
    )
    sparse_encoder = MagicMock()
    sparse_encoder.encode_query.return_value = models.SparseVector(indices=[1], values=[0.7])
    with (
        patch("rag_challenge.core.retriever.get_settings", return_value=settings),
        patch("rag_challenge.core.retriever.BM25SparseEncoder", return_value=sparse_encoder),
    ):
        yield settings


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.embed_query = AsyncMock(return_value=[0.1] * 8)
    return embedder


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.collection_name = "legal_chunks"
    store.page_collection_name = "legal_pages"
    store.shadow_collection_name = "legal_chunks_shadow"
    store.client = AsyncMock()
    store.client.collection_exists = AsyncMock(return_value=True)
    store.client.scroll = AsyncMock(return_value=([], None))

    points: list[object] = []
    for i in range(5):
        point = SimpleNamespace(
            id=f"p{i}",
            score=0.9 - (i * 0.1),
            payload={
                "chunk_id": f"c{i}",
                "doc_id": "d1",
                "doc_title": "Test Doc",
                "doc_type": "statute",
                "section_path": f"Section {i}",
                "chunk_text": f"Legal text {i}",
                "doc_summary": "Summary",
            },
        )
        points.append(point)
    store.client.query_points = AsyncMock(return_value=SimpleNamespace(points=points))
    return store


def _make_page_point(
    *,
    page_id: str = "doc-1_16",
    doc_id: str = "doc-1",
    page_num: int = 16,
    page_role: str = "article_clause",
    page_family: str = "body",
    article_refs: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=page_id,
        score=0.95,
        payload={
            "page_id": page_id,
            "doc_id": doc_id,
            "page_num": page_num,
            "doc_title": "Operating Law 2018",
            "doc_type": "statute",
            "page_text": "Article 16 filing obligation",
            "page_role": page_role,
            "page_family": page_family,
            "article_refs": article_refs or ["Article 16"],
            "normalized_refs": ["Article 16"],
            "law_titles": ["Operating Law 2018"],
            "case_numbers": [],
            "amount_roles": [],
            "linked_refs": [],
            "doc_family": "law",
        },
    )


def _build_alias_registry_payload(tmp_path) -> str:
    registry = CorpusRegistry(
        source_doc_count=2,
        laws={
            "companies_law": LawObject(
                object_id="law:companies_law",
                doc_id="companies_law",
                title="DIFC Companies Law No. 3 of 2004",
                legal_doc_type=LegalDocType.LAW,
                short_title="Companies Law",
                law_number="3",
                year="2004",
                issuing_authority="Dubai Financial Services Authority",
            )
        },
        cases={
            "cfi_001_2024": CaseObject(
                object_id="case:cfi_001_2024",
                doc_id="cfi_001_2024",
                title="Alpha Ltd v Dubai Financial Services Authority",
                legal_doc_type=LegalDocType.CASE,
                case_number="CFI 001/2024",
                parties=[
                    CaseParty(name="Dubai Financial Services Authority", role="Respondent"),
                ],
            )
        },
    )
    resolver = EntityAliasResolver.build_from_registry(registry)
    return str(resolver.export(tmp_path / "canonical_entity_registry.json"))


@pytest.mark.asyncio
async def test_retrieve_returns_chunks_and_tracks_ids(mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    chunks = await retriever.retrieve("What is the statute of limitations?")
    debug = retriever.get_last_retrieval_debug()

    assert len(chunks) == 5
    assert chunks[0].chunk_id == "c0"
    assert chunks[0].score == pytest.approx(0.9)
    assert retriever.get_last_retrieved_ids() == [f"c{i}" for i in range(5)]
    assert debug["retrieval_mode"] == "hybrid"
    assert debug["fail_open_triggered"] is False
    assert debug["initial_chunk_count"] == 5
    assert debug["final_chunk_count"] == 5
    assert debug["source_hits"] == {"baseline": 5, "shadow": 0, "anchor": 0, "segment": 0, "bridge": 0}
    mock_embedder.embed_query.assert_awaited_once()
    mock_store.client.query_points.assert_awaited_once()

    kwargs = mock_store.client.query_points.await_args.kwargs
    assert kwargs["limit"] == 80
    assert isinstance(kwargs["query"], models.FusionQuery)
    assert isinstance(kwargs["prefetch"], list)
    assert len(kwargs["prefetch"]) == 2


@pytest.mark.asyncio
async def test_retrieve_breaks_equal_scores_deterministically(mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    points = [
        SimpleNamespace(
            id="p1",
            score=0.9,
            payload={
                "chunk_id": "chunk-b",
                "doc_id": "doc-b",
                "doc_title": "Doc B",
                "doc_type": "statute",
                "section_path": "Section 2",
                "chunk_text": "Text B",
                "doc_summary": "Summary",
            },
        ),
        SimpleNamespace(
            id="p2",
            score=0.9,
            payload={
                "chunk_id": "chunk-c",
                "doc_id": "doc-a",
                "doc_title": "Doc A",
                "doc_type": "statute",
                "section_path": "Section 2",
                "chunk_text": "Text C",
                "doc_summary": "Summary",
            },
        ),
        SimpleNamespace(
            id="p3",
            score=0.9,
            payload={
                "chunk_id": "chunk-a",
                "doc_id": "doc-a",
                "doc_title": "Doc A",
                "doc_type": "statute",
                "section_path": "Section 1",
                "chunk_text": "Text A",
                "doc_summary": "Summary",
            },
        ),
    ]
    mock_store.client.query_points = AsyncMock(return_value=SimpleNamespace(points=points))

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    first = await retriever.retrieve("What is the statute of limitations?")
    second = await retriever.retrieve("What is the statute of limitations?")

    expected_chunk_ids = ["chunk-a", "chunk-c", "chunk-b"]
    assert [chunk.chunk_id for chunk in first] == expected_chunk_ids
    assert [chunk.chunk_id for chunk in second] == expected_chunk_ids


@pytest.mark.asyncio
async def test_payload_boosts_use_canonical_entity_ids(tmp_path, mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    mock_settings.pipeline.enable_entity_boosts = True
    mock_settings.pipeline.canonical_entity_registry_path = _build_alias_registry_payload(tmp_path)

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    boosted = retriever._apply_payload_boosts(
        chunks=[
            RetrievedChunk(
                chunk_id="match",
                doc_id="d1",
                doc_title="Companies Law",
                doc_type=DocType.STATUTE,
                text="Dubai Financial Services Authority acted under the law.",
                score=0.5,
                canonical_entity_ids=[
                    "law_title:companies-law",
                    "party:dubai-financial-services-authority",
                ],
            ),
            RetrievedChunk(
                chunk_id="miss",
                doc_id="d2",
                doc_title="Other",
                doc_type=DocType.STATUTE,
                text="Unrelated text.",
                score=0.55,
            ),
        ],
        query="Did DFSA act under the Companies Law?",
        exact_legal_refs=[],
        retrieval_debug={},
    )

    assert boosted[0].chunk_id == "match"
    assert boosted[0].score > boosted[1].score


@pytest.mark.asyncio
async def test_retrieve_merges_segment_surface_chunks(mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    mock_settings.pipeline.enable_segment_retrieval = True
    baseline_points = mock_store.client.query_points.return_value.points
    segment_point = SimpleNamespace(
        id="segment-1",
        score=0.88,
        payload={
            "segment_id": "law_doc:segment:article:article-1:law_doc_1",
            "segment_type": "article",
            "doc_id": "law_doc",
            "doc_title": "Operating Law 2018",
            "doc_type": "statute",
            "canonical_doc_id": "law:law_doc",
            "legal_path": "Operating Law 2018 > Article 1",
            "text": "Article 1 - Definitions",
            "page_ids": ["law_doc_1"],
            "start_page": 1,
            "end_page": 1,
            "canonical_entity_ids": ["law:law_doc"],
            "search_text": "Operating Law 2018\nArticle 1 - Definitions",
        },
    )
    mock_store.client.query_points = AsyncMock(
        side_effect=[
            SimpleNamespace(points=baseline_points),
            SimpleNamespace(points=[segment_point]),
        ]
    )
    mock_store.client.scroll = AsyncMock(
        return_value=(
            [
                SimpleNamespace(
                    id="segment-chunk",
                    payload={
                        "chunk_id": "segment-chunk",
                        "doc_id": "law_doc",
                        "doc_title": "Operating Law 2018",
                        "doc_type": "statute",
                        "section_path": "page:1",
                        "chunk_text": "Article 1 - Definitions\nApproved Person means a person approved by the Authority.",
                        "doc_summary": "",
                        "segment_id": "law_doc:segment:article:article-1:law_doc_1",
                    },
                )
            ],
            None,
        )
    )

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    chunks = await retriever.retrieve("What does Article 1 say?", query_vector=[0.1] * 8)
    debug = retriever.get_last_retrieval_debug()

    assert any(chunk.chunk_id == "segment-chunk" for chunk in chunks)
    assert debug["segment_retrieval_used"] is True
    assert debug["segment_hit_count"] == 1
    assert debug["source_hits"]["segment"] == 1


@pytest.mark.asyncio
async def test_retrieve_merges_bridge_fact_surface_chunks(mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    mock_settings.pipeline.enable_bridge_fact_retrieval = True
    baseline_points = mock_store.client.query_points.return_value.points
    bridge_point = SimpleNamespace(
        id="bridge-1",
        score=0.83,
        payload={
            "fact_id": "bridge:1",
            "fact_type": "case_party",
            "canonical_text": "Case CFI 001/2024 involves Alpha Ltd as claimant.",
            "source_entity_ids": ["case_number:cfi0012024", "party:alpha-ltd"],
            "source_doc_ids": ["law_doc"],
            "evidence_page_ids": ["law_doc_1"],
            "attributes": {"case_number": "CFI 001/2024", "party_name": "Alpha Ltd"},
        },
    )
    mock_store.client.query_points = AsyncMock(
        side_effect=[
            SimpleNamespace(points=baseline_points),
            SimpleNamespace(points=[bridge_point]),
        ]
    )
    mock_store.client.scroll = AsyncMock(
        return_value=(
            [
                SimpleNamespace(
                    id="bridge-chunk",
                    payload={
                        "chunk_id": "bridge-chunk",
                        "doc_id": "law_doc",
                        "doc_title": "Operating Law 2018",
                        "doc_type": "statute",
                        "section_path": "page:1",
                        "chunk_text": "Alpha Ltd appears in the cited case materials.",
                        "doc_summary": "",
                    },
                )
            ],
            None,
        )
    )

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    chunks = await retriever.retrieve("Which case involves Alpha Ltd?", query_vector=[0.1] * 8)
    debug = retriever.get_last_retrieval_debug()

    assert any(chunk.chunk_id == "bridge-chunk" for chunk in chunks)
    assert debug["bridge_fact_retrieval_used"] is True
    assert debug["bridge_fact_hit_count"] == 1
    assert debug["source_hits"]["bridge"] == 1


@pytest.mark.asyncio
async def test_retrieve_with_retry_changes_bias_and_uses_expanded_query(
    mock_settings,
    mock_embedder,
    mock_store,
):
    from rag_challenge.core.retriever import HybridRetriever

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    chunks = await retriever.retrieve_with_retry(
        "statute of limitations",
        expanded_query="limitation period years statute law",
    )

    assert len(chunks) == 5
    mock_embedder.embed_query.assert_awaited_with("limitation period years statute law")

    kwargs = mock_store.client.query_points.await_args.kwargs
    prefetch = kwargs["prefetch"]
    assert isinstance(prefetch, list)
    dense = prefetch[0]
    sparse = prefetch[1]
    assert dense.limit == 40
    assert sparse.limit == 90


@pytest.mark.asyncio
async def test_retrieve_with_filter_passes_filter_to_prefetch(mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    chunks = await retriever.retrieve(
        "contract terms",
        doc_type_filter=DocType.CONTRACT,
        jurisdiction_filter="US",
    )

    assert len(chunks) == 5
    kwargs = mock_store.client.query_points.await_args.kwargs
    prefetch = kwargs["prefetch"]
    assert isinstance(prefetch, list)
    first_filter = prefetch[0].filter
    assert isinstance(first_filter, models.Filter)
    must = first_filter.must
    assert isinstance(must, list)
    assert len(must) == 2


@pytest.mark.asyncio
async def test_retrieve_with_doc_refs_adds_citation_filter_and_soft_case_law_filter(
    mock_settings,
    mock_embedder,
    mock_store,
):
    from rag_challenge.core.retriever import HybridRetriever

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    await retriever.retrieve("What happened in CFI 010/2024?", doc_refs=["CFI 010/2024"])

    kwargs = mock_store.client.query_points.await_args.kwargs
    prefetch = kwargs["prefetch"]
    assert isinstance(prefetch, list)
    first_filter = prefetch[0].filter
    assert isinstance(first_filter, models.Filter)
    must = first_filter.must
    assert isinstance(must, list)
    keys = {getattr(cond, "key", "") for cond in must}
    assert "doc_type" in keys
    # Ticket 2005: doc-ref filter is now nested as Filter(should=...) inside must
    nested_filters = [c for c in must if isinstance(c, models.Filter)]
    assert len(nested_filters) == 1
    should = nested_filters[0].should
    assert isinstance(should, list)
    assert any(getattr(cond, "key", "") == "citations" for cond in should)


def test_build_filter_adds_doc_title_should_for_title_year_refs():
    from rag_challenge.core.retriever import HybridRetriever

    where = HybridRetriever._build_filter(
        doc_type_filter=None,
        jurisdiction_filter=None,
        doc_refs=["Operating Law 2018"],
    )

    assert isinstance(where, models.Filter)
    # Ticket 2005: doc-ref filter is nested as Filter(should=...) inside must
    must = where.must
    assert isinstance(must, list)
    nested_filters = [c for c in must if isinstance(c, models.Filter)]
    assert len(nested_filters) == 1
    should = nested_filters[0].should
    assert isinstance(should, list)
    assert len(should) == 2
    citations_condition = should[0]
    title_condition = should[1]
    assert getattr(citations_condition, "key", "") == "citations"
    assert getattr(title_condition, "key", "") == "doc_title"
    title_match = getattr(title_condition, "match", None)
    assert isinstance(title_match, models.MatchAny)
    assert "Operating Law" in title_match.any


def test_expand_doc_ref_variants_adds_yearless_title_variants():
    from rag_challenge.core.retriever import HybridRetriever

    variants = HybridRetriever._expand_doc_ref_variants(["Employment Law 2019"])

    assert "Employment Law 2019" in variants
    assert "Employment Law" in variants


def test_build_sparse_query_boosts_exact_legal_refs_for_statute_queries():
    from rag_challenge.core.retriever import HybridRetriever

    sparse_query = HybridRetriever._build_sparse_query(
        query="According to Article 16 ( 1 ) of the Operating Law 2018, what document must be filed?",
        extracted_refs=["Operating Law 2018"],
    )

    assert "According to Article 16 ( 1 ) of the Operating Law 2018" in sparse_query
    assert sparse_query.count("Article 16(1)") >= 2
    assert sparse_query.count("Operating Law 2018") >= 2


def test_build_sparse_query_skips_case_law_queries():
    from rag_challenge.core.retriever import HybridRetriever

    query = "What happened in CFI 010/2024?"
    sparse_query = HybridRetriever._build_sparse_query(query=query, extracted_refs=["CFI 010/2024"])

    assert sparse_query == query


@pytest.mark.asyncio
async def test_dense_only_when_cloud_inference_disabled(mock_embedder, mock_store):
    settings = SimpleNamespace(
        qdrant=SimpleNamespace(
            prefetch_dense=60,
            prefetch_sparse=60,
            use_cloud_inference=False,
            enable_sparse_bm25=False,
            sparse_model="Qdrant/bm25",
            fusion_method="RRF",
            circuit_failure_threshold=3,
            circuit_reset_timeout_s=60.0,
        ),
        reranker=SimpleNamespace(rerank_candidates=80),
        pipeline=SimpleNamespace(retry_dense_bias=40, retry_sparse_bias=90),
    )

    with patch("rag_challenge.core.retriever.get_settings", return_value=settings):
        from rag_challenge.core.retriever import HybridRetriever

        retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
        await retriever.retrieve("query")

    kwargs = mock_store.client.query_points.await_args.kwargs
    assert kwargs["using"] == "dense"
    assert kwargs["query"] == [0.1] * 8


@pytest.mark.asyncio
async def test_dense_only_query_uses_query_filter_argument(mock_embedder, mock_store):
    settings = SimpleNamespace(
        qdrant=SimpleNamespace(
            prefetch_dense=60,
            prefetch_sparse=60,
            use_cloud_inference=False,
            enable_sparse_bm25=False,
            sparse_model="Qdrant/bm25",
            fusion_method="RRF",
            circuit_failure_threshold=3,
            circuit_reset_timeout_s=60.0,
        ),
        reranker=SimpleNamespace(rerank_candidates=80),
        pipeline=SimpleNamespace(retry_dense_bias=40, retry_sparse_bias=90),
    )

    with patch("rag_challenge.core.retriever.get_settings", return_value=settings):
        from rag_challenge.core.retriever import HybridRetriever

        retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
        await retriever.retrieve("query", doc_type_filter=DocType.STATUTE)

    kwargs = mock_store.client.query_points.await_args.kwargs
    assert "query_filter" in kwargs
    assert kwargs["query_filter"] is not None
    assert "filter" not in kwargs


@pytest.mark.asyncio
async def test_sparse_only_query_uses_query_filter_argument(mock_embedder, mock_store):
    settings = SimpleNamespace(
        qdrant=SimpleNamespace(
            prefetch_dense=60,
            prefetch_sparse=60,
            use_cloud_inference=False,
            enable_sparse_bm25=True,
            sparse_model="Qdrant/bm25",
            fusion_method="RRF",
            circuit_failure_threshold=3,
            circuit_reset_timeout_s=60.0,
        ),
        reranker=SimpleNamespace(rerank_candidates=80),
        pipeline=SimpleNamespace(retry_dense_bias=40, retry_sparse_bias=90),
    )
    sparse_encoder = MagicMock()
    sparse_encoder.encode_query.return_value = models.SparseVector(indices=[1], values=[0.7])

    with (
        patch("rag_challenge.core.retriever.get_settings", return_value=settings),
        patch("rag_challenge.core.retriever.BM25SparseEncoder", return_value=sparse_encoder),
    ):
        from rag_challenge.core.retriever import HybridRetriever

        retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
        await retriever.retrieve("query", doc_type_filter=DocType.STATUTE, sparse_only=True)

    kwargs = mock_store.client.query_points.await_args.kwargs
    assert kwargs["using"] == "bm25"
    assert "query_filter" in kwargs
    assert kwargs["query_filter"] is not None
    assert "filter" not in kwargs


def test_map_results_skips_bad_points():
    from rag_challenge.core.retriever import HybridRetriever

    good = SimpleNamespace(
        id="p1",
        score=0.5,
        payload={
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_title": "Doc",
            "doc_type": "contract",
            "section_path": "S1",
            "chunk_text": "Text",
            "doc_summary": "",
        },
    )
    bad = SimpleNamespace(id="p2", score=0.1, payload="not-a-dict")
    result = SimpleNamespace(points=[good, bad])

    chunks = HybridRetriever._map_results(result)
    assert len(chunks) == 1
    assert chunks[0].doc_type == DocType.CONTRACT


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_dense_only_on_hybrid_failure(mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    points = mock_store.client.query_points.return_value
    mock_store.client.query_points = AsyncMock(side_effect=[RuntimeError("bm25 failure"), points])

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    chunks = await retriever.retrieve("query")

    assert len(chunks) == 5
    assert mock_store.client.query_points.await_count == 2
    first_kwargs = mock_store.client.query_points.await_args_list[0].kwargs
    second_kwargs = mock_store.client.query_points.await_args_list[1].kwargs
    assert "prefetch" in first_kwargs
    assert second_kwargs["using"] == "dense"
    debug = retriever.get_last_retrieval_debug()
    assert debug["retrieval_mode"] == "dense_only"
    assert debug["hybrid_degraded_to_dense_only"] is True
    assert debug["dense_only_reason"] == "hybrid_failure"


@pytest.mark.asyncio
async def test_retrieve_disables_cloud_bm25_after_inference_unavailable(mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    points = mock_store.client.query_points.return_value
    inference_error = UnexpectedResponse(
        status_code=500,
        reason_phrase="Internal Server Error",
        content=b'{"status":{"error":"InferenceService is not initialized"}}',
        headers=Headers(),
    )
    mock_store.client.query_points = AsyncMock(side_effect=[inference_error, points, points])

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    chunks_first = await retriever.retrieve("query one")
    chunks_second = await retriever.retrieve("query two")

    assert len(chunks_first) == 5
    assert len(chunks_second) == 5
    assert mock_store.client.query_points.await_count == 3
    third_kwargs = mock_store.client.query_points.await_args_list[2].kwargs
    assert third_kwargs["using"] == "dense"


@pytest.mark.asyncio
async def test_retrieve_records_exact_ref_fail_open_debug(mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    points = mock_store.client.query_points.return_value.points
    mock_store.client.query_points = AsyncMock(
        side_effect=[
            SimpleNamespace(points=[]),
            SimpleNamespace(points=[]),
            SimpleNamespace(points=points),
        ]
    )

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    chunks = await retriever.retrieve(
        "According to Article 16(1) of the Operating Law 2018, what document must be filed?",
        doc_refs=["Operating Law 2018"],
    )
    debug = retriever.get_last_retrieval_debug()

    assert len(chunks) == 5
    assert debug["has_exact_legal_refs_in_query"] is True
    assert debug["initial_doc_ref_filter_applied"] is True
    assert debug["fail_open_triggered"] is True
    assert debug["fail_open_stages"] == ["drop_doc_type", "drop_doc_refs"]
    assert debug["fail_open_stage"] == "drop_doc_refs"
    assert debug["initial_chunk_count"] == 0
    assert debug["drop_doc_type_chunk_count"] == 0
    assert debug["drop_doc_refs_chunk_count"] == 5
    assert debug["final_chunk_count"] == 5
    assert debug["final_doc_ref_filter_applied"] is False
    assert debug["final_doc_type_filter_applied"] == ""


@pytest.mark.asyncio
async def test_retrieve_unions_shadow_results_when_enabled(mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    mock_settings.pipeline.enable_shadow_search_text = True
    baseline_points = mock_store.client.query_points.return_value.points
    shadow_points = [
        SimpleNamespace(
            id="shadow-1",
            score=1.4,
            payload={
                "chunk_id": "shadow-c1",
                "doc_id": "d2",
                "doc_title": "Operating Law 2018",
                "doc_type": "statute",
                "section_path": "page:2",
                "chunk_text": "Article 16 filing obligation",
                "doc_summary": "Summary",
                "article_refs": ["Article 16"],
                "shadow_search_text": "Operating Law 2018 Article 16 filing obligation",
            },
        )
    ]
    mock_store.client.query_points = AsyncMock(
        side_effect=[SimpleNamespace(points=baseline_points), SimpleNamespace(points=shadow_points)]
    )

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    chunks = await retriever.retrieve("According to Article 16 of the Operating Law 2018, what must be filed?")
    debug = retriever.get_last_retrieval_debug()

    assert any(chunk.chunk_id == "shadow-c1" for chunk in chunks)
    assert debug["shadow_collection_used"] is True
    assert debug["source_hits"]["shadow"] == 1
    assert debug["source_survivors"]["shadow"] >= 1


@pytest.mark.asyncio
async def test_retrieve_runs_bounded_anchor_retrieval_when_enabled(mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    mock_settings.pipeline.enable_parallel_anchor_retrieval = True
    baseline_points = mock_store.client.query_points.return_value.points
    anchor_points = [
        SimpleNamespace(
            id="anchor-1",
            score=1.3,
            payload={
                "chunk_id": "anchor-c1",
                "doc_id": "d1",
                "doc_title": "Test Doc",
                "doc_type": "statute",
                "section_path": "page:1",
                "chunk_text": "COMMENCEMENT Article 16",
                "doc_summary": "Summary",
                "chunk_type": "commencement_anchor",
                "article_refs": ["Article 16"],
            },
        )
    ]
    mock_store.client.query_points = AsyncMock(
        side_effect=[SimpleNamespace(points=baseline_points), SimpleNamespace(points=anchor_points)]
    )

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    chunks = await retriever.retrieve("What is the commencement page for Article 16?")
    debug = retriever.get_last_retrieval_debug()

    assert any(chunk.chunk_id == "anchor-c1" for chunk in chunks)
    assert debug["anchor_retrieval_used"] is True
    assert debug["source_hits"]["anchor"] == 1


def test_build_page_filter_includes_all_page_level_constraints() -> None:
    from rag_challenge.core.retriever import HybridRetriever

    where = HybridRetriever._build_page_filter(
        doc_ids=["doc-1"],
        page_nums=[16],
        article_refs=["Article 16"],
        page_roles=["article_clause"],
        page_families=["body"],
    )

    assert isinstance(where, models.Filter)
    must = where.must
    assert isinstance(must, list)
    keys = {getattr(condition, "key", "") for condition in must}
    assert keys == {"doc_id", "page_num", "article_refs", "page_role", "page_family"}


@pytest.mark.asyncio
async def test_retrieve_pages_uses_hybrid_fusion_and_page_filters(mock_settings, mock_embedder, mock_store) -> None:
    from rag_challenge.core.retriever import HybridRetriever

    mock_store.client.query_points = AsyncMock(return_value=SimpleNamespace(points=[_make_page_point()]))
    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)

    pages = await retriever.retrieve_pages(
        "According to Article 16 of the Operating Law 2018, what document must be filed?",
        top_k=3,
        doc_refs=["Operating Law 2018"],
        doc_ids=["doc-1"],
        page_nums=[16],
        article_refs=["Article 16"],
        page_roles=["article_clause"],
        page_families=["body"],
    )

    assert len(pages) == 1
    kwargs = mock_store.client.query_points.await_args.kwargs
    assert kwargs["collection_name"] == "legal_pages"
    assert kwargs["limit"] == 3
    assert isinstance(kwargs["query"], models.FusionQuery)
    prefetch = kwargs["prefetch"]
    assert isinstance(prefetch, list)
    assert len(prefetch) == 2
    for prefetch_query in prefetch:
        assert prefetch_query.limit == 6
        assert isinstance(prefetch_query.filter, models.Filter)
        must = prefetch_query.filter.must
        assert isinstance(must, list)
        keys = {getattr(condition, "key", "") for condition in must}
        assert keys == {"doc_id", "page_num", "article_refs", "page_role", "page_family"}
