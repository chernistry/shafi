from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Headers
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_challenge.models import DocType


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
        ),
        reranker=SimpleNamespace(rerank_candidates=80),
        pipeline=SimpleNamespace(retry_dense_bias=40, retry_sparse_bias=90),
    )
    with patch("rag_challenge.core.retriever.get_settings", return_value=settings):
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
    store.client = AsyncMock()

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


@pytest.mark.asyncio
async def test_retrieve_returns_chunks_and_tracks_ids(mock_settings, mock_embedder, mock_store):
    from rag_challenge.core.retriever import HybridRetriever

    retriever = HybridRetriever(store=mock_store, embedder=mock_embedder)
    chunks = await retriever.retrieve("What is the statute of limitations?")

    assert len(chunks) == 5
    assert chunks[0].chunk_id == "c0"
    assert chunks[0].score == pytest.approx(0.9)
    assert retriever.get_last_retrieved_ids() == [f"c{i}" for i in range(5)]
    mock_embedder.embed_query.assert_awaited_once()
    mock_store.client.query_points.assert_awaited_once()

    kwargs = mock_store.client.query_points.await_args.kwargs
    assert kwargs["limit"] == 80
    assert isinstance(kwargs["query"], models.FusionQuery)
    assert isinstance(kwargs["prefetch"], list)
    assert len(kwargs["prefetch"]) == 2


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
    should = first_filter.should
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
    should = where.should
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
