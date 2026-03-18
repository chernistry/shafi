from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_challenge.core.grounding.evidence_selector import GroundingEvidenceSelector
from rag_challenge.core.grounding.query_scope_classifier import extract_explicit_page_numbers
from rag_challenge.models import DocType, RankedChunk, RetrievedPage


def _make_ranked_chunk(*, chunk_id: str, doc_id: str, section_path: str, text: str) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title="Doc",
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


@pytest.mark.asyncio
async def test_grounding_sidecar_activates_for_single_doc_date_scope() -> None:
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
    kwargs = retriever.retrieve_pages.await_args.kwargs
    assert kwargs["doc_ids"] == ["law"]
    assert "issued_by_block" in kwargs["page_roles"]


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
async def test_grounding_sidecar_passes_article_anchor_filter_for_single_doc_scope() -> None:
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
    kwargs = retriever.retrieve_pages.await_args.kwargs
    assert kwargs["doc_ids"] == ["law"]
    assert kwargs["article_refs"] == ["Article 16"]
    assert "article_clause" in kwargs["page_roles"]


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
