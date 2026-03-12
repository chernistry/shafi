from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rag_challenge.models import Chunk, DocType, DocumentSection, ParsedDocument, ProvidedChunk


@pytest.fixture
def mock_chunker_settings():
    settings = SimpleNamespace(
        ingestion=SimpleNamespace(
            chunk_size_tokens=100,
            chunk_overlap_tokens=10,
            sac_summary_max_tokens=150,
            sac_summary_min_tokens=80,
        )
    )
    with patch("rag_challenge.ingestion.chunker.get_settings", return_value=settings):
        yield settings


@pytest.fixture
def mock_sac_settings():
    settings = SimpleNamespace(
        ingestion=SimpleNamespace(
            sac_summary_max_tokens=150,
            sac_summary_min_tokens=80,
            sac_doc_excerpt_chars=3000,
        ),
        llm=SimpleNamespace(summary_model="gpt-4o-mini"),
    )
    with patch("rag_challenge.ingestion.sac.get_settings", return_value=settings):
        yield settings


def _make_doc(
    text: str,
    *,
    sections: list[DocumentSection] | None = None,
    provided_chunks: list[ProvidedChunk] | None = None,
) -> ParsedDocument:
    return ParsedDocument(
        doc_id="test_doc",
        title="Test Document",
        doc_type=DocType.STATUTE,
        jurisdiction="US",
        full_text=text,
        sections=sections or [],
        provided_chunks=provided_chunks or [],
    )


def test_chunk_short_document(mock_chunker_settings):
    from rag_challenge.ingestion.chunker import LegalChunker

    chunker = LegalChunker()
    doc = _make_doc("This is a short document about legal matters.")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].chunk_text == "This is a short document about legal matters."
    assert chunks[0].doc_id == "test_doc"
    assert chunks[0].chunk_text_for_embedding == chunks[0].chunk_text


def test_chunk_respects_sections(mock_chunker_settings):
    from rag_challenge.ingestion.chunker import LegalChunker

    chunker = LegalChunker()
    sections = [
        DocumentSection(
            heading="Section 1",
            section_path="Section 1",
            text=("Text of section one. " * 60).strip(),
            level=1,
        ),
        DocumentSection(
            heading="Section 2",
            section_path="Section 2",
            text=("Text of section two. " * 60).strip(),
            level=1,
        ),
    ]
    chunks = chunker.chunk_document(_make_doc("", sections=sections))

    assert len(chunks) >= 2
    section_paths = {chunk.section_path for chunk in chunks}
    assert "Section 1" in section_paths
    assert "Section 2" in section_paths


def test_chunk_id_deterministic(mock_chunker_settings):
    from rag_challenge.ingestion.chunker import LegalChunker

    chunker = LegalChunker()
    doc = _make_doc("Some legal text here.")
    chunks1 = chunker.chunk_document(doc)
    chunks2 = chunker.chunk_document(doc)

    assert chunks1[0].chunk_id == chunks2[0].chunk_id


def test_preserves_provided_chunk_ids(mock_chunker_settings):
    from rag_challenge.ingestion.chunker import LegalChunker

    doc = _make_doc(
        "",
        provided_chunks=[
            ProvidedChunk(chunk_id="EVAL_CHUNK_1", text="Text 1", section_path="Section 1"),
            ProvidedChunk(chunk_id="EVAL_CHUNK_2", text="Text 2", section_path="Section 2"),
        ],
    )
    chunker = LegalChunker()
    chunks = chunker.chunk_document(doc)

    assert [chunk.chunk_id for chunk in chunks] == ["EVAL_CHUNK_1", "EVAL_CHUNK_2"]
    assert chunks[0].chunk_text == "Text 1"
    assert chunks[1].section_path == "Section 2"


def test_citation_extraction(mock_chunker_settings):
    from rag_challenge.ingestion.chunker import LegalChunker

    chunker = LegalChunker()
    doc = _make_doc("Under § 501(a), the plaintiff in Smith v. Jones argued that 42 USC § 1983 applies.")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert any("§ 501" in citation for citation in chunks[0].citations)
    assert any("Smith v. Jones" in citation for citation in chunks[0].citations)


def test_difc_identifier_citations_include_doc_title_refs(mock_chunker_settings):
    from rag_challenge.ingestion.chunker import LegalChunker

    chunker = LegalChunker()
    doc = ParsedDocument(
        doc_id="test_doc",
        title="cfi 10/2024 Fursa Consulting v Bay Gate Investment LLC",
        doc_type=DocType.CASE_LAW,
        jurisdiction="DIFC",
        full_text="Some body text without the identifier.",
        sections=[],
        provided_chunks=[],
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert "CFI 010/2024" in chunks[0].citations


def test_difc_law_title_and_schedule_citations(mock_chunker_settings):
    from rag_challenge.ingestion.chunker import LegalChunker

    chunker = LegalChunker()
    doc = _make_doc("In the Trust Law 2018, see Schedule 1 and Article 5(2).")
    chunks = chunker.chunk_document(doc)

    assert "Trust Law 2018" in chunks[0].citations
    assert "Schedule 1" in chunks[0].citations
    assert "Article 5(2)" in chunks[0].citations


def test_defined_term_extraction(mock_chunker_settings):
    from rag_challenge.ingestion.chunker import LegalChunker

    chunker = LegalChunker()
    doc = _make_doc('The "Service Agreement" defines "Confidential Information" as follows.')
    chunks = chunker.chunk_document(doc)

    assert "Service Agreement" in chunks[0].anchors
    assert "Confidential Information" in chunks[0].anchors


def test_count_tokens(mock_chunker_settings):
    from rag_challenge.ingestion.chunker import LegalChunker

    chunker = LegalChunker()
    assert chunker.count_tokens("hello world") > 0


def test_chunker_adds_anchor_microchunks_for_pdf_pages(mock_chunker_settings):
    from rag_challenge.ingestion.chunker import LegalChunker

    chunker = LegalChunker()
    sections = [
        DocumentSection(
            heading="Page 1",
            section_path="page:1",
            text=(
                "BETWEEN\n"
                "Architeriors Interior Design (L.L.C)\n"
                "Applicant\n"
                "and\n"
                "Coinmena B.S.C. (C)\n"
                "Respondent\n\n"
                "Article 5 Definitions\n"
                "This Law may be cited as the Operating Law 2018.\n"
            ),
            level=0,
            page_number=1,
            page_type="page",
        ),
        DocumentSection(
            heading="Page 2",
            section_path="page:2",
            text=(
                "Claim No. ENF-316-2023/2\n"
                "IT IS HEREBY ORDERED THAT the appeal is dismissed with costs.\n"
            ),
            level=0,
            page_number=2,
            page_type="page",
        ),
    ]

    chunks = chunker.chunk_document(_make_doc("", sections=sections))

    page_types = {chunk.page_type for chunk in chunks if chunk.page_type}
    assert "title_anchor" in page_types
    assert "caption_anchor" in page_types
    assert "page2_anchor" in page_types
    assert "heading_window" in page_types

    title_anchor = next(chunk for chunk in chunks if chunk.page_type == "title_anchor")
    page2_anchor = next(chunk for chunk in chunks if chunk.page_type == "page2_anchor")
    assert title_anchor.page_number == 1
    assert page2_anchor.page_number == 2
    assert "Operating Law 2018" in title_anchor.doc_refs
    assert page2_anchor.has_order_terms is True


@pytest.mark.asyncio
async def test_sac_generate_doc_summary_and_augment(mock_sac_settings):
    from rag_challenge.ingestion.sac import SACGenerator
    from rag_challenge.llm.provider import LLMResult

    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=LLMResult(
            text="This statute sets limitation periods and key filing deadlines.",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            model="gpt-4o-mini",
            latency_ms=10.0,
        )
    )

    sac = SACGenerator(llm=llm)
    doc = _make_doc("A statute text. " * 300)
    summary = await sac.generate_doc_summary(doc)

    assert summary.startswith("This statute")
    llm.generate.assert_awaited_once()
    kwargs = llm.generate.await_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["max_tokens"] == 150

    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Test Law 2024",
        doc_type=DocType.STATUTE,
        jurisdiction="US",
        section_path="Section 1",
        chunk_text="Original text here.",
        chunk_text_for_embedding="Original text here.",
        doc_summary="",
        citations=["Test Law 2024", "Law No. 7 of 2024", "Article 5"],
        token_count=5,
    )
    augmented = sac.augment_chunks([chunk], summary)

    assert len(augmented) == 1
    assert augmented[0].chunk_text == "Original text here."
    assert "[DOC_TITLE]" in augmented[0].chunk_text_for_embedding
    assert "[DOC_ALIASES]" in augmented[0].chunk_text_for_embedding
    assert "[DOC_SUMMARY]" in augmented[0].chunk_text_for_embedding
    assert "[SECTION_PATH]" in augmented[0].chunk_text_for_embedding
    assert "[CHUNK]" in augmented[0].chunk_text_for_embedding
    assert "Law No. 7 of 2024" in augmented[0].chunk_text_for_embedding
    assert "Article 5" not in augmented[0].chunk_text_for_embedding
    assert augmented[0].doc_summary == summary


def test_sac_augment_without_summary_still_builds_retrieval_text(mock_sac_settings):
    from rag_challenge.ingestion.sac import SACGenerator

    sac = SACGenerator(llm=AsyncMock())
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Test Law 2024",
        doc_type=DocType.STATUTE,
        jurisdiction="US",
        section_path="Page 1",
        chunk_text="Original text here.",
        chunk_text_for_embedding="Original text here.",
        doc_summary="",
        citations=["Law No. 7 of 2024", "Schedule 1"],
        token_count=5,
    )

    augmented = sac.augment_chunks([chunk], "   ")
    assert augmented[0].chunk_text_for_embedding.startswith("[DOC_TITLE]\nTest Law 2024")
    assert "[DOC_SUMMARY]" not in augmented[0].chunk_text_for_embedding
    assert "Law No. 7 of 2024" in augmented[0].chunk_text_for_embedding
    assert "Schedule 1" not in augmented[0].chunk_text_for_embedding
    assert augmented[0].doc_summary == ""
