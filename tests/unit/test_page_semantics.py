from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_challenge.ingestion.page_semantics import (
    catalog_page_semantics,
    classify_document_template_family,
    extract_page_top_structure,
)
from rag_challenge.models import DocType, PageRole, ParsedDocument


def test_extract_page_top_structure_detects_issue_block_fields() -> None:
    top = extract_page_top_structure(
        "DIFC Employment Law\nLaw No. 2 of 2019\nIssued by: The President\nDate of Issue: 01 January 2019",
        page_num=1,
        total_pages=12,
    )

    assert top.top_lines[:2] == ["DIFC Employment Law", "Law No. 2 of 2019"]
    assert "Issued by" in top.field_labels_present
    assert "Date of Issue" in top.field_labels_present
    assert "Law No" in top.field_labels_present
    assert top.has_title_like_header is True
    assert top.has_issued_by_pattern is True
    assert top.has_date_of_issue_pattern is True
    assert top.has_law_number_pattern is True


def test_extract_page_top_structure_detects_enactment_notice_panel_fields() -> None:
    top = extract_page_top_structure(
        "Enactment Notice\nDIFC Employment Law\nLaw No. 2 of 2019\nIssued by: The President\nDate of Issue: 01 January 2019",
        page_num=1,
        total_pages=12,
    )

    assert "Enactment Notice" in top.field_labels_present
    assert "Issued by" in top.field_labels_present
    assert "Date of Issue" in top.field_labels_present
    assert top.has_title_like_header is True
    assert top.has_issued_by_pattern is True
    assert top.has_date_of_issue_pattern is True
    assert top.has_law_number_pattern is True


def test_extract_page_top_structure_detects_case_caption() -> None:
    top = extract_page_top_structure(
        "CFI 010/2024\nACME LTD v BETA LLC\nClaim No. CFI 010/2024\nJUDGMENT",
        page_num=1,
        total_pages=5,
    )

    assert top.has_caption_block is True
    assert top.has_claim_number_pattern is True
    assert "Claim No" in top.field_labels_present


def test_catalog_page_semantics_marks_reference_like_contents() -> None:
    top = extract_page_top_structure("TABLE OF CONTENTS\nSchedule 1\nAppendix A", page_num=2, total_pages=20)

    catalog = catalog_page_semantics(
        doc_family="consolidated_law",
        document_template_family="issued_law_instrument",
        page_family="contents_like",
        page_role=PageRole.OTHER,
        page_text="TABLE OF CONTENTS\nSchedule 1\nAppendix A",
        page_num=2,
        total_pages=20,
        top_structure=top,
    )

    assert catalog.page_template_family == "duplicate_or_reference_like"
    assert catalog.officialness_score == pytest.approx(0.10)
    assert catalog.source_vs_reference_prior == pytest.approx(0.05)


def test_catalog_page_semantics_keeps_plain_title_cover_as_title_cover() -> None:
    top = extract_page_top_structure("DIFC Employment Law\nLaw No. 2 of 2019", page_num=1, total_pages=12)

    catalog = catalog_page_semantics(
        doc_family="consolidated_law",
        document_template_family="issued_law_instrument",
        page_family="citation_title_like",
        page_role=PageRole.TITLE_COVER,
        page_text="DIFC Employment Law\nLaw No. 2 of 2019",
        page_num=1,
        total_pages=12,
        top_structure=top,
    )

    assert catalog.page_template_family == "title_cover"
    assert catalog.officialness_score == pytest.approx(0.92)
    assert catalog.source_vs_reference_prior == pytest.approx(0.95)


def test_catalog_page_semantics_promotes_notice_panel_title_page_to_authority() -> None:
    top = extract_page_top_structure(
        "Enactment Notice\nDIFC Employment Law\nLaw No. 2 of 2019\nIssued by: The President\nDate of Issue: 01 January 2019",
        page_num=1,
        total_pages=12,
    )

    catalog = catalog_page_semantics(
        doc_family="consolidated_law",
        document_template_family="issued_law_instrument",
        page_family="citation_title_like",
        page_role=PageRole.TITLE_COVER,
        page_text="Enactment Notice\nDIFC Employment Law\nLaw No. 2 of 2019\nIssued by: The President\nDate of Issue: 01 January 2019",
        page_num=1,
        total_pages=12,
        top_structure=top,
    )

    assert catalog.page_template_family == "issued_by_authority"
    assert catalog.officialness_score == pytest.approx(0.95)
    assert catalog.source_vs_reference_prior == pytest.approx(0.97)


def test_classify_document_template_family_prefers_captioned_case_file() -> None:
    assert (
        classify_document_template_family(
            "judgment",
            first_page_text="CFI 010/2024\nACME LTD v BETA LLC\nJUDGMENT",
        )
        == "captioned_case_file"
    )


@pytest.mark.asyncio
async def test_upsert_pages_enriches_page_semantics_payload() -> None:
    from rag_challenge.ingestion.pipeline import IngestionPipeline

    settings = SimpleNamespace(
        ingestion=SimpleNamespace(
            ingest_version="v1",
            build_shadow_collection=False,
            manifest_dir="",
            manifest_filename=".rag_challenge_ingestion_manifest.json",
            manifest_hash_chunk_size_bytes=1024 * 1024,
            manifest_schema_version=1,
            sac_concurrency=4,
        )
    )
    parser = MagicMock()
    chunker = MagicMock()
    sac = MagicMock()
    embedder = AsyncMock()
    embedder.embed_documents = AsyncMock(return_value=[[0.1] * 8])
    store = AsyncMock()
    store.upsert_pages = AsyncMock(return_value=1)
    store.ensure_support_fact_collection = AsyncMock()
    store.ensure_support_fact_payload_indexes = AsyncMock()
    store.upsert_support_facts = AsyncMock(return_value=0)

    doc = ParsedDocument(
        doc_id="law-1",
        title="DIFC Employment Law",
        doc_type=DocType.STATUTE,
        sections=[
            {
                "heading": "Page 1",
                "section_path": "page:1",
                "text": "Enactment Notice\nDIFC Employment Law\nLaw No. 2 of 2019\nIssued by: The President\nDate of Issue: 01 January 2019",
                "level": 0,
            }
        ],
    )

    with patch("rag_challenge.ingestion.pipeline.get_settings", return_value=settings):
        pipeline = IngestionPipeline(parser=parser, chunker=chunker, sac=sac, embedder=embedder, store=store)
        upserted = await pipeline._upsert_pages_for_doc(doc, "summary")

    assert upserted == 1
    stored_page = store.upsert_pages.await_args.args[0][0]
    assert stored_page.top_lines[:2] == ["Enactment Notice", "DIFC Employment Law"]
    assert stored_page.page_template_family == "issued_by_authority"
    assert stored_page.document_template_family == "enactment_notice"
    assert stored_page.has_issued_by_pattern is True
    assert stored_page.officialness_score > 0.8
