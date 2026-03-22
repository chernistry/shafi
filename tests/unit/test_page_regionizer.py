from __future__ import annotations

from rag_challenge.ingestion.page_regionizer import (
    HardPageDetector,
    PageRegionizer,
    PageTextBlock,
    RegionEnricher,
    RenderedPage,
)
from rag_challenge.models.legal_objects import (
    BoundingBox,
    CorpusRegistry,
    LawObject,
    LegalDocType,
    RegionType,
)


def _rendered_page() -> RenderedPage:
    return RenderedPage(
        page_id="law-a_1",
        doc_id="law-a",
        width=800,
        height=1200,
        text_blocks=[
            PageTextBlock(text="EMPLOYMENT LAW 2020", bbox=BoundingBox(x=40, y=40, w=320, h=50)),
            PageTextBlock(text="Claimant: Alice Smith", bbox=BoundingBox(x=40, y=120, w=260, h=40)),
            PageTextBlock(text="Schedule 1 | Filing | Annual return", bbox=BoundingBox(x=40, y=360, w=500, h=120)),
            PageTextBlock(
                text="Issued by DIFC Authority\nSigned by Registrar\n1 January 2020",
                bbox=BoundingBox(x=420, y=900, w=260, h=140),
            ),
        ],
    )


def _notice_rendered_page() -> RenderedPage:
    return RenderedPage(
        page_id="law-a_1",
        doc_id="law-a",
        width=800,
        height=1200,
        text_blocks=[
            PageTextBlock(text="ENACTMENT NOTICE", bbox=BoundingBox(x=40, y=30, w=300, h=40)),
            PageTextBlock(text="DIFC Employment Law\nLaw No. 2 of 2019", bbox=BoundingBox(x=40, y=90, w=340, h=70)),
            PageTextBlock(text="Issued by DIFC Authority\nDate of Issue: 1 January 2020", bbox=BoundingBox(x=40, y=920, w=360, h=90)),
        ],
    )


def test_hard_page_detector_flags_caption_notice_and_schedule_pages() -> None:
    detector = HardPageDetector()

    assert detector.is_hard_page("Schedule 1 | Filing | Annual return", {"ocr_artifact": False}) is True
    assert detector.is_hard_page("Tiny text", {}) is True
    assert detector.is_hard_page("Enactment Notice\nDIFC Employment Law\nLaw No. 2 of 2019", {}) is True
    assert detector.is_hard_page("CFI 010/2024\nACME LTD v BETA LLC\nClaimant: Alice Smith", {}) is True


def test_page_regionizer_extracts_caption_table_and_signature_regions() -> None:
    regions = PageRegionizer().extract_regions(_rendered_page(), "law-a_1")
    region_types = {region.region_type for region in regions}

    assert RegionType.HEADER in region_types or RegionType.TITLE in region_types
    assert RegionType.CAPTION in region_types
    assert RegionType.SCHEDULE in region_types
    assert RegionType.SIGNATURE in region_types


def test_page_regionizer_extracts_notice_title_and_judge_signature_regions() -> None:
    regions = PageRegionizer().extract_regions(_notice_rendered_page(), "law-a_1")
    region_types = {region.region_type for region in regions}

    assert RegionType.TITLE in region_types
    assert RegionType.SIGNATURE in region_types


def test_extract_table_structure_and_signature_block_are_typed() -> None:
    regionizer = PageRegionizer()
    table = regionizer.extract_table_structure("Schedule 1 | Filing | Annual return\nRow A | Due | 30 days")
    signature = regionizer.extract_signature_block("Issued by DIFC Authority\nSigned by Registrar\n1 January 2020")
    judge_signature = regionizer.extract_signature_block("Judge: Justice Amina Hassan\n1 January 2020")

    assert table.headers == ["Schedule 1", "Filing", "Annual return"]
    assert table.rows == [["Row A", "Due", "30 days"]]
    assert signature.authority == "DIFC Authority"
    assert signature.signer == "Registrar"
    assert signature.date == "1 January 2020"
    assert judge_signature.signer == "Justice Amina Hassan"
    assert judge_signature.date == "1 January 2020"


def test_region_enricher_attaches_regions_to_registry_pages() -> None:
    registry = CorpusRegistry(
        laws={
            "law-a": LawObject(
                object_id="law:law-a",
                doc_id="law-a",
                title="Employment Law 2020",
                legal_doc_type=LegalDocType.LAW,
                page_ids=["law-a_1"],
                page_texts={"law-a_1": "Schedule 1 | Filing | Annual return"},
            )
        }
    )
    page = _rendered_page()
    regions = PageRegionizer().extract_regions(page, page.page_id)
    enriched = RegionEnricher().enrich_corpus_registry(registry, {page.page_id: regions})

    assert enriched.laws["law-a"].visual_regions["law-a_1"]
    assert enriched.laws["law-a"].visual_regions["law-a_1"][0].page_id == "law-a_1"
