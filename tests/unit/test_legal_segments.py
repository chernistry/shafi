from shafi.ingestion.legal_segments import SegmentCompiler
from shafi.models import Chunk, DocType, DocumentSection, ParsedDocument, SegmentType
from shafi.models.legal_objects import CorpusRegistry, LawObject, LegalDocType


def _build_registry() -> CorpusRegistry:
    return CorpusRegistry(
        source_doc_count=1,
        laws={
            "law_doc": LawObject(
                object_id="law:law_doc",
                doc_id="law_doc",
                title="Operating Law 2018",
                legal_doc_type=LegalDocType.LAW,
                short_title="Operating Law",
                law_number="3",
                year="2018",
                issuing_authority="DIFC Authority",
                page_ids=["law_doc_1", "law_doc_2", "law_doc_3"],
            )
        },
    )


def _build_law_doc() -> ParsedDocument:
    return ParsedDocument(
        doc_id="law_doc",
        title="Operating Law 2018",
        doc_type=DocType.STATUTE,
        sections=[
            DocumentSection(
                heading="page 1",
                section_path="page:1",
                text=(
                    "Part I - General\n"
                    "Chapter 1 - Introductory\n"
                    "Article 1 - Registration\n"
                    "An applicant must file the prescribed form.\n"
                    "This Article continues onto the next page."
                ),
            ),
            DocumentSection(
                heading="page 2",
                section_path="page:2",
                text=(
                    "The obligations in Article 1 continue here.\n"
                    "Definitions\n"
                    '"Approved Person" means a person approved by the Authority.\n'
                    "Issued by: Dubai International Financial Centre Authority\n"
                    "Article 2 - Registration\n"
                    "A firm shall register with the Registrar."
                ),
            ),
            DocumentSection(
                heading="page 3",
                section_path="page:3",
                text=("Schedule 1 - Fees\nTable of annual fees.\nOperative text follows."),
            ),
        ],
    )


def test_compile_segments_detects_cross_page_articles_and_schedule() -> None:
    compiler = SegmentCompiler()
    segments = compiler.compile_segments(_build_law_doc(), _build_registry())

    article_segments = [segment for segment in segments if segment.segment_type is SegmentType.ARTICLE]
    schedule_segments = [segment for segment in segments if segment.segment_type is SegmentType.SCHEDULE]
    definition_segments = [segment for segment in segments if segment.segment_type is SegmentType.DEFINITION]
    issued_by_segments = [segment for segment in segments if segment.segment_type is SegmentType.ISSUED_BY]

    assert article_segments
    assert schedule_segments
    assert definition_segments
    assert issued_by_segments
    assert article_segments[0].page_ids == ["law_doc_1", "law_doc_2"]
    assert article_segments[0].canonical_doc_id == "law:law_doc"
    assert "Operating Law 2018" in article_segments[0].legal_path
    assert compiler.project_to_pages(article_segments[0]) == ["law_doc_1", "law_doc_2"]


def test_annotate_chunks_adds_segment_ids_for_matching_pages() -> None:
    compiler = SegmentCompiler()
    segments = compiler.compile_segments(_build_law_doc(), _build_registry())
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="law_doc",
            doc_title="Operating Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            chunk_text="Article 1 - Definitions\nApproved Person means a person approved by the Authority.",
            chunk_text_for_embedding="Article 1 - Definitions\nApproved Person means a person approved by the Authority.",
        ),
        Chunk(
            chunk_id="c2",
            doc_id="law_doc",
            doc_title="Operating Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            chunk_text="Schedule 1 - Fees\nTable of annual fees.",
            chunk_text_for_embedding="Schedule 1 - Fees\nTable of annual fees.",
        ),
    ]

    annotated = compiler.annotate_chunks(chunks, segments)

    assert annotated[0].segment_id
    assert annotated[1].segment_id
    assert annotated[0].segment_id != annotated[1].segment_id
