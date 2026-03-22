from __future__ import annotations

from pathlib import Path

from rag_challenge.ingestion.corpus_compiler import CorpusCompiler
from rag_challenge.ingestion.parser import DocumentParser
from rag_challenge.models import DocType, DocumentSection, ParsedDocument
from rag_challenge.models.legal_objects import (
    CaseObject,
    CaseParty,
    CorpusRegistry,
    LawObject,
    LegalDocType,
)

FIXTURE_DIR = Path("/Users/sasha/IdeaProjects/personal_projects/rag_challenge/tests/fixtures/docs")


def _parse_fixture(name: str) -> ParsedDocument:
    parser = DocumentParser()
    return parser.parse_file(FIXTURE_DIR / name)


def test_resolve_document_type_prefers_law_for_statute_fixture() -> None:
    doc = _parse_fixture("limitation_act.txt")

    compiler = CorpusCompiler()

    assert compiler.resolve_document_type(doc) == LegalDocType.LAW


def test_extract_law_metadata_reads_short_title_and_article_tree() -> None:
    doc = _parse_fixture("limitation_act.txt")

    compiler = CorpusCompiler()
    compiled = compiler.extract_law_metadata(doc)

    assert isinstance(compiled, LawObject)
    assert compiled.short_title == "This Act may be cited as the Limitation Act 2020"
    assert compiled.year == "2020"
    assert [node.label for node in compiled.article_tree[:2]] == ["Section 1", "Section 2"]


def test_extract_law_metadata_falls_back_to_ruler_style_authority() -> None:
    doc = ParsedDocument(
        doc_id="data-protection-law",
        title="DIFC Data Protection Law No. 5 of 2020",
        doc_type=DocType.STATUTE,
        source_path="data_protection_law.pdf",
        full_text=(
            "We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact this Law No. 5 of 2020.\n"
            "This law comes into force on 1 January 2021."
        ),
        sections=[
            DocumentSection(
                heading="Page 1",
                section_path="page:1",
                text="We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact this Law No. 5 of 2020.",
                level=0,
            ),
            DocumentSection(
                heading="Page 2",
                section_path="page:2",
                text="This law comes into force on 1 January 2021.",
                level=0,
            ),
        ],
    )

    compiler = CorpusCompiler()
    compiled = compiler.extract_law_metadata(doc)

    assert compiled.issuing_authority == "Mohammed bin Rashid Al Maktoum, Ruler of Dubai"
    assert compiled.field_page_ids["issued_by"] == ["data-protection-law_1"]
    assert compiled.field_page_ids["authority"] == ["data-protection-law_1"]


def test_extract_law_metadata_reads_in_force_date_from_regulation_text() -> None:
    doc = ParsedDocument(
        doc_id="financial-collateral-regulations",
        title="Financial Collateral Regulations",
        doc_type=DocType.REGULATION,
        source_path="financial_collateral_regulations.pdf",
        full_text=(
            "FINANCIAL COLLATERAL REGULATIONS\n"
            "In force on 1 November 2019.\n"
            "These Regulations apply in the DIFC."
        ),
        sections=[
            DocumentSection(
                heading="Page 1",
                section_path="page:1",
                text="FINANCIAL COLLATERAL REGULATIONS\nIn force on 1 November 2019.",
                level=0,
            )
        ],
    )

    compiler = CorpusCompiler()
    compiled = compiler.extract_law_metadata(doc, legal_doc_type=LegalDocType.REGULATION)

    assert compiled.commencement_date == "1 November 2019"
    assert compiled.field_page_ids["commencement_date"] == ["financial-collateral-regulations_1"]
    assert compiled.field_page_ids["date"] == ["financial-collateral-regulations_1"]


def test_extract_case_metadata_reads_caption_and_outcome() -> None:
    doc = _parse_fixture("smith_v_jones.txt")

    compiler = CorpusCompiler()
    compiled = compiler.extract_case_metadata(doc)

    assert isinstance(compiled, CaseObject)
    assert compiled.case_number == "[2021] UKSC 45"
    assert [party.name for party in compiled.parties] == ["Smith", "Jones"]
    assert "termination clause" in compiled.outcome_summary.casefold()


def test_extract_case_metadata_ignores_role_noise_from_order_body() -> None:
    doc = ParsedDocument(
        doc_id="cfi_010_2024",
        title="CFI 010/2024 Fursa Consulting v Bay Gate Investment LLC",
        doc_type=DocType.CASE_LAW,
        source_path="cfi_010_2024.pdf",
        full_text=(
            "BETWEEN\n"
            "FURSA CONSULTING\n"
            "Claimant\n"
            "and\n"
            "BAY GATE INVESTMENT LLC\n"
            "Defendant\n"
            "AND UPON the Claimant's Application dated 19 November 2025\n"
            "AND UPON all documents filed in this matter\n"
            "IT IS HEREBY ORDERED THAT:\n"
            "1. The Application is dismissed.\n"
        ),
        sections=[
            DocumentSection(
                heading="Page 1",
                section_path="page:1",
                text=(
                    "BETWEEN\n"
                    "FURSA CONSULTING\n"
                    "Claimant\n"
                    "and\n"
                    "BAY GATE INVESTMENT LLC\n"
                    "Defendant\n"
                    "AND UPON the Claimant's Application dated 19 November 2025\n"
                ),
                level=0,
            )
        ],
    )

    compiler = CorpusCompiler()
    compiled = compiler.extract_case_metadata(doc)

    assert isinstance(compiled, CaseObject)
    assert compiled.parties == [
        CaseParty(name="Fursa Consulting", role="claimant"),
        CaseParty(name="Bay Gate Investment LLC", role="respondent"),
    ]


def test_extract_case_metadata_normalizes_appellate_caption_roles() -> None:
    doc = ParsedDocument(
        doc_id="cfi_011_2024",
        title="CFI 011/2024 Alpha Holdings Ltd (Appellant) v Beta Holdings Ltd (Respondent)",
        doc_type=DocType.CASE_LAW,
        source_path="cfi_011_2024.pdf",
        full_text=(
            "BETWEEN\n"
            "ALPHA HOLDINGS LTD\n"
            "Appellant\n"
            "and\n"
            "BETA HOLDINGS LTD\n"
            "Respondent\n"
            "The appeal is allowed.\n"
        ),
        sections=[
            DocumentSection(
                heading="Page 1",
                section_path="page:1",
                text=(
                    "CFI 011/2024 Alpha Holdings Ltd (Appellant) v Beta Holdings Ltd (Respondent)\n"
                    "ALPHA HOLDINGS LTD\n"
                    "Appellant\n"
                ),
                level=0,
            )
        ],
    )

    compiler = CorpusCompiler()
    compiled = compiler.extract_case_metadata(doc)

    assert isinstance(compiled, CaseObject)
    assert compiled.parties == [
        CaseParty(name="Alpha Holdings Ltd", role="appellant"),
        CaseParty(name="Beta Holdings Ltd", role="respondent"),
    ]
    assert compiled.field_page_ids["appellant"] == ["cfi_011_2024_1"]
    assert compiled.field_page_ids["respondent"] == ["cfi_011_2024_1"]


def test_extract_case_metadata_filters_registrar_and_party_noise_from_judges_and_normalizes_court_title() -> None:
    doc = ParsedDocument(
        doc_id="cfi_010_2024",
        title="CFI 010/2024 Alpha Ltd v Registrar",
        doc_type=DocType.CASE_LAW,
        source_path="cfi_010_2024.pdf",
        full_text=(
            "BETWEEN\n"
            "ALPHA LTD\n"
            "Claimant\n"
            "and\n"
            "BETA LTD\n"
            "Respondent\n"
            "Before: Justice Jane Smith\n"
            "Court of Appeal of the DIFC Courts\n"
        ),
        sections=[
            DocumentSection(
                heading="Page 1",
                section_path="page:1",
                text=(
                    "BETWEEN\n"
                    "ALPHA LTD\n"
                    "Claimant\n"
                    "and\n"
                    "BETA LTD\n"
                    "Respondent\n"
                ),
                level=0,
            ),
            DocumentSection(
                heading="Page 2",
                section_path="page:2",
                text="Before: Justice Jane Smith\nCourt of Appeal of the DIFC Courts\n",
                level=0,
            ),
        ],
    )

    compiler = CorpusCompiler()
    compiled = compiler.extract_case_metadata(doc)

    assert compiled.judges == ["Justice Jane Smith"]
    assert compiled.court == "Court of Appeal"
    assert compiled.field_page_ids["judge"] == ["cfi_010_2024_2"]
    assert compiled.field_page_ids["court"] == ["cfi_010_2024_2"]
    assert "Registrar" not in compiled.judges


def test_is_plausible_party_value_rejects_role_fragment_and_placeholder_noise() -> None:
    assert CorpusCompiler._is_plausible_party_value("pondent") is False
    assert CorpusCompiler._is_plausible_party_value("/") is False
    assert CorpusCompiler._is_plausible_party_value("Creditor") is False
    assert CorpusCompiler._is_plausible_party_value("Debtor") is False
    assert CorpusCompiler._is_plausible_party_value("Natixis") is True


def test_compile_document_populates_source_text_page_texts_and_field_pages() -> None:
    doc = ParsedDocument(
        doc_id="companies_law",
        title="DIFC Companies Law No. 3 of 2004",
        doc_type=DocType.STATUTE,
        source_path="companies_law.pdf",
        full_text=(
            "DIFC Companies Law No. 3 of 2004\nIssued by DIFC Authority.\n"
            "This law comes into force on 1 January 2005."
        ),
        sections=[
            DocumentSection(
                heading="Page 1",
                section_path="page:1",
                text="DIFC Companies Law No. 3 of 2004\nIssued by DIFC Authority.",
                level=0,
            ),
            DocumentSection(
                heading="Page 2",
                section_path="page:2",
                text="This law comes into force on 1 January 2005.",
                level=0,
            ),
        ],
    )

    compiler = CorpusCompiler()
    compiled = compiler.extract_law_metadata(doc)

    assert compiled.source_text.startswith("DIFC Companies Law No. 3 of 2004")
    assert compiled.page_texts == {
        "companies_law_1": "DIFC Companies Law No. 3 of 2004\nIssued by DIFC Authority.",
        "companies_law_2": "This law comes into force on 1 January 2005.",
    }
    assert compiled.field_page_ids["title"] == ["companies_law_1"]
    assert compiled.field_page_ids["issued_by"] == ["companies_law_1"]
    assert compiled.field_page_ids["commencement_date"] == ["companies_law_2"]


def test_compile_documents_builds_cross_reference_links() -> None:
    law_doc = ParsedDocument(
        doc_id="employment-rights-act",
        title="Employment Rights Act",
        doc_type=DocType.STATUTE,
        source_path="employment_rights_act.txt",
        full_text="Employment Rights Act",
        sections=[],
    )
    case_doc = _parse_fixture("smith_v_jones.txt")

    compiler = CorpusCompiler()
    registry = compiler.compile_documents([law_doc, case_doc])

    assert any(link.target_doc_id == "employment-rights-act" for link in registry.links)
    assert all(link.source_doc_id != link.target_doc_id for link in registry.links)


def test_merge_registries_accumulates_all_object_buckets() -> None:
    compiler = CorpusCompiler()
    law_registry = compiler.compile_documents([_parse_fixture("limitation_act.txt")])
    case_registry = compiler.compile_documents([_parse_fixture("smith_v_jones.txt")])

    merged = compiler.merge_registries([law_registry, case_registry])

    assert isinstance(merged, CorpusRegistry)
    assert len(merged.laws) == 1
    assert len(merged.cases) == 1


def test_compile_documents_is_idempotent_for_fixture_pack() -> None:
    compiler = CorpusCompiler()
    docs = [_parse_fixture("limitation_act.txt"), _parse_fixture("smith_v_jones.txt"), _parse_fixture("sample_contract.txt")]

    left = compiler.compile_documents(docs).model_dump_json()
    right = compiler.compile_documents(docs).model_dump_json()

    assert left == right
