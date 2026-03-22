from __future__ import annotations

from rag_challenge.core.field_lookup import FieldLookupTable, FieldType
from rag_challenge.models.legal_objects import (
    CaseObject,
    CaseParty,
    CorpusRegistry,
    LawObject,
    LegalDocType,
    OrderObject,
)


def _registry() -> CorpusRegistry:
    return CorpusRegistry(
        source_doc_count=2,
        laws={
            "companies_law": LawObject(
                object_id="law:companies_law",
                doc_id="companies_law",
                title="DIFC Companies Law No. 3 of 2004",
                source_path="companies_law.pdf",
                page_ids=["companies_law_1", "companies_law_2"],
                source_text="Issued by DIFC Authority. Commencement date 1 January 2005.",
                page_texts={
                    "companies_law_1": "DIFC Companies Law No. 3 of 2004. Issued by DIFC Authority.",
                    "companies_law_2": "This law comes into force on 1 January 2005.",
                },
                field_page_ids={
                    "title": ["companies_law_1"],
                    "law_number": ["companies_law_1"],
                    "issued_by": ["companies_law_1"],
                    "authority": ["companies_law_1"],
                    "commencement_date": ["companies_law_2"],
                    "date": ["companies_law_2"],
                },
                legal_doc_type=LegalDocType.LAW,
                short_title="Companies Law",
                law_number="3",
                year="2004",
                issuing_authority="DIFC Authority",
                commencement_date="1 January 2005",
            )
        },
        orders={
            "data_protection_notice": OrderObject(
                object_id="enactment_notice:data_protection_notice",
                doc_id="data_protection_notice",
                title="_______________________________________________",
                source_path="notice.pdf",
                page_ids=["data_protection_notice_1"],
                source_text=(
                    "ENACTMENT NOTICE\n"
                    "We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact\n"
                    "on this 01 day of March 2024\n"
                    "the\n"
                    "DATA PROTECTION LAW\n"
                    "DIFC LAW NO. 5 OF 2020\n"
                ),
                page_texts={"data_protection_notice_1": "DATA PROTECTION LAW\nDIFC LAW NO. 5 OF 2020"},
                field_page_ids={
                    "title": ["data_protection_notice_1"],
                    "date": ["data_protection_notice_1"],
                    "authority": ["data_protection_notice_1"],
                    "issued_by": ["data_protection_notice_1"],
                },
                legal_doc_type=LegalDocType.ENACTMENT_NOTICE,
                issued_by="the",
                effective_date="",
            )
        },
        cases={
            "cfi_001_2024": CaseObject(
                object_id="case:cfi_001_2024",
                doc_id="cfi_001_2024",
                title="Alpha Ltd v Beta Ltd",
                source_path="cfi_001_2024.pdf",
                page_ids=["cfi_001_2024_1", "cfi_001_2024_2"],
                source_text="Claimant: Alpha Ltd. Respondent: Beta Ltd. Judge Jane Smith.",
                page_texts={
                    "cfi_001_2024_1": "Alpha Ltd v Beta Ltd. Claimant: Alpha Ltd. Respondent: Beta Ltd.",
                    "cfi_001_2024_2": "Before Justice Jane Smith. The claim was dismissed.",
                },
                field_page_ids={
                    "title": ["cfi_001_2024_1"],
                    "case_number": ["cfi_001_2024_1"],
                    "claimant": ["cfi_001_2024_1"],
                    "respondent": ["cfi_001_2024_1"],
                    "appellant": ["cfi_001_2024_1"],
                    "appellee": ["cfi_001_2024_1"],
                    "party": ["cfi_001_2024_1"],
                    "judge": ["cfi_001_2024_2"],
                    "outcome": ["cfi_001_2024_2"],
                },
                legal_doc_type=LegalDocType.CASE,
                case_number="CFI 001/2024",
                judges=["Justice Jane Smith"],
                parties=[
                    CaseParty(name="Alpha Ltd", role="appellant"),
                    CaseParty(name="Beta Ltd", role="respondent"),
                ],
                outcome_summary="The claim was dismissed.",
            )
        },
    )


def test_field_lookup_builds_rows_for_laws_and_cases() -> None:
    lookup = FieldLookupTable.build_from_registry(_registry())

    issued_by = lookup.lookup("law:companies_law", FieldType.ISSUED_BY)
    judge = lookup.lookup("case:cfi_001_2024", FieldType.JUDGE)
    appellant = lookup.lookup("case:cfi_001_2024", FieldType.APPELLANT)

    assert issued_by is not None
    assert issued_by.value == "DIFC Authority"
    assert issued_by.source_page_ids == ["companies_law_1"]
    assert judge is not None
    assert judge.value == "Justice Jane Smith"
    assert judge.source_page_ids == ["cfi_001_2024_2"]
    assert appellant is not None
    assert appellant.value == "Alpha Ltd"
    assert appellant.source_page_ids == ["cfi_001_2024_1"]


def test_field_lookup_reverse_search_and_round_trip(tmp_path) -> None:
    lookup = FieldLookupTable.build_from_registry(_registry())
    path = lookup.export(tmp_path / "field_lookup.json")

    restored = FieldLookupTable.load(path)

    assert restored.search_by_field(FieldType.CASE_NUMBER, "CFI 001/2024") == ["case:cfi_001_2024"]
    assert restored.get_all_fields("law:companies_law")[FieldType.TITLE] == "DIFC Companies Law No. 3 of 2004"
    assert restored.get_all_fields("case:cfi_001_2024")[FieldType.APPELLANT] == "Alpha Ltd"


def test_field_lookup_derives_law_fields_from_enactment_notice() -> None:
    lookup = FieldLookupTable.build_from_registry(_registry())

    title = lookup.lookup("order:data_protection_notice", FieldType.TITLE)
    law_number = lookup.lookup("order:data_protection_notice", FieldType.LAW_NUMBER)
    authority = lookup.lookup("order:data_protection_notice", FieldType.AUTHORITY)
    date = lookup.lookup("order:data_protection_notice", FieldType.DATE)

    assert title is not None
    assert title.value == "DATA PROTECTION LAW"
    assert law_number is not None
    assert law_number.value == "5"
    assert authority is not None
    assert authority.value == "Mohammed bin Rashid Al Maktoum, Ruler of Dubai"
    assert date is not None
    assert date.value == "1 March 2024"


def test_field_lookup_backfills_law_authority_from_source_text() -> None:
    registry = CorpusRegistry(
        source_doc_count=1,
        laws={
            "data_protection_law": LawObject(
                object_id="law:data_protection_law",
                doc_id="data_protection_law",
                title="DIFC Data Protection Law No. 5 of 2020",
                source_path="data_protection_law.pdf",
                page_ids=["data_protection_law_1", "data_protection_law_2"],
                source_text=(
                    "We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact this Law No. 5 of 2020.\n"
                    "In force on 1 January 2021."
                ),
                page_texts={
                    "data_protection_law_1": "We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact this Law No. 5 of 2020.",
                    "data_protection_law_2": "In force on 1 January 2021.",
                },
                field_page_ids={
                    "title": ["data_protection_law_1"],
                    "law_number": ["data_protection_law_1"],
                    "issued_by": ["data_protection_law_1"],
                    "authority": ["data_protection_law_1"],
                    "commencement_date": ["data_protection_law_2"],
                    "date": ["data_protection_law_2"],
                },
                legal_doc_type=LegalDocType.LAW,
                short_title="Data Protection Law",
                law_number="5",
                year="2020",
                issuing_authority="",
                commencement_date="",
            )
        },
    )

    lookup = FieldLookupTable.build_from_registry(registry)

    issued_by = lookup.lookup("law:data_protection_law", FieldType.ISSUED_BY)
    authority = lookup.lookup("law:data_protection_law", FieldType.AUTHORITY)
    date = lookup.lookup("law:data_protection_law", FieldType.DATE)
    commencement_date = lookup.lookup("law:data_protection_law", FieldType.COMMENCEMENT_DATE)

    assert issued_by is not None
    assert issued_by.value == "Mohammed bin Rashid Al Maktoum, Ruler of Dubai"
    assert authority is not None
    assert authority.value == "Mohammed bin Rashid Al Maktoum, Ruler of Dubai"
    assert date is not None
    assert date.value == "1 January 2021"
    assert commencement_date is not None
    assert commencement_date.value == "1 January 2021"
