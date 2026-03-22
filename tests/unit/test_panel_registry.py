from __future__ import annotations

from rag_challenge.core.panel_registry import PanelRegistry
from rag_challenge.models.legal_objects import (
    CaseObject,
    CaseParty,
    CorpusRegistry,
    LawObject,
    LegalDocType,
    LegalEntity,
    LegalEntityType,
)


def _registry() -> CorpusRegistry:
    return CorpusRegistry(
        source_doc_count=3,
        laws={
            "companies_law": LawObject(
                object_id="law:companies_law",
                doc_id="companies_law",
                title="DIFC Companies Law No. 3 of 2004",
                short_title="Companies Law",
                law_number="3",
                year="2004",
                issuing_authority="DIFC Authority",
                commencement_date="1 January 2005",
                legal_doc_type=LegalDocType.LAW,
                page_ids=["companies_law_1", "companies_law_2"],
                field_page_ids={
                    "title": ["companies_law_1"],
                    "authority": ["companies_law_1"],
                    "law_number": ["companies_law_1"],
                    "date": ["companies_law_2"],
                },
            )
        },
        cases={
            "cfi_001_2024": CaseObject(
                object_id="case:cfi_001_2024",
                doc_id="cfi_001_2024",
                title="Alpha Ltd v Beta Ltd",
                case_number="CFI 001/2024",
                judges=["Justice Jane Smith"],
                parties=[
                    CaseParty(name="Alpha Ltd", role="claimant"),
                    CaseParty(name="Beta Ltd", role="respondent"),
                ],
                date="1 January 2024",
                legal_doc_type=LegalDocType.CASE,
                page_ids=["cfi_001_2024_1", "cfi_001_2024_2"],
                field_page_ids={
                    "case_number": ["cfi_001_2024_1"],
                    "party": ["cfi_001_2024_1"],
                    "judge": ["cfi_001_2024_2"],
                    "date": ["cfi_001_2024_2"],
                },
            ),
            "cfi_002_2024": CaseObject(
                object_id="case:cfi_002_2024",
                doc_id="cfi_002_2024",
                title="Gamma Ltd v Beta Ltd",
                case_number="CFI 002/2024",
                judges=["Justice Jane Smith"],
                parties=[
                    CaseParty(name="Gamma Ltd", role="claimant"),
                    CaseParty(name="Beta Ltd", role="respondent"),
                ],
                date="4 January 2024",
                legal_doc_type=LegalDocType.CASE,
                page_ids=["cfi_002_2024_1", "cfi_002_2024_2"],
                field_page_ids={
                    "case_number": ["cfi_002_2024_1"],
                    "party": ["cfi_002_2024_1"],
                    "judge": ["cfi_002_2024_2"],
                    "date": ["cfi_002_2024_2"],
                },
            ),
        },
        entities={
            "party:beta-ltd": LegalEntity(
                entity_id="party:beta-ltd",
                name="Beta Ltd",
                canonical_name="Beta Ltd",
                entity_type=LegalEntityType.PARTY,
                source_doc_ids=["cfi_001_2024", "cfi_002_2024"],
            ),
            "judge:jane-smith": LegalEntity(
                entity_id="judge:jane-smith",
                name="Justice Jane Smith",
                canonical_name="Justice Jane Smith",
                entity_type=LegalEntityType.JUDGE,
                source_doc_ids=["cfi_001_2024", "cfi_002_2024"],
            ),
        },
    )


def test_panel_registry_resolves_selector_documents() -> None:
    registry = PanelRegistry.build_from_corpus(_registry())

    assert registry.resolve_document_ids(["case_number:cfi0012024"]) == ["cfi_001_2024"]
    assert registry.resolve_document_ids(["law_title:companies-law"], source_doc_ids=["companies_law"]) == ["companies_law"]


def test_panel_registry_intersects_entities_and_attributes() -> None:
    registry = PanelRegistry.build_from_corpus(_registry())

    assert registry.intersect_entities(["cfi_001_2024", "cfi_002_2024"]) == [
        "judge:jane-smith",
        "party:beta-ltd",
    ]
    assert registry.intersect_attributes(["cfi_001_2024", "cfi_002_2024"], "judge") == ["Justice Jane Smith"]
    assert registry.intersect_attributes(["cfi_001_2024", "cfi_002_2024"], "party") == ["Beta Ltd"]


def test_panel_registry_normalizes_case_court_and_filters_judge_noise() -> None:
    noisy_registry = CorpusRegistry(
        source_doc_count=1,
        cases={
            "cfi_010_2024": CaseObject(
                object_id="case:cfi_010_2024",
                doc_id="cfi_010_2024",
                title="Alpha Ltd v Registrar",
                case_number="CFI 010/2024",
                judges=["Registrar", "H.E. Justice Jane Smith", "Justice Jane Smith"],
                parties=[
                    CaseParty(name="Alpha Ltd", role="claimant"),
                    CaseParty(name="Beta Ltd", role="respondent"),
                ],
                court="The Court of Appeal of the DIFC Courts",
                legal_doc_type=LegalDocType.CASE,
            )
        },
    )

    registry = PanelRegistry.build_from_corpus(noisy_registry)

    assert registry.intersect_attributes(["cfi_010_2024"], "judge") == ["Justice Jane Smith"]
    assert registry.intersect_attributes(["cfi_010_2024"], "court") == ["Court of Appeal"]
    assert registry.get_documents_by_field("court", "Court of Appeal") == ["cfi_010_2024"]
    assert registry.get_documents_by_field("judge", "Registrar") == []
