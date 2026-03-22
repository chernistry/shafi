from __future__ import annotations

from typing import TYPE_CHECKING

from rag_challenge.ingestion.canonical_entities import build_law_aliases
from rag_challenge.ingestion.corpus_compiler import CorpusCompiler
from rag_challenge.ingestion.manual_domain_overrides import (
    ManualDomainOverride,
    apply_manual_domain_override,
)
from rag_challenge.models import DocType, DocumentSection, ParsedDocument
from rag_challenge.models.legal_objects import CaseObject, CaseParty, CorpusRegistry, LawObject, LegalDocType

if TYPE_CHECKING:
    import pytest


def test_apply_manual_law_override_updates_fields_and_aliases() -> None:
    compiled = LawObject(
        object_id="law:companies",
        doc_id="companies",
        title="COMPANIES LAW",
        legal_doc_type=LegalDocType.LAW,
        short_title="COMPANIES LAW",
        aliases=["COMPANIES LAW"],
        law_number="",
        year="",
        issuing_authority="",
        commencement_date="",
    )
    override = ManualDomainOverride(
        title="Companies Law DIFC Law No. 5 of 2018",
        short_title="Companies Law",
        aliases=("DIFC Law No. 5 of 2018",),
        law_number="5",
        year="2018",
        issued_by="Board of Directors of the DIFCA",
        commencement_date="12 November 2018",
    )

    patched = apply_manual_domain_override(compiled, override)

    assert isinstance(patched, LawObject)
    assert patched.title == "Companies Law DIFC Law No. 5 of 2018"
    assert patched.short_title == "Companies Law"
    assert patched.law_number == "5"
    assert patched.year == "2018"
    assert patched.issuing_authority == "Board of Directors of the DIFCA"
    assert patched.commencement_date == "12 November 2018"
    assert patched.aliases == ["COMPANIES LAW", "DIFC Law No. 5 of 2018"]


def test_apply_manual_case_override_replaces_target_roles_only() -> None:
    compiled = CaseObject(
        object_id="case:lxt",
        doc_id="lxt",
        title="CA 005/2025 LXT Real Estate Broker L.L.C v SIR Real Estate LLC",
        legal_doc_type=LegalDocType.CASE,
        case_number="CA 005/2025",
        judges=["Registrar Date"],
        parties=[
            CaseParty(name="LXT Real Estate Broker L.L.C", role="claimant"),
            CaseParty(name="pondent", role="respondent"),
            CaseParty(name="SIR Real Estate LLC", role="appellant"),
        ],
    )
    override = ManualDomainOverride(
        claimant=("LXT Real Estate Broker L.L.C",),
        respondent=("SIR Real Estate LLC",),
        judges=("Chief Justice Wayne Martin", "Justice Sir Peter Gross", "Justice Rene Le Miere"),
    )

    patched = apply_manual_domain_override(compiled, override)

    assert isinstance(patched, CaseObject)
    assert patched.judges == [
        "Chief Justice Wayne Martin",
        "Justice Sir Peter Gross",
        "Justice Rene Le Miere",
    ]
    assert patched.parties == [
        CaseParty(name="LXT Real Estate Broker L.L.C", role="claimant"),
        CaseParty(name="SIR Real Estate LLC", role="respondent"),
        CaseParty(name="SIR Real Estate LLC", role="appellant"),
    ]


def test_corpus_compiler_applies_manual_override_and_refreshes_field_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = ParsedDocument(
        doc_id="companies",
        title="COMPANIES LAW",
        doc_type=DocType.STATUTE,
        source_path="companies.pdf",
        full_text=(
            "Companies Law DIFC Law No. 5 of 2018\n"
            "Issued by Board of Directors of the DIFCA\n"
            "Date of Issue: 12 November 2018\n"
        ),
        sections=[
            DocumentSection(
                heading="Page 1",
                section_path="page:1",
                text=(
                    "Companies Law DIFC Law No. 5 of 2018\n"
                    "Issued by Board of Directors of the DIFCA\n"
                    "Date of Issue: 12 November 2018\n"
                ),
                level=0,
            )
        ],
    )
    override = ManualDomainOverride(
        title="Companies Law DIFC Law No. 5 of 2018",
        short_title="Companies Law",
        aliases=("DIFC Law No. 5 of 2018",),
        law_number="5",
        year="2018",
        issued_by="Board of Directors of the DIFCA",
        commencement_date="12 November 2018",
    )
    monkeypatch.setattr(
        "rag_challenge.ingestion.corpus_compiler.get_manual_domain_override",
        lambda doc_id: override if doc_id == "companies" else None,
    )

    compiler = CorpusCompiler()
    _legal_doc_type, compiled = compiler.compile_document(doc)

    assert isinstance(compiled, LawObject)
    assert compiled.field_page_ids["title"] == ["companies_1"]
    assert compiled.field_page_ids["law_number"] == ["companies_1"]
    assert compiled.field_page_ids["issued_by"] == ["companies_1"]
    assert compiled.field_page_ids["commencement_date"] == ["companies_1"]


def test_manual_aliases_propagate_into_law_alias_clusters() -> None:
    registry = CorpusRegistry(
        source_doc_count=1,
        laws={
            "companies": LawObject(
                object_id="law:companies",
                doc_id="companies",
                title="Companies Law DIFC Law No. 5 of 2018",
                short_title="Companies Law",
                aliases=["COMPANIES LAW", "DIFC Law No. 5 of 2018"],
                law_number="5",
                year="2018",
                legal_doc_type=LegalDocType.LAW,
            )
        },
    )

    clusters = build_law_aliases(registry)

    assert len(clusters) == 1
    assert "DIFC Law No. 5 of 2018" in clusters[0].aliases
    assert "COMPANIES LAW" in clusters[0].aliases
