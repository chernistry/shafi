from __future__ import annotations

from shafi.core.entity_registry import EntityRegistry
from shafi.ingestion.canonical_entities import (
    CanonicalEntityType,
    EntityAliasResolver,
)
from shafi.models.legal_objects import (
    CaseObject,
    CaseParty,
    CorpusRegistry,
    LawObject,
    LegalDocType,
    OrderObject,
)


def _build_registry() -> CorpusRegistry:
    return CorpusRegistry(
        source_doc_count=2,
        laws={
            "companies_law": LawObject(
                object_id="law:companies_law",
                doc_id="companies_law",
                title="DIFC Companies Law No. 3 of 2004",
                legal_doc_type=LegalDocType.LAW,
                short_title="Companies Law",
                law_number="3",
                year="2004",
                issuing_authority="Dubai Financial Services Authority",
            )
        },
        orders={
            "data_protection_notice": OrderObject(
                object_id="enactment_notice:data_protection_notice",
                doc_id="data_protection_notice",
                title="_______________________________________________",
                legal_doc_type=LegalDocType.ENACTMENT_NOTICE,
                source_text=(
                    "ENACTMENT NOTICE\n"
                    "We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact\n"
                    "on this 01 day of March 2024\n"
                    "the\n"
                    "DATA PROTECTION LAW\n"
                    "DIFC LAW NO. 5 OF 2020\n"
                ),
            )
        },
        cases={
            "cfi_001_2024": CaseObject(
                object_id="case:cfi_001_2024",
                doc_id="cfi_001_2024",
                title="Alpha Ltd v Dubai Financial Services Authority",
                legal_doc_type=LegalDocType.CASE,
                case_number="CFI 001/2024",
                court="DIFC Courts",
                judges=["Justice Sir Jeremy Cooke"],
                parties=[
                    CaseParty(name="Alpha Ltd", role="Claimant"),
                    CaseParty(name="Dubai Financial Services Authority", role="Respondent"),
                ],
            )
        },
    )


def test_law_aliases_resolve_short_and_numbered_forms() -> None:
    resolver = EntityAliasResolver.build_from_registry(_build_registry())

    short_id = resolver.resolve("Companies Law", CanonicalEntityType.LAW_TITLE)
    numbered_id = resolver.resolve("Law No. 3 of 2004", CanonicalEntityType.LAW_TITLE)
    full_id = resolver.resolve("DIFC Companies Law No. 3 of 2004", CanonicalEntityType.LAW_TITLE)

    assert short_id
    assert short_id == numbered_id == full_id


def test_judge_aliases_resolve_abbreviated_forms() -> None:
    resolver = EntityAliasResolver.build_from_registry(_build_registry())

    full_id = resolver.resolve("Justice Sir Jeremy Cooke", CanonicalEntityType.JUDGE)
    short_id = resolver.resolve("Cooke J", CanonicalEntityType.JUDGE)

    assert full_id
    assert full_id == short_id


def test_law_aliases_include_enactment_notice_titles() -> None:
    resolver = EntityAliasResolver.build_from_registry(_build_registry())

    full_id = resolver.resolve("Data Protection Law", CanonicalEntityType.LAW_TITLE)
    numbered_id = resolver.resolve("Law No. 5 of 2020", CanonicalEntityType.LAW_TITLE)

    assert full_id
    assert full_id == numbered_id


def test_case_number_aliases_resolve_compact_variant() -> None:
    resolver = EntityAliasResolver.build_from_registry(_build_registry())

    spaced_id = resolver.resolve("CFI 001/2024", CanonicalEntityType.CASE_NUMBER)
    compact_id = resolver.resolve("CFI001/2024", CanonicalEntityType.CASE_NUMBER)

    assert spaced_id
    assert spaced_id == compact_id


def test_entity_registry_enriches_query_with_canonical_ids(tmp_path) -> None:
    resolver = EntityAliasResolver.build_from_registry(_build_registry())
    path = resolver.export(tmp_path / "canonical_entity_registry.json")

    registry = EntityRegistry.load(path)
    enriched = registry.enrich_query("Did DFSA act under the Companies Law in CFI 001/2024 before Cooke J?")

    assert len(enriched.canonical_entity_ids) >= 3


def test_export_and_load_round_trip_preserves_cluster_count(tmp_path) -> None:
    resolver = EntityAliasResolver.build_from_registry(_build_registry())
    path = resolver.export(tmp_path / "canonical_entity_registry.json")

    loaded = EntityAliasResolver.load(path)

    assert len(loaded.iter_clusters()) == len(resolver.iter_clusters())
