from shafi.ingestion.bridge_facts import BridgeFactGenerator
from shafi.ingestion.canonical_entities import EntityAliasResolver
from shafi.models import BridgeFactType, SegmentType
from shafi.models.legal_objects import (
    ArticleNode,
    CaseObject,
    CaseParty,
    CorpusRegistry,
    LawObject,
    LegalDocType,
    LegalEntity,
    LegalEntityType,
)
from shafi.models.schemas import DocType, LegalSegment


def _build_registry() -> CorpusRegistry:
    return CorpusRegistry(
        source_doc_count=2,
        laws={
            "operating_law": LawObject(
                object_id="law:operating_law",
                doc_id="operating_law",
                title="Operating Law 2018",
                legal_doc_type=LegalDocType.LAW,
                short_title="Operating Law",
                law_number="7",
                year="2018",
                issuing_authority="Dubai Financial Services Authority",
                commencement_date="2018-08-01",
                amendment_refs=["Amendment Law No. 1 of 2019"],
                page_ids=["operating_law_1", "operating_law_2"],
                article_tree=[
                    ArticleNode(
                        article_id="art-1",
                        label="Article 1",
                        title="Definitions",
                        page_ids=["operating_law_1"],
                    )
                ],
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
                cited_law_titles=["Operating Law 2018"],
                outcome_summary="The claim was dismissed.",
                page_ids=["cfi_001_2024_1", "cfi_001_2024_2"],
            )
        },
        entities={
            "party:alpha-ltd": LegalEntity(
                entity_id="party:alpha-ltd",
                name="Alpha Ltd",
                canonical_name="Alpha Ltd",
                entity_type=LegalEntityType.PARTY,
                aliases=["Alpha Ltd"],
                source_doc_ids=["cfi_001_2024"],
            )
        },
    )


def test_generate_bridge_facts_covers_core_fact_types() -> None:
    registry = _build_registry()
    resolver = EntityAliasResolver.build_from_registry(registry)
    generator = BridgeFactGenerator()
    segments = [
        LegalSegment(
            segment_id="operating_law:segment:definition:definitions:operating_law_1",
            segment_type=SegmentType.DEFINITION,
            doc_id="operating_law",
            doc_title="Operating Law 2018",
            doc_type=DocType.STATUTE,
            canonical_doc_id="law:operating_law",
            text='"Approved Person" means a person approved by the Authority.',
            page_ids=["operating_law_1"],
            start_page=1,
            end_page=1,
        )
    ]

    facts = generator.generate_all(corpus_registry=registry, entity_resolver=resolver, segments=segments)
    fact_types = {fact.fact_type for fact in facts}

    assert BridgeFactType.CASE_PARTY in fact_types
    assert BridgeFactType.CASE_JUDGE in fact_types
    assert BridgeFactType.CASE_LAW in fact_types
    assert BridgeFactType.CASE_OUTCOME in fact_types
    assert BridgeFactType.LAW_AUTHORITY in fact_types
    assert BridgeFactType.LAW_AMENDMENT in fact_types
    assert BridgeFactType.LAW_DEFINITION in fact_types
    assert BridgeFactType.LAW_COMMENCEMENT in fact_types
    assert BridgeFactType.ARTICLE_LOCATION in fact_types
    assert BridgeFactType.ENTITY_DOCUMENT in fact_types


def test_bridge_fact_deduplication_merges_doc_and_page_evidence() -> None:
    fact_a = BridgeFactGenerator().generate_entity_document_facts(
        entity=LegalEntity(
            entity_id="party:alpha-ltd",
            name="Alpha Ltd",
            canonical_name="Alpha Ltd",
            entity_type=LegalEntityType.PARTY,
            aliases=["Alpha Ltd"],
            source_doc_ids=["doc-a"],
        )
    )[0]
    fact_b = fact_a.model_copy(
        update={
            "source_doc_ids": ["doc-b"],
            "evidence_page_ids": ["doc-b_1"],
        }
    )

    deduped = BridgeFactGenerator.deduplicate([fact_a, fact_b])

    assert len(deduped) == 1
    assert deduped[0].source_doc_ids == ["doc-a", "doc-b"]
    assert deduped[0].evidence_page_ids == ["doc-b_1"]


def test_bridge_facts_skip_invalid_hash_like_case_number() -> None:
    registry = CorpusRegistry(
        source_doc_count=1,
        cases={
            "hash_case": CaseObject(
                object_id="case:hash_case",
                doc_id="hash_case",
                title="Untitled judgment",
                legal_doc_type=LegalDocType.CASE,
                case_number="be59024d9c82dbc4a8b716e94bd1169387066998e415fb598e677eb9ad104c29",
                judges=["Justice Sir Jeremy Cooke"],
                parties=[CaseParty(name="Alpha Ltd", role="Claimant")],
                page_ids=["hash_case_1"],
            )
        },
    )
    resolver = EntityAliasResolver.build_from_registry(registry)
    generator = BridgeFactGenerator()

    facts = generator.generate_all(corpus_registry=registry, entity_resolver=resolver)

    assert all(fact.fact_type not in {BridgeFactType.CASE_PARTY, BridgeFactType.CASE_JUDGE} for fact in facts)


def test_bridge_facts_skip_low_signal_judge_names() -> None:
    registry = CorpusRegistry(
        source_doc_count=1,
        cases={
            "case_1": CaseObject(
                object_id="case:case_1",
                doc_id="case_1",
                title="Alpha Ltd v Beta Ltd",
                legal_doc_type=LegalDocType.CASE,
                case_number="CFI 001/2024",
                judges=["Registrar\nDate", "Justice Sir Jeremy Cooke"],
                parties=[CaseParty(name="Alpha Ltd", role="Claimant")],
                page_ids=["case_1_1"],
            )
        },
    )
    resolver = EntityAliasResolver.build_from_registry(registry)
    generator = BridgeFactGenerator()

    facts = generator.generate_case_judge_facts(
        case_obj=registry.cases["case_1"],
        entity_resolver=resolver,
    )

    assert len(facts) == 1
    assert facts[0].attributes["judge_name"] == "Justice Sir Jeremy Cooke"


def test_bridge_facts_skip_noisy_party_name_and_entity_document() -> None:
    registry = CorpusRegistry(
        source_doc_count=1,
        cases={
            "case_1": CaseObject(
                object_id="case:case_1",
                doc_id="case_1",
                title="Alpha Ltd v Beta Ltd",
                legal_doc_type=LegalDocType.CASE,
                case_number="CFI 001/2024",
                judges=["Justice Sir Jeremy Cooke"],
                parties=[
                    CaseParty(
                        name="Alpha Ltd, and, shall, filed, intends, relies, served",
                        role="Claimant",
                    )
                ],
                page_ids=["case_1_1"],
            )
        },
        entities={
            "party:noisy": LegalEntity(
                entity_id="party:noisy",
                name="Alpha Ltd, and, shall, filed, intends, relies, served",
                canonical_name="Alpha Ltd, and, shall, filed, intends, relies, served",
                entity_type=LegalEntityType.PARTY,
                aliases=["Alpha Ltd, and, shall, filed, intends, relies, served"],
                source_doc_ids=["case_1"],
            )
        },
    )
    resolver = EntityAliasResolver.build_from_registry(registry)
    generator = BridgeFactGenerator()

    case_party_facts = generator.generate_case_party_facts(
        case_obj=registry.cases["case_1"],
        entity_resolver=resolver,
    )
    entity_document_facts = generator.generate_entity_document_facts(
        entity=registry.entities["party:noisy"],
    )

    assert case_party_facts == []
    assert entity_document_facts == []
