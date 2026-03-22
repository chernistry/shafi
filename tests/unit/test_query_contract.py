from __future__ import annotations

from rag_challenge.core.query_contract import (
    ExecutionEngine,
    Polarity,
    PredicateType,
    QueryContractCompiler,
    TimeScopeType,
)
from rag_challenge.ingestion.canonical_entities import EntityAliasResolver
from rag_challenge.models.legal_objects import (
    CaseObject,
    CaseParty,
    CorpusRegistry,
    LawObject,
    LegalDocType,
)


def _build_registry() -> CorpusRegistry:
    return CorpusRegistry(
        source_doc_count=3,
        laws={
            "companies_law": LawObject(
                object_id="law:companies_law",
                doc_id="companies_law",
                title="DIFC Companies Law No. 3 of 2004",
                legal_doc_type=LegalDocType.LAW,
                short_title="Companies Law",
                law_number="3",
                year="2004",
            )
        },
        cases={
            "cfi_001_2024": CaseObject(
                object_id="case:cfi_001_2024",
                doc_id="cfi_001_2024",
                title="Alpha Ltd v Beta Ltd",
                legal_doc_type=LegalDocType.CASE,
                case_number="CFI 001/2024",
                judges=["Justice Sir Jeremy Cooke"],
                parties=[
                    CaseParty(name="Alpha Ltd", role="Claimant"),
                    CaseParty(name="Beta Ltd", role="Respondent"),
                ],
            ),
            "cfi_002_2024": CaseObject(
                object_id="case:cfi_002_2024",
                doc_id="cfi_002_2024",
                title="Gamma Ltd v Beta Ltd",
                legal_doc_type=LegalDocType.CASE,
                case_number="CFI 002/2024",
                judges=["Justice Sir Jeremy Cooke"],
                parties=[
                    CaseParty(name="Gamma Ltd", role="Claimant"),
                    CaseParty(name="Beta Ltd", role="Respondent"),
                ],
            ),
        },
    )


def _build_compiler() -> QueryContractCompiler:
    resolver = EntityAliasResolver.build_from_registry(_build_registry())
    return QueryContractCompiler(alias_resolver=resolver)


def test_compile_detects_lookup_field_and_entity_resolution() -> None:
    compiler = _build_compiler()

    contract = compiler.compile(
        "Who issued the Companies Law?",
        answer_type="name",
    )

    assert contract.predicate is PredicateType.LOOKUP_FIELD
    assert contract.field_name == "issued_by"
    assert contract.execution_plan == [ExecutionEngine.FIELD_LOOKUP]
    assert contract.primary_entities[0].canonical_id.startswith("law_title:")
    assert contract.required_support_slots == ["page_with_issued_by_field", "page_for_target_entity"]
    assert contract.confidence >= 0.8


def test_compile_detects_compare_axes_and_primary_entities() -> None:
    compiler = _build_compiler()

    contract = compiler.compile(
        "Did CFI 001/2024 and CFI 002/2024 share a judge?",
        answer_type="boolean",
    )

    assert contract.predicate is PredicateType.COMPARE
    assert contract.execution_plan == [ExecutionEngine.COMPARE_JOIN]
    assert "judge" in contract.comparison_axes
    assert len(contract.primary_entities) == 2
    assert contract.constraint_entities == []


def test_compile_detects_temporal_specific_date_scope() -> None:
    compiler = _build_compiler()

    contract = compiler.compile(
        "Was the Companies Law in force on 1 January 2005?",
        answer_type="boolean",
    )

    assert contract.predicate is PredicateType.TEMPORAL
    assert contract.execution_plan == [ExecutionEngine.TEMPORAL_QUERY]
    assert contract.time_scope.scope_type is TimeScopeType.SPECIFIC_DATE
    assert contract.time_scope.reference_date == "1 January 2005"


def test_compile_detects_lookup_provision_queries() -> None:
    compiler = _build_compiler()

    contract = compiler.compile(
        "What does Article 5 of the Companies Law provide?",
        answer_type="free_text",
    )

    assert contract.predicate is PredicateType.LOOKUP_PROVISION
    assert contract.execution_plan == [ExecutionEngine.STANDARD_RAG]
    assert contract.required_support_slots == ["page_with_provision_text"]


def test_article_anchored_issue_date_question_stays_out_of_field_lookup() -> None:
    compiler = _build_compiler()

    contract = compiler.compile(
        "According to Article 9(9)(a) of the Companies Law, what is the issue date of a licence?",
        answer_type="date",
    )

    assert contract.predicate is PredicateType.LOOKUP_PROVISION
    assert contract.execution_plan == [ExecutionEngine.STANDARD_RAG]


def test_compile_detects_enumeration_and_negative_polarity() -> None:
    compiler = _build_compiler()

    contract = compiler.compile(
        "Which cases did not involve Beta Ltd?",
        answer_type="free_text",
    )

    assert contract.predicate is PredicateType.ENUMERATE
    assert contract.polarity is Polarity.NEGATIVE


def test_resolve_entities_partitions_primary_and_constraint_mentions() -> None:
    compiler = _build_compiler()

    primary_entities, constraint_entities = compiler.resolve_entities(
        "Compare CFI 001/2024 and CFI 002/2024 under the Companies Law."
    )

    assert [entity.canonical_id for entity in primary_entities] == [
        "case_number:cfi0012024",
        "case_number:cfi0022024",
    ]
    assert len(constraint_entities) == 1
    assert constraint_entities[0].canonical_id.startswith("law_title:")


def test_low_confidence_contract_falls_back_to_standard_rag() -> None:
    compiler = QueryContractCompiler()

    contract = compiler.compile("Explain this.", answer_type="free_text")

    assert contract.execution_plan == [ExecutionEngine.STANDARD_RAG]
    assert contract.confidence < 0.55


def test_lookup_field_prioritizes_case_number_over_generic_party_noise() -> None:
    compiler = _build_compiler()

    contract = compiler.compile(
        "Who were the claimants in case CFI 001/2024?",
        answer_type="names",
    )

    assert contract.predicate is PredicateType.LOOKUP_FIELD
    assert contract.field_name == "claimant"
    assert contract.primary_entities
    assert contract.primary_entities[0].canonical_id == "case_number:cfi0012024"
    assert all(entity.mention_text.casefold() not in {"who", "were", "the", "case"} for entity in contract.primary_entities)
    assert all(entity.mention_text.casefold() not in {"who", "were", "the", "case"} for entity in contract.constraint_entities)


def test_lookup_field_extracts_explicit_case_number_without_alias_resolver() -> None:
    compiler = QueryContractCompiler()

    contract = compiler.compile(
        "Who were the claimants in case CFI 001/2024?",
        answer_type="names",
    )

    assert contract.predicate is PredicateType.LOOKUP_FIELD
    assert contract.field_name == "claimant"
    assert contract.primary_entities
    assert contract.primary_entities[0].canonical_id == "case_number:cfi0012024"
    assert contract.primary_entities[0].source_doc_ids == ["cfi_001_2024"]


def test_lookup_field_prioritizes_law_title_for_law_number_queries() -> None:
    compiler = _build_compiler()

    contract = compiler.compile(
        "What is the law number of the Companies Law?",
        answer_type="number",
    )

    assert contract.predicate is PredicateType.LOOKUP_FIELD
    assert contract.field_name == "law_number"
    assert contract.primary_entities
    assert contract.primary_entities[0].canonical_id.startswith("law_title:")


def test_multi_entity_field_query_promotes_to_compare() -> None:
    compiler = _build_compiler()

    contract = compiler.compile(
        "Which case has an earlier decision date: CFI 001/2024 or CFI 002/2024?",
        answer_type="name",
    )

    assert contract.predicate is PredicateType.COMPARE
    assert contract.execution_plan == [ExecutionEngine.COMPARE_JOIN]
    assert len(contract.primary_entities) == 2


def test_common_party_phrase_promotes_lookup_to_compare() -> None:
    compiler = _build_compiler()

    contract = compiler.compile(
        "Which party was common to CFI 001/2024 and CFI 002/2024?",
        answer_type="name",
    )

    assert contract.predicate is PredicateType.COMPARE
    assert contract.execution_plan == [ExecutionEngine.COMPARE_JOIN]
    assert "party" in contract.comparison_axes
    assert len(contract.primary_entities) == 2
