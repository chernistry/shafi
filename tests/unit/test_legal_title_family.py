from __future__ import annotations

from shafi.core.grounding.law_family_graph import (
    build_candidate_law_family_bundle,
    build_query_law_family_bundle,
    law_family_match_score,
)
from shafi.core.grounding.scope_policy import (
    QueryScopePrediction,
    ScopeMode,
    select_sidecar_doc_scope,
)
from shafi.core.legal_title_family import (
    canonical_law_family,
    derive_law_title_aliases,
    derive_related_law_families,
    extract_query_law_families,
)
from shafi.models import DocType, RankedChunk


def _make_chunk(*, doc_id: str, doc_title: str, text: str, law_titles: list[str] | None = None) -> RankedChunk:
    return RankedChunk(
        chunk_id=f"{doc_id}:0:0:abc123",
        doc_id=doc_id,
        doc_title=doc_title,
        doc_type=DocType.STATUTE,
        section_path="page:1",
        text=text,
        retrieval_score=0.9,
        rerank_score=0.9,
        law_titles=law_titles or [],
    )


def test_canonical_law_family_drops_numbering_noise() -> None:
    assert canonical_law_family("DIFC Employment Law No. 2 of 2019") == "DIFC Employment Law"


def test_canonical_law_family_preserves_numbered_generic_shells() -> None:
    assert canonical_law_family("DIFC Law No. 2 of 2019") == "DIFC Law No. 2 of 2019"
    assert canonical_law_family("Ruler of Dubai Law No. 3 of 2021") == "Ruler of Dubai Law No. 3 of 2021"


def test_derive_law_title_aliases_includes_base_family_for_amendment_law() -> None:
    aliases = derive_law_title_aliases("DIFC Employment Law Amendment Law 2020")

    assert "DIFC Employment Law Amendment Law" in aliases
    assert "Employment Law" in " | ".join(aliases)
    assert "DIFC Employment Law" in aliases


def test_derive_law_title_aliases_preserves_specific_numbered_shell_aliases() -> None:
    aliases = derive_law_title_aliases("DIFC Law No. 2 of 2019")

    assert "DIFC Law No. 2 of 2019" in aliases
    assert "DIFC Law" in aliases


def test_derive_law_title_aliases_strips_jurisdiction_prefix_for_substantive_laws() -> None:
    aliases = derive_law_title_aliases("DIFC Trust Law No. 4 of 2018")

    assert "DIFC Trust Law No. 4 of 2018" in aliases
    assert "DIFC Trust Law" in aliases
    assert "Trust Law" in aliases


def test_derive_law_title_aliases_expands_safe_abbreviations() -> None:
    aliases = derive_law_title_aliases("DIFC Intellectual Property Law")

    assert "DIFC Intellectual Property Law" in aliases
    assert "Intellectual Property Law" in aliases
    assert "IP Law" in aliases


def test_derive_related_law_families_links_amendment_to_base_family() -> None:
    families = derive_related_law_families("DIFC Employment Law Amendment Law 2020")

    assert "DIFC Employment Law Amendment Law" in families
    assert "DIFC Employment Law" in families


def test_extract_query_law_families_finds_explicit_law_titles() -> None:
    families = extract_query_law_families("Under the DIFC Employment Law Amendment Law 2020, what changed?")

    assert "employment law amendment law" in families[0]


def test_extract_query_law_families_expands_prefix_and_abbreviation_variants() -> None:
    families = extract_query_law_families("Under the DIFC Trust Law, what applies to IP Law?")

    assert "trust law" in families
    assert "ip law" in families
    assert "intellectual property law" in families


def test_extract_query_law_families_preserves_numbered_generic_shells() -> None:
    families = extract_query_law_families("According to the DIFC Law No. 2 of 2019, what is the title?")

    assert any("law no 2 of 2019" in family for family in families)


def test_select_sidecar_doc_scope_prefers_matching_law_family() -> None:
    scope = QueryScopePrediction(scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC, page_budget=1)
    selected = select_sidecar_doc_scope(
        query="According to the DIFC Employment Law Amendment Law 2020, what is the effective date?",
        scope=scope,
        context_chunks=[
            _make_chunk(
                doc_id="law-a",
                doc_title="DIFC Employment Law Amendment Law 2020",
                text="Title page",
                law_titles=["DIFC Employment Law Amendment Law 2020"],
            ),
            _make_chunk(
                doc_id="law-b",
                doc_title="DIFC Insolvency Law",
                text="Title page",
                law_titles=["DIFC Insolvency Law"],
            ),
        ],
    )

    assert selected == ["law-a"]


def test_select_sidecar_doc_scope_prefers_numbered_generic_shells() -> None:
    scope = QueryScopePrediction(scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC, page_budget=1)
    selected = select_sidecar_doc_scope(
        query="According to the DIFC Law No. 2 of 2019, what is the title?",
        scope=scope,
        context_chunks=[
            _make_chunk(
                doc_id="law-a",
                doc_title="DIFC Law No. 2 of 2019",
                text="Title page",
                law_titles=["DIFC Law No. 2 of 2019"],
            ),
            _make_chunk(
                doc_id="law-b",
                doc_title="DIFC Insolvency Law",
                text="Title page",
                law_titles=["DIFC Insolvency Law"],
            ),
        ],
    )

    assert selected == ["law-a"]


def test_select_sidecar_doc_scope_compare_pair_uses_case_numbers_for_party_name_titles() -> None:
    """Compare-pair scope must select both docs even when one has a party-name title.

    Real case: CA 006/2024 has title "Tr88house Restaurant ... v Bond" — no "CA 006/2024"
    token in the title.  OCR text is empty.  Only case_numbers metadata carries the ref.
    Without the case_numbers fallback, _matches_case_ref returns False → one doc is missed
    → per_doc_cmp only covers one case → wrong or null answer.
    """
    scope = QueryScopePrediction(
        scope_mode=ScopeMode.COMPARE_PAIR,
        page_budget=2,
        target_page_roles=[],
    )
    # "CA 006/2024" doc has party-name title + empty text but case_numbers populated
    ca_chunk = _make_chunk(
        doc_id="ca-006",
        doc_title="Tr88house Restaurant LLC v Bond Interior",
        text="",
    )
    # Inject case_numbers into the chunk (field exists on RankedChunk)
    ca_chunk_with_cn = ca_chunk.model_copy(update={"case_numbers": ["CA 006/2024"]})

    # "ARB 014/2025" doc has normal title with case ref
    arb_chunk = _make_chunk(
        doc_id="arb-014",
        doc_title="ARB 014/2025 Claimant v Respondent",
        text="",
    )

    selected = select_sidecar_doc_scope(
        query="Is there any main party common to case CA 006/2024 and case ARB 014/2025?",
        scope=scope,
        context_chunks=[arb_chunk, ca_chunk_with_cn],
    )

    assert "ca-006" in selected, "CA 006/2024 doc must be selected via case_numbers fallback"
    assert "arb-014" in selected, "ARB 014/2025 doc must be selected via title match"


def test_law_family_graph_matches_enactment_notice_to_base_family() -> None:
    query_bundle = build_query_law_family_bundle(
        "According to the DIFC Employment Law Enactment Notice, when did it commence?"
    )
    candidate_bundle = build_candidate_law_family_bundle(
        "DIFC Employment Law 2019",
        ("DIFC Employment Law 2019",),
    )

    assert law_family_match_score(query_bundle, candidate_bundle) > 0.0
