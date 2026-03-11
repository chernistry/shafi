from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_challenge.models import DocType, QueryComplexity, RankedChunk


def _make_chunks(n: int) -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id=f"doc1:{i}:0:abc",
            doc_id="doc1",
            doc_title="Limitation Act",
            doc_type=DocType.STATUTE,
            section_path=f"Section {i + 1}",
            text=f"The limitation period for action type {i} is {i + 3} years.",
            retrieval_score=0.9 - (i * 0.1),
            rerank_score=0.95 - (i * 0.05),
            doc_summary="",
        )
        for i in range(n)
    ]


@pytest.fixture
def mock_settings():
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            max_context_tokens=2500,
            simple_model="gpt-4o-mini",
            simple_max_tokens=300,
            fallback_model="gpt-4o-mini",
        ),
        pipeline=SimpleNamespace(max_answer_words=250),
    )
    with patch("rag_challenge.llm.generator.get_settings", return_value=settings):
        yield settings


def test_build_prompt_format(mock_settings):
    from rag_challenge.llm.generator import RAGGenerator

    generator = RAGGenerator(llm=MagicMock())
    chunks = _make_chunks(3)

    system, user = generator.build_prompt("What is the limitation period?", chunks)

    assert "legal QA assistant" in system
    assert "ONLY the provided sources" in system
    assert "1. [doc1:0:0:abc] Limitation Act | Section 1" in user
    assert "limitation period for action type 0" in user


def test_build_prompt_complex_uses_complex_prompt(mock_settings):
    from rag_challenge.llm.generator import RAGGenerator

    generator = RAGGenerator(llm=MagicMock())
    system, _ = generator.build_prompt(
        "Compare the difference between X and Y",
        _make_chunks(1),
        complexity=QueryComplexity.COMPLEX,
    )
    assert "Maximum 250 words" in system
    assert "conflict" in system.lower()


def test_build_prompt_strict_uses_strict_user_template(mock_settings):
    from rag_challenge.llm.generator import RAGGenerator

    generator = RAGGenerator(llm=MagicMock())
    _system, user = generator.build_prompt(
        "What is the claim amount?",
        _make_chunks(1),
        answer_type="number",
    )
    assert "Answer type: number" in user
    assert "Output rules:" in user


def test_extract_citations():
    from rag_challenge.llm.generator import RAGGenerator

    chunks = _make_chunks(3)
    answer = (
        "The period is 3 years (cite: doc1:0:0:abc). "
        "For type 1 it is 4 years (cite: doc1:1:0:abc)."
    )

    citations = RAGGenerator.extract_citations(answer, chunks)
    assert len(citations) == 2
    assert citations[0].chunk_id == "doc1:0:0:abc"
    assert citations[0].doc_title == "Limitation Act"
    assert citations[1].chunk_id == "doc1:1:0:abc"


def test_extract_multiple_citations_in_one():
    from rag_challenge.llm.generator import RAGGenerator

    citations = RAGGenerator.extract_cited_chunk_ids("Both apply (cite: c1, c2).")
    assert citations == ["c1", "c2"]


def test_extract_unknown_citation():
    from rag_challenge.llm.generator import RAGGenerator

    citations = RAGGenerator.extract_citations("Unknown (cite: ghost-id).", _make_chunks(1))
    assert len(citations) == 1
    assert citations[0].doc_title == "unknown"


def test_render_context_blocks_groups_named_multi_title_lookup_by_document(mock_settings) -> None:
    from rag_challenge.llm.generator import RAGGenerator

    generator = RAGGenerator(llm=MagicMock())
    chunks = [
        RankedChunk(
            chunk_id="arb:intro",
            doc_id="arb",
            doc_title="Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="We hereby enact the Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="arb:title",
            doc_id="arb",
            doc_title="Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text="This Law may be cited as Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013.",
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="gp:intro",
            doc_id="gp",
            doc_title="General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="We hereby enact the General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013.",
            retrieval_score=0.93,
            rerank_score=0.93,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="gp:title",
            doc_id="gp",
            doc_title="General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text="This Law may be cited as General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013.",
            retrieval_score=0.92,
            rerank_score=0.92,
            doc_summary="",
        ),
    ]

    parts, budget = generator._render_context_blocks(
        question="What are the titles of DIFC Law No. 6 of 2013 and DIFC Law No. 3 of 2013?",
        chunks=chunks,
        complexity=QueryComplexity.SIMPLE,
        answer_type="free_text",
    )

    assert budget == 1600
    assert len(parts) == 2
    assert "DOCUMENT: Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013" in parts[0]
    assert "QUESTION_REF: Law No. 6 of 2013" in parts[0]
    assert "[arb:title]" in parts[0]
    assert "DOCUMENT: General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013" in parts[1]
    assert "QUESTION_REF: Law No. 3 of 2013" in parts[1]
    assert "[gp:title]" in parts[1]


def test_cleanup_list_answer_postamble_trims_trailing_summary() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "1. ICC Regulations (cite: doc1:0:0:abc). "
        "2. IC Regulations (cite: doc2:0:0:def). "
        "No other laws match. Therefore, the laws are: 1."
    )

    cleaned = RAGGenerator.cleanup_list_answer_postamble(answer)

    assert cleaned == "1. ICC Regulations (cite: doc1:0:0:abc). 2. IC Regulations (cite: doc2:0:0:def)."


def test_cleanup_final_answer_removes_trailing_summary_block() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "1. Foundations Law 2018 (cite: c0)\n"
        "2. Trust Law 2018 (cite: c1)\n"
        "Summary:\n"
        "- Foundations Law 2018 applies.\n"
        "- Trust Law 2018 applies."
    )

    cleaned = RAGGenerator.cleanup_final_answer(answer)

    assert cleaned == "1. Foundations Law 2018 (cite: c0)\n2. Trust Law 2018 (cite: c1)"


def test_cleanup_final_answer_drops_truncated_trailing_bullet_line() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "1. Enactment Date:\n"
        "The DIFC Laws Amendment Law, DIFC Law No. 8 of 2018, was enacted on the 5th day of November 2018 "
        "(cite: c0).\n"
        "2. Laws Amended:\n"
        "- TRUST LAW (DIFC Law No. 4 of 2018) (cite: c1)\n"
        "- LIMITED PARTNERSHIP LAW (DIFC Law No."
    )

    cleaned = RAGGenerator.cleanup_final_answer(answer)

    assert cleaned == (
        "1. Enactment Date:\n"
        "The DIFC Laws Amendment Law, DIFC Law No. 8 of 2018, was enacted on the 5th day of November 2018 "
        "(cite: c0).\n"
        "2. Laws Amended:\n"
        "- TRUST LAW (DIFC Law No. 4 of 2018) (cite: c1)"
    )


def test_cleanup_named_multi_title_lookup_answer_rebuilds_last_updated_items() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="companies:0",
            doc_id="companies",
            doc_title="COMPANIES LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="COMPANIES LAW DIFC LAW NO. 5 OF 2018 Consolidated Version (March 2022)",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** COMPANIES LAW",
        ),
        RankedChunk(
            chunk_id="ip:0",
            doc_id="ip",
            doc_title="Intellectual Property Law",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="ENACTMENT NOTICE Intellectual Property Law DIFC Law No. 4 of 2019",
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="**Document Title:** Intellectual Property Law",
        ),
    ]
    answer = (
        "1. DIFC Law No. 5 of 2018 - Title: COMPANIES LAW (cite: companies:0) - Last updated "
        "(consolidated version): March 2022 (cite: companies:0) 2. DIFC Law No. 4 of 2019 - Title: "
        "Intellectual Property Law (cite: ip:0) - The sources do not state when the consolidated version "
        "was last updated for DIFC Law No. 4 of"
    )

    cleaned = RAGGenerator.cleanup_named_multi_title_lookup_answer(
        answer,
        question=(
            "What is the title of DIFC Law No. 5 of 2018 and DIFC Law No. 4 of 2019, and when were their "
            "consolidated versions last updated?"
        ),
        chunks=chunks,
    )

    assert cleaned == (
        "1. Law No. 5 of 2018 - Title: Companies Law 2018 - Last updated (consolidated version): March 2022 "
        "(cite: companies:0)\n"
        "2. Law No. 4 of 2019 - Title: Intellectual Property Law (cite: ip:0) - The provided sources do not "
        "state when the consolidated version was last updated."
    )


def test_cleanup_named_multi_title_lookup_answer_prefers_law_doc_with_consolidated_version() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="ip-law:0",
            doc_id="ip-law",
            doc_title="INTELLECTUAL PROPERTY LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="INTELLECTUAL PROPERTY LAW DIFC LAW No. 4 of 2019 Consolidated Version (March 2022)",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary='**Document Title:** Intellectual Property Law',
        ),
        RankedChunk(
            chunk_id="ip-law:1",
            doc_id="ip-law",
            doc_title="INTELLECTUAL PROPERTY LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text='This Law may be cited as the "Intellectual Property Law 2019".',
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary='**Document Title:** Intellectual Property Law',
        ),
        RankedChunk(
            chunk_id="ip-notice:0",
            doc_id="ip-notice",
            doc_title="_______________________________________________",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="ENACTMENT NOTICE the Intellectual Property Law DIFC Law No. 4 of 2019",
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="Document type: Enactment Notice",
        ),
        RankedChunk(
            chunk_id="companies:0",
            doc_id="companies",
            doc_title="COMPANIES LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="COMPANIES LAW DIFC LAW NO. 5 OF 2018 Consolidated Version (March 2022)",
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary='**Document Title:** COMPANIES LAW',
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_multi_title_lookup_answer(
        "",
        question=(
            "What is the title of DIFC Law No. 5 of 2018 and DIFC Law No. 4 of 2019, and when were their "
            "consolidated versions last updated?"
        ),
        chunks=chunks,
        doc_refs=["Law No. 5 of 2018", "Law No. 4 of 2019"],
    )

    assert "Law No. 4 of 2019 - Title: Intellectual Property Law" in cleaned
    assert "The provided sources do not state when the consolidated version was last updated" not in cleaned
    assert "March 2022" in cleaned
    assert "ip-law:0" in cleaned


def test_cleanup_named_multi_title_lookup_answer_prefers_identity_page_for_title_support() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="companies:cover",
            doc_id="companies",
            doc_title="COMPANIES LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="COMPANIES LAW DIFC LAW NO. 5 OF 2018 Consolidated Version (March 2022)",
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary="**Document Title:** Companies Law 2018",
        ),
        RankedChunk(
            chunk_id="ip:cover",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="INTELLECTUAL PROPERTY LAW DIFC LAW No. 4 of 2019 Consolidated Version (March 2022)",
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="**Document Title:** Intellectual Property Law 2019",
        ),
        RankedChunk(
            chunk_id="ip:title",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text='This Law may be cited as the "Intellectual Property Law 2019".',
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** Intellectual Property Law 2019",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_multi_title_lookup_answer(
        "",
        question=(
            "What is the title of DIFC Law No. 5 of 2018 and DIFC Law No. 4 of 2019, and when were their "
            "consolidated versions last updated?"
        ),
        chunks=chunks,
        doc_refs=["Law No. 5 of 2018", "Law No. 4 of 2019"],
    )

    assert "ip:title" not in cleaned
    assert "ip:cover" in cleaned


def test_cleanup_named_multi_title_lookup_answer_strips_historical_year_from_amendment_titles() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="law6:cover",
            doc_id="law6",
            doc_title="ARBITRATION LAW AMENDMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="ARBITRATION LAW AMENDMENT LAW DIFC LAW No. 6 of 2013",
            retrieval_score=0.98,
            rerank_score=0.98,
            doc_summary="**Document Title:** Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013",
        ),
        RankedChunk(
            chunk_id="law3:cover",
            doc_id="law3",
            doc_title="GENERAL PARTNERSHIP LAW AMENDMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="GENERAL PARTNERSHIP LAW AMENDMENT LAW DIFC LAW No. 3 of 2013",
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary="**Document Title:** General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_multi_title_lookup_answer(
        "",
        question="What are the titles of DIFC Law No. 6 of 2013 and DIFC Law No. 3 of 2013?",
        chunks=chunks,
        doc_refs=["Law No. 6 of 2013", "Law No. 3 of 2013"],
    )

    assert "Arbitration Law Amendment Law, DIFC Law No. 6 of 2013" in cleaned
    assert "General Partnership Law Amendment Law, DIFC Law No. 3 of 2013" in cleaned
    assert "Arbitration Law of 2008 Amendment Law" not in cleaned
    assert "General Partnership Law 2004 Amendment Law" not in cleaned
    assert (
        RAGGenerator._clean_amendment_title_historical_year("Arbitration Law of 2008 Amendment Law")
        == "Arbitration Law Amendment Law"
    )
    assert (
        RAGGenerator._clean_amendment_title_historical_year("General Partnership Law 2004 Amendment Law")
        == "General Partnership Law Amendment Law"
    )


def test_recover_doc_title_from_chunks_prefers_legal_title_over_body_clause() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="companies:0",
            doc_id="companies",
            doc_title="COMPANIES LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text='This Law No. 5 of 2018 may be cited as the "Companies Law". Consolidated Version (March 2022).',
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary='This contract document, titled "COMPANIES LAW," is a consolidated version of DIFC Law No. 5 of 2018.',
        ),
        RankedChunk(
            chunk_id="companies:77",
            doc_id="companies",
            doc_title="COMPANIES LAW",
            doc_type=DocType.STATUTE,
            section_path="page:77",
            text=(
                "Directors to ensure that any accounts prepared by the Company under this Part "
                "comply with the requirements of this Law."
            ),
            retrieval_score=0.8,
            rerank_score=0.8,
            doc_summary='This contract document, titled "COMPANIES LAW," is a consolidated version of DIFC Law No. 5 of 2018.',
        ),
    ]

    recovered = RAGGenerator._recover_doc_title_from_chunks(chunks, prefer_citation_title=True)

    assert recovered == "COMPANIES LAW"


def test_extract_doc_title_from_summary_supports_plain_narrative_summary() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    summary = (
        'This document is the DIFC Leasing Law, consolidated as of March 2022 and amended by '
        'DIFC Law No. 2 of 2022.'
    )

    extracted = RAGGenerator._extract_doc_title_from_summary(summary)

    assert extracted == "DIFC Leasing Law"


def test_extract_doc_title_from_text_supports_cover_title_with_law_year() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    extracted = RAGGenerator._extract_doc_title_from_text(
        "COMPANIES LAW\nDIFC LAW NO. 5 OF 2018\n\nConsolidated Version (March 2022)"
    )

    assert extracted == "Companies Law 2018"


def test_cleanup_named_amendment_answer_rebuilds_complete_amended_law_list() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="amender:0",
            doc_id="amender",
            doc_title="ENACTMENT NOTICE",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We hereby enact on this 5th day of November 2018 the DIFC Laws Amendment Law "
                "DIFC Law No. 8 of 2018."
            ),
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary="**Document Title:** DIFC Laws Amendment Law DIFC Law No. 8 of 2018",
        ),
        RankedChunk(
            chunk_id="trust:0",
            doc_id="trust",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "TRUST LAW DIFC LAW NO. 4 OF 2018 As Amended by DIFC Laws Amendment Law "
                "DIFC Law No. 8 of 2018"
            ),
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="**Document Title:** TRUST LAW (DIFC Law No. 4 of 2018)",
        ),
        RankedChunk(
            chunk_id="lp:0",
            doc_id="lp",
            doc_title="LIMITED PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "LIMITED PARTNERSHIP LAW DIFC LAW NO. 4 OF 2006 As Amended by DIFC Laws Amendment Law "
                "DIFC Law No. 8 of 2018"
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** LIMITED PARTNERSHIP LAW (DIFC Law No. 4 of 2006)",
        ),
    ]
    answer = (
        "1. Enactment Date:\n"
        "The DIFC Laws Amendment Law, DIFC Law No. 8 of 2018, was enacted on the 5th day of November 2018 "
        "(cite: amender:0).\n"
        "2. Laws Amended:\n"
        "- TRUST LAW (DIFC Law No. 4 of 2018) (cite: trust:0)\n"
        "- LIMITED PARTNERSHIP LAW (DIFC Law No."
    )

    cleaned = RAGGenerator.cleanup_named_amendment_answer(
        answer,
        question="When was the DIFC Laws Amendment Law, DIFC Law No. 8 of 2018 enacted and what law did it amend?",
        chunks=chunks,
    )

    assert cleaned == (
        "1. Enactment Date:\n"
        "DIFC Laws Amendment Law DIFC Law No. 8 of 2018 was enacted on 5th day of November 2018 "
        "(cite: amender:0).\n"
        "2. Laws Amended:\n"
        "- TRUST LAW (DIFC Law No. 4 of 2018) (cite: trust:0)\n"
        "- LIMITED PARTNERSHIP LAW (DIFC Law No. 4 of 2006) (cite: lp:0)"
    )


def test_build_structured_free_text_answer_handles_named_amendment_query(mock_settings) -> None:
    from rag_challenge.llm.generator import RAGGenerator

    generator = RAGGenerator(llm=MagicMock())
    chunks = [
        RankedChunk(
            chunk_id="amender:0",
            doc_id="amender",
            doc_title="ENACTMENT NOTICE",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We hereby enact on this 5th day of November 2018 the DIFC Laws Amendment Law "
                "DIFC Law No. 8 of 2018."
            ),
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary="**Document Title:** DIFC Laws Amendment Law DIFC Law No. 8 of 2018",
        ),
        RankedChunk(
            chunk_id="trust:0",
            doc_id="trust",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "TRUST LAW DIFC LAW NO. 4 OF 2018 As Amended by DIFC Laws Amendment Law "
                "DIFC Law No. 8 of 2018"
            ),
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="**Document Title:** TRUST LAW (DIFC Law No. 4 of 2018)",
        ),
    ]

    answer = generator.build_structured_free_text_answer(
        question="When was the DIFC Laws Amendment Law, DIFC Law No. 8 of 2018 enacted and what law did it amend?",
        chunks=chunks,
        doc_refs=["DIFC Law No. 8 of 2018"],
    )

    assert "Enactment Date" in answer
    assert "Laws Amended" in answer
    assert "trust:0" in answer


def test_build_structured_free_text_answer_handles_amended_by_enumeration_query(mock_settings) -> None:
    from rag_challenge.llm.generator import RAGGenerator

    generator = RAGGenerator(llm=MagicMock())
    chunks = [
        RankedChunk(
            chunk_id="foundations:0",
            doc_id="foundations",
            doc_title="FOUNDATIONS LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "FOUNDATIONS LAW DIFC LAW NO. 3 OF 2018 As Amended by DIFC Laws Amendment Law "
                "DIFC Law No. 2 of 2022"
            ),
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary="**Document Title:** FOUNDATIONS LAW (DIFC Law No. 3 of 2018)",
        ),
        RankedChunk(
            chunk_id="companies:0",
            doc_id="companies",
            doc_title="COMPANIES LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "COMPANIES LAW DIFC LAW NO. 5 OF 2018 As Amended by DIFC Laws Amendment Law "
                "DIFC Law No. 2 of 2022"
            ),
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="**Document Title:** COMPANIES LAW (DIFC Law No. 5 of 2018)",
        ),
    ]

    answer = generator.build_structured_free_text_answer(
        question="Which specific DIFC Laws were amended by DIFC Law No. 2 of 2022?",
        chunks=chunks,
        doc_refs=["Law No. 2 of 2022"],
    )

    assert answer == (
        "1. FOUNDATIONS LAW (DIFC Law No. 3 of 2018) (cite: foundations:0)\n"
        "2. COMPANIES LAW (DIFC Law No. 5 of 2018) (cite: companies:0)"
    )


def test_strip_negative_subclaims_removes_explicit_mention_disclaimer() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "1. ICC Regulations (cite: doc1:0:0:abc). "
        "2. IC Regulations (cite: doc2:0:0:def). "
        "There is no explicit mention of the Insolvency Law 2009 in the other regulations."
    )

    cleaned = RAGGenerator.strip_negative_subclaims(answer)

    assert cleaned == "1. ICC Regulations (cite: doc1:0:0:abc). 2. IC Regulations (cite: doc2:0:0:def)."


def test_strip_negative_subclaims_removes_inline_common_elements_disclaimer() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "1. Schedule 1 Structure: All three laws state that Schedule 1 contains interpretative provisions "
        "and a list of defined terms (cite: c0, c1, c2). "
        "The Trust Law 2018 only states that Schedule 1 contains interpretative provisions and a list of "
        "defined terms, but does not provide the substantive interpretative rules themselves."
    )

    cleaned = RAGGenerator.strip_negative_subclaims(answer)

    assert cleaned == (
        "1. Schedule 1 Structure: All three laws state that Schedule 1 contains interpretative provisions "
        "and a list of defined terms (cite: c0, c1, c2)."
    )


def test_cleanup_list_answer_preamble_trims_uncited_preface() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "The following are the common elements found in the interpretation sections. "
        "1. Schedule 1 Structure: All three laws state that Schedule 1 contains interpretative provisions "
        "(cite: c0, c1, c2)."
    )

    cleaned = RAGGenerator.cleanup_list_answer_preamble(answer)

    assert cleaned == "1. Schedule 1 Structure: All three laws state that Schedule 1 contains interpretative provisions (cite: c0, c1, c2)."


def test_cleanup_numbered_list_items_drops_common_elements_doc_specific_items() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "1. Schedule 1 of the Operating Law 2018 contains interpretative provisions (cite: c0). "
        "2. Schedule 1 of the Trust Law 2018 contains interpretative provisions (cite: c1). "
        "3. Interpretative provisions apply to the respective Law, and defined terms are listed (cite: c0, c1)."
    )

    cleaned = RAGGenerator.cleanup_numbered_list_items(
        answer,
        question="What are the common elements found in Schedule 1 of the Operating Law 2018 and the Trust Law 2018?",
        common_elements=True,
    )

    assert cleaned == "1. Interpretative provisions apply to the respective Law, and defined terms are listed (cite: c0, c1)."


def test_cleanup_common_elements_canonical_answer_merges_by_supported_signatures() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="operating:0",
            doc_id="operating",
            doc_title="OPERATING LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="Schedule 1 contains interpretative provisions which apply to the Law and a list of defined terms used in the Law.",
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="**Document Title:** Operating Law 2018",
        ),
        RankedChunk(
            chunk_id="trust:0",
            doc_id="trust",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="Schedule 1 contains interpretative provisions which apply to the Law and a list of defined terms used in the Law.",
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="**Document Title:** Trust Law 2018",
        ),
    ]
    answer = (
        "1. Schedule 1 of the Operating Law 2018 contains interpretative provisions which apply to the Law "
        "(cite: operating:0). 2. Schedule 1 of the Trust Law 2018 contains interpretative provisions which "
        "apply to the Law (cite: trust:0). 3. A list of defined terms used in the respective Law "
        "(cite: operating:0, trust:0)."
    )

    cleaned = RAGGenerator.cleanup_common_elements_canonical_answer(
        answer,
        question="What are the common elements found in Schedule 1 of the Operating Law 2018 and the Trust Law 2018?",
        chunks=chunks,
    )

    assert cleaned == (
        "1. Schedule 1 contains interpretative provisions which apply to the Law. (cite: operating:0, trust:0)\n"
        "2. Schedule 1 contains a list of defined terms used in the Law. (cite: operating:0, trust:0)"
    )


def test_question_named_refs_strips_interpretation_sections_prefix() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    refs = RAGGenerator._question_named_refs(
        question=(
            "What are the common elements found in the interpretation sections of the "
            "Operating Law 2018, Trust Law 2018, and Common Reporting Standard Law 2018?"
        )
    )

    assert refs == [
        "Operating Law 2018",
        "Trust Law 2018",
        "Common Reporting Standard Law 2018",
    ]


def test_cleanup_common_elements_canonical_answer_rebuilds_interpretation_section_overlap() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="operating:schedule",
            doc_id="operating",
            doc_title="OPERATING LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="Schedule 1 contains interpretative provisions which apply to the Law and a list of defined terms used in the Law.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** Operating Law 2018",
        ),
        RankedChunk(
            chunk_id="operating:rule",
            doc_id="operating",
            doc_title="OPERATING LAW",
            doc_type=DocType.STATUTE,
            section_path="page:39",
            text=(
                "SCHEDULE 1 INTERPRETATION. Rules of interpretation. "
                "A reference to a statutory provision includes a reference to the statutory provision as amended "
                "or re-enacted from time to time. A reference to a person includes any natural person, body "
                "corporate or body unincorporate, including a company, partnership, unincorporated association, "
                "government or state."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="**Document Title:** Operating Law 2018",
        ),
        RankedChunk(
            chunk_id="trust:schedule",
            doc_id="trust",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="Schedule 1 contains interpretative provisions which apply to the Law and a list of defined terms used in the Law.",
            retrieval_score=0.93,
            rerank_score=0.93,
            doc_summary="**Document Title:** Trust Law 2018",
        ),
        RankedChunk(
            chunk_id="trust:rule",
            doc_id="trust",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:43",
            text=(
                "SCHEDULE 1 INTERPRETATION. Rules of interpretation. "
                "A reference to a statutory provision includes a reference to the statutory provision as amended "
                "or re-enacted from time to time. A reference to a person includes any natural person, body "
                "corporate or body unincorporate, including a company, partnership, unincorporated association, "
                "government or state."
            ),
            retrieval_score=0.92,
            rerank_score=0.92,
            doc_summary="**Document Title:** Trust Law 2018",
        ),
        RankedChunk(
            chunk_id="crs:rule",
            doc_id="crs",
            doc_title="COMMON REPORTING STANDARD LAW",
            doc_type=DocType.STATUTE,
            section_path="page:14",
            text=(
                "Rules of Interpretation. A reference to a statutory provision includes a reference to the "
                "statutory provision as amended or re-enacted from time to time. A reference to a person "
                "includes any natural person, body corporate or body unincorporate, including a company, "
                "partnership, unincorporated association, government or state."
            ),
            retrieval_score=0.91,
            rerank_score=0.91,
            doc_summary="**Document Title:** Common Reporting Standard Law 2018",
        ),
        RankedChunk(
            chunk_id="crs:defined-terms",
            doc_id="crs",
            doc_title="COMMON REPORTING STANDARD LAW",
            doc_type=DocType.STATUTE,
            section_path="page:15",
            text="Section 3 defines terms used in the Law.",
            retrieval_score=0.90,
            rerank_score=0.90,
            doc_summary="**Document Title:** Common Reporting Standard Law 2018",
        ),
    ]
    answer = "1. Schedule 1 contains interpretative provisions which apply to the Law and a list of defined terms used in the Law (cite: operating:schedule, trust:schedule, crs:defined-terms)."

    cleaned = RAGGenerator.cleanup_common_elements_canonical_answer(
        answer,
        question=(
            "What are the common elements found in the interpretation sections of the Operating Law 2018, "
            "Trust Law 2018, and Common Reporting Standard Law 2018?"
        ),
        chunks=chunks,
    )

    assert cleaned == (
        "1. A statutory provision includes a reference to the statutory provision as amended or re-enacted from time to time. "
        "(cite: operating:rule, trust:rule, crs:rule)\n"
        "2. A reference to a person includes any natural person, body corporate or body unincorporate, including a company, "
        "partnership, unincorporated association, government or state. (cite: operating:rule, trust:rule, crs:rule)"
    )


def test_cleanup_numbered_list_items_drops_uncited_and_conclusion_items() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "1. ICC Regulations mention both the Companies Law 2018 and the Insolvency Law 2009 (cite: c0). "
        "2. The Investment Companies regulations mention the Insolvency Law 2009. "
        "3. Therefore, only the ICC Regulations match (cite: c0)."
    )

    cleaned = RAGGenerator.cleanup_numbered_list_items(
        answer,
        question="Which laws explicitly mention the Companies Law 2018 and the Insolvency Law 2009 in their regulations concerning company structures?",
        common_elements=False,
    )

    assert cleaned == "1. ICC Regulations mention both the Companies Law 2018 and the Insolvency Law 2009 (cite: c0)."


def test_cleanup_broad_enumeration_titles_only_strips_verbose_schedule_detail() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "1. This is confirmed by the text in the Foundations Law 2018 document and the explicit Schedule 2 titled "
        "\"APPLICATION OF THE ARBITRATION LAW,\" which contains detailed provisions on how the Arbitration Law applies "
        "to Foundations (cite: c0). "
        "2. TRUST LAW includes provisions relating to the application of the Arbitration Law in its Schedule 2 "
        "(cite: c1)."
    )

    cleaned = RAGGenerator.cleanup_broad_enumeration_titles_only(
        answer,
        question="Which laws, enacted in 2018, include provisions relating to the application of the Arbitration Law in their Schedule 2?",
    )

    assert cleaned == "1. Foundations Law 2018 (cite: c0)\n2. TRUST LAW (cite: c1)"


def test_cleanup_broad_enumeration_titles_only_recovers_placeholder_from_cited_doc() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="found:0",
            doc_id="found",
            doc_title="FOUNDATIONS LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="Schedule 2 titled APPLICATION OF THE ARBITRATION LAW applies to Foundations.",
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="**Document Title:** Foundations Law 2018",
        ),
        RankedChunk(
            chunk_id="trust:0",
            doc_id="trust",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="Schedule 2 titled APPLICATION OF THE ARBITRATION LAW applies to trusts.",
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="**Document Title:** Trust Law 2018",
        ),
    ]
    answer = (
        "1. Foundations Law 2018 (cite: found:0)\n"
        "2. This is shown by the statement (cite: trust:0)"
    )

    cleaned = RAGGenerator.cleanup_broad_enumeration_titles_only(
        answer,
        question="Which laws, enacted in 2018, include provisions relating to the application of the Arbitration Law in their Schedule 2?",
        chunks=chunks,
    )

    assert cleaned == "1. Foundations Law 2018 (cite: found:0)\n2. Trust Law 2018 (cite: trust:0)"


def test_cleanup_broad_enumeration_titles_only_overrides_wrong_model_title_with_citation_title() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="found:0",
            doc_id="found",
            doc_title="FOUNDATIONS LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="Schedule 2 titled APPLICATION OF THE ARBITRATION LAW applies to Foundations.",
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="**Document Title:** Foundations Law 2018",
        ),
        RankedChunk(
            chunk_id="trust:0",
            doc_id="trust",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="Schedule 2 titled APPLICATION OF THE ARBITRATION LAW applies to trusts.",
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="**Document Title:** Trust Law 2018",
        ),
    ]
    answer = (
        "1. Foundations Law 2018 (cite: found:0)\n"
        "2. Foundations Law 2018 (cite: trust:0)"
    )

    cleaned = RAGGenerator.cleanup_broad_enumeration_titles_only(
        answer,
        question="Which laws, enacted in 2018, include provisions relating to the application of the Arbitration Law in their Schedule 2?",
        chunks=chunks,
    )

    assert cleaned == "1. Foundations Law 2018 (cite: found:0)\n2. Trust Law 2018 (cite: trust:0)"


def test_cleanup_registrar_enumeration_items_splits_multi_doc_item() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="gp:admin",
            doc_id="gp",
            doc_title="General Partnership Law 2004",
            doc_type=DocType.STATUTE,
            section_path="page:21",
            text="The Registrar shall administer this Law.",
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="llp:admin",
            doc_id="llp",
            doc_title="Limited Liability Partnership Law 2004",
            doc_type=DocType.STATUTE,
            section_path="page:22",
            text="The Registrar shall administer this Law.",
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="",
        ),
    ]
    answer = (
        "1. General Partnership Law, DIFC Law No.11 of 2004 — This law was enacted in 2004 and is administered "
        "by the Registrar (cite: gp:admin) (cite: llp:admin)."
    )

    cleaned = RAGGenerator.cleanup_registrar_enumeration_items(answer, chunks)

    assert cleaned == (
        "1. General Partnership Law 2004 (cite: gp:admin)\n"
        "2. Limited Liability Partnership Law 2004 (cite: llp:admin)"
    )


def test_cleanup_registrar_enumeration_items_recovers_citation_title_placeholder() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="gp:title",
            doc_id="gp",
            doc_title="GENERAL PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text='This Law may be cited as "General Partnership Law 2004". The Registrar shall administer this Law.',
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="**Document Title:** General Partnership Law 2004",
        ),
        RankedChunk(
            chunk_id="llp:title",
            doc_id="llp",
            doc_title="LIMITED LIABILITY PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text='This Law may be cited as "Limited Liability Partnership Law 2004". The Registrar shall administer this Law.',
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="**Document Title:** Limited Liability Partnership Law 2004",
        ),
    ]
    answer = "1. Citation title (cite: gp:title, llp:title)"

    cleaned = RAGGenerator.cleanup_registrar_enumeration_items(answer, chunks)

    assert cleaned == (
        "1. General Partnership Law 2004 (cite: gp:title)\n"
        "2. Limited Liability Partnership Law 2004 (cite: llp:title)"
    )


def test_cleanup_named_commencement_answer_requires_per_law_support() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="law3:0",
            doc_id="law3",
            doc_title="Enactment Notice",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "The DIFC Laws Amendment Law DIFC Law No. 3 of 2024 made by the Ruler of Dubai. "
                "This Law shall come into force on the 5th business day after enactment "
                "(not counting the day of enactment for this purpose)."
            ),
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="**Document Title:** DIFC Laws Amendment Law DIFC Law No. 3 of 2024",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_commencement_answer(
        "The common commencement date is the 5th business day after enactment for both laws.",
        question=(
            "What is the common commencement date for the DIFC Laws Amendment Law, DIFC Law No. 3 of 2024, "
            "and the Law of Security Law, DIFC Law No. 4 of 2024?"
        ),
        chunks=chunks,
        doc_refs=["Law No. 3 of 2024", "Law No. 4 of 2024"],
    )

    assert cleaned == (
        "1. DIFC Laws Amendment Law DIFC Law No. 3 of 2024: "
        "Shall come into force on the 5th business day after enactment "
        "(not counting the day of enactment for this purpose) (cite: law3:0)\n"
        "2. Law No. 4 of 2024: The provided sources do not contain its commencement provision."
    )


def test_cleanup_named_commencement_answer_merges_shared_rule_when_all_named_laws_supported() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="law3:0",
            doc_id="law3",
            doc_title="Enactment Notice",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "The DIFC Laws Amendment Law DIFC Law No. 3 of 2024 made by the Ruler of Dubai. "
                "This Law shall come into force on the 5th business day after enactment "
                "(not counting the day of enactment for this purpose)."
            ),
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="**Document Title:** DIFC Laws Amendment Law DIFC Law No. 3 of 2024",
        ),
        RankedChunk(
            chunk_id="law4:0",
            doc_id="law4",
            doc_title="Enactment Notice",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "The Law of Security Law DIFC Law No. 4 of 2024 made by the Ruler of Dubai. "
                "This Law shall come into force on the 5th business day after enactment "
                "(not counting the day of enactment for this purpose)."
            ),
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="**Document Title:** Law of Security Law DIFC Law No. 4 of 2024",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_commencement_answer(
        "The common commencement date is the 5th business day after enactment for both laws.",
        question=(
            "What is the common commencement date for the DIFC Laws Amendment Law, DIFC Law No. 3 of 2024, "
            "and the Law of Security Law, DIFC Law No. 4 of 2024?"
        ),
        chunks=chunks,
        doc_refs=["Law No. 3 of 2024", "Law No. 4 of 2024"],
    )

    assert cleaned == (
        "The common commencement rule for DIFC Laws Amendment Law DIFC Law No. 3 of 2024 and "
        "Law of Security Law DIFC Law No. 4 of 2024 is Shall come into force on the 5th business "
        "day after enactment (not counting the day of enactment for this purpose) "
        "(cite: law3:0, law4:0)"
    )


def test_cleanup_named_commencement_answer_prefers_exact_title_support_over_amendment_doc() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="amend:0",
            doc_id="amend",
            doc_title="DIFC AMENDMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text=(
                "This Law may be cited as the DIFC Laws Amendment Law, DIFC Law No.1 of 2025. "
                "This Law comes into force on the date specified in the Enactment Notice in respect of this Law."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** DIFC Laws Amendment Law DIFC Law No.1 of 2025",
        ),
        RankedChunk(
            chunk_id="data:title",
            doc_id="data",
            doc_title=".",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text='This Law may be cited as the "Data Protection Law 2020".',
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="data:comm",
            doc_id="data",
            doc_title=".",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="4. Commencement This Law comes into force on 1 July 2020.",
            retrieval_score=0.93,
            rerank_score=0.93,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="employment:title",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text='This Employment Law 2019 may be cited as the "Employment Law 2019".',
            retrieval_score=0.92,
            rerank_score=0.92,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="employment:comm",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text=(
                "6. Commencement This Law comes into force on the date ninety (90) days following "
                "the date specified in the Enactment Notice."
            ),
            retrieval_score=0.91,
            rerank_score=0.91,
            doc_summary="",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_commencement_answer(
        (
            "1. DIFC Laws Amendment Law, DIFC Law No.1 of 2025: Comes into force on the date specified "
            "in the Enactment Notice in respect of this Law (cite: amend:0)\n"
            "2. Employment Law 2019: The provided sources do not contain its commencement provision."
        ),
        question="What is the commencement date for the Data Protection Law 2020 and the Employment Law 2019?",
        chunks=chunks,
        doc_refs=["Data Protection Law 2020", "Employment Law 2019"],
    )

    assert cleaned == (
        "1. Data Protection Law 2020: Comes into force on 1 July 2020 (cite: data:comm)\n"
        "2. EMPLOYMENT LAW: Comes into force on the date ninety (90) days following the date specified in the Enactment Notice (cite: employment:comm)"
    )


def test_cleanup_named_commencement_answer_prefers_enactment_notice_over_cross_reference_doc() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="law3:0",
            doc_id="law3",
            doc_title="Enactment Notice",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We hereby enact the DIFC Laws Amendment Law DIFC Law No. 3 of 2024. "
                "This Law shall come into force on the 5th business day after enactment "
                "(not counting the day of enactment for this purpose)."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** DIFC Laws Amendment Law DIFC Law No. 3 of 2024",
        ),
        RankedChunk(
            chunk_id="law4:0",
            doc_id="law4",
            doc_title="Enactment Notice",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We hereby enact the Law of Security Law DIFC Law No. 4 of 2024. "
                "This Law shall come into force on the 5th business day after enactment "
                "(not counting the day of enactment for this purpose)."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="**Document Title:** Law of Security Law DIFC Law No. 4 of 2024",
        ),
        RankedChunk(
            chunk_id="real:0",
            doc_id="real",
            doc_title="REAL PROPERTY LAW",
            doc_type=DocType.STATUTE,
            section_path="page:8",
            text=(
                "Article 9 refers to Law No. 4 of 2024. "
                "Date of commencement: This Law comes into force on the date specified in the Enactment Notice for the Law."
            ),
            retrieval_score=0.93,
            rerank_score=0.93,
            doc_summary="**Document Title:** Real Property Law",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_commencement_answer(
        (
            "1. DIFC Law Amendment Law DIFC Law No. 3 of 2024: Shall come into force on the 5th business day after enactment "
            "(not counting the day of enactment for this purpose) (cite: law3:0)\n"
            "2. REAL PROPERTY LAW: Comes into force on the date specified in the Enactment Notice (cite: real:0)"
        ),
        question=(
            "What is the common commencement date for the DIFC Laws Amendment Law, DIFC Law No. 3 of 2024, "
            "and the Law of Security Law, DIFC Law No. 4 of 2024?"
        ),
        chunks=chunks,
        doc_refs=["Law No. 3 of 2024", "Law No. 4 of 2024"],
    )

    assert "REAL PROPERTY LAW" not in cleaned
    assert "Law of Security Law DIFC Law No. 4 of 2024" in cleaned
    assert "law3:0, law4:0" in cleaned


def test_cleanup_named_commencement_answer_strips_enactment_boilerplate_from_labels() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="law3:0",
            doc_id="law3",
            doc_title="in the form now attached the DIFC Law Amendment Law 2024",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We hereby enact the DIFC Laws Amendment Law DIFC Law No. 3 of 2024. "
                "This Law shall come into force on the 5th business day after enactment "
                "(not counting the day of enactment for this purpose)."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** DIFC Laws Amendment Law DIFC Law No. 3 of 2024",
        ),
        RankedChunk(
            chunk_id="law4:0",
            doc_id="law4",
            doc_title="in the form now attached the Law of Security Law 2024",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We hereby enact the Law of Security Law DIFC Law No. 4 of 2024. "
                "This Law shall come into force on the 5th business day after enactment "
                "(not counting the day of enactment for this purpose)."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_commencement_answer(
        "The common commencement date is the 5th business day after enactment for both laws.",
        question=(
            "What is the common commencement date for the DIFC Laws Amendment Law, DIFC Law No. 3 of 2024, "
            "and the Law of Security Law, DIFC Law No. 4 of 2024?"
        ),
        chunks=chunks,
        doc_refs=["Law No. 3 of 2024", "Law No. 4 of 2024"],
    )

    assert "in the form now attached" not in cleaned
    assert cleaned == (
        "The common commencement rule for DIFC Laws Amendment Law DIFC Law No. 3 of 2024 and "
        "Law of Security Law 2024 is Shall come into force on the 5th business day after enactment "
        "(not counting the day of enactment for this purpose) (cite: law3:0, law4:0)"
    )


def test_cleanup_named_administration_answer_rebuilds_per_law_clauses() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="llp:admin",
            doc_id="llp",
            doc_title="LIMITED LIABILITY PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                "This Law may be cited as the Limited Liability Partnership Law 2004. "
                "This Law and any legislation made for the purpose of this Law is administered by the Registrar."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="npio:admin",
            doc_id="npio",
            doc_title="NON PROFIT INCORPORATED ORGANISATIONS LAW",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text=(
                "This Law may be cited as the Non Profit Incorporated Organisations Law. "
                "This Law and any legislation made for the purposes of this Law are administered by the Registrar."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_administration_answer(
        "Both laws are administered by the Registrar.",
        question="How do the Limited Liability Partnership Law and the Non Profit Incorporated Organisations Law define their administration?",
        chunks=chunks,
        doc_refs=["Limited Liability Partnership Law", "Non Profit Incorporated Organisations Law"],
    )

    assert cleaned == (
        "1. Limited Liability Partnership Law 2004: This Law and any legislation made for the purpose of this Law is administered by the Registrar (cite: llp:admin)\n"
        "2. Non Profit Incorporated Organisations Law: This Law and any legislation made for the purposes of this Law are administered by the Registrar (cite: npio:admin)"
    )


def test_cleanup_named_administration_answer_supports_difca_and_regulations_made_under_it() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="leasing:admin",
            doc_id="leasing",
            doc_title="LEASING LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                'This Law No. 1 of 2020 may be cited as the "Leasing Law 2020". '
                "This Law and any Regulations made under it shall be administered by the DIFCA."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="trust:admin",
            doc_id="trust",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                'This Law may be cited as the "Trust Law 2018". '
                "Administration of this Law This Law is administered by the DIFCA."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_administration_answer(
        "",
        question="What entity administers the Leasing Law 2020 and the Trust Law 2018?",
        chunks=chunks,
        doc_refs=["Leasing Law 2020", "Trust Law 2018"],
    )

    assert cleaned == (
        "1. Leasing Law 2020: DIFCA (cite: leasing:admin)\n"
        "2. Trust Law 2018: DIFCA (cite: trust:admin)"
    )


def test_cleanup_named_administration_answer_prefers_limited_liability_over_limited_partnership() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="lp:admin",
            doc_id="lp",
            doc_title="LIMITED PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                'This Law may be cited as the "Limited Partnership Law 2006". '
                "This Law is administered by the Registrar."
            ),
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary="**Document Title:** Limited Partnership Law 2006",
        ),
        RankedChunk(
            chunk_id="llp:admin",
            doc_id="llp",
            doc_title="LIMITED LIABILITY PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                'This Law may be cited as the "Limited Liability Partnership Law 2004". '
                "This Law and any legislation made for the purpose of this Law is administered by the Registrar."
            ),
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="**Document Title:** Limited Liability Partnership Law 2004",
        ),
        RankedChunk(
            chunk_id="npio:admin",
            doc_id="npio",
            doc_title="NON PROFIT INCORPORATED ORGANISATIONS LAW",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text=(
                'This Law may be cited as the "Non Profit Incorporated Organisations Law". '
                "This Law and any legislation made for the purposes of this Law are administered by the Registrar."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** Non Profit Incorporated Organisations Law",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_administration_answer(
        "",
        question="How do the Limited Liability Partnership Law and the Non Profit Incorporated Organisations Law define their administration?",
        chunks=chunks,
        doc_refs=["Limited Liability Partnership Law", "Non Profit Incorporated Organisations Law"],
    )

    assert "Limited Partnership Law 2006" not in cleaned
    assert cleaned == (
        "1. Limited Liability Partnership Law 2004: This Law and any legislation made for the purpose of this Law is administered by the Registrar (cite: llp:admin)\n"
        "2. Non Profit Incorporated Organisations Law: This Law and any legislation made for the purposes of this Law are administered by the Registrar (cite: npio:admin)"
    )


def test_question_named_refs_strips_how_do_lead_tokens() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    refs = RAGGenerator._question_named_refs(
        question="How do the Limited Liability Partnership Law and the Non Profit Incorporated Organisations Law define their administration?",
        extra_refs=None,
        prefer_extra_refs=False,
    )

    assert refs == [
        "Limited Liability Partnership Law",
        "Non Profit Incorporated Organisations Law",
    ]


def test_question_named_refs_strips_generic_question_leads_from_named_law_titles() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    refs = RAGGenerator._question_named_refs(
        question="On what date was the Employment Law Amendment Law enacted?",
        extra_refs=None,
        prefer_extra_refs=False,
    )

    assert refs == ["Employment Law Amendment Law"]


def test_question_named_refs_strips_preposition_leads_from_named_law_titles() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    refs = RAGGenerator._question_named_refs(
        question=(
            "What are the effective dates for pre-existing and new accounts under the "
            "Common Reporting Standard Law 2018, and what is the date of its enactment?"
        ),
        extra_refs=None,
        prefer_extra_refs=False,
    )

    assert refs == ["Common Reporting Standard Law 2018"]


def test_build_registrar_enumeration_answer_rebuilds_titles_from_sources() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="gp:title",
            doc_id="gp",
            doc_title="GENERAL PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="This Law may be cited as the General Partnership Law 2004.",
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="gp:admin",
            doc_id="gp",
            doc_title="GENERAL PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="This Law and any legislation made for the purpose of this Law is administered by the Registrar.",
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="lp:title",
            doc_id="lp",
            doc_title="LIMITED PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="This Law may be cited as the Limited Partnership Law 2006.",
            retrieval_score=0.88,
            rerank_score=0.88,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="lp:admin",
            doc_id="lp",
            doc_title="LIMITED PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="This Law and any legislation made for the purpose of this Law is administered by the Registrar.",
            retrieval_score=0.87,
            rerank_score=0.87,
            doc_summary="",
        ),
    ]

    built = RAGGenerator.build_registrar_enumeration_answer(
        question="Which laws are administered by the Registrar and what are their respective citation titles?",
        chunks=chunks,
    )

    assert built == (
        "1. General Partnership Law 2004 (cite: gp:title, gp:admin)\n"
        "2. Limited Partnership Law 2006 (cite: lp:title, lp:admin)"
    )


def test_build_registrar_enumeration_answer_respects_enacted_year_filter() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="gp:title",
            doc_id="gp",
            doc_title="GENERAL PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="This Law may be cited as the General Partnership Law 2004.",
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="gp:admin",
            doc_id="gp",
            doc_title="GENERAL PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="This Law and any legislation made for the purpose of this Law is administered by the Registrar.",
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="amend:title",
            doc_id="amend",
            doc_title="GENERAL PARTNERSHIP LAW AMENDMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="This Law may be cited as the General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013.",
            retrieval_score=0.88,
            rerank_score=0.88,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="amend:admin",
            doc_id="amend",
            doc_title="GENERAL PARTNERSHIP LAW AMENDMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="This Law and any legislation made for the purpose of this Law is administered by the Registrar.",
            retrieval_score=0.87,
            rerank_score=0.87,
            doc_summary="",
        ),
    ]

    built = RAGGenerator.build_registrar_enumeration_answer(
        question="Which laws are administered by the Registrar and were enacted in 2004?",
        chunks=chunks,
    )

    assert built == "1. General Partnership Law 2004 (cite: gp:title, gp:admin)"


def test_build_ruler_authority_year_enumeration_answer_rebuilds_titles() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="found:title",
            doc_id="found",
            doc_title="FOUNDATIONS LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="This Law may be cited as the Foundations Law 2018.",
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="found:ruler",
            doc_id="found",
            doc_title="FOUNDATIONS LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai, hereby enact this Law.",
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="trust:title",
            doc_id="trust",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="This Law may be cited as the Trust Law 2018.",
            retrieval_score=0.88,
            rerank_score=0.88,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="trust:ruler",
            doc_id="trust",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai, hereby enact this Law.",
            retrieval_score=0.87,
            rerank_score=0.87,
            doc_summary="",
        ),
    ]

    built = RAGGenerator.build_ruler_authority_year_enumeration_answer(
        question="Which laws mention the Ruler of Dubai as the legislative authority and were enacted in 2018?",
        chunks=chunks,
    )

    assert built == (
        "1. Foundations Law 2018 (cite: found:title, found:ruler)\n"
        "2. Trust Law 2018 (cite: trust:title, trust:ruler)"
    )


def test_cleanup_account_effective_dates_answer_rebuilds_crs_dates() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="crs:0",
            doc_id="crs",
            doc_title="COMMON REPORTING",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text=(
                "This Law may be cited as the Common Reporting Standard Law 2018. "
                "The Law is enacted on the date specified in the Enactment Notice in respect of this Law. "
                "The Law comes into force on the date specified in the Enactment Notice in respect of this Law, except "
                "in respect of Pre-existing Accounts that are subject to due diligence requirements under this Law, "
                "the effective date is 31 December, 2016; and in respect of New Accounts that are subject to due "
                "diligence requirements under this Law, the effective date is 1 January, 2017."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** Common Reporting Standard Law 2018",
        ),
        RankedChunk(
            chunk_id="crs:notice",
            doc_id="crs-notice",
            doc_title="Enactment Notice",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We hereby enact on this 14th day of March 2018 the Common Reporting Standard Law DIFC Law No. 2 of 2018. "
                "This Law shall come into force on the 5th business day after enactment."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="**Document Title:** Common Reporting Standard Law DIFC Law No. 2 of 2018",
        ),
    ]

    cleaned = RAGGenerator.cleanup_account_effective_dates_answer(
        (
            "1. DIFC References to legislation in the Law: Comes into force on the date specified in the Enactment Notice "
            "(cite: crs:0)\n"
            "2. Common Reporting Standard Law DIFC Law No. 2 of 2018: Shall come into force on the 5th business day after enactment "
            "(cite: crs:notice)"
        ),
        question=(
            "What are the effective dates for pre-existing and new accounts under the Common Reporting Standard Law 2018, "
            "and what is the date of its enactment?"
        ),
        chunks=chunks,
        doc_refs=["Common Reporting Standard Law 2018"],
    )

    assert cleaned == (
        "1. Pre-existing Accounts: The effective date is 31 December, 2016 (cite: crs:0)\n"
        "2. New Accounts: The effective date is 1 January, 2017 (cite: crs:0)\n"
        "3. Common Reporting Standard Law 2018: The date of enactment is 14th day of March 2018 (cite: crs:notice)"
    )


def test_cleanup_account_effective_dates_answer_falls_back_to_enactment_notice_reference() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="crs:0",
            doc_id="crs",
            doc_title="COMMON REPORTING",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text=(
                "This Law may be cited as the Common Reporting Standard Law 2018. "
                "This Law is enacted on the date specified in the Enactment Notice in respect of this Law. "
                "This Law comes into force on the date specified in the Enactment Notice in respect of this Law, except "
                "in respect of Pre-existing Accounts that are subject to due diligence requirements under this Law, "
                "the effective date is 31 December, 2016; and in respect of New Accounts that are subject to due "
                "diligence requirements under this Law, the effective date is 1 January, 2017."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** Common Reporting Standard Law 2018",
        ),
    ]

    cleaned = RAGGenerator.cleanup_account_effective_dates_answer(
        "",
        question=(
            "What are the effective dates for pre-existing and new accounts under the Common Reporting Standard Law 2018, "
            "and what is the date of its enactment?"
        ),
        chunks=chunks,
        doc_refs=["Common Reporting Standard Law 2018"],
    )

    assert cleaned == (
        "1. Pre-existing Accounts: The effective date is 31 December, 2016 (cite: crs:0)\n"
        "2. New Accounts: The effective date is 1 January, 2017 (cite: crs:0)\n"
        "3. Common Reporting Standard Law 2018: The date of enactment is the date specified in the Enactment Notice in respect of this Law (cite: crs:0)"
    )


def test_cleanup_account_effective_dates_answer_uses_opaque_notice_chunk_for_exact_date() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="crs:0",
            doc_id="crs",
            doc_title="COMMON REPORTING",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text=(
                "This Law may be cited as the Common Reporting Standard Law 2018. "
                "This Law is enacted on the date specified in the Enactment Notice in respect of this Law. "
                "This Law comes into force on the date specified in the Enactment Notice in respect of this Law, except "
                "in respect of Pre-existing Accounts that are subject to due diligence requirements under this Law, "
                "the effective date is 31 December, 2016; and in respect of New Accounts that are subject to due "
                "diligence requirements under this Law, the effective date is 1 January, 2017."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="crs:notice",
            doc_id="crs-notice",
            doc_title="_______________________________________________",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We hereby enact on this 14th day of March 2018 the Common Reporting Standard Law DIFC Law No. 2 of 2018. "
                "This Law shall come into force on the 5th business day after enactment."
            ),
            retrieval_score=0.41,
            rerank_score=0.41,
            doc_summary="",
        ),
    ]

    cleaned = RAGGenerator.cleanup_account_effective_dates_answer(
        "",
        question=(
            "What are the effective dates for pre-existing and new accounts under the Common Reporting Standard Law 2018, "
            "and what is the date of its enactment?"
        ),
        chunks=chunks,
        doc_refs=["Common Reporting Standard Law 2018"],
    )

    assert cleaned == (
        "1. Pre-existing Accounts: The effective date is 31 December, 2016 (cite: crs:0)\n"
        "2. New Accounts: The effective date is 1 January, 2017 (cite: crs:0)\n"
        "3. Common Reporting Standard Law 2018: The date of enactment is 14th day of March 2018 (cite: crs:notice)"
    )


def test_cleanup_account_effective_dates_answer_handles_effective_dates_without_enactment_request() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="crs:0",
            doc_id="crs",
            doc_title="COMMON REPORTING",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text=(
                "This Law comes into force on the date specified in the Enactment Notice in respect of this Law, except "
                "in respect of Pre-existing Accounts that are subject to due diligence requirements under this Law, "
                "the effective date is 31 December, 2016; and in respect of New Accounts that are subject to due "
                "diligence requirements under this Law, the effective date is 1 January, 2017."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** Common Reporting Standard Law 2018",
        ),
    ]

    cleaned = RAGGenerator.cleanup_account_effective_dates_answer(
        "",
        question=(
            "What is the effective date for due diligence requirements for Pre-existing Accounts and "
            "New Accounts under the Common Reporting Standard Law 2018?"
        ),
        chunks=chunks,
        doc_refs=["Common Reporting Standard Law 2018"],
    )

    assert cleaned == (
        "1. Pre-existing Accounts: The effective date is 31 December, 2016 (cite: crs:0)\n"
        "2. New Accounts: The effective date is 1 January, 2017 (cite: crs:0)"
    )


def test_cleanup_named_retention_period_answer_rebuilds_crs_and_general_partnership() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="crs:0",
            doc_id="crs",
            doc_title="COMMON REPORTING STANDARD LAW",
            doc_type=DocType.STATUTE,
            section_path="page:7",
            text=(
                "All records required to be kept by Reporting Financial Institutions pursuant to the provisions of "
                "this Law and the Regulations shall be retained in an electronically readable format for a retention "
                "period of six (6) years after the date of reporting the information."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** Common Reporting Standard Law 2018",
        ),
        RankedChunk(
            chunk_id="gp:0",
            doc_id="gp",
            doc_title="GENERAL PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                "A General Partnership's Accounting Records shall be preserved by the General Partnership for at "
                "least six (6) years from the date upon which they were created, or for some other period as may "
                "be prescribed in the Regulations."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="**Document Title:** General Partnership Law 2004",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_retention_period_answer(
        "",
        question=(
            "According to Article 12(4) of the Common Reporting Standard Law and Article 18(2)(b) of the "
            "General Partnership Law, what are the retention periods for records?"
        ),
        chunks=chunks,
        doc_refs=["Common Reporting Standard Law 2018", "General Partnership Law 2004"],
    )

    assert cleaned == (
        "1. Common Reporting Standard Law 2018: Records must be retained for a retention period of six (6) years "
        "after the date of reporting the information. (cite: crs:0)\n"
        "2. General Partnership Law 2004: Accounting Records must be preserved by the General Partnership for at "
        "least six (6) years from the date upon which they were created. (cite: gp:0)"
    )


def test_cleanup_named_retention_period_answer_rebuilds_general_partnership_and_llp() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="gp:0",
            doc_id="gp",
            doc_title="GENERAL PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                "A General Partnership's Accounting Records shall be preserved by the General Partnership for at "
                "least six (6) years from the date upon which they were created, or for some other period as may "
                "be prescribed in the Regulations."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** General Partnership Law 2004",
        ),
        RankedChunk(
            chunk_id="llp:0",
            doc_id="llp",
            doc_title="LIMITED LIABILITY PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:10",
            text=(
                "A Limited Liability Partnership's Accounting Records shall be preserved by the Limited Liability "
                "Partnership for at least six (6) years from the date upon which they were created, or for some "
                "other period as may be Prescribed in the Regulations."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="**Document Title:** Limited Liability Partnership Law 2018",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_retention_period_answer(
        "",
        question=(
            "What is the minimum period for which a DIFC-incorporated General Partnership and a DIFC-incorporated "
            "Limited Liability Partnership must preserve their accounting records?"
        ),
        chunks=chunks,
    )

    assert cleaned == (
        "1. General Partnership Law 2004: Accounting Records must be preserved by the General Partnership for at "
        "least six (6) years from the date upon which they were created. (cite: gp:0)\n"
        "2. Limited Liability Partnership Law 2018: Accounting Records must be preserved by the Limited Liability "
        "Partnership for at least six (6) years from the date upon which they were created. (cite: llp:0)"
    )


def test_cleanup_named_administration_answer_handles_single_law_without_duplication() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="employment:0",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text="This Law and any Regulations made under it shall be administered by the DIFCA.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** Employment Law 2019",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_administration_answer(
        "",
        question="Who is responsible for administering the Employment Law and any Regulations made under it?",
        chunks=chunks,
        doc_refs=["Employment Law 2019"],
    )

    assert cleaned == "DIFCA administers Employment Law 2019 and any Regulations made under it (cite: employment:0)"


def test_cleanup_interpretative_provisions_enumeration_items_recovers_law_titles() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    chunks = [
        RankedChunk(
            chunk_id="gp:schedule",
            doc_id="gp",
            doc_title="General Partnership Law 2004",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="Schedule 1 contains: (a) interpretative provisions which apply to this Law; and (b) defined terms.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** General Partnership Law 2004",
        ),
        RankedChunk(
            chunk_id="rp:schedule",
            doc_id="rp",
            doc_title="REAL PROPERTY LAW",
            doc_type=DocType.STATUTE,
            section_path="page:8",
            text=(
                "Property within the jurisdiction of the DIFC governed by this Law and which forms part of a Lot "
                "for which a Folio has been created under the provisions of this Law. "
                "(3) Schedule 1 contains: (a) interpretative provisions which apply to this Law; and "
                "(b) a list of defined terms used in this Law."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="**Document Title:** Real Property Law 2018",
        ),
    ]

    cleaned = RAGGenerator.cleanup_interpretative_provisions_enumeration_items(
        "1. General Partnership Law 2004 (cite: gp:schedule)\n"
        "2. jurisdiction of the DIFC governed by this Law and which forms part of a Lot for which a Folio has been created "
        "under the provisions of this Law (cite: rp:schedule)",
        chunks=chunks,
    )

    assert cleaned == (
        "1. General Partnership Law 2004 (cite: gp:schedule)\n"
        "2. Real Property Law 2018 (cite: rp:schedule)"
    )


def test_cleanup_named_penalty_answer_rebuilds_per_regulation_clauses() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    question = (
        "What is the prescribed penalty for an offense against the Strata Title Law under the Strata Title Regulations, "
        "and what is the penalty for using leased premises for an illegal purpose under the Leasing Regulations?"
    )
    chunks = [
        RankedChunk(
            chunk_id="strata:penalty",
            doc_id="strata",
            doc_title="STRATA TITLE REGULATIONS",
            doc_type=DocType.REGULATION,
            section_path="page:2",
            text="PENALTY FOR OFFENCES AGAINST THE LAW. The sources here do not state a specific monetary amount.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="**Document Title:** Strata Title Regulations",
        ),
        RankedChunk(
            chunk_id="leasing:penalty",
            doc_id="leasing",
            doc_title="LEASING REGULATIONS",
            doc_type=DocType.REGULATION,
            section_path="page:5",
            text="Use of the Leased Premises for any purpose that is illegal 10,000",
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="**Document Title:** Leasing Regulations",
        ),
        RankedChunk(
            chunk_id="leasing:transitional",
            doc_id="leasing",
            doc_title="LEASING REGULATIONS",
            doc_type=DocType.REGULATION,
            section_path="page:4",
            text=(
                "Where a Security Deposit has been paid by a Lessee in connection with a Residential Lease entered into "
                "prior to the date of commencement of the Law, the provisions of Articles 24 to 29 of the Law shall only "
                "apply to such Lease upon the same parties entering into a new Residential Lease of the same Leased Premises."
            ),
            retrieval_score=0.93,
            rerank_score=0.93,
            doc_summary="This statute, titled \"Leasing Regulations,\" governs leasing activities in the DIFC.",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_penalty_answer(
        "",
        question=question,
        chunks=chunks,
        doc_refs=["Strata Title Regulations", "Leasing Regulations"],
    )

    assert cleaned == (
        "1. Strata Title Regulations: The provided sources do not specify the penalty.\n"
        "2. Leasing Regulations: The penalty is 10,000 USD (cite: leasing:penalty)"
    )


def test_cleanup_named_penalty_answer_prefers_doc_group_with_actual_penalty_clause() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    question = (
        "What is the prescribed penalty for an offense against the Strata Title Law under the Strata Title Regulations, "
        "and what is the penalty for using leased premises for an illegal purpose under the Leasing Regulations?"
    )
    chunks = [
        RankedChunk(
            chunk_id="strata:toc",
            doc_id="strata-toc",
            doc_title="STRATA TITLE REGULATIONS",
            doc_type=DocType.REGULATION,
            section_path="page:2",
            text="PENALTY FOR OFFENCES AGAINST THE LAW",
            retrieval_score=0.98,
            rerank_score=0.98,
            doc_summary='Statute: Strata Title Regulations (DIFC jurisdiction), effective 12 November 2018.',
        ),
        RankedChunk(
            chunk_id="strata:detail",
            doc_id="strata-detail",
            doc_title="STRATA TITLE REGULATIONS",
            doc_type=DocType.REGULATION,
            section_path="page:3",
            text=(
                "3. PENALTY FOR OFFENCES AGAINST THE LAW. "
                "The penalty for an offence against the Law is one thousand dollars (US$ 1,000)."
            ),
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary='Statute: Strata Title Regulations (DIFC jurisdiction), effective 12 November 2018.',
        ),
        RankedChunk(
            chunk_id="leasing:detail",
            doc_id="leasing-detail",
            doc_title="LEASING REGULATIONS",
            doc_type=DocType.REGULATION,
            section_path="page:5",
            text="Use of the Leased Premises for any purpose that is illegal 10,000",
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary='Statute: Leasing Regulations (DIFC jurisdiction).',
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_penalty_answer(
        "",
        question=question,
        chunks=chunks,
        doc_refs=["Strata Title Regulations", "Leasing Regulations"],
    )

    assert cleaned == (
        "1. Strata Title Regulations: The penalty is 1,000 USD (cite: strata:detail)\n"
        "2. Leasing Regulations: The penalty is 10,000 USD (cite: leasing:detail)"
    )


def test_extract_doc_title_from_text_rejects_commencement_clause_as_title() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    extracted = RAGGenerator._extract_doc_title_from_text(
        "Where a Security Deposit has been paid by a Lessee in connection with a Residential Lease entered "
        "into prior to the date of commencement of the Law, the provisions of Articles 24 to 29 of the Law "
        "shall only apply to such Lease."
    )

    assert extracted == ""


def test_get_context_debug_stats_uses_compact_budget_for_slow_broad_enumeration(mock_settings) -> None:
    from rag_challenge.llm.generator import RAGGenerator

    generator = RAGGenerator(llm=MagicMock())
    chunks = _make_chunks(12)

    count, budget = generator.get_context_debug_stats(
        question="Which laws are administered by the Registrar and what are their respective citation titles?",
        chunks=chunks,
        complexity=QueryComplexity.COMPLEX,
        answer_type="free_text",
    )

    assert count > 0
    assert budget == 1600


def test_get_context_debug_stats_uses_named_multi_lookup_budget(mock_settings) -> None:
    from rag_challenge.llm.generator import RAGGenerator

    generator = RAGGenerator(llm=MagicMock())
    count, budget = generator.get_context_debug_stats(
        question="What is the commencement date for the Data Protection Law 2020 and the Employment Law 2019?",
        chunks=[
            RankedChunk(
                chunk_id="data:0",
                doc_id="data",
                doc_title="Data Protection Law 2020",
                doc_type=DocType.STATUTE,
                section_path="page:4",
                text="This Law comes into force on 1 July 2020.",
                retrieval_score=0.9,
                rerank_score=0.9,
                doc_summary="",
            ),
            RankedChunk(
                chunk_id="employment:0",
                doc_id="employment",
                doc_title="Employment Law 2019",
                doc_type=DocType.STATUTE,
                section_path="page:5",
                text="This Law comes into force 90 days following the date specified in the Enactment Notice.",
                retrieval_score=0.89,
                rerank_score=0.89,
                doc_summary="",
            ),
        ],
        complexity=QueryComplexity.COMPLEX,
        answer_type="free_text",
    )

    assert count == 2
    assert budget == 1600


def test_cleanup_final_answer_strips_dangling_numbered_tail() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    cleaned = RAGGenerator.cleanup_final_answer("1. Foundations Law 2018 (cite: c0)\n2.")

    assert cleaned == "1. Foundations Law 2018 (cite: c0)"


def test_cleanup_final_answer_strips_trailing_inline_list_postamble() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "1. Enactment Date: 5 November 2018 (cite: enact:0). "
        "2. Law(s) Amended: TRUST LAW (cite: trust:0) - FOUNDATIONS LAW (cite: foundations:0) "
        "These are the laws explicitly identified in the provided sources as having been amended by DIFC Law No. 8 of"
    )

    cleaned = RAGGenerator.cleanup_final_answer(answer)

    assert cleaned == (
        "1. Enactment Date: 5 November 2018 (cite: enact:0). "
        "2. Law(s) Amended: TRUST LAW (cite: trust:0) - FOUNDATIONS LAW (cite: foundations:0)."
    )


def test_cleanup_final_answer_strips_incomplete_negative_tail_after_complete_sentence() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "DIFC Law No. 2 of 2022 amended the Foundations Law 2018, as indicated in the consolidated version "
        "of the Foundations Law 2018 which lists DIFC Law No. 2 of 2022 among the amending laws "
        "(cite: foundations:0). There is no information in the provided sources about other specific DIFC Laws "
        "amended by DIFC Law No. 2 of"
    )

    cleaned = RAGGenerator.cleanup_final_answer(answer)

    assert cleaned == (
        "DIFC Law No. 2 of 2022 amended the Foundations Law 2018, as indicated in the consolidated version "
        "of the Foundations Law 2018 which lists DIFC Law No. 2 of 2022 among the amending laws "
        "(cite: foundations:0)."
    )


def test_cleanup_named_ref_enumeration_items_rebinds_to_supporting_chunk() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = (
        "1. Incorporated Cell Company (ICC) Regulations — explicitly mention both the Companies Law 2018 and "
        "the Insolvency Law 2009 in their regulations concerning company structures (cite: icc:late).\n"
        "2. Investment Companies (IC) Regulations (cite: ic:good)"
    )
    chunks = [
        RankedChunk(
            chunk_id="icc:good",
            doc_id="icc",
            doc_title="Incorporated Cell Company (ICC) Regulations",
            doc_type=DocType.REGULATION,
            section_path="1.1.2",
            text="These Regulations are made under the Companies Law 2018 and define terms in the Insolvency Law 2009.",
            retrieval_score=1.0,
            rerank_score=1.0,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="icc:late",
            doc_id="icc",
            doc_title="Incorporated Cell Company (ICC) Regulations",
            doc_type=DocType.REGULATION,
            section_path="1.6.6",
            text="The provisions in the Law and the Insolvency Law apply to the Articles of Association.",
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="ic:good",
            doc_id="ic",
            doc_title="Investment Companies (IC) Regulations",
            doc_type=DocType.REGULATION,
            section_path="1.1.2",
            text="Capitalised terms are defined in the Companies Law 2018 and the Insolvency Law 2009.",
            retrieval_score=0.8,
            rerank_score=0.8,
            doc_summary="",
        ),
    ]

    cleaned = RAGGenerator.cleanup_named_ref_enumeration_items(
        answer,
        question="Which laws explicitly mention the Companies Law 2018 and the Insolvency Law 2009 in their regulations concerning company structures?",
        chunks=chunks,
    )

    assert cleaned == (
        "1. Incorporated Cell Company (ICC) Regulations (cite: icc:good)\n"
        "2. Investment Companies (IC) Regulations (cite: ic:good)"
    )


def test_cleanup_ruler_enactment_enumeration_items_keeps_only_supported_laws() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = "1. Wrong Law (cite: bad:1)"
    chunks = [
        RankedChunk(
            chunk_id="good:1",
            doc_id="good",
            doc_title="_______________________________________________",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                "We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact "
                "in the form now attached the The Law of Damages and Remedies DIFC Law No. 7 of 2005. "
                "This Law shall come into force on the 5th business day after enactment."
            ),
            retrieval_score=1.0,
            rerank_score=1.0,
            doc_summary="**Document Title:** The Law of Damages and Remedies 2005 Enactment Notice",
        ),
        RankedChunk(
            chunk_id="bad:1",
            doc_id="bad",
            doc_title="DIFC Laws Amendment Law",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="This Law comes into force on the date specified in the Enactment Notice in respect of this Law.",
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        ),
    ]

    cleaned = RAGGenerator.cleanup_ruler_enactment_enumeration_items(answer, chunks=chunks)

    assert cleaned == (
        "1. The Law of Damages and Remedies 2005: Shall come into force on the 5th business day after enactment "
        "(cite: good:1)"
    )


def test_cleanup_ruler_enactment_enumeration_items_recovers_relative_commencement_and_title() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    answer = "1. Assets Law (cite: d:1)\n2. Employment Law (cite: e:1)"
    chunks = [
        RankedChunk(
            chunk_id="d:1",
            doc_id="digital",
            doc_title="_______________________________________________",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact "
                "in the form now attached the Digital Assets Law DIFC Law No. 2 of 2024. "
                "This Law shall come into force on the 5th business day after enactment."
            ),
            retrieval_score=1.0,
            rerank_score=1.0,
            doc_summary="**Document Title:** Digital Assets Law Enactment Notice",
        ),
        RankedChunk(
            chunk_id="e:1",
            doc_id="employment-amendment",
            doc_title="_______________________________________________",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact "
                "in the form now attached the Employment Law Amendment Law DIFC Law No. 4 of 2021. "
                "This Law shall come into force on 90 days after enactment."
            ),
            retrieval_score=0.98,
            rerank_score=0.98,
            doc_summary="**Document Title:** Employment Law Amendment Law Enactment Notice",
        ),
    ]

    cleaned = RAGGenerator.cleanup_ruler_enactment_enumeration_items(answer, chunks=chunks)

    assert cleaned == (
        "1. Digital Assets Law DIFC Law No. 2 of 2024: Shall come into force on the 5th business day after enactment (cite: d:1)\n"
        "2. Employment Law Amendment Law DIFC Law No. 4 of 2021: Shall come into force on 90 days after enactment (cite: e:1)"
    )


def test_common_elements_queries_do_not_use_irac() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    assert (
        RAGGenerator._should_use_irac(
            "What are the common elements found in the interpretation sections of the Operating Law 2018 and the Trust Law 2018?"
        )
        is False
    )


def test_display_doc_title_prefers_extracted_citation_title_over_generic_uppercase_header() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    generator = RAGGenerator(llm=MagicMock())
    chunk = RankedChunk(
        chunk_id="doc1:0:0:abc",
        doc_id="doc1",
        doc_title="LIMITED PARTNERSHIP LAW",
        doc_type=DocType.STATUTE,
        section_path="page:1",
        text='Title This Law may be cited as the "Limited Partnership Law 2006".',
        retrieval_score=0.9,
        rerank_score=0.9,
        doc_summary="",
    )

    assert generator._display_doc_title(chunk) == "Limited Partnership Law 2006"


def test_display_doc_title_extracts_quoted_title_without_optional_the_before_quote() -> None:
    from rag_challenge.llm.generator import RAGGenerator

    generator = RAGGenerator(llm=MagicMock())
    chunk = RankedChunk(
        chunk_id="doc2:0:0:def",
        doc_id="doc2",
        doc_title="LAW OF DAMAGES AND REMEDIES",
        doc_type=DocType.STATUTE,
        section_path="page:1",
        text='Title This Law may be cited as "The Law of Damages and Remedies 2005”.',
        retrieval_score=0.9,
        rerank_score=0.9,
        doc_summary="",
    )

    assert generator._display_doc_title(chunk) == "The Law of Damages and Remedies 2005"


def test_context_token_budget(mock_settings):
    from rag_challenge.llm.generator import RAGGenerator

    mock_settings.llm.max_context_tokens = 50
    generator = RAGGenerator(llm=MagicMock())
    chunks = _make_chunks(10)

    _, user = generator.build_prompt("test", chunks)
    chunk_count = user.count("[doc1:")
    assert chunk_count < 10


@pytest.mark.asyncio
async def test_generate_returns_answer_and_citations(mock_settings):
    from rag_challenge.llm.generator import RAGGenerator
    from rag_challenge.llm.provider import LLMResult

    llm = MagicMock()
    llm.generate_with_cascade = AsyncMock(
        return_value=LLMResult(
            text="Answer text (cite: doc1:0:0:abc).",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            model="gpt-4o-mini",
            latency_ms=12.0,
        )
    )

    generator = RAGGenerator(llm=llm)
    answer, citations = await generator.generate("Q?", _make_chunks(2))

    assert "Answer text" in answer
    assert len(citations) == 1
    assert citations[0].chunk_id == "doc1:0:0:abc"


@pytest.mark.asyncio
async def test_generate_stream_updates_collector_usage(mock_settings):
    from rag_challenge.llm.generator import RAGGenerator
    from rag_challenge.telemetry import TelemetryCollector

    async def fake_stream_generate(**_: object):
        for token in ["Hello", " ", "world"]:
            yield token

    llm = MagicMock()
    llm.stream_generate_with_cascade = fake_stream_generate
    llm.get_last_stream_model = MagicMock(return_value="")
    llm.get_last_stream_usage = MagicMock(
        return_value={"prompt_tokens": 77, "completion_tokens": 9, "total_tokens": 86}
    )

    generator = RAGGenerator(llm=llm)
    collector = TelemetryCollector(request_id="req-1")
    tokens = [
        token
        async for token in generator.generate_stream(
            "Q?",
            _make_chunks(1),
            model="gpt-4o-mini",
            max_tokens=50,
            collector=collector,
        )
    ]

    payload = collector.finalize()
    assert "".join(tokens) == "Hello world"
    assert payload.model_llm == "gpt-4o-mini"
    assert payload.prompt_tokens == 77
    assert payload.completion_tokens == 9
    assert payload.total_tokens == 86


@pytest.mark.asyncio
async def test_generate_stream_falls_back_to_estimated_usage_when_provider_missing(mock_settings):
    from rag_challenge.llm.generator import RAGGenerator
    from rag_challenge.telemetry import TelemetryCollector

    async def fake_stream_generate(**_: object):
        for token in ["Hello", " ", "world"]:
            yield token

    llm = MagicMock()
    llm.stream_generate_with_cascade = fake_stream_generate
    llm.get_last_stream_model = MagicMock(return_value="")
    llm.get_last_stream_usage = MagicMock(
        return_value={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )

    generator = RAGGenerator(llm=llm)
    collector = TelemetryCollector(request_id="req-2")
    tokens = [
        token
        async for token in generator.generate_stream(
            "Q?",
            _make_chunks(1),
            model="gpt-4o-mini",
            max_tokens=50,
            collector=collector,
        )
    ]

    payload = collector.finalize()
    assert "".join(tokens) == "Hello world"
    assert payload.prompt_tokens > 0
    assert payload.completion_tokens > 0
    assert payload.total_tokens >= payload.prompt_tokens + payload.completion_tokens


@pytest.mark.asyncio
async def test_generate_falls_back_to_estimated_usage_when_provider_missing(mock_settings):
    from rag_challenge.llm.generator import RAGGenerator
    from rag_challenge.llm.provider import LLMResult
    from rag_challenge.telemetry import TelemetryCollector

    llm = MagicMock()
    llm.generate_with_cascade = AsyncMock(
        return_value=LLMResult(
            text="Answer text (cite: doc1:0:0:abc).",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            model="gpt-4o-mini",
            latency_ms=12.0,
        )
    )

    generator = RAGGenerator(llm=llm)
    collector = TelemetryCollector(request_id="req-3")
    answer, _citations = await generator.generate(
        "Q?",
        _make_chunks(2),
        collector=collector,
    )

    payload = collector.finalize()
    assert "Answer text" in answer
    assert payload.prompt_tokens > 0
    assert payload.completion_tokens > 0
    assert payload.total_tokens >= payload.prompt_tokens + payload.completion_tokens
