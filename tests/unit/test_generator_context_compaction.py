from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from shafi.llm.generator import RAGGenerator
from shafi.models import DocType, QueryComplexity, RankedChunk


def _chunk(text: str) -> RankedChunk:
    return RankedChunk(
        chunk_id="doc:0:0:abc",
        doc_id="doc",
        doc_title="Doc",
        doc_type=DocType.CASE_LAW,
        section_path="Section 1",
        text=text,
        retrieval_score=0.9,
        rerank_score=0.9,
    )


def test_strict_context_is_compacted_and_budgeted() -> None:
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            max_context_tokens=2500,
            simple_model="gpt-4o-mini",
            simple_max_tokens=300,
            fallback_model="gpt-4o-mini",
        ),
        pipeline=SimpleNamespace(
            max_answer_words=250,
            context_token_budget_boolean=900,
            context_token_budget_strict=1100,
            context_token_budget_free_text_simple=1800,
            context_token_budget_free_text_complex=2400,
        ),
    )

    irrelevant = "This sentence is irrelevant to the question."
    relevant = "The DIFC Court states there is no jury system in these proceedings."
    huge_text = " ".join([irrelevant] * 80 + [relevant] + [irrelevant] * 80)

    with patch("shafi.llm.generator.get_settings", return_value=settings):
        generator = RAGGenerator(llm=MagicMock())
        _system, user = generator.build_prompt(
            "What did the jury decide in ENF 053/2025?",
            [_chunk(huge_text)],
            complexity=QueryComplexity.SIMPLE,
            answer_type="boolean",
        )

    assert "no jury system" in user.lower()
    # Ensure compaction prevents runaway context growth.
    assert len(user) < len(huge_text)


def test_broad_enumeration_prioritizes_unique_law_titles() -> None:
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            max_context_tokens=2500,
            simple_model="gpt-4o-mini",
            simple_max_tokens=300,
            fallback_model="gpt-4o-mini",
        ),
        pipeline=SimpleNamespace(
            max_answer_words=250,
            context_token_budget_boolean=900,
            context_token_budget_strict=1100,
            context_token_budget_free_text_simple=1800,
            context_token_budget_free_text_complex=2400,
        ),
    )

    enactment_notice = RankedChunk(
        chunk_id="notice:0:0:abc",
        doc_id="notice",
        doc_title="ENACTMENT NOTICE",
        doc_type=DocType.STATUTE,
        section_path="page:1",
        text=(
            "ENACTMENT NOTICE We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact "
            "in the form now attached the Employment Law DIFC Law No. 2 of 2019."
        ),
        retrieval_score=0.9,
        rerank_score=0.9,
    )
    law_page = RankedChunk(
        chunk_id="law:0:0:def",
        doc_id="law",
        doc_title="Employment Law DIFC Law No. 2 of 2019",
        doc_type=DocType.STATUTE,
        section_path="page:1",
        text="Employment Law DIFC Law No. 2 of 2019",
        retrieval_score=0.8,
        rerank_score=0.8,
    )

    with patch("shafi.llm.generator.get_settings", return_value=settings):
        generator = RAGGenerator(llm=MagicMock())
        prioritized = generator._prioritize_unique_titles([enactment_notice, law_page])

    assert prioritized[0].chunk_id == "notice:0:0:abc"
    assert prioritized[1].chunk_id == "law:0:0:def"


def test_broad_enumeration_with_multiple_named_titles_keeps_one_chunk_per_title_first() -> None:
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            max_context_tokens=2500,
            simple_model="gpt-4o-mini",
            simple_max_tokens=300,
            fallback_model="gpt-4o-mini",
        ),
        pipeline=SimpleNamespace(
            max_answer_words=250,
            context_token_budget_boolean=900,
            context_token_budget_strict=1100,
            context_token_budget_free_text_simple=1800,
            context_token_budget_free_text_complex=2400,
        ),
    )

    icc_page_1 = RankedChunk(
        chunk_id="icc:0:0:abc",
        doc_id="icc",
        doc_title="INCORPORATED CELL COMPANY (ICC) REGULATIONS",
        doc_type=DocType.REGULATION,
        section_path="page:1",
        text="Companies Law 2018 and Insolvency Law 2009 are both referenced here.",
        retrieval_score=0.9,
        rerank_score=0.9,
    )
    icc_page_2 = RankedChunk(
        chunk_id="icc:1:0:def",
        doc_id="icc",
        doc_title="INCORPORATED CELL COMPANY (ICC) REGULATIONS",
        doc_type=DocType.REGULATION,
        section_path="page:2",
        text="More ICC content.",
        retrieval_score=0.85,
        rerank_score=0.85,
    )
    ic_page_1 = RankedChunk(
        chunk_id="ic:0:0:ghi",
        doc_id="ic",
        doc_title="INVESTMENT COMPANIES (IC) REGULATIONS",
        doc_type=DocType.REGULATION,
        section_path="page:1",
        text="Companies Law 2018 and Insolvency Law 2009 are both referenced here as well.",
        retrieval_score=0.8,
        rerank_score=0.8,
    )

    with patch("shafi.llm.generator.get_settings", return_value=settings):
        generator = RAGGenerator(llm=MagicMock())
        context = generator._format_context(
            question="Which laws explicitly mention the Companies Law 2018 and the Insolvency Law 2009 in their regulations concerning company structures?",
            chunks=[icc_page_1, icc_page_2, ic_page_1],
            complexity=QueryComplexity.COMPLEX,
            answer_type="free_text",
        )

    assert context.index("[icc:0:0:abc]") < context.index("[icc:1:0:def]")
    assert context.index("[ic:0:0:ghi]") < context.index("[icc:1:0:def]")


def test_broad_enumeration_registrar_prioritizes_title_and_admin_clause_per_law() -> None:
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            max_context_tokens=2500,
            simple_model="gpt-4o-mini",
            simple_max_tokens=300,
            fallback_model="gpt-4o-mini",
        ),
        pipeline=SimpleNamespace(
            max_answer_words=250,
            context_token_budget_boolean=900,
            context_token_budget_strict=1100,
            context_token_budget_free_text_simple=1800,
            context_token_budget_free_text_complex=2400,
        ),
    )

    gp_title = RankedChunk(
        chunk_id="gp:title",
        doc_id="gp",
        doc_title="General Partnership Law 2004",
        doc_type=DocType.STATUTE,
        section_path="page:1",
        text="General Partnership Law 2004 made by the Ruler of Dubai.",
        retrieval_score=0.9,
        rerank_score=0.9,
    )
    gp_admin = RankedChunk(
        chunk_id="gp:admin",
        doc_id="gp",
        doc_title="General Partnership Law 2004",
        doc_type=DocType.STATUTE,
        section_path="page:21",
        text="Administration of this Law. The Registrar shall administer this Law and the Regulations made under it.",
        retrieval_score=0.89,
        rerank_score=0.89,
    )
    llp_title = RankedChunk(
        chunk_id="llp:title",
        doc_id="llp",
        doc_title="Limited Liability Partnership Law 2004",
        doc_type=DocType.STATUTE,
        section_path="page:1",
        text="Limited Liability Partnership Law 2004 made by the Ruler of Dubai.",
        retrieval_score=0.88,
        rerank_score=0.88,
    )
    llp_admin = RankedChunk(
        chunk_id="llp:admin",
        doc_id="llp",
        doc_title="Limited Liability Partnership Law 2004",
        doc_type=DocType.STATUTE,
        section_path="page:22",
        text="Administration of this Law. The Registrar shall administer this Law and the Regulations made under it.",
        retrieval_score=0.87,
        rerank_score=0.87,
    )

    with patch("shafi.llm.generator.get_settings", return_value=settings):
        generator = RAGGenerator(llm=MagicMock())
        context = generator._format_context(
            question="Which laws are administered by the Registrar and were enacted in 2004?",
            chunks=[gp_title, llp_title, gp_admin, llp_admin],
            complexity=QueryComplexity.COMPLEX,
            answer_type="free_text",
        )

    assert context.index("[gp:admin]") < context.index("[llp:title]")
    assert context.index("[llp:admin]") < context.index("[gp:title]")


def test_common_elements_context_prioritizes_unique_titles() -> None:
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            max_context_tokens=2500,
            simple_model="gpt-4o-mini",
            simple_max_tokens=300,
            fallback_model="gpt-4o-mini",
        ),
        pipeline=SimpleNamespace(
            max_answer_words=250,
            context_token_budget_boolean=900,
            context_token_budget_strict=1100,
            context_token_budget_free_text_simple=1800,
            context_token_budget_free_text_complex=2400,
        ),
    )

    operating_general = RankedChunk(
        chunk_id="operating:0:0:abc",
        doc_id="operating",
        doc_title="Operating Law 2018",
        doc_type=DocType.STATUTE,
        section_path="page:1",
        text="PART 1: GENERAL. Title and application of the Operating Law 2018.",
        retrieval_score=0.9,
        rerank_score=0.9,
    )
    operating_interpretation = RankedChunk(
        chunk_id="operating:1:0:def",
        doc_id="operating",
        doc_title="Operating Law 2018",
        doc_type=DocType.STATUTE,
        section_path="page:2",
        text="SCHEDULE 1 INTERPRETATION. Rules of interpretation. Schedule 1 contains interpretative provisions and a list of defined terms.",
        retrieval_score=0.85,
        rerank_score=0.85,
    )
    trust_structural = RankedChunk(
        chunk_id="trust:0:0:ghi",
        doc_id="trust",
        doc_title="Trust Law 2018",
        doc_type=DocType.STATUTE,
        section_path="page:1",
        text="Schedule 1 contains interpretative provisions and a list of defined terms.",
        retrieval_score=0.8,
        rerank_score=0.8,
    )

    with patch("shafi.llm.generator.get_settings", return_value=settings):
        generator = RAGGenerator(llm=MagicMock())
        context = generator._format_context(
            question="What are the common elements found in the interpretation sections of the Operating Law 2018 and the Trust Law 2018?",
            chunks=[operating_general, operating_interpretation, trust_structural],
            complexity=QueryComplexity.COMPLEX,
            answer_type="free_text",
        )

    assert context.index("[operating:1:0:def]") < context.index("[operating:0:0:abc]")
    assert context.index("[trust:0:0:ghi]") < context.index("[operating:0:0:abc]")


def test_common_elements_context_merges_yearless_duplicate_titles() -> None:
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            max_context_tokens=2500,
            simple_model="gpt-4o-mini",
            simple_max_tokens=300,
            fallback_model="gpt-4o-mini",
        ),
        pipeline=SimpleNamespace(
            max_answer_words=250,
            context_token_budget_boolean=900,
            context_token_budget_strict=1100,
            context_token_budget_free_text_simple=1800,
            context_token_budget_free_text_complex=2400,
        ),
    )

    operating_intro = RankedChunk(
        chunk_id="operating:intro",
        doc_id="operating-a",
        doc_title="Operating Law 2018",
        doc_type=DocType.STATUTE,
        section_path="page:1",
        text="Schedule 1 contains interpretative provisions and a list of defined terms.",
        retrieval_score=0.9,
        rerank_score=0.9,
    )
    operating_interpretation = RankedChunk(
        chunk_id="operating:interp",
        doc_id="operating-b",
        doc_title="OPERATING LAW",
        doc_type=DocType.STATUTE,
        section_path="page:39",
        text="SCHEDULE 1 INTERPRETATION. Rules of interpretation. A statutory provision includes a reference to the statutory provision as amended or re-enacted from time to time.",
        retrieval_score=0.85,
        rerank_score=0.85,
    )
    operating_continuation = RankedChunk(
        chunk_id="operating:cont",
        doc_id="operating-b",
        doc_title="OPERATING LAW",
        doc_type=DocType.STATUTE,
        section_path="page:40",
        text="Rules of interpretation continue with defined terms and interpretative rules.",
        retrieval_score=0.8,
        rerank_score=0.8,
    )
    trust_interpretation = RankedChunk(
        chunk_id="trust:interp",
        doc_id="trust-a",
        doc_title="Trust Law 2018",
        doc_type=DocType.STATUTE,
        section_path="page:49",
        text="SCHEDULE 1 INTERPRETATION. Rules of interpretation. A statutory provision includes a reference to the statutory provision as amended or re-enacted from time to time.",
        retrieval_score=0.75,
        rerank_score=0.75,
    )

    with patch("shafi.llm.generator.get_settings", return_value=settings):
        generator = RAGGenerator(llm=MagicMock())
        prioritized = generator._prioritize_common_elements_titles(
            "What are the common elements found in the interpretation sections of the Operating Law 2018 and the Trust Law 2018?",
            [operating_intro, operating_interpretation, operating_continuation, trust_interpretation],
        )

    assert [chunk.chunk_id for chunk in prioritized[:3]] == ["operating:interp", "trust:interp", "operating:cont"]
    assert prioritized[-1].chunk_id == "operating:intro"
