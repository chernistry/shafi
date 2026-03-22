from shafi.core.conflict_detector import ConflictDetector
from shafi.core.decomposer import QueryDecomposer
from shafi.core.strict_answerer import StrictAnswerer
from shafi.models import DocType, QueryComplexity, RankedChunk


def _chunk(
    *,
    chunk_id: str,
    text: str,
    doc_title: str = "CFI 010/2024 Example v Example",
    doc_type: DocType = DocType.STATUTE,
) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        doc_title=doc_title,
        doc_type=doc_type,
        section_path="Article 10",
        text=text,
        retrieval_score=0.8,
        rerank_score=0.7,
    )


def test_query_decomposer_detects_multi_hop_complex_query() -> None:
    decomposer = QueryDecomposer()
    query = "Compare Law No. 2 of 2019 and Law No. 10 of 2018, and explain the difference in limitation period."
    assert decomposer.should_decompose(query, QueryComplexity.COMPLEX) is True
    parts = decomposer.decompose(query, max_subqueries=3)
    assert 1 <= len(parts) <= 3


def test_conflict_detector_finds_obligation_conflict() -> None:
    detector = ConflictDetector()
    chunks = [
        _chunk(chunk_id="c1", text="Article 10 says parties shall disclose material facts."),
        _chunk(chunk_id="c2", text="Article 10 says parties shall not disclose material facts."),
    ]
    report = detector.detect(chunks)
    assert report.has_conflict is True
    assert len(report.conflicts) >= 1
    assert "Conflict advisory" in report.to_prompt_context()


def test_strict_answerer_extracts_number_and_returns_evidence_ids() -> None:
    answerer = StrictAnswerer()
    chunks = [_chunk(chunk_id="c1", text="The claim value is 250000 AED.")]
    result = answerer.answer(
        answer_type="number",
        query="What was the claim value?",
        context_chunks=chunks,
        max_chunks=2,
    )
    assert result is not None
    assert result.answer == "250000"
    assert result.cited_chunk_ids == ["c1"]


def test_strict_answerer_normalizes_day_first_textual_date_to_iso() -> None:
    answerer = StrictAnswerer()
    chunks = [_chunk(chunk_id="c1", text="This Law comes into force on 11 March 2025.")]

    result = answerer.answer(
        answer_type="date",
        query="On what date does this Law come into force?",
        context_chunks=chunks,
        max_chunks=2,
    )

    assert result is not None
    assert result.answer == "2025-03-11"
    assert result.cited_chunk_ids == ["c1"]


def test_strict_answerer_normalizes_month_first_textual_date_to_iso() -> None:
    answerer = StrictAnswerer()
    chunks = [_chunk(chunk_id="c1", text="The enactment notice was issued on March 11, 2025.")]

    result = answerer.answer(
        answer_type="date",
        query="What is the date of the enactment notice?",
        context_chunks=chunks,
        max_chunks=2,
    )

    assert result is not None
    assert result.answer == "2025-03-11"
    assert result.cited_chunk_ids == ["c1"]


def test_strict_answerer_compares_same_year_for_named_law_titles() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="employment:cover",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="EMPLOYMENT LAW DIFC LAW NO. 2 of 2019 Consolidated Version No. 5 (July 2025)",
            retrieval_score=0.9,
            rerank_score=0.9,
        ),
        RankedChunk(
            chunk_id="ip:general",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="This Law may be cited as the Intellectual Property Law 2019. This Law is enacted on the date specified in the Enactment Notice in respect of this Law.",
            retrieval_score=0.88,
            rerank_score=0.88,
        ),
    ]

    result = answerer.answer(
        answer_type="boolean",
        query="Was the Employment Law enacted in the same year as the Intellectual Property Law?",
        context_chunks=chunks,
        max_chunks=4,
    )

    assert result is not None
    assert result.answer == "Yes"
    assert result.cited_chunk_ids == ["employment:cover", "ip:general"]


def test_strict_answerer_same_year_prefers_current_title_year_over_repealed_reference() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="employment:title",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="This Employment Law 2019 repeals and replaces the Employment Law 2005 (DIFC Law No. 4 of 2005).",
            retrieval_score=0.9,
            rerank_score=0.9,
        ),
        RankedChunk(
            chunk_id="ip:title",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="This Law may be cited as the Intellectual Property Law 2019.",
            retrieval_score=0.88,
            rerank_score=0.88,
        ),
    ]

    result = answerer.answer(
        answer_type="boolean",
        query="Was the Employment Law enacted in the same year as the Intellectual Property Law?",
        context_chunks=chunks,
        max_chunks=4,
    )

    assert result is not None
    assert result.answer == "Yes"
    assert result.cited_chunk_ids == ["employment:title", "ip:title"]


def test_strict_answerer_restriction_effectiveness_ignores_same_doc_distractor() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="pp:distractor",
            doc_id="pp-law",
            doc_title="PERSONAL PROPERTY LAW",
            doc_type=DocType.STATUTE,
            section_path="page:9",
            text=(
                "Article 22. A restriction on transfer of a security may be noted in the register. "
                "This provision does not address actual knowledge or whether the restriction is effective."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
        ),
        RankedChunk(
            chunk_id="pp:article23",
            doc_id="pp-law",
            doc_title="PERSONAL PROPERTY LAW",
            doc_type=DocType.STATUTE,
            section_path="page:10",
            text=(
                "Article 23. A restriction on transfer of a security is ineffective against any person "
                "other than a person who had actual knowledge of the restriction. "
                "If the security is uncertificated and the registered owner has been notified of the restriction, "
                "the restriction remains effective against a person with actual knowledge."
            ),
            retrieval_score=0.88,
            rerank_score=0.88,
        ),
    ]

    result = answerer.answer(
        answer_type="boolean",
        query=(
            "Under Article 23 of the Personal Property Law 2005, is a restriction on transfer of a security "
            "imposed by the issuer effective against a person who had actual knowledge of such third party "
            "property interest, if the security is uncertificated and the registered owner has been notified "
            "of the restriction?"
        ),
        context_chunks=chunks,
        max_chunks=4,
    )

    assert result is not None
    assert result.answer == "Yes"
    assert result.cited_chunk_ids == ["pp:article23"]


def test_strict_answerer_answers_no_for_trust_article_28_good_faith_purchaser_clause() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="trust:article28",
            doc_id="trust-law",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:15",
            text=(
                "Applications and orders under Articles 24 to 27. "
                "No order may be made under Article 28(3) which would prejudice any purchaser in good faith "
                "for value of any trust property without notice of the matters which render the transfer voidable."
            ),
            retrieval_score=0.93,
            rerank_score=0.93,
        )
    ]

    result = answerer.answer(
        answer_type="boolean",
        query=(
            "According to Article 28(4) of the DIFC Trust Law 2018, can an order made consequential to a "
            "declaration under Articles 24 to 27 prejudice a purchaser in good faith for value of trust property "
            "without notice of the voidable matters?"
        ),
        context_chunks=chunks,
        max_chunks=4,
    )

    assert result is not None
    assert result.answer == "No"
    assert result.cited_chunk_ids == ["trust:article28"]


def test_strict_answerer_answers_no_for_employing_child_under_sixteen() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="employment:article13",
            doc_id="employment-law",
            doc_title="EMPLOYMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:7",
            text=(
                "Article 13. Hiring children. "
                "An Employer shall not employ a child who is under sixteen (16) years of age."
            ),
            retrieval_score=0.91,
            rerank_score=0.91,
        )
    ]

    result = answerer.answer(
        answer_type="boolean",
        query="According to Article 13 of the Employment Law 2019, can an Employer employ a child who is under sixteen years of age?",
        context_chunks=chunks,
        max_chunks=4,
    )

    assert result is not None
    assert result.answer == "No"
    assert result.cited_chunk_ids == ["employment:article13"]


def test_strict_answerer_handles_presided_over_both_judge_compare_phrase() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="arb:page1",
            doc_id="arb-034",
            doc_title="ARB 034/2025 Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="ORDER WITH REASONS OF H.E. JUSTICE SHAMLAN AL SAWALEHI.",
            retrieval_score=0.95,
            rerank_score=0.95,
        ),
        RankedChunk(
            chunk_id="cfi:page1",
            doc_id="cfi-067",
            doc_title="CFI 067/2025 Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="ORDER WITH REASONS OF H.E. JUSTICE MICHAEL BLACK KC.",
            retrieval_score=0.94,
            rerank_score=0.94,
        ),
    ]

    result = answerer.answer(
        answer_type="boolean",
        query="Is there a judge who presided over both case ARB 034/2025 and case CFI 067/2025?",
        context_chunks=chunks,
        max_chunks=4,
    )

    assert result is not None
    assert result.answer == "No"
    assert result.cited_chunk_ids == ["arb:page1", "cfi:page1"]
