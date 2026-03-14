from __future__ import annotations

from rag_challenge.core.strict_answerer import StrictAnswerer
from rag_challenge.models import DocType, RankedChunk


def _case_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    doc_title: str,
    page: int,
    text: str,
) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=doc_title,
        doc_type=DocType.CASE_LAW,
        section_path=f"page:{page}",
        text=text,
        retrieval_score=0.9,
        rerank_score=0.9,
        doc_summary="",
    )


def test_normalize_name_preserves_dotted_corporate_suffixes() -> None:
    assert StrictAnswerer._normalize_name("Architeriors Interior Design (L.L.C).") == "Architeriors Interior Design (L.L.C)"
    assert StrictAnswerer._normalize_name("Coinmena B.S.C. (C)") == "Coinmena B.S.C. (C)"
    assert StrictAnswerer._normalize_name("Union Properties P.J.S.C.") == "Union Properties P.J.S.C"


def test_answer_name_returns_case_ref_for_issue_date_comparison() -> None:
    answerer = StrictAnswerer()
    chunks = [
        _case_chunk(
            chunk_id="cfi010:0",
            doc_id="cfi010",
            doc_title="CFI 010/2024 Fursa Consulting v Bay Gate Investment LLC",
            page=1,
            text="CFI 010/2024\nDate of Issue: 23 January 2026\nJudgment text follows.",
        ),
        _case_chunk(
            chunk_id="cfi016:0",
            doc_id="cfi016",
            doc_title="CFI 016/2025 Obadiah",
            page=1,
            text="CFI 016/2025\nDate of Issue: 16 February 2026\nJudgment text follows.",
        ),
    ]

    result = answerer.answer(
        answer_type="name",
        query="Which case has an earlier Date of Issue: CFI 010/2024 or CFI 016/2025?",
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "CFI 010/2024"
    assert result.cited_chunk_ids == ["cfi010:0", "cfi016:0"]


def test_answer_name_returns_case_ref_for_higher_monetary_claim() -> None:
    answerer = StrictAnswerer()
    chunks = [
        _case_chunk(
            chunk_id="sct169:0",
            doc_id="sct169",
            doc_title="SCT 169/2025 Case Title",
            page=1,
            text=(
                "SCT 169/2025\nDate of Issue: 24 December 2025\n"
                "The Claimant seeks payment in the amount of AED 391,123.45."
            ),
        ),
        _case_chunk(
            chunk_id="sct295:0",
            doc_id="sct295",
            doc_title="SCT 295/2025 Olexa v Odon",
            page=1,
            text=(
                "SCT 295/2025\nDate of Issue: 10 December 2025\n"
                "The Claimant's outstanding claim was for four months of his basic salary at AED 165,000. "
                "A penalty of AED 300,162.86 was also discussed."
            ),
        ),
    ]

    result = answerer.answer(
        answer_type="name",
        query="Identify the case with the higher monetary claim: SCT 169/2025 or SCT 295/2025.",
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "SCT 169/2025"
    assert result.cited_chunk_ids == ["sct169:0", "sct295:0"]


def test_answer_name_extracts_origin_claim_number_from_page_two_anchor() -> None:
    answerer = StrictAnswerer()
    chunks = [
        _case_chunk(
            chunk_id="ca009:1",
            doc_id="ca009",
            doc_title="CA 009/2024 Oskar v Oron",
            page=1,
            text=(
                "Claim No. CA 009/2024\n"
                "This appeal concerns earlier proceedings in Claim No. ENF 316/2023."
            ),
        ),
        _case_chunk(
            chunk_id="ca009:2",
            doc_id="ca009",
            doc_title="CA 009/2024 Oskar v Oron",
            page=2,
            text=(
                "UPON the hearing of the Appellant's appeal against the Order dated 17 May 2024, "
                "granting the relief sought by the Respondents in their Urgent Application of 1 April 2024 "
                'in Claim No. ENF-316-2023/2, (the "Application").'
            ),
        ),
    ]

    result = answerer.answer(
        answer_type="name",
        query="According to page 2 of the judgment, from which specific claim number did the appeal in CA 009/2024 originate?",
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "ENF-316-2023/2"
    assert result.cited_chunk_ids == ["ca009:2"]


def test_answer_boolean_detects_common_judge_across_multiple_case_documents() -> None:
    answerer = StrictAnswerer()
    chunks = [
        _case_chunk(
            chunk_id="ca005:orders",
            doc_id="ca005-orders",
            doc_title="CA 005/2025 LXT Real Estate Broker L.L.C v SIR Real Estate LLC",
            page=1,
            text=(
                "Claim No: CA 005/2025\n"
                "hearing held before H.E. Chief Justice Wayne Martin, "
                "H.E. Justice Rene Le Miere and H.E. Justice Sir Peter Gross."
            ),
        ),
        _case_chunk(
            chunk_id="tcd001:first-instance",
            doc_id="tcd001-first",
            doc_title="TCD 001/2024 Architeriors Interior Design v Emirates National Investment Co",
            page=1,
            text=(
                "Claim No: TCD 001/2024\n"
                "ORDER WITH REASONS OF H.E. JUSTICE ROGER STEWART."
            ),
        ),
        _case_chunk(
            chunk_id="tcd001:appeal",
            doc_id="tcd001-appeal",
            doc_title="TCD 001/2024 Architeriors Interior Design v Emirates National Investment Co",
            page=1,
            text=(
                "Claim No: TCD 001/2024\n"
                "ORDER WITH REASONS OF H.E. CHIEF JUSTICE WAYNE MARTIN."
            ),
        ),
    ]

    result = answerer.answer(
        answer_type="boolean",
        query="Considering all documents across case CA 005/2025 and case TCD 001/2024, was there any judge who participated in both cases?",
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "Yes"
    assert result.cited_chunk_ids == ["ca005:orders", "tcd001:appeal"]


def test_answer_boolean_detects_party_overlap_from_first_page_caption_text() -> None:
    answerer = StrictAnswerer()
    chunks = [
        _case_chunk(
            chunk_id="ca004:0",
            doc_id="ca004",
            doc_title="CA 004/2025",
            page=1,
            text=(
                "Claim No: CA 004/2025\n"
                "BETWEEN\n"
                "ALPHA HOLDINGS LTD\n"
                "Claimant/Appellant\n"
                "and\n"
                "BETA LLC\n"
                "Defendant/Respondent\n"
            ),
        ),
        _case_chunk(
            chunk_id="sct295:0",
            doc_id="sct295",
            doc_title="SCT 295/2025",
            page=1,
            text=(
                "Claim No: SCT 295/2025\n"
                "BETWEEN\n"
                "OLEXA\n"
                "Claimant/Applicant\n"
                "and\n"
                "ALPHA HOLDINGS LTD\n"
                "Defendant/Respondent\n"
            ),
        ),
    ]

    result = answerer.answer(
        answer_type="boolean",
        query="Do cases CA 004/2025 and SCT 295/2025 involve any of the same legal entities or individuals as parties?",
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "Yes"
    assert result.cited_chunk_ids == ["ca004:0", "sct295:0"]


def test_answer_boolean_handles_operating_law_bad_faith_carve_out_across_chunks() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="op:1",
            doc_id="op",
            doc_title="OPERATING LAW",
            doc_type=DocType.STATUTE,
            section_path="page:7",
            text=(
                "Subject to Article 7(8), neither the Registrar nor any delegate or agent of the Registrar can be held liable "
                "for anything done or omitted to be done in the performance of the functions of the Registrar."
            ),
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="op:2",
            doc_id="op",
            doc_title="OPERATING LAW",
            doc_type=DocType.STATUTE,
            section_path="page:7",
            text="Article 7(7) does not apply if the act or omission is shown to have been in bad faith.",
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="",
        ),
    ]

    result = answerer.answer(
        answer_type="boolean",
        query="Under the Operating Law 2018, can the Registrar be held liable for acts or omissions in performing their functions if the act or omission is shown to have been in bad faith, according to Article 7(8)?",
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "Yes"
    assert result.cited_chunk_ids == ["op:1", "op:2"]


def test_answer_boolean_handles_trust_law_governing_law_validity_clause() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="trust:1",
            doc_id="trust",
            doc_title="TRUST LAW 2018",
            doc_type=DocType.STATUTE,
            section_path="page:6",
            text=(
                "A term of the trust expressly declaring that the laws of the DIFC shall govern the trust "
                "is valid, effective and conclusive regardless of any other circumstance."
            ),
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        )
    ]

    result = answerer.answer(
        answer_type="boolean",
        query="Under Article 11(5) of the DIFC Trust Law 2018, is a term of the trust expressly declaring that the laws of the DIFC shall govern the trust valid, effective, and conclusive regardless of any other circumstance?",
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "Yes"
    assert result.cited_chunk_ids == ["trust:1"]


def test_answer_boolean_detects_party_overlap_phrase_variant() -> None:
    answerer = StrictAnswerer()
    chunks = [
        _case_chunk(
            chunk_id="cfi057:title",
            doc_id="cfi057",
            doc_title="CFI 057/2025 Clyde & Co LLP v Union Properties PJSC",
            page=1,
            text="Claim No: CFI 057/2025",
        ),
        _case_chunk(
            chunk_id="sct295:title",
            doc_id="sct295",
            doc_title="SCT 295/2025 Olexa v Odon",
            page=1,
            text="Claim No: SCT 295/2025",
        ),
    ]

    result = answerer.answer(
        answer_type="boolean",
        query="Is there any main party that appeared in both cases CFI 057/2025 and SCT 295/2025?",
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "No"
    assert result.cited_chunk_ids == ["cfi057:title", "sct295:title"]


def test_answer_boolean_handles_operating_law_article_8_1() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="operating:8-1",
            doc_id="operating",
            doc_title="OPERATING LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                "Article 8. Conducting Business in the DIFC. "
                "No person shall operate or conduct business in or from the DIFC unless incorporated, "
                "registered or continued under a Prescribed Law."
            ),
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        )
    ]

    result = answerer.answer(
        answer_type="boolean",
        query=(
            "Under Article 8(1) of the Operating Law 2018, is a person permitted to operate or conduct "
            "business in or from the DIFC without being incorporated, registered, or continued under a "
            "Prescribed Law or other Legislation administered by the Registrar?"
        ),
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "No"
    assert result.cited_chunk_ids == ["operating:8-1"]


def test_answer_boolean_handles_general_partnership_article_11_body_corporate() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="gp:11",
            doc_id="gp",
            doc_title="GENERAL PARTNERSHIP LAW",
            doc_type=DocType.STATUTE,
            section_path="page:5",
            text=(
                "An unincorporated body of persons carrying on a business for profit is deemed to be a "
                "General Partnership unless the agreement otherwise provides, including that it is a body corporate."
            ),
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        )
    ]

    result = answerer.answer(
        answer_type="boolean",
        query=(
            "Under Article 11 of the General Partnership Law 2004, is an unincorporated body of persons "
            "carrying on business for profit automatically deemed a partnership if there is an agreement "
            "specifying it as a body corporate?"
        ),
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "No"
    assert result.cited_chunk_ids == ["gp:11"]


def test_answer_boolean_handles_employment_article_11_2_b() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="employment:11-2b",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:7",
            text=(
                "Nothing in this Article shall prevent an Employee from waiving any right under this Law in a "
                "written agreement with the Employer to terminate the Employee's employment or to resolve a dispute, "
                "provided the Employee is given the opportunity to receive independent legal advice or otherwise "
                "takes part in mediation proceedings."
            ),
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        )
    ]

    result = answerer.answer(
        answer_type="boolean",
        query=(
            "Under Article 11(2)(b) of the Employment Law 2019, can an Employee waive any right under "
            "this Law by entering into a written agreement with their Employer to terminate employment, "
            "provided they were given an opportunity to receive independent legal advice or took part in mediation?"
        ),
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "Yes"
    assert result.cited_chunk_ids == ["employment:11-2b"]


def test_answer_boolean_handles_claimant_application_not_granted_when_defendant_set_aside_is_granted() -> None:
    answerer = StrictAnswerer()
    chunks = [
        _case_chunk(
            chunk_id="arb034:2",
            doc_id="arb034",
            doc_title="ARB 034/2025 Ohtli v Onora",
            page=2,
            text=(
                "IT IS HEREBY ORDERED THAT: "
                "1. The ASI Order is discharged with immediate effect. "
                "2. The Defendant's Set Aside Application is granted. "
                "3. The Claimant shall pay the Defendant its costs of the Set Aside Application on the standard basis."
            ),
        ),
        _case_chunk(
            chunk_id="arb034:4",
            doc_id="arb034",
            doc_title="ARB 034/2025 Ohtli v Onora",
            page=4,
            text=(
                "I am not persuaded that such concern justifies the continuation of an injunction where the "
                "conduct complained of has ceased and the dispute is progressing through arbitration."
            ),
        ),
    ]

    result = answerer.answer(
        answer_type="boolean",
        query=(
            "Based on the second page of the document, was the Claimant's application for continuation "
            "of the ASI Order granted by the court in case ARB 034/2025?"
        ),
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "No"
    assert result.cited_chunk_ids == ["arb034:2"]


def test_answer_boolean_handles_crs_article_17_b() -> None:
    answerer = StrictAnswerer()
    chunks = [
        RankedChunk(
            chunk_id="crs:17b",
            doc_id="crs",
            doc_title="COMMON REPORTING STANDARD LAW",
            doc_type=DocType.STATUTE,
            section_path="page:8",
            text=(
                "A person commits an offence if the person obstructs an Inspector in the performance of the "
                "Inspector's powers and functions, including a failure to give or produce information or documents "
                "specified by an Inspector."
            ),
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        )
    ]

    result = answerer.answer(
        answer_type="boolean",
        query=(
            "If a Reporting Financial Institution fails to provide specified information during an investigation, "
            "does this constitute obstruction of Inspectors under Article 17(b) of the Common Reporting Standard Law 2018?"
        ),
        context_chunks=chunks,
    )

    assert result is not None
    assert result.answer == "Yes"
    assert result.cited_chunk_ids == ["crs:17b"]
