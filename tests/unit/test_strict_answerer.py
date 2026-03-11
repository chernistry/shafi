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
