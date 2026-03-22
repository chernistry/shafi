from __future__ import annotations

from shafi.core.local_page_reranker import score_pages_from_chunk_scores, select_top_pages_per_doc
from shafi.models import DocType, RankedChunk


def _chunk(
    *,
    chunk_id: str,
    doc_id: str,
    page: int,
    rerank_score: float,
    retrieval_score: float,
    text: str = "text",
) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=doc_id,
        doc_type=DocType.CASE_LAW,
        section_path=f"page:{page}",
        text=text,
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
        doc_summary="",
    )


def test_score_pages_from_chunk_scores_biases_page_one_when_signal_is_close() -> None:
    scored = score_pages_from_chunk_scores(
        chunks=[
            _chunk(chunk_id="doca:body", doc_id="doca", page=5, rerank_score=0.93, retrieval_score=0.94),
            _chunk(chunk_id="doca:title", doc_id="doca", page=1, rerank_score=0.84, retrieval_score=0.82),
        ],
        doc_ids={"doca"},
        page_one_bias=0.18,
        early_page_bias=0.04,
    )

    assert [row.page_id for row in scored] == ["doca_1", "doca_5"]
    assert scored[0].best_chunk_id == "doca:title"


def test_select_top_pages_per_doc_respects_doc_order() -> None:
    scored = score_pages_from_chunk_scores(
        chunks=[
            _chunk(chunk_id="doca:title", doc_id="doca", page=1, rerank_score=0.82, retrieval_score=0.80),
            _chunk(chunk_id="docb:title", doc_id="docb", page=1, rerank_score=0.81, retrieval_score=0.79),
            _chunk(chunk_id="doca:body", doc_id="doca", page=4, rerank_score=0.91, retrieval_score=0.92),
            _chunk(chunk_id="docb:body", doc_id="docb", page=6, rerank_score=0.88, retrieval_score=0.90),
        ],
        doc_ids={"doca", "docb"},
        page_one_bias=0.18,
        early_page_bias=0.04,
    )

    selected = select_top_pages_per_doc(scored_pages=scored, doc_order=["docb", "doca"], per_doc_pages=1)

    assert [row.page_id for row in selected] == ["docb_1", "doca_1"]
