from __future__ import annotations

import fitz
from scripts.build_local_page_reranker_candidate import _candidate_pages_for_doc, _select_pages

from rag_challenge.core.local_page_reranker import PageRerankScore


def test_candidate_pages_for_doc_keeps_baseline_neighbors_and_anchors(tmp_path) -> None:
    pdf_path = tmp_path / "doca.pdf"
    doc = fitz.open()
    try:
        for _ in range(12):
            doc.new_page()
        doc.save(pdf_path)
    finally:
        doc.close()
    cache: dict[str, int] = {}
    page_ids = _candidate_pages_for_doc(
        doc_id="doca",
        baseline_pages=[5, 9],
        dataset_dir=tmp_path,
        page_count_cache=cache,
        include_page_one=True,
        include_page_two=True,
        include_last_page=True,
        neighbor_radius=1,
        max_pages_per_doc=8,
    )
    assert page_ids == ["doca_5", "doca_9", "doca_4", "doca_6", "doca_8", "doca_10", "doca_1", "doca_2"]


def test_select_pages_picks_top_page_per_doc_then_extra() -> None:
    selected = _select_pages(
        scored_pages=[
            PageRerankScore(page_id="doca_5", score=0.5),
            PageRerankScore(page_id="doca_2", score=0.7),
            PageRerankScore(page_id="docb_1", score=0.8),
            PageRerankScore(page_id="docb_3", score=0.6),
        ],
        per_doc_pages=1,
        extra_global_pages=1,
    )
    assert selected == ["doca_2", "docb_1", "docb_3"]
