from __future__ import annotations

from scripts.build_embedding_doc_family_candidate import PageScore, _group_retrieved_chunk_pages, _select_pages


def test_select_pages_picks_per_doc_then_extra_global() -> None:
    scored_pages = [
        PageScore(page_id="doca_2", score=0.9),
        PageScore(page_id="doca_1", score=0.8),
        PageScore(page_id="docb_4", score=0.95),
        PageScore(page_id="docb_1", score=0.7),
    ]
    selected = _select_pages(
        scored_pages=scored_pages,
        per_doc_pages=1,
        extra_global_pages=1,
    )
    assert selected == ["doca_2", "docb_4", "doca_1"]


def test_group_retrieved_chunk_pages_groups_sorted_pages() -> None:
    grouped = _group_retrieved_chunk_pages(["b_3", "a_2", "a_1", "b_3"])
    assert grouped == [
        {"doc_id": "a", "page_numbers": [1, 2]},
        {"doc_id": "b", "page_numbers": [3]},
    ]
