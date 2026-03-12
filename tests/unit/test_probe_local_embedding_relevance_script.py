from __future__ import annotations

from scripts.probe_local_embedding_relevance import (
    CandidatePage,
    _cosine_similarity,
    _parse_page_id,
    _pick_distractor_page_ids,
    _same_doc_neighbor_page_ids,
    _score_case,
)


def test_parse_page_id_extracts_doc_and_page() -> None:
    assert _parse_page_id("abc123_4") == ("abc123", 4)


def test_pick_distractor_page_ids_excludes_gold_and_deduplicates() -> None:
    raw_result = {
        "telemetry": {
            "used_page_ids": ["gold_1", "other_2", "other_2"],
            "context_page_ids": ["other_3", "gold_1"],
            "retrieved_page_ids": ["other_4"],
        }
    }

    distractors = _pick_distractor_page_ids(
        gold_page_ids=["gold_1"],
        raw_result=raw_result,
        max_distractors=3,
    )

    assert distractors == ["other_2", "other_3", "other_4"]


def test_cosine_similarity_matches_expected_vector_ordering() -> None:
    score = _cosine_similarity([1.0, 0.0], [0.5, 0.0])
    assert score == 1.0


def test_score_case_flags_gold_top1_when_best_page_is_gold() -> None:
    candidates = [
        CandidatePage(page_id="gold_1", text="gold text", is_gold=True),
        CandidatePage(page_id="other_2", text="other text", is_gold=False),
    ]

    result = _score_case(
        question_id="qid-1",
        question="test question",
        candidates=candidates,
        query_embedding=[1.0, 0.0],
        page_embeddings=[[0.9, 0.0], [0.1, 0.9]],
    )

    assert result.gold_top1 is True
    assert result.best_gold_rank == 1
    assert result.top_page_id == "gold_1"
    assert result.gold_margin > 0.0


def test_same_doc_neighbor_page_ids_adds_adjacent_pages(tmp_path) -> None:
    pdf_path = tmp_path / "doc123.pdf"
    import fitz

    doc = fitz.open()
    try:
        for index in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"page {index + 1}")
        doc.save(str(pdf_path))
    finally:
        doc.close()

    neighbors = _same_doc_neighbor_page_ids(
        gold_page_ids=["doc123_2"],
        dataset_dir=tmp_path,
        page_count_cache={},
    )

    assert neighbors == ["doc123_1", "doc123_3"]
