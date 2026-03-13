from __future__ import annotations

from scripts.probe_local_embedding_relevance import (
    CandidatePage,
    _cosine_similarity,
    _load_scaffold_cases,
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


def test_load_scaffold_cases_filters_by_verdict_failure_class_and_pages(tmp_path) -> None:
    scaffold = tmp_path / "scaffold.json"
    scaffold.write_text(
        """
        {
          "records": [
            {
              "question_id": "q1",
              "manual_verdict": "correct",
              "failure_class": "support_undercoverage",
              "minimal_required_support_pages": ["doca_1"]
            },
            {
              "question_id": "q2",
              "manual_verdict": "incorrect",
              "failure_class": "weak_path_fallback",
              "minimal_required_support_pages": ["docb_2"]
            },
            {
              "question_id": "q3",
              "manual_verdict": "correct",
              "failure_class": "support_undercoverage",
              "minimal_required_support_pages": []
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    cases = _load_scaffold_cases(
        scaffold,
        question_ids=None,
        manual_verdicts={"correct"},
        failure_classes={"support_undercoverage"},
        max_cases=None,
    )

    assert cases == [
        {
            "question_id": "q1",
            "gold_page_ids": ["doca_1"],
            "source_manual_verdict": "correct",
            "source_failure_class": "support_undercoverage",
        }
    ]
