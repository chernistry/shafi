from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.mine_embedding_support_opportunities import (
    SupportOpportunity,
    _opportunity_sort_key,
    _raw_results_current_pages,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_raw_results_current_pages_merges_unique_page_ids_in_priority_order(tmp_path: Path) -> None:
    raw_results = tmp_path / "raw_results.json"
    raw_results.write_text(
        json.dumps(
            [
                {
                    "telemetry": {
                        "question_id": "qid-1",
                        "used_page_ids": ["doc_2", "doc_1"],
                        "context_page_ids": ["doc_1", "doc_3"],
                        "retrieved_page_ids": ["doc_3", "doc_4"],
                    }
                }
            ]
        ),
        encoding="utf-8",
    )

    pages = _raw_results_current_pages(raw_results)
    assert pages["qid-1"] == ["doc_2", "doc_1", "doc_3", "doc_4"]


def test_opportunity_sort_key_prefers_new_gold_gain_then_top1_then_margin() -> None:
    weaker = SupportOpportunity(
        question_id="a",
        question="Q",
        failure_class="support_undercoverage",
        manual_verdict="correct",
        gold_page_ids=["doc_2"],
        current_page_ids=["doc_1"],
        selected_page_ids=["doc_2"],
        gold_top1=False,
        gold_top3=True,
        current_has_gold=False,
        selected_has_gold=True,
        new_gold_gain=True,
        best_gold_rank=2,
        top_page_id="doc_1",
        gold_margin=0.1,
        scored_pages=[],
    )
    stronger = SupportOpportunity(
        question_id="b",
        question="Q",
        failure_class="support_undercoverage",
        manual_verdict="correct",
        gold_page_ids=["doc_1"],
        current_page_ids=["doc_2"],
        selected_page_ids=["doc_1"],
        gold_top1=True,
        gold_top3=True,
        current_has_gold=False,
        selected_has_gold=True,
        new_gold_gain=True,
        best_gold_rank=1,
        top_page_id="doc_1",
        gold_margin=0.2,
        scored_pages=[],
    )

    assert _opportunity_sort_key(stronger) > _opportunity_sort_key(weaker)
