from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import fitz
from scripts.build_colbert_page_reranker_candidate import _build_candidate

from rag_challenge.core.local_page_reranker import PageRerankScore

if TYPE_CHECKING:
    from pathlib import Path


class _FakeLateInteractionReranker:
    def __init__(self, *, model_name: str, max_chars: int, max_query_chars: int) -> None:
        self.model_name = model_name
        self.max_chars = max_chars
        self.max_query_chars = max_query_chars

    def score_pages(self, *, query: str, pages: list[tuple[str, str]]) -> list[PageRerankScore]:
        assert query == "Who is the claimant?"
        page_ids = [page_id for page_id, _ in pages]
        assert "doca_1" in page_ids
        return [
            PageRerankScore(page_id="doca_1", score=0.9),
            PageRerankScore(page_id="doca_2", score=0.4),
        ]


def _write_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    try:
        for text in pages:
            page = doc.new_page()
            page.insert_text((72, 72), text)
        doc.save(path)
    finally:
        doc.close()


def test_build_candidate_patches_selected_pages(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "docs"
    dataset_dir.mkdir()
    _write_pdf(dataset_dir / "doca.pdf", ["Claimant Alice", "Body page two"])

    baseline_submission = tmp_path / "submission.json"
    baseline_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "qid-1",
                        "answer": "Alice",
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doca", "page_numbers": [2]}]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_raw_results = tmp_path / "raw_results.json"
    baseline_raw_results.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "qid-1", "question": "Who is the claimant?"},
                    "telemetry": {
                        "question_id": "qid-1",
                        "retrieved_page_ids": ["doca_2"],
                        "context_page_ids": ["doca_2"],
                        "cited_page_ids": ["doca_2"],
                        "used_page_ids": ["doca_2"],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    baseline_preflight = tmp_path / "preflight.json"
    baseline_preflight.write_text(json.dumps({"question_count": 1}), encoding="utf-8")

    monkeypatch.setattr(
        "scripts.build_colbert_page_reranker_candidate.LocalLateInteractionReranker",
        _FakeLateInteractionReranker,
    )

    args = SimpleNamespace(
        baseline_submission=baseline_submission,
        baseline_raw_results=baseline_raw_results,
        baseline_preflight=baseline_preflight,
        dataset_documents=dataset_dir,
        label="v10_local_colbert_page_rerank_r1",
        model="answerdotai/answerai-colbert-small-v1",
        max_chars=4000,
        max_query_chars=1200,
        qid=["qid-1"],
        qids_file=None,
        page_source="retrieved",
        include_page_one=True,
        include_page_two=True,
        include_last_page=False,
        neighbor_radius=1,
        max_pages_per_doc=4,
        per_doc_pages=1,
        extra_global_pages=0,
        report_top_k=3,
    )

    selections, submission_payload, raw_results_payload, preflight = _build_candidate(args)

    assert selections[0].selected_page_ids == ["doca_1"]
    assert submission_payload["answers"][0]["telemetry"]["retrieval"]["retrieved_chunk_pages"] == [
        {"doc_id": "doca", "page_numbers": [1]}
    ]
    assert raw_results_payload[0]["telemetry"]["used_page_ids"] == ["doca_1"]
    projection = preflight["counterfactual_projection"]
    assert projection["page_source_policy"]["kind"] == "local_late_interaction_reranker"
    assert projection["page_source_policy"]["model"] == "answerdotai/answerai-colbert-small-v1"
