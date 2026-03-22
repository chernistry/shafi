from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_search_within_doc_rerank_subsets_finds_best_subset(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    baseline_raw_results = tmp_path / "baseline_raw_results.json"
    scaffold = tmp_path / "scaffold.json"
    benchmark = tmp_path / "benchmark.json"
    preflight = tmp_path / "preflight.json"
    seed_qids = tmp_path / "seed_qids.txt"
    qids = tmp_path / "qids.txt"
    out_dir = tmp_path / "out"

    baseline_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {"question_id": "q1", "answer": "A", "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc1", "page_numbers": [2]}]}}},
                    {"question_id": "q2", "answer": "B", "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc2", "page_numbers": [4]}]}}},
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_raw_results.write_text(
        json.dumps(
            [
                {"case": {"case_id": "q1"}, "answer_text": "A", "telemetry": {"used_page_ids": ["doc1_2"], "context_page_ids": ["doc1_2"], "retrieved_page_ids": ["doc1_2"]}},
                {"case": {"case_id": "q2"}, "answer_text": "B", "telemetry": {"used_page_ids": ["doc2_4"], "context_page_ids": ["doc2_4"], "retrieved_page_ids": ["doc2_4"]}},
            ]
        ),
        encoding="utf-8",
    )
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {"question_id": "q1", "manual_verdict": "correct", "failure_class": "support_undercoverage", "minimal_required_support_pages": ["doc1_1"]},
                    {"question_id": "q2", "manual_verdict": "correct", "failure_class": "support_undercoverage", "minimal_required_support_pages": ["doc2_4"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {"question_id": "q1", "gold_page_ids": ["doc1_1"], "gold_items": [], "items": []},
                    {"question_id": "q2", "gold_page_ids": ["doc2_4"], "gold_items": [], "items": []},
                ]
            }
        ),
        encoding="utf-8",
    )
    preflight.write_text(json.dumps({"page_count_distribution": {"p95": 2}}), encoding="utf-8")
    seed_qids.write_text("q1\nq2\n", encoding="utf-8")
    qids.write_text("q1\nq2\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/search_within_doc_rerank_subsets.py",
            "--baseline-submission",
            str(baseline_submission),
            "--baseline-raw-results",
            str(baseline_raw_results),
            "--scaffold",
            str(scaffold),
            "--benchmark",
            str(benchmark),
            "--baseline-preflight",
            str(preflight),
            "--seed-qids-file",
            str(seed_qids),
            "--qids-file",
            str(qids),
            "--out-dir",
            str(out_dir),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads((out_dir / "subset_search.json").read_text(encoding="utf-8"))
    results = payload["results"]
    assert results[0]["qids"] == ["q1"]
    assert results[0]["benchmark_trusted_candidate"] >= results[-1]["benchmark_trusted_candidate"]
