from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_build_scaffold_support_surrogate_patches_only_selected_qids(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    baseline_raw_results = tmp_path / "baseline_raw_results.json"
    scaffold = tmp_path / "scaffold.json"
    out_submission = tmp_path / "out_submission.json"
    out_raw_results = tmp_path / "out_raw_results.json"
    out_report = tmp_path / "out_report.json"

    baseline_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q1",
                        "answer": "A",
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doca", "page_numbers": [2]}]}},
                    },
                    {
                        "question_id": "q2",
                        "answer": "B",
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "docb", "page_numbers": [5]}]}},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_raw_results.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q1"},
                    "answer_text": "A",
                    "telemetry": {
                        "retrieved_page_ids": ["doca_2"],
                        "context_page_ids": ["doca_2"],
                        "used_page_ids": ["doca_2"],
                        "doc_shortlist": ["doca"],
                    },
                },
                {
                    "case": {"case_id": "q2"},
                    "answer_text": "B",
                    "telemetry": {
                        "retrieved_page_ids": ["docb_5"],
                        "context_page_ids": ["docb_5"],
                        "used_page_ids": ["docb_5"],
                        "doc_shortlist": ["docb"],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {"question_id": "q1", "minimal_required_support_pages": ["doca_1", "docc_3"]},
                    {"question_id": "q2", "minimal_required_support_pages": ["docb_7"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_scaffold_support_surrogate.py",
            "--baseline-submission",
            str(baseline_submission),
            "--baseline-raw-results",
            str(baseline_raw_results),
            "--scaffold",
            str(scaffold),
            "--qid",
            "q1",
            "--out-submission",
            str(out_submission),
            "--out-raw-results",
            str(out_raw_results),
            "--out-report",
            str(out_report),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    submission_payload = json.loads(out_submission.read_text(encoding="utf-8"))
    answer_q1 = next(item for item in submission_payload["answers"] if item["question_id"] == "q1")
    answer_q2 = next(item for item in submission_payload["answers"] if item["question_id"] == "q2")
    assert answer_q1["telemetry"]["retrieval"]["retrieved_chunk_pages"] == [
        {"doc_id": "doca", "page_numbers": [1]},
        {"doc_id": "docc", "page_numbers": [3]},
    ]
    assert answer_q2["telemetry"]["retrieval"]["retrieved_chunk_pages"] == [{"doc_id": "docb", "page_numbers": [5]}]

    raw_payload = json.loads(out_raw_results.read_text(encoding="utf-8"))
    raw_q1 = next(item for item in raw_payload if item["case"]["case_id"] == "q1")
    raw_q2 = next(item for item in raw_payload if item["case"]["case_id"] == "q2")
    assert raw_q1["telemetry"]["used_page_ids"] == ["doca_1", "docc_3"]
    assert raw_q1["telemetry"]["context_page_ids"] == ["doca_1", "docc_3"]
    assert raw_q1["telemetry"]["retrieved_page_ids"] == ["doca_1", "docc_3"]
    assert raw_q1["telemetry"]["doc_shortlist"] == ["doca", "docc"]
    assert raw_q2["telemetry"]["used_page_ids"] == ["docb_5"]

    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["patched_qids"] == ["q1"]
    assert report["answer_changed_count_vs_baseline"] == 0
    assert report["page_projection_changed_count_vs_baseline"] == 1
