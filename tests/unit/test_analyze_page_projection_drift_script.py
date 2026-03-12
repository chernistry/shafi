from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_analyze_page_projection_drift_groups_changed_questions(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    candidate_submission = tmp_path / "candidate_submission.json"
    candidate_raw = tmp_path / "candidate_raw.json"
    candidate_scaffold = tmp_path / "candidate_scaffold.json"
    report_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    baseline_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q1",
                        "answer": True,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doca", "page_numbers": [1]}]}},
                    },
                    {
                        "question_id": "q2",
                        "answer": False,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "docb", "page_numbers": [1]}]}},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q1",
                        "answer": True,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doca", "page_numbers": [2]}]}},
                    },
                    {
                        "question_id": "q2",
                        "answer": False,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "docb", "page_numbers": [1]}]}},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_raw.write_text(
        json.dumps(
            [
                {"case": {"case_id": "q1", "question": "Based on page 2, what happened?", "answer_type": "boolean"}},
                {"case": {"case_id": "q2", "question": "Who won the case?", "answer_type": "boolean"}},
            ]
        ),
        encoding="utf-8",
    )
    candidate_scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {"question_id": "q1", "route_family": "strict", "failure_class": "support_undercoverage"},
                    {"question_id": "q2", "route_family": "model", "failure_class": ""},
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_page_projection_drift.py",
            "--baseline-label",
            "baseline",
            "--candidate-label",
            "candidate",
            "--baseline-submission",
            str(baseline_submission),
            "--candidate-submission",
            str(candidate_submission),
            "--candidate-raw-results",
            str(candidate_raw),
            "--candidate-scaffold",
            str(candidate_scaffold),
            "--out",
            str(report_path),
            "--json-out",
            str(json_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Changed page projections: `1`" in report
    assert "by_question_flag: `{'page2': 1}`" in report
    assert "## q1" in report

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["changed_count"] == 1
    assert payload["rows"][0]["question_id"] == "q1"
