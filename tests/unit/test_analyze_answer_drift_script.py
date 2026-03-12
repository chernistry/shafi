from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_analyze_answer_drift_reports_changed_answers_by_type_and_route(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline.json"
    candidate_submission = tmp_path / "candidate.json"
    scaffold = tmp_path / "scaffold.json"
    report_path = tmp_path / "drift.md"
    json_path = tmp_path / "drift.json"

    baseline_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q1",
                        "answer": "old",
                        "telemetry": {"model_name": "strict-extractor"},
                    },
                    {
                        "question_id": "q2",
                        "answer": "same",
                        "telemetry": {"model_name": "gpt-4.1-mini"},
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
                        "answer": "new",
                        "telemetry": {"model_name": "structured-extractor"},
                    },
                    {
                        "question_id": "q2",
                        "answer": "same",
                        "telemetry": {"model_name": "gpt-4.1-mini"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "answer_type": "free_text",
                        "route_family": "structured",
                        "question": "What happened?",
                    },
                    {
                        "question_id": "q2",
                        "answer_type": "boolean",
                        "route_family": "model",
                        "question": "Was it granted?",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_answer_drift.py",
            "--baseline-submission",
            str(baseline_submission),
            "--candidate-submission",
            str(candidate_submission),
            "--scaffold",
            str(scaffold),
            "--baseline-label",
            "baseline",
            "--label",
            "candidate",
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
    assert "- Changed answers: `1`" in report
    assert "by_answer_type: `{'free_text': 1}`" in report
    assert "by_route_family: `{'structured': 1}`" in report
    assert "- baseline_answer: `old`" in report
    assert "- candidate_answer: `new`" in report

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["changed_answer_count"] == 1
    assert payload["records"][0]["question_id"] == "q1"
