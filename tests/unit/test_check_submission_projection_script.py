from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_check_submission_projection_script_reports_projection_summary(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.json"
    report_path = tmp_path / "report.md"

    eval_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q-bool",
                        "answer_type": "boolean",
                        "answer": "Yes",
                        "telemetry": {"used_page_ids": ["doca_1"], "ttft_ms": 10},
                    },
                    {
                        "question_id": "q-free",
                        "answer_type": "free_text",
                        "answer": "1. Alpha (cite: doca:0:0:x)\n2. " + ("Beta " * 100),
                        "telemetry": {"used_page_ids": ["docb_2"], "ttft_ms": 20},
                    },
                    {
                        "question_id": "q-date",
                        "answer_type": "date",
                        "answer": "01/01/2020",
                        "telemetry": {"used_page_ids": ["docc_3"], "ttft_ms": 30},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/check_submission_projection.py",
            "--eval",
            str(eval_path),
            "--out",
            str(report_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Submission Projection Check" in report
    assert "Submission-compliant cases: `2/3`" in report
    assert "Cases with projected answer changes vs eval artifact: `2`" in report
    assert "Boolean JSON-safe after projection: `1/1`" in report
    assert "`q-date` [date] `answer`: date must be ISO YYYY-MM-DD or null" in report
