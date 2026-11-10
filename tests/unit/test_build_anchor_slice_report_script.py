from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.build_anchor_slice_report import build_rows, render_report

if TYPE_CHECKING:
    from pathlib import Path


def test_build_rows_marks_support_improvement_without_answer_drift(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    candidate_submission = tmp_path / "candidate_submission.json"
    baseline_raw = tmp_path / "baseline_raw.json"
    candidate_raw = tmp_path / "candidate_raw.json"
    baseline_scaffold = tmp_path / "baseline_scaffold.json"
    candidate_scaffold = tmp_path / "candidate_scaffold.json"

    baseline_submission.write_text(
        json.dumps({"answers": [{"question_id": "q1", "answer": False}]}),
        encoding="utf-8",
    )
    candidate_submission.write_text(
        json.dumps({"answers": [{"question_id": "q1", "answer": False}]}),
        encoding="utf-8",
    )
    baseline_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q1"},
                    "answer_text": "False",
                    "telemetry": {"used_page_ids": ["doc_1"], "context_page_ids": ["doc_1"]},
                }
            ]
        ),
        encoding="utf-8",
    )
    candidate_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q1"},
                    "answer_text": "False",
                    "telemetry": {"used_page_ids": ["doc_2"], "context_page_ids": ["doc_2"]},
                }
            ]
        ),
        encoding="utf-8",
    )
    baseline_scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "question": "Based on page 2, was the order granted?",
                        "answer_type": "boolean",
                        "route_family": "model",
                        "expected_answer": "true",
                        "manual_verdict": "correct",
                        "failure_class": "support_undercoverage",
                        "manual_exactness_labels": ["anchor_missing"],
                        "exactness_review_flags": ["required_page_anchor_missing_in_candidates"],
                        "minimal_required_support_pages": ["doc_2"],
                        "required_page_anchor": {"page": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "minimal_required_support_pages": ["doc_2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = build_rows(
        baseline_submission_path=baseline_submission,
        candidate_submission_path=candidate_submission,
        baseline_scaffold_path=baseline_scaffold,
        candidate_scaffold_path=candidate_scaffold,
        baseline_raw_results_path=baseline_raw,
        candidate_raw_results_path=candidate_raw,
        qids=["q1"],
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "support_improved"
    assert row.answer_changed is False
    report = render_report(rows, baseline_label="base", candidate_label="cand")
    assert "status: `support_improved`" in report
    assert "required_page_anchor" in report
