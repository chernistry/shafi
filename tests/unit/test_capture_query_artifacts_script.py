from __future__ import annotations

import asyncio
import json
from pathlib import Path

from shafi.submission.common import SubmissionCase
from shafi.submission.platform import PlatformCaseResult
from shafi.submission.query_capture import capture_query_artifacts, summarize_results


def test_summarize_results_reports_page_distribution() -> None:
    results = [
        PlatformCaseResult(
            case=SubmissionCase(case_id="q1", question="Question 1", answer_type="name"),
            answer_text="Alice",
            telemetry={"used_page_ids": ["doc_1"]},
            total_ms=10,
        ),
        PlatformCaseResult(
            case=SubmissionCase(case_id="q2", question="Question 2", answer_type="boolean"),
            answer_text="null",
            telemetry={"used_page_ids": []},
            total_ms=30,
        ),
    ]

    summary = summarize_results(
        results,
        questions_path=Path("/tmp/questions.json"),
        docs_dir=Path("/tmp/docs"),
        anomaly_repairs={"repaired_case_ids": ["q2"]},
        truth_audit_path=Path("/tmp/truth.json"),
    )

    assert summary["case_count"] == 2
    assert summary["answer_null_count"] == 1
    assert summary["page_count_distribution"]["p50"] == 0
    assert summary["page_count_distribution"]["zero_count"] == 1
    assert summary["truth_audit_path"] == "/tmp/truth.json"


def test_capture_query_artifacts_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    questions_path = tmp_path / "questions.json"
    raw_results_out = tmp_path / "raw_results.json"
    submission_out = tmp_path / "submission.json"
    summary_out = tmp_path / "summary.json"
    truth_audit_out = tmp_path / "truth_audit.json"
    truth_audit_workbook_out = tmp_path / "truth_audit.md"
    docs_dir = tmp_path / "docs"

    questions_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "case_id": "q1",
                    "question": "Who were the claimants?",
                    "answer_type": "names",
                }
            ]
        ),
        encoding="utf-8",
    )
    docs_dir.mkdir()

    fake_result = PlatformCaseResult(
        case=SubmissionCase(case_id="q1", question="Who were the claimants?", answer_type="names"),
        answer_text="Alice",
        telemetry={"used_page_ids": ["doc_1"], "question_id": "q1"},
        total_ms=11,
    )

    async def _fake_run_questions(*args, **kwargs):
        return [fake_result]

    async def _fake_repair(results, *, fail_fast):
        return results, {"repaired_case_ids": [], "unchanged_case_ids": ["q1"], "skipped_case_ids": []}

    monkeypatch.setattr("shafi.submission.query_capture._run_questions", _fake_run_questions)
    monkeypatch.setattr("shafi.submission.query_capture._repair_anomalous_results", _fake_repair)
    monkeypatch.setattr(
        "shafi.submission.query_capture.build_truth_audit_scaffold",
        lambda **kwargs: {"summary": {"questions_count": 1}, "records": []},
    )
    monkeypatch.setattr(
        "shafi.submission.query_capture.render_truth_audit_workbook",
        lambda scaffold: "# workbook\n",
    )

    summary = asyncio.run(
        capture_query_artifacts(
            questions_path=questions_path,
            raw_results_path=raw_results_out,
            submission_path=submission_out,
            summary_path=summary_out,
            docs_dir=docs_dir,
            truth_audit_path=truth_audit_out,
            truth_audit_workbook_path=truth_audit_workbook_out,
            concurrency=1,
            fail_fast=False,
            ingest_doc_dir=None,
        )
    )

    assert summary["case_count"] == 1
    assert json.loads(raw_results_out.read_text(encoding="utf-8"))[0]["case"]["case_id"] == "q1"
    assert json.loads(submission_out.read_text(encoding="utf-8"))["answers"][0]["question_id"] == "q1"
    assert truth_audit_out.exists()
    assert truth_audit_workbook_out.read_text(encoding="utf-8") == "# workbook\n"
