from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _submission_record(*, qid: str, answer: object, page_id: str, model_name: str = "baseline-model") -> dict[str, object]:
    return {
        "question_id": qid,
        "answer": answer,
        "telemetry": {
            "model_name": model_name,
            "retrieval": {
                "retrieved_chunk_pages": [
                    {"doc_id": page_id.rsplit("_", 1)[0], "page_numbers": [int(page_id.rsplit("_", 1)[1])]}
                ]
            },
        },
    }


def _raw_record(*, qid: str, answer_text: str, used_page_id: str, context_page_id: str) -> dict[str, object]:
    return {
        "case": {"case_id": qid, "question": qid, "answer_type": "name"},
        "answer_text": answer_text,
        "telemetry": {
            "retrieved_chunk_ids": [f"{used_page_id}:chunk"],
            "retrieved_page_ids": [used_page_id],
            "context_chunk_ids": [f"{context_page_id}:chunk"],
            "context_page_ids": [context_page_id],
            "used_chunk_ids": [f"{used_page_id}:chunk"],
            "used_page_ids": [used_page_id],
        },
        "total_ms": 10,
    }


def _preflight(*, p95: int) -> dict[str, object]:
    return {
        "phase": "warmup",
        "answer_type_counts": {"name": 1},
        "page_count_distribution": {"min": 1, "p50": 1, "p95": p95, "max": p95},
        "code_archive_sha256": "code",
        "questions_sha256": "questions",
        "documents_zip_sha256": "docs",
        "pdf_count": 1,
        "phase_collection_name": "collection",
        "qdrant_point_count": 10,
        "truth_audit_workbook_path": "workbook.md",
    }


def test_build_counterfactual_candidate_uses_baseline_answers_and_candidate_pages(tmp_path: Path) -> None:
    answer_submission = tmp_path / "answer_submission.json"
    answer_raw = tmp_path / "answer_raw.json"
    answer_preflight = tmp_path / "answer_preflight.json"
    page_submission = tmp_path / "page_submission.json"
    page_raw = tmp_path / "page_raw.json"
    page_preflight = tmp_path / "page_preflight.json"
    out_submission = tmp_path / "out_submission.json"
    out_raw = tmp_path / "out_raw.json"
    out_preflight = tmp_path / "out_preflight.json"
    out_report = tmp_path / "out_report.json"

    answer_submission.write_text(
        json.dumps({"architecture_summary": {"mode": "baseline"}, "answers": [_submission_record(qid="q1", answer="A", page_id="doca_1")]}),
        encoding="utf-8",
    )
    answer_raw.write_text(json.dumps([_raw_record(qid="q1", answer_text="A", used_page_id="doca_1", context_page_id="doca_1")]), encoding="utf-8")
    answer_preflight.write_text(json.dumps(_preflight(p95=1)), encoding="utf-8")

    page_submission.write_text(
        json.dumps({"architecture_summary": {"mode": "candidate"}, "answers": [_submission_record(qid="q1", answer="B", page_id="docb_2", model_name="candidate-model")]}),
        encoding="utf-8",
    )
    page_raw.write_text(json.dumps([_raw_record(qid="q1", answer_text="B", used_page_id="docb_2", context_page_id="docb_2")]), encoding="utf-8")
    page_preflight.write_text(json.dumps(_preflight(p95=2)), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_counterfactual_candidate.py",
            "--answer-source-submission",
            str(answer_submission),
            "--answer-source-raw-results",
            str(answer_raw),
            "--answer-source-preflight",
            str(answer_preflight),
            "--page-source-submission",
            str(page_submission),
            "--page-source-raw-results",
            str(page_raw),
            "--page-source-preflight",
            str(page_preflight),
            "--out-submission",
            str(out_submission),
            "--out-raw-results",
            str(out_raw),
            "--out-preflight",
            str(out_preflight),
            "--out-report",
            str(out_report),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    merged_submission = json.loads(out_submission.read_text(encoding="utf-8"))
    merged_answer = merged_submission["answers"][0]
    assert merged_answer["answer"] == "A"
    assert merged_answer["telemetry"]["retrieval"]["retrieved_chunk_pages"][0]["doc_id"] == "docb"

    merged_raw = json.loads(out_raw.read_text(encoding="utf-8"))[0]
    assert merged_raw["answer_text"] == "A"
    assert merged_raw["telemetry"]["used_page_ids"] == ["docb_2"]

    merged_preflight = json.loads(out_preflight.read_text(encoding="utf-8"))
    assert merged_preflight["page_count_distribution"]["p95"] == 1
    assert merged_preflight["counterfactual_projection"]["submission_policy"] == "NO_SUBMIT_WITHOUT_USER_APPROVAL"

    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["answer_changed_count_vs_answer_source"] == 0
    assert report["page_projection_changed_count_vs_answer_source"] == 1


def test_build_counterfactual_candidate_can_allowlist_candidate_answers(tmp_path: Path) -> None:
    answer_submission = tmp_path / "answer_submission.json"
    answer_raw = tmp_path / "answer_raw.json"
    answer_preflight = tmp_path / "answer_preflight.json"
    page_submission = tmp_path / "page_submission.json"
    page_raw = tmp_path / "page_raw.json"
    page_preflight = tmp_path / "page_preflight.json"
    out_submission = tmp_path / "out_submission.json"
    out_raw = tmp_path / "out_raw.json"
    out_preflight = tmp_path / "out_preflight.json"
    out_report = tmp_path / "out_report.json"

    answer_submission.write_text(
        json.dumps({"architecture_summary": {}, "answers": [_submission_record(qid="q1", answer="A", page_id="doca_1")]}),
        encoding="utf-8",
    )
    answer_raw.write_text(json.dumps([_raw_record(qid="q1", answer_text="A", used_page_id="doca_1", context_page_id="doca_1")]), encoding="utf-8")
    answer_preflight.write_text(json.dumps(_preflight(p95=1)), encoding="utf-8")
    page_submission.write_text(
        json.dumps({"architecture_summary": {}, "answers": [_submission_record(qid="q1", answer="B", page_id="docb_2", model_name="candidate-model")]}),
        encoding="utf-8",
    )
    page_raw.write_text(json.dumps([_raw_record(qid="q1", answer_text="B", used_page_id="docb_2", context_page_id="docb_2")]), encoding="utf-8")
    page_preflight.write_text(json.dumps(_preflight(p95=2)), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_counterfactual_candidate.py",
            "--answer-source-submission",
            str(answer_submission),
            "--answer-source-raw-results",
            str(answer_raw),
            "--answer-source-preflight",
            str(answer_preflight),
            "--page-source-submission",
            str(page_submission),
            "--page-source-raw-results",
            str(page_raw),
            "--page-source-preflight",
            str(page_preflight),
            "--page-source-answer-qid",
            "q1",
            "--out-submission",
            str(out_submission),
            "--out-raw-results",
            str(out_raw),
            "--out-preflight",
            str(out_preflight),
            "--out-report",
            str(out_report),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    merged_submission = json.loads(out_submission.read_text(encoding="utf-8"))
    assert merged_submission["answers"][0]["answer"] == "B"

    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["answer_changed_count_vs_answer_source"] == 1
    assert report["page_projection_changed_count_vs_answer_source"] == 1


def test_build_counterfactual_candidate_can_limit_page_projection_to_allowlist(tmp_path: Path) -> None:
    answer_submission = tmp_path / "answer_submission.json"
    answer_raw = tmp_path / "answer_raw.json"
    answer_preflight = tmp_path / "answer_preflight.json"
    page_submission = tmp_path / "page_submission.json"
    page_raw = tmp_path / "page_raw.json"
    page_preflight = tmp_path / "page_preflight.json"
    out_submission = tmp_path / "out_submission.json"
    out_raw = tmp_path / "out_raw.json"
    out_preflight = tmp_path / "out_preflight.json"
    out_report = tmp_path / "out_report.json"

    answer_submission.write_text(
        json.dumps(
            {
                "architecture_summary": {},
                "answers": [
                    _submission_record(qid="q1", answer="A", page_id="doca_1"),
                    _submission_record(qid="q2", answer="B", page_id="docb_1"),
                ],
            }
        ),
        encoding="utf-8",
    )
    answer_raw.write_text(
        json.dumps(
            [
                _raw_record(qid="q1", answer_text="A", used_page_id="doca_1", context_page_id="doca_1"),
                _raw_record(qid="q2", answer_text="B", used_page_id="docb_1", context_page_id="docb_1"),
            ]
        ),
        encoding="utf-8",
    )
    answer_preflight.write_text(json.dumps(_preflight(p95=1)), encoding="utf-8")
    page_submission.write_text(
        json.dumps(
            {
                "architecture_summary": {},
                "answers": [
                    _submission_record(qid="q1", answer="A*", page_id="docx_2", model_name="candidate-model"),
                    _submission_record(qid="q2", answer="B*", page_id="docy_2", model_name="candidate-model"),
                ],
            }
        ),
        encoding="utf-8",
    )
    page_raw.write_text(
        json.dumps(
            [
                _raw_record(qid="q1", answer_text="A*", used_page_id="docx_2", context_page_id="docx_2"),
                _raw_record(qid="q2", answer_text="B*", used_page_id="docy_2", context_page_id="docy_2"),
            ]
        ),
        encoding="utf-8",
    )
    page_preflight.write_text(json.dumps(_preflight(p95=2)), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_counterfactual_candidate.py",
            "--answer-source-submission",
            str(answer_submission),
            "--answer-source-raw-results",
            str(answer_raw),
            "--answer-source-preflight",
            str(answer_preflight),
            "--page-source-submission",
            str(page_submission),
            "--page-source-raw-results",
            str(page_raw),
            "--page-source-preflight",
            str(page_preflight),
            "--page-source-page-qid",
            "q1",
            "--out-submission",
            str(out_submission),
            "--out-raw-results",
            str(out_raw),
            "--out-preflight",
            str(out_preflight),
            "--out-report",
            str(out_report),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    merged_submission = json.loads(out_submission.read_text(encoding="utf-8"))
    answers = {row["question_id"]: row for row in merged_submission["answers"]}
    assert answers["q1"]["telemetry"]["retrieval"]["retrieved_chunk_pages"][0]["doc_id"] == "docx"
    assert answers["q2"]["telemetry"]["retrieval"]["retrieved_chunk_pages"][0]["doc_id"] == "docb"

    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["answer_changed_count_vs_answer_source"] == 0
    assert report["page_projection_changed_count_vs_answer_source"] == 1
