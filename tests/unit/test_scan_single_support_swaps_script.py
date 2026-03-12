from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _submission_record(*, qid: str, answer: object, page_id: str) -> dict[str, object]:
    doc_id, page_num = page_id.rsplit("_", 1)
    return {
        "question_id": qid,
        "answer": answer,
        "telemetry": {
            "model_name": "baseline-model",
            "retrieval": {
                "retrieved_chunk_pages": [
                    {"doc_id": doc_id, "page_numbers": [int(page_num)]},
                ]
            },
        },
    }


def _raw_record(*, qid: str, answer_text: str, used_page_id: str, context_page_id: str) -> dict[str, object]:
    return {
        "case": {"case_id": qid, "question": qid, "answer_type": "boolean"},
        "answer_text": answer_text,
        "telemetry": {
            "question_id": qid,
            "used_page_ids": [used_page_id],
            "context_page_ids": [context_page_id],
            "retrieved_page_ids": [context_page_id],
        },
        "total_ms": 10,
    }


def _preflight(*, p95: int) -> dict[str, object]:
    return {
        "phase": "warmup",
        "page_count_distribution": {"min": 1, "p50": 1, "p95": p95, "max": p95},
        "code_archive_sha256": "code",
        "questions_sha256": "questions",
        "documents_zip_sha256": "docs",
        "pdf_count": 1,
        "phase_collection_name": "collection",
        "qdrant_point_count": 1,
        "truth_audit_workbook_path": "workbook.md",
    }


def test_scan_single_support_swaps_prefers_gold_page_candidate(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    baseline_raw = tmp_path / "baseline_raw.json"
    baseline_preflight = tmp_path / "baseline_preflight.json"
    page_submission = tmp_path / "page_submission.json"
    page_raw = tmp_path / "page_raw.json"
    page_preflight = tmp_path / "page_preflight.json"
    benchmark = tmp_path / "benchmark.json"
    questions = tmp_path / "questions.json"
    docs_dir = tmp_path / "docs"
    out_dir = tmp_path / "out"

    docs_dir.mkdir()

    q1 = "q1"
    q2 = "q2"
    baseline_submission.write_text(
        json.dumps(
            {
                "architecture_summary": {},
                "answers": [
                    _submission_record(qid=q1, answer=False, page_id="doc_bad_1"),
                    _submission_record(qid=q2, answer=False, page_id="doc_bad_1"),
                ],
            }
        ),
        encoding="utf-8",
    )
    baseline_raw.write_text(
        json.dumps(
            [
                _raw_record(qid=q1, answer_text="False", used_page_id="doc_bad_1", context_page_id="doc_bad_1"),
                _raw_record(qid=q2, answer_text="False", used_page_id="doc_bad_1", context_page_id="doc_bad_1"),
            ]
        ),
        encoding="utf-8",
    )
    baseline_preflight.write_text(json.dumps(_preflight(p95=1)), encoding="utf-8")

    page_submission.write_text(
        json.dumps(
            {
                "architecture_summary": {},
                "answers": [
                    _submission_record(qid=q1, answer=False, page_id="doc_good_2"),
                    _submission_record(qid=q2, answer=False, page_id="doc_bad_2"),
                ],
            }
        ),
        encoding="utf-8",
    )
    page_raw.write_text(
        json.dumps(
            [
                _raw_record(qid=q1, answer_text="False", used_page_id="doc_good_2", context_page_id="doc_good_2"),
                _raw_record(qid=q2, answer_text="False", used_page_id="doc_bad_2", context_page_id="doc_bad_2"),
            ]
        ),
        encoding="utf-8",
    )
    page_preflight.write_text(json.dumps(_preflight(p95=1)), encoding="utf-8")

    benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": q1,
                        "gold_page_ids": ["doc_good_2"],
                        "gold_items": [],
                        "items": [],
                        "wrong_document_risk": False,
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                    },
                    {
                        "question_id": q2,
                        "gold_page_ids": ["doc_other_3"],
                        "gold_items": [],
                        "items": [],
                        "wrong_document_risk": False,
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    questions.write_text(
        json.dumps(
            [
                {"id": q1, "question": "question one", "answer_type": "boolean"},
                {"id": q2, "question": "question two", "answer_type": "boolean"},
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/scan_single_support_swaps.py",
            "--baseline-label",
            "baseline",
            "--page-source-label",
            "candidate",
            "--baseline-submission",
            str(baseline_submission),
            "--baseline-raw-results",
            str(baseline_raw),
            "--baseline-preflight",
            str(baseline_preflight),
            "--page-source-submission",
            str(page_submission),
            "--page-source-raw-results",
            str(page_raw),
            "--page-source-preflight",
            str(page_preflight),
            "--benchmark",
            str(benchmark),
            "--questions",
            str(questions),
            "--docs-dir",
            str(docs_dir),
            "--out-dir",
            str(out_dir),
            "--judge-top-k",
            "0",
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads((out_dir / "single_support_swap_scan.json").read_text(encoding="utf-8"))
    assert payload["results"][0]["question_id"] == "q1"
    assert payload["results"][0]["recommendation"] == "PROMISING"
    assert payload["submission_policy"] == "NO_SUBMIT_WITHOUT_USER_APPROVAL"


def test_scan_single_support_swaps_honors_include_qids_file(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    baseline_raw = tmp_path / "baseline_raw.json"
    baseline_preflight = tmp_path / "baseline_preflight.json"
    page_submission = tmp_path / "page_submission.json"
    page_raw = tmp_path / "page_raw.json"
    page_preflight = tmp_path / "page_preflight.json"
    benchmark = tmp_path / "benchmark.json"
    questions = tmp_path / "questions.json"
    qids_file = tmp_path / "qids.txt"
    docs_dir = tmp_path / "docs"
    out_dir = tmp_path / "out"

    docs_dir.mkdir()
    q1 = "q1"
    q2 = "q2"

    baseline_submission.write_text(
        json.dumps(
            {
                "architecture_summary": {},
                "answers": [
                    _submission_record(qid=q1, answer=False, page_id="doc_bad_1"),
                    _submission_record(qid=q2, answer=False, page_id="doc_bad_1"),
                ],
            }
        ),
        encoding="utf-8",
    )
    baseline_raw.write_text(
        json.dumps(
            [
                _raw_record(qid=q1, answer_text="False", used_page_id="doc_bad_1", context_page_id="doc_bad_1"),
                _raw_record(qid=q2, answer_text="False", used_page_id="doc_bad_1", context_page_id="doc_bad_1"),
            ]
        ),
        encoding="utf-8",
    )
    baseline_preflight.write_text(json.dumps(_preflight(p95=1)), encoding="utf-8")

    page_submission.write_text(
        json.dumps(
            {
                "architecture_summary": {},
                "answers": [
                    _submission_record(qid=q1, answer=False, page_id="doc_good_2"),
                    _submission_record(qid=q2, answer=False, page_id="doc_good_2"),
                ],
            }
        ),
        encoding="utf-8",
    )
    page_raw.write_text(
        json.dumps(
            [
                _raw_record(qid=q1, answer_text="False", used_page_id="doc_good_2", context_page_id="doc_good_2"),
                _raw_record(qid=q2, answer_text="False", used_page_id="doc_good_2", context_page_id="doc_good_2"),
            ]
        ),
        encoding="utf-8",
    )
    page_preflight.write_text(json.dumps(_preflight(p95=1)), encoding="utf-8")

    benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": q1,
                        "gold_page_ids": ["doc_good_2"],
                        "gold_items": [],
                        "items": [],
                        "wrong_document_risk": False,
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                    },
                    {
                        "question_id": q2,
                        "gold_page_ids": ["doc_good_2"],
                        "gold_items": [],
                        "items": [],
                        "wrong_document_risk": False,
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    questions.write_text(
        json.dumps(
            [
                {"id": q1, "question": "question one", "answer_type": "boolean"},
                {"id": q2, "question": "question two", "answer_type": "boolean"},
            ]
        ),
        encoding="utf-8",
    )
    qids_file.write_text("q1\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/scan_single_support_swaps.py",
            "--baseline-label",
            "baseline",
            "--page-source-label",
            "candidate",
            "--baseline-submission",
            str(baseline_submission),
            "--baseline-raw-results",
            str(baseline_raw),
            "--baseline-preflight",
            str(baseline_preflight),
            "--page-source-submission",
            str(page_submission),
            "--page-source-raw-results",
            str(page_raw),
            "--page-source-preflight",
            str(page_preflight),
            "--benchmark",
            str(benchmark),
            "--questions",
            str(questions),
            "--docs-dir",
            str(docs_dir),
            "--out-dir",
            str(out_dir),
            "--judge-top-k",
            "0",
            "--include-qids-file",
            str(qids_file),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads((out_dir / "single_support_swap_scan.json").read_text(encoding="utf-8"))
    assert payload["candidates_scanned"] == 1
    assert payload["include_qids"] == ["q1"]
    assert payload["results"][0]["question_id"] == "q1"
