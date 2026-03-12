from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_audit_comparison_title_page_candidates_identifies_page1_rescue(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    truth_path = tmp_path / "truth.json"
    baseline_raw_path = tmp_path / "baseline_raw.json"
    source_raw_path = tmp_path / "source_raw.json"
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    out_seed = tmp_path / "seed_qids.json"

    qid = "qid-9f9"
    _write_json(
        questions_path,
        [
            {
                "id": qid,
                "question": "Do cases CA 004/2025 and SCT 295/2025 involve any of the same legal entities or individuals as parties?",
                "answer_type": "boolean",
            },
            {
                "id": "other",
                "question": "What is the result of case X?",
                "answer_type": "free_text",
            },
        ],
    )
    _write_json(
        truth_path,
        {
            "records": [
                {
                    "question_id": qid,
                    "question": "Do cases CA 004/2025 and SCT 295/2025 involve any of the same legal entities or individuals as parties?",
                    "answer_type": "boolean",
                    "support_shape_class": "comparison",
                    "resolved_doc_titles": {
                        "doc-a": "CA 004/2025",
                        "doc-b": "SCT 295/2025",
                    },
                    "minimal_required_support_pages": ["doc-a_1", "doc-b_1"],
                    "notes": "Both title pages are required to compare parties.",
                }
            ]
        },
    )
    _write_json(
        baseline_raw_path,
        [
            {
                "case": {
                    "case_id": qid,
                    "question": "Do cases CA 004/2025 and SCT 295/2025 involve any of the same legal entities or individuals as parties?",
                    "answer_type": "boolean",
                },
                "answer_text": "false",
                "telemetry": {
                    "used_page_ids": ["doc-a_2", "doc-b_7"],
                    "context_page_ids": ["doc-a_1", "doc-a_2", "doc-b_1", "doc-b_7"],
                    "retrieved_page_ids": ["doc-a_1", "doc-a_2", "doc-b_1", "doc-b_7"],
                },
                "total_ms": 1,
            }
        ],
    )
    _write_json(
        source_raw_path,
        [
            {
                "case": {
                    "case_id": qid,
                    "question": "Do cases CA 004/2025 and SCT 295/2025 involve any of the same legal entities or individuals as parties?",
                    "answer_type": "boolean",
                },
                "answer_text": "false",
                "telemetry": {
                    "used_page_ids": ["doc-a_1", "doc-b_1"],
                    "context_page_ids": ["doc-a_1", "doc-b_1"],
                    "retrieved_page_ids": ["doc-a_1", "doc-b_1"],
                },
                "total_ms": 1,
            }
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_comparison_title_page_candidates.py",
            "--questions",
            str(questions_path),
            "--truth-audit",
            str(truth_path),
            "--baseline-raw-results",
            str(baseline_raw_path),
            "--baseline-label",
            "baseline",
            "--source-raw-results",
            f"iter12={source_raw_path}",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--out-seed-qids",
            str(out_seed),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        check=True,
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["submission_policy"] == "NO_SUBMIT_WITHOUT_USER_APPROVAL"
    record = report["records"][0]
    assert record["question_id"] == qid
    assert record["compare_kind"] == "party_overlap"
    assert record["recommendation"] == "PROMISING"
    assert record["baseline_used_page1_doc_hits"] == 0
    assert record["baseline_retrieved_page1_doc_hits"] == 2
    assert record["missing_used_page1_doc_ids"] == ["doc-a", "doc-b"]
    source_signal = record["source_signals"][0]
    assert source_signal["label"] == "iter12"
    assert source_signal["answer_changed"] is False
    assert source_signal["page1_doc_hits"] == 2
    seed_qids = json.loads(out_seed.read_text(encoding="utf-8"))
    assert seed_qids == [qid]
    markdown = out_md.read_text(encoding="utf-8")
    assert "NO_SUBMIT_WITHOUT_USER_APPROVAL" in markdown
    assert qid in markdown
