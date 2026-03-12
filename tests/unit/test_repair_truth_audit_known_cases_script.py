from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_repair_truth_audit_known_cases_updates_stale_dotted_suffix_expectations(tmp_path: Path) -> None:
    scaffold_path = tmp_path / "truth_audit_scaffold.json"
    workbook_path = tmp_path / "truth_audit_workbook.md"
    report_path = tmp_path / "repair_report.json"

    scaffold_path.write_text(
        json.dumps(
            {
                "summary": {
                    "manual_exactness_label_counts": {},
                    "manual_verdict_counts": {
                        "deterministic_complete": 1,
                        "deterministic_incomplete": 0,
                        "free_text_complete": 0,
                        "free_text_incomplete": 0,
                    },
                },
                "records": [
                    {
                        "question_id": "43f77ed8a37c7af9b3e52b0532c593c768f8f1159db9b9ca717a700d6b0a47f9",
                        "answer_type": "names",
                        "question": "From the header/caption section of each document in case TCD 001/2024, identify all parties listed as Claimant.",
                        "current_answer": ["Architeriors Interior Design (LLC)"],
                        "current_answer_text": '["Architeriors Interior Design (LLC)"]',
                        "manual_verdict": "correct",
                        "expected_answer": ["Architeriors Interior Design (LLC)"],
                        "manual_exactness_labels": [],
                        "failure_class": "",
                        "notes": "",
                        "review_packet": {"manual_exactness_labels": []},
                        "support_shape_class": "generic",
                        "audit_priority": 30,
                    }
                ],
                "deterministic_cases": [],
                "free_text_cases": [],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/repair_truth_audit_known_cases.py",
            "--scaffold",
            str(scaffold_path),
            "--rewrite-workbook",
            "--json-out",
            str(report_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    repaired = json.loads(scaffold_path.read_text(encoding="utf-8"))
    record = repaired["records"][0]
    assert record["manual_verdict"] == "incorrect"
    assert record["expected_answer"] == ["Architeriors Interior Design (L.L.C)"]
    assert record["manual_exactness_labels"] == ["platform_exact_risk", "suffix_risk"]
    assert record["failure_class"] == "wrong_strict_extraction"
    assert "Platform exactness already validated" in record["notes"]
    assert repaired["summary"]["manual_exactness_label_counts"] == {
        "platform_exact_risk": 1,
        "suffix_risk": 1,
    }
    assert repaired["summary"]["manual_verdict_counts"]["deterministic_complete"] == 1
    assert "Architeriors Interior Design (L.L.C)" in workbook_path.read_text(encoding="utf-8")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["results"][0]["changed_question_ids"] == [
        "43f77ed8a37c7af9b3e52b0532c593c768f8f1159db9b9ca717a700d6b0a47f9"
    ]


def test_repair_truth_audit_known_cases_leaves_already_correct_scaffold_untouched(tmp_path: Path) -> None:
    scaffold_path = tmp_path / "truth_audit_scaffold.json"
    report_path = tmp_path / "repair_report.json"
    original = {
        "summary": {
            "manual_exactness_label_counts": {"semantic_correct": 1},
            "manual_verdict_counts": {
                "deterministic_complete": 1,
                "deterministic_incomplete": 0,
                "free_text_complete": 0,
                "free_text_incomplete": 0,
            },
        },
        "records": [
            {
                "question_id": "f950917f9b85f687161b1022a11c3ce31e4f6ab459af69dfea311c20893fc8a7",
                "answer_type": "names",
                "question": "Who are listed as the claimants in the case documents for CFI 067/2025?",
                "current_answer": ["Coinmena B.S.C. (C)"],
                "current_answer_text": '["Coinmena B.S.C. (C)"]',
                "manual_verdict": "correct",
                "expected_answer": ["Coinmena B.S.C. (C)"],
                "manual_exactness_labels": ["semantic_correct"],
                "failure_class": "",
                "notes": "",
                "review_packet": {"manual_exactness_labels": ["semantic_correct"]},
                "support_shape_class": "generic",
                "audit_priority": 30,
            }
        ],
        "deterministic_cases": [],
        "free_text_cases": [],
    }
    scaffold_path.write_text(json.dumps(original), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/repair_truth_audit_known_cases.py",
            "--scaffold",
            str(scaffold_path),
            "--json-out",
            str(report_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    repaired = json.loads(scaffold_path.read_text(encoding="utf-8"))
    record = repaired["records"][0]
    assert record["expected_answer"] == ["Coinmena B.S.C. (C)"]
    assert record["manual_exactness_labels"] == ["semantic_correct"]
    assert repaired["deterministic_cases"][0]["question_id"] == record["question_id"]
    assert repaired["summary"]["deterministic_count"] == 1
    assert repaired["summary"]["manual_exactness_label_counts"] == {"semantic_correct": 1}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["results"][0]["changed_question_ids"] == []
