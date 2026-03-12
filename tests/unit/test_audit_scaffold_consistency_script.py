from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _scaffold(*, question_id: str, expected_answer: object, manual_verdict: str, failure_class: str, labels: list[str]) -> dict[str, object]:
    return {
        "records": [
            {
                "question_id": question_id,
                "expected_answer": expected_answer,
                "manual_verdict": manual_verdict,
                "failure_class": failure_class,
                "manual_exactness_labels": labels,
            }
        ]
    }


def test_audit_scaffold_consistency_reports_variant_mismatches(tmp_path: Path) -> None:
    scaffold_a = tmp_path / "truth_audit_scaffold_a.json"
    scaffold_b = tmp_path / "truth_audit_scaffold_b.json"
    report = tmp_path / "report.md"
    json_out = tmp_path / "report.json"

    scaffold_a.write_text(
        json.dumps(
            _scaffold(
                question_id="q5046",
                expected_answer="ENF-316-2023/2",
                manual_verdict="incorrect",
                failure_class="weak_path_fallback",
                labels=["platform_exact_risk", "page_specific_exact_risk"],
            )
        ),
        encoding="utf-8",
    )
    scaffold_b.write_text(
        json.dumps(
            _scaffold(
                question_id="q5046",
                expected_answer="ENF 316/2023",
                manual_verdict="correct",
                failure_class="support_undercoverage",
                labels=["semantic_correct"],
            )
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_scaffold_consistency.py",
            "--scaffold-glob",
            "missing/*.json",
            "--scaffold",
            str(scaffold_a),
            "--scaffold",
            str(scaffold_b),
            "--qid",
            "q5046",
            "--out",
            str(report),
            "--json-out",
            str(json_out),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report_text = report.read_text(encoding="utf-8")
    assert "# Truth-Audit Scaffold Consistency Report" in report_text
    assert "q5046" in report_text
    assert "ENF-316-2023/2" in report_text
    assert "ENF 316/2023" in report_text
    assert "- consistent: `False`" in report_text

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    report_payload = payload["reports"][0]
    expected_field = next(field for field in report_payload["fields"] if field["field"] == "expected_answer")
    assert expected_field["consistent"] is False
    assert len(expected_field["variants"]) == 2


def test_audit_scaffold_consistency_reports_missing_records(tmp_path: Path) -> None:
    scaffold_a = tmp_path / "truth_audit_scaffold_a.json"
    scaffold_b = tmp_path / "truth_audit_scaffold_b.json"
    json_out = tmp_path / "report.json"

    scaffold_a.write_text(
        json.dumps(
            _scaffold(
                question_id="q43",
                expected_answer=["Architeriors Interior Design (L.L.C)"],
                manual_verdict="incorrect",
                failure_class="wrong_strict_extraction",
                labels=["platform_exact_risk", "suffix_risk"],
            )
        ),
        encoding="utf-8",
    )
    scaffold_b.write_text(json.dumps({"records": []}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_scaffold_consistency.py",
            "--scaffold-glob",
            "missing/*.json",
            "--scaffold",
            str(scaffold_a),
            "--scaffold",
            str(scaffold_b),
            "--qid",
            "q43",
            "--json-out",
            str(json_out),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    report_payload = payload["reports"][0]
    assert report_payload["found_in"] == 1
    assert report_payload["missing_paths"] == [str(scaffold_b.resolve())]
