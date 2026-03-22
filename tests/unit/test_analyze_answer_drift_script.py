from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts import analyze_answer_drift as mod

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_report_categorizes_drift_cases() -> None:
    canary = {
        "baseline_concurrency": 1,
        "candidate_concurrency": 2,
        "total_cases": 100,
        "answer_drift_case_ids": ["q1", "q2"],
        "answer_drift_count": 2,
        "page_drift_count": 0,
        "model_drift_count": 0,
        "missing_case_ids": [],
    }
    questions = {
        "q1": {"id": "q1", "question": "Question one?", "answer_type": "boolean"},
        "q2": {"id": "q2", "question": "Question two?", "answer_type": "name"},
    }
    truth = {
        "q1": {"question_id": "q1", "route_family": "strict", "support_shape_class": "comparison", "question_refs": ["CA 001/2025"], "audit_priority": 20},
        "q2": {"question_id": "q2", "route_family": "strict", "support_shape_class": "generic", "question_refs": ["CFI 010/2024"], "audit_priority": 30},
    }

    report = mod._build_report(canary=canary, questions=questions, truth_audit=truth)

    assert report["runtime_recommendation"] == "query_concurrency=1"
    assert report["answer_drift_rate"] == 0.02
    assert report["by_answer_type"] == {"boolean": 1, "name": 1}
    assert report["by_route_family"] == {"strict": 2}
    assert report["by_support_shape_class"] == {"comparison": 1, "generic": 1}


def test_main_writes_reports(tmp_path: Path) -> None:
    canary = tmp_path / "canary.json"
    questions = tmp_path / "questions.json"
    truth = tmp_path / "truth.json"
    out_dir = tmp_path / "out"

    _write_json(
        canary,
        {
            "baseline_concurrency": 1,
            "candidate_concurrency": 2,
            "total_cases": 10,
            "answer_drift_case_ids": ["q1"],
            "answer_drift_count": 1,
            "page_drift_count": 0,
            "model_drift_count": 0,
            "missing_case_ids": [],
        },
    )
    _write_json(questions, [{"id": "q1", "question": "Question one?", "answer_type": "boolean"}])
    _write_json(
        truth,
        {
            "records": [
                {
                    "question_id": "q1",
                    "question": "Question one?",
                    "answer_type": "boolean",
                    "route_family": "strict",
                    "support_shape_class": "comparison",
                    "question_refs": ["CA 001/2025"],
                    "audit_priority": 20,
                }
            ]
        },
    )

    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "analyze_answer_drift.py",
            "--canary",
            str(canary),
            "--questions",
            str(questions),
            "--truth-audit",
            str(truth),
            "--out-dir",
            str(out_dir),
        ]
        mod.main()
    finally:
        sys.argv = old_argv

    payload = json.loads((out_dir / "concurrency_drift_report.json").read_text(encoding="utf-8"))
    assert payload["answer_drift_count"] == 1
    assert payload["runtime_recommendation"] == "query_concurrency=1"
    markdown = (out_dir / "concurrency_drift_report.md").read_text(encoding="utf-8")
    assert "Concurrency Drift Audit" in markdown
    assert "Question one?" in markdown
