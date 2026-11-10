from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_analyze_page_projection_drift_separates_noise_classes(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    ledger_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "qid": "qid-mixed",
                        "doc_family": "comparison_party",
                        "route": "strict",
                        "failure_stage": "retained_to_used",
                        "gold_pages": ["doc-a_1"],
                        "used_pages": ["doc-a_1", "doc-b_4"],
                        "false_positive_pages": ["doc-b_4"],
                        "page_budget_overrun": 1,
                        "wrong_document_risk": True,
                    },
                    {
                        "qid": "qid-orphan",
                        "doc_family": "same_doc",
                        "route": "strict",
                        "failure_stage": "retained_to_used",
                        "gold_pages": ["doc-c_3"],
                        "used_pages": ["doc-c_7"],
                        "false_positive_pages": ["doc-c_7"],
                        "page_budget_overrun": 0,
                        "wrong_document_risk": False,
                    },
                    {
                        "qid": "qid-low-confidence",
                        "doc_family": "same_doc",
                        "route": "model",
                        "failure_stage": "retained_to_used",
                        "gold_pages": ["doc-d_2"],
                        "used_pages": ["doc-d_2", "doc-d_5"],
                        "false_positive_pages": ["doc-d_5"],
                        "page_budget_overrun": 0,
                        "wrong_document_risk": False,
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_page_projection_drift.py",
            "--page-trace-ledger",
            str(ledger_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["false_positive_case_count"] == 3
    assert summary["mixed_doc_case_count"] == 1
    assert summary["orphan_case_count"] == 1
    assert summary["low_confidence_case_count"] == 1
    assert summary["recommended_max_page_drift"] == 2
    assert (
        report["high_false_positive_qids"][0]["qid"] == "qid-low-confidence"
        or report["high_false_positive_qids"][0]["qid"] == "qid-mixed"
    )
    markdown = out_md.read_text(encoding="utf-8")
    assert "Page Projection Drift Audit" in markdown
    assert "comparison_party" in markdown
