from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_audit_explicit_page_reference_candidates_reads_page_trace_ledger(tmp_path: Path) -> None:
    page_trace_ledger = tmp_path / "page_trace_ledger.json"
    page_trace_ledger.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "qid": "q1",
                        "question": "According to page 2 of the judgment, what is the claim number?",
                        "failure_stage": "used_pages",
                        "route": "model",
                        "trust_tier": "trusted",
                        "gold_in_used": False,
                        "gold_pages": ["doc_2"],
                        "used_pages": ["doc_1"],
                        "false_positive_pages": ["doc_1"],
                        "page_budget_overrun": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out_md = tmp_path / "out.md"
    out_json = tmp_path / "out.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_explicit_page_reference_candidates.py",
            "--page-trace-ledger",
            str(page_trace_ledger),
            "--min-meaningful-qids",
            "1",
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    record = payload["records"][0]
    assert summary["meaningful_qid_count"] == 1
    assert summary["trusted_qid_count"] == 1
    assert summary["phrase_type_counts"]["second_page"] == 1
    assert summary["failure_stage_counts"]["used_pages"] == 1
    assert summary["verdict"] == "continue_to_ticket_14"
    assert record["qid"] == "q1"
    assert record["phrase_type"] == "second_page"
    assert record["requested_page"] == 2
    assert record["gold_pages"] == ["doc_2"]
    assert record["used_pages"] == ["doc_1"]
