from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_build_page_trace_ledger_localizes_failure_stages(tmp_path: Path) -> None:
    raw_results = tmp_path / "raw_results.json"
    benchmark = tmp_path / "benchmark.json"
    scaffold = tmp_path / "scaffold.json"
    out_json = tmp_path / "ledger.json"
    out_md = tmp_path / "ledger.md"

    raw_results.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q_lost_after_context", "question": "Who were the claimants in case CFI 010/2024?"},
                    "telemetry": {
                        "question_id": "q_lost_after_context",
                        "retrieved_page_ids": ["docA_1", "docA_2"],
                        "context_page_ids": ["docA_1"],
                        "used_page_ids": ["docA_2"],
                    },
                },
                {
                    "case": {"case_id": "q_wrong_page_same_doc", "question": "Compare the parties in cases CA 004/2025 and SCT 295/2025."},
                    "telemetry": {
                        "question_id": "q_wrong_page_same_doc",
                        "retrieved_page_ids": ["docB_2", "docC_7"],
                        "context_page_ids": ["docB_2"],
                        "used_page_ids": ["docB_2"],
                    },
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q_lost_after_context",
                        "gold_page_ids": ["docA_1"],
                        "trust_tier": "trusted",
                        "wrong_document_risk": False,
                    },
                    {
                        "question_id": "q_wrong_page_same_doc",
                        "gold_page_ids": ["docB_1"],
                        "trust_tier": "trusted",
                        "wrong_document_risk": True,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q_lost_after_context",
                        "question": "Who were the claimants in case CFI 010/2024?",
                        "failure_class": "support_undercoverage",
                        "support_shape_class": "named_metadata",
                        "route_family": "model",
                    },
                    {
                        "question_id": "q_wrong_page_same_doc",
                        "question": "Compare the parties in cases CA 004/2025 and SCT 295/2025.",
                        "failure_class": "support_undercoverage",
                        "support_shape_class": "comparison",
                        "route_family": "model",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_page_trace_ledger.py",
            "--raw-results",
            str(raw_results),
            "--benchmark",
            str(benchmark),
            "--scaffold",
            str(scaffold),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    records = {row["qid"]: row for row in payload["records"]}
    assert records["q_lost_after_context"]["failure_stage"] == "lost_after_context"
    assert records["q_wrong_page_same_doc"]["failure_stage"] == "wrong_page_used_same_doc"
    assert payload["summary"]["gold_in_retrieved_count"] == 1
    assert payload["summary"]["gold_in_used_count"] == 0
    assert "Page Trace Ledger" in out_md.read_text(encoding="utf-8")
