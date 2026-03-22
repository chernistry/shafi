from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_audit_duplicate_families_reports_pair(tmp_path: Path) -> None:
    scan_results = tmp_path / "scan_results.jsonl"
    scan_results.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": "doc-a",
                        "sha256": "sha-a",
                        "collision_doc_ids": ["sha-b"],
                        "duplicate_same_family_doc_ids": [],
                    }
                ),
                json.dumps(
                    {
                        "doc_id": "doc-b",
                        "sha256": "sha-b",
                        "collision_doc_ids": ["sha-a"],
                        "duplicate_same_family_doc_ids": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scaffold = tmp_path / "scaffold.json"
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {"question_id": "q1", "retrieved_chunk_pages": [{"doc_id": "doc-a", "page_numbers": [1]}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    submission = tmp_path / "submission.json"
    submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q1",
                        "answer": "x",
                        "telemetry": {
                            "retrieval": {
                                "retrieved_chunk_pages": [
                                    {"doc_id": "doc-a", "page_numbers": [1]},
                                    {"doc_id": "doc-b", "page_numbers": [2]},
                                ]
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_duplicate_families.py",
            "--scan-results-jsonl",
            str(scan_results),
            "--scaffold",
            str(scaffold),
            "--submission",
            str(submission),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/shafi",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["pair_count"] == 1
    assert payload["pairs"][0]["recommendation"] == "risky"
    assert (
        payload["pairs"][0]["grounding_g_score_beta_2_5_current"]
        < payload["pairs"][0]["grounding_g_score_beta_2_5_without_partner"]
    )
    assert payload["pairs"][0]["grounding_g_score_beta_2_5_delta"] > 0
    assert payload["pairs"][0]["improved_qids"] == ["q1"]
    assert "Duplicate Family Audit" in out_md.read_text(encoding="utf-8")
