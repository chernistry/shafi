from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

from scripts.mine_within_doc_rerank_opportunities import mine_within_doc_rerank_opportunities

if TYPE_CHECKING:
    from pathlib import Path


def test_mine_within_doc_rerank_opportunities_detects_opportunity(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    raw_results = tmp_path / "raw_results.json"

    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "question": "According to page 2 of case CFI 1/2024, what is the claim number?",
                        "manual_verdict": "correct",
                        "failure_class": "support_undercoverage",
                        "support_shape_class": "generic",
                        "route_family": "strict",
                        "minimal_required_support_pages": ["doc_2"],
                    },
                    {
                        "question_id": "q2",
                        "question": "According to the title page, who is the claimant?",
                        "manual_verdict": "correct",
                        "failure_class": "support_undercoverage",
                        "support_shape_class": "named_metadata",
                        "route_family": "strict",
                        "minimal_required_support_pages": ["doc_1"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_results.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q1"},
                    "telemetry": {
                        "retrieved_page_ids": ["doc_1", "doc_2"],
                        "context_page_ids": ["doc_1", "doc_2"],
                        "used_page_ids": ["doc_1"],
                    },
                },
                {
                    "case": {"case_id": "q2"},
                    "telemetry": {
                        "retrieved_page_ids": ["doc_1"],
                        "context_page_ids": ["doc_1"],
                        "used_page_ids": ["doc_1"],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    rows, summaries = mine_within_doc_rerank_opportunities(
        scaffold_path=scaffold,
        raw_results_path=raw_results,
    )

    assert len(rows) == 2
    row1 = next(row for row in rows if row.question_id == "q1")
    assert row1.gold_in_retrieved is True
    assert row1.gold_in_used is False
    assert row1.within_doc_rerank_opportunity is True

    row2 = next(row for row in rows if row.question_id == "q2")
    assert row2.within_doc_rerank_opportunity is False

    summary = next(item for item in summaries if item.family == "explicit_page_two")
    assert summary.opportunity_count == 1
    assert summary.likely_actionable is False


def test_cli_writes_reports(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    raw_results = tmp_path / "raw_results.json"
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"

    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "question": "According to page 2 of case CFI 1/2024, what is the claim number?",
                        "manual_verdict": "correct",
                        "failure_class": "support_undercoverage",
                        "support_shape_class": "generic",
                        "route_family": "strict",
                        "minimal_required_support_pages": ["doc_2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_results.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q1"},
                    "telemetry": {
                        "retrieved_page_ids": ["doc_1", "doc_2"],
                        "context_page_ids": ["doc_1", "doc_2"],
                        "used_page_ids": ["doc_1"],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/mine_within_doc_rerank_opportunities.py",
            "--scaffold",
            str(scaffold),
            "--raw-results",
            str(raw_results),
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
    assert payload["case_count"] == 1
    assert payload["opportunity_count"] == 1
    assert "Within-Doc Rerank Opportunities" in out_md.read_text(encoding="utf-8")
