from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.audit_explicit_anchor_gaps import build_gaps

if TYPE_CHECKING:
    from pathlib import Path


def _write_scaffold(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"summary": {}, "records": records}, indent=2), encoding="utf-8")


def _write_raw_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def test_build_gaps_reports_only_missing_explicit_anchor_pages(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    raw_results = tmp_path / "raw_results.json"
    _write_scaffold(
        scaffold,
        [
            {
                "question_id": "qid-1",
                "question": "According to page 2 of the judgment, what is the claim number?",
                "manual_verdict": "correct",
                "failure_class": "support_undercoverage",
                "minimal_required_support_pages": ["doc_2"],
            },
            {
                "question_id": "qid-2",
                "question": "According to the title page, what is the law number?",
                "manual_verdict": "correct",
                "failure_class": "support_undercoverage",
                "minimal_required_support_pages": ["docb_1"],
            },
        ],
    )
    _write_raw_results(
        raw_results,
        [
            {
                "case": {"case_id": "qid-1"},
                "telemetry": {
                    "used_page_ids": ["doc_1"],
                    "context_page_ids": ["doc_3"],
                    "retrieved_page_ids": ["doc_4"],
                },
            },
            {
                "case": {"case_id": "qid-2"},
                "telemetry": {
                    "used_page_ids": ["docb_1"],
                    "context_page_ids": [],
                    "retrieved_page_ids": [],
                },
            },
        ],
    )

    gaps = build_gaps(
        scaffold_path=scaffold,
        current_raw_results_path=raw_results,
        manual_verdicts={"correct"},
        failure_classes={"support_undercoverage"},
    )

    assert len(gaps) == 1
    assert gaps[0].question_id == "qid-1"
    assert gaps[0].missing_gold_page_ids == ["doc_2"]
    assert gaps[0].current_has_gold is False


def test_build_gaps_ignores_non_explicit_anchor_questions(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    raw_results = tmp_path / "raw_results.json"
    _write_scaffold(
        scaffold,
        [
            {
                "question_id": "qid-1",
                "question": "What is the law number?",
                "manual_verdict": "correct",
                "failure_class": "support_undercoverage",
                "minimal_required_support_pages": ["doc_1"],
            }
        ],
    )
    _write_raw_results(
        raw_results,
        [{"case": {"case_id": "qid-1"}, "telemetry": {"used_page_ids": [], "context_page_ids": [], "retrieved_page_ids": []}}],
    )

    gaps = build_gaps(
        scaffold_path=scaffold,
        current_raw_results_path=raw_results,
        manual_verdicts={"correct"},
        failure_classes={"support_undercoverage"},
    )

    assert gaps == []
