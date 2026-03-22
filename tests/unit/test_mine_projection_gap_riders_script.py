from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.mine_projection_gap_riders import build_projection_gap_opportunities

if TYPE_CHECKING:
    from pathlib import Path


def _write_submission(path: Path, *, qid: str, page_numbers: list[int]) -> None:
    payload = {
        "architecture_summary": {},
        "answers": [
            {
                "question_id": qid,
                "answer": "false",
                "telemetry": {
                    "retrieval": {
                        "retrieved_chunk_pages": [
                            {
                                "doc_id": "docA",
                                "page_numbers": page_numbers,
                            }
                        ]
                    }
                },
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_mine_projection_gap_opportunities_finds_page_number_gain(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    scaffold.write_text(
        json.dumps(
            {
                "summary": {},
                "records": [
                    {
                        "question_id": "qid-1",
                        "question": "According to page 2 of the judgment, what claim number applied?",
                        "manual_verdict": "correct",
                        "failure_class": "support_undercoverage",
                        "minimal_required_support_pages": ["docZ_2"],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    submissions_dir = tmp_path / "subs"
    submissions_dir.mkdir()
    baseline = submissions_dir / "submission_v_current.json"
    better = submissions_dir / "submission_v_better.json"
    _write_submission(baseline, qid="qid-1", page_numbers=[1])
    _write_submission(better, qid="qid-1", page_numbers=[2])

    rows = build_projection_gap_opportunities(
        scaffold_path=scaffold,
        baseline_submission_path=baseline,
        submissions_dir=submissions_dir,
    )

    assert len(rows) == 1
    assert rows[0].question_id == "qid-1"
    assert rows[0].source_submission == "submission_v_better.json"
    assert rows[0].baseline_target_hits == 0
    assert rows[0].source_target_hits == 1


def test_mine_projection_gap_opportunities_ignores_non_support_cases(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    scaffold.write_text(
        json.dumps(
            {
                "summary": {},
                "records": [
                    {
                        "question_id": "qid-1",
                        "question": "Who won?",
                        "manual_verdict": "correct",
                        "failure_class": "",
                        "minimal_required_support_pages": ["docZ_2"],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    submissions_dir = tmp_path / "subs"
    submissions_dir.mkdir()
    baseline = submissions_dir / "submission_v_current.json"
    better = submissions_dir / "submission_v_better.json"
    _write_submission(baseline, qid="qid-1", page_numbers=[1])
    _write_submission(better, qid="qid-1", page_numbers=[2])

    rows = build_projection_gap_opportunities(
        scaffold_path=scaffold,
        baseline_submission_path=baseline,
        submissions_dir=submissions_dir,
    )

    assert rows == []
