from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from scripts.select_anchor_slice_qids import select_qids

if TYPE_CHECKING:
    from pathlib import Path


def test_select_qids_filters_to_used_support_hits(tmp_path: Path) -> None:
    rows = [
        {
            "question_id": "keep-improved",
            "status": "support_improved",
            "answer_changed": False,
            "candidate_used_hit": True,
            "candidate_used_equivalent_hit": False,
        },
        {
            "question_id": "drop-missing-used-hit",
            "status": "support_equivalent_or_held",
            "answer_changed": False,
            "candidate_used_hit": False,
            "candidate_used_equivalent_hit": False,
        },
        {
            "question_id": "keep-equivalent",
            "status": "support_equivalent_or_held",
            "answer_changed": False,
            "candidate_used_hit": False,
            "candidate_used_equivalent_hit": True,
        },
    ]

    selected, report = select_qids(
        rows=rows,
        include_statuses={"support_improved", "support_equivalent_or_held"},
        exclude_statuses=set(),
        require_no_answer_change=True,
        require_used_support=True,
        excluded_qids=set(),
    )

    assert selected == ["keep-improved", "keep-equivalent"]
    assert report["selected_count"] == 2
    assert report["rejection_reasons_by_qid"]["drop-missing-used-hit"] == "missing_used_support_hit"


def test_cli_writes_qid_file_and_report(tmp_path: Path) -> None:
    anchor_slice_json = tmp_path / "anchor_slice.json"
    out = tmp_path / "qids.txt"
    json_out = tmp_path / "selection.json"
    anchor_slice_json.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "question_id": "q1",
                        "status": "support_improved",
                        "answer_changed": False,
                        "candidate_used_hit": True,
                        "candidate_used_equivalent_hit": False,
                    },
                    {
                        "question_id": "q2",
                        "status": "mixed_or_no_hit",
                        "answer_changed": False,
                        "candidate_used_hit": True,
                        "candidate_used_equivalent_hit": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    from scripts.select_anchor_slice_qids import main

    argv = sys.argv
    try:
        sys.argv = [
            "select_anchor_slice_qids.py",
            "--anchor-slice-json",
            str(anchor_slice_json),
            "--include-status",
            "support_improved",
            "--require-no-answer-change",
            "--require-used-support",
            "--out",
            str(out),
            "--json-out",
            str(json_out),
        ]
        main()
    finally:
        sys.argv = argv

    assert out.read_text(encoding="utf-8") == "q1\n"
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["selected_qids"] == ["q1"]
