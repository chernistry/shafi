from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.run_bounded_counterfactual_cycle import _write_summary
from scripts.select_anchor_slice_qids import select_qids

if TYPE_CHECKING:
    from pathlib import Path


def test_select_qids_can_be_augmented_with_extra_ids() -> None:
    rows = [
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
            "candidate_used_hit": False,
            "candidate_used_equivalent_hit": False,
        },
    ]
    selected, report = select_qids(
        rows=rows,
        include_statuses={"support_improved"},
        exclude_statuses=set(),
        require_no_answer_change=True,
        require_used_support=True,
        excluded_qids=set(),
    )
    final = list(dict.fromkeys([*selected, "trusted-qid"]))
    assert final == ["q1", "trusted-qid"]
    assert report["selected_count"] == 1


def test_write_summary_sets_no_submit_policy(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    _write_summary(
        path=summary_path,
        label="candidate_x",
        baseline_label="baseline_y",
        selected_qids=["q1", "q2"],
        selection_json=tmp_path / "selection.json",
        submission_path=tmp_path / "submission.json",
        raw_results_path=tmp_path / "raw_results.json",
        preflight_path=tmp_path / "preflight.json",
        gate_report_path=tmp_path / "gate.md",
        anchor_slice_path=tmp_path / "anchor_slice.md",
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["selected_qids"] == ["q1", "q2"]
    assert payload["submission_policy"] == "NO_SUBMIT_WITHOUT_USER_APPROVAL"
