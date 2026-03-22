from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.run_offline_hypothesis_cycle import _write_cycle_summary

if TYPE_CHECKING:
    from pathlib import Path


def test_write_cycle_summary_marks_no_submit_policy(tmp_path: Path) -> None:
    out = tmp_path / "cycle.json"
    _write_cycle_summary(
        path=out,
        suffix="candidate",
        candidate_submission=tmp_path / "submission_candidate.json",
        candidate_preflight=tmp_path / "preflight_candidate.json",
        candidate_raw_results=tmp_path / "raw_candidate.json",
        gate_report=tmp_path / "gate.md",
        anchor_slice_report=tmp_path / "anchor.md",
        scoring_report=tmp_path / "score.md",
        supervisor_report=tmp_path / "supervisor.md",
        exactness_queue_report=tmp_path / "queue.md",
    )

    payload = json.loads(out.read_text())
    assert payload["artifact_suffix"] == "candidate"
    assert payload["anchor_slice_report"].endswith("anchor.md")
    assert payload["scoring_report"].endswith("score.md")
    assert payload["submission_policy"] == "NO_SUBMIT_WITHOUT_USER_APPROVAL"
