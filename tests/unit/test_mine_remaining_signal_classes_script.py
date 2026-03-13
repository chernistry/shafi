from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

from scripts.mine_remaining_signal_classes import summarize_remaining_signal_classes

if TYPE_CHECKING:
    from pathlib import Path


def test_summarize_remaining_signal_classes_marks_multi_qid_family_actionable(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "question": "Which claimant appears on the title page of both CA 1/2024 and SCT 2/2024?",
                        "manual_verdict": "correct",
                        "failure_class": "support_undercoverage",
                        "support_shape_class": "comparison",
                        "minimal_required_support_pages": ["doc1_1", "doc2_1"],
                    },
                    {
                        "question_id": "q2",
                        "question": "Which respondent appears on the title page of both CA 3/2024 and SCT 4/2024?",
                        "manual_verdict": "correct",
                        "failure_class": "support_undercoverage",
                        "support_shape_class": "comparison",
                        "minimal_required_support_pages": ["doc3_1", "doc4_1"],
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    oracle = tmp_path / "oracle.json"
    oracle.write_text(
        json.dumps(
            {
                "opportunities": [
                    {
                        "question_id": "q1",
                        "question": "Which claimant appears on the title page of both CA 1/2024 and SCT 2/2024?",
                        "failure_class": "support_undercoverage",
                        "exact_gold_recovered": False,
                    },
                    {
                        "question_id": "q2",
                        "question": "Which respondent appears on the title page of both CA 3/2024 and SCT 4/2024?",
                        "failure_class": "support_undercoverage",
                        "exact_gold_recovered": False,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    known = tmp_path / "known.txt"
    known.write_text("q0\n", encoding="utf-8")

    summaries = summarize_remaining_signal_classes(
        scaffold_path=scaffold,
        oracle_opportunities_path=oracle,
        known_qids_path=known,
    )

    by_family = {item.family: item for item in summaries}
    family = by_family["comparison_title_party"]
    assert family.uncovered_qid_count == 2
    assert family.oracle_new_qid_count == 2
    assert family.likely_actionable is True


def test_mine_remaining_signal_classes_script_writes_reports(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "question": "According to the cover page of the law, what law number is stated?",
                        "manual_verdict": "correct",
                        "failure_class": "support_undercoverage",
                        "support_shape_class": "named_metadata",
                        "minimal_required_support_pages": ["doc1_1"],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"opportunities": []}, indent=2), encoding="utf-8")
    known = tmp_path / "known.txt"
    known.write_text("", encoding="utf-8")
    out_json = tmp_path / "remaining.json"
    out_md = tmp_path / "remaining.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/mine_remaining_signal_classes.py",
            "--scaffold",
            str(scaffold),
            "--oracle-opportunities-json",
            str(oracle),
            "--known-qids-file",
            str(known),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["family_count"] == 1
    assert payload["summaries"][0]["family"] == "single_doc_title_cover"
    report = out_md.read_text(encoding="utf-8")
    assert "# Remaining Signal Classes" in report
