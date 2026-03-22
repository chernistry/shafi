from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

from scripts.cluster_historical_support_families import summarize_historical_support_families

if TYPE_CHECKING:
    from pathlib import Path


def test_summarize_historical_support_families_counts_new_qids(tmp_path: Path) -> None:
    opportunities_json = tmp_path / "opportunities.json"
    opportunities_json.write_text(
        json.dumps(
            {
                "opportunity_count": 3,
                "opportunities": [
                    {
                        "question_id": "q1",
                        "question": "According to the cover page of the DIFC Trust Law, what law number is stated?",
                        "failure_class": "support_undercoverage",
                        "source_submission": "submission_v_a.json",
                        "exact_gold_recovered": True,
                    },
                    {
                        "question_id": "q2",
                        "question": "According to page 2 of the judgment, what claim number applied?",
                        "failure_class": "support_undercoverage",
                        "source_submission": "submission_v_b.json",
                        "exact_gold_recovered": True,
                    },
                    {
                        "question_id": "q3",
                        "question": "Which claimant appears on the title page of both CA 1/2024 and SCT 2/2024?",
                        "failure_class": "support_undercoverage",
                        "source_submission": "submission_v_c.json",
                        "exact_gold_recovered": False,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summaries = summarize_historical_support_families(
        opportunities_json=opportunities_json,
        known_qids={"q1"},
    )

    by_family = {item.family: item for item in summaries}
    assert by_family["single_doc_title_cover"].new_qid_count == 0
    assert by_family["explicit_page_two"].new_qid_count == 1
    assert by_family["comparison_title_party"].new_qid_count == 1


def test_cluster_historical_support_families_script_writes_reports(tmp_path: Path) -> None:
    opportunities_json = tmp_path / "opportunities.json"
    opportunities_json.write_text(
        json.dumps(
            {
                "opportunity_count": 1,
                "opportunities": [
                    {
                        "question_id": "q1",
                        "question": "According to page 2 of the judgment, what claim number applied?",
                        "failure_class": "support_undercoverage",
                        "source_submission": "submission_v_a.json",
                        "exact_gold_recovered": True,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    known_qids = tmp_path / "known_qids.txt"
    known_qids.write_text("q0\n", encoding="utf-8")
    out_json = tmp_path / "families.json"
    out_md = tmp_path / "families.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/cluster_historical_support_families.py",
            "--opportunities-json",
            str(opportunities_json),
            "--known-qids-file",
            str(known_qids),
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
    assert payload["family_count"] == 1
    assert payload["summaries"][0]["family"] == "explicit_page_two"
    report = out_md.read_text(encoding="utf-8")
    assert "# Historical Support Family Summary" in report
    assert "`explicit_page_two`" in report
