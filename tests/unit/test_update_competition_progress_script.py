from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_update_competition_progress_script_renders_budget_and_estimate(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    ledger_json = tmp_path / "ledger.json"
    scoring_json = tmp_path / "scoring.json"
    anchor_slice_json = tmp_path / "anchor.json"
    out = tmp_path / "progress.md"

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","TopTeam","0.867424","1","0.666667","0.921596","0.996","1.05","80","6","2026-03-12T14:02:54"\n'
        '"2","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17"\n',
        encoding="utf-8",
    )
    ledger_json.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "label": "iter7",
                        "recommendation": "EXPERIMENTAL_NO_SUBMIT",
                        "answer_changed_count": 12,
                        "retrieval_page_projection_changed_count": 22,
                        "benchmark_trusted_baseline": 0.0,
                        "benchmark_trusted_candidate": 0.02,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    scoring_json.write_text(
        json.dumps(
            {
                "det_lattice_denominator": 420,
                "asst_lattice_denominator": 150,
                "delta_total_per_full_deterministic_answer": 0.0083,
                "delta_total_per_free_text_step": 0.0017,
                "exactness_estimate": {
                    "answer_changed_count": 2,
                    "page_changed_count": 0,
                    "strict_upper_bound_total_if_all_answer_changes_are_real": 0.75816,
                },
            }
        ),
        encoding="utf-8",
    )
    anchor_slice_json.write_text(
        json.dumps(
            {
                "status_counts": {
                    "support_improved": 4,
                    "mixed_or_no_hit": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/update_competition_progress.py",
            "--leaderboard",
            str(leaderboard),
            "--team",
            "Tzur Labs",
            "--ledger-json",
            str(ledger_json),
            "--scoring-json",
            str(scoring_json),
            "--anchor-slice-json",
            str(anchor_slice_json),
            "--out",
            str(out),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "Competition Progress Snapshot" in report
    assert "Warm-up submissions remaining: `1`" in report
    assert "Det lattice denominator: `420`" in report
    assert "`support_improved`: `4`" in report
    assert "Default: **NO SUBMIT**" in report
