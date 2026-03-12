from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_reverse_engineer_platform_scoring_reports_lattice_and_exactness_bound(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    history = tmp_path / "history.md"
    exactness_report = tmp_path / "exactness.json"
    out = tmp_path / "report.md"
    json_out = tmp_path / "summary.json"

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","TopTeam","0.867424","1","0.666667","0.921596","0.996","1.05","80","6","2026-03-12T14:02:54"\n'
        '"2","MyTeam","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17"\n'
        '"3","Other","0.753713","0.985714","0.673333","0.904631","0.996","0.9378","961","3","2026-03-12T18:45:46"\n',
        encoding="utf-8",
    )
    history.write_text(
        "\n".join(
            [
                "| Version | Strategy | Det | Asst | G | Total | Result |",
                "|---------|----------|-----|------|---|-------|--------|",
                "| v5 | Baseline | 0.943 | 0.667 | 0.801 | 0.718 | OK |",
                "| v6 | Dotted suffix fix | 0.971 | 0.693 | 0.801 | 0.742 | BEST |",
                "| v7 | ALL context pages | 0.971 | 0.647 | 0.608 | 0.554 | G CRASH |",
                "| v9 | Reranked pages | 0.971 | 0.700 | 0.654 | 0.606 | G CRASH |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    exactness_report.write_text(
        json.dumps(
            {
                "answer_changed_count": 2,
                "page_changed_count": 0,
                "page_metrics_identical": True,
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/reverse_engineer_platform_scoring.py",
            "--leaderboard",
            str(leaderboard),
            "--team",
            "MyTeam",
            "--history-md",
            str(history),
            "--exactness-report",
            str(exactness_report),
            "--out",
            str(out),
            "--json-out",
            str(json_out),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "Platform Scoring Reverse-Engineering Report" in report
    assert "Asst appears to lie on a 1/150 lattice" in report
    assert "strict upper-bound total if every answer delta is real" in report

    summary = json.loads(json_out.read_text(encoding="utf-8"))
    assert summary["asst_lattice_denominator"] == 150
    assert summary["exactness_estimate"]["answer_changed_count"] == 2

