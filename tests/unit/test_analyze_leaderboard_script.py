from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_analyze_leaderboard_script_reports_gap_math(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    out = tmp_path / "report.md"
    json_out = tmp_path / "summary.json"
    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","TopTeam","0.900000","1","0.7","0.95","1","1","100","2","2026-03-12T10:00:00"\n'
        '"2","MyTeam","0.756000","0.95","0.65","0.84","1","1","200","9","2026-03-12T11:00:00"\n',
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_leaderboard.py",
            "--leaderboard",
            str(leaderboard),
            "--team",
            "MyTeam",
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
    assert "Leaderboard Geometry Report" in report
    assert "Assumed formula: `S = 0.7*Det + 0.3*Asst`" in report
    assert "- Team: `MyTeam`" in report
    assert "- Rank `1` `TopTeam` total `0.900000`" in report
    assert "Public #1 is not reachable by exactness-only" in report

    summary = json_out.read_text(encoding="utf-8")
    assert '"team_name": "MyTeam"' in summary
    assert '"submissions": 9' in summary
