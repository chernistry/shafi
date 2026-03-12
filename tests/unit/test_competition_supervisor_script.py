from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_competition_supervisor_prefers_no_submit_when_budget_is_almost_spent(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    backlog_dir = tmp_path / "backlog"
    ledger_json = tmp_path / "ledger.json"
    exactness_report = tmp_path / "exactness_report.json"
    out = tmp_path / "supervisor.md"
    runs_json = tmp_path / "runs.json"

    backlog_dir.mkdir()
    (backlog_dir / "31-a.md").write_text("# a\n", encoding="utf-8")
    (backlog_dir / "32-b.md").write_text("# b\n", encoding="utf-8")
    (backlog_dir / "39-c.md").write_text("# c\n", encoding="utf-8")

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","TopTeam","0.900000","1","0.7","0.95","1","1","100","2","2026-03-12T10:00:00"\n'
        '"2","Tzur Labs","0.756000","0.95","0.65","0.84","1","1","200","9","2026-03-12T11:00:00"\n',
        encoding="utf-8",
    )
    ledger_json.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "label": "anchor-branch",
                        "recommendation": "EXPERIMENTAL_NO_SUBMIT",
                        "answer_changed_count": 12,
                        "retrieval_page_projection_changed_count": 30,
                        "benchmark_trusted_baseline": 0.02,
                        "benchmark_trusted_candidate": 0.02,
                    }
                ]
            }
        ),
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
            "scripts/competition_supervisor.py",
            "--leaderboard",
            str(leaderboard),
            "--team",
            "Tzur Labs",
            "--backlog-dir",
            str(backlog_dir),
            "--ledger-json",
            str(ledger_json),
            "--exactness-report",
            str(exactness_report),
            "--out",
            str(out),
            "--runs-json",
            str(runs_json),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "Competition Supervisor Report" in report
    assert "- Submissions remaining: `1`" in report
    assert "- Action: `no_submit_continue_offline`" in report
    assert "lineage proof is missing" in report
    assert "- `31-a.md`" in report
    assert "- `32-b.md`" in report

    runs = json.loads(runs_json.read_text(encoding="utf-8"))
    assert runs["runs"][0]["decision"]["action"] == "no_submit_continue_offline"


def test_competition_supervisor_accepts_nested_page_metrics_identical_with_lineage_proof(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    backlog_dir = tmp_path / "backlog"
    ledger_json = tmp_path / "ledger.json"
    exactness_report = tmp_path / "exactness_report.json"
    equivalence_json = tmp_path / "equivalence.json"
    out = tmp_path / "supervisor.md"
    runs_json = tmp_path / "runs.json"

    backlog_dir.mkdir()
    (backlog_dir / "31-a.md").write_text("# a\n", encoding="utf-8")

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","TopTeam","0.900000","1","0.7","0.95","1","1","100","2","2026-03-12T10:00:00"\n'
        '"2","Tzur Labs","0.756000","0.95","0.65","0.84","1","1","200","9","2026-03-12T11:00:00"\n',
        encoding="utf-8",
    )
    ledger_json.write_text(json.dumps({"experiments": []}), encoding="utf-8")
    exactness_report.write_text(
        json.dumps(
            {
                "answer_changed_count": 2,
                "page_changed_count": 0,
                "hidden_g_page_benchmark": {"page_metrics_identical": True},
            }
        ),
        encoding="utf-8",
    )
    equivalence_json.write_text(
        json.dumps(
            {
                "safe_baselines": [
                    "/tmp/submission_v6_context_seed.json",
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/competition_supervisor.py",
            "--leaderboard",
            str(leaderboard),
            "--team",
            "Tzur Labs",
            "--backlog-dir",
            str(backlog_dir),
            "--ledger-json",
            str(ledger_json),
            "--exactness-report",
            str(exactness_report),
            "--equivalence-json",
            str(equivalence_json),
            "--required-safe-baseline-substring",
            "submission_v6_context_seed.json",
            "--out",
            str(out),
            "--runs-json",
            str(runs_json),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "- Action: `exactness_only_candidate`" in report
    assert "submission_v6_context_seed.json" in report


def test_competition_supervisor_rejects_lineage_safe_fallback_for_wrong_baseline(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    backlog_dir = tmp_path / "backlog"
    ledger_json = tmp_path / "ledger.json"
    exactness_report = tmp_path / "exactness_report.json"
    equivalence_json = tmp_path / "equivalence.json"
    out = tmp_path / "supervisor.md"
    runs_json = tmp_path / "runs.json"

    backlog_dir.mkdir()
    (backlog_dir / "31-a.md").write_text("# a\n", encoding="utf-8")

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","TopTeam","0.900000","1","0.7","0.95","1","1","100","2","2026-03-12T10:00:00"\n'
        '"2","Tzur Labs","0.756000","0.95","0.65","0.84","1","1","200","9","2026-03-12T11:00:00"\n',
        encoding="utf-8",
    )
    ledger_json.write_text(json.dumps({"experiments": []}), encoding="utf-8")
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
    equivalence_json.write_text(
        json.dumps(
            {
                "safe_baselines": [
                    "/tmp/submission_v4_anchor_lineage.json",
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/competition_supervisor.py",
            "--leaderboard",
            str(leaderboard),
            "--team",
            "Tzur Labs",
            "--backlog-dir",
            str(backlog_dir),
            "--ledger-json",
            str(ledger_json),
            "--exactness-report",
            str(exactness_report),
            "--equivalence-json",
            str(equivalence_json),
            "--required-safe-baseline-substring",
            "submission_v6_context_seed.json",
            "--out",
            str(out),
            "--runs-json",
            str(runs_json),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "- Action: `no_submit_continue_offline`" in report
    assert "not lineage-safe for the required champion baseline" in report
