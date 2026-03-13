from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

REPO_ROOT = "/Users/sasha/IdeaProjects/.codex-worktrees/rag_challenge-main"


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
        cwd=REPO_ROOT,
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
        cwd=REPO_ROOT,
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
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "- Action: `no_submit_continue_offline`" in report
    assert "not lineage-safe for the required champion baseline" in report


def test_competition_supervisor_marks_small_diff_ceiling_reached_for_rank_one(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    backlog_dir = tmp_path / "backlog"
    ledger_json = tmp_path / "ledger.json"
    ceiling_json = tmp_path / "ceiling.json"
    out = tmp_path / "supervisor.md"
    runs_json = tmp_path / "runs.json"

    backlog_dir.mkdir()
    (backlog_dir / "32-a.md").write_text("# a\n", encoding="utf-8")
    (backlog_dir / "43-b.md").write_text("# b\n", encoding="utf-8")

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","TopTeam","0.867424","1","0.666667","0.921596","0.996","1.05","80","6","2026-03-12T14:02:54.255238"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17.082289"\n',
        encoding="utf-8",
    )
    ledger_json.write_text(json.dumps({"experiments": []}), encoding="utf-8")
    ceiling_json.write_text(
        json.dumps(
            {
                "ranked_candidates": [
                    {
                        "label": "triad_f331_e0798_plus_dotted",
                        "paranoid_total_estimate": 0.752000,
                        "strict_total_estimate": 0.760607,
                        "upper_total_estimate": 0.780940,
                        "paranoid_rank_estimate": 7,
                        "strict_rank_estimate": 7,
                        "upper_rank_estimate": 6,
                    }
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
            "--candidate-ceiling-cycle",
            str(ceiling_json),
            "--target-rank",
            "1",
            "--out",
            str(out),
            "--runs-json",
            str(runs_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "- Action: `small_diff_ceiling_reached`" in report
    assert "- Paranoid rank estimate: `7`" in report
    assert "- Upper rank estimate: `6`" in report


def test_competition_supervisor_keeps_offline_loop_alive_when_blindspot_gains_exist(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    backlog_dir = tmp_path / "backlog"
    ledger_json = tmp_path / "ledger.json"
    ceiling_json = tmp_path / "ceiling.json"
    out = tmp_path / "supervisor.md"
    runs_json = tmp_path / "runs.json"

    backlog_dir.mkdir()
    (backlog_dir / "32-a.md").write_text("# a\n", encoding="utf-8")
    (backlog_dir / "43-b.md").write_text("# b\n", encoding="utf-8")

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","TopTeam","0.867424","1","0.666667","0.921596","0.996","1.05","80","6","2026-03-12T14:02:54.255238"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17.082289"\n',
        encoding="utf-8",
    )
    ledger_json.write_text(json.dumps({"experiments": []}), encoding="utf-8")
    ceiling_json.write_text(
        json.dumps(
            {
                "ranked_candidates": [
                    {
                        "label": "projection-gap-leader",
                        "paranoid_total_estimate": 0.752000,
                        "strict_total_estimate": 0.760607,
                        "upper_total_estimate": 0.780940,
                        "paranoid_rank_estimate": 7,
                        "strict_rank_estimate": 7,
                        "upper_rank_estimate": 6,
                        "blindspot_improved_case_count": 5,
                        "blindspot_support_undercoverage_case_count": 4,
                    }
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
            "--candidate-ceiling-cycle",
            str(ceiling_json),
            "--target-rank",
            "1",
            "--out",
            str(out),
            "--runs-json",
            str(runs_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "- Action: `no_submit_continue_offline`" in report
    assert "- Blindspot improved cases: `5`" in report
    assert "benchmark-blind page-family gains are still active" in report


def test_competition_supervisor_reports_non_improving_paranoid_estimate(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    backlog_dir = tmp_path / "backlog"
    ledger_json = tmp_path / "ledger.json"
    ceiling_json = tmp_path / "ceiling.json"
    out = tmp_path / "supervisor.md"
    runs_json = tmp_path / "runs.json"

    backlog_dir.mkdir()
    (backlog_dir / "32-a.md").write_text("# a\n", encoding="utf-8")

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","TopTeam","0.867424","1","0.666667","0.921596","0.996","1.05","80","6","2026-03-12T14:02:54.255238"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17.082289"\n',
        encoding="utf-8",
    )
    ledger_json.write_text(json.dumps({"experiments": []}), encoding="utf-8")
    ceiling_json.write_text(
        json.dumps(
            {
                "ranked_candidates": [
                    {
                        "label": "pessimistic-ceiling",
                        "paranoid_total_estimate": 0.741000,
                        "strict_total_estimate": 0.760607,
                        "upper_total_estimate": 0.780940,
                        "paranoid_rank_estimate": 8,
                        "strict_rank_estimate": 7,
                        "upper_rank_estimate": 6,
                    }
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
            "--candidate-ceiling-cycle",
            str(ceiling_json),
            "--target-rank",
            "1",
            "--out",
            str(out),
            "--runs-json",
            str(runs_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "paranoid estimate is non-improving relative to the current public baseline" in report


def test_competition_supervisor_marks_ceiling_when_no_actionable_signal_families_remain(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    backlog_dir = tmp_path / "backlog"
    ledger_json = tmp_path / "ledger.json"
    ceiling_json = tmp_path / "ceiling.json"
    remaining_signal_json = tmp_path / "remaining_signal.json"
    out = tmp_path / "supervisor.md"
    runs_json = tmp_path / "runs.json"

    backlog_dir.mkdir()
    (backlog_dir / "32-a.md").write_text("# a\n", encoding="utf-8")

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","TopTeam","0.867424","1","0.666667","0.921596","0.996","1.05","80","6","2026-03-12T14:02:54.255238"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17.082289"\n',
        encoding="utf-8",
    )
    ledger_json.write_text(json.dumps({"experiments": []}), encoding="utf-8")
    ceiling_json.write_text(
        json.dumps(
            {
                "ranked_candidates": [
                    {
                        "label": "projection-gap-leader",
                        "paranoid_total_estimate": 0.744607,
                        "strict_total_estimate": 0.760607,
                        "upper_total_estimate": 0.780940,
                        "paranoid_rank_estimate": 8,
                        "strict_rank_estimate": 7,
                        "upper_rank_estimate": 6,
                        "blindspot_improved_case_count": 5,
                        "blindspot_support_undercoverage_case_count": 4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    remaining_signal_json.write_text(
        json.dumps(
            {
                "summaries": [
                    {"family": "comparison_party_metadata", "likely_actionable": False},
                    {"family": "single_doc_title_cover", "likely_actionable": False},
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
            "--candidate-ceiling-cycle",
            str(ceiling_json),
            "--remaining-signal-json",
            str(remaining_signal_json),
            "--target-rank",
            "1",
            "--out",
            str(out),
            "--runs-json",
            str(runs_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "- Action: `small_diff_ceiling_reached`" in report
    assert "actionable_families=0/2" in report
    assert "no likely actionable family-level signals remain" in report


def test_competition_supervisor_holds_budget_when_no_actionable_families_and_alt_branch_fails(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    backlog_dir = tmp_path / "backlog"
    ledger_json = tmp_path / "ledger.json"
    ceiling_json = tmp_path / "ceiling.json"
    remaining_signal_json = tmp_path / "remaining_signal.json"
    alternative_gate_json = tmp_path / "alternative_gate.json"
    out = tmp_path / "supervisor.md"
    runs_json = tmp_path / "runs.json"

    backlog_dir.mkdir()
    (backlog_dir / "32-a.md").write_text("# a\n", encoding="utf-8")

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","TopTeam","0.867424","1","0.666667","0.921596","0.996","1.05","80","6","2026-03-12T14:02:54.255238"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17.082289"\n',
        encoding="utf-8",
    )
    ledger_json.write_text(json.dumps({"experiments": []}), encoding="utf-8")
    ceiling_json.write_text(
        json.dumps(
            {
                "ranked_candidates": [
                    {
                        "label": "projection-gap-leader",
                        "paranoid_total_estimate": 0.744607,
                        "strict_total_estimate": 0.760607,
                        "upper_total_estimate": 0.780940,
                        "paranoid_rank_estimate": 8,
                        "strict_rank_estimate": 7,
                        "upper_rank_estimate": 6,
                        "blindspot_improved_case_count": 5,
                        "blindspot_support_undercoverage_case_count": 4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    remaining_signal_json.write_text(
        json.dumps(
            {
                "summaries": [
                    {"family": "comparison_party_metadata", "likely_actionable": False},
                    {"family": "single_doc_title_cover", "likely_actionable": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    alternative_gate_json.write_text(
        json.dumps(
            {
                "label": "embeddinggemma-fullcollection",
                "recommendation": "NO_SUBMIT",
                "answer_changed_count": 14,
                "retrieval_page_projection_changed_count": 50,
                "benchmark_trusted_baseline": 0.0425,
                "benchmark_trusted_candidate": 0.0227,
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
            "--candidate-ceiling-cycle",
            str(ceiling_json),
            "--remaining-signal-json",
            str(remaining_signal_json),
            "--alternative-gate-json",
            str(alternative_gate_json),
            "--target-rank",
            "1",
            "--out",
            str(out),
            "--runs-json",
            str(runs_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "- Action: `local_ceiling_reached_hold_budget`" in report
    assert "actionable_families=0/2" in report
    assert "no tested alternative branch class currently clears the bounded offline gate" in report
    assert "embeddinggemma-fullcollection" in report


def test_competition_supervisor_requires_manual_lineage_review_when_candidate_is_otherwise_ready(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    backlog_dir = tmp_path / "backlog"
    production_mimic = tmp_path / "production_mimic.json"
    out = tmp_path / "supervisor.md"
    runs_json = tmp_path / "runs.json"

    backlog_dir.mkdir()
    (backlog_dir / "18-manual-submit-readiness-decision.md").write_text("# 18\n", encoding="utf-8")

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","Leader","0.867424","1","0.666667","0.921596","0.996","1.05","80","6","2026-03-12T14:02:54"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17"\n',
        encoding="utf-8",
    )
    production_mimic.write_text(
        json.dumps(
            {
                "production_mimic": {
                    "candidate_class": "triad_f331_e0798_plus_dotted",
                    "lineage_confidence": "medium",
                    "platform_like_total_estimate": 0.760000,
                    "strict_total_estimate": 0.768000,
                    "paranoid_total_estimate": 0.752000,
                    "hidden_g_trusted": {"delta": 0.0425},
                    "exactness": {"still_mismatched_incorrect_qids": []},
                    "judge": {"judge_timeout_or_failure": False},
                    "submit_eligibility": False,
                }
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
            "--production-mimic-json",
            str(production_mimic),
            "--out",
            str(out),
            "--runs-json",
            str(runs_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "- Action: `candidate_requires_manual_lineage_review`" in report
    assert "candidate clears strict local bar except for lineage confidence" in report


def test_competition_supervisor_marks_candidate_ready_for_manual_submit_review(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    backlog_dir = tmp_path / "backlog"
    production_mimic = tmp_path / "production_mimic.json"
    out = tmp_path / "supervisor.md"
    runs_json = tmp_path / "runs.json"

    backlog_dir.mkdir()
    (backlog_dir / "18-manual-submit-readiness-decision.md").write_text("# 18\n", encoding="utf-8")

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","Leader","0.867424","1","0.666667","0.921596","0.996","1.05","80","6","2026-03-12T14:02:54"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17"\n',
        encoding="utf-8",
    )
    production_mimic.write_text(
        json.dumps(
            {
                "production_mimic": {
                    "candidate_class": "triad_f331_e0798_plus_dotted",
                    "lineage_confidence": "high",
                    "platform_like_total_estimate": 0.760000,
                    "strict_total_estimate": 0.768000,
                    "paranoid_total_estimate": 0.752000,
                    "hidden_g_trusted": {"delta": 0.0425},
                    "exactness": {"still_mismatched_incorrect_qids": []},
                    "judge": {"judge_timeout_or_failure": False, "pass_rate": 1.0},
                    "submit_eligibility": True,
                }
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
            "--production-mimic-json",
            str(production_mimic),
            "--out",
            str(out),
            "--runs-json",
            str(runs_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "- Action: `candidate_ready_for_manual_submit_review`" in report
    assert "requires only explicit user approval" in report


def test_competition_supervisor_reports_branch_freeze_and_private_only_demotions(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    backlog_dir = tmp_path / "backlog"
    ledger_json = tmp_path / "ledger.json"
    ceiling_json = tmp_path / "ceiling.json"
    out = tmp_path / "supervisor.md"
    runs_json = tmp_path / "runs.json"

    backlog_dir.mkdir()
    (backlog_dir / "11-export-bounded-miss-pack.md").write_text("# 11\n", encoding="utf-8")
    (backlog_dir / "12-cohere-fast-page-falsifier.md").write_text("# 12\n", encoding="utf-8")

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","Leader","0.867424","1","0.666667","0.921596","0.996","1.05","80","6","2026-03-12T14:02:54"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17"\n',
        encoding="utf-8",
    )
    ledger_json.write_text(json.dumps({"experiments": []}), encoding="utf-8")
    ceiling_json.write_text(
        json.dumps(
            {
                "branch_class_summary": [
                    {
                        "branch_class": "small_diff_support_rider",
                        "status": "frozen",
                        "active_before_march17": False,
                        "best_label": "dead-rider",
                        "best_upper_rank_estimate": 6,
                        "reason": "proven dead mechanism: small-diff support rider ceiling",
                    },
                    {
                        "branch_class": "ocr_visual_rerank",
                        "status": "private_only",
                        "active_before_march17": False,
                        "best_label": "ocr-vision",
                        "best_upper_rank_estimate": 2,
                        "reason": "reserved for after March 18 / private-phase work",
                    },
                    {
                        "branch_class": "doc_page_rerank_core",
                        "status": "active",
                        "active_before_march17": True,
                        "best_label": "page-core",
                        "best_upper_rank_estimate": 1,
                        "reason": "active pre-March-17 class",
                    },
                ],
                "ranked_candidates": [
                    {
                        "label": "ocr-vision",
                        "branch_class": "ocr_visual_rerank",
                        "branch_status": "private_only",
                        "paranoid_total_estimate": 0.752000,
                        "strict_total_estimate": 0.760607,
                        "upper_total_estimate": 0.780940,
                        "paranoid_rank_estimate": 7,
                        "strict_rank_estimate": 7,
                        "upper_rank_estimate": 2,
                        "blindspot_improved_case_count": 1,
                        "blindspot_support_undercoverage_case_count": 0,
                    }
                ],
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
            "--candidate-ceiling-cycle",
            str(ceiling_json),
            "--target-rank",
            "1",
            "--out",
            str(out),
            "--runs-json",
            str(runs_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "## Branch Freeze" in report
    assert "small_diff_support_rider" in report
    assert "ocr_visual_rerank" in report
    assert "private-only branch classes will not displace active pre-March-17 work" in report
    assert "best ranked ceiling row is demoted by branch policy" in report
