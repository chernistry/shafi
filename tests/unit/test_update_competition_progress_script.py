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


def test_update_competition_progress_script_builds_canonical_matrix(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    specs_json = tmp_path / "matrix_specs.json"
    history_md = tmp_path / "history.md"
    candidate_cycle = tmp_path / "cycle.json"
    production_mimic = tmp_path / "production_mimic.json"
    supervisor_runs = tmp_path / "runs.json"
    matrix_json = tmp_path / "competition_matrix.json"
    matrix_md = tmp_path / "competition_matrix.md"

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","Leader","0.867424","1","0.666667","0.921596","0.996","1.05","80","6","2026-03-12T14:02:54"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17"\n',
        encoding="utf-8",
    )
    specs_json.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "label": "v6_public_exactness_champion",
                        "date": "2026-03-12",
                        "status": "submitted",
                        "branch_class": "answer_only_exactness",
                        "git_commit": "unknown",
                        "baseline": "v5",
                        "external_version": "v6",
                        "lineage_confidence": "low",
                        "notes": "public champion",
                    },
                    {
                        "label": "triad_f331_e0798_plus_dotted",
                        "date": "2026-03-13",
                        "status": "ceiling",
                        "branch_class": "combined_small_diff_ceiling",
                        "git_commit": "0343e02",
                        "baseline": "submission_v6_context_seed",
                        "candidate_label": "triad_f331_e0798_plus_dotted",
                        "lineage_confidence": "high",
                        "notes": "best offline candidate",
                    },
                    {
                        "label": "v10_local_page_reranker_r1",
                        "date": "2026-03-13",
                        "status": "rejected",
                        "branch_class": "real_page_reranker_bounded_candidates",
                        "git_commit": "a9ad8a7",
                        "baseline": "triad_f331_e0798_plus_dotted",
                        "candidate_label": "v10_local_page_reranker_r1",
                        "lineage_confidence": "medium",
                        "notes": "rejected bounded page reranker candidate",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    history_md.write_text(
        "| Version | Strategy | Det | Asst | G | Total | Result |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| **v6 (champion)** | **Dotted suffix fix** | **0.971** | **0.693** | **0.801** | **0.742** | **BEST** |\n",
        encoding="utf-8",
    )
    candidate_cycle.write_text(
        json.dumps(
            {
                "ranked_candidates": [
                    {
                        "label": "triad_f331_e0798_plus_dotted",
                        "strict_total_estimate": 0.760607,
                        "paranoid_total_estimate": 0.744607,
                        "hidden_g_trusted_delta": 0.0425,
                        "hidden_g_all_delta": 0.0206,
                        "answer_drift": 2,
                        "page_drift": 4,
                        "recommendation": "PROMISING",
                    },
                    {
                        "label": "v10_local_page_reranker_r1",
                        "strict_total_estimate": 0.728266,
                        "paranoid_total_estimate": 0.717766,
                        "hidden_g_trusted_delta": -0.0198,
                        "hidden_g_all_delta": -0.0144,
                        "answer_drift": 0,
                        "page_drift": 2,
                        "recommendation": "NO_SUBMIT",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    production_mimic.write_text(
        json.dumps(
            {
                "production_mimic": {
                    "candidate_class": "triad_f331_e0798_plus_dotted",
                    "lineage_confidence": "high",
                    "platform_like_total_estimate": 0.748,
                    "strict_total_estimate": 0.756,
                    "paranoid_total_estimate": 0.744,
                    "judge": {
                        "pass_rate": 1.0,
                        "avg_grounding": 5.0,
                        "avg_accuracy": 5.0,
                    },
                    "hidden_g_trusted": {"delta": 0.0425},
                    "submit_eligibility": False,
                }
            }
        ),
        encoding="utf-8",
    )
    supervisor_runs.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "decision": {
                            "action": "local_ceiling_reached_hold_budget",
                        }
                    }
                ]
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
            "--matrix-row-specs-json",
            str(specs_json),
            "--history-md",
            str(history_md),
            "--candidate-ceiling-cycle",
            str(candidate_cycle),
            "--production-mimic-json",
            str(production_mimic),
            "--supervisor-runs-json",
            str(supervisor_runs),
            "--matrix-json-out",
            str(matrix_json),
            "--matrix-md-out",
            str(matrix_md),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    matrix_payload = json.loads(matrix_json.read_text(encoding="utf-8"))
    matrix_report = matrix_md.read_text(encoding="utf-8")
    assert matrix_payload["summary"]["current_default_decision"] == "local_ceiling_reached_hold_budget"
    assert matrix_payload["summary"]["current_public_best_label"] == "v6_public_exactness_champion"
    assert matrix_payload["summary"]["current_best_offline_label"] == "triad_f331_e0798_plus_dotted"
    assert "# Competition Matrix" in matrix_report
    assert "Current default decision: `local_ceiling_reached_hold_budget`" in matrix_report
    assert "triad_f331_e0798_plus_dotted" in matrix_report
    assert "v6_public_exactness_champion" in matrix_report
