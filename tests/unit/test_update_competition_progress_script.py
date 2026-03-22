from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_update_competition_progress_module():
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("update_competition_progress")


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
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert "Competition Progress Snapshot" in report
    assert "Warm-up submissions used: `9 / 15`" in report
    assert "Warm-up submissions remaining: `6`" in report
    assert "Det lattice denominator: `420`" in report
    assert "`support_improved`: `4`" in report
    assert "Default: **NO SUBMIT**" in report


def test_update_competition_progress_script_builds_canonical_matrix(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    specs_json = tmp_path / "matrix_specs.json"
    history_md = tmp_path / "history.md"
    candidate_cycle = tmp_path / "cycle.json"
    production_mimic = tmp_path / "production_mimic.json"
    run_manifest = tmp_path / "run_manifest.json"
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
                        "git_commit": "unknown",
                        "baseline": "submission_v6_context_seed",
                        "candidate_label": "triad_f331_e0798_plus_dotted",
                        "lineage_confidence": "high",
                        "run_manifest_json": str(run_manifest),
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
    run_manifest.write_text(
        json.dumps(
            {
                "run_manifest": {
                    "candidate_label": "triad_f331_e0798_plus_dotted",
                    "fingerprint": "manifest1234567890",
                    "git": {"sha": "feedfacecafebeef"},
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
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    matrix_payload = json.loads(matrix_json.read_text(encoding="utf-8"))
    matrix_report = matrix_md.read_text(encoding="utf-8")
    assert matrix_payload["summary"]["current_default_decision"] == "local_ceiling_reached_hold_budget"
    assert matrix_payload["summary"]["current_public_best_label"] == "v6_public_exactness_champion"
    assert matrix_payload["summary"]["current_best_offline_label"] == "triad_f331_e0798_plus_dotted"
    rows = {row["label"]: row for row in matrix_payload["rows"]}
    assert rows["triad_f331_e0798_plus_dotted"]["run_manifest_status"] == "present"
    assert rows["triad_f331_e0798_plus_dotted"]["run_manifest_fingerprint"] == "manifest1234567890"
    assert rows["triad_f331_e0798_plus_dotted"]["git_commit"] == "feedfac"
    assert rows["v10_local_page_reranker_r1"]["run_manifest_status"] == "legacy_unknown"
    assert rows["triad_f331_e0798_plus_dotted"]["platform_like_rank_estimate"] is not None
    assert rows["triad_f331_e0798_plus_dotted"]["strict_rank_estimate"] is not None
    assert rows["triad_f331_e0798_plus_dotted"]["paranoid_rank_estimate"] is not None
    assert "# Competition Matrix" in matrix_report
    assert "paranoid_rank=`" in matrix_report
    assert "Current default decision: `local_ceiling_reached_hold_budget`" in matrix_report
    assert "Run manifest coverage:" in matrix_report
    assert "triad_f331_e0798_plus_dotted" in matrix_report
    assert "v6_public_exactness_champion" in matrix_report


def test_update_competition_progress_script_merges_partial_specs_with_defaults(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    specs_json = tmp_path / "matrix_specs.json"
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
                        "label": "v10_local_page_candidates_r1",
                        "date": "2026-03-13",
                        "status": "rejected",
                        "branch_class": "page_candidate_generator_family_aware",
                        "git_commit": "a2ef637",
                        "baseline": "triad_f331_e0798_plus_dotted",
                        "notes": "ticket 23 rejected",
                    }
                ]
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
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    matrix_payload = json.loads(matrix_json.read_text(encoding="utf-8"))
    matrix_report = matrix_md.read_text(encoding="utf-8")
    labels = {str(row["label"]) for row in matrix_payload["rows"]}

    assert "v6_public_exactness_champion" in labels
    assert "triad_f331_e0798_plus_dotted" in labels
    assert "v10_local_page_candidates_r1" in labels
    assert matrix_payload["summary"]["current_best_offline_label"] == "triad_f331_e0798_plus_dotted"
    assert "Current best offline candidate: `triad_f331_e0798_plus_dotted`" in matrix_report
    rejected_row = next(row for row in matrix_payload["rows"] if row["label"] == "v10_local_page_candidates_r1")
    assert rejected_row["run_manifest_status"] == "legacy_unknown"


def test_load_specs_autodiscovers_ticket_specs_and_applies_known_overrides(tmp_path: Path) -> None:
    module = _load_update_competition_progress_module()
    root = tmp_path
    research = root / ".sdd" / "researches"
    ticket_dir = research / "ticket23_page_candidates_r1_2026-03-13"
    ticket_dir.mkdir(parents=True)
    page_localizer_dir = research / "matrix_specs_with_ticket19_2026-03-13.json"
    prod_dir = research / "production_mimic_v10_local_page_localizer_r1_2026-03-13"
    prod_dir.mkdir(parents=True)
    (prod_dir / "production_mimic.json").write_text("{}", encoding="utf-8")
    page_localizer_dir.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "label": "v10_local_page_localizer_r1",
                        "status": "rejected",
                        "branch_class": "page_localizer_doc_to_page_rerank",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (research / "ticket23_page_candidates_r1_2026-03-13" / "spec.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "label": "v10_local_page_candidates_r1",
                        "status": "rejected",
                        "branch_class": "page_candidate_generator_family_aware",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    specs = module._load_specs(None, root=root)
    by_label = {str(spec["label"]): spec for spec in specs}

    assert "v10_local_page_candidates_r1" in by_label
    assert by_label["v10_local_page_candidates_r1"]["status"] == "rejected"
    assert by_label["v10_local_page_localizer_r1"]["production_mimic_json"] == str(
        root / ".sdd" / "researches" / "production_mimic_v10_local_page_localizer_r1_2026-03-13" / "production_mimic.json"
    )


def test_hydrate_row_prefers_row_specific_cycle_and_production_mimic(tmp_path: Path) -> None:
    module = _load_update_competition_progress_module()
    cycle_path = tmp_path / "cycle.json"
    cycle_path.write_text(
        json.dumps(
            {
                "ranked_candidates": [
                    {
                        "label": "v10_local_page_localizer_r1",
                        "strict_total_estimate": 0.77,
                        "paranoid_total_estimate": 0.38,
                        "hidden_g_trusted_delta": -0.04,
                        "hidden_g_all_delta": 0.03,
                        "answer_drift": 97,
                        "page_drift": 93,
                        "recommendation": "NO_SUBMIT",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    production_path = tmp_path / "production_mimic.json"
    production_path.write_text(
        json.dumps(
            {
                "production_mimic": {
                    "candidate_class": "v10_local_page_localizer_r1",
                    "lineage_confidence": "low",
                    "platform_like_total_estimate": 0.7595,
                    "strict_total_estimate": 0.7700,
                    "paranoid_total_estimate": 0.3830,
                    "judge": {
                        "pass_rate": 0.0,
                        "avg_grounding": 1.0,
                        "avg_accuracy": 1.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "label": "v10_local_page_localizer_r1",
        "status": "rejected",
        "candidate_label": "v10_local_page_localizer_r1",
        "candidate_cycle_json": str(cycle_path),
        "production_mimic_json": str(production_path),
    }

    row = module._hydrate_row(
        spec=spec,
        history_rows={},
        global_cycle_index={},
        cycle_index_cache={},
        global_production_mimic=None,
        supervisor_action=None,
    )

    assert row["answer_drift"] == 97
    assert row["page_drift"] == 93
    assert row["platform_like_total_estimate"] == 0.7595
    assert row["strict_total_estimate"] == 0.77
    assert row["paranoid_total_estimate"] == 0.383
    assert row["judge_pass_rate"] == 0.0
    assert row["run_manifest_status"] == "legacy_unknown"


def test_hydrate_row_blocks_active_candidates_without_run_manifest() -> None:
    module = _load_update_competition_progress_module()
    spec = {
        "label": "triad_f331_e0798_plus_dotted",
        "status": "ceiling",
        "candidate_label": "triad_f331_e0798_plus_dotted",
        "notes": "best offline candidate",
    }

    row = module._hydrate_row(
        spec=spec,
        history_rows={},
        global_cycle_index={},
        cycle_index_cache={},
        global_production_mimic=None,
        supervisor_action=None,
    )

    assert row["run_manifest_status"] == "missing_blocking"
    assert "manifest=missing_blocking" in str(row["notes"])


def test_candidate_fingerprint_script_marks_duplicate_and_matrix_links_it(tmp_path: Path) -> None:
    module = _load_update_competition_progress_module()
    submission_a = tmp_path / "submission_a.json"
    submission_b = tmp_path / "submission_b.json"
    raw_results_a = tmp_path / "raw_results_a.json"
    raw_results_b = tmp_path / "raw_results_b.json"
    fingerprint_a = tmp_path / "fingerprint_a.json"
    fingerprint_b = tmp_path / "fingerprint_b.json"

    submission_payload = {
        "architecture_summary": "test",
        "answers": [
            {
                "question_id": "q1",
                "answer": "Acme Ltd",
                "telemetry": {"model_name": "gpt-4o-mini"},
            },
            {
                "question_id": "q2",
                "answer": True,
                "telemetry": {"model_name": "gpt-4o-mini"},
            },
        ],
    }
    raw_results_payload = [
        {
            "telemetry": {
                "question_id": "q1",
                "answer_type": "name",
                "context_page_ids": ["docA_1", "docA_2"],
                "used_page_ids": ["docA_1"],
                "model_embed": "kanon-2-embedder",
                "model_rerank": "zerank-2",
                "model_llm": "gpt-4o-mini",
                "generation_mode": "strict",
                "llm_provider": "openai",
            }
        },
        {
            "telemetry": {
                "question_id": "q2",
                "answer_type": "boolean",
                "context_page_ids": ["docB_4"],
                "used_page_ids": ["docB_4"],
                "model_embed": "kanon-2-embedder",
                "model_rerank": "zerank-2",
                "model_llm": "gpt-4o-mini",
                "generation_mode": "strict",
                "llm_provider": "openai",
            }
        },
    ]

    submission_a.write_text(json.dumps(submission_payload), encoding="utf-8")
    submission_b.write_text(json.dumps(submission_payload), encoding="utf-8")
    raw_results_a.write_text(json.dumps(raw_results_payload), encoding="utf-8")
    raw_results_b.write_text(json.dumps(raw_results_payload), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/candidate_fingerprint.py",
            "--label",
            "candidate_alpha",
            "--submission-json",
            str(submission_a),
            "--raw-results-json",
            str(raw_results_a),
            "--out-json",
            str(fingerprint_a),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/candidate_fingerprint.py",
            "--label",
            "candidate_beta",
            "--submission-json",
            str(submission_b),
            "--raw-results-json",
            str(raw_results_b),
            "--known-candidate-fingerprint-json",
            str(fingerprint_a),
            "--out-json",
            str(fingerprint_b),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    duplicate_payload = json.loads(fingerprint_b.read_text(encoding="utf-8"))["candidate_fingerprint"]
    assert duplicate_payload["should_skip"] is True
    assert duplicate_payload["duplicate_of_label"] == "candidate_alpha"
    assert duplicate_payload["answers_hash"]
    assert duplicate_payload["used_pages_hash"]
    assert duplicate_payload["context_pages_hash"]
    assert duplicate_payload["route_map_hash"]

    row = module._hydrate_row(
        spec={
            "label": "candidate_beta",
            "status": "candidate",
            "candidate_fingerprint_json": str(fingerprint_b),
        },
        history_rows={},
        global_cycle_index={},
        cycle_index_cache={},
        global_production_mimic=None,
        supervisor_action=None,
    )

    assert row["candidate_fingerprint"] == duplicate_payload["fingerprint"]
    assert row["duplicate_of_label"] == "candidate_alpha"
    assert "duplicate_of=candidate_alpha" in str(row["notes"])


def test_update_competition_progress_recomputes_stale_rank_estimates_from_latest_leaderboard(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    specs_json = tmp_path / "matrix_specs.json"
    production_mimic = tmp_path / "production_mimic.json"
    matrix_json = tmp_path / "competition_matrix.json"
    matrix_md = tmp_path / "competition_matrix.md"

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","Leader","0.900000","1","0.700000","0.950000","1","1.0500","80","2","2026-03-12T14:02:54"\n'
        '"2","Team2","0.840000","1","0.700000","0.900000","1","1.0500","81","2","2026-03-12T14:02:55"\n'
        '"3","Team3","0.820000","1","0.700000","0.890000","1","1.0500","82","2","2026-03-12T14:02:56"\n'
        '"4","Team4","0.810000","1","0.700000","0.880000","1","1.0500","83","2","2026-03-12T14:02:57"\n'
        '"5","Team5","0.800000","1","0.700000","0.870000","1","1.0500","84","2","2026-03-12T14:02:58"\n'
        '"6","Team6","0.781000","1","0.700000","0.860000","1","1.0500","85","2","2026-03-12T14:02:59"\n'
        '"7","Team7","0.761000","1","0.700000","0.850000","1","1.0500","86","2","2026-03-12T14:03:00"\n'
        '"8","Borderline","0.753713","0.985714","0.673333","0.904631","0.996","0.9378","961","3","2026-03-12T18:45:46"\n'
        '"9","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-13T14:56:17"\n',
        encoding="utf-8",
    )
    production_mimic.write_text(
        json.dumps(
            {
                "production_mimic": {
                    "candidate_class": "triad_rank_refresh_test",
                    "lineage_confidence": "medium",
                    "platform_like_total_estimate": 0.757142,
                    "platform_like_rank_estimate": 7,
                    "strict_total_estimate": 0.760607,
                    "strict_rank_estimate": 7,
                    "paranoid_total_estimate": 0.74156,
                    "paranoid_rank_estimate": 8,
                }
            }
        ),
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
                        "external_version": "v6",
                        "external_rank": 8,
                        "external_total": 0.74156,
                        "git_commit": "unknown",
                        "baseline": "v5",
                        "lineage_confidence": "high",
                    },
                    {
                        "label": "triad_rank_refresh_test",
                        "date": "2026-03-13",
                        "status": "ceiling",
                        "branch_class": "combined_small_diff_ceiling",
                        "candidate_label": "triad_rank_refresh_test",
                        "git_commit": "unknown",
                        "baseline": "submission_v6_context_seed",
                        "lineage_confidence": "medium",
                        "production_mimic_json": str(production_mimic),
                    },
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
            "--matrix-json-out",
            str(matrix_json),
            "--matrix-md-out",
            str(matrix_md),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    rows = json.loads(matrix_json.read_text(encoding="utf-8"))["rows"]
    by_label = {row["label"]: row for row in rows}
    assert by_label["v6_public_exactness_champion"]["external_rank"] == 9
    assert by_label["triad_rank_refresh_test"]["platform_like_rank_estimate"] == 8
    assert by_label["triad_rank_refresh_test"]["strict_rank_estimate"] == 8
    assert by_label["triad_rank_refresh_test"]["paranoid_rank_estimate"] == 9
