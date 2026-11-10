from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_run_production_mimic_eval_script_writes_ranked_output(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    candidate_cycle = tmp_path / "cycle.json"
    exactness = tmp_path / "exactness.json"
    equivalence = tmp_path / "equivalence.json"
    history = tmp_path / "history.json"
    scaffold = tmp_path / "scaffold.json"
    raw_results = tmp_path / "raw_results.json"
    cheap_eval = tmp_path / "cheap_eval.json"
    strict_eval = tmp_path / "strict_eval.json"
    page_trace = tmp_path / "page_trace.json"
    out_json = tmp_path / "production_mimic.json"
    out_md = tmp_path / "production_mimic.md"

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","Leader","0.860000","1","0.70","0.920000","0.996","1.05","100","6","2026-03-12T10:00:00"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17"\n',
        encoding="utf-8",
    )
    candidate_cycle.write_text(
        json.dumps(
            {
                "ranked_candidates": [
                    {
                        "label": "triad_f331_e0798_plus_dotted",
                        "branch_class": "combined_small_diff_ceiling",
                        "strict_total_estimate": 0.7800,
                        "upper_total_estimate": 0.8000,
                        "paranoid_total_estimate": 0.7700,
                        "hidden_g_trusted_delta": 0.0425,
                        "lineage_ok": True,
                        "page_drift": 4,
                        "raw_results": str(raw_results),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    exactness.write_text(
        json.dumps(
            {
                "resolved_incorrect_qids": ["43f77", "f950"],
                "still_mismatched_incorrect_qids": [],
            }
        ),
        encoding="utf-8",
    )
    equivalence.write_text(json.dumps({"safe_baselines": ["/tmp/submission_v6_context_seed.json"]}), encoding="utf-8")
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "question": "Compare the parties in these cases.",
                        "support_shape_class": "comparison",
                        "support_shape_requirements": {"requires_title_anchor": True},
                        "minimal_required_support_pages": ["docA_1", "docB_1"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_results.write_text(
        json.dumps(
            [
                {
                    "case": {"question": "Compare the parties in these cases."},
                    "telemetry": {
                        "question_id": "q1",
                        "used_page_ids": ["docA_3", "docB_2"],
                        "retrieved_page_ids": ["docA_1", "docA_3", "docB_1", "docB_2"],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    history.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "external_total": 0.72,
                        "strict_total_estimate": 0.74,
                        "paranoid_total_estimate": 0.73,
                        "platform_like_total_estimate": 0.735,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cheap_eval.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "answer": "Alpha",
                        "cited_ids": ["docA:1:0:a"],
                        "context_fingerprint": "ctx-shared",
                        "judge_prompt_version": "judge-v1",
                    }
                ],
                "production_mimic": {
                    "eval": {
                        "citation_coverage": 0.9,
                        "answer_type_format_compliance": 1.0,
                        "grounding_g_score_beta_2_5": 0.81,
                    },
                    "judge": {
                        "cases": 2,
                        "pass_rate": 1.0,
                        "avg_accuracy": 5.0,
                        "avg_grounding": 5.0,
                        "judge_failures": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    strict_eval.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "answer": "Alpha",
                        "cited_ids": ["docA:1:0:a"],
                        "context_fingerprint": "ctx-shared",
                        "judge_prompt_version": "judge-v1",
                    }
                ],
                "production_mimic": {
                    "eval": {
                        "citation_coverage": 0.92,
                        "answer_type_format_compliance": 1.0,
                        "grounding_g_score_beta_2_5": 0.8,
                    },
                    "judge": {
                        "cases": 1,
                        "pass_rate": 0.5,
                        "avg_accuracy": 4.8,
                        "avg_grounding": 4.7,
                        "judge_failures": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    page_trace.write_text(
        json.dumps(
            {
                "summary": {
                    "cases_scored": 1,
                    "trusted_case_count": 1,
                    "gold_in_retrieved_count": 1,
                    "gold_in_reranked_count": 0,
                    "gold_in_used_count": 0,
                    "false_positive_case_count": 1,
                    "failure_stage_counts": {"wrong_page_used_same_doc": 1},
                    "stage_examples": {"wrong_page_used_same_doc": ["q1"]},
                    "explained_ratio": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_production_mimic_eval.py",
            "--leaderboard",
            str(leaderboard),
            "--team",
            "Tzur Labs",
            "--candidate-cycle-json",
            str(candidate_cycle),
            "--candidate-label",
            "triad_f331_e0798_plus_dotted",
            "--exactness-json",
            str(exactness),
            "--equivalence-json",
            str(equivalence),
            "--history-json",
            str(history),
            "--cheap-eval-json",
            str(cheap_eval),
            "--strict-eval-json",
            str(strict_eval),
            "--scaffold-json",
            str(scaffold),
            "--page-trace-json",
            str(page_trace),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    report = payload["production_mimic"]
    markdown = out_md.read_text(encoding="utf-8")

    assert report["lineage_confidence"] == "high"
    assert report["judge"]["strict_requested"] is True
    assert report["judge"]["strict_used"] is True
    assert report["judge"]["cache"]["shared_cache_key_count"] == 1
    assert report["judge"]["cache"]["cache_hit_count"] == 1
    assert report["judge_penalties"]["disagreement_penalty"] == 0.002
    assert report["support_shape"]["weak_same_doc_anchor_case_count"] == 1
    assert report["page_trace"]["cases_scored"] == 1
    assert report["page_trace"]["failure_stage_counts"] == {"wrong_page_used_same_doc": 1}
    assert report["platform_like_rank_estimate"] >= 1
    assert report["strict_rank_estimate"] >= 1
    assert report["paranoid_rank_estimate"] >= 1
    assert "Production-Mimic Local Eval" in markdown
    assert "triad_f331_e0798_plus_dotted" in markdown
    assert "shared_cache_key_count" in markdown
    assert "## Judge Penalties" in markdown
    assert "## Page Trace" in markdown


def test_run_production_mimic_eval_skips_strict_judge_when_candidate_is_not_near_promote(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    candidate_cycle = tmp_path / "cycle.json"
    cheap_eval = tmp_path / "cheap_eval.json"
    strict_eval = tmp_path / "strict_eval.json"
    raw_results = tmp_path / "raw_results.json"
    out_json = tmp_path / "production_mimic.json"
    out_md = tmp_path / "production_mimic.md"

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","Leader","0.860000","1","0.70","0.920000","0.996","1.05","100","6","2026-03-12T10:00:00"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17"\n',
        encoding="utf-8",
    )
    candidate_cycle.write_text(
        json.dumps(
            {
                "ranked_candidates": [
                    {
                        "label": "dead_branch",
                        "branch_class": "dead_branch",
                        "strict_total_estimate": 0.70,
                        "upper_total_estimate": 0.71,
                        "paranoid_total_estimate": 0.69,
                        "hidden_g_trusted_delta": 0.0,
                        "lineage_ok": True,
                        "raw_results": str(raw_results),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cheap_eval.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "answer": "Alpha",
                        "cited_ids": ["docA:1:0:a"],
                        "context_fingerprint": "ctx-shared",
                        "judge_prompt_version": "judge-v1",
                    }
                ],
                "summary": {
                    "citation_coverage": 1.0,
                    "answer_type_format_compliance": 1.0,
                    "grounding_g_score_beta_2_5": 0.9,
                    "judge": {
                        "cases": 1,
                        "pass_rate": 1.0,
                        "avg_accuracy": 5.0,
                        "avg_grounding": 5.0,
                        "judge_failures": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    raw_results.write_text(json.dumps([]), encoding="utf-8")
    strict_eval.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "answer": "Alpha",
                        "cited_ids": ["docA:1:0:a"],
                        "context_fingerprint": "ctx-shared",
                        "judge_prompt_version": "judge-v1",
                    }
                ],
                "summary": {
                    "citation_coverage": 0.5,
                    "answer_type_format_compliance": 1.0,
                    "grounding_g_score_beta_2_5": 0.4,
                    "judge": {
                        "cases": 1,
                        "pass_rate": 0.0,
                        "avg_accuracy": 1.0,
                        "avg_grounding": 1.0,
                        "judge_failures": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_production_mimic_eval.py",
            "--leaderboard",
            str(leaderboard),
            "--team",
            "Tzur Labs",
            "--candidate-cycle-json",
            str(candidate_cycle),
            "--candidate-label",
            "dead_branch",
            "--cheap-eval-json",
            str(cheap_eval),
            "--strict-eval-json",
            str(strict_eval),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    report = payload["production_mimic"]

    assert report["judge"]["strict_requested"] is True
    assert report["judge"]["strict_used"] is False
    assert report["judge"]["strict_skip_reason"] == "candidate_not_near_promote"
    assert report["judge"]["strict_present"] is False
    assert report["judge"]["pass_rate"] == 1.0
    assert report["judge_penalties"]["disagreement_penalty"] == 0.0


def test_run_production_mimic_eval_batch_mode_is_byte_identical_with_parallel_workers(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    candidate_cycle = tmp_path / "cycle.json"
    raw_results = tmp_path / "raw_results.json"
    cheap_eval = tmp_path / "cheap_eval.json"
    strict_eval = tmp_path / "strict_eval.json"
    batch_a = tmp_path / "batch_a"
    batch_b = tmp_path / "batch_b"

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","Leader","0.860000","1","0.70","0.920000","0.996","1.05","100","6","2026-03-12T10:00:00"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17"\n',
        encoding="utf-8",
    )
    raw_results.write_text(json.dumps([]), encoding="utf-8")
    candidate_cycle.write_text(
        json.dumps(
            {
                "ranked_candidates": [
                    {
                        "label": "cand_b",
                        "branch_class": "offline_pack",
                        "strict_total_estimate": 0.75,
                        "upper_total_estimate": 0.76,
                        "paranoid_total_estimate": 0.74,
                        "hidden_g_trusted_delta": 0.01,
                        "lineage_ok": True,
                        "raw_results": str(raw_results),
                    },
                    {
                        "label": "cand_a",
                        "branch_class": "offline_pack",
                        "strict_total_estimate": 0.77,
                        "upper_total_estimate": 0.78,
                        "paranoid_total_estimate": 0.76,
                        "hidden_g_trusted_delta": 0.02,
                        "lineage_ok": True,
                        "raw_results": str(raw_results),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cheap_eval.write_text(
        json.dumps(
            {
                "summary": {
                    "citation_coverage": 1.0,
                    "answer_type_format_compliance": 1.0,
                    "grounding_g_score_beta_2_5": 0.9,
                    "judge": {
                        "cases": 1,
                        "pass_rate": 1.0,
                        "avg_accuracy": 5.0,
                        "avg_grounding": 5.0,
                        "judge_failures": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    strict_eval.write_text(
        json.dumps(
            {
                "summary": {
                    "citation_coverage": 1.0,
                    "answer_type_format_compliance": 1.0,
                    "grounding_g_score_beta_2_5": 0.9,
                    "judge": {
                        "cases": 1,
                        "pass_rate": 1.0,
                        "avg_accuracy": 5.0,
                        "avg_grounding": 5.0,
                        "judge_failures": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    base_cmd = [
        sys.executable,
        "scripts/run_production_mimic_eval.py",
        "--leaderboard",
        str(leaderboard),
        "--team",
        "Tzur Labs",
        "--candidate-cycle-json",
        str(candidate_cycle),
        "--candidate-label",
        "cand_b",
        "--candidate-label",
        "cand_a",
        "--cheap-eval-json",
        str(cheap_eval),
        "--strict-eval-json",
        str(strict_eval),
        "--batch-out-dir",
        str(batch_a),
        "--parallel-workers",
        "2",
    ]
    subprocess.run(
        base_cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    rerun_cmd = list(base_cmd)
    rerun_cmd[rerun_cmd.index(str(batch_a))] = str(batch_b)
    subprocess.run(
        rerun_cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    batch_a_summary = (batch_a / "batch_summary.json").read_bytes()
    batch_b_summary = (batch_b / "batch_summary.json").read_bytes()
    assert batch_a_summary == batch_b_summary
    assert (batch_a / "cand_a" / "production_mimic.json").read_bytes() == (
        batch_b / "cand_a" / "production_mimic.json"
    ).read_bytes()
    assert (batch_a / "cand_b" / "production_mimic.json").read_bytes() == (
        batch_b / "cand_b" / "production_mimic.json"
    ).read_bytes()

    payload = json.loads(batch_a_summary.decode("utf-8"))
    assert payload["parallel_workers_used"] == 2
    assert payload["canonical_candidate_build_concurrency"] == 1
    assert payload["deterministic_inputs_sorted"] is True
    assert [row["candidate_label"] for row in payload["artifacts"]] == ["cand_a", "cand_b"]
