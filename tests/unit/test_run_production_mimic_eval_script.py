from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_run_production_mimic_eval_script_writes_ranked_output(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    candidate_cycle = tmp_path / "cycle.json"
    exactness = tmp_path / "exactness.json"
    equivalence = tmp_path / "equivalence.json"
    history = tmp_path / "history.json"
    scaffold = tmp_path / "scaffold.json"
    raw_results = tmp_path / "raw_results.json"
    cheap_eval = tmp_path / "cheap_eval.json"
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
                }
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
            "--scaffold-json",
            str(scaffold),
            "--page-trace-json",
            str(page_trace),
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
    report = payload["production_mimic"]
    markdown = out_md.read_text(encoding="utf-8")

    assert report["lineage_confidence"] == "high"
    assert report["support_shape"]["weak_same_doc_anchor_case_count"] == 1
    assert report["page_trace"]["cases_scored"] == 1
    assert report["page_trace"]["failure_stage_counts"] == {"wrong_page_used_same_doc": 1}
    assert report["platform_like_rank_estimate"] >= 1
    assert report["strict_rank_estimate"] >= 1
    assert report["paranoid_rank_estimate"] >= 1
    assert "Production-Mimic Local Eval" in markdown
    assert "triad_f331_e0798_plus_dotted" in markdown
    assert "## Page Trace" in markdown
