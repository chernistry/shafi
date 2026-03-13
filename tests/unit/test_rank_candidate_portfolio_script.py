from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rank_candidate_portfolio_filters_and_orders(tmp_path: Path) -> None:
    source = tmp_path / "portfolio.json"
    source.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "qids": ["q1"],
                        "recommendation": "PROMISING",
                        "answer_changed_count": 0,
                        "retrieval_page_projection_changed_count": 2,
                        "candidate_page_p95": 4,
                        "improved_seed_cases": ["q1"],
                        "blindspot_improved_cases": [],
                        "blindspot_support_undercoverage_cases": [],
                        "benchmark_trusted_baseline": 0.0,
                        "benchmark_trusted_candidate": 0.05,
                        "benchmark_all_baseline": 0.10,
                        "benchmark_all_candidate": 0.12,
                        "judge_pass_rate_baseline": 0.0,
                        "judge_pass_rate_candidate": 1.0,
                        "judge_grounding_baseline": 0.0,
                        "judge_grounding_candidate": 5.0,
                    },
                    {
                        "qids": ["q2"],
                        "recommendation": "PROMISING",
                        "answer_changed_count": 0,
                        "retrieval_page_projection_changed_count": 3,
                        "candidate_page_p95": 4,
                        "improved_seed_cases": ["q2"],
                        "blindspot_improved_cases": ["q2"],
                        "blindspot_support_undercoverage_cases": ["q2"],
                        "benchmark_trusted_baseline": 0.0,
                        "benchmark_trusted_candidate": 0.05,
                        "benchmark_all_baseline": 0.10,
                        "benchmark_all_candidate": 0.12,
                        "judge_pass_rate_baseline": 0.0,
                        "judge_pass_rate_candidate": 1.0,
                        "judge_grounding_baseline": 0.0,
                        "judge_grounding_candidate": 5.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_md = tmp_path / "ranked.md"
    out_json = tmp_path / "ranked.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/rank_candidate_portfolio.py",
            "--source-json",
            str(source),
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
            "--max-page-drift",
            "6",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["filtered_count"] == 2
    assert payload["ranked_candidates"][0]["qids"] == ["q1"]
    assert payload["ranked_candidates"][0]["portfolio_role"] == "promotion_candidate"
    assert payload["ranked_candidates"][1]["portfolio_role"] == "promotion_candidate"
    assert payload["ranked_candidates"][1]["undercoverage_count"] == 1
    assert "`q1`" in out_md.read_text(encoding="utf-8")
