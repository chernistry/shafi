from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_analyze_portfolio_marginal_contribution_flags_plateau_items(tmp_path: Path) -> None:
    source = tmp_path / "search.json"
    source.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "qids": ["a", "b"],
                        "recommendation": "PROMISING",
                        "answer_changed_count": 0,
                        "retrieval_page_projection_changed_count": 2,
                        "candidate_page_p95": 4,
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
                        "qids": ["a", "c"],
                        "recommendation": "PROMISING",
                        "answer_changed_count": 0,
                        "retrieval_page_projection_changed_count": 3,
                        "candidate_page_p95": 4,
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
                        "qids": ["a"],
                        "recommendation": "PROMISING",
                        "answer_changed_count": 0,
                        "retrieval_page_projection_changed_count": 1,
                        "candidate_page_p95": 4,
                        "benchmark_trusted_baseline": 0.0,
                        "benchmark_trusted_candidate": 0.04,
                        "benchmark_all_baseline": 0.10,
                        "benchmark_all_candidate": 0.11,
                        "judge_pass_rate_baseline": 0.0,
                        "judge_pass_rate_candidate": 1.0,
                        "judge_grounding_baseline": 0.0,
                        "judge_grounding_candidate": 4.0,
                    },
                    {
                        "qids": ["b"],
                        "recommendation": "PROMISING",
                        "answer_changed_count": 0,
                        "retrieval_page_projection_changed_count": 1,
                        "candidate_page_p95": 4,
                        "benchmark_trusted_baseline": 0.0,
                        "benchmark_trusted_candidate": 0.02,
                        "benchmark_all_baseline": 0.10,
                        "benchmark_all_candidate": 0.105,
                        "judge_pass_rate_baseline": 0.0,
                        "judge_pass_rate_candidate": 0.0,
                        "judge_grounding_baseline": 0.0,
                        "judge_grounding_candidate": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_md = tmp_path / "out.md"
    out_json = tmp_path / "out.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_portfolio_marginal_contribution.py",
            "--source-json",
            str(source),
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
            "--max-answer-drift",
            "0",
            "--max-page-drift",
            "6",
            "--max-page-p95",
            "4",
        ],
        check=True,
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    reports = {row["qid"]: row for row in payload["item_reports"]}
    assert reports["a"]["is_plateau_item"] is False
    assert reports["b"]["is_plateau_item"] is True
    assert reports["c"]["is_plateau_item"] is True
    assert reports["a"]["trusted_delta_gain"] > 0.0
