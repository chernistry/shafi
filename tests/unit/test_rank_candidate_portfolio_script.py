from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


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
                        "retrieval_page_projection_changed_count": 7,
                        "candidate_page_p95": 4,
                        "benchmark_trusted_baseline": 0.0,
                        "benchmark_trusted_candidate": 0.08,
                        "benchmark_all_baseline": 0.10,
                        "benchmark_all_candidate": 0.13,
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
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["filtered_count"] == 1
    assert payload["ranked_candidates"][0]["qids"] == ["q1"]
    assert "`q1`" in out_md.read_text(encoding="utf-8")
