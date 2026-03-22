from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_build_portfolio_from_single_support_scan_filters_on_judge_improvement(tmp_path: Path) -> None:
    scan_json = tmp_path / "scan.json"
    single_swap_dir = tmp_path / "single_swaps"
    out = tmp_path / "portfolio.json"
    single_swap_dir.mkdir()

    for qid in ("good", "flat"):
        for prefix in ("submission_single_swap_", "raw_results_single_swap_", "preflight_summary_single_swap_"):
            (single_swap_dir / f"{prefix}{qid}.json").write_text("{}", encoding="utf-8")

    scan_json.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "question_id": "good",
                        "recommendation": "PROMISING",
                        "judge_pass_rate_baseline": 0.0,
                        "judge_pass_rate_candidate": 1.0,
                        "judge_grounding_baseline": 0.0,
                        "judge_grounding_candidate": 5.0,
                        "retrieval_page_projection_changed_count": 1,
                    },
                    {
                        "question_id": "flat",
                        "recommendation": "PROMISING",
                        "judge_pass_rate_baseline": 0.0,
                        "judge_pass_rate_candidate": 0.0,
                        "judge_grounding_baseline": 0.0,
                        "judge_grounding_candidate": 0.0,
                        "retrieval_page_projection_changed_count": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_portfolio_from_single_support_scan.py",
            "--scan-json",
            str(scan_json),
            "--single-swap-dir",
            str(single_swap_dir),
            "--out",
            str(out),
            "--require-judge-pass-improvement",
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/shafi",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [row["qid"] for row in payload] == ["good"]
