from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _write_eval(path: Path, *, pass_rate: float, grounding: float) -> None:
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "judge": {
                        "pass_rate": pass_rate,
                        "avg_grounding": grounding,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_build_comparison_support_portfolio_merges_base_and_promising_single_swaps(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q_keep",
                        "compare_kind": "party_overlap",
                        "recommendation": "PROMISING",
                        "minimal_required_page1_count": 2,
                        "baseline_used_page1_doc_hits": 0,
                    },
                    {
                        "question_id": "q_skip",
                        "compare_kind": "judge_overlap",
                        "recommendation": "WATCH",
                        "minimal_required_page1_count": 2,
                        "baseline_used_page1_doc_hits": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    single_swap_dir = tmp_path / "single"
    single_swap_dir.mkdir()
    for prefix in ("submission", "raw_results", "preflight_summary"):
        (single_swap_dir / f"{prefix}_single_swap_q_keep.json").write_text("{}", encoding="utf-8")
    _write_eval(single_swap_dir / "eval_candidate_debug_single_swap_q_keep.json", pass_rate=1.0, grounding=4.0)
    baseline_eval = tmp_path / "baseline_eval.json"
    _write_eval(baseline_eval, pass_rate=0.0, grounding=1.0)

    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            [
                {
                    "qid": "q_existing",
                    "label": "existing",
                    "submission_path": "/tmp/existing_submission.json",
                    "raw_results_path": "/tmp/existing_raw.json",
                    "preflight_path": "/tmp/existing_preflight.json",
                    "notes": "existing note",
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_comparison_support_portfolio.py",
            "--comparison-audit-json",
            str(audit),
            "--single-swap-dir",
            str(single_swap_dir),
            "--base-portfolio-json",
            str(base),
            "--baseline-eval-json",
            str(baseline_eval),
            "--out",
            str(out),
        ],
        check=True,
        cwd="/Users/sasha/IdeaProjects/personal_projects/shafi",
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [row["qid"] for row in payload] == ["q_existing", "q_keep"]
    assert "judge_pass_delta=+1.0000" in payload[1]["notes"]
    assert "missing_used_page1_docs=2" in payload[1]["notes"]
    assert payload[1]["label"].startswith("q_keep_")
