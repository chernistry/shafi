from __future__ import annotations

import json
from argparse import Namespace
from typing import TYPE_CHECKING

from scripts.run_comparison_title_portfolio_cycle import (
    _build_commands,
    _cycle_paths,
    _write_summary,
)

if TYPE_CHECKING:
    from pathlib import Path


def _args() -> Namespace:
    return Namespace(
        label="iter13",
        questions="platform_runs/warmup/questions.json",
        truth_audit="platform_runs/warmup/truth_audit_scaffold.json",
        baseline_label="v6_context_seed",
        baseline_submission="platform_runs/warmup/submission_v6_context_seed.json",
        baseline_raw_results="platform_runs/warmup/raw_results_v6_context_seed.json",
        baseline_preflight="platform_runs/warmup/preflight_summary_v6_context_seed.json",
        source_label="iter13_partyclaimants",
        source_raw_results="platform_runs/warmup/raw_results_v_anchor_rebuild_iter13_partyclaimants.json",
        single_swap_dir=".sdd/researches/scan_single_support_swaps_iter13",
        benchmark="platform_runs/warmup/trusted_hidden_g_benchmark.json",
        docs_dir="platform_runs/warmup/documents",
        out_dir=".sdd/researches/comparison_title_cycle_iter13_2026-03-13",
        base_portfolio_json=None,
        baseline_eval_json=None,
        include_recommendation=["PROMISING", "WATCH"],
        max_new_items=0,
        require_judge_non_inferior=False,
        require_judge_pass_improvement=False,
        combo_min_size=2,
        combo_max_size=4,
        combo_top_k=30,
        combo_judge_top_k=12,
        rank_top_k=10,
        max_answer_drift=0,
        max_page_drift=6,
        max_page_p95=4,
    )


def test_build_commands_covers_full_cycle(tmp_path: Path) -> None:
    root = tmp_path
    out_dir = tmp_path / "out"
    paths = _cycle_paths(out_dir=out_dir, label="iter13")
    args = _args()

    commands = _build_commands(root=root, args=args, paths=paths)

    assert len(commands) == 5
    assert commands[0][1].endswith("audit_comparison_title_page_candidates.py")
    assert "--out-seed-qids" in commands[0]
    assert commands[1][1].endswith("build_comparison_support_portfolio.py")
    assert commands[2][1].endswith("search_portfolio_support_combos.py")
    assert commands[3][1].endswith("rank_candidate_portfolio.py")
    assert commands[4][1].endswith("analyze_portfolio_marginal_contribution.py")
    assert str(paths.combo_dir / "portfolio_support_combo_search.json") in commands[3]
    assert str(paths.combo_dir / "portfolio_support_combo_search.json") in commands[4]


def test_write_summary_marks_no_submit_policy(tmp_path: Path) -> None:
    paths = _cycle_paths(out_dir=tmp_path / "out", label="iter13")
    _write_summary(path=paths.summary_json, args=_args(), cycle_paths=paths)

    payload = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert payload["label"] == "iter13"
    assert payload["baseline_label"] == "v6_context_seed"
    assert payload["combo_results_md"].endswith("portfolio_support_combo_search.md")
    assert payload["submission_policy"] == "NO_SUBMIT_WITHOUT_USER_APPROVAL"
