from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CyclePaths:
    audit_json: Path
    audit_md: Path
    seed_qids: Path
    portfolio_json: Path
    combo_dir: Path
    rank_md: Path
    rank_json: Path
    marginal_md: Path
    marginal_json: Path
    summary_json: Path


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _resolve(root: Path, raw: str | Path) -> Path:
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (root / value).resolve()


def _cycle_paths(*, out_dir: Path, label: str) -> CyclePaths:
    combo_dir = out_dir / f"combo_search_{label}"
    return CyclePaths(
        audit_json=out_dir / f"comparison_title_audit_{label}.json",
        audit_md=out_dir / f"comparison_title_audit_{label}.md",
        seed_qids=out_dir / f"comparison_title_seed_qids_{label}.txt",
        portfolio_json=out_dir / f"comparison_title_portfolio_{label}.json",
        combo_dir=combo_dir,
        rank_md=out_dir / f"comparison_title_portfolio_rank_{label}.md",
        rank_json=out_dir / f"comparison_title_portfolio_rank_{label}.json",
        marginal_md=out_dir / f"comparison_title_portfolio_marginal_{label}.md",
        marginal_json=out_dir / f"comparison_title_portfolio_marginal_{label}.json",
        summary_json=out_dir / f"comparison_title_cycle_{label}.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the comparison/title-page offline portfolio cycle end-to-end.",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--truth-audit", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline-submission", required=True)
    parser.add_argument("--baseline-raw-results", required=True)
    parser.add_argument("--baseline-preflight", required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--source-raw-results", required=True)
    parser.add_argument("--single-swap-dir", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-portfolio-json", default=None)
    parser.add_argument("--baseline-eval-json", default=None)
    parser.add_argument("--include-recommendation", action="append", default=["PROMISING"])
    parser.add_argument("--max-new-items", type=int, default=0)
    parser.add_argument("--require-judge-non-inferior", action="store_true")
    parser.add_argument("--require-judge-pass-improvement", action="store_true")
    parser.add_argument("--combo-min-size", type=int, default=2)
    parser.add_argument("--combo-max-size", type=int, default=4)
    parser.add_argument("--combo-top-k", type=int, default=40)
    parser.add_argument("--combo-judge-top-k", type=int, default=20)
    parser.add_argument("--rank-top-k", type=int, default=12)
    parser.add_argument("--max-answer-drift", type=int, default=0)
    parser.add_argument("--max-page-drift", type=int, default=6)
    parser.add_argument("--max-page-p95", type=int, default=4)
    return parser.parse_args()


def _build_commands(*, root: Path, args: argparse.Namespace, paths: CyclePaths) -> list[list[str]]:
    questions = _resolve(root, args.questions)
    truth_audit = _resolve(root, args.truth_audit)
    baseline_submission = _resolve(root, args.baseline_submission)
    baseline_raw_results = _resolve(root, args.baseline_raw_results)
    baseline_preflight = _resolve(root, args.baseline_preflight)
    source_raw_results = _resolve(root, args.source_raw_results)
    single_swap_dir = _resolve(root, args.single_swap_dir)
    benchmark = _resolve(root, args.benchmark)
    docs_dir = _resolve(root, args.docs_dir)

    commands: list[list[str]] = []

    audit_cmd = [
        sys.executable,
        "scripts/audit_comparison_title_page_candidates.py",
        "--questions",
        str(questions),
        "--truth-audit",
        str(truth_audit),
        "--baseline-raw-results",
        str(baseline_raw_results),
        "--baseline-label",
        args.baseline_label,
        "--source-raw-results",
        f"{args.source_label}={source_raw_results}",
        "--out-json",
        str(paths.audit_json),
        "--out-md",
        str(paths.audit_md),
        "--out-seed-qids",
        str(paths.seed_qids),
    ]
    commands.append(audit_cmd)

    portfolio_cmd = [
        sys.executable,
        "scripts/build_comparison_support_portfolio.py",
        "--comparison-audit-json",
        str(paths.audit_json),
        "--single-swap-dir",
        str(single_swap_dir),
        "--out",
        str(paths.portfolio_json),
        "--max-new-items",
        str(int(args.max_new_items)),
    ]
    for recommendation in args.include_recommendation:
        portfolio_cmd.extend(["--include-recommendation", str(recommendation)])
    if args.base_portfolio_json:
        portfolio_cmd.extend(["--base-portfolio-json", str(_resolve(root, args.base_portfolio_json))])
    if args.baseline_eval_json:
        portfolio_cmd.extend(["--baseline-eval-json", str(_resolve(root, args.baseline_eval_json))])
    if bool(args.require_judge_non_inferior):
        portfolio_cmd.append("--require-judge-non-inferior")
    if bool(args.require_judge_pass_improvement):
        portfolio_cmd.append("--require-judge-pass-improvement")
    commands.append(portfolio_cmd)

    combo_cmd = [
        sys.executable,
        "scripts/search_portfolio_support_combos.py",
        "--baseline-label",
        args.baseline_label,
        "--baseline-submission",
        str(baseline_submission),
        "--baseline-raw-results",
        str(baseline_raw_results),
        "--baseline-preflight",
        str(baseline_preflight),
        "--portfolio-json",
        str(paths.portfolio_json),
        "--benchmark",
        str(benchmark),
        "--questions",
        str(questions),
        "--docs-dir",
        str(docs_dir),
        "--out-dir",
        str(paths.combo_dir),
        "--min-size",
        str(int(args.combo_min_size)),
        "--max-size",
        str(int(args.combo_max_size)),
        "--top-k",
        str(int(args.combo_top_k)),
        "--judge-top-k",
        str(int(args.combo_judge_top_k)),
    ]
    commands.append(combo_cmd)

    combo_results_json = paths.combo_dir / "portfolio_support_combo_search.json"
    rank_cmd = [
        sys.executable,
        "scripts/rank_candidate_portfolio.py",
        "--source-json",
        str(combo_results_json),
        "--out-md",
        str(paths.rank_md),
        "--out-json",
        str(paths.rank_json),
        "--top-k",
        str(int(args.rank_top_k)),
        "--max-answer-drift",
        str(int(args.max_answer_drift)),
        "--max-page-drift",
        str(int(args.max_page_drift)),
        "--max-page-p95",
        str(int(args.max_page_p95)),
    ]
    commands.append(rank_cmd)

    marginal_cmd = [
        sys.executable,
        "scripts/analyze_portfolio_marginal_contribution.py",
        "--source-json",
        str(combo_results_json),
        "--out-md",
        str(paths.marginal_md),
        "--out-json",
        str(paths.marginal_json),
        "--max-answer-drift",
        str(int(args.max_answer_drift)),
        "--max-page-drift",
        str(int(args.max_page_drift)),
        "--max-page-p95",
        str(int(args.max_page_p95)),
    ]
    commands.append(marginal_cmd)
    return commands


def _write_summary(*, path: Path, args: argparse.Namespace, cycle_paths: CyclePaths) -> None:
    payload = {
        "label": args.label,
        "baseline_label": args.baseline_label,
        "source_label": args.source_label,
        "audit_json": str(cycle_paths.audit_json),
        "audit_md": str(cycle_paths.audit_md),
        "seed_qids": str(cycle_paths.seed_qids),
        "portfolio_json": str(cycle_paths.portfolio_json),
        "combo_dir": str(cycle_paths.combo_dir),
        "combo_results_json": str(cycle_paths.combo_dir / "portfolio_support_combo_search.json"),
        "combo_results_md": str(cycle_paths.combo_dir / "portfolio_support_combo_search.md"),
        "rank_json": str(cycle_paths.rank_json),
        "rank_md": str(cycle_paths.rank_md),
        "marginal_json": str(cycle_paths.marginal_json),
        "marginal_md": str(cycle_paths.marginal_md),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = _resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cycle_paths = _cycle_paths(out_dir=out_dir, label=args.label)

    commands = _build_commands(root=root, args=args, paths=cycle_paths)
    for command in commands:
        _run(command, cwd=root)

    _write_summary(path=cycle_paths.summary_json, args=args, cycle_paths=cycle_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
