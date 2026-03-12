from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _artifact_path(root: Path, suffix: str, stem: str, extension: str = ".json") -> Path:
    return root / "platform_runs" / "warmup" / f"{stem}_{suffix}{extension}"


def _write_cycle_summary(
    *,
    path: Path,
    suffix: str,
    candidate_submission: Path,
    candidate_preflight: Path,
    candidate_raw_results: Path,
    gate_report: Path,
    anchor_slice_report: Path | None,
    scoring_report: Path | None,
    supervisor_report: Path,
    exactness_queue_report: Path,
) -> None:
    summary = {
        "artifact_suffix": suffix,
        "candidate_submission": str(candidate_submission),
        "candidate_preflight": str(candidate_preflight),
        "candidate_raw_results": str(candidate_raw_results),
        "gate_report": str(gate_report),
        "anchor_slice_report": str(anchor_slice_report) if anchor_slice_report is not None else None,
        "scoring_report": str(scoring_report) if scoring_report is not None else None,
        "supervisor_report": str(supervisor_report),
        "exactness_queue_report": str(exactness_queue_report),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one offline no-submit competition hypothesis cycle.",
    )
    parser.add_argument("--artifact-suffix", required=True)
    parser.add_argument("--baseline-label", default="v6_context_seed")
    parser.add_argument("--baseline-submission", required=True)
    parser.add_argument("--baseline-raw-results", required=True)
    parser.add_argument("--baseline-scaffold", required=True)
    parser.add_argument(
        "--audit-scaffold",
        default="platform_runs/warmup/truth_audit_scaffold.json",
        help="Canonical strict audit scaffold used for seed-slice/gate comparisons.",
    )
    parser.add_argument("--baseline-preflight", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--seed-qid", action="append", default=[])
    parser.add_argument("--seed-qids-file", default=None)
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--exactness-report", required=True)
    parser.add_argument("--history-md", default=".sdd/researches/getting_to_1st_001.md")
    parser.add_argument("--backlog-dir", required=True)
    parser.add_argument("--ledger-json", required=True)
    parser.add_argument("--warmup-budget", type=int, default=10)
    parser.add_argument("--query-concurrency", type=int, default=1)
    parser.add_argument("--research-dir", default=".sdd/researches")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    research_dir = (root / args.research_dir).resolve()
    research_dir.mkdir(parents=True, exist_ok=True)
    audit_scaffold = (root / args.audit_scaffold).resolve() if not Path(args.audit_scaffold).is_absolute() else Path(args.audit_scaffold)

    env = dict(os.environ)
    env.setdefault("QDRANT_URL", "http://localhost:6333")

    seed_qids: list[str] = []
    for raw in args.seed_qid:
        text = str(raw).strip()
        if text:
            seed_qids.append(text)
    if args.seed_qids_file is not None:
        qids_path = (root / args.seed_qids_file).resolve() if not Path(args.seed_qids_file).is_absolute() else Path(args.seed_qids_file)
        for line in qids_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                seed_qids.append(text)
    seed_qids = list(dict.fromkeys(seed_qids))
    if not seed_qids:
        raise ValueError("At least one seed qid or --seed-qids-file is required")

    platform_cmd = [
        sys.executable,
        "-m",
        "rag_challenge.submission.platform",
        "--skip-ingest",
        "--artifact-suffix",
        args.artifact_suffix,
        "--query-concurrency",
        str(args.query_concurrency),
    ]
    _run(platform_cmd, cwd=root, env=env)

    candidate_submission = _artifact_path(root, args.artifact_suffix, "submission")
    candidate_preflight = _artifact_path(root, args.artifact_suffix, "preflight_summary")
    candidate_raw_results = _artifact_path(root, args.artifact_suffix, "raw_results")
    candidate_scaffold = _artifact_path(root, args.artifact_suffix, "truth_audit_scaffold")

    gate_report = research_dir / f"experiment_gate_{args.artifact_suffix}_vs_{args.baseline_label}.md"
    gate_cmd = [
        sys.executable,
        "scripts/run_experiment_gate.py",
        "--label",
        args.artifact_suffix,
        "--baseline-label",
        args.baseline_label,
        "--baseline-submission",
        args.baseline_submission,
        "--candidate-submission",
        str(candidate_submission),
        "--baseline-raw-results",
        args.baseline_raw_results,
        "--candidate-raw-results",
        str(candidate_raw_results),
        "--benchmark",
        args.benchmark,
        "--scaffold",
        str(audit_scaffold),
        "--candidate-scaffold",
        str(candidate_scaffold),
        "--baseline-preflight",
        args.baseline_preflight,
        "--candidate-preflight",
        str(candidate_preflight),
        "--out",
        str(gate_report),
        "--ledger-json",
        args.ledger_json,
    ]
    for qid in seed_qids:
        gate_cmd.extend(["--seed-qid", qid])
    _run(gate_cmd, cwd=root, env=env)

    anchor_slice_report = research_dir / f"anchor_slice_{args.artifact_suffix}_vs_{args.baseline_label}.md"
    anchor_slice_json = research_dir / f"anchor_slice_{args.artifact_suffix}_vs_{args.baseline_label}.json"
    anchor_slice_cmd = [
        sys.executable,
        "scripts/build_anchor_slice_report.py",
        "--baseline-label",
        args.baseline_label,
        "--candidate-label",
        args.artifact_suffix,
        "--baseline-submission",
        args.baseline_submission,
        "--candidate-submission",
        str(candidate_submission),
        "--baseline-raw-results",
        args.baseline_raw_results,
        "--candidate-raw-results",
        str(candidate_raw_results),
        "--baseline-scaffold",
        str(audit_scaffold),
        "--candidate-scaffold",
        str(candidate_scaffold),
        "--out",
        str(anchor_slice_report),
        "--json-out",
        str(anchor_slice_json),
    ]
    for qid in seed_qids:
        anchor_slice_cmd.extend(["--qid", qid])
    _run(anchor_slice_cmd, cwd=root, env=env)

    exactness_queue_report = research_dir / "exactness_review_queue_2026-03-12.md"
    exactness_queue_cmd = [
        sys.executable,
        "scripts/build_exactness_review_queue.py",
        "--scaffold",
        str(root / "platform_runs" / "warmup" / "truth_audit_scaffold.json"),
        "--limit",
        "15",
        "--out",
        str(exactness_queue_report),
    ]
    _run(exactness_queue_cmd, cwd=root, env=env)

    scoring_report = research_dir / f"platform_scoring_{args.artifact_suffix}.md"
    scoring_json = research_dir / f"platform_scoring_{args.artifact_suffix}.json"
    scoring_cmd = [
        sys.executable,
        "scripts/reverse_engineer_platform_scoring.py",
        "--leaderboard",
        args.leaderboard,
        "--team",
        args.team,
        "--history-md",
        args.history_md,
        "--exactness-report",
        args.exactness_report,
        "--out",
        str(scoring_report),
        "--json-out",
        str(scoring_json),
    ]
    _run(scoring_cmd, cwd=root, env=env)

    supervisor_report = research_dir / "competition_supervisor_2026-03-12.md"
    supervisor_runs = research_dir / "competition_supervisor_runs.json"
    supervisor_cmd = [
        sys.executable,
        "scripts/competition_supervisor.py",
        "--leaderboard",
        args.leaderboard,
        "--team",
        args.team,
        "--backlog-dir",
        args.backlog_dir,
        "--ledger-json",
        args.ledger_json,
        "--exactness-report",
        args.exactness_report,
        "--scoring-json",
        str(scoring_json),
        "--min-ticket",
        "31",
        "--warmup-budget",
        str(args.warmup_budget),
        "--out",
        str(supervisor_report),
        "--runs-json",
        str(supervisor_runs),
    ]
    _run(supervisor_cmd, cwd=root, env=env)

    progress_report = research_dir / "competition_progress_2026-03-12.md"
    progress_cmd = [
        sys.executable,
        "scripts/update_competition_progress.py",
        "--leaderboard",
        args.leaderboard,
        "--team",
        args.team,
        "--ledger-json",
        args.ledger_json,
        "--scoring-json",
        str(scoring_json),
        "--anchor-slice-json",
        str(anchor_slice_json),
        "--warmup-budget",
        str(args.warmup_budget),
        "--out",
        str(progress_report),
    ]
    _run(progress_cmd, cwd=root, env=env)

    cycle_summary = research_dir / f"offline_cycle_{args.artifact_suffix}.json"
    _write_cycle_summary(
        path=cycle_summary,
        suffix=args.artifact_suffix,
        candidate_submission=candidate_submission,
        candidate_preflight=candidate_preflight,
        candidate_raw_results=candidate_raw_results,
        gate_report=gate_report,
        anchor_slice_report=anchor_slice_report,
        scoring_report=scoring_report,
        supervisor_report=supervisor_report,
        exactness_queue_report=exactness_queue_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
