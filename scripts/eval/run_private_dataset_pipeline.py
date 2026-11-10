#!/usr/bin/env python3
"""Orchestrate the full private-dataset dual-pipeline replay workflow.

Phases:
  ingest-a   Ingest PDFs into 1024-dim Qdrant collection (v6 profile).
  ingest-b   Ingest PDFs into 1792-dim Qdrant collection.
  run-a      Run answer pipeline A (1024-dim, best answers).
  run-b      Run page pipeline B (1792-dim, best grounding).
  merge      Replay-merge: freeze A answers, swap B pages.
  analyze    Generate per-type statistics and comparison report.
  all        Execute all phases sequentially.

Each phase is idempotent — it checks for existing artifacts before running.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = ROOT / "profiles"
PLATFORM_RUNS_DIR = ROOT / "platform_runs" / "final"

V6_PROFILE = PROFILES_DIR / "private_v6_regime.env"
V1792_PROFILE = PROFILES_DIR / "private_1792_regime.env"

JsonDict = dict[str, Any]


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _elapsed(start: float) -> str:
    secs = time.monotonic() - start
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{secs / 60:.1f}m"


def _load_profile_env(profile_path: Path) -> dict[str, str]:
    """Parse a .env profile file into a dict (ignoring comments and blanks)."""
    env: dict[str, str] = {}
    for line in profile_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            env[key.strip()] = value.strip()
    return env


def _build_env(profile_path: Path) -> dict[str, str]:
    """Merge current process env with profile overrides."""
    env = dict(os.environ)
    env.update(_load_profile_env(profile_path))
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def _artifact_exists(suffix: str) -> bool:
    """Check if a submission artifact already exists for the given suffix."""
    return (PLATFORM_RUNS_DIR / f"submission_{suffix}.json").exists()


def _raw_results_path(suffix: str) -> Path:
    return PLATFORM_RUNS_DIR / f"raw_results_{suffix}.json"


def _submission_path(suffix: str) -> Path:
    return PLATFORM_RUNS_DIR / f"submission_{suffix}.json"


def _preflight_path(suffix: str) -> Path:
    return PLATFORM_RUNS_DIR / f"preflight_summary_{suffix}.json"


def _run_subprocess(cmd: list[str], env: dict[str, str], label: str) -> None:
    """Run a subprocess, streaming output, and raise on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    print(f"  [{label}] completed in {_elapsed(start)} (exit={result.returncode})")
    if result.returncode != 0:
        raise SystemExit(f"FAILED: {label} (exit {result.returncode})")


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def phase_ingest_a(args: argparse.Namespace) -> None:
    """Ingest private PDFs into 1024-dim collection."""
    print("\n>>> Phase: ingest-a (1024-dim v6 collection)")
    env = _build_env(V6_PROFILE)
    env["EVAL_PHASE"] = "final"
    _run_subprocess(
        [
            sys.executable,
            "-m",
            "shafi.submission.platform",
            "--archive-only",  # we only want ingestion, not full run
        ],
        env=env,
        label="Ingest 1024-dim (v6 profile)",
    )
    # Note: actual ingestion happens when we run the pipeline without --skip-ingest.
    # The archive-only mode just validates the environment.
    print("  Ingest-a: collection ready (will ingest on first run-a if needed)")


def phase_ingest_b(args: argparse.Namespace) -> None:
    """Ingest private PDFs into 1792-dim collection."""
    print("\n>>> Phase: ingest-b (1792-dim collection)")
    env = _build_env(V1792_PROFILE)
    env["EVAL_PHASE"] = "final"
    _run_subprocess(
        [
            sys.executable,
            "-m",
            "shafi.submission.platform",
            "--archive-only",
        ],
        env=env,
        label="Ingest 1792-dim (kanon2 profile)",
    )
    print("  Ingest-b: collection ready (will ingest on first run-b if needed)")


def phase_run_a(args: argparse.Namespace) -> str:
    """Run Pipeline A (1024-dim) for answer extraction."""
    suffix = args.suffix_a or f"private_run_a_v6_{args.run_id}"
    print(f"\n>>> Phase: run-a (1024-dim answer source, suffix={suffix})")

    if _artifact_exists(suffix) and not args.force:
        print(f"  SKIP: artifacts already exist for suffix '{suffix}'")
        return suffix

    env = _build_env(V6_PROFILE)
    env["EVAL_PHASE"] = "final"
    cmd = [
        sys.executable,
        "-m",
        "shafi.submission.platform",
        "--artifact-suffix",
        suffix,
    ]
    if args.skip_ingest:
        cmd.append("--skip-ingest")
    _run_subprocess(cmd, env=env, label=f"Pipeline A (1024-dim) → {suffix}")
    return suffix


def phase_run_b(args: argparse.Namespace) -> str:
    """Run Pipeline B (1792-dim) for page grounding."""
    suffix = args.suffix_b or f"private_run_b_1792_{args.run_id}"
    print(f"\n>>> Phase: run-b (1792-dim page source, suffix={suffix})")

    if _artifact_exists(suffix) and not args.force:
        print(f"  SKIP: artifacts already exist for suffix '{suffix}'")
        return suffix

    env = _build_env(V1792_PROFILE)
    env["EVAL_PHASE"] = "final"
    cmd = [
        sys.executable,
        "-m",
        "shafi.submission.platform",
        "--artifact-suffix",
        suffix,
    ]
    if args.skip_ingest:
        cmd.append("--skip-ingest")
    _run_subprocess(cmd, env=env, label=f"Pipeline B (1792-dim) → {suffix}")
    return suffix


def phase_merge(args: argparse.Namespace, suffix_a: str, suffix_b: str) -> Path:
    """Merge answers from A + pages from B via replay."""
    out_dir = PLATFORM_RUNS_DIR / f"replay_{args.run_id}"
    print(f"\n>>> Phase: merge (A={suffix_a}, B={suffix_b})")

    merged_submission = out_dir / "submission_answer_stable_replay.json"
    if merged_submission.exists() and not args.force:
        print(f"  SKIP: merged artifacts already exist at {out_dir}")
        return out_dir

    # Verify all required artifacts exist
    for label, suffix in [("A", suffix_a), ("B", suffix_b)]:
        for kind, path_fn in [("submission", _submission_path), ("raw_results", _raw_results_path), ("preflight", _preflight_path)]:
            p = path_fn(suffix)
            if not p.exists():
                raise SystemExit(f"MISSING: Pipeline {label} {kind} artifact: {p}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    _run_subprocess(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_answer_stable_grounding_replay.py"),
            "--answer-source-submission",
            str(_submission_path(suffix_a)),
            "--answer-source-raw-results",
            str(_raw_results_path(suffix_a)),
            "--answer-source-preflight",
            str(_preflight_path(suffix_a)),
            "--page-source-submission",
            str(_submission_path(suffix_b)),
            "--page-source-raw-results",
            str(_raw_results_path(suffix_b)),
            "--page-source-preflight",
            str(_preflight_path(suffix_b)),
            "--page-source-pages-default",
            "all",
            "--skip-scoring",
            "--out-dir",
            str(out_dir),
        ],
        env=env,
        label="Replay merge (freeze A answers, swap B pages)",
    )

    # Verify zero answer drift
    summary_path = out_dir / "replay_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        drift = summary.get("drift", {})
        answer_changed = drift.get("answer_changed_count", -1)
        page_changed = drift.get("page_changed_count", 0)
        print(f"  Merge result: answers_changed={answer_changed}, pages_changed={page_changed}")
        if answer_changed != 0:
            print("  WARNING: Non-zero answer drift detected! Review replay_summary.json")

    return out_dir


def phase_analyze(args: argparse.Namespace, suffix_a: str, suffix_b: str, merged_dir: Path) -> None:
    """Generate per-type statistics and comparison report."""
    print("\n>>> Phase: analyze")
    report_lines: list[str] = ["# Private Dataset Dual-Pipeline Report", ""]

    # Load submissions
    sub_a_path = _submission_path(suffix_a)
    sub_b_path = _submission_path(suffix_b)
    merged_sub_path = merged_dir / "submission_answer_stable_replay.json"

    if not all(p.exists() for p in [sub_a_path, sub_b_path, merged_sub_path]):
        print("  SKIP: not all artifacts available for analysis")
        return

    sub_a = json.loads(sub_a_path.read_text(encoding="utf-8"))
    sub_b = json.loads(sub_b_path.read_text(encoding="utf-8"))
    sub_merged = json.loads(merged_sub_path.read_text(encoding="utf-8"))

    answers_a = {a["question_id"]: a for a in sub_a.get("answers", [])}
    answers_b = {a["question_id"]: a for a in sub_b.get("answers", [])}
    answers_m = {a["question_id"]: a for a in sub_merged.get("answers", [])}

    report_lines.append(f"- Pipeline A (1024-dim): `{suffix_a}` — {len(answers_a)} questions")
    report_lines.append(f"- Pipeline B (1792-dim): `{suffix_b}` — {len(answers_b)} questions")
    report_lines.append(f"- Merged: {len(answers_m)} questions")
    report_lines.append("")

    # Answer type distribution
    type_counts: dict[str, int] = {}
    for a in answers_a.values():
        at = str(a.get("telemetry", {}).get("answer_type", "unknown"))
        type_counts[at] = type_counts.get(at, 0) + 1
    report_lines.append("## Answer Type Distribution")
    report_lines.append("")
    for at in sorted(type_counts):
        report_lines.append(f"- `{at}`: {type_counts[at]}")
    report_lines.append("")

    # Page comparison: count how many questions have different pages between A and B
    page_diffs = 0
    for qid in answers_a:
        pages_a = json.dumps(
            answers_a[qid].get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages", []),
            sort_keys=True,
        )
        pages_b = json.dumps(
            answers_b.get(qid, {}).get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages", []),
            sort_keys=True,
        )
        if pages_a != pages_b:
            page_diffs += 1

    report_lines.append("## Page Grounding Comparison")
    report_lines.append("")
    report_lines.append(f"- Questions with different page projections (A vs B): {page_diffs}/{len(answers_a)}")
    report_lines.append("")

    # Answer comparison: count where A and B disagree on answers
    answer_diffs = 0
    for qid in answers_a:
        ans_a = json.dumps(answers_a[qid].get("answer"), sort_keys=True)
        ans_b = json.dumps(answers_b.get(qid, {}).get("answer"), sort_keys=True)
        if ans_a != ans_b:
            answer_diffs += 1

    report_lines.append("## Answer Comparison (A vs B)")
    report_lines.append("")
    report_lines.append(f"- Answer disagreements: {answer_diffs}/{len(answers_a)}")
    report_lines.append("")

    # Null answer counts
    null_a = sum(1 for a in answers_a.values() if a.get("answer") is None)
    null_b = sum(1 for a in answers_b.values() if a.get("answer") is None)
    null_m = sum(1 for a in answers_m.values() if a.get("answer") is None)
    report_lines.append("## Null Answers")
    report_lines.append("")
    report_lines.append(f"- Pipeline A: {null_a}")
    report_lines.append(f"- Pipeline B: {null_b}")
    report_lines.append(f"- Merged: {null_m}")
    report_lines.append("")

    # Merged answer stability verification
    answer_drift = 0
    for qid in answers_a:
        ans_a = json.dumps(answers_a[qid].get("answer"), sort_keys=True)
        ans_m = json.dumps(answers_m.get(qid, {}).get("answer"), sort_keys=True)
        if ans_a != ans_m:
            answer_drift += 1

    report_lines.append("## Merge Verification")
    report_lines.append("")
    report_lines.append(f"- Answer drift (A vs Merged): **{answer_drift}** (must be 0)")
    report_lines.append("")

    report_text = "\n".join(report_lines) + "\n"
    report_path = merged_dir / "analysis_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  Report written to {report_path}")
    print(report_text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["ingest-a", "ingest-b", "run-a", "run-b", "merge", "analyze", "all"],
        default="all",
        help="Which phase to execute (default: all)",
    )
    parser.add_argument(
        "--run-id",
        default=_timestamp(),
        help="Run identifier for artifact naming (default: timestamp)",
    )
    parser.add_argument(
        "--suffix-a",
        default=None,
        help="Override artifact suffix for pipeline A",
    )
    parser.add_argument(
        "--suffix-b",
        default=None,
        help="Override artifact suffix for pipeline B",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion (reuse existing Qdrant collections)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if artifacts already exist",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    phase = args.phase
    start_total = time.monotonic()

    PLATFORM_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Private Dataset Pipeline — run_id={args.run_id}, phase={phase}")
    print(f"  Profile A: {V6_PROFILE}")
    print(f"  Profile B: {V1792_PROFILE}")
    print(f"  Output: {PLATFORM_RUNS_DIR}")

    if not V6_PROFILE.exists():
        raise SystemExit(f"Missing profile: {V6_PROFILE}")
    if not V1792_PROFILE.exists():
        raise SystemExit(f"Missing profile: {V1792_PROFILE}")

    suffix_a = args.suffix_a or f"private_run_a_v6_{args.run_id}"
    suffix_b = args.suffix_b or f"private_run_b_1792_{args.run_id}"
    merged_dir = PLATFORM_RUNS_DIR / f"replay_{args.run_id}"

    if phase in ("ingest-a", "all"):
        phase_ingest_a(args)

    if phase in ("ingest-b", "all"):
        phase_ingest_b(args)

    if phase in ("run-a", "all"):
        suffix_a = phase_run_a(args)

    if phase in ("run-b", "all"):
        suffix_b = phase_run_b(args)

    if phase in ("merge", "all"):
        merged_dir = phase_merge(args, suffix_a, suffix_b)

    if phase in ("analyze", "all"):
        phase_analyze(args, suffix_a, suffix_b, merged_dir)

    print(f"\nTotal elapsed: {_elapsed(start_total)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
