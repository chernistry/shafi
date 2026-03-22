#!/usr/bin/env python3
"""Run the pipeline across multiple profiles and collect comparison results.

For each profile:
  1. Load the profile's env vars
  2. Run the pipeline via subprocess
  3. Save artifacts to output_dir/{profile_name}/
  4. Auto-run ab_eval_compare.py between consecutive profiles

Usage:
    python scripts/run_profile_matrix.py \
        --profiles profiles/private_v6_regime.env profiles/private_v7_enhanced.env \
        --output-dir matrix_results/ \
        --golden eval_golden_warmup.json \
        --skip-ingest
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _elapsed(start: float) -> str:
    secs = time.monotonic() - start
    return f"{secs:.1f}s" if secs < 60 else f"{secs / 60:.1f}m"


def _profile_name(profile_path: Path) -> str:
    """Extract a short name from a profile path for artifact naming."""
    return profile_path.stem.replace("_regime", "").replace("private_", "")


def _load_profile_env(profile_path: Path) -> dict[str, str]:
    """Parse a .env profile file into a dict."""
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


def _run_pipeline(
    profile_path: Path,
    output_dir: Path,
    suffix: str,
    skip_ingest: bool,
    eval_phase: str,
) -> dict[str, Path]:
    """Run the pipeline for a single profile and return artifact paths."""
    env = _build_env(profile_path)
    env["EVAL_PHASE"] = eval_phase

    cmd = [
        sys.executable,
        "-m",
        "shafi.submission.platform",
        "--artifact-suffix",
        suffix,
    ]
    if skip_ingest:
        cmd.append("--skip-ingest")

    label = f"Pipeline [{_profile_name(profile_path)}]"
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    print(f"  [{label}] completed in {_elapsed(start)} (exit={result.returncode})")

    if result.returncode != 0:
        raise SystemExit(f"FAILED: {label} (exit {result.returncode})")

    # Collect artifacts into profile-specific directory
    profile_dir = output_dir / _profile_name(profile_path)
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Find the artifacts in the platform_runs dir
    runs_dir = ROOT / "platform_runs" / eval_phase
    artifacts: dict[str, Path] = {}
    for kind in ("submission", "raw_results", "preflight_summary"):
        src_path = runs_dir / f"{kind}_{suffix}.json"
        if src_path.exists():
            dst_path = profile_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            artifacts[kind] = dst_path
            print(f"  Copied {src_path.name} → {profile_dir.name}/")
        else:
            print(f"  WARNING: {src_path.name} not found")

    return artifacts


def _run_comparison(
    baseline_artifacts: dict[str, Path],
    candidate_artifacts: dict[str, Path],
    golden_path: Path,
    output_dir: Path,
    baseline_name: str,
    candidate_name: str,
) -> None:
    """Run A/B comparison between two pipeline outputs."""
    baseline_sub = baseline_artifacts.get("submission")
    candidate_sub = candidate_artifacts.get("submission")

    if not baseline_sub or not baseline_sub.exists():
        print(f"  SKIP comparison: baseline submission missing for {baseline_name}")
        return
    if not candidate_sub or not candidate_sub.exists():
        print(f"  SKIP comparison: candidate submission missing for {candidate_name}")
        return

    compare_dir = output_dir / f"compare_{baseline_name}_vs_{candidate_name}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ab_eval_compare.py"),
        "--baseline-submission", str(baseline_sub),
        "--candidate-submission", str(candidate_sub),
        "--golden", str(golden_path),
        "--output", str(compare_dir / "comparison"),
    ]

    baseline_pf = baseline_artifacts.get("preflight_summary")
    candidate_pf = candidate_artifacts.get("preflight_summary")
    if baseline_pf and baseline_pf.exists():
        cmd.extend(["--baseline-preflight", str(baseline_pf)])
    if candidate_pf and candidate_pf.exists():
        cmd.extend(["--candidate-preflight", str(candidate_pf)])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    label = f"Compare {baseline_name} vs {candidate_name}"
    print(f"\n--- {label} ---")
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(f"  WARNING: comparison failed (exit {result.returncode})")
    else:
        print(f"  Comparison report: {compare_dir}/")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        nargs="+",
        required=True,
        help="Profile .env files to run (in order)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "matrix_results",
        help="Directory for all matrix outputs",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "eval_golden_warmup.json",
        help="Golden evaluation labels JSON",
    )
    parser.add_argument(
        "--eval-phase",
        default="warmup",
        help="Evaluation phase (warmup or final, default: warmup)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion (reuse existing Qdrant collections)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier for artifact naming (default: timestamp)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip pipeline execution, only run comparisons on existing artifacts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or _timestamp()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate profiles exist
    for profile in args.profiles:
        if not profile.exists():
            print(f"ERROR: Profile not found: {profile}", file=sys.stderr)
            return 1
    if not args.golden.exists():
        print(f"ERROR: Golden file not found: {args.golden}", file=sys.stderr)
        return 1

    print(f"Profile Matrix Run — run_id={run_id}")
    print(f"  Profiles: {[str(p) for p in args.profiles]}")
    print(f"  Output: {output_dir}")
    print(f"  Golden: {args.golden}")

    start_total = time.monotonic()
    all_artifacts: list[tuple[str, dict[str, Path]]] = []

    for profile in args.profiles:
        name = _profile_name(profile)
        suffix = f"matrix_{name}_{run_id}"
        profile_dir = output_dir / name

        if args.compare_only:
            # Look for existing artifacts
            artifacts: dict[str, Path] = {}
            for kind in ("submission", "raw_results", "preflight_summary"):
                p = profile_dir / f"{kind}_{suffix}.json"
                if p.exists():
                    artifacts[kind] = p
            if not artifacts:
                # Try to find any submission in the profile dir
                subs = sorted(profile_dir.glob("submission_*.json"))
                if subs:
                    artifacts["submission"] = subs[-1]
                    rr = sorted(profile_dir.glob("raw_results_*.json"))
                    if rr:
                        artifacts["raw_results"] = rr[-1]
                    pf = sorted(profile_dir.glob("preflight_summary_*.json"))
                    if pf:
                        artifacts["preflight_summary"] = pf[-1]
            if artifacts:
                all_artifacts.append((name, artifacts))
                print(f"  Found existing artifacts for {name}")
            else:
                print(f"  WARNING: No artifacts found for {name} in {profile_dir}")
        else:
            artifacts = _run_pipeline(
                profile_path=profile,
                output_dir=output_dir,
                suffix=suffix,
                skip_ingest=args.skip_ingest,
                eval_phase=args.eval_phase,
            )
            all_artifacts.append((name, artifacts))

    # Run comparisons between consecutive profiles
    if len(all_artifacts) >= 2:
        print("\n" + "=" * 60)
        print("  Running pairwise comparisons")
        print("=" * 60)
        for i in range(len(all_artifacts) - 1):
            base_name, base_art = all_artifacts[i]
            cand_name, cand_art = all_artifacts[i + 1]
            _run_comparison(
                baseline_artifacts=base_art,
                candidate_artifacts=cand_art,
                golden_path=args.golden.resolve(),
                output_dir=output_dir,
                baseline_name=base_name,
                candidate_name=cand_name,
            )

    # Write matrix summary
    summary: dict[str, Any] = {
        "run_id": run_id,
        "profiles": [str(p) for p in args.profiles],
        "profile_names": [_profile_name(p) for p in args.profiles],
        "artifacts": {
            name: {k: str(v) for k, v in arts.items()}
            for name, arts in all_artifacts
        },
    }
    import json
    summary_path = output_dir / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nMatrix run complete in {_elapsed(start_total)}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
