"""Export deterministic grounding-sidecar ML datasets from existing artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_challenge.ml.grounding_dataset import export_grounding_ml_dataset


def _default_path(*candidates: str) -> Path:
    """Return the first existing candidate path, or the first candidate if none exist.

    Args:
        *candidates: Absolute candidate path strings.

    Returns:
        First existing path, or the first candidate converted to a Path.
    """
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[0])


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured argument parser for the export script.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-raw-results",
        type=Path,
        default=_default_path(
            str(
                repo_root
                / ".sdd"
                / "researches"
                / "624_reviewed_heuristic_grounding_repair_r1_2026-03-19"
                / "raw_results_reviewed_public100_legacy_baseline.json"
            ),
            "/Users/sasha/IdeaProjects/.codex-worktrees/rag_challenge-grounding-sidecar-recovery/platform_runs/warmup/raw_results_v4_grounding_sidecar_r1_legacy.json",
            str(repo_root / "platform_runs" / "warmup" / "raw_results_v4_grounding_sidecar.json"),
        ),
    )
    parser.add_argument(
        "--sidecar-raw-results",
        type=Path,
        default=_default_path(
            str(
                repo_root
                / ".sdd"
                / "researches"
                / "624_reviewed_heuristic_grounding_repair_r1_2026-03-19"
                / "raw_results_reviewed_public100_sidecar_r1.json"
            ),
            "/Users/sasha/IdeaProjects/.codex-worktrees/rag_challenge-grounding-sidecar-recovery/platform_runs/warmup/raw_results_v4_grounding_sidecar_r1_sidecar.json",
            str(repo_root / "platform_runs" / "warmup" / "raw_results_v4_grounding_sidecar.json"),
        ),
    )
    parser.add_argument(
        "--golden-labels",
        type=Path,
        default=_default_path(
            str(repo_root / ".sdd" / "golden" / "reviewed" / "corrected_golden_labels_v3.json"),
            str(repo_root / ".sdd" / "golden" / "synthetic-ai-generated" / "golden_labels_v2.json"),
        ),
    )
    parser.add_argument(
        "--page-benchmark",
        type=Path,
        default=_default_path(
            str(repo_root / ".sdd" / "golden" / "reviewed" / "reviewed_page_benchmark_all_100.json"),
            str(repo_root / ".sdd" / "golden" / "page_benchmark_v2.json"),
        ),
    )
    parser.add_argument(
        "--suspect-labels",
        type=Path,
        default=_default_path(
            "/Users/sasha/IdeaProjects/.codex-worktrees/rag_challenge-grounding-sidecar-recovery/.sdd/researches/golden_label_self_audit_2026-03-18/matched_suspect_labels.json",
            str(repo_root / ".sdd" / "researches" / "missing_suspect_labels.json"),
        ),
    )
    parser.add_argument(
        "--reviewed-labels",
        type=Path,
        default=_default_path(
            str(repo_root / ".sdd" / "golden" / "reviewed" / "reviewed_all_100.json"),
            str(repo_root / ".sdd" / "golden" / "missing_reviewed_labels.json"),
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "derived" / "grounding_ml" / "v2_reviewed",
    )
    parser.add_argument("--split-seed", type=int, default=601)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    return parser


def main() -> int:
    """Run the export CLI.

    Returns:
        Process exit code.
    """
    args = build_arg_parser().parse_args()
    manifest = export_grounding_ml_dataset(
        legacy_raw_results_path=args.legacy_raw_results,
        sidecar_raw_results_path=args.sidecar_raw_results,
        golden_labels_path=args.golden_labels,
        page_benchmark_path=args.page_benchmark,
        suspect_labels_path=args.suspect_labels if args.suspect_labels.exists() else None,
        reviewed_labels_path=args.reviewed_labels,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        dev_ratio=args.dev_ratio,
    )
    print(args.output_dir)
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
