#!/usr/bin/env python3
"""Build a calibrated replay candidate from two platform artifact suffixes.

This script operationalizes the current answer-stable replay workflow without
requiring manual path assembly. It does not run the platform pipeline itself;
it assumes both source artifact bundles already exist under
``platform_runs/<phase>/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]
ROOT = Path(__file__).resolve().parents[1]


def artifact_bundle(*, root: Path, phase: str, suffix: str) -> dict[str, Path]:
    """Resolve the three required artifact paths for one suffix.

    Args:
        root: Repository root.
        phase: Platform phase, typically ``warmup`` or ``final``.
        suffix: Artifact suffix passed to ``submission.platform``.

    Returns:
        dict[str, Path]: Submission, raw-results, and preflight paths.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """

    phase_dir = root / "platform_runs" / phase
    bundle = {
        "submission": phase_dir / f"submission_{suffix}.json",
        "raw_results": phase_dir / f"raw_results_{suffix}.json",
        "preflight": phase_dir / f"preflight_summary_{suffix}.json",
    }
    missing = [str(path) for path in bundle.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifact bundle for suffix {suffix}: {missing}")
    return bundle


def default_out_dir(*, root: Path, phase: str, answer_source_suffix: str, page_source_suffix: str) -> Path:
    """Build the default replay candidate output directory.

    Args:
        root: Repository root.
        phase: Platform phase.
        answer_source_suffix: Suffix for the frozen answer source.
        page_source_suffix: Suffix for the page-source challenger.

    Returns:
        Path: Default replay output directory.
    """

    label = f"replay_{phase}_answers_{answer_source_suffix}__pages_{page_source_suffix}"
    return root / "platform_runs" / phase / "replay_candidates" / label


def build_replay_command(
    *,
    answer_bundle: dict[str, Path],
    page_bundle: dict[str, Path],
    out_dir: Path,
    reviewed_all: Path,
    reviewed_high: Path,
    page_source_pages_default: str,
    answer_qids: list[str],
    answer_qids_file: Path | None,
    page_qids: list[str],
    page_qids_file: Path | None,
) -> list[str]:
    """Build the replay subprocess command.

    Args:
        answer_bundle: Frozen-answer artifact bundle.
        page_bundle: Page-source artifact bundle.
        out_dir: Replay output directory.
        reviewed_all: Reviewed all_100 golden path.
        reviewed_high: Reviewed high_81 golden path.
        page_source_pages_default: Default page projection policy.
        answer_qids: Allowlisted answer QIDs from the page source.
        answer_qids_file: Optional file of answer QIDs.
        page_qids: Allowlisted page QIDs from the page source.
        page_qids_file: Optional file of page QIDs.

    Returns:
        list[str]: Subprocess command.
    """

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_answer_stable_grounding_replay.py"),
        "--answer-source-submission",
        str(answer_bundle["submission"]),
        "--answer-source-raw-results",
        str(answer_bundle["raw_results"]),
        "--answer-source-preflight",
        str(answer_bundle["preflight"]),
        "--page-source-submission",
        str(page_bundle["submission"]),
        "--page-source-raw-results",
        str(page_bundle["raw_results"]),
        "--page-source-preflight",
        str(page_bundle["preflight"]),
        "--page-source-pages-default",
        page_source_pages_default,
        "--reviewed-all",
        str(reviewed_all),
        "--reviewed-high",
        str(reviewed_high),
        "--out-dir",
        str(out_dir),
    ]
    for qid in answer_qids:
        command.extend(["--page-source-answer-qid", qid])
    if answer_qids_file is not None:
        command.extend(["--page-source-answer-qids-file", str(answer_qids_file)])
    for qid in page_qids:
        command.extend(["--page-source-page-qid", qid])
    if page_qids_file is not None:
        command.extend(["--page-source-page-qids-file", str(page_qids_file)])
    return command


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for calibrated replay candidate construction.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("warmup", "final"), default="warmup")
    parser.add_argument("--answer-source-suffix", required=True)
    parser.add_argument("--page-source-suffix", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--page-source-pages-default", choices=("all", "none"), default="all")
    parser.add_argument("--page-source-answer-qid", action="append", default=[])
    parser.add_argument("--page-source-answer-qids-file", type=Path, default=None)
    parser.add_argument("--page-source-page-qid", action="append", default=[])
    parser.add_argument("--page-source-page-qids-file", type=Path, default=None)
    parser.add_argument("--reviewed-all", type=Path, default=ROOT / ".sdd" / "golden" / "reviewed" / "reviewed_all_100.json")
    parser.add_argument(
        "--reviewed-high",
        type=Path,
        default=ROOT / ".sdd" / "golden" / "reviewed" / "reviewed_high_confidence_81.json",
    )
    return parser.parse_args()


def main() -> int:
    """Build the calibrated replay artifact.

    Returns:
        int: Process exit code.
    """

    args = parse_args()
    answer_bundle = artifact_bundle(
        root=ROOT,
        phase=args.phase,
        suffix=str(args.answer_source_suffix),
    )
    page_bundle = artifact_bundle(
        root=ROOT,
        phase=args.phase,
        suffix=str(args.page_source_suffix),
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else default_out_dir(
            root=ROOT,
            phase=str(args.phase),
            answer_source_suffix=str(args.answer_source_suffix),
            page_source_suffix=str(args.page_source_suffix),
        )
    )
    command = build_replay_command(
        answer_bundle=answer_bundle,
        page_bundle=page_bundle,
        out_dir=out_dir,
        reviewed_all=args.reviewed_all.resolve(),
        reviewed_high=args.reviewed_high.resolve(),
        page_source_pages_default=str(args.page_source_pages_default),
        answer_qids=[str(item) for item in args.page_source_answer_qid],
        answer_qids_file=args.page_source_answer_qids_file.resolve() if args.page_source_answer_qids_file else None,
        page_qids=[str(item) for item in args.page_source_page_qid],
        page_qids_file=args.page_source_page_qids_file.resolve() if args.page_source_page_qids_file else None,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: JsonDict = {
        "phase": args.phase,
        "answer_source_suffix": args.answer_source_suffix,
        "page_source_suffix": args.page_source_suffix,
        "answer_bundle": {key: str(path) for key, path in answer_bundle.items()},
        "page_bundle": {key: str(path) for key, path in page_bundle.items()},
        "reviewed_all": str(args.reviewed_all.resolve()),
        "reviewed_high": str(args.reviewed_high.resolve()),
        "page_source_pages_default": args.page_source_pages_default,
        "page_source_answer_qids": list(args.page_source_answer_qid),
        "page_source_answer_qids_file": (
            str(args.page_source_answer_qids_file.resolve()) if args.page_source_answer_qids_file else ""
        ),
        "page_source_page_qids": list(args.page_source_page_qid),
        "page_source_page_qids_file": (
            str(args.page_source_page_qids_file.resolve()) if args.page_source_page_qids_file else ""
        ),
        "command": command,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    (out_dir / "replay_build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
