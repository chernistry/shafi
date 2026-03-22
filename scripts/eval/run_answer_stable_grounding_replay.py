#!/usr/bin/env python3
"""Build and score an answer-stable grounding replay artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

try:
    from score_against_golden import score as score_against_golden
except ModuleNotFoundError:  # pragma: no cover
    from scripts.score_against_golden import score as score_against_golden

from rag_challenge.submission.replay import (
    build_counterfactual_preflight,
    compare_submission_drift,
    load_json_dict,
    load_json_list,
    merge_answer_stable_records,
    qid_allowlist,
    sha256_json_file,
)

JsonDict = dict[str, object]


def _json_dict(value: object) -> JsonDict:
    """Coerce an arbitrary value into a JSON dict.

    Args:
        value: Arbitrary JSON-like value.

    Returns:
        JsonDict: Mapping or empty dict.
    """

    return cast("JsonDict", value) if isinstance(value, dict) else {}


def _family_mean(value: object) -> float | None:
    """Extract a by-family mean F-beta value.

    Args:
        value: Family payload from the scorer.

    Returns:
        float | None: Mean F-beta when available.
    """

    family = _json_dict(value)
    f_beta = _json_dict(family.get("f_beta"))
    mean = f_beta.get("mean")
    return float(mean) if isinstance(mean, int | float) else None


def _family_delta(
    baseline_score: JsonDict,
    candidate_score: JsonDict,
) -> dict[str, float]:
    """Compute per-family grounding deltas between two reviewed scores.

    Args:
        baseline_score: Baseline reviewed summary.
        candidate_score: Candidate reviewed summary.

    Returns:
        dict[str, float]: Per-family F-beta deltas.
    """

    baseline_families = cast("dict[str, object]", baseline_score.get("family_f_beta") or {})
    candidate_families = cast("dict[str, object]", candidate_score.get("family_f_beta") or {})
    keys = sorted(set(baseline_families) | set(candidate_families))
    out: dict[str, float] = {}
    for key in keys:
        baseline_value = baseline_families.get(key)
        candidate_value = candidate_families.get(key)
        baseline_float = float(baseline_value) if isinstance(baseline_value, int | float) else 0.0
        candidate_float = float(candidate_value) if isinstance(candidate_value, int | float) else 0.0
        out[key] = round(candidate_float - baseline_float, 6)
    return out


def _reviewed_summary(raw_results_path: Path, golden_path: Path) -> JsonDict:
    """Score one raw-results artifact against a reviewed slice.

    Args:
        raw_results_path: Raw results JSON path.
        golden_path: Reviewed golden path.

    Returns:
        JsonDict: Condensed reviewed score summary.
    """

    score = score_against_golden(raw_results_path, golden_path)
    summary = _json_dict(score.get("summary"))
    by_family = _json_dict(score.get("by_family"))
    return {
        "overall_grounding_f_beta": summary.get("overall_grounding_f_beta"),
        "trusted_grounding_f_beta": summary.get("trusted_grounding_f_beta"),
        "weighted_grounding_f_beta": summary.get("weighted_grounding_f_beta"),
        "family_f_beta": {
            key: mean
            for key, value in by_family.items()
            if (mean := _family_mean(value)) is not None
        },
    }


def _page_count_delta_distribution(page_count_deltas: dict[str, int]) -> JsonDict:
    """Summarize used-page count deltas for a replay.

    Args:
        page_count_deltas: Per-question page count deltas.

    Returns:
        JsonDict: Simple delta distribution summary.
    """

    values = sorted(page_count_deltas.values())
    if not values:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0, "positive_count": 0, "negative_count": 0}

    def _percentile(q: float) -> int:
        idx = min(len(values) - 1, max(0, int((len(values) - 1) * q)))
        return int(values[idx])

    return {
        "min": values[0],
        "p50": _percentile(0.50),
        "p95": _percentile(0.95),
        "max": values[-1],
        "positive_count": len([value for value in values if value > 0]),
        "negative_count": len([value for value in values if value < 0]),
    }


def _render_markdown(payload: JsonDict) -> str:
    """Render replay results into markdown.

    Args:
        payload: Replay summary payload.

    Returns:
        str: Markdown report.
    """

    drift = cast("dict[str, object]", payload["drift"])
    reviewed_all = _json_dict(payload.get("reviewed_all_100"))
    reviewed_high = _json_dict(payload.get("reviewed_high_confidence_81"))
    scoring_skipped = bool(reviewed_all.get("skipped"))
    baseline_all = _json_dict(reviewed_all.get("baseline"))
    candidate_all = _json_dict(reviewed_all.get("candidate"))
    baseline_high = _json_dict(reviewed_high.get("baseline"))
    candidate_high = _json_dict(reviewed_high.get("candidate"))
    artifact_contract = _json_dict(payload.get("artifact_contract"))
    lines = [
        "# 641 Answer-Stable Grounding Replay",
        "",
        f"- command: `{payload['command']}`",
        f"- answer_source_submission: `{artifact_contract.get('answer_source_submission')}`",
        f"- page_source_submission: `{artifact_contract.get('page_source_submission')}`",
        "",
        "## Drift",
        "",
        f"- answer_changed_count: `{drift['answer_changed_count']}`",
        f"- page_changed_count: `{drift['page_changed_count']}`",
        f"- used_page_count_delta_distribution: `{drift['used_page_count_delta_distribution']}`",
        f"- answer_changed_qids: `{drift['answer_changed_qids']}`",
        f"- page_changed_qids: `{drift['page_changed_qids']}`",
    ]
    if scoring_skipped:
        lines.extend(["", "## Reviewed Scoring", "", "- Scoring skipped (--skip-scoring or missing golden files)"])
    else:
        lines.extend([
            "",
            "## Reviewed all_100",
            "",
            f"- baseline_overall_grounding_f_beta: `{baseline_all.get('overall_grounding_f_beta')}`",
            f"- candidate_overall_grounding_f_beta: `{candidate_all.get('overall_grounding_f_beta')}`",
            f"- baseline_trusted_grounding_f_beta: `{baseline_all.get('trusted_grounding_f_beta')}`",
            f"- candidate_trusted_grounding_f_beta: `{candidate_all.get('trusted_grounding_f_beta')}`",
            f"- family_delta: `{reviewed_all.get('family_delta')}`",
            "",
            "## Reviewed high_confidence_81",
            "",
            f"- baseline_overall_grounding_f_beta: `{baseline_high.get('overall_grounding_f_beta')}`",
            f"- candidate_overall_grounding_f_beta: `{candidate_high.get('overall_grounding_f_beta')}`",
            f"- baseline_trusted_grounding_f_beta: `{baseline_high.get('trusted_grounding_f_beta')}`",
            f"- candidate_trusted_grounding_f_beta: `{candidate_high.get('trusted_grounding_f_beta')}`",
            f"- family_delta: `{reviewed_high.get('family_delta')}`",
        ])
    lines.extend([
        "",
        "## Contract Notes",
        "",
        "- Answers are frozen from the answer-source artifact unless explicitly allowlisted.",
        "- Replay fails loudly if submission/raw-results question sets do not align.",
        "- This harness is grounding-only; answer-path prompts and retrieval are not mutated here.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the replay harness.

    Returns:
        argparse.Namespace: Parsed CLI args.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-source-submission", type=Path, required=True)
    parser.add_argument("--answer-source-raw-results", type=Path, required=True)
    parser.add_argument("--answer-source-preflight", type=Path, required=True)
    parser.add_argument("--page-source-submission", type=Path, required=True)
    parser.add_argument("--page-source-raw-results", type=Path, required=True)
    parser.add_argument("--page-source-preflight", type=Path, required=True)
    parser.add_argument("--page-source-answer-qid", action="append", default=[])
    parser.add_argument("--page-source-answer-qids-file", type=Path, default=None)
    parser.add_argument("--page-source-page-qid", action="append", default=[])
    parser.add_argument("--page-source-page-qids-file", type=Path, default=None)
    parser.add_argument("--page-source-pages-default", choices=("all", "none"), default="all")
    parser.add_argument("--reviewed-all", type=Path, default=Path(".sdd/golden/reviewed/reviewed_all_100.json"))
    parser.add_argument(
        "--reviewed-high",
        type=Path,
        default=Path(".sdd/golden/reviewed/reviewed_high_confidence_81.json"),
    )
    parser.add_argument("--skip-scoring", action="store_true", help="Skip reviewed golden scoring (for private set)")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run the answer-stable grounding replay harness.

    Returns:
        int: Process exit code.
    """

    args = parse_args()
    answer_allowlist = qid_allowlist(
        values=args.page_source_answer_qid,
        file_path=args.page_source_answer_qids_file.resolve() if args.page_source_answer_qids_file else None,
    )
    page_allowlist = qid_allowlist(
        values=args.page_source_page_qid,
        file_path=args.page_source_page_qids_file.resolve() if args.page_source_page_qids_file else None,
    )

    answer_source_submission = load_json_dict(args.answer_source_submission.resolve())
    answer_source_raw_results = load_json_list(args.answer_source_raw_results.resolve())
    answer_source_preflight = load_json_dict(args.answer_source_preflight.resolve())
    page_source_submission = load_json_dict(args.page_source_submission.resolve())
    page_source_raw_results = load_json_list(args.page_source_raw_results.resolve())
    page_source_preflight = load_json_dict(args.page_source_preflight.resolve())

    merged_submission, merged_raw_results, merge_report = merge_answer_stable_records(
        answer_source_submission=answer_source_submission,
        answer_source_raw_results=answer_source_raw_results,
        page_source_submission=page_source_submission,
        page_source_raw_results=page_source_raw_results,
        allowlisted_qids=answer_allowlist,
        page_allowlisted_qids=page_allowlist,
        page_source_pages_default=str(args.page_source_pages_default),
    )
    merged_preflight = build_counterfactual_preflight(
        merged_payload=merged_submission,
        answer_source_preflight=answer_source_preflight,
        page_source_preflight=page_source_preflight,
        answer_source_submission=args.answer_source_submission.resolve(),
        page_source_submission=args.page_source_submission.resolve(),
        allowlisted_qids=answer_allowlist,
        page_allowlisted_qids=page_allowlist,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_submission_path = out_dir / "submission_answer_stable_replay.json"
    merged_raw_results_path = out_dir / "raw_results_answer_stable_replay.json"
    merged_preflight_path = out_dir / "preflight_answer_stable_replay.json"

    merged_submission_path.write_text(
        json.dumps(merged_submission, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    merged_raw_results_path.write_text(
        json.dumps(merged_raw_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    merged_preflight["raw_results_path"] = str(merged_raw_results_path)
    merged_preflight["submission_sha256"] = sha256_json_file(merged_submission_path)
    merged_preflight_path.write_text(
        json.dumps(merged_preflight, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    drift = compare_submission_drift(
        baseline_submission=answer_source_submission,
        candidate_submission=merged_submission,
        baseline_raw_results=answer_source_raw_results,
        candidate_raw_results=merged_raw_results,
    )
    skip_scoring = getattr(args, "skip_scoring", False) or not (
        args.reviewed_all.resolve().exists() and args.reviewed_high.resolve().exists()
    )

    if skip_scoring:
        baseline_all = candidate_all = baseline_high = candidate_high = {}
    else:
        baseline_all = _reviewed_summary(args.answer_source_raw_results.resolve(), args.reviewed_all.resolve())
        candidate_all = _reviewed_summary(merged_raw_results_path, args.reviewed_all.resolve())
        baseline_high = _reviewed_summary(args.answer_source_raw_results.resolve(), args.reviewed_high.resolve())
        candidate_high = _reviewed_summary(merged_raw_results_path, args.reviewed_high.resolve())

    payload: JsonDict = {
        "command": " ".join(sys.argv),
        "artifact_contract": {
            "answer_source_submission": str(args.answer_source_submission.resolve()),
            "answer_source_raw_results": str(args.answer_source_raw_results.resolve()),
            "page_source_submission": str(args.page_source_submission.resolve()),
            "page_source_raw_results": str(args.page_source_raw_results.resolve()),
            "page_source_answer_qids": sorted(answer_allowlist),
            "page_source_page_qids": sorted(page_allowlist),
            "page_source_pages_default": args.page_source_pages_default,
        },
        "merge_report": merge_report,
        "drift": {
            "answer_changed_count": drift.answer_changed_count,
            "page_changed_count": drift.page_changed_count,
            "answer_changed_qids": drift.answer_changed_qids,
            "page_changed_qids": drift.page_changed_qids,
            "used_page_count_delta_distribution": _page_count_delta_distribution(drift.used_page_count_deltas),
        },
        "reviewed_all_100": {
            "baseline": baseline_all,
            "candidate": candidate_all,
            "family_delta": _family_delta(baseline_all, candidate_all),
        } if not skip_scoring else {"skipped": True},
        "reviewed_high_confidence_81": {
            "baseline": baseline_high,
            "candidate": candidate_high,
            "family_delta": _family_delta(baseline_high, candidate_high),
        } if not skip_scoring else {"skipped": True},
        "outputs": {
            "submission": str(merged_submission_path),
            "raw_results": str(merged_raw_results_path),
            "preflight": str(merged_preflight_path),
        },
    }

    (out_dir / "replay_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "replay_summary.md").write_text(_render_markdown(payload), encoding="utf-8")
    (out_dir / "closeout.md").write_text(
        "\n".join(
            [
                "# Ticket 641 Closeout",
                "",
                "## Commands Run",
                "",
                f"- `{ ' '.join(sys.argv) }`",
                "",
                "## Artifact Contract",
                "",
                f"- answer_source_submission: `{args.answer_source_submission.resolve()}`",
                f"- page_source_submission: `{args.page_source_submission.resolve()}`",
                f"- page_source_pages_default: `{args.page_source_pages_default}`",
                f"- page_source_answer_qids: `{sorted(answer_allowlist)}`",
                f"- page_source_page_qids: `{sorted(page_allowlist)}`",
                "",
                "## Answer Drift Proof",
                "",
                f"- answer_changed_count: `{drift.answer_changed_count}`",
                f"- answer_changed_qids: `{drift.answer_changed_qids}`",
                "",
                "## Outputs",
                "",
                "- `submission_answer_stable_replay.json`",
                "- `raw_results_answer_stable_replay.json`",
                "- `preflight_answer_stable_replay.json`",
                "- `replay_summary.json`",
                "- `replay_summary.md`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
