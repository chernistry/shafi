# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from analyze_leaderboard import build_summary as build_leaderboard_summary
from analyze_leaderboard import load_rows as load_leaderboard_rows


def _load_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("dict[str, object]", obj)


def _as_float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    return default


def _as_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    return default


def _latest_experiment(ledger: dict[str, object]) -> dict[str, object]:
    experiments = ledger.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return {}
    latest = experiments[-1]
    return cast("dict[str, object]", latest) if isinstance(latest, dict) else {}


def build_report(
    *,
    leaderboard_path: Path,
    team_name: str,
    ledger_path: Path | None,
    scoring_json: Path | None,
    anchor_slice_json: Path | None,
    warmup_budget: int,
) -> str:
    leaderboard_summary = build_leaderboard_summary(load_leaderboard_rows(leaderboard_path), team_name=team_name)
    ledger = _load_json(ledger_path)
    latest_experiment = _latest_experiment(ledger)
    scoring = _load_json(scoring_json)
    anchor_slice = _load_json(anchor_slice_json)
    exactness_estimate = cast("dict[str, object]", scoring.get("exactness_estimate") or {})
    anchor_counts = cast("dict[str, object]", anchor_slice.get("status_counts") or {})

    submissions_used = _as_int(leaderboard_summary.get("submissions"))
    submissions_remaining = max(0, warmup_budget - submissions_used)
    lines = [
        "# Competition Progress Snapshot",
        "",
        "## Submission Budget",
        "",
        f"- Warm-up submissions used: `{submissions_used} / {warmup_budget}`",
        f"- Warm-up submissions remaining: `{submissions_remaining}`",
        "- Rule for this thread: **do not submit anything without explicit user approval**",
        "",
        "## Current Public State",
        "",
        f"- Team: `{leaderboard_summary.get('team_name')}`",
        f"- Rank: `{leaderboard_summary.get('rank')}`",
        f"- Total: `{_as_float(leaderboard_summary.get('total')):.6f}`",
        f"- S: `{_as_float(leaderboard_summary.get('s')):.6f}`",
        f"- G: `{_as_float(leaderboard_summary.get('g')):.6f}`",
        f"- Perfect `S=1.0` total at current `G/T/F`: `{_as_float(leaderboard_summary.get('perfect_s_total')):.6f}`",
        "",
        "## Strict Local Estimate",
        "",
        f"- Det lattice denominator: `{scoring.get('det_lattice_denominator')}`",
        f"- Asst lattice denominator: `{scoring.get('asst_lattice_denominator')}`",
        f"- `+1` deterministic full-answer upper bound: `+{_as_float(scoring.get('delta_total_per_full_deterministic_answer')):.6f}` total",
        f"- `+0.2` free-text judge step upper bound: `+{_as_float(scoring.get('delta_total_per_free_text_step')):.6f}` total",
        "",
        "## Latest Experiment",
        "",
    ]
    if latest_experiment:
        lines.extend(
            [
                f"- Label: `{latest_experiment.get('label')}`",
                f"- Recommendation: `{latest_experiment.get('recommendation')}`",
                f"- Answer drift: `{latest_experiment.get('answer_changed_count')}`",
                f"- Retrieval-page projection drift: `{latest_experiment.get('retrieval_page_projection_changed_count')}`",
                f"- Hidden-G trusted baseline: `{_as_float(latest_experiment.get('benchmark_trusted_baseline')):.4f}`",
                f"- Hidden-G trusted candidate: `{_as_float(latest_experiment.get('benchmark_trusted_candidate')):.4f}`",
            ]
        )
    else:
        lines.append("- none")

    lines.extend(["", "## Anchor Slice", ""])
    if anchor_counts:
        for key in sorted(anchor_counts):
            lines.append(f"- `{key}`: `{_as_int(anchor_counts[key])}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Exactness-Only Fallback", ""])
    if exactness_estimate:
        upper_bound = exactness_estimate.get("strict_upper_bound_total_if_all_answer_changes_are_real")
        lines.extend(
            [
                f"- answer_changed_count: `{exactness_estimate.get('answer_changed_count')}`",
                f"- page_changed_count: `{exactness_estimate.get('page_changed_count')}`",
                f"- strict upper-bound total: `{_as_float(upper_bound):.6f}`" if upper_bound is not None else "- strict upper-bound total: `(not applicable)`",
            ]
        )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Current Decision",
            "",
            "- Default: **NO SUBMIT**",
            "- Spend the last warm-up attempt only on a branch that clears trusted local gates and still looks worth it under the strict estimate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the human-readable competition progress snapshot.")
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--ledger-json", type=Path, default=None)
    parser.add_argument("--scoring-json", type=Path, default=None)
    parser.add_argument("--anchor-slice-json", type=Path, default=None)
    parser.add_argument("--warmup-budget", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        leaderboard_path=args.leaderboard,
        team_name=args.team,
        ledger_path=args.ledger_json,
        scoring_json=args.scoring_json,
        anchor_slice_json=args.anchor_slice_json,
        warmup_budget=args.warmup_budget,
    )
    args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
