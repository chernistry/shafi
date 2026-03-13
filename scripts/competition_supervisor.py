# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from analyze_leaderboard import build_summary as build_leaderboard_summary
from analyze_leaderboard import load_rows as load_leaderboard_rows


@dataclass(frozen=True)
class SupervisorDecision:
    action: str
    rationale: list[str]
    submissions_remaining: int
    next_tickets: list[str]


def _load_json(path: Path) -> dict[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("dict[str, object]", obj)


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


def _load_latest_experiment(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    payload = _load_json(path)
    experiments_obj = payload.get("experiments")
    if not isinstance(experiments_obj, list) or not experiments_obj:
        return None
    last = experiments_obj[-1]
    return cast("dict[str, object]", last) if isinstance(last, dict) else None


def _load_exactness_report(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _load_equivalence_report(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _load_scoring_summary(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _load_candidate_ceiling_cycle(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _load_production_mimic(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    payload = _load_json(path)
    nested = payload.get("production_mimic")
    if isinstance(nested, dict):
        return cast("dict[str, object]", nested)
    return payload


def _load_remaining_signal_summary(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _load_alternative_gate_reports(paths: list[Path] | None) -> list[dict[str, object]]:
    if not paths:
        return []
    reports: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        reports.append(_load_json(path))
    return reports


def _exactness_page_metrics_identical(report: dict[str, object]) -> bool:
    direct = report.get("page_metrics_identical")
    if isinstance(direct, bool):
        return direct
    nested_obj = report.get("hidden_g_page_benchmark")
    if isinstance(nested_obj, dict):
        nested = cast("dict[str, object]", nested_obj).get("page_metrics_identical")
        if isinstance(nested, bool):
            return nested
    return False


def _ticket_num(path: Path) -> int:
    match = re.match(r"^(\d+)-", path.name)
    if match is None:
        return 0
    return int(match.group(1))


def _open_tickets(backlog_dir: Path, *, min_ticket: int) -> list[str]:
    if not backlog_dir.exists():
        return []
    tickets = [path.name for path in sorted(backlog_dir.iterdir(), key=_ticket_num) if _ticket_num(path) >= min_ticket]
    return tickets


def _safe_exactness_baselines(
    equivalence_report: dict[str, object] | None,
    *,
    required_safe_baseline_substrings: list[str],
) -> tuple[list[str], list[str]]:
    if equivalence_report is None:
        return [], []
    safe_obj = equivalence_report.get("safe_baselines")
    if not isinstance(safe_obj, list):
        return [], []
    safe_baselines = [str(item).strip() for item in safe_obj if str(item).strip()]
    if not required_safe_baseline_substrings:
        return safe_baselines, safe_baselines
    matched = [
        baseline
        for baseline in safe_baselines
        if any(required in baseline for required in required_safe_baseline_substrings)
    ]
    return safe_baselines, matched


def _remaining_signal_stats(summary: dict[str, object] | None) -> tuple[int, int]:
    if summary is None:
        return 0, 0
    summaries_obj = summary.get("summaries")
    if not isinstance(summaries_obj, list):
        return 0, 0
    family_count = 0
    actionable_count = 0
    for item in summaries_obj:
        if not isinstance(item, dict):
            continue
        family_count += 1
        if bool(item.get("likely_actionable")):
            actionable_count += 1
    return family_count, actionable_count


def _decide(
    *,
    subject_summary: dict[str, object],
    latest_experiment: dict[str, object] | None,
    exactness_report: dict[str, object] | None,
    equivalence_report: dict[str, object] | None,
    candidate_ceiling_cycle: dict[str, object] | None,
    production_mimic: dict[str, object] | None,
    remaining_signal_summary: dict[str, object] | None,
    alternative_gate_reports: list[dict[str, object]],
    required_safe_baseline_substrings: list[str],
    ticket_names: list[str],
    warmup_budget: int,
    target_rank: int,
) -> SupervisorDecision:
    submissions_used = _as_int(subject_summary.get("submissions"))
    submissions_remaining = max(0, warmup_budget - submissions_used)
    rationale: list[str] = []

    if submissions_remaining <= 1:
        rationale.append("warm-up budget is nearly exhausted")

    if latest_experiment is not None:
        recommendation = str(latest_experiment.get("recommendation") or "").strip()
        rationale.append(f"latest experiment recommendation={recommendation or 'n/a'}")
        if recommendation == "PROMISING":
            action = "rebuild_branch_promising"
            rationale.append("branch passed bounded offline gate")
            return SupervisorDecision(
                action=action,
                rationale=rationale,
                submissions_remaining=submissions_remaining,
                next_tickets=ticket_names[:5],
            )

    if production_mimic is not None:
        candidate_class = str(production_mimic.get("candidate_class") or "unknown").strip() or "unknown"
        lineage_confidence = str(production_mimic.get("lineage_confidence") or "unknown").strip() or "unknown"
        platform_like_total = _as_float(production_mimic.get("platform_like_total_estimate"))
        strict_total = _as_float(production_mimic.get("strict_total_estimate"))
        paranoid_total = _as_float(production_mimic.get("paranoid_total_estimate"))
        submit_eligibility = bool(production_mimic.get("submit_eligibility"))
        hidden_g_trusted = cast("dict[str, object]", production_mimic.get("hidden_g_trusted") or {})
        exactness_block = cast("dict[str, object]", production_mimic.get("exactness") or {})
        judge_block = cast("dict[str, object]", production_mimic.get("judge") or {})
        eval_block = cast("dict[str, object]", production_mimic.get("eval") or {})
        unresolved_qids = cast("list[object]", exactness_block.get("still_mismatched_incorrect_qids") or [])
        judge_timeout = bool(judge_block.get("judge_timeout_or_failure"))
        citation_hard_floor_blocked = bool(eval_block.get("citation_hard_floor_blocked"))
        citation_page_trace_disagreement = bool(eval_block.get("citation_page_trace_disagreement"))
        rationale.append(
            "production-like eval "
            f"candidate_class={candidate_class} lineage={lineage_confidence} "
            f"platform_like={platform_like_total:.6f} strict={strict_total:.6f} paranoid={paranoid_total:.6f}"
        )
        if citation_hard_floor_blocked:
            rationale.append("citation hard floor blocked promotion")
        if citation_page_trace_disagreement:
            rationale.append("citation/page-trace disagreement still requires explanation")
        if (
            platform_like_total > _as_float(subject_summary.get("total")) + 1e-9
            and _as_float(hidden_g_trusted.get("delta")) >= 0.0
            and not unresolved_qids
            and not judge_timeout
            and not citation_hard_floor_blocked
        ):
            if lineage_confidence != "high":
                rationale.append("candidate clears strict local bar except for lineage confidence")
                return SupervisorDecision(
                    action="candidate_requires_manual_lineage_review",
                    rationale=rationale,
                    submissions_remaining=submissions_remaining,
                    next_tickets=ticket_names[:5],
                )
            if submit_eligibility:
                rationale.append("candidate clears production-like local bar and requires only explicit user approval")
                return SupervisorDecision(
                    action="candidate_ready_for_manual_submit_review",
                    rationale=rationale,
                    submissions_remaining=submissions_remaining,
                    next_tickets=ticket_names[:5],
                )

    if candidate_ceiling_cycle is not None:
        remaining_family_count, remaining_actionable_count = _remaining_signal_stats(remaining_signal_summary)
        ranked_obj = candidate_ceiling_cycle.get("ranked_candidates")
        ranked = cast("list[object]", ranked_obj) if isinstance(ranked_obj, list) else []
        if ranked and isinstance(ranked[0], dict):
            best = cast("dict[str, object]", ranked[0])
            best_label = str(best.get("label") or "unknown").strip() or "unknown"
            paranoid_rank = _as_int(best.get("paranoid_rank_estimate"), default=10_000)
            paranoid_total = _as_float(best.get("paranoid_total_estimate"))
            strict_rank = _as_int(best.get("strict_rank_estimate"), default=10_000)
            upper_rank = _as_int(best.get("upper_rank_estimate"), default=10_000)
            blindspot_improved = _as_int(best.get("blindspot_improved_case_count"))
            support_undercoverage_blindspots = _as_int(best.get("blindspot_support_undercoverage_case_count"))
            strict_total = _as_float(best.get("strict_total_estimate"))
            rationale.append(
                f"best small-diff ceiling candidate={best_label} paranoid_rank≈{paranoid_rank} strict_rank≈{strict_rank} upper_rank≈{upper_rank} blindspot_gains={blindspot_improved}"
            )
            if paranoid_total <= _as_float(subject_summary.get("total")):
                rationale.append("paranoid estimate is non-improving relative to the current public baseline")
            if paranoid_rank > target_rank and strict_rank > target_rank and upper_rank > target_rank:
                rationale.append("best current small-diff path still misses the requested target rank even under the upper estimate")
                if remaining_family_count > 0:
                    rationale.append(
                        f"remaining family-level signal scan: actionable_families={remaining_actionable_count}/{remaining_family_count}"
                    )
                if remaining_family_count > 0 and remaining_actionable_count == 0:
                    rationale.append("no likely actionable family-level signals remain after investigated/manual-tested coverage")
                    if alternative_gate_reports:
                        alternative_labels = [
                            str(report.get("label") or "unknown").strip() or "unknown"
                            for report in alternative_gate_reports
                        ]
                        promising_alternatives = [
                            report
                            for report in alternative_gate_reports
                            if str(report.get("recommendation") or "").strip() == "PROMISING"
                        ]
                        rationale.append(
                            "alternative branch reports considered: "
                            + ", ".join(alternative_labels)
                        )
                        if not promising_alternatives:
                            rationale.append("no tested alternative branch class currently clears the bounded offline gate")
                            return SupervisorDecision(
                                action="local_ceiling_reached_hold_budget",
                                rationale=rationale,
                                submissions_remaining=submissions_remaining,
                                next_tickets=ticket_names[:5],
                            )
                    return SupervisorDecision(
                        action="small_diff_ceiling_reached",
                        rationale=rationale,
                        submissions_remaining=submissions_remaining,
                        next_tickets=ticket_names[:5],
                    )
                if blindspot_improved > 0 or support_undercoverage_blindspots > 0:
                    rationale.append("benchmark-blind page-family gains are still active, so the small-diff path is not yet fully exhausted")
                elif submissions_remaining <= 1:
                    return SupervisorDecision(
                        action="small_diff_ceiling_reached",
                        rationale=rationale,
                        submissions_remaining=submissions_remaining,
                        next_tickets=ticket_names[:5],
                    )
            elif strict_total > _as_float(subject_summary.get("total")):
                rationale.append("best current small-diff candidate still has plausible upside under strict estimate")

    if exactness_report is not None:
        answer_changed = _as_int(exactness_report.get("answer_changed_count"))
        page_changed = _as_int(exactness_report.get("page_changed_count"))
        page_metrics_identical = _exactness_page_metrics_identical(exactness_report)
        if answer_changed > 0 and page_changed == 0 and page_metrics_identical:
            safe_baselines, matched_baselines = _safe_exactness_baselines(
                equivalence_report,
                required_safe_baseline_substrings=required_safe_baseline_substrings,
            )
            if equivalence_report is None:
                rationale.append("exactness-only fallback exists but lineage proof is missing")
            elif not safe_baselines:
                rationale.append("exactness-only fallback exists but has no lineage-safe baseline")
            elif required_safe_baseline_substrings and not matched_baselines:
                rationale.append("exactness-only fallback exists but is not lineage-safe for the required champion baseline")
            else:
                baseline_note = ", ".join(matched_baselines or safe_baselines)
                rationale.append(f"audit-safe exactness-only fallback exists for baseline {baseline_note}")
                return SupervisorDecision(
                    action="exactness_only_candidate",
                    rationale=rationale,
                    submissions_remaining=submissions_remaining,
                    next_tickets=ticket_names[:5],
                )

    rationale.append("no branch is submit-safe enough yet")
    return SupervisorDecision(
        action="no_submit_continue_offline",
        rationale=rationale,
        submissions_remaining=submissions_remaining,
        next_tickets=ticket_names[:5],
    )


def _render_report(
    *,
    team_name: str,
    subject_summary: dict[str, object],
    latest_experiment: dict[str, object] | None,
    scoring_summary: dict[str, object] | None,
    candidate_ceiling_cycle: dict[str, object] | None,
    production_mimic: dict[str, object] | None,
    alternative_gate_reports: list[dict[str, object]],
    decision: SupervisorDecision,
) -> str:
    gap_targets = cast("list[dict[str, object]]", subject_summary.get("gap_targets") or [])
    lines = [
        "# Competition Supervisor Report",
        "",
        f"- Team: `{team_name}`",
        f"- Rank: `{subject_summary.get('rank')}`",
        f"- Total: `{_as_float(subject_summary.get('total')):.6f}`",
        f"- S: `{_as_float(subject_summary.get('s')):.6f}`",
        f"- G: `{_as_float(subject_summary.get('g')):.6f}`",
        f"- Perfect `S=1.0` total at current `G/T/F`: `{_as_float(subject_summary.get('perfect_s_total')):.6f}`",
        f"- `+0.01 G` => `+{_as_float(subject_summary.get('delta_total_per_g_0_01')):.6f}` total",
        f"- `+0.01 S` => `+{_as_float(subject_summary.get('delta_total_per_s_0_01')):.6f}` total",
        "",
        "## Budget",
        "",
        f"- Submissions used: `{subject_summary.get('submissions')}`",
        f"- Submissions remaining: `{decision.submissions_remaining}`",
        "",
        "## Higher-Rank Gaps",
        "",
    ]
    for target in gap_targets[:4]:
        lines.extend(
            [
                f"- Rank `{target['rank']}` `{target['team_name']}`",
                f"  - Need `ΔG={_as_float(target['delta_g_at_current_s']):+.6f}` at current `S/T/F`",
                f"  - Need `ΔS={_as_float(target['delta_s_at_current_g']):+.6f}` at current `G/T/F`",
            ]
        )

    lines.extend(["", "## Latest Experiment", ""])
    if latest_experiment is None:
        lines.append("- none")
    else:
        lines.extend(
            [
                f"- Label: `{latest_experiment.get('label')}`",
                f"- Recommendation: `{latest_experiment.get('recommendation')}`",
                f"- Answer changes: `{latest_experiment.get('answer_changed_count')}`",
                f"- Retrieval-page projection changes: `{latest_experiment.get('retrieval_page_projection_changed_count')}`",
                f"- Trusted hidden-G baseline: `{_as_float(latest_experiment.get('benchmark_trusted_baseline')):.4f}`",
                f"- Trusted hidden-G candidate: `{_as_float(latest_experiment.get('benchmark_trusted_candidate')):.4f}`",
            ]
        )

    lines.extend(["", "## Strict Estimate", ""])
    if scoring_summary is None:
        lines.append("- none")
    else:
        exactness_summary = cast("dict[str, object]", scoring_summary.get("exactness_estimate") or {})
        lines.extend(
            [
                f"- Det lattice denominator: `{scoring_summary.get('det_lattice_denominator')}`",
                f"- Asst lattice denominator: `{scoring_summary.get('asst_lattice_denominator')}`",
                f"- `+1` deterministic full-answer upper bound: `+{_as_float(scoring_summary.get('delta_total_per_full_deterministic_answer')):.6f}` total",
                f"- `+0.2` free-text judge step upper bound: `+{_as_float(scoring_summary.get('delta_total_per_free_text_step')):.6f}` total",
            ]
        )
        upper_bound = exactness_summary.get("strict_upper_bound_total_if_all_answer_changes_are_real")
        if upper_bound is not None:
            lines.append(f"- Exactness-only strict upper-bound total: `{_as_float(upper_bound):.6f}`")

    lines.extend(["", "## Ceiling Cycle", ""])
    if candidate_ceiling_cycle is None:
        lines.append("- none")
    else:
        ranked_obj = candidate_ceiling_cycle.get("ranked_candidates")
        ranked = cast("list[object]", ranked_obj) if isinstance(ranked_obj, list) else []
        if not ranked or not isinstance(ranked[0], dict):
            lines.append("- none")
        else:
            best = cast("dict[str, object]", ranked[0])
            lines.extend(
                [
                    f"- Best candidate: `{best.get('label')}`",
                    f"- Paranoid total estimate: `{_as_float(best.get('paranoid_total_estimate')):.6f}`",
                    f"- Strict total estimate: `{_as_float(best.get('strict_total_estimate')):.6f}`",
                    f"- Upper total estimate: `{_as_float(best.get('upper_total_estimate')):.6f}`",
                    f"- Paranoid rank estimate: `{_as_int(best.get('paranoid_rank_estimate'))}`",
                    f"- Strict rank estimate: `{_as_int(best.get('strict_rank_estimate'))}`",
                    f"- Upper rank estimate: `{_as_int(best.get('upper_rank_estimate'))}`",
                    f"- Blindspot improved cases: `{_as_int(best.get('blindspot_improved_case_count'))}`",
                    f"- Support-undercoverage blindspots: `{_as_int(best.get('blindspot_support_undercoverage_case_count'))}`",
                ]
            )

    lines.extend(["", "## Production-Like Eval", ""])
    if production_mimic is None:
        lines.append("- none")
    else:
        hidden_g_trusted = cast("dict[str, object]", production_mimic.get("hidden_g_trusted") or {})
        exactness = cast("dict[str, object]", production_mimic.get("exactness") or {})
        judge = cast("dict[str, object]", production_mimic.get("judge") or {})
        lines.extend(
            [
                f"- Candidate class: `{production_mimic.get('candidate_class')}`",
                f"- Lineage confidence: `{production_mimic.get('lineage_confidence')}`",
                f"- Submit eligibility: `{production_mimic.get('submit_eligibility')}`",
                f"- No-submit reason: `{production_mimic.get('no_submit_reason') or 'none'}`",
                f"- Platform-like total estimate: `{_as_float(production_mimic.get('platform_like_total_estimate')):.6f}`",
                f"- Strict total estimate: `{_as_float(production_mimic.get('strict_total_estimate')):.6f}`",
                f"- Paranoid total estimate: `{_as_float(production_mimic.get('paranoid_total_estimate')):.6f}`",
                f"- Platform-like rank estimate: `{_as_int(production_mimic.get('platform_like_rank_estimate'))}`",
                f"- Strict rank estimate: `{_as_int(production_mimic.get('strict_rank_estimate'))}`",
                f"- Paranoid rank estimate: `{_as_int(production_mimic.get('paranoid_rank_estimate'))}`",
                f"- Trusted hidden-G delta: `{_as_float(hidden_g_trusted.get('delta')):.4f}`",
                f"- Unresolved incorrect qids: `{len(cast('list[object]', exactness.get('still_mismatched_incorrect_qids') or []))}`",
                f"- Judge timeout/failure: `{judge.get('judge_timeout_or_failure')}`",
            ]
        )

    lines.extend(["", "## Alternative Branches", ""])
    if not alternative_gate_reports:
        lines.append("- none")
    else:
        for report in alternative_gate_reports:
            label = str(report.get("label") or "unknown")
            lines.extend(
                [
                    f"- `{label}`",
                    f"  - Recommendation: `{report.get('recommendation')}`",
                    f"  - Answer changes: `{_as_int(report.get('answer_changed_count'))}`",
                    f"  - Retrieval-page projection changes: `{_as_int(report.get('retrieval_page_projection_changed_count'))}`",
                    f"  - Trusted hidden-G baseline: `{_as_float(report.get('benchmark_trusted_baseline')):.4f}`",
                    f"  - Trusted hidden-G candidate: `{_as_float(report.get('benchmark_trusted_candidate')):.4f}`",
                ]
            )

    lines.extend(["", "## Decision", "", f"- Action: `{decision.action}`"])
    for note in decision.rationale:
        lines.append(f"- Rationale: {note}")

    lines.extend(["", "## Next Tickets", ""])
    for ticket in decision.next_tickets:
        lines.append(f"- `{ticket}`")
    return "\n".join(lines) + "\n"


def _append_runs(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        current = _load_json(path)
        runs_obj = current.get("runs")
        runs = cast("list[object]", runs_obj) if isinstance(runs_obj, list) else []
    else:
        runs = []
    runs.append(payload)
    path.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bounded no-submit competition supervisor report.")
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--backlog-dir", type=Path, required=True)
    parser.add_argument("--ledger-json", type=Path, default=None)
    parser.add_argument("--exactness-report", type=Path, default=None)
    parser.add_argument("--equivalence-json", type=Path, default=None)
    parser.add_argument("--candidate-ceiling-cycle", type=Path, default=None)
    parser.add_argument("--production-mimic-json", type=Path, default=None)
    parser.add_argument("--remaining-signal-json", type=Path, default=None)
    parser.add_argument("--alternative-gate-json", action="append", type=Path, default=[])
    parser.add_argument("--required-safe-baseline-substring", action="append", default=[])
    parser.add_argument("--scoring-json", type=Path, default=None)
    parser.add_argument("--min-ticket", type=int, default=31)
    parser.add_argument("--warmup-budget", type=int, default=10)
    parser.add_argument("--target-rank", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--runs-json", type=Path, default=None)
    args = parser.parse_args()

    rows = load_leaderboard_rows(args.leaderboard)
    subject_summary = build_leaderboard_summary(rows, team_name=args.team)
    latest_experiment = _load_latest_experiment(args.ledger_json)
    exactness_report = _load_exactness_report(args.exactness_report)
    equivalence_report = _load_equivalence_report(args.equivalence_json)
    candidate_ceiling_cycle = _load_candidate_ceiling_cycle(args.candidate_ceiling_cycle)
    production_mimic = _load_production_mimic(args.production_mimic_json)
    remaining_signal_summary = _load_remaining_signal_summary(args.remaining_signal_json)
    alternative_gate_reports = _load_alternative_gate_reports(args.alternative_gate_json)
    scoring_summary = _load_scoring_summary(args.scoring_json)
    ticket_names = _open_tickets(args.backlog_dir, min_ticket=args.min_ticket)
    decision = _decide(
        subject_summary=subject_summary,
        latest_experiment=latest_experiment,
        exactness_report=exactness_report,
        equivalence_report=equivalence_report,
        candidate_ceiling_cycle=candidate_ceiling_cycle,
        production_mimic=production_mimic,
        remaining_signal_summary=remaining_signal_summary,
        alternative_gate_reports=alternative_gate_reports,
        required_safe_baseline_substrings=[
            str(item).strip() for item in args.required_safe_baseline_substring if str(item).strip()
        ],
        ticket_names=ticket_names,
        warmup_budget=args.warmup_budget,
        target_rank=args.target_rank,
    )
    report = _render_report(
        team_name=args.team,
        subject_summary=subject_summary,
        latest_experiment=latest_experiment,
        scoring_summary=scoring_summary,
        candidate_ceiling_cycle=candidate_ceiling_cycle,
        production_mimic=production_mimic,
        alternative_gate_reports=alternative_gate_reports,
        decision=decision,
    )

    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report)

    if args.runs_json is not None:
        run_payload = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "team": args.team,
            "decision": asdict(decision),
            "leaderboard_summary": subject_summary,
            "latest_experiment_label": latest_experiment.get("label") if latest_experiment is not None else None,
            "production_mimic_candidate_class": production_mimic.get("candidate_class") if production_mimic is not None else None,
            "strict_estimate_upper_bound_total": (
                cast("dict[str, object]", scoring_summary.get("exactness_estimate") or {}).get(
                    "strict_upper_bound_total_if_all_answer_changes_are_real"
                )
                if scoring_summary is not None
                else None
            ),
        }
        _append_runs(args.runs_json, run_payload)


if __name__ == "__main__":
    main()
