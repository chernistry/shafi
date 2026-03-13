from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

try:
    from analyze_leaderboard import (  # noqa: I001
        LeaderboardRow,
        build_summary as build_leaderboard_summary,
        load_rows as load_leaderboard_rows,
    )
    from impact_router import route_changed_files
except ModuleNotFoundError:  # pragma: no cover - import path differs under pytest/module import
    from scripts.analyze_leaderboard import (  # noqa: I001
        LeaderboardRow,
        build_summary as build_leaderboard_summary,
        load_rows as load_leaderboard_rows,
    )
    from scripts.impact_router import route_changed_files

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    submission: Path
    raw_results: Path
    preflight: Path | None
    candidate_scaffold: Path | None
    allowed_answer_qids: list[str]
    allowed_page_qids: list[str]
    changed_files: list[str]
    completed_packs: list[str]
    branch_class: str
    timeline_scope: str


_FROZEN_BRANCH_CLASS_REASONS: dict[str, str] = {
    "small_diff_support_rider": "proven dead mechanism: small-diff support rider ceiling",
    "broad_page_inflation": "proven dead mechanism: broad page inflation",
    "full_collection_embedder_swap": "proven dead mechanism: full-collection embedder swap",
    "global_prompt_churn": "proven dead mechanism: global prompt churn",
    "latency_vps_chase_for_score_only": "proven dead mechanism: latency/VPS score chase",
}


def _resolve(root: Path, raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (root / value).resolve()


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _coerce_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _coerce_branch_class(value: object) -> str:
    text = str(value or "").strip()
    return text or "unclassified"


def _coerce_timeline_scope(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"private_only", "deferred_after_18", "after_private_release"}:
        return "private_only"
    return "active"


def _load_qid_set(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _candidate_paths(*, out_dir: Path, label: str) -> dict[str, Path]:
    candidate_dir = out_dir / label
    return {
        "dir": candidate_dir,
        "allowed_answer_qids": candidate_dir / "allowed_answer_qids.txt",
        "allowed_page_qids": candidate_dir / "allowed_page_qids.txt",
        "lineage_json": candidate_dir / "lineage.json",
        "lineage_md": candidate_dir / "lineage.md",
        "gate_json": candidate_dir / "gate.json",
        "gate_md": candidate_dir / "gate.md",
        "exactness_json": candidate_dir / "exactness.json",
        "exactness_md": candidate_dir / "exactness.md",
    }


def _load_manifest(path: Path, *, root: Path) -> list[CandidateSpec]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    candidates_obj = cast("JsonDict", obj).get("candidates")
    if not isinstance(candidates_obj, list):
        raise ValueError(f"Manifest at {path} is missing candidates[]")
    out: list[CandidateSpec] = []
    seen: set[str] = set()
    candidates = cast("list[object]", candidates_obj)
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        row = cast("JsonDict", raw)
        label = str(row.get("label") or "").strip()
        if not label:
            raise ValueError("Candidate manifest entry missing label")
        if label in seen:
            raise ValueError(f"Duplicate candidate label in manifest: {label}")
        seen.add(label)
        submission = _resolve(root, row.get("submission"))
        raw_results = _resolve(root, row.get("raw_results"))
        if submission is None or raw_results is None:
            raise ValueError(f"Candidate {label} is missing submission/raw_results")
        out.append(
            CandidateSpec(
                label=label,
                submission=submission,
                raw_results=raw_results,
                preflight=_resolve(root, row.get("preflight")),
                candidate_scaffold=_resolve(root, row.get("candidate_scaffold")),
                allowed_answer_qids=_coerce_str_list(row.get("allowed_answer_qids")),
                allowed_page_qids=_coerce_str_list(row.get("allowed_page_qids")),
                changed_files=_coerce_str_list(row.get("changed_files")),
                completed_packs=_coerce_str_list(row.get("completed_packs")),
                branch_class=_coerce_branch_class(row.get("branch_class")),
                timeline_scope=_coerce_timeline_scope(row.get("timeline_scope")),
            )
        )
    if not out:
        raise ValueError(f"No valid candidates in manifest: {path}")
    return out


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    out: list[str] = []
    for raw in items:
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def _write_qids(path: Path, qids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{qid}\n" for qid in qids), encoding="utf-8")


def _verify_lineage(
    *,
    root: Path,
    baseline_submission: Path,
    candidate: CandidateSpec,
    paths: dict[str, Path],
) -> JsonDict:
    _write_qids(paths["allowed_answer_qids"], candidate.allowed_answer_qids)
    _write_qids(paths["allowed_page_qids"], candidate.allowed_page_qids)
    cmd = [
        sys.executable,
        "scripts/verify_candidate_lineage.py",
        "--baseline-submission",
        str(baseline_submission),
        "--candidate-submission",
        str(candidate.submission),
        "--allowed-answer-qids-file",
        str(paths["allowed_answer_qids"]),
        "--allowed-page-qids-file",
        str(paths["allowed_page_qids"]),
        "--out-json",
        str(paths["lineage_json"]),
        "--out-md",
        str(paths["lineage_md"]),
    ]
    _run(cmd, cwd=root)
    return _load_json(paths["lineage_json"])


def _run_gate(
    *,
    root: Path,
    baseline_label: str,
    baseline_submission: Path,
    baseline_raw_results: Path,
    baseline_preflight: Path | None,
    benchmark: Path,
    scaffold: Path,
    candidate: CandidateSpec,
    paths: dict[str, Path],
) -> JsonDict:
    cmd = [
        sys.executable,
        "scripts/run_experiment_gate.py",
        "--label",
        candidate.label,
        "--baseline-label",
        baseline_label,
        "--baseline-submission",
        str(baseline_submission),
        "--candidate-submission",
        str(candidate.submission),
        "--baseline-raw-results",
        str(baseline_raw_results),
        "--candidate-raw-results",
        str(candidate.raw_results),
        "--benchmark",
        str(benchmark),
        "--scaffold",
        str(scaffold),
        "--out",
        str(paths["gate_md"]),
        "--out-json",
        str(paths["gate_json"]),
    ]
    if candidate.candidate_scaffold is not None:
        cmd.extend(["--candidate-scaffold", str(candidate.candidate_scaffold)])
    if baseline_preflight is not None:
        cmd.extend(["--baseline-preflight", str(baseline_preflight)])
    if candidate.preflight is not None:
        cmd.extend(["--candidate-preflight", str(candidate.preflight)])
    _run(cmd, cwd=root)
    return _load_json(paths["gate_json"])


def _run_exactness(
    *,
    root: Path,
    baseline_label: str,
    baseline_submission: Path,
    scaffold: Path,
    candidate: CandidateSpec,
    paths: dict[str, Path],
) -> JsonDict:
    cmd = [
        sys.executable,
        "scripts/audit_exactness_candidate.py",
        "--baseline-label",
        baseline_label,
        "--baseline-submission",
        str(baseline_submission),
        "--candidate-label",
        candidate.label,
        "--candidate-submission",
        str(candidate.submission),
        "--truth-audit-scaffold",
        str(scaffold),
        "--out-json",
        str(paths["exactness_json"]),
        "--out-md",
        str(paths["exactness_md"]),
        "--judge-scope",
        "none",
    ]
    _run(cmd, cwd=root)
    return _load_json(paths["exactness_json"])


def _run_family_debug(
    *,
    root: Path,
    baseline_label: str,
    baseline_raw_results: Path,
    questions: Path,
    docs_dir: Path,
    include_qids_file: Path,
    candidates: list[CandidateSpec],
    out_dir: Path,
    judge_scope: str,
) -> JsonDict:
    out_json = out_dir / "family_debug_rank.json"
    out_md = out_dir / "family_debug_rank.md"
    cmd = [
        sys.executable,
        "scripts/compare_candidate_family_debug.py",
        "--baseline-label",
        baseline_label,
        "--baseline-raw-results",
        str(baseline_raw_results),
        "--questions",
        str(questions),
        "--docs-dir",
        str(docs_dir),
        "--include-qids-file",
        str(include_qids_file),
        "--family-label",
        "candidate_ceiling_portfolio",
        "--out-dir",
        str(out_dir),
        "--judge-scope",
        judge_scope,
        "--case-scope",
        "all",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    for candidate in candidates:
        cmd.extend(["--candidate", f"{candidate.label}={candidate.raw_results}"])
    _run(cmd, cwd=root)
    return _load_json(out_json)


def _combined_score(
    row: JsonDict,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float, int, int, str]:
    active_before_deadline = bool(row.get("active_before_march17", True))
    branch_active = str(row.get("branch_status") or "active") == "active"
    impacted_packs_ok = not bool(row.get("impact_router_blocked"))
    lineage_ok = bool(row.get("lineage_ok"))
    recommendation = str(row.get("recommendation") or "")
    recommendation_bonus = {"PROMISING": 2.0, "EXPERIMENTAL_NO_SUBMIT": 1.0}.get(recommendation.upper(), 0.0)
    return (
        1.0 if active_before_deadline else 0.0,
        1.0 if branch_active else 0.0,
        1.0 if impacted_packs_ok else 0.0,
        1.0 if lineage_ok else 0.0,
        recommendation_bonus,
        _coerce_float(row.get("paranoid_total_estimate")),
        _coerce_float(row.get("strict_total_estimate")),
        _coerce_float(row.get("upper_total_estimate")),
        float(_coerce_int(row.get("blindspot_support_undercoverage_case_count"))),
        float(_coerce_int(row.get("blindspot_improved_case_count"))),
        _coerce_float(row.get("hidden_g_trusted_delta")),
        _coerce_float(row.get("hidden_g_all_delta")),
        _coerce_float(row.get("judge_pass_delta")),
        _coerce_float(row.get("judge_grounding_delta")) + (_coerce_int(row.get("resolved_incorrect_count")) * 0.5),
        -_coerce_int(row.get("page_drift")),
        -_coerce_int(row.get("answer_drift")),
        str(row.get("label") or ""),
    )


def _summarize_branch_classes(*, rows: list[JsonDict], target_rank: int) -> list[JsonDict]:
    grouped: dict[str, list[JsonDict]] = {}
    for row in rows:
        branch_class = _coerce_branch_class(row.get("branch_class"))
        grouped.setdefault(branch_class, []).append(row)

    summaries: list[JsonDict] = []
    for branch_class, branch_rows in grouped.items():
        best_upper = min(branch_rows, key=lambda row: _coerce_int(row.get("upper_rank_estimate")) or 10_000)
        best_strict = min(branch_rows, key=lambda row: _coerce_int(row.get("strict_rank_estimate")) or 10_000)
        best_paranoid = min(branch_rows, key=lambda row: _coerce_int(row.get("paranoid_rank_estimate")) or 10_000)
        best_upper_rank = _coerce_int(best_upper.get("upper_rank_estimate")) or 10_000
        best_strict_rank = _coerce_int(best_strict.get("strict_rank_estimate")) or 10_000
        best_paranoid_rank = _coerce_int(best_paranoid.get("paranoid_rank_estimate")) or 10_000
        timeline_scopes = sorted({str(row.get("timeline_scope") or "active") for row in branch_rows})

        status = "active"
        reason = "active pre-March-17 class"
        active_before_march17 = True
        if branch_class in _FROZEN_BRANCH_CLASS_REASONS:
            status = "frozen"
            reason = _FROZEN_BRANCH_CLASS_REASONS[branch_class]
            active_before_march17 = False
        elif all(scope == "private_only" for scope in timeline_scopes):
            status = "private_only"
            reason = "reserved for after March 18 / private-phase work"
            active_before_march17 = False
        elif (
            best_upper_rank > target_rank
            and best_strict_rank > target_rank
            and best_paranoid_rank > target_rank
        ):
            status = "frozen"
            reason = f"mathematical ceiling misses target rank {target_rank} even under upper estimate"
            active_before_march17 = False

        summaries.append(
            {
                "branch_class": branch_class,
                "status": status,
                "reason": reason,
                "active_before_march17": active_before_march17,
                "timeline_scopes": timeline_scopes,
                "candidate_count": len(branch_rows),
                "promising_candidate_count": sum(
                    1 for row in branch_rows if str(row.get("recommendation") or "").upper() == "PROMISING"
                ),
                "best_label": str(best_upper.get("label") or ""),
                "best_upper_rank_estimate": best_upper_rank,
                "best_strict_rank_estimate": best_strict_rank,
                "best_paranoid_rank_estimate": best_paranoid_rank,
                "best_upper_total_estimate": _coerce_float(best_upper.get("upper_total_estimate")),
            }
        )

    summaries.sort(
        key=lambda row: (
            1.0 if bool(row.get("active_before_march17")) else 0.0,
            _coerce_float(row.get("best_upper_total_estimate")),
            str(row.get("branch_class") or ""),
        ),
        reverse=True,
    )
    return summaries


def _apply_branch_class_policy(*, rows: list[JsonDict], target_rank: int) -> list[JsonDict]:
    summaries = _summarize_branch_classes(rows=rows, target_rank=target_rank)
    by_branch = {
        _coerce_branch_class(summary.get("branch_class")): summary
        for summary in summaries
    }
    for row in rows:
        summary = by_branch.get(_coerce_branch_class(row.get("branch_class")))
        if summary is None:
            row["branch_status"] = "active"
            row["branch_status_reason"] = "active pre-March-17 class"
            row["active_before_march17"] = True
            continue
        row["branch_status"] = summary.get("status")
        row["branch_status_reason"] = summary.get("reason")
        row["active_before_march17"] = bool(summary.get("active_before_march17"))
    return summaries


def _apply_impact_router(candidate: CandidateSpec, row: JsonDict) -> None:
    impact = route_changed_files(candidate.changed_files, completed_packs=candidate.completed_packs)
    row["required_packs"] = impact.get("required_packs") or []
    row["completed_packs"] = impact.get("completed_packs") or []
    row["missing_packs"] = impact.get("missing_packs") or []
    row["impact_router_blocked"] = bool(impact.get("should_block_promotion"))
    if row["impact_router_blocked"]:
        row["recommendation"] = "BLOCKED_MISSING_IMPACT_PACK"


def _paranoid_total_penalty(row: JsonDict) -> float:
    page_drift = _coerce_int(row.get("page_drift"))
    answer_drift = _coerce_int(row.get("answer_drift"))
    page_p95 = _coerce_int(row.get("page_p95"))
    still_mismatched = _coerce_int(row.get("still_mismatched_incorrect_count"))
    recommendation = str(row.get("recommendation") or "").upper()
    judge_timeout = bool(row.get("judge_timeout"))
    penalty = 0.0
    penalty += 0.0025 * page_drift
    penalty += 0.0015 * answer_drift
    penalty += 0.0020 * still_mismatched
    penalty += 0.0010 * max(0, page_p95 - 4)
    if recommendation != "PROMISING":
        penalty += 0.0030
    if judge_timeout:
        penalty += 0.0040
    return penalty


def _rank_for_total(rows: list[LeaderboardRow], *, team_name: str, total: float) -> int:
    return 1 + sum(1 for row in rows if row.team_name != team_name and row.total > total + 1e-9)


def _candidate_score_estimates(
    *,
    row: JsonDict,
    subject_summary: JsonDict,
    leaderboard_rows: list[LeaderboardRow],
    team_name: str,
    public_realized_exactness_qids: set[str] | None,
) -> JsonDict:
    current_s = _coerce_float(subject_summary.get("s"))
    current_g = _coerce_float(subject_summary.get("g"))
    current_t = _coerce_float(subject_summary.get("t"))
    current_f = _coerce_float(subject_summary.get("f"))
    current_total = _coerce_float(subject_summary.get("total"))

    resolved_qids = _coerce_str_list(row.get("resolved_incorrect_qids"))
    if public_realized_exactness_qids is None:
        strict_resolved_qids: list[str] = []
        upper_resolved_qids = resolved_qids
        exactness_basis = "strict exactness disabled until public-realized qids are supplied"
    else:
        strict_resolved_qids = [qid for qid in resolved_qids if qid not in public_realized_exactness_qids]
        upper_resolved_qids = strict_resolved_qids
        exactness_basis = "counts only qids not already realized in the public baseline"

    strict_s = current_s + (0.01 * len(strict_resolved_qids))
    upper_s = current_s + (0.01 * len(upper_resolved_qids))
    strict_g = current_g + _coerce_float(row.get("hidden_g_all_delta"))
    upper_g = current_g + max(_coerce_float(row.get("hidden_g_all_delta")), _coerce_float(row.get("hidden_g_trusted_delta")))
    strict_total = strict_s * strict_g * current_t * current_f
    upper_total = upper_s * upper_g * current_t * current_f
    paranoid_penalty = _paranoid_total_penalty(row)
    paranoid_total = max(0.0, strict_total - paranoid_penalty)

    return {
        "strict_resolved_incorrect_qids": strict_resolved_qids,
        "upper_resolved_incorrect_qids": upper_resolved_qids,
        "strict_resolved_incorrect_count": len(strict_resolved_qids),
        "upper_resolved_incorrect_count": len(upper_resolved_qids),
        "strict_s_estimate": strict_s,
        "upper_s_estimate": upper_s,
        "strict_g_estimate": strict_g,
        "upper_g_estimate": upper_g,
        "paranoid_total_penalty": paranoid_penalty,
        "paranoid_total_estimate": paranoid_total,
        "paranoid_total_delta": paranoid_total - current_total,
        "strict_total_estimate": strict_total,
        "upper_total_estimate": upper_total,
        "strict_total_delta": strict_total - current_total,
        "upper_total_delta": upper_total - current_total,
        "paranoid_rank_estimate": _rank_for_total(leaderboard_rows, team_name=team_name, total=paranoid_total),
        "strict_rank_estimate": _rank_for_total(leaderboard_rows, team_name=team_name, total=strict_total),
        "upper_rank_estimate": _rank_for_total(leaderboard_rows, team_name=team_name, total=upper_total),
        "strict_exactness_basis": exactness_basis,
    }


def _render_markdown(
    *,
    rows: list[JsonDict],
    branch_class_summary: list[JsonDict],
    baseline_label: str,
    include_qids_file: Path,
    team_name: str | None,
    leaderboard_path: Path | None,
) -> str:
    lines = [
        "# Candidate Ceiling Cycle",
        "",
        f"- baseline_label: `{baseline_label}`",
        f"- include_qids_file: `{include_qids_file}`",
        f"- candidates: `{len(rows)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
    ]
    if team_name is not None and leaderboard_path is not None:
        lines.extend(
            [
                f"- leaderboard: `{leaderboard_path}`",
                f"- team_name: `{team_name}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Branch Classes",
            "",
            "| Branch class | Status | Active before March 17 | Best label | Best upper rank | Candidates | Reason |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for summary in branch_class_summary:
        lines.append(
            "| "
            f"`{summary['branch_class']}` | `{summary['status']}` | `{summary['active_before_march17']}` | "
            f"`{summary['best_label']}` | `{_coerce_int(summary.get('best_upper_rank_estimate'))}` | "
            f"`{_coerce_int(summary.get('candidate_count'))}` | {summary['reason']} |"
        )
    lines.extend(["", "## Ranked Candidates", ""])
    lines.extend(
        [
            "| Rank | Label | Recommendation | Lineage | Paranoid Total | Strict Total | Upper Total | Paranoid Rank | Strict Rank | Upper Rank | Blindspot Gains | SU Blindspots | Hidden-G Trusted Δ | Hidden-G All Δ | Judge Pass Δ | Judge Grounding Δ | Resolved Exactness | Answer Drift | Page Drift | Page p95 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, row in enumerate(sorted(rows, key=_combined_score, reverse=True), start=1):
        paranoid_total = row.get("paranoid_total_estimate")
        paranoid_rank = row.get("paranoid_rank_estimate")
        strict_total = row.get("strict_total_estimate")
        upper_total = row.get("upper_total_estimate")
        strict_rank = row.get("strict_rank_estimate")
        upper_rank = row.get("upper_rank_estimate")
        paranoid_total_cell = f"{_coerce_float(paranoid_total):.6f}" if paranoid_total is not None else "n/a"
        paranoid_rank_cell = str(_coerce_int(paranoid_rank)) if paranoid_rank is not None else "n/a"
        strict_total_cell = f"{_coerce_float(strict_total):.6f}" if strict_total is not None else "n/a"
        upper_total_cell = f"{_coerce_float(upper_total):.6f}" if upper_total is not None else "n/a"
        strict_rank_cell = str(_coerce_int(strict_rank)) if strict_rank is not None else "n/a"
        upper_rank_cell = str(_coerce_int(upper_rank)) if upper_rank is not None else "n/a"
        lines.append(
            "| "
            f"{index} | `{row['label']}` | `{row['recommendation']}` | `{row['lineage_ok']}` | "
            f"{paranoid_total_cell} | {strict_total_cell} | {upper_total_cell} | {paranoid_rank_cell} | {strict_rank_cell} | {upper_rank_cell} | "
            f"{_coerce_int(row.get('blindspot_improved_case_count'))} | {_coerce_int(row.get('blindspot_support_undercoverage_case_count'))} | "
            f"{_coerce_float(row.get('hidden_g_trusted_delta')):.4f} | {_coerce_float(row.get('hidden_g_all_delta')):.4f} | "
            f"{_coerce_float(row.get('judge_pass_delta')):+.4f} | {_coerce_float(row.get('judge_grounding_delta')):+.4f} | "
            f"{_coerce_int(row.get('resolved_incorrect_count'))} | {_coerce_int(row.get('answer_drift'))} | "
            f"{_coerce_int(row.get('page_drift'))} | {_coerce_int(row.get('page_p95'))} |"
        )
    return "\n".join(lines) + "\n"


def _family_rows_by_label(payload: JsonDict) -> dict[str, JsonDict]:
    ranked_obj = payload.get("ranked_candidates")
    if not isinstance(ranked_obj, list):
        return {}
    rows = cast("list[object]", ranked_obj)
    out: dict[str, JsonDict] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = cast("JsonDict", raw)
        label = str(row.get("label") or "").strip()
        if label:
            out[label] = row
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lineage + gate + exactness + family-debug ranking for a candidate ceiling manifest.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline-submission", required=True)
    parser.add_argument("--baseline-raw-results", required=True)
    parser.add_argument("--baseline-preflight", default=None)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--scaffold", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--judge-scope", choices=("all", "free_text", "none"), default="all")
    parser.add_argument("--leaderboard", default=None)
    parser.add_argument("--team-name", default=None)
    parser.add_argument("--public-realized-exactness-qids-file", default=None)
    parser.add_argument("--target-rank", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline_submission = cast("Path", _resolve(root, args.baseline_submission))
    baseline_raw_results = cast("Path", _resolve(root, args.baseline_raw_results))
    baseline_preflight = _resolve(root, args.baseline_preflight)
    benchmark = cast("Path", _resolve(root, args.benchmark))
    scaffold = cast("Path", _resolve(root, args.scaffold))
    questions = cast("Path", _resolve(root, args.questions))
    docs_dir = cast("Path", _resolve(root, args.docs_dir))
    manifest_json = cast("Path", _resolve(root, args.manifest_json))
    out_dir = cast("Path", _resolve(root, args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = _resolve(root, args.leaderboard)
    public_realized_qids_path = _resolve(root, args.public_realized_exactness_qids_file)

    leaderboard_rows: list[LeaderboardRow] = []
    subject_summary: JsonDict | None = None
    if leaderboard_path is not None:
        if args.team_name is None:
            raise ValueError("--team-name is required when --leaderboard is provided")
        leaderboard_rows = load_leaderboard_rows(leaderboard_path)
        subject_summary = cast("JsonDict", build_leaderboard_summary(leaderboard_rows, team_name=str(args.team_name)))
    public_realized_exactness_qids = _load_qid_set(public_realized_qids_path) if public_realized_qids_path is not None else None

    candidates = _load_manifest(manifest_json, root=root)
    include_qids = sorted({qid for candidate in candidates for qid in [*candidate.allowed_answer_qids, *candidate.allowed_page_qids]})
    include_qids_file = out_dir / "include_qids.txt"
    _write_qids(include_qids_file, include_qids)

    rows: list[JsonDict] = []
    for candidate in candidates:
        paths = _candidate_paths(out_dir=out_dir, label=candidate.label)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        lineage = _verify_lineage(
            root=root,
            baseline_submission=baseline_submission,
            candidate=candidate,
            paths=paths,
        )
        gate = _run_gate(
            root=root,
            baseline_label=str(args.baseline_label),
            baseline_submission=baseline_submission,
            baseline_raw_results=baseline_raw_results,
            baseline_preflight=baseline_preflight,
            benchmark=benchmark,
            scaffold=scaffold,
            candidate=candidate,
            paths=paths,
        )
        exactness = _run_exactness(
            root=root,
            baseline_label=str(args.baseline_label),
            baseline_submission=baseline_submission,
            scaffold=scaffold,
            candidate=candidate,
            paths=paths,
        )
        row: JsonDict = {
            "label": candidate.label,
            "submission": str(candidate.submission),
            "raw_results": str(candidate.raw_results),
            "preflight": None if candidate.preflight is None else str(candidate.preflight),
            "branch_class": candidate.branch_class,
            "timeline_scope": candidate.timeline_scope,
            "recommendation": gate.get("recommendation"),
            "lineage_ok": lineage.get("lineage_ok"),
            "answer_drift": lineage.get("answer_changed_count"),
            "page_drift": lineage.get("page_changed_count"),
            "page_p95": gate.get("candidate_page_p95"),
            "hidden_g_trusted_delta": _coerce_float(gate.get("benchmark_trusted_candidate")) - _coerce_float(gate.get("benchmark_trusted_baseline")),
            "hidden_g_all_delta": _coerce_float(gate.get("benchmark_all_candidate")) - _coerce_float(gate.get("benchmark_all_baseline")),
            "blindspot_improved_case_count": len(cast("list[object]", gate.get("blindspot_improved_cases") or [])),
            "blindspot_support_undercoverage_case_count": len(
                cast("list[object]", gate.get("blindspot_support_undercoverage_cases") or [])
            ),
            "resolved_incorrect_qids": _coerce_str_list(exactness.get("resolved_incorrect_qids")),
            "resolved_incorrect_count": len(cast("list[object]", exactness.get("resolved_incorrect_qids") or [])),
            "still_mismatched_incorrect_count": len(cast("list[object]", exactness.get("still_mismatched_incorrect_qids") or [])),
            "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        }
        _apply_impact_router(candidate, row)
        if subject_summary is not None:
            row.update(
                _candidate_score_estimates(
                    row=row,
                    subject_summary=subject_summary,
                    leaderboard_rows=leaderboard_rows,
                    team_name=str(args.team_name),
                    public_realized_exactness_qids=public_realized_exactness_qids,
                )
            )
        rows.append(row)

    family_debug = _run_family_debug(
        root=root,
        baseline_label=str(args.baseline_label),
        baseline_raw_results=baseline_raw_results,
        questions=questions,
        docs_dir=docs_dir,
        include_qids_file=include_qids_file,
        candidates=candidates,
        out_dir=out_dir / "family_debug",
        judge_scope=str(args.judge_scope),
    )
    family_rows = _family_rows_by_label(family_debug)
    for row in rows:
        family = family_rows.get(str(row["label"]), {})
        row["judge_pass_delta"] = family.get("judge_pass_delta", 0.0)
        row["judge_grounding_delta"] = family.get("judge_grounding_delta", 0.0)
        row["judge_accuracy_delta"] = family.get("judge_accuracy_delta", 0.0)
        row["citation_delta"] = family.get("citation_delta", 0.0)
        row["format_delta"] = family.get("format_delta", 0.0)
        row["judge_timeout"] = family.get("judge_timeout", False)

    branch_class_summary = _apply_branch_class_policy(rows=rows, target_rank=args.target_rank)
    ranked = sorted(rows, key=_combined_score, reverse=True)
    payload = {
        "baseline_label": args.baseline_label,
        "include_qids_file": str(include_qids_file),
        "manifest_json": str(manifest_json),
        "family_debug_json": str((out_dir / "family_debug" / "family_debug_rank.json").resolve()),
        "leaderboard": None if leaderboard_path is None else str(leaderboard_path),
        "team_name": args.team_name,
        "public_realized_exactness_qids_file": None if public_realized_qids_path is None else str(public_realized_qids_path),
        "target_rank": args.target_rank,
        "branch_class_summary": branch_class_summary,
        "ranked_candidates": ranked,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    out_json = out_dir / "candidate_ceiling_cycle.json"
    out_md = out_dir / "candidate_ceiling_cycle.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(
        _render_markdown(
            rows=rows,
            branch_class_summary=branch_class_summary,
            baseline_label=str(args.baseline_label),
            include_qids_file=include_qids_file,
            team_name=args.team_name,
            leaderboard_path=leaderboard_path,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
