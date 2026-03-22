from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

try:
    from analyze_leaderboard import build_summary as build_leaderboard_summary
    from analyze_leaderboard import load_rows as load_leaderboard_rows
except ModuleNotFoundError:  # pragma: no cover
    from scripts.analyze_leaderboard import build_summary as build_leaderboard_summary
    from scripts.analyze_leaderboard import load_rows as load_leaderboard_rows

from shafi.eval.production_mimic import (
    JsonDict,
    JsonList,
    build_public_history_calibration,
    estimate_production_mimic,
)


@dataclass(frozen=True)
class CandidateEvalTask:
    leaderboard: str
    team_name: str
    candidate_cycle_json: str
    candidate_label: str
    exactness_json: str | None
    equivalence_json: str | None
    cheap_eval_json: str | None
    strict_eval_json: str | None
    history_json: str | None
    scaffold_json: str | None
    page_trace_json: str | None


def _load_json(path: Path | None) -> JsonDict | None:
    if path is None or not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _load_json_list(path: Path | None) -> JsonList | None:
    if path is None or not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list in {path}")
    out: JsonList = []
    for raw in cast("list[object]", obj):
        if isinstance(raw, dict):
            out.append(cast("JsonDict", raw))
    return out


def _load_history_rows(path: Path | None) -> list[JsonDict]:
    payload = _load_json(path)
    if payload is None:
        return []
    rows_obj = payload.get("rows")
    if not isinstance(rows_obj, list):
        return []
    out: list[JsonDict] = []
    for raw in cast("list[object]", rows_obj):
        if isinstance(raw, dict):
            out.append(cast("JsonDict", raw))
    return out


def _load_candidate_row(path: Path, *, label: str) -> JsonDict:
    payload = _load_json(path)
    if payload is None:
        raise ValueError(f"Missing candidate cycle JSON: {path}")
    ranked_obj = payload.get("ranked_candidates")
    if not isinstance(ranked_obj, list):
        raise ValueError(f"Candidate cycle JSON missing ranked_candidates: {path}")
    for raw in cast("list[object]", ranked_obj):
        if not isinstance(raw, dict):
            continue
        row = cast("JsonDict", raw)
        if str(row.get("label") or "").strip() == label:
            return row
    raise ValueError(f"Candidate label not found in {path}: {label}")


def _resolve_optional_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    return Path(raw).expanduser()


def _rank_for_total(
    *,
    leaderboard_path: Path,
    team_name: str,
    total: float,
) -> int:
    rows = load_leaderboard_rows(leaderboard_path)
    return 1 + sum(1 for row in rows if row.team_name != team_name and row.total > total + 1e-9)


def _render_markdown(
    *,
    label: str,
    leaderboard_path: Path,
    team_name: str,
    production_mimic: JsonDict,
) -> str:
    hidden_g_trusted = cast("JsonDict", production_mimic.get("hidden_g_trusted") or {})
    hidden_g_all = cast("JsonDict", production_mimic.get("hidden_g_all") or {})
    exactness = cast("JsonDict", production_mimic.get("exactness") or {})
    judge = cast("JsonDict", production_mimic.get("judge") or {})
    judge_penalties = cast("JsonDict", production_mimic.get("judge_penalties") or {})
    eval_block = cast("JsonDict", production_mimic.get("eval") or {})
    page_trace = cast("JsonDict", production_mimic.get("page_trace") or {})
    trusted_bootstrap = cast("JsonDict", page_trace.get("trusted_bootstrap") or {})
    policy_debt = cast("JsonDict", production_mimic.get("policy_debt") or {})
    platform_like_total = cast("float", production_mimic.get("platform_like_total_estimate", 0.0))
    strict_total = cast("float", production_mimic.get("strict_total_estimate", 0.0))
    strict_raw_total = cast("float", production_mimic.get("strict_raw_total_estimate", strict_total))
    strict_policy_blocked_total = cast(
        "float",
        production_mimic.get("strict_policy_blocked_total_estimate", strict_total),
    )
    paranoid_total = cast("float", production_mimic.get("paranoid_total_estimate", 0.0))
    lines = [
        "# Production-Mimic Local Eval",
        "",
        f"- label: `{label}`",
        f"- leaderboard: `{leaderboard_path}`",
        f"- team: `{team_name}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "## Candidate Envelope",
        "",
        f"- candidate_class: `{production_mimic.get('candidate_class')}`",
        f"- lineage_confidence: `{production_mimic.get('lineage_confidence')}`",
        f"- submit_eligibility: `{production_mimic.get('submit_eligibility')}`",
        f"- no_submit_reason: `{production_mimic.get('no_submit_reason') or 'none'}`",
        "",
        "## Hidden-G",
        "",
        f"- trusted delta: `{cast('float', hidden_g_trusted.get('delta', 0.0)):.4f}`",
        f"- all-case delta: `{cast('float', hidden_g_all.get('delta', 0.0)):.4f}`",
        "",
        "## Exactness",
        "",
        f"- resolved_incorrect_qids: `{len(cast('list[object]', exactness.get('resolved_incorrect_qids') or []))}`",
        f"- unresolved_incorrect_qids: `{len(cast('list[object]', exactness.get('still_mismatched_incorrect_qids') or []))}`",
        "",
        "## Hybrid Strict Judge",
        "",
        f"- pass_rate: `{judge.get('pass_rate')}`",
        f"- avg_grounding: `{judge.get('avg_grounding')}`",
        f"- avg_accuracy: `{judge.get('avg_accuracy')}`",
        f"- judge_failures: `{judge.get('judge_failures')}`",
        f"- disagreement: `{judge.get('disagreement')}`",
        f"- strict_requested: `{judge.get('strict_requested')}`",
        f"- strict_used: `{judge.get('strict_used')}`",
        f"- strict_skip_reason: `{judge.get('strict_skip_reason') or 'none'}`",
        f"- cache_hit_count: `{cast('JsonDict', judge.get('cache') or {}).get('cache_hit_count')}`",
        f"- cache_miss_count: `{cast('JsonDict', judge.get('cache') or {}).get('cache_miss_count')}`",
        f"- shared_cache_key_count: `{cast('JsonDict', judge.get('cache') or {}).get('shared_cache_key_count')}`",
        "",
        "## Judge Penalties",
        "",
        f"- pass_rate_penalty: `{judge_penalties.get('pass_rate_penalty')}`",
        f"- grounding_penalty: `{judge_penalties.get('grounding_penalty')}`",
        f"- accuracy_penalty: `{judge_penalties.get('accuracy_penalty')}`",
        f"- disagreement_penalty: `{judge_penalties.get('disagreement_penalty')}`",
        f"- timeout_penalty: `{judge_penalties.get('timeout_penalty')}`",
        f"- total: `{judge_penalties.get('total')}`",
        "",
        "## Eval Aggregates",
        "",
        f"- citation_coverage: `{eval_block.get('citation_coverage')}`",
        f"- citation_coverage_by_answer_type: `{eval_block.get('citation_coverage_by_answer_type')}`",
        f"- citation_floor_failure_count: `{eval_block.get('citation_floor_failure_count')}`",
        f"- citation_floor_failure_answer_types: `{eval_block.get('citation_floor_failure_answer_types')}`",
        f"- citation_floor_failures: `{eval_block.get('citation_floor_failures')}`",
        f"- answer_type_format_compliance: `{eval_block.get('answer_type_format_compliance')}`",
        f"- grounding_g_score_beta_2_5: `{eval_block.get('grounding_g_score_beta_2_5')}`",
        "",
        "## Page Trace",
        "",
        f"- cases_scored: `{page_trace.get('cases_scored')}`",
        f"- trusted_case_count: `{page_trace.get('trusted_case_count')}`",
        f"- gold_in_retrieved_count: `{page_trace.get('gold_in_retrieved_count')}`",
        f"- gold_in_reranked_count: `{page_trace.get('gold_in_reranked_count')}`",
        f"- gold_in_used_count: `{page_trace.get('gold_in_used_count')}`",
        f"- false_positive_case_count: `{page_trace.get('false_positive_case_count')}`",
        f"- page_precision: `{page_trace.get('page_precision')}`",
        f"- page_recall: `{page_trace.get('page_recall')}`",
        f"- trusted_page_precision: `{page_trace.get('trusted_page_precision')}`",
        f"- trusted_page_recall: `{page_trace.get('trusted_page_recall')}`",
        f"- trusted_bootstrap_record_count: `{trusted_bootstrap.get('record_count')}`",
        f"- trusted_bootstrap_samples: `{trusted_bootstrap.get('sample_count')}`",
        f"- trusted_bootstrap_f_beta_2_5_p05_p50_p95: `({trusted_bootstrap.get('f_beta_2_5_p05')}, {trusted_bootstrap.get('f_beta_2_5_p50')}, {trusted_bootstrap.get('f_beta_2_5_p95')})`",
        f"- trusted_bootstrap_precision_p05_p50_p95: `({trusted_bootstrap.get('precision_p05')}, {trusted_bootstrap.get('precision_p50')}, {trusted_bootstrap.get('precision_p95')})`",
        f"- trusted_bootstrap_recall_p05_p50_p95: `({trusted_bootstrap.get('recall_p05')}, {trusted_bootstrap.get('recall_p50')}, {trusted_bootstrap.get('recall_p95')})`",
        f"- trusted_bootstrap_unstable_small_slice: `{trusted_bootstrap.get('unstable_small_slice')}`",
        f"- explained_ratio: `{page_trace.get('explained_ratio')}`",
        "",
        "## Score Envelope",
        "",
        f"- strict_raw_total_estimate: `{strict_raw_total:.6f}`",
        f"- strict_policy_blocked_total_estimate: `{strict_policy_blocked_total:.6f}`",
        f"- platform_like_total_estimate: `{platform_like_total:.6f}`",
        f"- strict_total_estimate: `{strict_total:.6f}`",
        f"- paranoid_total_estimate: `{paranoid_total:.6f}`",
        f"- platform_like_rank_estimate: `{_rank_for_total(leaderboard_path=leaderboard_path, team_name=team_name, total=platform_like_total)}`",
        f"- strict_rank_estimate: `{_rank_for_total(leaderboard_path=leaderboard_path, team_name=team_name, total=strict_total)}`",
        f"- paranoid_rank_estimate: `{_rank_for_total(leaderboard_path=leaderboard_path, team_name=team_name, total=paranoid_total)}`",
        "",
        "## Policy Debt",
        "",
        f"- total: `{policy_debt.get('total')}`",
        f"- lineage_penalty: `{policy_debt.get('lineage_penalty')}`",
        f"- unresolved_exactness_penalty: `{policy_debt.get('unresolved_exactness_penalty')}`",
        f"- format_penalty: `{policy_debt.get('format_penalty')}`",
        f"- citation_aggregate_penalty: `{policy_debt.get('citation_aggregate_penalty')}`",
        f"- citation_floor_penalty: `{policy_debt.get('citation_floor_penalty')}`",
        f"- citation_page_trace_penalty: `{policy_debt.get('citation_page_trace_penalty')}`",
        f"- grounding_penalty: `{policy_debt.get('grounding_penalty')}`",
        f"- judge_penalty: `{policy_debt.get('judge_penalty')}`",
        f"- page_drift_without_gain_penalty: `{policy_debt.get('page_drift_without_gain_penalty')}`",
        f"- page_precision_penalty: `{policy_debt.get('page_precision_penalty')}`",
        f"- page_recall_penalty: `{policy_debt.get('page_recall_penalty')}`",
        f"- trusted_slice_penalty: `{policy_debt.get('trusted_slice_penalty')}`",
        f"- page_trace_explained_penalty: `{policy_debt.get('page_trace_explained_penalty')}`",
        f"- support_shape_penalty: `{policy_debt.get('support_shape_penalty')}`",
    ]
    return "\n".join(lines) + "\n"


def _build_payload_for_task(task: CandidateEvalTask) -> tuple[str, JsonDict, str]:
    leaderboard_path = Path(task.leaderboard).expanduser()
    subject_summary = build_leaderboard_summary(load_leaderboard_rows(leaderboard_path), team_name=task.team_name)
    candidate_row = _load_candidate_row(Path(task.candidate_cycle_json).expanduser(), label=task.candidate_label)
    calibration = build_public_history_calibration(_load_history_rows(_resolve_optional_path(task.history_json)))
    raw_results_path = Path(str(candidate_row.get("raw_results") or "")).expanduser()
    raw_results_payload = _load_json_list(raw_results_path if raw_results_path.exists() else None)
    result = estimate_production_mimic(
        subject_summary=subject_summary,
        candidate_row=candidate_row,
        exactness_report=_load_json(_resolve_optional_path(task.exactness_json)),
        equivalence_report=_load_json(_resolve_optional_path(task.equivalence_json)),
        cheap_eval_payload=_load_json(_resolve_optional_path(task.cheap_eval_json)),
        strict_eval_payload=_load_json(_resolve_optional_path(task.strict_eval_json)),
        calibration=calibration,
        raw_results_payload=raw_results_payload,
        scaffold_payload=_load_json(_resolve_optional_path(task.scaffold_json)),
        page_trace_payload=_load_json(_resolve_optional_path(task.page_trace_json)),
    )
    result["platform_like_rank_estimate"] = _rank_for_total(
        leaderboard_path=leaderboard_path,
        team_name=task.team_name,
        total=cast("float", result.get("platform_like_total_estimate", 0.0)),
    )
    result["strict_rank_estimate"] = _rank_for_total(
        leaderboard_path=leaderboard_path,
        team_name=task.team_name,
        total=cast("float", result.get("strict_total_estimate", 0.0)),
    )
    result["paranoid_rank_estimate"] = _rank_for_total(
        leaderboard_path=leaderboard_path,
        team_name=task.team_name,
        total=cast("float", result.get("paranoid_total_estimate", 0.0)),
    )
    payload: JsonDict = {
        "leaderboard": str(leaderboard_path),
        "team_name": task.team_name,
        "candidate_label": task.candidate_label,
        "candidate_row": candidate_row,
        "calibration": calibration,
        "production_mimic": result,
        "parallel_eval_mode": "offline_only",
        "canonical_candidate_build_concurrency": 1,
    }
    markdown = _render_markdown(
        label=task.candidate_label,
        leaderboard_path=leaderboard_path,
        team_name=task.team_name,
        production_mimic=result,
    )
    return task.candidate_label, payload, markdown


def _evaluate_tasks(tasks: list[CandidateEvalTask], *, parallel_workers: int) -> list[tuple[str, JsonDict, str]]:
    if parallel_workers <= 1 or len(tasks) <= 1:
        return [_build_payload_for_task(task) for task in tasks]
    worker_count = min(parallel_workers, len(tasks))
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("spawn")) as executor:
        return list(executor.map(_build_payload_for_task, tasks))


def _json_text(payload: JsonDict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _payload_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_batch_markdown(*, labels: list[str], parallel_workers: int, batch_summary: JsonDict) -> str:
    artifacts = cast("list[JsonDict]", batch_summary.get("artifacts") or [])
    lines = [
        "# Production-Mimic Batch Eval",
        "",
        f"- labels: `{labels}`",
        f"- parallel_workers_used: `{parallel_workers}`",
        "- parallel_eval_mode: `offline_only`",
        "- canonical_candidate_build_concurrency: `1`",
        "- deterministic_inputs_sorted: `true`",
        "",
        "## Artifacts",
        "",
    ]
    for row in artifacts:
        lines.extend(
            [
                f"### `{row['candidate_label']}`",
                f"- json_path: `{row['out_json']}`",
                f"- markdown_path: `{row['out_md']}`",
                f"- json_sha256: `{row['json_sha256']}`",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict production-like local evaluation envelope for one candidate.")
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--candidate-cycle-json", type=Path, required=True)
    parser.add_argument("--candidate-label", action="append", required=True)
    parser.add_argument("--exactness-json", type=Path, default=None)
    parser.add_argument("--equivalence-json", type=Path, default=None)
    parser.add_argument("--cheap-eval-json", type=Path, default=None)
    parser.add_argument("--strict-eval-json", type=Path, default=None)
    parser.add_argument("--history-json", type=Path, default=None)
    parser.add_argument("--scaffold-json", type=Path, default=None)
    parser.add_argument("--page-trace-json", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--batch-out-dir", type=Path, default=None)
    parser.add_argument("--parallel-workers", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = sorted({label.strip() for label in args.candidate_label if label.strip()})
    tasks = [
        CandidateEvalTask(
            leaderboard=str(args.leaderboard.resolve()),
            team_name=args.team,
            candidate_cycle_json=str(args.candidate_cycle_json.resolve()),
            candidate_label=label,
            exactness_json=None if args.exactness_json is None else str(args.exactness_json.resolve()),
            equivalence_json=None if args.equivalence_json is None else str(args.equivalence_json.resolve()),
            cheap_eval_json=None if args.cheap_eval_json is None else str(args.cheap_eval_json.resolve()),
            strict_eval_json=None if args.strict_eval_json is None else str(args.strict_eval_json.resolve()),
            history_json=None if args.history_json is None else str(args.history_json.resolve()),
            scaffold_json=None if args.scaffold_json is None else str(args.scaffold_json.resolve()),
            page_trace_json=None if args.page_trace_json is None else str(args.page_trace_json.resolve()),
        )
        for label in labels
    ]
    results = _evaluate_tasks(tasks, parallel_workers=max(1, int(args.parallel_workers)))

    if args.batch_out_dir is None:
        if len(results) != 1:
            raise ValueError("Single-output mode requires exactly one --candidate-label; use --batch-out-dir for multiple labels")
        if args.out_json is None or args.out_md is None:
            raise ValueError("--out-json and --out-md are required in single-output mode")
        _label, payload, markdown = results[0]
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(_json_text(payload), encoding="utf-8")
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown, encoding="utf-8")
        return 0

    batch_out_dir = args.batch_out_dir.resolve()
    batch_out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[JsonDict] = []
    worker_count = min(max(1, int(args.parallel_workers)), len(results))
    for label, payload, markdown in results:
        candidate_dir = batch_out_dir / label
        out_json = candidate_dir / "production_mimic.json"
        out_md = candidate_dir / "production_mimic.md"
        relative_json = f"{label}/production_mimic.json"
        relative_md = f"{label}/production_mimic.md"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        json_text = _json_text(payload)
        out_json.write_text(json_text, encoding="utf-8")
        out_md.write_text(markdown, encoding="utf-8")
        artifacts.append(
            {
                "candidate_label": label,
                "out_json": relative_json,
                "out_md": relative_md,
                "json_sha256": _payload_sha256(json_text),
            }
        )
    batch_summary: JsonDict = {
        "leaderboard": str(args.leaderboard.resolve()),
        "team_name": args.team,
        "labels": labels,
        "parallel_eval_mode": "offline_only",
        "parallel_workers_used": worker_count,
        "canonical_candidate_build_concurrency": 1,
        "deterministic_inputs_sorted": True,
        "artifacts": artifacts,
    }
    (batch_out_dir / "batch_summary.json").write_text(_json_text(batch_summary), encoding="utf-8")
    (batch_out_dir / "batch_summary.md").write_text(
        _render_batch_markdown(labels=labels, parallel_workers=worker_count, batch_summary=batch_summary),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
