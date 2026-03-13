# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from analyze_leaderboard import build_summary as build_leaderboard_summary
from analyze_leaderboard import load_rows as load_leaderboard_rows

JsonDict = dict[str, object]
ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path | None) -> JsonDict:
    if path is None or not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _latest_experiment(ledger: JsonDict) -> JsonDict:
    experiments = ledger.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return {}
    latest = experiments[-1]
    return cast("JsonDict", latest) if isinstance(latest, dict) else {}


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
    exactness_estimate = cast("JsonDict", scoring.get("exactness_estimate") or {})
    anchor_counts = cast("JsonDict", anchor_slice.get("status_counts") or {})

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


def _fmt_float(value: object) -> str:
    if value is None or value == "":
        return "-"
    return f"{_as_float(value):.6f}"


def _fmt_small(value: object) -> str:
    if value is None or value == "":
        return "-"
    return f"{_as_float(value):.4f}"


def _fmt_int(value: object) -> str:
    if value is None or value == "":
        return "-"
    return str(_as_int(value))


def _short_qids(value: object, *, limit: int = 4) -> str:
    if not isinstance(value, list):
        return "-"
    items = [str(item).strip() for item in cast("list[object]", value) if str(item).strip()]
    if not items:
        return "-"
    shown = [item[:6] for item in items[:limit]]
    if len(items) > limit:
        shown.append(f"+{len(items) - limit}")
    return ",".join(shown)


def _parse_public_history_rows(path: Path | None) -> dict[str, JsonDict]:
    if path is None or not path.exists():
        return {}
    history: dict[str, JsonDict] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [part.strip().replace("**", "") for part in line.strip("|").split("|")]
        if len(cells) != 7 or cells[0].lower() == "version" or set(cells[0]) == {"-"}:
            continue
        version = cells[0].split()[0]
        if not version.startswith("v"):
            continue
        try:
            history[version] = {
                "version": version,
                "strategy": cells[1],
                "external_det": float(cells[2]),
                "external_asst": float(cells[3]),
                "external_g": float(cells[4]),
                "external_total": float(cells[5]),
                "result": cells[6],
            }
        except ValueError:
            continue
    return history


def _load_candidate_cycle_index(path: Path | None) -> dict[str, JsonDict]:
    payload = _load_json(path)
    ranked_obj = payload.get("ranked_candidates")
    if not isinstance(ranked_obj, list):
        return {}
    index: dict[str, JsonDict] = {}
    for item in cast("list[object]", ranked_obj):
        if isinstance(item, dict):
            row = cast("JsonDict", item)
            label = str(row.get("label") or "").strip()
            if label:
                index[label] = row
    return index


def _load_lineage(path: Path | None) -> JsonDict | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _infer_lineage_confidence(spec: JsonDict, lineage_payload: JsonDict | None) -> str:
    if spec.get("lineage_confidence"):
        return str(spec["lineage_confidence"])
    if lineage_payload is None:
        return "unknown"
    if _as_bool(lineage_payload.get("lineage_ok")):
        unexpected_answer = cast("list[object]", lineage_payload.get("unexpected_answer_qids") or [])
        unexpected_page = cast("list[object]", lineage_payload.get("unexpected_page_qids") or [])
        if not unexpected_answer and not unexpected_page:
            return "high"
        return "medium"
    return "low"


def _load_gate(path: Path | None) -> JsonDict | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _load_supervisor_action(path: Path | None) -> str | None:
    payload = _load_json(path)
    runs_obj = payload.get("runs")
    if not isinstance(runs_obj, list) or not runs_obj:
        return None
    last = runs_obj[-1]
    if not isinstance(last, dict):
        return None
    decision = cast("JsonDict", last.get("decision") or {})
    action = str(decision.get("action") or "").strip()
    return action or None


def _load_production_mimic(path: Path | None) -> JsonDict | None:
    payload = _load_json(path)
    if not payload:
        return None
    nested = payload.get("production_mimic")
    if isinstance(nested, dict):
        return cast("JsonDict", nested)
    return payload


def _load_exactness(path: Path | None) -> JsonDict | None:
    payload = _load_json(path)
    return payload or None


def _status_metrics(payload: JsonDict | None) -> JsonDict:
    metrics = cast("JsonDict", (payload or {}).get("metrics") or {})
    return {
        "external_det": metrics.get("deterministic"),
        "external_asst": metrics.get("assistant"),
        "external_g": metrics.get("grounding"),
        "external_t": metrics.get("telemetry"),
        "external_f": metrics.get("ttft_multiplier"),
        "external_total": metrics.get("total_score"),
    }


def _matrix_default_specs(root: Path) -> list[JsonDict]:
    warmup = root / "platform_runs" / "warmup"
    research = root / ".sdd" / "researches"
    return [
        {
            "label": "submitted_unknown_01",
            "date": "2026-03-11",
            "status": "submitted",
            "branch_class": "unknown",
            "git_commit": "unknown",
            "baseline": "unknown",
            "lineage_confidence": "low",
            "notes": "Warm-up submission counted on the public board, but no recoverable local lineage artifact was found.",
        },
        {
            "label": "submitted_unknown_02",
            "date": "2026-03-11",
            "status": "submitted",
            "branch_class": "unknown",
            "git_commit": "unknown",
            "baseline": "unknown",
            "lineage_confidence": "low",
            "notes": "Warm-up submission counted on the public board, but no recoverable local lineage artifact was found.",
        },
        {
            "label": "submitted_unknown_03",
            "date": "2026-03-11",
            "status": "submitted",
            "branch_class": "unknown",
            "git_commit": "unknown",
            "baseline": "unknown",
            "lineage_confidence": "low",
            "notes": "Warm-up submission counted on the public board, but no recoverable local lineage artifact was found.",
        },
        {
            "label": "submitted_unknown_04",
            "date": "2026-03-11",
            "status": "submitted",
            "branch_class": "unknown",
            "git_commit": "unknown",
            "baseline": "unknown",
            "lineage_confidence": "low",
            "notes": "Warm-up submission counted on the public board, but no recoverable local lineage artifact was found.",
        },
        {
            "label": "v5_public_support_baseline",
            "date": "2026-03-12",
            "status": "submitted",
            "branch_class": "support_calibrated_baseline",
            "git_commit": "unknown",
            "baseline": "public",
            "external_version": "v5",
            "external_status_json": str(warmup / "submission_status_v4_anchor_lineage.json"),
            "lineage_confidence": "high",
            "notes": "Last clean accepted lineage artifact available in repo.",
        },
        {
            "label": "v6_public_exactness_champion",
            "date": "2026-03-12",
            "status": "submitted",
            "branch_class": "answer_only_exactness",
            "git_commit": "unknown",
            "baseline": "v5_public_support_baseline",
            "external_version": "v6",
            "lineage_confidence": "low",
            "notes": "Best public result so far; local artifact lineage is ambiguous relative to the public score state.",
        },
        {
            "label": "v7_public_all_context_failure",
            "date": "2026-03-12",
            "status": "submitted",
            "branch_class": "broad_page_inflation",
            "git_commit": "unknown",
            "baseline": "v6_public_exactness_champion",
            "external_version": "v7",
            "lineage_confidence": "low",
            "notes": "All-context page broadening; catastrophic public grounding regression.",
        },
        {
            "label": "v8_public_onora_no_gain",
            "date": "2026-03-12",
            "status": "submitted",
            "branch_class": "answer_only_exactness",
            "git_commit": "unknown",
            "baseline": "v6_public_exactness_champion",
            "external_version": "v8",
            "lineage_confidence": "low",
            "notes": "ONORA casing tweak; no real public gain over v6.",
        },
        {
            "label": "v9_public_reranked_pages_failure",
            "date": "2026-03-12",
            "status": "submitted",
            "branch_class": "mixed_page_rerank_failure",
            "git_commit": "unknown",
            "baseline": "v6_public_exactness_champion",
            "external_version": "v9",
            "external_status_json": str(warmup / "submission_status.json"),
            "lineage_confidence": "high",
            "notes": "Reranked-page submission; public grounding crash.",
        },
        {
            "label": "v10_exactness_only",
            "date": "2026-03-12",
            "status": "candidate",
            "branch_class": "exactness_only_fallback",
            "git_commit": "unknown",
            "baseline": "submission_v4_anchor_lineage",
            "lineage_json": str(research / "candidate_lineage_equivalence_v10_2026-03-12.json"),
            "notes": "Exactness-only safety artifact; defensive fallback only because lineage is anchored to v4 lineage rather than clean public v6.",
        },
        {
            "label": "v5046_exactness_only_from_v6_context_seed",
            "date": "2026-03-12",
            "status": "candidate",
            "branch_class": "exactness_only_fallback",
            "git_commit": "unknown",
            "baseline": "submission_v6_context_seed",
            "lineage_json": str(research / "candidate_lineage_equivalence_v5046_exactness_only_2026-03-12.json"),
            "notes": "Local page-stable exactness fallback relative to v6_context_seed only; not promoted for public submit.",
        },
        {
            "label": "triad_f331_e0798",
            "date": "2026-03-13",
            "status": "ceiling",
            "branch_class": "support_only_offense",
            "git_commit": "0343e02",
            "baseline": "submission_v6_context_seed",
            "candidate_label": "triad_f331_e0798",
            "candidate_cycle_json": str(research / "candidate_ceiling_cycle_2026-03-13_estimated" / "candidate_ceiling_cycle.json"),
            "lineage_json": str(research / "candidate_ceiling_cycle_2026-03-13_estimated" / "triad_f331_e0798" / "lineage.json"),
            "notes": "Best pure-G small-diff ceiling candidate under local gates.",
        },
        {
            "label": "triad_f331_e0798_plus_dotted",
            "date": "2026-03-13",
            "status": "ceiling",
            "branch_class": "combined_small_diff_ceiling",
            "git_commit": "0343e02",
            "baseline": "submission_v6_context_seed",
            "candidate_label": "triad_f331_e0798_plus_dotted",
            "candidate_cycle_json": str(research / "candidate_ceiling_cycle_2026-03-13_estimated" / "candidate_ceiling_cycle.json"),
            "lineage_json": str(research / "candidate_ceiling_cycle_2026-03-13_estimated" / "triad_f331_e0798_plus_dotted" / "lineage.json"),
            "exactness_json": str(research / "current_combined_exactness_audit_all_incorrect_2026-03-13.json"),
            "notes": "Current best bounded combined candidate; local small-diff ceiling leader.",
        },
        {
            "label": "triad_f331_e0798_plus_dotted_5046",
            "date": "2026-03-13",
            "status": "candidate",
            "branch_class": "combined_small_diff_with_fallback_rider",
            "git_commit": "0343e02",
            "baseline": "submission_v6_context_seed",
            "candidate_label": "triad_f331_e0798_plus_dotted_5046",
            "candidate_cycle_json": str(research / "candidate_ceiling_cycle_2026-03-13_estimated" / "candidate_ceiling_cycle.json"),
            "lineage_json": str(research / "candidate_ceiling_cycle_2026-03-13_estimated" / "triad_f331_e0798_plus_dotted_5046" / "lineage.json"),
            "exactness_json": str(research / "current_combined_exactness_audit_all_incorrect_plus_5046_2026-03-13.json"),
            "notes": "Local exactness-max variant; resolves the full known incorrect scaffold tail but still not promoted.",
        },
        {
            "label": "v_embeddinggemma_fullcollection_iter14_invalid",
            "date": "2026-03-13",
            "status": "invalid",
            "branch_class": "embedding_fullcollection_invalid_wiring",
            "git_commit": "3317825",
            "baseline": "triad_f331_e0798_plus_dotted",
            "lineage_confidence": "n/a",
            "notes": "Initial full-collection embedding branch run invalidated by collection wiring mismatch; replaced by iter14c.",
        },
        {
            "label": "v_embeddinggemma_fullcollection_iter14c",
            "date": "2026-03-13",
            "status": "rejected",
            "branch_class": "embedding_fullcollection_branch",
            "git_commit": "3317825",
            "baseline": "triad_f331_e0798_plus_dotted",
            "gate_json": str(research / "embedding_fullcollection_iter14c" / "experiment_gate_v_embeddinggemma_fullcollection_iter14c_vs_v_support_combo_9f9_cdddeb_f331_e0798_plus_dotted_iter13.json"),
            "notes": "Full-collection embedding branch; bounded gate says NO_SUBMIT against current leader.",
        },
        {
            "label": "v_within_doc_rerank_surrogate_iter13",
            "date": "2026-03-13",
            "status": "rejected",
            "branch_class": "within_doc_rerank_surrogate",
            "git_commit": "0343e02",
            "baseline": "triad_f331_e0798_plus_dotted",
            "gate_json": str(research / "within_doc_rerank_surrogate_iter13" / "experiment_gate.json"),
            "notes": "Within-doc rerank/localization branch; best subsets are non-inferior only, not better.",
        },
    ]


def _load_specs(path: Path | None, *, root: Path) -> list[JsonDict]:
    if path is None or not path.exists():
        return _matrix_default_specs(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows_obj = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows_obj, list):
        raise ValueError(f"Expected list/rows object in {path}")
    specs: list[JsonDict] = []
    for item in cast("list[object]", rows_obj):
        if isinstance(item, dict):
            specs.append(cast("JsonDict", item))
    return specs


def _resolve_path(value: object) -> Path | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _hydrate_row(
    *,
    spec: JsonDict,
    history_rows: dict[str, JsonDict],
    cycle_index: dict[str, JsonDict],
    global_production_mimic: JsonDict | None,
    supervisor_action: str | None,
) -> JsonDict:
    row: JsonDict = {
        "label": spec.get("label"),
        "date": spec.get("date"),
        "status": spec.get("status"),
        "branch_class": spec.get("branch_class"),
        "git_commit": spec.get("git_commit") or "unknown",
        "baseline": spec.get("baseline") or "-",
        "notes": spec.get("notes") or "",
        "lineage_confidence": spec.get("lineage_confidence") or "unknown",
        "answer_drift": None,
        "page_drift": None,
        "hidden_g_trusted": None,
        "hidden_g_all": None,
        "judge_pass_rate": None,
        "judge_grounding": None,
        "judge_accuracy": None,
        "exactness_resolved_qids": [],
        "exactness_unresolved_qids": [],
        "external_det": None,
        "external_asst": None,
        "external_g": None,
        "external_t": None,
        "external_f": None,
        "external_total": None,
        "external_rank": None,
        "platform_like_total_estimate": None,
        "strict_total_estimate": None,
        "paranoid_total_estimate": None,
        "supervisor_action": spec.get("supervisor_action") or None,
    }

    version = str(spec.get("external_version") or "").strip()
    if version and version in history_rows:
        row.update(history_rows[version])

    status_payload = _load_json(_resolve_path(spec.get("external_status_json")))
    if status_payload:
        row.update(_status_metrics(status_payload))

    candidate_label = str(spec.get("candidate_label") or "").strip()
    if candidate_label and candidate_label in cycle_index:
        candidate_row = cycle_index[candidate_label]
        row["answer_drift"] = candidate_row.get("answer_drift")
        row["page_drift"] = candidate_row.get("page_drift")
        row["hidden_g_trusted"] = candidate_row.get("hidden_g_trusted_delta")
        row["hidden_g_all"] = candidate_row.get("hidden_g_all_delta")
        row["platform_like_total_estimate"] = candidate_row.get("strict_total_estimate")
        row["strict_total_estimate"] = candidate_row.get("strict_total_estimate")
        row["paranoid_total_estimate"] = candidate_row.get("paranoid_total_estimate")
        row["notes"] = (
            f"{row['notes']} [candidate_cycle={candidate_row.get('recommendation')}]".strip()
            if row["notes"]
            else f"candidate_cycle={candidate_row.get('recommendation')}"
        )

    gate_payload = _load_gate(_resolve_path(spec.get("gate_json")))
    if gate_payload is not None:
        row["answer_drift"] = gate_payload.get("answer_changed_count", row["answer_drift"])
        row["page_drift"] = gate_payload.get("retrieval_page_projection_changed_count", row["page_drift"])
        baseline_all = _as_float(gate_payload.get("benchmark_all_baseline"))
        candidate_all = _as_float(gate_payload.get("benchmark_all_candidate"))
        baseline_trusted = _as_float(gate_payload.get("benchmark_trusted_baseline"))
        candidate_trusted = _as_float(gate_payload.get("benchmark_trusted_candidate"))
        row["hidden_g_all"] = candidate_all - baseline_all
        row["hidden_g_trusted"] = candidate_trusted - baseline_trusted
        row["notes"] = (
            f"{row['notes']} [gate={gate_payload.get('recommendation')}]".strip()
            if row["notes"]
            else f"gate={gate_payload.get('recommendation')}"
        )

    lineage_payload = _load_lineage(_resolve_path(spec.get("lineage_json")))
    row["lineage_confidence"] = _infer_lineage_confidence(spec, lineage_payload)

    exactness_payload = _load_exactness(_resolve_path(spec.get("exactness_json")))
    if exactness_payload is not None:
        row["exactness_resolved_qids"] = exactness_payload.get("resolved_incorrect_qids") or []
        row["exactness_unresolved_qids"] = exactness_payload.get("still_mismatched_incorrect_qids") or []

    if global_production_mimic is not None and candidate_label and candidate_label == str(global_production_mimic.get("candidate_class") or candidate_label):
        row["lineage_confidence"] = global_production_mimic.get("lineage_confidence", row["lineage_confidence"])
        row["judge_pass_rate"] = cast("JsonDict", global_production_mimic.get("judge") or {}).get("pass_rate")
        row["judge_grounding"] = cast("JsonDict", global_production_mimic.get("judge") or {}).get("avg_grounding")
        row["judge_accuracy"] = cast("JsonDict", global_production_mimic.get("judge") or {}).get("avg_accuracy")
        row["platform_like_total_estimate"] = global_production_mimic.get("platform_like_total_estimate")
        row["strict_total_estimate"] = global_production_mimic.get("strict_total_estimate")
        row["paranoid_total_estimate"] = global_production_mimic.get("paranoid_total_estimate")
        row["supervisor_action"] = supervisor_action or row["supervisor_action"]

    return row


def _best_offline_row(rows: list[JsonDict]) -> JsonDict | None:
    candidates = [row for row in rows if str(row.get("status")) in {"candidate", "ceiling", "rejected"}]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            _as_float(row.get("paranoid_total_estimate"), default=-1.0),
            _as_float(row.get("strict_total_estimate"), default=-1.0),
            _as_float(row.get("hidden_g_trusted"), default=-999.0),
        ),
    )


def _best_public_row(rows: list[JsonDict]) -> JsonDict | None:
    submitted = [row for row in rows if str(row.get("status")) == "submitted" and row.get("external_total") is not None]
    if not submitted:
        return None
    return max(submitted, key=lambda row: _as_float(row.get("external_total"), default=-1.0))


def _summary_block(*, leaderboard_summary: JsonDict, rows: list[JsonDict], supervisor_action: str | None) -> list[str]:
    public_best = _best_public_row(rows)
    offline_best = _best_offline_row(rows)
    gap_targets = cast("list[JsonDict]", leaderboard_summary.get("gap_targets") or [])
    used = _as_int(leaderboard_summary.get("submissions"))
    remaining = max(0, 10 - used)
    current_path_ceilinged = supervisor_action in {"small_diff_ceiling_reached", "local_ceiling_reached_hold_budget"}
    lines = [
        "# Competition Matrix",
        "",
        "## Current Status",
        "",
        f"- Current public best: `{public_best.get('label') if public_best else 'unknown'}`"
        + (
            f" total=`{_fmt_float(public_best.get('external_total') if public_best and public_best.get('external_rank') is not None else leaderboard_summary.get('total'))}` "
            f"rank=`{_fmt_int(public_best.get('external_rank') if public_best and public_best.get('external_rank') is not None else leaderboard_summary.get('rank'))}`"
            if public_best
            else ""
        ),
        f"- Current best offline candidate: `{offline_best.get('label') if offline_best else 'unknown'}`"
        + (
            f" paranoid=`{_fmt_float(offline_best.get('paranoid_total_estimate'))}`"
            if offline_best
            else ""
        ),
        f"- Warm-up submissions used/remaining: `{used}/10`, remaining=`{remaining}`",
        f"- Current default decision: `{supervisor_action or 'no_submit_continue_offline'}`",
        f"- Current S: `{_fmt_float(leaderboard_summary.get('s'))}`",
        f"- Current G: `{_fmt_float(leaderboard_summary.get('g'))}`",
    ]
    gap_map = {int(_as_int(item.get("rank"))): item for item in gap_targets}
    for rank in (1, 3, 5):
        item = gap_map.get(rank)
        if item is None:
            continue
        lines.append(
            f"- Required `G` to beat rank `{rank}`: `{_fmt_float(item.get('target_g_at_current_s'))}` "
            f"(ΔG `{_as_float(item.get('delta_g_at_current_s')):+.6f}`)"
        )
    lines.extend(
        [
            f"- #1 reachable through current small-diff path: `{'no' if current_path_ceilinged else 'unknown'}`",
            f"- Current path locally ceilinged: `{'yes' if current_path_ceilinged else 'unknown'}`",
            "",
            "## Matrix",
            "",
        ]
    )
    return lines


def _render_matrix(rows: list[JsonDict], *, leaderboard_summary: JsonDict, supervisor_action: str | None) -> str:
    headers = [
        "label",
        "date",
        "status",
        "branch_class",
        "git_commit",
        "baseline",
        "lineage_confidence",
        "answer_drift",
        "page_drift",
        "hidden_g_trusted",
        "hidden_g_all",
        "judge_pass_rate",
        "judge_grounding",
        "judge_accuracy",
        "exactness_resolved_qids",
        "exactness_unresolved_qids",
        "external_det",
        "external_asst",
        "external_g",
        "external_t",
        "external_f",
        "external_total",
        "external_rank",
        "platform_like_total_estimate",
        "strict_total_estimate",
        "paranoid_total_estimate",
        "supervisor_action",
        "notes",
    ]
    lines = _summary_block(leaderboard_summary=leaderboard_summary, rows=rows, supervisor_action=supervisor_action)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        rendered = [
            str(row.get("label") or "-"),
            str(row.get("date") or "-"),
            str(row.get("status") or "-"),
            str(row.get("branch_class") or "-"),
            str(row.get("git_commit") or "-"),
            str(row.get("baseline") or "-"),
            str(row.get("lineage_confidence") or "-"),
            _fmt_int(row.get("answer_drift")),
            _fmt_int(row.get("page_drift")),
            _fmt_small(row.get("hidden_g_trusted")),
            _fmt_small(row.get("hidden_g_all")),
            _fmt_small(row.get("judge_pass_rate")),
            _fmt_small(row.get("judge_grounding")),
            _fmt_small(row.get("judge_accuracy")),
            _short_qids(row.get("exactness_resolved_qids")),
            _short_qids(row.get("exactness_unresolved_qids")),
            _fmt_small(row.get("external_det")),
            _fmt_small(row.get("external_asst")),
            _fmt_small(row.get("external_g")),
            _fmt_small(row.get("external_t")),
            _fmt_small(row.get("external_f")),
            _fmt_float(row.get("external_total")),
            _fmt_int(row.get("external_rank")),
            _fmt_float(row.get("platform_like_total_estimate")),
            _fmt_float(row.get("strict_total_estimate")),
            _fmt_float(row.get("paranoid_total_estimate")),
            str(row.get("supervisor_action") or "-"),
            str(row.get("notes") or "-").replace("\n", " "),
        ]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines) + "\n"


def build_competition_matrix(
    *,
    leaderboard_path: Path,
    team_name: str,
    specs_path: Path | None,
    history_path: Path | None,
    candidate_cycle_json: Path | None,
    production_mimic_json: Path | None,
    supervisor_runs_json: Path | None,
) -> JsonDict:
    leaderboard_summary = build_leaderboard_summary(load_leaderboard_rows(leaderboard_path), team_name=team_name)
    history_rows = _parse_public_history_rows(history_path)
    cycle_index = _load_candidate_cycle_index(candidate_cycle_json)
    production_mimic = _load_production_mimic(production_mimic_json)
    supervisor_action = _load_supervisor_action(supervisor_runs_json)
    specs = _load_specs(specs_path, root=ROOT)
    production_mimic_label = str(production_mimic.get("candidate_class") or "") if production_mimic is not None else ""
    rows = [
        _hydrate_row(
            spec=spec,
            history_rows=history_rows,
            cycle_index=cycle_index,
            global_production_mimic=production_mimic
            if production_mimic is not None
            and (
                str(spec.get("label") or "") == production_mimic_label
                or str(spec.get("candidate_label") or "") == production_mimic_label
            )
            else None,
            supervisor_action=supervisor_action
            if production_mimic is not None
            and (
                str(spec.get("label") or "") == production_mimic_label
                or str(spec.get("candidate_label") or "") == production_mimic_label
            )
            else None,
        )
        for spec in specs
    ]
    current_total = _as_float(leaderboard_summary.get("total"))
    for row in rows:
        if str(row.get("status")) != "submitted":
            continue
        external_total = row.get("external_total")
        if external_total is None:
            continue
        if abs(_as_float(external_total) - current_total) <= 1e-6:
            row["external_rank"] = leaderboard_summary.get("rank")
            row["external_t"] = leaderboard_summary.get("t")
            row["external_f"] = leaderboard_summary.get("f")

    public_best = _best_public_row(rows)
    offline_best = _best_offline_row(rows)
    if public_best is not None and public_best.get("external_rank") is None:
        public_best["external_rank"] = leaderboard_summary.get("rank")
        public_best["external_total"] = leaderboard_summary.get("total")
        public_best["external_t"] = leaderboard_summary.get("t")
        public_best["external_f"] = leaderboard_summary.get("f")
    payload: JsonDict = {
        "summary": {
            "team": team_name,
            "current_public_best_label": public_best.get("label") if public_best else None,
            "current_best_offline_label": offline_best.get("label") if offline_best else None,
            "current_default_decision": supervisor_action or "no_submit_continue_offline",
            "submissions_used": leaderboard_summary.get("submissions"),
            "submissions_remaining": max(0, 10 - _as_int(leaderboard_summary.get("submissions"))),
            "current_s": leaderboard_summary.get("s"),
            "current_g": leaderboard_summary.get("g"),
            "perfect_s_total": leaderboard_summary.get("perfect_s_total"),
            "small_diff_path_reaches_rank_1": False,
            "current_path_locally_ceilinged": supervisor_action in {"small_diff_ceiling_reached", "local_ceiling_reached_hold_budget"},
        },
        "leaderboard_summary": leaderboard_summary,
        "rows": rows,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the competition progress snapshot and optionally build the canonical competition matrix.")
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--ledger-json", type=Path, default=None)
    parser.add_argument("--scoring-json", type=Path, default=None)
    parser.add_argument("--anchor-slice-json", type=Path, default=None)
    parser.add_argument("--warmup-budget", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--matrix-row-specs-json", type=Path, default=None)
    parser.add_argument("--matrix-json-out", type=Path, default=None)
    parser.add_argument("--matrix-md-out", type=Path, default=None)
    parser.add_argument("--history-md", type=Path, default=None)
    parser.add_argument("--candidate-ceiling-cycle", type=Path, default=None)
    parser.add_argument("--production-mimic-json", type=Path, default=None)
    parser.add_argument("--supervisor-runs-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.out is not None:
        report = build_report(
            leaderboard_path=args.leaderboard,
            team_name=args.team,
            ledger_path=args.ledger_json,
            scoring_json=args.scoring_json,
            anchor_slice_json=args.anchor_slice_json,
            warmup_budget=args.warmup_budget,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")

    if args.matrix_json_out is not None or args.matrix_md_out is not None:
        matrix_payload = build_competition_matrix(
            leaderboard_path=args.leaderboard,
            team_name=args.team,
            specs_path=args.matrix_row_specs_json,
            history_path=args.history_md,
            candidate_cycle_json=args.candidate_ceiling_cycle,
            production_mimic_json=args.production_mimic_json,
            supervisor_runs_json=args.supervisor_runs_json,
        )
        if args.matrix_json_out is not None:
            args.matrix_json_out.parent.mkdir(parents=True, exist_ok=True)
            args.matrix_json_out.write_text(json.dumps(matrix_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.matrix_md_out is not None:
            args.matrix_md_out.parent.mkdir(parents=True, exist_ok=True)
            args.matrix_md_out.write_text(
                _render_matrix(
                    cast("list[JsonDict]", matrix_payload["rows"]),
                    leaderboard_summary=cast("JsonDict", matrix_payload["leaderboard_summary"]),
                    supervisor_action=str(cast("JsonDict", matrix_payload["summary"]).get("current_default_decision") or ""),
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
