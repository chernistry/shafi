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


def _short_hash(value: object, *, length: int = 12) -> str:
    if value is None or value == "":
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    return text[:length]


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


def _discover_spec_paths(root: Path) -> list[Path]:
    research = root / ".sdd" / "researches"
    if not research.exists():
        return []
    paths = sorted(research.glob("matrix_specs_with_ticket*.json"))
    paths.extend(sorted(research.glob("ticket*_*/spec.json")))
    return [path.resolve() for path in paths]


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


def _load_run_manifest(path: Path | None) -> JsonDict | None:
    payload = _load_json(path)
    if not payload:
        return None
    nested = payload.get("run_manifest")
    if isinstance(nested, dict):
        return cast("JsonDict", nested)
    return payload


def _load_candidate_fingerprint(path: Path | None) -> JsonDict | None:
    payload = _load_json(path)
    if not payload:
        return None
    nested = payload.get("candidate_fingerprint")
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
            "external_det": 0.971429,
            "external_asst": 0.693333,
            "external_g": 0.800729,
            "external_total": 0.74156,
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
            "external_det": 0.971429,
            "external_asst": 0.646667,
            "external_g": 0.608,
            "external_total": 0.554,
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
            "external_det": 0.971429,
            "external_asst": 0.686667,
            "external_g": 0.800729,
            "external_total": 0.740,
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
        {
            "label": "v10_ticket501_current",
            "date": "2026-03-16",
            "status": "submitted",
            "branch_class": "current_codebase_warmup_submit",
            "git_commit": "unknown",
            "baseline": "ticket64_private_rehearsal_fix3_run_a",
            "lineage_confidence": "high",
            "notes": "10th warm-up submission from current codebase. Scored Total=0.698012, G=0.771627 on platform. Board retained historical best 0.74156. Drift ledger shows only 3 answer changes vs frozen control.",
        },
    ]


def _known_row_overrides(root: Path) -> dict[str, JsonDict]:
    research = root / ".sdd" / "researches"
    return {
        "triad_f331_e0798": {
            "production_mimic_json": str(
                research / "production_mimic_triad_f331_e0798_2026-03-13" / "production_mimic.json"
            ),
        },
        "triad_f331_e0798_plus_dotted": {
            "production_mimic_json": str(
                research / "production_mimic_current_combined_2026-03-13" / "production_mimic.json"
            ),
        },
        "triad_f331_e0798_plus_dotted_5046": {
            "production_mimic_json": str(
                research
                / "production_mimic_triad_f331_e0798_plus_dotted_5046_2026-03-13"
                / "production_mimic.json"
            ),
        },
        "v5046_exactness_only_from_v6_context_seed": {
            "production_mimic_json": str(
                research
                / "production_mimic_v5046_exactness_only_from_v6_context_seed_2026-03-13"
                / "production_mimic.json"
            ),
        },
        "v_embeddinggemma_fullcollection_iter14c": {
            "production_mimic_json": str(
                research
                / "production_mimic_v_embeddinggemma_fullcollection_iter14c_2026-03-13"
                / "production_mimic.json"
            ),
        },
        "v_within_doc_rerank_surrogate_iter13": {
            "production_mimic_json": str(
                research
                / "production_mimic_v_within_doc_rerank_surrogate_iter13_2026-03-13"
                / "production_mimic.json"
            ),
        },
        "v10_local_page_localizer_r1": {
            "production_mimic_json": str(research / "production_mimic_v10_local_page_localizer_r1_2026-03-13" / "production_mimic.json"),
        },
        "v10_local_page_reranker_r1": {
            "production_mimic_json": str(research / "ticket21_support_shape_v2_2026-03-13" / "production_mimic_rejected_page_reranker.json"),
        },
        "v10_local_support_shape_v2_r1": {
            "production_mimic_json": str(research / "ticket21_support_shape_v2_2026-03-13" / "production_mimic_current_leader.json"),
        },
        "v10_local_docfamily_collapse_r1": {
            "production_mimic_json": str(research / "ticket22_docfamily_collapse_r1_2026-03-13" / "production_mimic_v10_local_docfamily_collapse_r1.json"),
        },
        "v10_local_page_candidates_r1": {
            "production_mimic_json": str(research / "ticket23_page_candidates_r1_2026-03-13" / "production_mimic_v10_local_page_candidates_r1.json"),
        },
        "v10_local_colbert_page_rerank_r1": {
            "production_mimic_json": str(
                research / "ticket28_local_colbert_sidecar_2026-03-13" / "production_mimic_v10_local_colbert_page_rerank_r1.json"
            ),
        },
    }


def _merge_specs(
    base_specs: list[JsonDict],
    extra_paths: list[Path],
    *,
    root: Path,
) -> list[JsonDict]:
    merged_specs: dict[str, JsonDict] = {}
    ordered_labels: list[str] = []
    for spec in base_specs:
        label = str(spec.get("label") or "").strip()
        if not label:
            continue
        merged_specs[label] = dict(spec)
        ordered_labels.append(label)
    for path in extra_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows_obj = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows_obj, list):
            raise ValueError(f"Expected list/rows object in {path}")
        for item in cast("list[object]", rows_obj):
            if not isinstance(item, dict):
                continue
            spec = cast("JsonDict", item)
            label = str(spec.get("label") or "").strip()
            if not label:
                continue
            merged_specs[label] = {**merged_specs.get(label, {}), **spec}
            if label not in ordered_labels:
                ordered_labels.append(label)
    for label, override in _known_row_overrides(root).items():
        if label in merged_specs:
            merged_specs[label] = {**merged_specs[label], **override}
    return [merged_specs[label] for label in ordered_labels if label in merged_specs]


def _load_specs(path: Path | None, *, root: Path) -> list[JsonDict]:
    default_specs = _matrix_default_specs(root)
    extra_paths = _discover_spec_paths(root)
    if path is not None and path.exists():
        extra_paths.append(path.resolve())
    return _merge_specs(default_specs, extra_paths, root=root)


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
    global_cycle_index: dict[str, JsonDict],
    cycle_index_cache: dict[str, dict[str, JsonDict]],
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
        "run_manifest_status": spec.get("run_manifest_status") or "unknown",
        "run_manifest_fingerprint": None,
        "candidate_fingerprint": None,
        "duplicate_of_label": None,
        "answer_drift": None,
        "page_drift": None,
        "hidden_g_trusted": None,
        "hidden_g_all": None,
        "judge_pass_rate": None,
        "judge_grounding": None,
        "judge_accuracy": None,
        "exactness_resolved_qids": [],
        "exactness_unresolved_qids": [],
        "external_det": spec.get("external_det"),
        "external_asst": spec.get("external_asst"),
        "external_g": spec.get("external_g"),
        "external_t": spec.get("external_t"),
        "external_f": spec.get("external_f"),
        "external_total": spec.get("external_total"),
        "external_rank": spec.get("external_rank"),
        "platform_like_total_estimate": None,
        "strict_total_estimate": None,
        "paranoid_total_estimate": None,
        "platform_like_rank_estimate": None,
        "strict_rank_estimate": None,
        "paranoid_rank_estimate": None,
        "supervisor_action": spec.get("supervisor_action") or None,
    }

    version = str(spec.get("external_version") or "").strip()
    if version and version in history_rows:
        history_row = history_rows[version]
        for key, value in history_row.items():
            current_value = row.get(key)
            if current_value is None or current_value == "" or current_value == []:
                row[key] = value

    status_payload = _load_json(_resolve_path(spec.get("external_status_json")))
    if status_payload:
        row.update(_status_metrics(status_payload))

    candidate_label = str(spec.get("candidate_label") or "").strip()
    cycle_path = _resolve_path(spec.get("candidate_cycle_json"))
    cycle_index = global_cycle_index
    if cycle_path is not None:
        cache_key = str(cycle_path.resolve())
        if cache_key not in cycle_index_cache:
            cycle_index_cache[cache_key] = _load_candidate_cycle_index(cycle_path)
        cycle_index = cycle_index_cache[cache_key]
    if candidate_label and candidate_label in cycle_index:
        candidate_row = cycle_index[candidate_label]
        row["answer_drift"] = candidate_row.get("answer_drift")
        row["page_drift"] = candidate_row.get("page_drift")
        row["hidden_g_trusted"] = candidate_row.get("hidden_g_trusted_delta")
        row["hidden_g_all"] = candidate_row.get("hidden_g_all_delta")
        row["platform_like_total_estimate"] = candidate_row.get("platform_like_total_estimate", candidate_row.get("strict_total_estimate"))
        row["strict_total_estimate"] = candidate_row.get("strict_total_estimate")
        row["paranoid_total_estimate"] = candidate_row.get("paranoid_total_estimate")
        row["platform_like_rank_estimate"] = candidate_row.get("platform_like_rank_estimate")
        row["strict_rank_estimate"] = candidate_row.get("strict_rank_estimate")
        row["paranoid_rank_estimate"] = candidate_row.get("paranoid_rank_estimate")
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

    run_manifest_path = _resolve_path(spec.get("run_manifest_json") or spec.get("manifest_json"))
    run_manifest_payload = _load_run_manifest(run_manifest_path)
    if run_manifest_payload is not None:
        row["run_manifest_status"] = "present"
        row["run_manifest_fingerprint"] = run_manifest_payload.get("fingerprint")
        manifest_git = cast("JsonDict", run_manifest_payload.get("git") or {})
        manifest_sha = str(manifest_git.get("sha") or "").strip()
        if manifest_sha and str(row.get("git_commit") or "unknown") == "unknown":
            row["git_commit"] = manifest_sha[:7]
    else:
        status = str(row.get("status") or "").strip()
        if status in {"candidate", "ceiling"}:
            row["run_manifest_status"] = "missing_blocking"
            row["notes"] = (
                f"{row['notes']} [manifest=missing_blocking]".strip()
                if row["notes"]
                else "manifest=missing_blocking"
            )
        else:
            row["run_manifest_status"] = "legacy_unknown"

    candidate_fingerprint_path = _resolve_path(spec.get("candidate_fingerprint_json") or spec.get("fingerprint_json"))
    candidate_fingerprint_payload = _load_candidate_fingerprint(candidate_fingerprint_path)
    if candidate_fingerprint_payload is not None:
        row["candidate_fingerprint"] = candidate_fingerprint_payload.get("fingerprint")
        row["duplicate_of_label"] = candidate_fingerprint_payload.get("duplicate_of_label")
        duplicate_of_label = str(row.get("duplicate_of_label") or "").strip()
        if duplicate_of_label:
            row["notes"] = (
                f"{row['notes']} [duplicate_of={duplicate_of_label}]".strip()
                if row["notes"]
                else f"duplicate_of={duplicate_of_label}"
            )

    exactness_payload = _load_exactness(_resolve_path(spec.get("exactness_json")))
    if exactness_payload is not None:
        row["exactness_resolved_qids"] = exactness_payload.get("resolved_incorrect_qids") or []
        row["exactness_unresolved_qids"] = exactness_payload.get("still_mismatched_incorrect_qids") or []

    row_production_mimic = _load_production_mimic(_resolve_path(spec.get("production_mimic_json")))
    effective_production_mimic = row_production_mimic
    if effective_production_mimic is None and global_production_mimic is not None and candidate_label and candidate_label == str(global_production_mimic.get("candidate_class") or candidate_label):
        effective_production_mimic = global_production_mimic
    if effective_production_mimic is not None:
        row["lineage_confidence"] = effective_production_mimic.get("lineage_confidence", row["lineage_confidence"])
        row["judge_pass_rate"] = cast("JsonDict", effective_production_mimic.get("judge") or {}).get("pass_rate")
        row["judge_grounding"] = cast("JsonDict", effective_production_mimic.get("judge") or {}).get("avg_grounding")
        row["judge_accuracy"] = cast("JsonDict", effective_production_mimic.get("judge") or {}).get("avg_accuracy")
        row["platform_like_total_estimate"] = effective_production_mimic.get("platform_like_total_estimate")
        row["strict_total_estimate"] = effective_production_mimic.get("strict_total_estimate")
        row["paranoid_total_estimate"] = effective_production_mimic.get("paranoid_total_estimate")
        row["platform_like_rank_estimate"] = effective_production_mimic.get("platform_like_rank_estimate")
        row["strict_rank_estimate"] = effective_production_mimic.get("strict_rank_estimate")
        row["paranoid_rank_estimate"] = effective_production_mimic.get("paranoid_rank_estimate")
        if row_production_mimic is None:
            row["supervisor_action"] = supervisor_action or row["supervisor_action"]

    return row


def _best_offline_row(rows: list[JsonDict]) -> JsonDict | None:
    candidates = [row for row in rows if str(row.get("status")) in {"candidate", "ceiling"}]
    if not candidates:
        candidates = [row for row in rows if str(row.get("status")) == "rejected"]
    if not candidates:
        return None
    status_priority = {"ceiling": 2, "candidate": 1, "rejected": 0}
    return max(
        candidates,
        key=lambda row: (
            row.get("paranoid_total_estimate") is not None,
            _as_float(row.get("paranoid_total_estimate"), default=-1.0),
            status_priority.get(str(row.get("status")), -1),
            row.get("strict_total_estimate") is not None,
            _as_float(row.get("strict_total_estimate"), default=-1.0),
            _as_float(row.get("hidden_g_trusted"), default=-999.0),
            len(cast("list[object]", row.get("exactness_resolved_qids") or [])),
            -len(cast("list[object]", row.get("exactness_unresolved_qids") or [])),
            _as_float(row.get("judge_pass_rate"), default=-999.0),
            -_as_float(row.get("answer_drift"), default=999.0),
            -_as_float(row.get("page_drift"), default=999.0),
        ),
    )


def _best_public_row(rows: list[JsonDict]) -> JsonDict | None:
    submitted = [row for row in rows if str(row.get("status")) == "submitted" and row.get("external_total") is not None]
    if not submitted:
        return None
    return max(
        submitted,
        key=lambda row: (
            _as_float(row.get("external_total"), default=-1.0),
            _as_float(row.get("external_det"), default=-1.0),
            _as_float(row.get("external_asst"), default=-1.0),
            _as_float(row.get("external_g"), default=-1.0),
        ),
    )


def _estimate_rank_for_total(*, total: object, leaderboard_rows: list[object], team_name: str) -> int | None:
    if total is None or total == "":
        return None
    total_value = _as_float(total, default=-1.0)
    if total_value < 0:
        return None
    better = 0
    for raw in leaderboard_rows:
        row = cast("JsonDict", raw) if isinstance(raw, dict) else {}
        if str(row.get("team_name") or "") == team_name:
            continue
        if _as_float(row.get("total"), default=-1.0) > total_value + 1e-9:
            better += 1
    return better + 1


def _summary_block(*, leaderboard_summary: JsonDict, rows: list[JsonDict], supervisor_action: str | None) -> list[str]:
    public_best = _best_public_row(rows)
    offline_best = _best_offline_row(rows)
    gap_targets = cast("list[JsonDict]", leaderboard_summary.get("gap_targets") or [])
    used = _as_int(leaderboard_summary.get("submissions"))
    remaining = max(0, 10 - used)
    current_path_ceilinged = supervisor_action in {"small_diff_ceiling_reached", "local_ceiling_reached_hold_budget"}
    manifest_counts = {
        "present": sum(1 for row in rows if str(row.get("run_manifest_status") or "") == "present"),
        "missing_blocking": sum(1 for row in rows if str(row.get("run_manifest_status") or "") == "missing_blocking"),
        "legacy_unknown": sum(1 for row in rows if str(row.get("run_manifest_status") or "") == "legacy_unknown"),
    }
    duplicate_count = sum(1 for row in rows if str(row.get("duplicate_of_label") or "").strip())
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
            f" paranoid_rank=`{_fmt_int(offline_best.get('paranoid_rank_estimate'))}`"
            if offline_best
            else ""
        ),
        f"- Warm-up submissions used/remaining: `{used}/10`, remaining=`{remaining}`",
        f"- Current default decision: `{supervisor_action or 'no_submit_continue_offline'}`",
        f"- Current S: `{_fmt_float(leaderboard_summary.get('s'))}`",
        f"- Current G: `{_fmt_float(leaderboard_summary.get('g'))}`",
        f"- Run manifest coverage: present=`{manifest_counts['present']}` missing_blocking=`{manifest_counts['missing_blocking']}` legacy_unknown=`{manifest_counts['legacy_unknown']}`",
        f"- Candidate duplicates detected: `{duplicate_count}`",
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
        "run_manifest_status",
        "run_manifest_fingerprint",
        "candidate_fingerprint",
        "duplicate_of_label",
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
        "platform_like_rank_estimate",
        "strict_total_estimate",
        "strict_rank_estimate",
        "paranoid_total_estimate",
        "paranoid_rank_estimate",
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
            str(row.get("run_manifest_status") or "-"),
            _short_hash(row.get("run_manifest_fingerprint")),
            _short_hash(row.get("candidate_fingerprint")),
            str(row.get("duplicate_of_label") or "-"),
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
            _fmt_int(row.get("platform_like_rank_estimate")),
            _fmt_float(row.get("strict_total_estimate")),
            _fmt_int(row.get("strict_rank_estimate")),
            _fmt_float(row.get("paranoid_total_estimate")),
            _fmt_int(row.get("paranoid_rank_estimate")),
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
    leaderboard_rows = load_leaderboard_rows(leaderboard_path)
    leaderboard_summary = build_leaderboard_summary(leaderboard_rows, team_name=team_name)
    history_rows = _parse_public_history_rows(history_path)
    cycle_index = _load_candidate_cycle_index(candidate_cycle_json)
    cycle_index_cache: dict[str, dict[str, JsonDict]] = {}
    production_mimic = _load_production_mimic(production_mimic_json)
    supervisor_action = _load_supervisor_action(supervisor_runs_json)
    specs = _load_specs(specs_path, root=ROOT)
    production_mimic_label = str(production_mimic.get("candidate_class") or "") if production_mimic is not None else ""
    rows = [
        _hydrate_row(
            spec=spec,
            history_rows=history_rows,
            global_cycle_index=cycle_index,
            cycle_index_cache=cycle_index_cache,
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
    leaderboard_row_objs: list[object] = [{"team_name": row.team_name, "total": row.total} for row in leaderboard_rows]
    for row in rows:
        for total_key, rank_key in (
            ("platform_like_total_estimate", "platform_like_rank_estimate"),
            ("strict_total_estimate", "strict_rank_estimate"),
            ("paranoid_total_estimate", "paranoid_rank_estimate"),
        ):
            row[rank_key] = _estimate_rank_for_total(
                total=row.get(total_key),
                leaderboard_rows=leaderboard_row_objs,
                team_name=team_name,
            )
        if (
            row.get("platform_like_total_estimate") is None
            and row.get("strict_total_estimate") is None
            and row.get("paranoid_total_estimate") is None
        ):
            estimate_note = (
                "[estimates=not_applicable_invalid]"
                if str(row.get("status")) == "invalid"
                else "[estimates=unsupported_local_envelope]"
            )
            notes = str(row.get("notes") or "").strip()
            if estimate_note not in notes:
                row["notes"] = f"{notes} {estimate_note}".strip()
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
        if public_best.get("external_total") is None:
            public_best["external_total"] = leaderboard_summary.get("total")
        if public_best.get("external_t") is None:
            public_best["external_t"] = leaderboard_summary.get("t")
        if public_best.get("external_f") is None:
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
    parser.add_argument("--warmup-budget", type=int, default=15)
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
