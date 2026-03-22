#!/usr/bin/env python3
"""Build a calibrated resurrection matrix from tracked historical artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

try:
    from score_against_golden import score as score_against_golden
except ModuleNotFoundError:  # pragma: no cover
    from scripts.score_against_golden import score as score_against_golden

from rag_challenge.eval.resurrection_matrix import (
    HOT_SET_PREFIXES,
    LineageConfidence,
    ResurrectionEvidence,
    assess_resurrection_candidate,
    hot_set_overlap_prefixes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

JsonDict = dict[str, object]
ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = ROOT / ".sdd" / "researches"
PUBLIC_V6_TOTAL = 0.74156


@dataclass(frozen=True)
class CandidateSpec:
    """Describe one historical candidate/control to score in the matrix."""

    label: str
    baseline_label: str
    candidate_class: str
    raw_results_path: Path | None
    production_mimic_path: Path | None
    gate_path: Path | None
    lineage_path: Path | None
    control: bool = False


def _load_json(path: Path | None) -> JsonDict:
    """Load a JSON object from disk.

    Args:
        path: Optional JSON path.

    Returns:
        JsonDict: Parsed object or an empty dict when absent.
    """

    if path is None or not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _as_float(value: object) -> float | None:
    """Safely coerce a value into float.

    Args:
        value: Arbitrary scalar.

    Returns:
        float | None: Coerced float or ``None``.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _artifact_exists(path: Path | None) -> bool:
    """Return whether an optional artifact path exists.

    Args:
        path: Optional filesystem path.

    Returns:
        bool: ``True`` when the artifact exists.
    """

    return path is not None and path.exists()


def _json_dict(value: object) -> JsonDict:
    """Coerce an arbitrary value into a JSON dict.

    Args:
        value: Arbitrary JSON-like value.

    Returns:
        JsonDict: Mapping or empty dict.
    """

    return cast("JsonDict", value) if isinstance(value, dict) else {}


def _as_int(value: object) -> int:
    """Safely coerce a value into an int.

    Args:
        value: Arbitrary scalar value.

    Returns:
        int: Integer value or ``0`` when coercion fails.
    """

    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except ValueError:
            return 0
    return 0


def _family_mean(value: object) -> float | None:
    """Extract the mean f-beta value from a by-family payload row.

    Args:
        value: Family summary payload.

    Returns:
        float | None: Mean f-beta when present.
    """

    family = _json_dict(value)
    f_beta = _json_dict(family.get("f_beta"))
    return _as_float(f_beta.get("mean"))


def _candidate_manifest_row(spec: CandidateSpec) -> JsonDict:
    """Serialize a candidate spec into a JSON-friendly manifest row.

    Args:
        spec: Candidate spec to serialize.

    Returns:
        JsonDict: JSON-friendly manifest row.
    """

    return {
        "label": spec.label,
        "baseline_label": spec.baseline_label,
        "candidate_class": spec.candidate_class,
        "raw_results_path": str(spec.raw_results_path) if spec.raw_results_path is not None else None,
        "production_mimic_path": str(spec.production_mimic_path) if spec.production_mimic_path is not None else None,
        "gate_path": str(spec.gate_path) if spec.gate_path is not None else None,
        "lineage_path": str(spec.lineage_path) if spec.lineage_path is not None else None,
        "control": spec.control,
    }


def _lineage_confidence_label(value: object, *, control: bool) -> LineageConfidence:
    """Normalize lineage confidence into the project label set.

    Args:
        value: Arbitrary lineage confidence value.
        control: Whether the row is a control.

    Returns:
        LineageConfidence: Normalized lineage confidence label.
    """

    if control:
        return "control"
    text = str(value or "unknown").strip().lower() or "unknown"
    if text in {"high", "medium", "low", "unknown"}:
        return cast("LineageConfidence", text)
    return "unknown"


def _candidate_specs() -> list[CandidateSpec]:
    """Return the tracked candidate/control specs for ticket 640.

    Returns:
        list[CandidateSpec]: Candidate specs to evaluate.
    """

    return [
        CandidateSpec(
            label="v6_context_seed",
            baseline_label="v6_context_seed",
            candidate_class="public_anchor_control",
            raw_results_path=ROOT / "platform_runs" / "warmup" / "raw_results_v6_context_seed.json",
            production_mimic_path=None,
            gate_path=None,
            lineage_path=None,
            control=True,
        ),
        CandidateSpec(
            label="main_baseline_20260319",
            baseline_label="main_baseline_20260319",
            candidate_class="current_safe_main_control",
            raw_results_path=ROOT / "platform_runs" / "warmup" / "raw_results_main_baseline_20260319.json",
            production_mimic_path=None,
            gate_path=None,
            lineage_path=None,
            control=True,
        ),
        CandidateSpec(
            label="triad_f331_e0798",
            baseline_label="v6_context_seed",
            candidate_class="support_only_offense",
            raw_results_path=ROOT / "platform_runs" / "warmup" / "raw_results_v_support_combo_9f9_cdddeb_f331_e0798_iter13.json",
            production_mimic_path=RESEARCH_ROOT / "production_mimic_triad_f331_e0798_2026-03-13" / "production_mimic.json",
            gate_path=RESEARCH_ROOT / "candidate_ceiling_cycle_2026-03-13_estimated" / "triad_f331_e0798" / "gate.json",
            lineage_path=RESEARCH_ROOT / "candidate_ceiling_cycle_2026-03-13_estimated" / "triad_f331_e0798" / "lineage.json",
        ),
        CandidateSpec(
            label="v_t20_docfamily_collapse_surrogate_r1",
            baseline_label="v6_context_seed",
            candidate_class="doc_family_collapse_before_page_selection",
            raw_results_path=RESEARCH_ROOT
            / "ticket20_docfamily_collapse_bounded_impl_2026-03-13"
            / "raw_results_ticket20_docfamily_collapse_surrogate.json",
            production_mimic_path=RESEARCH_ROOT / "ticket20_docfamily_collapse_bounded_impl_2026-03-13" / "production_mimic.json",
            gate_path=RESEARCH_ROOT
            / "ticket20_docfamily_collapse_bounded_impl_2026-03-13"
            / "cycle"
            / "v_t20_docfamily_collapse_surrogate_r1"
            / "gate.json",
            lineage_path=RESEARCH_ROOT
            / "ticket20_docfamily_collapse_bounded_impl_2026-03-13"
            / "cycle"
            / "v_t20_docfamily_collapse_surrogate_r1"
            / "lineage.json",
        ),
        CandidateSpec(
            label="t54_single_doc_rerank_gate_r1",
            baseline_label="v6_context_seed",
            candidate_class="single_doc_explicit_provision_page_rerank_gate",
            raw_results_path=None,
            production_mimic_path=RESEARCH_ROOT / "ticket55_single_doc_rerank_gate_eval_2026-03-13" / "production_mimic.json",
            gate_path=RESEARCH_ROOT / "ticket55_single_doc_rerank_gate_eval_2026-03-13" / "experiment_gate.json",
            lineage_path=RESEARCH_ROOT / "ticket55_single_doc_rerank_gate_eval_2026-03-13" / "lineage.json",
        ),
        CandidateSpec(
            label="v_t21_doc_page_rerank_phase1_surrogate_r1",
            baseline_label="v6_context_seed",
            candidate_class="page_localization_rerank",
            raw_results_path=None,
            production_mimic_path=RESEARCH_ROOT / "ticket21_doc_page_rerank_core_phase1_2026-03-13" / "production_mimic.json",
            gate_path=RESEARCH_ROOT
            / "ticket21_doc_page_rerank_core_phase1_2026-03-13"
            / "cycle"
            / "v_t21_doc_page_rerank_phase1_surrogate_r1"
            / "gate.json",
            lineage_path=None,
        ),
        CandidateSpec(
            label="v_t26_within_doc_fallback_pack_r1",
            baseline_label="v6_context_seed",
            candidate_class="within_doc_defensive_fallback",
            raw_results_path=None,
            production_mimic_path=RESEARCH_ROOT / "ticket26_within_doc_defensive_fallback_pack_2026-03-13" / "production_mimic.json",
            gate_path=RESEARCH_ROOT
            / "ticket26_within_doc_defensive_fallback_pack_2026-03-13"
            / "cycle"
            / "v_t26_within_doc_fallback_pack_r1"
            / "gate.json",
            lineage_path=None,
        ),
    ]


def _reviewed_score(raw_results_path: Path | None, golden_path: Path) -> JsonDict:
    """Score one raw-results artifact against a reviewed golden slice.

    Args:
        raw_results_path: Candidate raw-results path.
        golden_path: Reviewed golden path.

    Returns:
        JsonDict: Reviewed score summary or a missing-artifact marker.
    """

    if raw_results_path is None or not raw_results_path.exists():
        return {"available": False}
    result = score_against_golden(raw_results_path, golden_path)
    summary = _json_dict(result.get("summary"))
    by_family = _json_dict(result.get("by_family"))
    return {
        "available": True,
        "overall_grounding_f_beta": _as_float(summary.get("overall_grounding_f_beta")),
        "trusted_grounding_f_beta": _as_float(summary.get("trusted_grounding_f_beta")),
        "weighted_grounding_f_beta": _as_float(summary.get("weighted_grounding_f_beta")),
        "family_f_beta": {
            key: mean
            for key, value in by_family.items()
            if (mean := _family_mean(value)) is not None
        },
    }


def _lineage_confidence(production_mimic: JsonDict, *, control: bool) -> LineageConfidence:
    """Resolve lineage confidence for one candidate row.

    Args:
        production_mimic: Production mimic payload.
        control: Whether this is a control row.

    Returns:
        LineageConfidence: Confidence label.
    """

    if control:
        return "control"
    return _lineage_confidence_label(production_mimic.get("lineage_confidence"), control=control)


def _candidate_question_ids(gate_payload: JsonDict, lineage_payload: JsonDict) -> list[str]:
    """Collect question IDs touched by a candidate artifact.

    Args:
        gate_payload: Gate JSON payload.
        lineage_payload: Lineage JSON payload.

    Returns:
        list[str]: Deterministically ordered touched QIDs.
    """

    qids: list[str] = []
    for key in ("answer_changed_qids", "page_changed_qids"):
        raw = lineage_payload.get(key) or gate_payload.get(key.replace("page_", "retrieval_page_projection_"))
        if isinstance(raw, list):
            qids.extend(str(item).strip() for item in cast("list[object]", raw) if str(item).strip())
    if "seed_deltas" in gate_payload and isinstance(gate_payload["seed_deltas"], list):
        qids.extend(
            str(cast("dict[str, object]", item).get("question_id") or "").strip()
            for item in cast("list[object]", gate_payload["seed_deltas"])
            if isinstance(item, dict) and str(cast("dict[str, object]", item).get("question_id") or "").strip()
        )
    return sorted(dict.fromkeys(qids))


def _build_evidence(spec: CandidateSpec, *, reviewed_all: Path, reviewed_high: Path) -> dict[str, object]:
    """Build a full evidence row for the resurrection matrix.

    Args:
        spec: Candidate/control spec.
        reviewed_all: Reviewed all-100 golden slice.
        reviewed_high: Reviewed high-confidence golden slice.

    Returns:
        dict[str, object]: Evidence row with assessment and reviewed scores.
    """

    production_mimic_outer = _load_json(spec.production_mimic_path)
    production_mimic = cast("JsonDict", production_mimic_outer.get("production_mimic") or {})
    gate_payload = _load_json(spec.gate_path)
    lineage_payload = _load_json(spec.lineage_path)
    question_ids = _candidate_question_ids(gate_payload, lineage_payload)
    hot_overlap = hot_set_overlap_prefixes(question_ids)

    no_submit_reason = str(production_mimic.get("no_submit_reason") or "").strip()
    if not no_submit_reason and spec.control:
        no_submit_reason = "control_row"

    evidence = ResurrectionEvidence(
        label=spec.label,
        candidate_class=str(production_mimic.get("candidate_class") or spec.candidate_class).strip() or spec.candidate_class,
        baseline_label=str(gate_payload.get("baseline_label") or spec.baseline_label).strip() or spec.baseline_label,
        lineage_confidence=_lineage_confidence(production_mimic, control=spec.control),
        answer_changed_count=_as_int(lineage_payload.get("answer_changed_count") or gate_payload.get("answer_changed_count")),
        page_changed_count=_as_int(
            lineage_payload.get("page_changed_count")
            or gate_payload.get("retrieval_page_projection_changed_count")
        ),
        hidden_g_trusted_delta=_as_float(cast("JsonDict", production_mimic.get("hidden_g_trusted") or {}).get("delta")) or 0.0,
        strict_total_estimate=_as_float(production_mimic.get("strict_total_estimate")),
        platform_like_total_estimate=_as_float(production_mimic.get("platform_like_total_estimate")),
        paranoid_total_estimate=_as_float(production_mimic.get("paranoid_total_estimate")),
        no_submit_reason=no_submit_reason,
        tracked_artifacts_ok=_artifact_exists(spec.raw_results_path)
        and (_artifact_exists(spec.lineage_path) or spec.control),
        hot_set_touched_prefixes=hot_overlap,
        control=spec.control,
    )
    assessment = assess_resurrection_candidate(evidence, public_anchor_total=PUBLIC_V6_TOTAL)
    return {
        "label": spec.label,
        "baseline_label": evidence.baseline_label,
        "candidate_class": evidence.candidate_class,
        "control": spec.control,
        "tracked_artifacts_ok": evidence.tracked_artifacts_ok,
        "artifacts": {
            "raw_results": str(spec.raw_results_path) if spec.raw_results_path is not None else None,
            "production_mimic": str(spec.production_mimic_path) if spec.production_mimic_path is not None else None,
            "gate": str(spec.gate_path) if spec.gate_path is not None else None,
            "lineage": str(spec.lineage_path) if spec.lineage_path is not None else None,
        },
        "lineage_confidence": evidence.lineage_confidence,
        "answer_changed_count": evidence.answer_changed_count,
        "page_changed_count": evidence.page_changed_count,
        "hidden_g_trusted_delta": evidence.hidden_g_trusted_delta,
        "strict_total_estimate": evidence.strict_total_estimate,
        "platform_like_total_estimate": evidence.platform_like_total_estimate,
        "paranoid_total_estimate": evidence.paranoid_total_estimate,
        "reviewed_all_100": _reviewed_score(spec.raw_results_path, reviewed_all),
        "reviewed_high_confidence_81": _reviewed_score(spec.raw_results_path, reviewed_high),
        "recommendation": gate_payload.get("recommendation"),
        "hot_set_prefixes_touched": hot_overlap,
        "question_ids_touched_count": len(question_ids),
        "question_ids_touched": question_ids,
        "assessment": assessment.as_dict(),
        "no_submit_reason": no_submit_reason,
    }


def _render_markdown(rows: Sequence[dict[str, object]], *, command: str) -> str:
    """Render the resurrection matrix into markdown.

    Args:
        rows: Evidence rows.
        command: Exact command invocation.

    Returns:
        str: Markdown report.
    """

    shortlist = [row["label"] for row in rows if cast("dict[str, object]", row["assessment"])["status"] == "replay_shortlist"]
    over_penalized = [
        row["label"] for row in rows if cast("dict[str, object]", row["assessment"])["status"] == "probably_over_penalized"
    ]
    toxic = [
        row["label"] for row in rows if cast("dict[str, object]", row["assessment"])["status"] == "toxic_even_if_locally_shiny"
    ]
    confounded = [
        row["label"] for row in rows if cast("dict[str, object]", row["assessment"])["status"] == "confounded_but_interesting"
    ]

    lines = [
        "# 640 Calibrated Resurrection Matrix",
        "",
        f"- command: `{command}`",
        f"- hot_set_prefixes: `{list(HOT_SET_PREFIXES)}`",
        "",
        "## Ranked Matrix",
        "",
        "| label | status | resurrection_score | over_penalized_score | toxicity_score | hidden_g_delta | answer_drift | page_drift | lineage | tracked | hot_set |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        assessment = cast("dict[str, object]", row["assessment"])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    f"`{assessment['status']}`",
                    f"{cast('float', assessment['resurrection_score']):.6f}",
                    f"{cast('float', assessment['over_penalized_score']):.6f}",
                    f"{cast('float', assessment['toxicity_score']):.6f}",
                    f"{cast('float', row['hidden_g_trusted_delta']):.4f}",
                    str(row["answer_changed_count"]),
                    str(row["page_changed_count"]),
                    f"`{row['lineage_confidence']}`",
                    f"`{row['tracked_artifacts_ok']}`",
                    f"`{','.join(cast('list[str]', row['hot_set_prefixes_touched'])) or '-'}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- ranked_shortlist: `{shortlist}`",
            f"- probably_over_penalized: `{over_penalized}`",
            f"- toxic_even_if_locally_shiny: `{toxic}`",
            f"- confounded_but_interesting: `{confounded}`",
            "",
            "## Notes",
            "",
            "- `t54_single_doc_rerank_gate_r1` remains confounded until its tracked candidate artifacts are restored into the main repo tree.",
            "- `v_t21_doc_page_rerank_phase1_surrogate_r1` and `v_t26_within_doc_fallback_pack_r1` are included to keep the toxic lane explicit rather than implicit.",
        ]
    )

    for row in rows:
        assessment = cast("dict[str, object]", row["assessment"])
        lines.extend(
            [
                "",
                f"## {row['label']}",
                "",
                f"- status: `{assessment['status']}`",
                f"- candidate_class: `{row['candidate_class']}`",
                f"- baseline_label: `{row['baseline_label']}`",
                f"- lineage_confidence: `{row['lineage_confidence']}`",
                f"- hidden_g_trusted_delta: `{row['hidden_g_trusted_delta']}`",
                f"- strict_total_estimate: `{row['strict_total_estimate']}`",
                f"- platform_like_total_estimate: `{row['platform_like_total_estimate']}`",
                f"- paranoid_total_estimate: `{row['paranoid_total_estimate']}`",
                f"- answer_changed_count: `{row['answer_changed_count']}`",
                f"- page_changed_count: `{row['page_changed_count']}`",
                f"- hot_set_prefixes_touched: `{row['hot_set_prefixes_touched']}`",
                f"- blocker_terms: `{assessment['blocker_terms']}`",
                f"- confounded_reasons: `{assessment['confounded_reasons']}`",
                f"- reviewed_all_100: `{row['reviewed_all_100']}`",
                f"- reviewed_high_confidence_81: `{row['reviewed_high_confidence_81']}`",
                f"- no_submit_reason: `{row['no_submit_reason'] or 'none'}`",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the matrix builder.

    Returns:
        argparse.Namespace: Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-all", type=Path, default=ROOT / ".sdd" / "golden" / "reviewed" / "reviewed_all_100.json")
    parser.add_argument(
        "--reviewed-high",
        type=Path,
        default=ROOT / ".sdd" / "golden" / "reviewed" / "reviewed_high_confidence_81.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Build the resurrection matrix and write report artifacts.

    Returns:
        int: Process exit code.
    """

    args = parse_args()
    rows = [_build_evidence(spec, reviewed_all=args.reviewed_all, reviewed_high=args.reviewed_high) for spec in _candidate_specs()]
    rows.sort(
        key=lambda row: (
            -cast("float", cast("dict[str, object]", row["assessment"])["resurrection_score"]),
            cast("float", cast("dict[str, object]", row["assessment"])["toxicity_score"]),
            row["label"],
        )
    )

    payload = {
        "command": " ".join(sys.argv),
        "public_anchor_total": PUBLIC_V6_TOTAL,
        "hot_set_prefixes": list(HOT_SET_PREFIXES),
        "ranked_rows": rows,
        "ranked_shortlist": [
            row["label"] for row in rows if cast("dict[str, object]", row["assessment"])["status"] == "replay_shortlist"
        ],
        "probably_over_penalized": [
            row["label"]
            for row in rows
            if cast("dict[str, object]", row["assessment"])["status"] == "probably_over_penalized"
        ],
        "toxic_even_if_locally_shiny": [
            row["label"]
            for row in rows
            if cast("dict[str, object]", row["assessment"])["status"] == "toxic_even_if_locally_shiny"
        ],
        "confounded_but_interesting": [
            row["label"]
            for row in rows
            if cast("dict[str, object]", row["assessment"])["status"] == "confounded_but_interesting"
        ],
        "artifacts_used": [_candidate_manifest_row(spec) for spec in _candidate_specs()],
    }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resurrection_matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "resurrection_matrix.md").write_text(
        _render_markdown(rows, command=" ".join(sys.argv)),
        encoding="utf-8",
    )
    (out_dir / "artifact_manifest.json").write_text(
        json.dumps(payload["artifacts_used"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "closeout.md").write_text(
        "\n".join(
            [
                "# Ticket 640 Closeout",
                "",
                "## Commands Run",
                "",
                f"- `{ ' '.join(sys.argv) }`",
                "",
                "## Outputs",
                "",
                "- `resurrection_matrix.json`",
                "- `resurrection_matrix.md`",
                "- `artifact_manifest.json`",
                "",
                "## Exact Candidates Compared",
                "",
                *[f"- `{spec.label}`" for spec in _candidate_specs()],
                "",
                "## Notes",
                "",
                "- Ranked shortlist, over-penalized list, toxic list, and confounded evidence are all derived from tracked artifacts only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
