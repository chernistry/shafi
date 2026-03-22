#!/usr/bin/env python3
"""Promote sparse compiled-memory entries from validated closeouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rag_challenge.core.compiled_memory import (
    CompiledMemoryEntry,
    CompiledMemoryKernel,
    MemoryPromotionDecision,
    MemoryPromotionStatus,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class MemoryBlueprint:
    """Static candidate blueprint derived from validated closeouts."""

    memory_id: str
    query_family: str
    route_target: str
    instruction_patch: str
    activation_rule: str
    promotion_score: float
    created_from_runs: tuple[str, ...]
    evidence_basis: tuple[str, ...]
    validation_summary: dict[str, object]
    reversible: bool = True


def _default_blueprints(source_root: Path) -> list[MemoryBlueprint]:
    """Build the default curated blueprints for the current branch state.

    Args:
        source_root: Research root containing closeouts.

    Returns:
        list[MemoryBlueprint]: Candidate memory blueprints.
    """

    def closeout_path(ticket_dir: str) -> Path:
        return source_root / ticket_dir / "closeout.md"

    blueprints = [
        MemoryBlueprint(
            memory_id="proof_fail_closed_free_text",
            query_family="free_text",
            route_target="answer",
            instruction_patch=(
                "Prefer proof-carrying compilation for free-text when support coverage is high; "
                "drop unsupported claims instead of inventing bridge text."
            ),
            activation_rule="predicate=explain;answer_type=free_text",
            promotion_score=0.86,
            created_from_runs=("1058",),
            evidence_basis=(str(closeout_path("1058_proof_carrying_answer_compiler_r2_2026-03-19")),),
            validation_summary={
                "validated_run_count": 2,
                "activated_subset_delta": 0.032,
                "global_baseline_delta": 0.012,
                "audit_safe": True,
                "false_positive_rate": 0.04,
            },
        ),
        MemoryBlueprint(
            memory_id="field_utility_offline_only",
            query_family="field_lookup",
            route_target="route_policy",
            instruction_patch=(
                "Use the retrieval utility predictor as an offline evidence signal only; "
                "do not treat it as a runtime route override without stronger structured routes."
            ),
            activation_rule="predicate=lookup_field;answer_type=name",
            promotion_score=0.82,
            created_from_runs=("1052",),
            evidence_basis=(str(closeout_path("1052_retrieval_utility_predictor_r2_2026-03-19")),),
            validation_summary={
                "validated_run_count": 2,
                "activated_subset_delta": 0.22999999999999998,
                "global_baseline_delta": 0.0,
                "audit_safe": True,
                "false_positive_rate": 0.01,
            },
        ),
        MemoryBlueprint(
            memory_id="compare_holdout_no_go",
            query_family="compare",
            route_target="route_policy",
            instruction_patch=(
                "Do not promote compare join on the current corpus substrate; keep compare handling "
                "on the fallback path until entity coverage and exact-match quality improve."
            ),
            activation_rule="predicate=compare;min_primary_entities>=2",
            promotion_score=0.39,
            created_from_runs=("1055",),
            evidence_basis=(str(closeout_path("1055_compare_join_engine_r1_2026-03-19")),),
            validation_summary={
                "validated_run_count": 1,
                "activated_subset_delta": -0.714,
                "global_baseline_delta": -0.02,
                "audit_safe": True,
                "false_positive_rate": 0.52,
            },
        ),
        MemoryBlueprint(
            memory_id="temporal_holdout_no_go",
            query_family="temporal",
            route_target="route_policy",
            instruction_patch=(
                "Keep temporal/applicability routing disabled until corpus coverage materially improves; "
                "the current substrate does not activate reliably."
            ),
            activation_rule="predicate=temporal;time_scope=current",
            promotion_score=0.31,
            created_from_runs=("1056",),
            evidence_basis=(str(closeout_path("1056_temporal_applicability_engine_r1_2026-03-19")),),
            validation_summary={
                "validated_run_count": 1,
                "activated_subset_delta": -1.0,
                "global_baseline_delta": -0.01,
                "audit_safe": True,
                "false_positive_rate": 0.0,
            },
            reversible=True,
        ),
    ]
    return blueprints


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / ".sdd" / "researches",
        help="Research root containing validated closeouts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".sdd" / "researches" / "1064_compiled_memory_kernel_r1_2026-03-19",
        help="Output directory for the memory store and report.",
    )
    parser.add_argument(
        "--memory-store-name",
        default="memory_store.json",
        help="JSON file name for the accepted memory entries.",
    )
    parser.add_argument(
        "--report-name",
        default="promotion_report.md",
        help="Markdown file name for the promotion report.",
    )
    return parser.parse_args()


def _build_memory_records(source_root: Path) -> tuple[list[CompiledMemoryEntry], list[MemoryPromotionDecision]]:
    """Build candidate memory records from the current validated closeouts.

    Args:
        source_root: Research root containing closeouts.

    Returns:
        tuple[list[CompiledMemoryEntry], list[MemoryPromotionDecision]]: Accepted entries and gate decisions.
    """

    accepted: list[CompiledMemoryEntry] = []
    decisions: list[MemoryPromotionDecision] = []
    available_closeouts = {path.parent.name for path in source_root.rglob("closeout.md")}
    for blueprint in _default_blueprints(source_root):
        if not any(
            any(run_id in closeout_name for closeout_name in available_closeouts)
            for run_id in blueprint.created_from_runs
        ):
            decisions.append(
                MemoryPromotionDecision(
                    status=MemoryPromotionStatus.PENDING,
                    reason=f"missing closeout evidence for {blueprint.memory_id}",
                    validation_summary=blueprint.validation_summary,
                )
            )
            continue
        entry = CompiledMemoryEntry(
            memory_id=blueprint.memory_id,
            query_family=blueprint.query_family,
            route_target=blueprint.route_target,
            instruction_patch=blueprint.instruction_patch,
            evidence_basis=blueprint.evidence_basis,
            activation_rule=blueprint.activation_rule,
            promotion_score=blueprint.promotion_score,
            created_from_runs=blueprint.created_from_runs,
            reversible=blueprint.reversible,
        )
        decision = CompiledMemoryKernel.evaluate_promotion(entry, blueprint.validation_summary)
        decisions.append(decision)
        if decision.status is MemoryPromotionStatus.ACCEPTED:
            accepted.append(entry)
    return accepted, decisions


def _render_report(
    *,
    accepted: list[CompiledMemoryEntry],
    decisions: list[MemoryPromotionDecision],
    source_root: Path,
    output_dir: Path,
) -> str:
    """Render a human-readable promotion report.

    Args:
        accepted: Accepted memory entries.
        decisions: Gate decisions.
        source_root: Research root containing closeouts.
        output_dir: Report output directory.

    Returns:
        str: Markdown report content.
    """

    status_counts = {
        "accepted": sum(decision.status is MemoryPromotionStatus.ACCEPTED for decision in decisions),
        "rejected": sum(decision.status is MemoryPromotionStatus.REJECTED for decision in decisions),
        "pending": sum(decision.status is MemoryPromotionStatus.PENDING for decision in decisions),
    }
    kernel = CompiledMemoryKernel.from_entries(accepted)
    reviewed_summary = _build_activation_summary(kernel, source_root=source_root)
    lines = [
        "# Compiled Memory Promotion Report",
        "",
        f"- source_root: `{source_root}`",
        f"- output_dir: `{output_dir}`",
        f"- accepted_entries: `{status_counts['accepted']}`",
        f"- rejected_entries: `{status_counts['rejected']}`",
        f"- pending_entries: `{status_counts['pending']}`",
        "",
        "## Accepted Entries",
    ]
    if accepted:
        for entry in accepted:
            lines.append(
                f"- `{entry.memory_id}` | family=`{entry.query_family}` | route=`{entry.route_target}` | score=`{entry.promotion_score:.3f}`"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Decisions",
        ]
    )
    for decision in decisions:
        lines.append(f"- `{decision.status.value}`: {decision.reason}")
    lines.extend(
        [
            "",
            "## Activation Summary",
            f"- reviewed_all_100_activation_rate: `{reviewed_summary['all_100']['activation_rate']:.3f}`",
            f"- reviewed_high_81_activation_rate: `{reviewed_summary['high_81']['activation_rate']:.3f}`",
            f"- activated_subset_label_weight_delta: `{reviewed_summary['label_weight_delta']:.3f}`",
            "- family_activation_rates:",
            "",
            "## Rollback",
            "- delete the generated memory store and re-run without loading compiled memory if any activation causes prompt bloat or route drift",
        ]
    )
    for family, rate in sorted(reviewed_summary["family_activation_rates"].items()):
        lines.insert(lines.index("- family_activation_rates:") + 1, f"  - {family}: `{rate:.3f}`")
    return "\n".join(lines) + "\n"


def _build_activation_summary(
    kernel: CompiledMemoryKernel,
    *,
    source_root: Path,
) -> dict[str, Any]:
    """Build a compact activation summary on reviewed slices.

    Args:
        kernel: Compiled memory kernel.
        source_root: Research root containing reviewed gold.

    Returns:
        dict[str, Any]: Summary with activation rates and label-weight proxy delta.
    """

    from rag_challenge.core.query_contract import QueryContractCompiler

    reviewed_all = _load_reviewed_slice(source_root.parent / "golden" / "reviewed" / "reviewed_all_100.json")
    reviewed_high = _load_reviewed_slice(source_root.parent / "golden" / "reviewed" / "reviewed_high_confidence_81.json")
    compiler = QueryContractCompiler()
    all_activation = _activation_rate(kernel, compiler, reviewed_all)
    high_activation = _activation_rate(kernel, compiler, reviewed_high)
    family_activation_rates = _family_activation_rates(kernel, compiler, reviewed_all)
    activated_weight = _label_weight_average(
        kernel,
        compiler,
        reviewed_all,
        active=True,
    )
    inactive_weight = _label_weight_average(
        kernel,
        compiler,
        reviewed_all,
        active=False,
    )
    return {
        "all_100": {"activation_rate": all_activation},
        "high_81": {"activation_rate": high_activation},
        "label_weight_delta": activated_weight - inactive_weight,
        "family_activation_rates": family_activation_rates,
    }


def _load_reviewed_slice(path: Path) -> list[dict[str, Any]]:
    """Load one reviewed benchmark slice.

    Args:
        path: JSON file path.

    Returns:
        list[dict[str, object]]: Reviewed rows.
    """

    if not path.exists():
        return []
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return cast("list[dict[str, Any]]", payload)
    return []


def _activation_rate(kernel: CompiledMemoryKernel, compiler: Any, rows: list[dict[str, Any]]) -> float:
    """Compute the activation rate on one reviewed slice.

    Args:
        kernel: Compiled memory kernel.
        compiler: Query contract compiler.
        rows: Reviewed rows.

    Returns:
        float: Activation rate.
    """

    if not rows:
        return 0.0
    active_count = 0
    for row in rows:
        contract = compiler.compile(
            str(row.get("question", "")),
            answer_type=str(row.get("answer_type", "free_text")),
        )
        if kernel.select_entries(contract, route_target=_route_target_for_contract(contract)):
            active_count += 1
    return active_count / len(rows)


def _family_activation_rates(
    kernel: CompiledMemoryKernel,
    compiler: Any,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute activation rates by inferred query family.

    Args:
        kernel: Compiled memory kernel.
        compiler: Query contract compiler.
        rows: Reviewed rows.

    Returns:
        dict[str, float]: Activation rate by family.
    """

    totals: dict[str, int] = {}
    hits: dict[str, int] = {}
    for row in rows:
        contract = compiler.compile(
            str(row.get("question", "")),
            answer_type=str(row.get("answer_type", "free_text")),
        )
        family = _family_for_contract(contract)
        totals[family] = totals.get(family, 0) + 1
        if kernel.select_entries(contract, route_target=_route_target_for_contract(contract)):
            hits[family] = hits.get(family, 0) + 1
    return {
        family: hits.get(family, 0) / total
        for family, total in totals.items()
        if total > 0
    }


def _label_weight_average(
    kernel: CompiledMemoryKernel,
    compiler: Any,
    rows: list[dict[str, Any]],
    *,
    active: bool,
) -> float:
    """Compute a simple label-weight proxy for activated or non-activated rows.

    Args:
        kernel: Compiled memory kernel.
        compiler: Query contract compiler.
        rows: Reviewed rows.
        active: Whether to average activated or non-activated rows.

    Returns:
        float: Average label weight proxy.
    """

    weights: list[float] = []
    for row in rows:
        contract = compiler.compile(
            str(row.get("question", "")),
            answer_type=str(row.get("answer_type", "free_text")),
        )
        is_active = bool(kernel.select_entries(contract, route_target=_route_target_for_contract(contract)))
        if is_active != active:
            continue
        try:
            weights.append(float(row.get("label_weight", 0.0)))
        except (TypeError, ValueError):
            continue
    if not weights:
        return 0.0
    return sum(weights) / len(weights)


def _route_target_for_contract(contract: Any) -> str:
    """Choose a sparse route target for one query contract.

    Args:
        contract: Query contract.

    Returns:
        str: Sparse route target label.
    """

    if contract.predicate.value == "lookup_field":
        return "answer"
    if contract.predicate.value in {"compare", "temporal"}:
        return "route_policy"
    return "answer"


def _family_for_contract(contract: Any) -> str:
    """Infer the sparse query family used for activation reporting.

    Args:
        contract: Query contract.

    Returns:
        str: Sparse family label.
    """

    if contract.predicate.value == "lookup_field":
        return "field_lookup"
    if contract.predicate.value == "compare":
        return "compare"
    if contract.predicate.value == "temporal":
        return "temporal"
    if contract.predicate.value == "lookup_provision":
        return "provision"
    if contract.predicate.value == "enumerate":
        return "enumeration"
    if contract.answer_type == "boolean":
        return "boolean"
    return "free_text"


def main() -> int:
    """Promote compiled memory entries and write a human-readable report.

    Returns:
        int: Process exit status.
    """

    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted, decisions = _build_memory_records(args.source_dir.resolve())
    store_payload = {
        "entries": [entry.model_dump(mode="json") for entry in accepted],
        "decisions": [
            {
                "status": decision.status.value,
                "reason": decision.reason,
                "validation_summary": decision.validation_summary,
            }
            for decision in decisions
        ],
    }
    (output_dir / args.memory_store_name).write_text(
        json.dumps(store_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = _render_report(
        accepted=accepted,
        decisions=decisions,
        source_root=args.source_dir.resolve(),
        output_dir=output_dir,
    )
    (output_dir / args.report_name).write_text(report, encoding="utf-8")
    (output_dir / "promotion_summary.json").write_text(
        json.dumps(
            {
                "accepted_count": len(accepted),
                "decision_count": len(decisions),
                "accepted_memory_ids": [entry.memory_id for entry in accepted],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
