"""Tests for the compiled-memory kernel."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from shafi.core.compiled_memory import (
    CompiledMemoryEntry,
    CompiledMemoryKernel,
    MemoryPromotionStatus,
)
from shafi.core.query_contract import QueryContractCompiler

if TYPE_CHECKING:
    from pathlib import Path


def _make_entry(
    memory_id: str,
    *,
    query_family: str,
    route_target: str,
    instruction_patch: str,
    activation_rule: str,
    promotion_score: float,
    reversible: bool = True,
) -> CompiledMemoryEntry:
    """Create one test memory entry.

    Args:
        memory_id: Stable identifier.
        query_family: Sparse query family label.
        route_target: Sparse route target label.
        instruction_patch: Prompt patch text.
        activation_rule: Sparse activation rule.
        promotion_score: Gate score.
        reversible: Whether the entry is reversible.

    Returns:
        CompiledMemoryEntry: Test entry.
    """

    return CompiledMemoryEntry(
        memory_id=memory_id,
        query_family=query_family,
        route_target=route_target,
        instruction_patch=instruction_patch,
        evidence_basis=(f"{memory_id}.md",),
        activation_rule=activation_rule,
        promotion_score=promotion_score,
        created_from_runs=(memory_id,),
        reversible=reversible,
    )


def test_load_entries_supports_wrapped_json_file(tmp_path: Path) -> None:
    """Load entries from a JSON payload with an ``entries`` wrapper."""

    payload = {
        "entries": [
            _make_entry(
                "memory-1",
                query_family="free_text",
                route_target="answer",
                instruction_patch="use proof",
                activation_rule="predicate=explain;answer_type=free_text",
                promotion_score=0.8,
            ).model_dump(mode="json")
        ]
    }
    path = tmp_path / "memory_store.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    entries = CompiledMemoryKernel.load_entries(path)

    assert len(entries) == 1
    assert entries[0].memory_id == "memory-1"


def test_select_entries_matches_query_family_and_activation_rule() -> None:
    """Select entries only when the query family and rule both match."""

    compiler = QueryContractCompiler()
    contract = compiler.compile("Who were the claimants in case CFI 010/2024?", answer_type="names")
    kernel = CompiledMemoryKernel.from_entries(
        [
            _make_entry(
                "field-1",
                query_family="field_lookup",
                route_target="answer",
                instruction_patch="prefer structured field lookup",
                activation_rule="predicate=lookup_field;answer_type=names",
                promotion_score=0.9,
            ),
            _make_entry(
                "field-2",
                query_family="field_lookup",
                route_target="route_policy",
                instruction_patch="use predictor only offline",
                activation_rule="predicate=lookup_field;answer_type=names",
                promotion_score=0.5,
            ),
            _make_entry(
                "compare-1",
                query_family="compare",
                route_target="route_policy",
                instruction_patch="do not compare",
                activation_rule="predicate=compare;min_primary_entities>=2",
                promotion_score=0.7,
            ),
        ]
    )

    selected = kernel.select_entries(contract, route_target="answer")

    assert [entry.memory_id for entry in selected] == ["field-1"]


def test_render_prompt_patch_and_route_hints_are_sparse() -> None:
    """Render a compact prompt patch and structured hints."""

    kernel = CompiledMemoryKernel.from_entries(
        [
            _make_entry(
                "proof-1",
                query_family="free_text",
                route_target="answer",
                instruction_patch="drop unsupported claims",
                activation_rule="predicate=explain;answer_type=free_text",
                promotion_score=0.88,
            ),
            _make_entry(
                "proof-2",
                query_family="free_text",
                route_target="answer",
                instruction_patch="attach provenance",
                activation_rule="predicate=explain;answer_type=free_text",
                promotion_score=0.83,
            ),
        ]
    )

    patch = kernel.render_prompt_patch(kernel.entries)
    hints = kernel.render_route_hints(kernel.entries)

    assert "Compiled Memory Patch" in patch
    assert "proof-1" in patch
    assert hints["active"] is True
    assert hints["memory_ids"] == ["proof-1", "proof-2"]
    assert hints["reversible"] is True


def test_promotion_gate_rejects_weak_entry() -> None:
    """Reject a weak memory candidate that lacks promotion evidence."""

    candidate = _make_entry(
        "weak-1",
        query_family="compare",
        route_target="route_policy",
        instruction_patch="avoid compare",
        activation_rule="predicate=compare;min_primary_entities>=2",
        promotion_score=0.1,
    )

    decision = CompiledMemoryKernel.evaluate_promotion(
        candidate,
        {
            "validated_run_count": 2,
            "activated_subset_delta": -0.2,
            "global_baseline_delta": -0.01,
            "audit_safe": True,
        },
    )

    assert decision.status is MemoryPromotionStatus.REJECTED


def test_promotion_gate_pending_and_disable_path() -> None:
    """Keep the kernel reversible and return pending for incomplete evidence."""

    candidate = _make_entry(
        "pending-1",
        query_family="free_text",
        route_target="answer",
        instruction_patch="keep proof lane",
        activation_rule="predicate=explain;answer_type=free_text",
        promotion_score=0.9,
    )

    pending = CompiledMemoryKernel.evaluate_promotion(
        candidate,
        {
            "validated_run_count": 1,
            "activated_subset_delta": 0.02,
            "global_baseline_delta": 0.0,
            "audit_safe": True,
        },
    )
    disabled_kernel = CompiledMemoryKernel.from_entries([candidate]).disabled_copy()

    assert pending.status is MemoryPromotionStatus.PENDING
    assert (
        disabled_kernel.select_entries(
            QueryContractCompiler().compile("Explain the case", answer_type="free_text"),
            route_target="answer",
        )
        == []
    )
