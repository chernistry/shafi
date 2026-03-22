"""Sparse compiled-memory kernel for auditable route-policy hints."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field

from shafi.core.query_contract import (
    PredicateType,
    QueryContract,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _tuple_str_factory() -> tuple[str, ...]:
    """Build a typed empty string tuple for Pydantic defaults.

    Returns:
        tuple[str, ...]: Empty string tuple.
    """

    return ()


class MemoryPromotionStatus(StrEnum):
    """Promotion status for one compiled-memory candidate."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending"


class CompiledMemoryEntry(BaseModel):
    """One sparse memory entry promoted from validated runs.

    Args:
        memory_id: Stable entry identifier.
        query_family: Query family this memory applies to.
        route_target: Intended consumer or route name.
        instruction_patch: Short prompt or policy fragment.
        evidence_basis: Human-readable provenance notes.
        activation_rule: Sparse activation rule for selection.
        promotion_score: Confidence score for promotion gating.
        created_from_runs: Source run or closeout identifiers.
        reversible: Whether the entry can be disabled cleanly.
    """

    model_config = ConfigDict(frozen=True)

    memory_id: str
    query_family: str
    route_target: str
    instruction_patch: str
    evidence_basis: tuple[str, ...] = Field(default_factory=_tuple_str_factory)
    activation_rule: str = ""
    promotion_score: float = 0.0
    created_from_runs: tuple[str, ...] = Field(default_factory=_tuple_str_factory)
    reversible: bool = True


class MemoryPromotionDecision(BaseModel):
    """Promotion decision for one compiled-memory candidate."""

    model_config = ConfigDict(frozen=True)

    status: MemoryPromotionStatus
    reason: str
    validation_summary: dict[str, object] = Field(default_factory=dict)


class CompiledMemoryKernel(BaseModel):
    """In-memory view over sparse compiled-memory entries."""

    model_config = ConfigDict(frozen=True)

    entries: tuple[CompiledMemoryEntry, ...] = Field(default_factory=tuple)
    enabled: bool = True

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[CompiledMemoryEntry],
        *,
        enabled: bool = True,
    ) -> CompiledMemoryKernel:
        """Build a compiled-memory kernel from typed entries.

        Args:
            entries: Candidate memory entries.
            enabled: Whether the kernel is active.

        Returns:
            CompiledMemoryKernel: Typed kernel wrapper.
        """

        return cls(entries=tuple(entries), enabled=enabled)

    @classmethod
    def from_path(cls, path: str | Path, *, enabled: bool = True) -> CompiledMemoryKernel:
        """Load a compiled-memory kernel from a persisted memory store.

        Args:
            path: JSON file or directory containing the memory store.
            enabled: Whether the kernel is active.

        Returns:
            CompiledMemoryKernel: Loaded kernel.
        """

        return cls(entries=tuple(cls.load_entries(path)), enabled=enabled)

    @classmethod
    def load_entries(cls, path: str | Path) -> list[CompiledMemoryEntry]:
        """Load compiled-memory entries from a JSON file or directory.

        Args:
            path: JSON file or directory containing ``memory_store.json``.

        Returns:
            list[CompiledMemoryEntry]: Loaded memory entries.
        """

        resolved_path = Path(path)
        if resolved_path.is_dir():
            candidate = resolved_path / "memory_store.json"
            if candidate.exists():
                resolved_path = candidate
            else:
                json_files = sorted(resolved_path.glob("*.json"))
                if not json_files:
                    return []
                resolved_path = json_files[0]
        if not resolved_path.exists():
            return []
        payload: Any = json.loads(resolved_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload_dict = cast("dict[str, Any]", payload)
            raw_entries_obj: Any = payload_dict.get("entries", payload_dict)
        else:
            raw_entries_obj = payload
        if not isinstance(raw_entries_obj, list):
            return []
        raw_entries = cast("list[Any]", raw_entries_obj)
        return [CompiledMemoryEntry.model_validate(item) for item in raw_entries]

    def disabled_copy(self) -> CompiledMemoryKernel:
        """Return a disabled copy of the current kernel.

        Returns:
            CompiledMemoryKernel: Disabled kernel copy.
        """

        return self.model_copy(update={"enabled": False})

    def select_entries(self, query_contract: QueryContract, route_target: str) -> list[CompiledMemoryEntry]:
        """Select the memory entries that apply to one query and route target.

        Args:
            query_contract: Typed query contract.
            route_target: Intended consumer route.

        Returns:
            list[CompiledMemoryEntry]: Matching entries, sorted by score.
        """

        if not self.enabled:
            return []

        resolved_family = _infer_query_family(query_contract)
        resolved_route_target = route_target.strip().casefold()
        selected: list[CompiledMemoryEntry] = []
        for entry in self.entries:
            if not entry.reversible:
                continue
            if entry.query_family.casefold() != resolved_family:
                continue
            if (
                resolved_route_target
                and resolved_route_target != "*"
                and entry.route_target.casefold() != resolved_route_target
            ):
                continue
            if not _activation_rule_matches(
                entry.activation_rule,
                query_contract,
                route_target=resolved_route_target,
                query_family=resolved_family,
            ):
                continue
            selected.append(entry)
        selected.sort(key=lambda item: (-item.promotion_score, item.memory_id))
        return selected

    def render_prompt_patch(self, entries: Sequence[CompiledMemoryEntry]) -> str:
        """Render a compact prompt patch from selected entries.

        Args:
            entries: Selected memory entries.

        Returns:
            str: Human-readable prompt fragment.
        """

        if not self.enabled or not entries:
            return ""
        lines = ["# Compiled Memory Patch"]
        for entry in entries:
            evidence = "; ".join(entry.evidence_basis) if entry.evidence_basis else "n/a"
            lines.append(
                f"- {entry.memory_id} | family={entry.query_family} | route={entry.route_target} | score={entry.promotion_score:.3f}"
            )
            lines.append(f"  instruction: {entry.instruction_patch}")
            lines.append(f"  evidence: {evidence}")
        return "\n".join(lines)

    def render_route_hints(self, entries: Sequence[CompiledMemoryEntry]) -> dict[str, object]:
        """Render structured route hints from selected entries.

        Args:
            entries: Selected memory entries.

        Returns:
            dict[str, object]: Structured route-hint payload.
        """

        if not self.enabled or not entries:
            return {}
        return {
            "active": True,
            "memory_ids": [entry.memory_id for entry in entries],
            "query_families": sorted({entry.query_family for entry in entries}),
            "route_targets": sorted({entry.route_target for entry in entries}),
            "instruction_patches": [entry.instruction_patch for entry in entries],
            "promotion_scores": {entry.memory_id: round(entry.promotion_score, 6) for entry in entries},
            "reversible": all(entry.reversible for entry in entries),
            "evidence_basis": [list(entry.evidence_basis) for entry in entries],
            "created_from_runs": [list(entry.created_from_runs) for entry in entries],
        }

    @staticmethod
    def evaluate_promotion(
        candidate: CompiledMemoryEntry,
        validation_summary: Mapping[str, object],
        *,
        min_promotion_score: float = 0.65,
        min_validated_runs: int = 2,
    ) -> MemoryPromotionDecision:
        """Evaluate whether one candidate clears the promotion gate.

        Args:
            candidate: Memory candidate under review.
            validation_summary: Validation metadata gathered from the source runs.
            min_promotion_score: Minimum score required for acceptance.
            min_validated_runs: Minimum number of validated runs required.

        Returns:
            MemoryPromotionDecision: Gate outcome with a reason.
        """

        summary = dict(validation_summary)
        if not candidate.reversible:
            return MemoryPromotionDecision(
                status=MemoryPromotionStatus.REJECTED,
                reason="candidate is not reversible",
                validation_summary=summary,
            )

        validated_run_count = _coerce_int(summary.get("validated_run_count", 0))
        if validated_run_count < min_validated_runs:
            return MemoryPromotionDecision(
                status=MemoryPromotionStatus.PENDING,
                reason=f"validated_run_count={validated_run_count} below threshold={min_validated_runs}",
                validation_summary=summary,
            )

        if candidate.promotion_score < min_promotion_score:
            return MemoryPromotionDecision(
                status=MemoryPromotionStatus.REJECTED,
                reason=f"promotion_score={candidate.promotion_score:.3f} below threshold={min_promotion_score:.3f}",
                validation_summary=summary,
            )

        activated_delta = _coerce_float(summary.get("activated_subset_delta", summary.get("activated_delta", 0.0)))
        global_delta = _coerce_float(summary.get("global_baseline_delta", summary.get("global_delta", 0.0)))
        if activated_delta <= 0.0 and global_delta <= 0.0:
            return MemoryPromotionDecision(
                status=MemoryPromotionStatus.REJECTED,
                reason="no positive activated-subset or global delta",
                validation_summary=summary,
            )

        if not bool(summary.get("audit_safe", True)):
            return MemoryPromotionDecision(
                status=MemoryPromotionStatus.REJECTED,
                reason="validation summary is not audit-safe",
                validation_summary=summary,
            )

        false_positive_rate = _coerce_float(summary.get("false_positive_rate", 0.0))
        if false_positive_rate > 0.15:
            return MemoryPromotionDecision(
                status=MemoryPromotionStatus.REJECTED,
                reason=f"false_positive_rate={false_positive_rate:.3f} above threshold=0.150",
                validation_summary=summary,
            )

        memory_bloat_tokens = _coerce_int(summary.get("memory_bloat_tokens", 0))
        if memory_bloat_tokens > 120:
            return MemoryPromotionDecision(
                status=MemoryPromotionStatus.REJECTED,
                reason=f"memory_bloat_tokens={memory_bloat_tokens} above threshold=120",
                validation_summary=summary,
            )

        if activated_delta > 0.0 or global_delta > 0.0:
            return MemoryPromotionDecision(
                status=MemoryPromotionStatus.ACCEPTED,
                reason="candidate clears promotion gate",
                validation_summary=summary,
            )
        return MemoryPromotionDecision(
            status=MemoryPromotionStatus.PENDING,
            reason="promotion evidence is still ambiguous",
            validation_summary=summary,
        )


def _infer_query_family(contract: QueryContract) -> str:
    """Infer a sparse query-family label from a query contract.

    Args:
        contract: Typed query contract.

    Returns:
        str: Sparse family label.
    """

    if contract.predicate is PredicateType.LOOKUP_FIELD:
        return "field_lookup"
    if contract.predicate is PredicateType.COMPARE:
        return "compare"
    if contract.predicate is PredicateType.TEMPORAL:
        return "temporal"
    if contract.predicate is PredicateType.LOOKUP_PROVISION:
        return "provision"
    if contract.predicate is PredicateType.ENUMERATE:
        return "enumeration"
    if contract.answer_type.strip().casefold() == "boolean":
        return "boolean"
    return "free_text"


def _activation_rule_matches(
    rule: str,
    contract: QueryContract,
    *,
    route_target: str,
    query_family: str,
) -> bool:
    """Check whether a sparse activation rule applies to one contract.

    Args:
        rule: Rule string in ``key<op>value`` form.
        contract: Typed query contract.
        route_target: Resolved route target.
        query_family: Resolved query family.

    Returns:
        bool: True when the rule matches or is empty.
    """

    normalized_rule = rule.strip()
    if not normalized_rule:
        return True
    clauses = [clause.strip() for clause in re.split(r"[;,]", normalized_rule) if clause.strip()]
    for clause in clauses:
        if not _match_clause(clause, contract, route_target=route_target, query_family=query_family):
            return False
    return True


def _match_clause(
    clause: str,
    contract: QueryContract,
    *,
    route_target: str,
    query_family: str,
) -> bool:
    """Evaluate one activation-rule clause against a query contract.

    Args:
        clause: One sparse rule clause.
        contract: Typed query contract.
        route_target: Resolved route target.
        query_family: Resolved query family.

    Returns:
        bool: True when the clause matches.
    """

    match = re.fullmatch(r"(?P<key>[a-z_]+)\s*(?P<op>>=|<=|=|>|<)\s*(?P<value>.+)", clause)
    if match is None:
        return False
    key = match.group("key").strip().casefold()
    op = match.group("op")
    raw_value = match.group("value").strip()
    text_value = raw_value.casefold()

    if key == "query_family":
        return op == "=" and query_family == text_value
    if key == "route_target":
        return op == "=" and route_target == text_value
    if key == "predicate":
        return op == "=" and contract.predicate.value == text_value
    if key == "answer_type":
        return op == "=" and contract.answer_type.casefold() == text_value
    if key == "field_name":
        return op == "=" and contract.field_name.casefold() == text_value
    if key == "time_scope":
        return op == "=" and contract.time_scope.scope_type.value == text_value
    if key == "contains":
        return op == "=" and text_value in contract.query_text.casefold()
    if key == "comparison_axis":
        return op == "=" and text_value in {axis.casefold() for axis in contract.comparison_axes}

    numeric_value: float
    if key == "confidence":
        numeric_value = contract.confidence
    elif key == "min_primary_entities":
        numeric_value = float(len(contract.primary_entities))
    elif key == "min_constraint_entities":
        numeric_value = float(len(contract.constraint_entities))
    else:
        return False

    try:
        threshold = float(raw_value)
    except ValueError:
        return False
    return _compare_numeric(numeric_value, threshold, op)


def _compare_numeric(left: float, right: float, op: str) -> bool:
    """Compare two numeric values using a simple operator.

    Args:
        left: Left-hand numeric value.
        right: Right-hand numeric value.
        op: Comparison operator.

    Returns:
        bool: Comparison result.
    """

    if op == "=":
        return left == right
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    return False


def _coerce_float(value: Any) -> float:
    """Coerce a loose value into a float.

    Args:
        value: Arbitrary validation value.

    Returns:
        float: Coerced float or ``0.0``.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(value: Any) -> int:
    """Coerce a loose value into an integer.

    Args:
        value: Arbitrary validation value.

    Returns:
        int: Coerced integer or ``0``.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
