from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

JsonDict = dict[str, object]
StageStatus = Literal["passed", "blocked", "pending", "skipped"]
PreconditionStatus = Literal["passed", "blocked", "assumed_passed"]

STAGE_ORDER: tuple[tuple[str, str], ...] = (
    ("stage0_static_safety", "Stage 0: static safety"),
    ("stage1_impact_routed_canary", "Stage 1: impact-routed canary"),
    ("stage2_changed_set_truth", "Stage 2: changed-set truth"),
    ("stage3_strict_local_production_mimic", "Stage 3: strict local production mimic"),
    ("stage4_judge", "Stage 4: judge"),
)


def _empty_json_dict() -> JsonDict:
    return {}


@dataclass(frozen=True)
class PreparedStage:
    key: str
    label: str
    status: StageStatus
    block_reason: str | None = None
    details: JsonDict = field(default_factory=_empty_json_dict)


def prepare_precondition_stage(
    *,
    key: str,
    status: PreconditionStatus,
    reason: str | None = None,
    details: JsonDict | None = None,
) -> PreparedStage:
    status_value: StageStatus = "blocked" if status == "blocked" else "passed"
    payload = dict(details or {})
    if status == "assumed_passed":
        payload["assumed"] = True
    if status == "blocked":
        payload["assumed"] = False
    label = dict(STAGE_ORDER)[key]
    return PreparedStage(
        key=key,
        label=label,
        status=status_value,
        block_reason=reason if status_value == "blocked" else None,
        details=payload,
    )


def prepare_measured_stage(
    *,
    key: str,
    passed: bool,
    reason: str | None = None,
    details: JsonDict | None = None,
) -> PreparedStage:
    label = dict(STAGE_ORDER)[key]
    return PreparedStage(
        key=key,
        label=label,
        status="passed" if passed else "blocked",
        block_reason=reason if not passed else None,
        details=dict(details or {}),
    )


def evaluate_stage_sequence(stages: list[PreparedStage]) -> JsonDict:
    stage_by_key = {stage.key: stage for stage in stages}
    ordered: list[JsonDict] = []
    blocked_stage: str | None = None
    block_reason: str | None = None
    completed: list[str] = []

    for key, label in STAGE_ORDER:
        stage = stage_by_key.get(key)
        if blocked_stage is not None:
            ordered.append(
                asdict(
                    PreparedStage(
                        key=key,
                        label=label,
                        status="skipped",
                        details={"blocked_by": blocked_stage},
                    )
                )
            )
            continue

        if stage is None:
            ordered.append(asdict(PreparedStage(key=key, label=label, status="pending")))
            continue

        ordered.append(asdict(stage))
        completed.append(key)
        if stage.status == "blocked":
            blocked_stage = key
            block_reason = stage.block_reason or f"{label} blocked"

    return {
        "stage_order": [key for key, _label in STAGE_ORDER],
        "completed_stages": completed,
        "blocked_stage": blocked_stage,
        "block_reason": block_reason,
        "next_stage": None if blocked_stage is not None else "stage3_strict_local_production_mimic",
        "should_run_stage3": blocked_stage is None,
        "stage_results": ordered,
    }
