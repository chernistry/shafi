"""Deterministic scoring helpers for calibrated candidate resurrection."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

LineageConfidence = Literal["high", "medium", "low", "unknown", "control"]

HOT_SET_PREFIXES: tuple[str, ...] = (
    "9f9fb4",
    "cdddeb",
    "e0798b",
    "f33177",
    "d6eb4a",
    "2e211d",
    "d374be",
)
_KNOWN_TOXIC_BLOCKERS = (
    "page drift without trusted hidden-g gain",
    "citation overbreadth exceeded strict local budget",
    "pages-per-doc exceeded strict local budget",
    "page-id precision below strict local floor",
    "page-id recall below strict local floor",
)


@dataclass(frozen=True)
class ResurrectionEvidence:
    """Minimal evidence needed to rank a historical candidate.

    Args:
        label: Candidate label.
        candidate_class: Candidate class/family.
        baseline_label: Baseline label used by the artifact.
        lineage_confidence: Evidence lineage confidence.
        answer_changed_count: Answer drift count.
        page_changed_count: Page drift count.
        hidden_g_trusted_delta: Trusted hidden-G delta.
        strict_total_estimate: Strict total estimate.
        platform_like_total_estimate: Platform-like total estimate.
        paranoid_total_estimate: Paranoid total estimate.
        no_submit_reason: Local no-submit rationale.
        tracked_artifacts_ok: Whether tracked artifacts needed for replay exist.
        hot_set_touched_prefixes: Consensus hot-set prefixes touched by the candidate.
        control: Whether this row is a control rather than a resurrection candidate.
    """

    label: str
    candidate_class: str
    baseline_label: str
    lineage_confidence: LineageConfidence
    answer_changed_count: int
    page_changed_count: int
    hidden_g_trusted_delta: float
    strict_total_estimate: float | None
    platform_like_total_estimate: float | None
    paranoid_total_estimate: float | None
    no_submit_reason: str
    tracked_artifacts_ok: bool
    hot_set_touched_prefixes: list[str]
    control: bool = False


@dataclass(frozen=True)
class ResurrectionAssessment:
    """Ranked resurrection assessment for one candidate.

    Args:
        label: Candidate label.
        status: Deterministic classification result.
        resurrection_score: Overall EV-style score.
        over_penalized_score: Score for “likely blocked by policy”.
        toxicity_score: Score for “likely still toxic”.
        confidence_weight: Numeric lineage confidence weight.
        blocker_terms: Parsed blocker terms.
        confounded_reasons: Concrete reasons the evidence remains confounded.
    """

    label: str
    status: str
    resurrection_score: float
    over_penalized_score: float
    toxicity_score: float
    confidence_weight: float
    blocker_terms: list[str]
    confounded_reasons: list[str]

    def as_dict(self) -> dict[str, object]:
        """Serialize the assessment to a JSON-friendly dict.

        Returns:
            dict[str, object]: Serialized assessment.
        """

        return asdict(self)


def confidence_weight(confidence: LineageConfidence) -> float:
    """Map lineage confidence to a deterministic numeric weight.

    Args:
        confidence: Candidate lineage confidence label.

    Returns:
        float: Weight used by the ranking heuristic.
    """

    return {
        "high": 1.0,
        "medium": 0.72,
        "low": 0.45,
        "unknown": 0.35,
        "control": 1.0,
    }.get(confidence, 0.35)


def parse_blocker_terms(reason: str) -> list[str]:
    """Split a no-submit reason into normalized blocker terms.

    Args:
        reason: Free-text local no-submit reason.

    Returns:
        list[str]: Lower-cased blocker strings.
    """

    if not reason.strip():
        return []
    parts = [part.strip().lower() for part in re.split(r"[;]", reason) if part.strip()]
    return list(dict.fromkeys(parts))


def hot_set_overlap_prefixes(question_ids: list[str]) -> list[str]:
    """Return consensus hot-set prefixes touched by the question list.

    Args:
        question_ids: Candidate-touched question IDs.

    Returns:
        list[str]: Overlapping hot-set prefixes.
    """

    hits: list[str] = []
    for prefix in HOT_SET_PREFIXES:
        if any(qid.startswith(prefix) for qid in question_ids):
            hits.append(prefix)
    return hits


def assess_resurrection_candidate(
    evidence: ResurrectionEvidence,
    *,
    public_anchor_total: float,
) -> ResurrectionAssessment:
    """Score and classify a resurrection candidate.

    Args:
        evidence: Candidate evidence bundle.
        public_anchor_total: Last strong public total anchor.

    Returns:
        ResurrectionAssessment: Deterministic ranking assessment.
    """

    blockers = parse_blocker_terms(evidence.no_submit_reason)
    confounded_reasons: list[str] = []
    if not evidence.tracked_artifacts_ok:
        confounded_reasons.append("missing_tracked_candidate_artifacts")
    if evidence.lineage_confidence == "low":
        confounded_reasons.append("low_lineage_confidence")
    if evidence.answer_changed_count > 0:
        confounded_reasons.append("answer_drift_present")
    if evidence.control:
        confounded_reasons.append("control_row")

    conf_weight = confidence_weight(evidence.lineage_confidence)
    strict_gain = max(0.0, (evidence.strict_total_estimate or 0.0) - public_anchor_total)
    platform_gain = max(0.0, (evidence.platform_like_total_estimate or 0.0) - public_anchor_total)
    paranoid_floor = (evidence.paranoid_total_estimate or 0.0) - public_anchor_total
    hot_overlap = len(evidence.hot_set_touched_prefixes)

    signal = (
        (evidence.hidden_g_trusted_delta * 8.0)
        + (strict_gain * 5.0)
        + (platform_gain * 6.0)
        + (hot_overlap * 0.01)
    )
    drift_penalty = (evidence.answer_changed_count * 0.02) + min(evidence.page_changed_count, 100) * 0.001
    resurrection_score = max(0.0, (signal * conf_weight) - drift_penalty)

    over_penalized_score = max(
        0.0,
        ((evidence.hidden_g_trusted_delta * 10.0) + (platform_gain * 4.0) + (strict_gain * 2.0))
        * conf_weight
        - (evidence.answer_changed_count * 0.03),
    )
    toxicity_score = 0.0
    if evidence.hidden_g_trusted_delta <= 0.0:
        toxicity_score += 0.5
    if evidence.answer_changed_count > 0:
        toxicity_score += 0.35
    if any(term in blockers for term in _KNOWN_TOXIC_BLOCKERS):
        toxicity_score += 0.3
    if paranoid_floor < -0.02:
        toxicity_score += 0.25

    if evidence.control:
        status = "control"
    elif confounded_reasons and evidence.hidden_g_trusted_delta > 0.0:
        status = "confounded_but_interesting"
    elif (evidence.hidden_g_trusted_delta <= 0.0 and (toxicity_score >= 0.5 or not evidence.tracked_artifacts_ok)) or (
        toxicity_score >= 0.8
    ):
        status = "toxic_even_if_locally_shiny"
    elif over_penalized_score >= 0.25 and evidence.hidden_g_trusted_delta > 0.0:
        status = "probably_over_penalized"
    else:
        status = "replay_shortlist"

    return ResurrectionAssessment(
        label=evidence.label,
        status=status,
        resurrection_score=round(resurrection_score, 6),
        over_penalized_score=round(over_penalized_score, 6),
        toxicity_score=round(toxicity_score, 6),
        confidence_weight=conf_weight,
        blocker_terms=blockers,
        confounded_reasons=confounded_reasons,
    )
