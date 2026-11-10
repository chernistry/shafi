"""Offline retrieval-utility predictor primitives.

This module defines a small, typed bundle-sufficiency predictor that can be
trained and benchmarked offline without changing the live retrieval or answer
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping


class EscalationTarget(StrEnum):
    """Possible escalation targets for a weak retrieval bundle."""

    NONE = "none"
    STRUCTURED_DB = "structured_db"
    TEMPORAL_ENGINE = "temporal_engine"
    COMPARE_ENGINE = "compare_engine"
    BROADER_RETRIEVAL = "broader_retrieval"


class BundleSnapshot(BaseModel):
    """Typed snapshot of a retrieved bundle and its surrounding signals.

    Args:
        question_id: Stable question identifier.
        question: Original question text.
        answer_type: Competition answer type.
        retrieved_page_count: Number of retrieved pages.
        context_page_count: Number of pages admitted to context.
        cited_page_count: Number of cited pages.
        used_page_count: Number of used pages.
        retrieved_chunk_count: Number of retrieved chunks.
        context_chunk_count: Number of context chunks.
        cited_chunk_count: Number of cited chunks.
        doc_ref_count: Number of document references supplied with the query.
        bridge_hit_count: Count of bridge-fact hits, if available.
        entity_confidence: Confidence score for entity resolution, if available.
        context_budget_tokens: Token budget assigned to the context window.
        compare_signal: Whether the question looks compare-like.
        temporal_signal: Whether the question looks temporal/applicability-like.
        field_signal: Whether the question looks like a direct field lookup.
    """

    model_config = ConfigDict(extra="ignore")

    question_id: str = ""
    question: str
    answer_type: str
    retrieved_page_count: int = 0
    context_page_count: int = 0
    cited_page_count: int = 0
    used_page_count: int = 0
    retrieved_chunk_count: int = 0
    context_chunk_count: int = 0
    cited_chunk_count: int = 0
    doc_ref_count: int = 0
    bridge_hit_count: int = 0
    entity_confidence: float = 0.0
    context_budget_tokens: int = 0
    compare_signal: bool = False
    temporal_signal: bool = False
    field_signal: bool = False

    @classmethod
    def from_raw_result(cls, row: Mapping[str, Any]) -> BundleSnapshot:
        """Build a bundle snapshot from a raw-results artifact row.

        Args:
            row: One raw-results row from a benchmark artifact.

        Returns:
            Bundle snapshot with pre-generation bundle signals.
        """
        case = cast("Mapping[str, Any]", row.get("case", {}))
        telemetry = cast("Mapping[str, Any]", row.get("telemetry", {}))
        question = str(case.get("question", ""))
        answer_type = str(case.get("answer_type", "free_text")).strip().lower()
        context_budget_tokens = int(telemetry.get("context_budget_tokens", 0) or 0)
        return cls(
            question_id=str(case.get("case_id") or telemetry.get("question_id") or telemetry.get("request_id") or ""),
            question=question,
            answer_type=answer_type,
            retrieved_page_count=len(_as_list(telemetry.get("retrieved_page_ids"))),
            context_page_count=len(_as_list(telemetry.get("context_page_ids"))),
            cited_page_count=len(_as_list(telemetry.get("cited_page_ids"))),
            used_page_count=len(_as_list(telemetry.get("used_page_ids"))),
            retrieved_chunk_count=len(_as_list(telemetry.get("retrieved_chunk_ids"))),
            context_chunk_count=len(_as_list(telemetry.get("context_chunk_ids"))),
            cited_chunk_count=len(_as_list(telemetry.get("cited_chunk_ids"))),
            doc_ref_count=len(_as_list(telemetry.get("doc_refs"))),
            bridge_hit_count=int(telemetry.get("bridge_hit_count", 0) or 0),
            entity_confidence=float(telemetry.get("grounding_relevance_verifier_confidence", 0.0) or 0.0),
            context_budget_tokens=context_budget_tokens,
            compare_signal=_contains_any(question, ("compare", "difference", "common", "both", "versus", "vs")),
            temporal_signal=_contains_any(
                question,
                ("date", "effective", "commencement", "amend", "supersed", "in force", "before", "after"),
            ),
            field_signal=_contains_any(question, ("who", "what", "which", "when", "name", "number", "amount", "date")),
        )


class BundlePrediction(BaseModel):
    """Prediction emitted by the retrieval-utility model.

    Args:
        bundle_sufficiency: Probability that the bundle is sufficient.
        escalation_target: Suggested next route when the bundle is weak.
        confidence_calibration: Confidence proxy derived from the probability.
        feature_vector: Feature dictionary used for the prediction.
        raw_probability: Raw model probability before thresholding.
        reason: Short human-readable explanation.
    """

    model_config = ConfigDict(extra="ignore")

    bundle_sufficiency: float
    escalation_target: EscalationTarget
    confidence_calibration: float
    feature_vector: dict[str, int | float | bool | str] = Field(default_factory=dict)
    raw_probability: float = 0.0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class UtilityPredictorArtifact:
    """Serialized retrieval-utility model bundle."""

    vectorizer: Any
    model: Any
    threshold: float
    feature_policy: str


class RetrievalUtilityPredictor:
    """Predict bundle sufficiency and route escalation recommendations."""

    def __init__(
        self,
        *,
        artifact: UtilityPredictorArtifact | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the predictor.

        Args:
            artifact: Optional fitted model bundle.
            threshold: Sufficiency threshold used for routing.
        """
        self._artifact = artifact
        self._threshold = threshold

    def predict(self, snapshot: BundleSnapshot) -> BundlePrediction:
        """Predict bundle sufficiency and escalation target.

        Args:
            snapshot: Typed retrieval bundle snapshot.

        Returns:
            Prediction with probability, route suggestion, and features.
        """
        features = self.extract_features(snapshot)
        if self._artifact is None:
            probability = self._heuristic_probability(snapshot, features)
        else:
            matrix = self._artifact.vectorizer.transform([features])
            probability = float(self._artifact.model.predict_proba(matrix)[0, 1])

        escalation_target = self._route_from_snapshot(snapshot, probability)
        confidence = max(probability, 1.0 - probability)
        reason = self._build_reason(snapshot, probability, escalation_target)
        return BundlePrediction(
            bundle_sufficiency=probability,
            escalation_target=escalation_target,
            confidence_calibration=confidence,
            feature_vector=features,
            raw_probability=probability,
            reason=reason,
        )

    def extract_features(self, snapshot: BundleSnapshot) -> dict[str, int | float | bool | str]:
        """Convert a bundle snapshot into model features.

        Args:
            snapshot: Input bundle snapshot.

        Returns:
            Sparse feature dictionary suitable for a DictVectorizer.
        """
        retrieved = max(snapshot.retrieved_page_count, 1)
        context = max(snapshot.context_page_count, 1)
        cited = max(snapshot.cited_page_count, 1)
        return {
            "answer_type": snapshot.answer_type,
            "question_len": len(snapshot.question),
            "retrieved_page_count": snapshot.retrieved_page_count,
            "context_page_count": snapshot.context_page_count,
            "cited_page_count": snapshot.cited_page_count,
            "used_page_count": snapshot.used_page_count,
            "retrieved_chunk_count": snapshot.retrieved_chunk_count,
            "context_chunk_count": snapshot.context_chunk_count,
            "cited_chunk_count": snapshot.cited_chunk_count,
            "doc_ref_count": snapshot.doc_ref_count,
            "bridge_hit_count": snapshot.bridge_hit_count,
            "entity_confidence": snapshot.entity_confidence,
            "context_budget_tokens": snapshot.context_budget_tokens,
            "compare_signal": snapshot.compare_signal,
            "temporal_signal": snapshot.temporal_signal,
            "field_signal": snapshot.field_signal,
            "retrieval_density": snapshot.context_page_count / float(retrieved),
            "citation_density": snapshot.cited_page_count / float(retrieved),
            "usage_density": snapshot.used_page_count / float(retrieved),
            "context_ratio": snapshot.context_page_count / float(context),
            "citation_ratio": snapshot.cited_page_count / float(cited),
            "multi_doc_signal": snapshot.doc_ref_count > 1,
            "strict_answer_type": snapshot.answer_type in {"boolean", "date", "name", "names", "number"},
            "free_text_answer_type": snapshot.answer_type == "free_text",
            "small_bundle_signal": snapshot.context_page_count <= 2,
        }

    def should_escalate(self, prediction: BundlePrediction) -> bool:
        """Return whether the bundle should escalate away from the base route.

        Args:
            prediction: Prediction returned by :meth:`predict`.

        Returns:
            True when the bundle is weak enough to escalate.
        """
        return prediction.bundle_sufficiency < self._threshold

    def _heuristic_probability(
        self,
        snapshot: BundleSnapshot,
        features: Mapping[str, int | float | bool | str],
    ) -> float:
        """Fallback probability when no fitted model is available.

        Args:
            snapshot: Input bundle snapshot.
            features: Extracted feature dictionary.

        Returns:
            Deterministic sufficiency score.
        """
        score = 0.0
        score += 0.35 if bool(features.get("strict_answer_type")) else 0.0
        score += 0.20 if snapshot.cited_page_count > 0 else 0.0
        score += 0.15 if snapshot.used_page_count > 0 else 0.0
        score += 0.10 if snapshot.context_page_count >= 3 else 0.0
        score += 0.05 if snapshot.doc_ref_count > 0 else 0.0
        score += 0.05 if snapshot.compare_signal or snapshot.temporal_signal or snapshot.field_signal else 0.0
        return min(score, 0.99)

    def _route_from_snapshot(self, snapshot: BundleSnapshot, probability: float) -> EscalationTarget:
        """Select an escalation route from the snapshot and score.

        Args:
            snapshot: Input bundle snapshot.
            probability: Sufficiency probability.

        Returns:
            Escalation target.
        """
        if probability >= self._threshold:
            return EscalationTarget.NONE
        if snapshot.compare_signal:
            return EscalationTarget.COMPARE_ENGINE
        if snapshot.temporal_signal:
            return EscalationTarget.TEMPORAL_ENGINE
        if snapshot.field_signal or snapshot.answer_type in {"boolean", "date", "name", "names", "number"}:
            return EscalationTarget.STRUCTURED_DB
        return EscalationTarget.BROADER_RETRIEVAL

    def _build_reason(
        self,
        snapshot: BundleSnapshot,
        probability: float,
        escalation_target: EscalationTarget,
    ) -> str:
        """Build a short human-readable explanation.

        Args:
            snapshot: Input bundle snapshot.
            probability: Sufficiency probability.
            escalation_target: Selected route.

        Returns:
            Concise reason string.
        """
        if escalation_target is EscalationTarget.NONE:
            return f"bundle_sufficient={probability:.3f} with citation_density={snapshot.cited_page_count}/{max(snapshot.retrieved_page_count, 1)}"
        return (
            f"bundle_weak={probability:.3f}; "
            f"route={escalation_target.value}; "
            f"cited_pages={snapshot.cited_page_count}; "
            f"context_pages={snapshot.context_page_count}"
        )


def _as_list(value: object) -> list[Any]:
    """Normalize a possibly-null value to a list.

    Args:
        value: Candidate list-like value.

    Returns:
        Flat list, or an empty list when unavailable.
    """
    if isinstance(value, list):
        return cast("list[Any]", value)
    if isinstance(value, tuple):
        return list(cast("tuple[Any, ...]", value))
    return []


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    """Return whether any needle appears in text.

    Args:
        text: Query text.
        needles: Candidate substrings.

    Returns:
        True when at least one needle is present.
    """
    lowered = text.lower()
    return any(needle in lowered for needle in needles)
