"""Models for closed-world failure cartography."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

JsonDict = dict[str, Any]


class FailureTaxonomy(StrEnum):
    """Failure categories used to cluster reviewed misses."""

    FIELD_MISS = "field_miss"
    PROVISION_MISS = "provision_miss"
    COMPARE_MISS = "compare_miss"
    TEMPORAL_MISS = "temporal_miss"
    ALIAS_MISS = "alias_miss"
    RETRIEVAL_MISS = "retrieval_miss"
    RANKING_MISS = "ranking_miss"
    GENERATION_MISS = "generation_miss"
    GROUNDING_MISS = "grounding_miss"


@dataclass(frozen=True)
class ReviewedGoldenCase:
    """Reviewed golden record for one labeled question.

    Args:
        question_id: Stable question identifier.
        question: Question text.
        answer_type: Declared answer type.
        golden_answer: Reviewed answer text.
        golden_page_ids: Reviewed page identifiers.
        trust_tier: Review trust tier.
        label_weight: Review label weight.
    """

    question_id: str
    question: str
    answer_type: str
    golden_answer: str
    golden_page_ids: list[str]
    trust_tier: str
    label_weight: float


@dataclass(frozen=True)
class RunObservation:
    """One run's prediction record for a reviewed question.

    Args:
        run_label: Human-readable run label.
        source_path: Origin artifact path.
        question_id: Stable question identifier.
        question: Question text.
        answer_type: Declared answer type.
        predicted_answer: Predicted answer text.
        used_page_ids: Final used page ids.
        retrieved_page_ids: Retrieved page ids.
    """

    run_label: str
    source_path: str
    question_id: str
    question: str
    answer_type: str
    predicted_answer: str
    used_page_ids: list[str]
    retrieved_page_ids: list[str]


@dataclass(frozen=True)
class RunFailureObservation:
    """Classified failure outcome for one run/question pair.

    Args:
        run_label: Historical run label.
        source_path: Origin artifact path.
        predicted_answer: Predicted answer text.
        used_page_ids: Final cited/used pages.
        retrieved_page_ids: Retrieved page ids.
        failure_types: Classified failure labels.
        answer_correct: Whether answer matched the reviewed answer.
    """

    run_label: str
    source_path: str
    predicted_answer: str
    used_page_ids: list[str]
    retrieved_page_ids: list[str]
    failure_types: list[str]
    answer_correct: bool


@dataclass(frozen=True)
class DriftRecord:
    """Drift summary across historical runs.

    Args:
        answer_variants: Distinct normalized answer variants.
        page_projection_variants: Distinct page projection variants.
        answer_drift_count: Number of non-baseline answer variants.
        page_drift_count: Number of non-baseline page variants.
    """

    answer_variants: list[str]
    page_projection_variants: list[list[str]]
    answer_drift_count: int
    page_drift_count: int


@dataclass(frozen=True)
class FailureRecord:
    """Aggregated failure ledger record for one reviewed question.

    Args:
        question_id: Stable question identifier.
        question: Question text.
        answer_type: Declared answer type.
        doc_family: Heuristic high-level document family.
        document_ids: Distinct document ids seen in gold/predictions.
        golden_answer: Reviewed answer text.
        golden_page_ids: Reviewed page ids.
        failure_types: Union of classified failure labels across runs.
        failure_type_counts: Per-label counts across runs.
        drift: Answer/page drift summary.
        run_observations: Per-run classified outcomes.
    """

    question_id: str
    question: str
    answer_type: str
    doc_family: str
    document_ids: list[str]
    golden_answer: str
    golden_page_ids: list[str]
    failure_types: list[str]
    failure_type_counts: dict[str, int]
    drift: DriftRecord
    run_observations: list[RunFailureObservation]


@dataclass(frozen=True)
class FailureLedger:
    """Machine-readable failure ledger plus aggregate summaries."""

    records: list[FailureRecord]
    summary: JsonDict

    def to_dict(self) -> JsonDict:
        """Serialize the ledger into JSON-compatible data.

        Returns:
            JsonDict: Ledger payload.
        """

        return {
            "summary": self.summary,
            "records": [asdict(record) for record in self.records],
        }
