"""Classification rules for closed-world failure cartography."""

from __future__ import annotations

import json
import re
from typing import cast

from shafi.eval.failure_cartography_models import (
    DriftRecord,
    FailureTaxonomy,
    ReviewedGoldenCase,
    RunObservation,
)

_WORD_RE = re.compile(r"\w+")


def classify_miss(golden: ReviewedGoldenCase, observation: RunObservation) -> list[FailureTaxonomy]:
    """Classify one reviewed miss into the failure taxonomy.

    Args:
        golden: Reviewed golden label.
        observation: Run observation.

    Returns:
        list[FailureTaxonomy]: Multi-label failure classification.
    """

    failure_types: list[FailureTaxonomy] = []
    question = golden.question or observation.question
    if _is_field_question(question):
        failure_types.append(FailureTaxonomy.FIELD_MISS)
    if _is_provision_question(question):
        failure_types.append(FailureTaxonomy.PROVISION_MISS)
    if _is_compare_question(question):
        failure_types.append(FailureTaxonomy.COMPARE_MISS)
    if _is_temporal_question(question):
        failure_types.append(FailureTaxonomy.TEMPORAL_MISS)
    if _is_alias_question(question):
        failure_types.append(FailureTaxonomy.ALIAS_MISS)

    gold_pages = set(golden.golden_page_ids)
    used_pages = set(observation.used_page_ids)
    retrieved_pages = set(observation.retrieved_page_ids)
    answer_correct = answers_match(
        answer_type=golden.answer_type,
        predicted=observation.predicted_answer,
        golden=golden.golden_answer,
    )
    has_used_gold = bool(gold_pages & used_pages)
    has_retrieved_gold = bool(gold_pages & retrieved_pages)

    if gold_pages and not has_retrieved_gold:
        failure_types.append(FailureTaxonomy.RETRIEVAL_MISS)
    elif gold_pages and has_retrieved_gold and not has_used_gold:
        failure_types.append(FailureTaxonomy.RANKING_MISS)

    if answer_correct and gold_pages and not has_used_gold:
        failure_types.append(FailureTaxonomy.GROUNDING_MISS)
    if not answer_correct and (has_used_gold or not gold_pages):
        failure_types.append(FailureTaxonomy.GENERATION_MISS)

    if not failure_types:
        failure_types.append(FailureTaxonomy.GENERATION_MISS)
    return list(dict.fromkeys(failure_types))


def compute_drift(observations: list[RunObservation]) -> DriftRecord:
    """Compute answer and page drift across run observations.

    Args:
        observations: Per-run observations for one question.

    Returns:
        DriftRecord: Drift summary.
    """

    answer_variants = list(dict.fromkeys(_normalize_for_drift(obs.predicted_answer) for obs in observations))
    page_variants_raw = list(
        dict.fromkeys(json.dumps(sorted(obs.used_page_ids), ensure_ascii=False) for obs in observations)
    )
    page_variants = [cast("list[str]", json.loads(item)) for item in page_variants_raw]
    return DriftRecord(
        answer_variants=answer_variants,
        page_projection_variants=page_variants,
        answer_drift_count=max(0, len(answer_variants) - 1),
        page_drift_count=max(0, len(page_variants) - 1),
    )


def answers_match(*, answer_type: str, predicted: str, golden: str) -> bool:
    """Judge whether a predicted answer matches the reviewed answer.

    Args:
        answer_type: Declared answer type.
        predicted: Predicted answer text.
        golden: Reviewed answer text.

    Returns:
        bool: Whether the answers match under a normalized comparison.
    """

    kind = answer_type.strip().lower()
    pred = _normalize_for_match(predicted)
    gold = _normalize_for_match(golden)
    if not gold:
        return False
    if kind == "boolean":
        return _canonical_boolean(pred) == _canonical_boolean(gold)
    if kind == "names":
        return sorted(_split_names(predicted)) == sorted(_split_names(golden))
    if kind in {"name", "number", "date"}:
        return pred == gold
    return pred == gold or (len(gold) >= 16 and gold in pred) or (len(pred) >= 16 and pred in gold)


def infer_doc_family(question: str, document_ids: list[str]) -> str:
    """Infer a coarse document family for aggregation.

    Args:
        question: Question text.
        document_ids: Distinct document ids involved in gold/predicted pages.

    Returns:
        str: Coarse family label.
    """

    lowered = question.lower()
    if any(term in lowered for term in ("article", "section", "schedule", "clause", "provision")):
        return "law_provision"
    if any(term in lowered for term in ("commence", "effective", "amend", "supersed", "repeal")):
        return "temporal_applicability"
    if any(term in lowered for term in ("common", "compare", "both", "same party", "same judge")):
        return "compare_join"
    if any(term in lowered for term in ("claimant", "defendant", "party", "judge", "issued by", "law no")):
        return "field_lookup"
    if len(document_ids) > 1:
        return "multi_document"
    return "generic_legal_document"


def page_docs(page_ids: list[str]) -> list[str]:
    """Extract document ids from page ids.

    Args:
        page_ids: Page identifiers.

    Returns:
        list[str]: Document ids.
    """

    docs: list[str] = []
    for page_id in page_ids:
        docs.append(page_id.rsplit("_", 1)[0] if "_" in page_id else page_id)
    return docs


def _normalize_for_match(value: str) -> str:
    return " ".join(_WORD_RE.findall(str(value).lower()))


def _normalize_for_drift(value: str) -> str:
    return _normalize_for_match(value) or str(value).strip().lower()


def _canonical_boolean(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"yes", "true", "1"}:
        return "true"
    if lowered in {"no", "false", "0"}:
        return "false"
    return lowered


def _split_names(value: str) -> list[str]:
    parts = re.split(r"[,\n;]+", value)
    return [_normalize_for_match(part) for part in parts if _normalize_for_match(part)]


def _is_field_question(question: str) -> bool:
    lowered = question.lower()
    return any(
        term in lowered
        for term in ("who is", "who was", "date", "issued by", "law no", "claimant", "judge", "party", "amount")
    )


def _is_provision_question(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in ("article", "section", "schedule", "clause", "provision"))


def _is_compare_question(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in ("common", "compare", "both", "same party", "same judge", "same claimant"))


def _is_temporal_question(question: str) -> bool:
    lowered = question.lower()
    return any(
        term in lowered for term in ("effective", "commence", "commencement", "amend", "supersed", "repeal", "replace")
    )


def _is_alias_question(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in ("short title", "also known", "abbreviat", "full title", "called"))
