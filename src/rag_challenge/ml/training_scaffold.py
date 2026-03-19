"""Shared training-scaffold helpers for grounding-sidecar ML tickets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rag_challenge.ml.grounding_dataset import GroundingMlRow, PageCandidateRecord

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

LabelMode = Literal[
    "reviewed_high_confidence",
    "reviewed_weighted",
    "reviewed_only",
    "soft_and_reviewed",
    "all",
]


@dataclass(frozen=True)
class RouterDataset:
    """Prepared router training dataset."""

    question_ids: list[str]
    texts: list[str]
    scope_targets: list[str]
    page_budget_targets: list[int]
    role_targets: list[list[str]]


@dataclass(frozen=True)
class PageTrainingExample:
    """One page-level supervised example for the offline scorer."""

    question_id: str
    page_id: str
    features: dict[str, str | int | float | bool]
    label: int
    sample_weight: float
    supervision_source: str


def load_grounding_rows(path: Path) -> list[GroundingMlRow]:
    """Load grounding ML rows from JSONL.

    Args:
        path: JSONL file path.

    Returns:
        Parsed grounding rows.
    """
    rows: list[GroundingMlRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(GroundingMlRow.model_validate_json(line))
    return rows


def deterministic_subset(rows: Sequence[GroundingMlRow], *, limit: int | None, seed: int) -> list[GroundingMlRow]:
    """Return a deterministic subset of rows.

    Args:
        rows: Source rows.
        limit: Optional maximum count.
        seed: Deterministic selection seed.

    Returns:
        Ordered subset of rows.
    """
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row.question_id}".encode()).hexdigest(),
            row.question_id,
        ),
    )
    if limit is None or limit <= 0 or limit >= len(ordered):
        return list(ordered)
    return list(ordered[:limit])


def build_router_dataset(rows: Sequence[GroundingMlRow]) -> RouterDataset:
    """Build router training targets from export rows.

    Args:
        rows: Grounding export rows.

    Returns:
        RouterDataset for scope, page-budget, and role prediction.
    """
    question_ids: list[str] = []
    texts: list[str] = []
    scope_targets: list[str] = []
    page_budget_targets: list[int] = []
    role_targets: list[list[str]] = []
    for row in rows:
        question_ids.append(row.question_id)
        texts.append(build_router_text(row))
        scope_targets.append(row.scope_mode)
        page_budget_targets.append(derive_page_budget_target(row))
        role_targets.append(list(row.target_page_roles))
    return RouterDataset(
        question_ids=question_ids,
        texts=texts,
        scope_targets=scope_targets,
        page_budget_targets=page_budget_targets,
        role_targets=role_targets,
    )


def build_router_text(row: GroundingMlRow) -> str:
    """Compose router input text from one export row.

    Args:
        row: Grounding ML row.

    Returns:
        Compact text prompt for the offline router model.
    """
    anchors = " ; ".join(row.hard_anchor_strings)
    roles = " ; ".join(row.target_page_roles)
    return (
        f"question: {row.question}\n"
        f"answer_type: {row.answer_type}\n"
        f"anchors: {anchors}\n"
        f"roles: {roles}\n"
        f"doc_ref_count: {row.support_fact_features.doc_ref_count}"
    ).strip()


def derive_page_budget_target(row: GroundingMlRow) -> int:
    """Derive a page-budget target from exported selections.

    Args:
        row: Grounding ML row.

    Returns:
        Integer page-budget target for offline router training.
    """
    if row.scope_mode == "negative_unanswerable":
        return 0
    preferred = row.sidecar_selected_pages or row.label_page_ids or row.legacy_selected_pages
    if not preferred:
        return 1
    return max(1, len(preferred))


def build_page_training_examples(
    rows: Sequence[GroundingMlRow],
    *,
    label_mode: LabelMode = "all",
) -> list[PageTrainingExample]:
    """Build page-level supervised examples from export rows.

    Args:
        rows: Grounding export rows.
        label_mode: Controls which label provenance types are trainable.

    Returns:
        Flat list of page training examples.
    """
    examples: list[PageTrainingExample] = []
    for row in rows:
        positive_pages, supervision_source, sample_weight = choose_page_supervision(row, label_mode=label_mode)
        if not positive_pages:
            continue
        doc_rank_map = {candidate.doc_id: index + 1 for index, candidate in enumerate(row.doc_candidates)}
        legacy_selected_page_ids = set(row.legacy_selected_pages)
        sidecar_selected_page_ids = set(row.sidecar_selected_pages)
        legacy_selected_docs = {
            candidate.doc_id for candidate in row.page_candidates if candidate.page_id in legacy_selected_page_ids
        }
        sidecar_selected_docs = {
            candidate.doc_id for candidate in row.page_candidates if candidate.page_id in sidecar_selected_page_ids
        }
        for page in row.page_candidates:
            examples.append(
                PageTrainingExample(
                    question_id=row.question_id,
                    page_id=page.page_id,
                    features=build_page_feature_dict(
                        row,
                        page,
                        doc_rank_map=doc_rank_map,
                        legacy_selected_docs=legacy_selected_docs,
                        sidecar_selected_docs=sidecar_selected_docs,
                    ),
                    label=1 if page.page_id in positive_pages else 0,
                    sample_weight=sample_weight,
                    supervision_source=supervision_source,
                )
            )
    return examples


def choose_page_supervision(
    row: GroundingMlRow,
    *,
    label_mode: LabelMode,
) -> tuple[set[str], str, float]:
    """Select the supervision source for a row's page targets.

    Args:
        row: Grounding ML row.
        label_mode: Controls which label-provenance rows are allowed.

    Returns:
        Tuple of positive page IDs, supervision source label, and sample weight.
    """
    reviewed_weight = internal_row_sample_weight(row, label_mode=label_mode)
    if row.label_source == "reviewed" and row.label_page_ids:
        if reviewed_weight > 0.0:
            return set(row.label_page_ids), "reviewed", 3.0 * reviewed_weight
        if label_mode in {"reviewed_high_confidence", "reviewed_weighted", "reviewed_only"}:
            return set(), "none", 0.0
    if row.label_source == "soft_ai_gold" and row.label_page_ids and label_mode in {"soft_and_reviewed", "all"}:
        return set(row.label_page_ids), "soft_ai_gold", 1.5
    if label_mode in {"reviewed_high_confidence", "reviewed_weighted", "reviewed_only"}:
        return set(), "none", 0.0
    if row.sidecar_selected_pages:
        return set(row.sidecar_selected_pages), "sidecar_selected", 0.35
    if row.legacy_selected_pages:
        return set(row.legacy_selected_pages), "legacy_selected", 0.2
    return set(), "none", 0.0


def internal_row_sample_weight(
    row: GroundingMlRow,
    *,
    label_mode: LabelMode,
) -> float:
    """Return the internal-training sample weight for one row.

    Args:
        row: Grounding ML row.
        label_mode: Reviewed-aware label selection mode.

    Returns:
        Non-negative sample weight. Zero means the row should be excluded from
        supervised training for the chosen mode.
    """
    if row.label_source == "reviewed":
        confidence = row.label_confidence.strip().lower()
        default_weight = row.label_weight if row.label_weight > 0.0 else _default_reviewed_weight(confidence)
        if label_mode == "reviewed_high_confidence":
            return 1.0 if confidence == "high" else 0.0
        if label_mode == "reviewed_weighted":
            return default_weight
        if label_mode == "reviewed_only":
            return 1.0
        if label_mode in {"soft_and_reviewed", "all"}:
            return default_weight
        return 0.0
    if row.label_source == "soft_ai_gold" and label_mode in {"soft_and_reviewed", "all"}:
        return 1.0
    return 0.0


def _default_reviewed_weight(confidence: str) -> float:
    """Return the fallback reviewed weight for a confidence tier.

    Args:
        confidence: Normalized reviewed confidence string.

    Returns:
        Default row weight when the export omitted an explicit numeric weight.
    """
    if confidence == "high":
        return 1.0
    if confidence == "medium":
        return 0.5
    if confidence == "low":
        return 0.0
    return 1.0


def build_page_feature_dict(
    row: GroundingMlRow,
    page: PageCandidateRecord,
    *,
    doc_rank_map: dict[str, int],
    legacy_selected_docs: set[str],
    sidecar_selected_docs: set[str],
) -> dict[str, str | int | float | bool]:
    """Build one feature dictionary for offline page scoring.

    Args:
        row: Parent grounding ML row.
        page: Candidate page record.
        doc_rank_map: 1-based document-rank mapping within the row.
        legacy_selected_docs: Docs already selected by legacy grounding.
        sidecar_selected_docs: Docs already selected by sidecar grounding.

    Returns:
        Feature dictionary for DictVectorizer.
    """
    answer_text = str(row.golden_answer) if row.golden_answer is not None else ""
    snippet = page.snippet_excerpt.casefold()
    return {
        "scope_mode": row.scope_mode,
        "answer_type": row.answer_type,
        "label_source": row.label_source,
        "doc_rank": doc_rank_map.get(page.doc_id, 0),
        "page_num": page.page_num,
        "is_first_page": page.page_num == 1,
        "doc_selected_by_legacy": page.doc_id in legacy_selected_docs,
        "doc_selected_by_sidecar": page.doc_id in sidecar_selected_docs,
        "doc_candidate_count": len(row.doc_candidates),
        "page_candidate_count": len(row.page_candidates),
        "anchor_hit_count": len(page.anchor_hits),
        "has_anchor_hit": bool(page.anchor_hits),
        "answer_in_snippet": bool(answer_text and answer_text.casefold() in snippet),
        "requires_all_docs_in_case": row.support_fact_features.requires_all_docs_in_case,
        "should_force_empty_grounding_on_null": row.support_fact_features.should_force_empty_grounding_on_null,
        "explicit_anchor_count": row.support_fact_features.explicit_anchor_count,
        "target_page_roles_count": row.support_fact_features.target_page_roles_count,
        "doc_ref_count": row.support_fact_features.doc_ref_count,
        "legacy_retrieved_page_count": row.page_retrieval_features.legacy_retrieved_page_count,
        "legacy_context_page_count": row.page_retrieval_features.legacy_context_page_count,
        "legacy_cited_page_count": row.page_retrieval_features.legacy_cited_page_count,
        "sidecar_retrieved_page_count": row.page_retrieval_features.sidecar_retrieved_page_count,
        "sidecar_context_page_count": row.page_retrieval_features.sidecar_context_page_count,
        "sidecar_cited_page_count": row.page_retrieval_features.sidecar_cited_page_count,
        "legacy_sidecar_used_overlap_count": row.page_retrieval_features.legacy_sidecar_used_overlap_count,
        "targets_title_cover": "title_cover" in row.target_page_roles,
        "targets_caption": "caption" in row.target_page_roles,
        "targets_issued_by_block": "issued_by_block" in row.target_page_roles,
        "targets_operative_order": "operative_order" in row.target_page_roles,
        "targets_costs_block": "costs_block" in row.target_page_roles,
        "targets_article_clause": "article_clause" in row.target_page_roles,
        "targets_schedule_table": "schedule_table" in row.target_page_roles,
        "legacy_retrieved_rank": page.legacy_retrieved_rank or 0,
        "legacy_context_rank": page.legacy_context_rank or 0,
        "legacy_cited_rank": page.legacy_cited_rank or 0,
        "sidecar_retrieved_rank": page.sidecar_retrieved_rank or 0,
        "sidecar_context_rank": page.sidecar_context_rank or 0,
        "sidecar_cited_rank": page.sidecar_cited_rank or 0,
        "from_legacy_retrieved": "legacy_retrieved" in page.candidate_sources,
        "from_legacy_context": "legacy_context" in page.candidate_sources,
        "from_legacy_cited": "legacy_cited" in page.candidate_sources,
        "from_legacy_used": "legacy_used" in page.candidate_sources,
        "from_sidecar_retrieved": "sidecar_retrieved" in page.candidate_sources,
        "from_sidecar_context": "sidecar_context" in page.candidate_sources,
        "from_sidecar_cited": "sidecar_cited" in page.candidate_sources,
        "from_sidecar_used": "sidecar_used" in page.candidate_sources,
    }


def group_page_examples(examples: Sequence[PageTrainingExample]) -> dict[str, list[PageTrainingExample]]:
    """Group page examples by question ID.

    Args:
        examples: Flat example list.

    Returns:
        Mapping from question ID to grouped examples.
    """
    grouped: dict[str, list[PageTrainingExample]] = {}
    for example in examples:
        grouped.setdefault(example.question_id, []).append(example)
    return grouped
