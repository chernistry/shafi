"""Helper functions for offline grounding-router training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_challenge.ml.external_grounding_data import NormalizedExternalRow
from rag_challenge.ml.training_scaffold import build_router_text, derive_page_budget_target

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rag_challenge.ml.grounding_dataset import GroundingMlRow


@dataclass(frozen=True)
class RouterTrainingExample:
    """One offline grounding-router training example.

    Args:
        sample_id: Stable sample identifier.
        text: Model input text.
        scope_target: Optional scope target.
        page_budget_target: Optional page-budget target.
        role_targets: Internal page-role targets.
        sample_weight: Training sample weight.
        source: Example provenance label.
    """

    sample_id: str
    text: str
    scope_target: str | None
    page_budget_target: int | None
    role_targets: list[str]
    sample_weight: float
    source: str


def build_internal_router_examples(rows: Sequence[GroundingMlRow]) -> list[RouterTrainingExample]:
    """Convert exported internal rows into router training examples.

    Args:
        rows: Exported internal grounding rows.

    Returns:
        Internal router examples.
    """
    return [
        RouterTrainingExample(
            sample_id=row.question_id,
            text=build_router_text(row),
            scope_target=row.scope_mode,
            page_budget_target=derive_page_budget_target(row),
            role_targets=list(row.target_page_roles),
            sample_weight=1.0,
            source="internal",
        )
        for row in rows
    ]


def load_external_normalized_rows(path: Path) -> list[NormalizedExternalRow]:
    """Load normalized external rows from JSONL.

    Args:
        path: Path to `normalized_rows.jsonl`.

    Returns:
        Parsed external rows.
    """
    rows: list[NormalizedExternalRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(NormalizedExternalRow.model_validate_json(line))
    return rows


def build_external_router_examples(
    rows: Sequence[NormalizedExternalRow],
    *,
    sample_weight: float,
) -> list[RouterTrainingExample]:
    """Map normalized external rows into internal-style router examples.

    Args:
        rows: Normalized external rows.
        sample_weight: Low-confidence training weight for external supervision.

    Returns:
        External router examples with coarse internal target mappings.
    """
    examples: list[RouterTrainingExample] = []
    for row in rows:
        scope_target = _map_external_scope_label(row)
        page_budget_target = _map_external_page_budget(row)
        role_targets = _map_external_role_targets(row)
        if scope_target is None and page_budget_target is None and not role_targets:
            continue
        examples.append(
            RouterTrainingExample(
                sample_id=f"{row.source_dataset}:{row.sample_id}",
                text=_build_external_router_text(row),
                scope_target=scope_target,
                page_budget_target=page_budget_target,
                role_targets=role_targets,
                sample_weight=sample_weight,
                source=row.source_dataset,
            )
        )
    return examples


def _build_external_router_text(row: NormalizedExternalRow) -> str:
    """Compose model input text for an external supervision row.

    Args:
        row: Normalized external row.

    Returns:
        Compact training text.
    """
    metadata = json.loads(row.metadata_json)
    return (
        f"question: {row.question}\n"
        f"label_type: {row.label_type}\n"
        f"role_label: {row.role_label}\n"
        f"scope_label: {row.scope_label}\n"
        f"support_label: {row.support_label}\n"
        f"source_dataset: {row.source_dataset}\n"
        f"text: {row.text[:400]}\n"
        f"metadata: {json.dumps(metadata, ensure_ascii=True, sort_keys=True)[:240]}"
    ).strip()


def _map_external_scope_label(row: NormalizedExternalRow) -> str | None:
    """Map an external scope label into the internal router scope family.

    Args:
        row: Normalized external row.

    Returns:
        Internal scope target, or `None` when no safe mapping exists.
    """
    if row.scope_label == "single_field_single_doc":
        return "single_field_single_doc"
    if row.scope_label == "pair_entailment":
        return "single_field_single_doc"
    return None


def _map_external_page_budget(row: NormalizedExternalRow) -> int | None:
    """Map an external row into a coarse page-budget target.

    Args:
        row: Normalized external row.

    Returns:
        Coarse page-budget target, or `None`.
    """
    if row.source_dataset == "obliqa":
        metadata = json.loads(row.metadata_json)
        passage_count = int(metadata.get("passage_count", 1))
        return 2 if passage_count > 1 else 1
    if row.source_dataset in {"cuad", "contractnli", "ledgar"}:
        return 1
    return None


def _map_external_role_targets(row: NormalizedExternalRow) -> list[str]:
    """Map external supervision into internal page-role labels.

    Args:
        row: Normalized external row.

    Returns:
        Zero or more internal page-role targets.
    """
    if row.source_dataset == "obliqa":
        return ["article_clause"]
    if row.source_dataset == "contractnli":
        return []

    role = row.role_label
    if not role:
        return []

    mapped: list[str] = []
    if any(token in role for token in ("document_name", "title", "agreement_name", "contract_name")):
        mapped.append("title_cover")
    if any(token in role for token in ("party", "parties")):
        mapped.append("caption")
    if any(token in role for token in ("cost", "costs", "expense", "expenses", "fee", "fees", "payment", "tax")):
        mapped.append("costs_block")
    if any(token in role for token in ("schedule", "appendix", "annex", "exhibit")):
        mapped.append("schedule_table")
    if any(
        token in role
        for token in (
            "date",
            "authority",
            "approval",
            "authorization",
            "execution",
            "effective",
        )
    ):
        mapped.append("issued_by_block")
    if any(
        token in role
        for token in (
            "termination",
            "remedies",
            "specific_performance",
            "arbitration",
            "waiver",
            "survival",
        )
    ):
        mapped.append("operative_order")
    if any(
        token in role
        for token in (
            "law",
            "jurisdiction",
            "term",
            "definition",
            "clause",
            "agreement",
            "obligation",
            "confidential",
            "indemn",
        )
    ):
        mapped.append("article_clause")

    if not mapped and row.label_type == "role_label":
        mapped.append("article_clause")
    return sorted(set(mapped))
