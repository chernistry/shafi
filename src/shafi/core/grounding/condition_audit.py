"""Typed grounding-condition audit helpers.

These helpers keep family-specific support checks deterministic and auditable.
They are used both for portfolio scoring and for fail-closed fallback when a
challenger page set does not meet the minimum support contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from shafi.core.grounding.authority_priors import matches_field_family
from shafi.models.schemas import RetrievedPage, ScopeMode

_DATE_QUERY_RE = re.compile(r"\b(date|effective|commence|commencement|when)\b|date of issue", re.IGNORECASE)
_AUTHORITY_QUERY_RE = re.compile(r"\b(judge|registrar|authority|issued|issued by|issued-by)\b", re.IGNORECASE)
_CLAIM_QUERY_RE = re.compile(r"\bclaim no|claim number\b", re.IGNORECASE)
_LAW_NUMBER_RE = re.compile(r"\blaw no|law number|enactment notice\b", re.IGNORECASE)
_PARTY_QUERY_RE = re.compile(r"\b(claimant|party|parties|title|caption)\b", re.IGNORECASE)
_ARTICLE_QUERY_RE = re.compile(r"\b(article|section|schedule)\b", re.IGNORECASE)
_OUTCOME_QUERY_RE = re.compile(r"\b(costs|cost awarded|final ruling|outcome|ordered)\b", re.IGNORECASE)
_COMPARE_QUERY_RE = re.compile(r"\b(compare|common|both|same|between)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ConditionAuditResult:
    """Typed support audit for one candidate grounding page set.

    Args:
        success: Whether all required slots were covered.
        required_slots: Ordered required support slots for the query family.
        covered_slots: Ordered slots covered by the candidate pages.
        failed_slots: Ordered slots that remain uncovered.
        reasons: Human-readable audit notes.
        coverage_ratio: Covered-slot ratio in ``[0.0, 1.0]``.
    """

    success: bool
    required_slots: tuple[str, ...]
    covered_slots: tuple[str, ...]
    failed_slots: tuple[str, ...]
    reasons: tuple[str, ...]
    coverage_ratio: float


def audit_candidate_pages(
    *,
    query: str,
    answer_type: str,
    scope_mode: ScopeMode,
    pages: list[RetrievedPage],
) -> ConditionAuditResult:
    """Audit whether candidate pages satisfy typed support conditions.

    Args:
        query: Raw user question.
        answer_type: Normalized answer type.
        scope_mode: Sidecar scope mode for the query.
        pages: Candidate pages in current portfolio order.

    Returns:
        ConditionAuditResult: Deterministic support audit summary.
    """

    required_slots = required_condition_slots(
        query=query,
        answer_type=answer_type,
        scope_mode=scope_mode,
    )
    if not required_slots:
        return ConditionAuditResult(
            success=True,
            required_slots=(),
            covered_slots=(),
            failed_slots=(),
            reasons=("no_typed_slots_required",),
            coverage_ratio=1.0,
        )

    covered_slots = tuple(slot for slot in required_slots if _slot_is_covered(slot=slot, pages=pages))
    failed_slots = tuple(slot for slot in required_slots if slot not in covered_slots)
    coverage_ratio = len(covered_slots) / len(required_slots)
    material_reasons: list[str] = []
    if covered_slots:
        material_reasons.append(f"covered:{','.join(covered_slots)}")
    if failed_slots:
        material_reasons.append(f"missing:{','.join(failed_slots)}")
    if scope_mode is ScopeMode.COMPARE_PAIR:
        material_reasons.append(f"compare_docs:{len({page.doc_id for page in pages if page.doc_id})}")
    return ConditionAuditResult(
        success=not failed_slots,
        required_slots=required_slots,
        covered_slots=covered_slots,
        failed_slots=failed_slots,
        reasons=tuple(material_reasons) or ("typed_audit_complete",),
        coverage_ratio=coverage_ratio,
    )


def required_condition_slots(
    *,
    query: str,
    answer_type: str,
    scope_mode: ScopeMode,
) -> tuple[str, ...]:
    """Build ordered typed support slots for a query family.

    Args:
        query: Raw user question.
        answer_type: Normalized answer type.
        scope_mode: Sidecar scope mode for the query.

    Returns:
        tuple[str, ...]: Ordered unique support slots.
    """

    slots: list[str] = []
    query_text = str(query or "")
    if scope_mode is ScopeMode.COMPARE_PAIR or _COMPARE_QUERY_RE.search(query_text):
        slots.append("compare_docs")
    if _DATE_QUERY_RE.search(query_text):
        slots.append("date")
    if _AUTHORITY_QUERY_RE.search(query_text):
        slots.append("authority")
    if _CLAIM_QUERY_RE.search(query_text):
        slots.append("claim")
    if _LAW_NUMBER_RE.search(query_text):
        slots.append("law_number")
    if _PARTY_QUERY_RE.search(query_text):
        slots.append("party_title")
    if _ARTICLE_QUERY_RE.search(query_text):
        slots.append("article_anchor")
    if _OUTCOME_QUERY_RE.search(query_text):
        slots.append("operative")
    if answer_type == "boolean":
        slots.append("boolean_support")

    ordered_unique: list[str] = []
    seen: set[str] = set()
    for slot in slots:
        if slot not in seen:
            seen.add(slot)
            ordered_unique.append(slot)
    return tuple(ordered_unique)


def _slot_is_covered(*, slot: str, pages: list[RetrievedPage]) -> bool:
    """Return whether at least one page covers a typed support slot.

    Args:
        slot: Typed support slot.
        pages: Candidate pages in current portfolio order.

    Returns:
        bool: True when the slot is sufficiently covered.
    """

    if slot == "compare_docs":
        return len({page.doc_id for page in pages if page.doc_id}) >= 2
    return any(_page_supports_slot(slot=slot, page=page) for page in pages)


def _page_supports_slot(*, slot: str, page: RetrievedPage) -> bool:
    """Return whether one page supports a typed slot.

    Args:
        slot: Typed support slot.
        page: Candidate page.

    Returns:
        bool: True when the page materially supports the slot.
    """

    if slot == "date":
        return (
            page.has_date_of_issue_pattern
            or page.page_role == "issued_by_block"
            or page.page_template_family in {"issued_by_authority", "title_cover", "official_primary"}
            or matches_field_family(page, "date")
            or "date of issue" in page.page_text.casefold()
        )
    if slot == "authority":
        return (
            page.has_issued_by_pattern
            or page.page_role == "issued_by_block"
            or page.page_template_family in {"caption_header", "issued_by_authority", "title_cover"}
            or matches_field_family(page, "authority")
            or "issued by" in page.page_text.casefold()
        )
    if slot == "claim":
        return page.has_claim_number_pattern or matches_field_family(page, "claim")
    if slot == "law_number":
        return page.has_law_number_pattern or matches_field_family(page, "law_number")
    if slot == "party_title":
        return (
            page.has_caption_block
            or page.page_role == "title_cover"
            or page.page_template_family
            in {
                "caption_header",
                "title_cover",
                "official_primary",
            }
        )
    if slot == "article_anchor":
        return (
            bool(page.article_refs)
            or page.page_role == "article_clause"
            or page.page_template_family in {"article_body", "schedule_table"}
        )
    if slot == "operative":
        return page.page_role == "operative_order" or page.page_template_family in {
            "operative_order",
            "official_primary",
        }
    if slot == "boolean_support":
        return bool(page.article_refs) or page.page_template_family in {
            "article_body",
            "operative_order",
            "schedule_table",
            "official_primary",
        }
    return False
