"""Rule-based query scope classifier for grounding evidence selection.

Classifies each question into a scope mode (single-field, explicit-page,
compare-pair, full-case-files, negative, broad) and predicts target page
roles and page budget. Used by the grounding sidecar to select minimal
evidentiary pages without touching the answer path.
"""

from __future__ import annotations

import re

from rag_challenge.models.schemas import PageRole, QueryScopePrediction, ScopeMode

_EXPLICIT_PAGE_RE = re.compile(r"\b(?:page|pages)\s+(\d+)\b", re.IGNORECASE)
_CASE_REF_RE = re.compile(r"\b[A-Z]{2,4}\s+\d{3}/\d{4}\b")
_ANCHOR_PATTERNS = (
    re.compile(r"\bArticle\s+\d+[A-Z]?(?:\(\w+\))?\b", re.IGNORECASE),
    re.compile(r"\bSection\s+\d+[A-Z]?(?:\(\w+\))?\b", re.IGNORECASE),
    re.compile(r"\bSchedule\s+\d+[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\bLaw\s+No\.?\s+\d+\s+of\s+\d{4}\b", re.IGNORECASE),
)

_FULL_CASE_PHRASES = (
    "full case files",
    "across all documents",
    "every document",
    "look through all documents",
    "through all documents",
)

_NEGATIVE_TERMS = ("jury", "parole", "miranda", "plea bargain", "plea")

_COSTS_TERMS = ("costs", "cost awarded", "final ruling", "outcome", "it is hereby ordered")


def classify_query_scope(query: str, answer_type: str) -> QueryScopePrediction:
    """Classify a question into a grounding scope prediction.

    Args:
        query: Raw user question text.
        answer_type: Normalized answer type (boolean, number, date, name, names, free_text).

    Returns:
        QueryScopePrediction with scope mode, target roles, page budget, etc.
    """
    q = (query or "").strip()
    ql = q.lower()

    # Explicit page reference
    if _EXPLICIT_PAGE_RE.search(ql):
        return QueryScopePrediction(
            scope_mode=ScopeMode.EXPLICIT_PAGE,
            target_page_roles=[],
            page_budget=1,
            requires_all_docs_in_case=False,
            hard_anchor_strings=[],
        )

    # Full-case scope
    if any(phrase in ql for phrase in _FULL_CASE_PHRASES):
        roles = [PageRole.TITLE_COVER.value]
        if any(term in ql for term in ("judge", "party", "claimant", "defendant")):
            roles = [PageRole.TITLE_COVER.value, PageRole.CAPTION.value]
        return QueryScopePrediction(
            scope_mode=ScopeMode.FULL_CASE_FILES,
            target_page_roles=roles,
            page_budget=4,
            requires_all_docs_in_case=True,
        )

    # Negative / unanswerable
    if any(term in ql for term in _NEGATIVE_TERMS):
        return QueryScopePrediction(
            scope_mode=ScopeMode.NEGATIVE_UNANSWERABLE,
            target_page_roles=[],
            page_budget=0,
            requires_all_docs_in_case=False,
            should_force_empty_grounding_on_null=True,
        )

    # Extract structural anchors from the query
    anchor_terms: list[str] = []
    for pattern in _ANCHOR_PATTERNS:
        anchor_terms.extend(re.findall(pattern, q))

    # Compare pair (multiple case refs)
    case_refs = _CASE_REF_RE.findall(q)
    if len(case_refs) >= 2:
        roles = [PageRole.TITLE_COVER.value]
        if any(term in ql for term in ("judge", "party")):
            roles = [PageRole.TITLE_COVER.value, PageRole.CAPTION.value]
        elif "date of issue" in ql:
            roles = [PageRole.ISSUED_BY_BLOCK.value, PageRole.TITLE_COVER.value]
        return QueryScopePrediction(
            scope_mode=ScopeMode.COMPARE_PAIR,
            target_page_roles=roles,
            page_budget=2,
            requires_all_docs_in_case=False,
            hard_anchor_strings=anchor_terms,
        )

    # Date of issue
    if "date of issue" in ql or "issue date" in ql:
        return QueryScopePrediction(
            scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
            target_page_roles=[PageRole.ISSUED_BY_BLOCK.value, PageRole.TITLE_COVER.value],
            page_budget=1,
            hard_anchor_strings=anchor_terms,
        )

    # Costs / outcome
    if any(term in ql for term in _COSTS_TERMS):
        return QueryScopePrediction(
            scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
            target_page_roles=[PageRole.OPERATIVE_ORDER.value, PageRole.COSTS_BLOCK.value],
            page_budget=1,
            hard_anchor_strings=anchor_terms,
        )

    # Article / section / schedule anchor
    if anchor_terms:
        return QueryScopePrediction(
            scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
            target_page_roles=[PageRole.ARTICLE_CLAUSE.value, PageRole.SCHEDULE_TABLE.value],
            page_budget=1,
            hard_anchor_strings=anchor_terms,
        )

    # Strict types default
    if answer_type in {"boolean", "number", "date", "name", "names"}:
        return QueryScopePrediction(
            scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
            target_page_roles=[PageRole.TITLE_COVER.value, PageRole.CAPTION.value],
            page_budget=1,
        )

    # Broad free text
    return QueryScopePrediction(
        scope_mode=ScopeMode.BROAD_FREE_TEXT,
        target_page_roles=[],
        page_budget=2,
    )
