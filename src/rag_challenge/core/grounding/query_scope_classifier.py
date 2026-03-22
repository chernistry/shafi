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
_CASE_REF_RE = re.compile(r"\b[A-Z]{2,4}(?:\s+\d{3}[-/]\d{4}|-\d{3}-\d{4}|/\d{3}/\d{4})\b", re.IGNORECASE)
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
_PARTY_SCOPE_TERMS = ("party", "parties", "claimant", "defendant", "appellant", "respondent")
_JUDGE_SCOPE_TERMS = ("judge", "registrar", "justice")


def extract_explicit_page_numbers(query: str) -> list[int]:
    """Extract explicit human page numbers from a query.

    Args:
        query: Raw user question text.

    Returns:
        Ordered unique positive page numbers mentioned in the query.
    """
    page_numbers: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"\b(?:page|pages)\s+((?:\d+(?:\s*(?:,|and)\s*)?)*)", query or "", re.IGNORECASE):
        for raw_num in re.findall(r"\d+", match.group(1) or ""):
            try:
                page_num = int(raw_num)
            except (TypeError, ValueError):
                continue
            if page_num <= 0 or page_num in seen:
                continue
            seen.add(page_num)
            page_numbers.append(page_num)
    return page_numbers


def _dedupe_roles(*roles: str) -> list[str]:
    """Return roles in input order without duplicates or blanks.

    Args:
        *roles: Candidate page-role values.

    Returns:
        Ordered unique role strings.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for role in roles:
        value = str(role or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _case_scope_roles(query_lower: str) -> list[str]:
    """Infer page roles for multi-doc compare/full-case questions.

    Args:
        query_lower: Lower-cased query text.

    Returns:
        Ordered page-role list for sidecar page retrieval.
    """
    roles = [PageRole.TITLE_COVER.value]
    if any(term in query_lower for term in _JUDGE_SCOPE_TERMS):
        roles.append(PageRole.ISSUED_BY_BLOCK.value)
    if any(term in query_lower for term in _PARTY_SCOPE_TERMS):
        roles.append(PageRole.CAPTION.value)
    return _dedupe_roles(*roles)


def _default_budget(settings: object) -> int:
    """Read grounding_page_budget_default from pipeline settings, fallback to 2.

    Args:
        settings: PipelineSettings or any object with grounding_page_budget_default attribute.

    Returns:
        Configured page budget, defaulting to 2 if not set.
    """
    return int(getattr(settings, "grounding_page_budget_default", 2))


def classify_query_scope(
    query: str,
    answer_type: str,
    *,
    settings: object = None,
) -> QueryScopePrediction:
    """Classify a question into a grounding scope prediction.

    Args:
        query: Raw user question text.
        answer_type: Normalized answer type (boolean, number, date, name, names, free_text).
        settings: Optional PipelineSettings — reads grounding_page_budget_default (default 2).

    Returns:
        QueryScopePrediction with scope mode, target roles, page budget, etc.
    """
    q = (query or "").strip()
    ql = q.lower()
    _budget = _default_budget(settings)

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
        roles = _case_scope_roles(ql)
        return QueryScopePrediction(
            scope_mode=ScopeMode.FULL_CASE_FILES,
            target_page_roles=roles,
            page_budget=max(4, _budget),
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

    # Detect multi-law queries (e.g., "Was Law X enacted the same year as Law Y?")
    # Match law titles with or without year/number suffix.
    _law_title_count = len(set(re.findall(
        r"\b(?:[A-Z][A-Za-z]+\s+){1,4}(?:Law|Regulations?)\b",
        q,
    )))
    multi_law_query = _law_title_count >= 2

    # Compare pair (multiple case refs)
    case_refs = _CASE_REF_RE.findall(q)
    if len(case_refs) >= 2:
        roles = _case_scope_roles(ql)
        if "date of issue" in ql:
            roles = _dedupe_roles(PageRole.ISSUED_BY_BLOCK.value, *roles)
        # Compare-pair needs pages from EACH referenced case. Budget must
        # be at least len(case_refs) to avoid recall loss from the recall
        # floor (which adds 1 page per missing doc). F-beta 2.5 math:
        # missing 1 gold page = -46% G, extra wrong page = -7% G.
        compare_budget = max(len(case_refs), _budget)
        return QueryScopePrediction(
            scope_mode=ScopeMode.COMPARE_PAIR,
            target_page_roles=roles,
            page_budget=compare_budget,
            requires_all_docs_in_case=False,
            hard_anchor_strings=anchor_terms,
        )

    # F-beta 2.5 recall optimization: page_budget=2 universally for SINGLE_FIELD.
    # EYAL's analysis: hit@1=65.6% vs hit@2=82.8% → +17.2pp recall from 2nd page.
    # Missing 1 gold page costs ~46% G; adding 1 wrong page costs ~6.5% G.
    # The recall gain massively outweighs the precision cost.

    # Multi-law comparisons need title/enactment pages, not article content.
    # This covers temporal (enacted/commencement), authority (administered by),
    # and identity (same entity/law) comparisons across laws.
    _multi_law_metadata_compare = multi_law_query and any(
        term in ql for term in (
            "enacted", "came into force", "commencement", "same year", "same date", "earlier",
            "administered", "same entity", "same authority", "same registrar",
        )
    )
    if _multi_law_metadata_compare:
        # Use COMPARE_PAIR so the sidecar's per-doc logic selects one page per law.
        # SINGLE_FIELD_SINGLE_DOC can fill both budget slots from the same law,
        # leaving the second law unrepresented (root cause of bb67fc19/b249b41b/d5bc7441 false nulls).
        # COMPARE_PAIR safe-sidecar: per_doc_cmp takes first page per unique doc_id.
        multi_law_budget = max(_law_title_count, _budget)
        return QueryScopePrediction(
            scope_mode=ScopeMode.COMPARE_PAIR,
            target_page_roles=[PageRole.TITLE_COVER.value, PageRole.ISSUED_BY_BLOCK.value],
            page_budget=multi_law_budget,
            hard_anchor_strings=anchor_terms,
        )

    # Date of issue
    if "date of issue" in ql or "issue date" in ql:
        return QueryScopePrediction(
            scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
            target_page_roles=[PageRole.ISSUED_BY_BLOCK.value, PageRole.TITLE_COVER.value],
            page_budget=_budget,
            hard_anchor_strings=anchor_terms,
        )

    # Costs / outcome
    if any(term in ql for term in _COSTS_TERMS):
        return QueryScopePrediction(
            scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
            target_page_roles=[PageRole.OPERATIVE_ORDER.value, PageRole.COSTS_BLOCK.value],
            page_budget=_budget,
            hard_anchor_strings=anchor_terms,
        )

    # SCT/CFI appeal status: "was SCT X appealed to the CFI?" — the permission-to-appeal
    # ruling is in the OPERATIVE_ORDER pages of the SCT judgment doc. Target those first.
    _case_refs_present = _CASE_REF_RE.findall(q)
    if (
        len(_case_refs_present) == 1
        and "appea" in ql
        and ("cfi" in ql or "court of first instance" in ql)
        and any(r.startswith("SCT") for r in _case_refs_present)
    ):
        return QueryScopePrediction(
            scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
            target_page_roles=[PageRole.OPERATIVE_ORDER.value, PageRole.TITLE_COVER.value],
            page_budget=_budget,
            hard_anchor_strings=anchor_terms,
        )

    # Article / section / schedule anchor
    if anchor_terms:
        return QueryScopePrediction(
            scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
            target_page_roles=[PageRole.ARTICLE_CLAUSE.value, PageRole.SCHEDULE_TABLE.value],
            page_budget=_budget,
            hard_anchor_strings=anchor_terms,
        )

    # Per-type page budgets based on gold-page distribution:
    # free_text: avg 5.6 gold pages → budget=4 for multi-provision answers
    # boolean/name: avg 1.5 → budget=2
    # number/date: avg 1.0 → budget=2 (safe margin)
    if answer_type in {"boolean", "number", "date", "name", "names"}:
        return QueryScopePrediction(
            scope_mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC,
            target_page_roles=[PageRole.TITLE_COVER.value, PageRole.CAPTION.value],
            page_budget=_budget,
        )

    # Broad free text — needs more pages (avg 5.6 gold pages)
    return QueryScopePrediction(
        scope_mode=ScopeMode.BROAD_FREE_TEXT,
        target_page_roles=[],
        page_budget=max(4, _budget),
    )
