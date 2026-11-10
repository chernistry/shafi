"""Typed evidence extraction for compare-style grounding families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shafi.core.grounding.authority_priors import select_authoritative_single_page
from shafi.core.grounding.scope_policy import extract_case_refs

if TYPE_CHECKING:
    from shafi.models.schemas import RetrievedPage


@dataclass(frozen=True, slots=True)
class TypedComparisonPanel:
    """One compare-side authoritative page.

    Args:
        doc_id: Document represented by this side.
        page_id: Selected authoritative page.
        case_ref: Optional case reference matched from the query.
    """

    doc_id: str
    page_id: str
    case_ref: str = ""


def build_typed_comparison_panel(
    *,
    query: str,
    ordered_pages: list[RetrievedPage],
) -> tuple[TypedComparisonPanel, ...]:
    """Build one authoritative evidence page per compare-side document.

    Args:
        query: Raw user question.
        ordered_pages: Candidate pages in current ranking order.

    Returns:
        tuple[TypedComparisonPanel, ...]: Up to two compare-side panels.
    """

    if len(ordered_pages) < 2:
        return ()

    case_refs = extract_case_refs(query)
    pages_by_doc: dict[str, list[RetrievedPage]] = {}
    for page in ordered_pages:
        if not page.doc_id or not page.page_id:
            continue
        pages_by_doc.setdefault(page.doc_id, []).append(page)
    if len(pages_by_doc) < 2:
        return ()

    ordered_docs = _ordered_compare_docs(
        query=query,
        pages_by_doc=pages_by_doc,
        case_refs=case_refs,
    )
    panels: list[TypedComparisonPanel] = []
    for doc_id, case_ref in ordered_docs[:2]:
        authoritative_page = select_authoritative_single_page(query, pages_by_doc[doc_id])
        if authoritative_page is None:
            continue
        panels.append(
            TypedComparisonPanel(
                doc_id=doc_id,
                page_id=authoritative_page.page_id,
                case_ref=case_ref,
            )
        )
    return tuple(panels)


def _ordered_compare_docs(
    *,
    query: str,
    pages_by_doc: dict[str, list[RetrievedPage]],
    case_refs: list[str],
) -> list[tuple[str, str]]:
    """Order compare documents using case refs first, then retrieval order.

    Args:
        query: Raw user question.
        pages_by_doc: Candidate pages grouped by document ID.
        case_refs: Case references explicitly mentioned in the query.

    Returns:
        list[tuple[str, str]]: Ordered ``(doc_id, case_ref)`` pairs.
    """

    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    if case_refs:
        for case_ref in case_refs:
            needle = case_ref.casefold()
            for doc_id, doc_pages in pages_by_doc.items():
                if doc_id in seen:
                    continue
                representative_page = doc_pages[0]
                haystacks = [
                    representative_page.doc_title,
                    " ".join(representative_page.top_lines),
                    " ".join(representative_page.heading_lines),
                    representative_page.page_text[:500],
                ]
                if any(needle in str(haystack).casefold() for haystack in haystacks):
                    seen.add(doc_id)
                    ordered.append((doc_id, case_ref))
                    break
    for page in ordered_pages_from_groups(pages_by_doc):
        if page.doc_id in seen:
            continue
        seen.add(page.doc_id)
        ordered.append((page.doc_id, ""))
    if not ordered and len(pages_by_doc) >= 2:
        return [(doc_id, "") for doc_id in pages_by_doc]
    del query
    return ordered


def ordered_pages_from_groups(pages_by_doc: dict[str, list[RetrievedPage]]) -> list[RetrievedPage]:
    """Flatten grouped pages while preserving original within-doc order.

    Args:
        pages_by_doc: Candidate pages grouped by document ID.

    Returns:
        list[RetrievedPage]: Representative flattened order.
    """

    flattened: list[RetrievedPage] = []
    for doc_pages in pages_by_doc.values():
        flattened.extend(doc_pages)
    return flattened
