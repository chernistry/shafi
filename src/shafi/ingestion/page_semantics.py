"""Deterministic page-structure and officialness enrichment for ingest."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from shafi.models import PageRole

_FIELD_LABEL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Date of Issue", re.compile(r"\bDate of Issue\b", re.IGNORECASE)),
    ("Date of Re-issue", re.compile(r"\bDate of Re-?issue\b", re.IGNORECASE)),
    ("Enactment Notice", re.compile(r"\bEnactment Notice\b", re.IGNORECASE)),
    ("Issued by", re.compile(r"\bIssued by\b", re.IGNORECASE)),
    ("Issued under", re.compile(r"\bIssued under\b", re.IGNORECASE)),
    ("Registrar", re.compile(r"\bRegistrar\b", re.IGNORECASE)),
    ("Judge", re.compile(r"\bJudge\b", re.IGNORECASE)),
    ("Claim No", re.compile(r"\bClaim\s+No\.?\b", re.IGNORECASE)),
    ("Law No", re.compile(r"\b(?:DIFC\s+)?Law\s+No\.?\b", re.IGNORECASE)),
)
_TITLE_LINE_RE = re.compile(
    r"\b(?:DIFC\s+Law|Law\s+No\.?|Judgment|Order|Regulation|Rules?|Schedule|Appendix|Annex)\b",
    re.IGNORECASE,
)
_CAPTION_RE = re.compile(r"\b(?:v\.?|vs\.?|versus)\b", re.IGNORECASE)
_ARTICLE_HEADING_RE = re.compile(
    r"^(?:ARTICLE|Article|SECTION|Section|SCHEDULE|Schedule|CHAPTER|Chapter|PART|Part)\s+\S+",
)
_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\s+[A-Z].+")
_ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z\s,&()/.-]{5,}$")
_CLAIM_NUMBER_RE = re.compile(r"\b(?:Claim|Case)\s+No\.?\s*[A-Za-z0-9/-]+\b", re.IGNORECASE)
_LAW_NUMBER_RE = re.compile(r"\b(?:DIFC\s+)?Law\s+No\.?\s+\d+\s+of\s+\d{4}\b", re.IGNORECASE)
_NOTICE_PANEL_RE = re.compile(r"\b(?:enactment notice|notice of enactment|issued under|published by)\b", re.IGNORECASE)
_ISSUED_BY_RE = re.compile(r"\bIssued by\b", re.IGNORECASE)
_DATE_OF_ISSUE_RE = re.compile(r"\bDate of Issue\b", re.IGNORECASE)
_REFERENCE_PAGE_RE = re.compile(
    r"\b(?:table of contents|contents|see also|amended by|appendix|annex|for reference|guidance notes?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PageTopStructure:
    """Structured top-of-page signals extracted deterministically.

    Args:
        top_lines: First non-empty normalized lines.
        heading_lines: Heading-like subset of the top lines.
        field_labels_present: Canonical explicit field labels found near the top.
        has_caption_block: Whether the page resembles a caption/header page.
        has_title_like_header: Whether the page resembles an official title/cover.
        has_issued_by_pattern: Whether an issued-by block is present.
        has_date_of_issue_pattern: Whether a date-of-issue field is present.
        has_claim_number_pattern: Whether a claim/case number is present.
        has_law_number_pattern: Whether a law-number pattern is present.
    """

    top_lines: list[str]
    heading_lines: list[str]
    field_labels_present: list[str]
    has_caption_block: bool
    has_title_like_header: bool
    has_issued_by_pattern: bool
    has_date_of_issue_pattern: bool
    has_claim_number_pattern: bool
    has_law_number_pattern: bool


@dataclass(frozen=True, slots=True)
class PageSemanticCatalog:
    """Deterministic template and authority catalog for a page.

    Args:
        document_template_family: Document-level template family.
        page_template_family: Page-level template/authority family.
        officialness_score: 0-1 prior that this page is authoritative.
        source_vs_reference_prior: 0-1 prior that the page is source-like, not reference-like.
    """

    document_template_family: str
    page_template_family: str
    officialness_score: float
    source_vs_reference_prior: float


def _normalize_line(text: str) -> str:
    """Normalize one page line for structure matching.

    Args:
        text: Raw line text.

    Returns:
        str: Normalized line.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip(" \t-:|")
    return normalized


def _top_lines(page_text: str, *, limit: int = 6) -> list[str]:
    """Extract the first non-empty normalized lines from a page.

    Args:
        page_text: Raw page text.
        limit: Maximum number of lines to keep.

    Returns:
        list[str]: Top non-empty lines.
    """

    lines: list[str] = []
    for raw in page_text.splitlines():
        line = _normalize_line(raw)
        if not line:
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _is_heading_line(line: str) -> bool:
    """Decide whether a line looks like an official heading.

    Args:
        line: Normalized line text.

    Returns:
        bool: True when the line looks heading-like.
    """

    return bool(
        _ARTICLE_HEADING_RE.match(line)
        or _NUMBERED_HEADING_RE.match(line)
        or _ALL_CAPS_HEADING_RE.match(line)
        or (_TITLE_LINE_RE.search(line) and len(line.split()) <= 16)
    )


def extract_page_top_structure(page_text: str, *, page_num: int, total_pages: int) -> PageTopStructure:
    """Extract deterministic top-of-page structure signals.

    Args:
        page_text: Raw page text.
        page_num: 1-based page number.
        total_pages: Total pages in the document.

    Returns:
        PageTopStructure: Extracted structure signals.
    """

    top_lines = _top_lines(page_text)
    heading_lines = [line for line in top_lines if _is_heading_line(line)]
    field_labels_present = [label for label, pattern in _FIELD_LABEL_PATTERNS if pattern.search("\n".join(top_lines))]
    joined_top = "\n".join(top_lines)
    has_claim_number = bool(_CLAIM_NUMBER_RE.search(joined_top))
    has_law_number = bool(_LAW_NUMBER_RE.search(joined_top))
    has_notice_panel = bool(_NOTICE_PANEL_RE.search(joined_top))
    has_issued_by = bool(_ISSUED_BY_RE.search(joined_top))
    has_date_of_issue = bool(_DATE_OF_ISSUE_RE.search(joined_top))
    has_caption = page_num == 1 and bool(_CAPTION_RE.search(joined_top))
    title_like = page_num == 1 and (
        bool(heading_lines)
        or has_law_number
        or has_notice_panel
        or bool(_TITLE_LINE_RE.search(joined_top))
        or (len(top_lines) >= 2 and all(len(line.split()) <= 14 for line in top_lines[:2]))
    )
    return PageTopStructure(
        top_lines=top_lines,
        heading_lines=heading_lines,
        field_labels_present=field_labels_present,
        has_caption_block=has_caption,
        has_title_like_header=title_like and not has_caption,
        has_issued_by_pattern=has_issued_by,
        has_date_of_issue_pattern=has_date_of_issue,
        has_claim_number_pattern=has_claim_number,
        has_law_number_pattern=has_law_number,
    )


def classify_document_template_family(doc_family: str, *, first_page_text: str) -> str:
    """Classify a document-level template family from current ingest signals.

    Args:
        doc_family: Existing deterministic document family.
        first_page_text: First non-empty page text.

    Returns:
        str: Document template family label.
    """

    top = extract_page_top_structure(first_page_text, page_num=1, total_pages=1)
    if doc_family in {"judgment", "order"} and top.has_caption_block:
        return "captioned_case_file"
    if doc_family in {"consolidated_law", "amendment_law", "regulations"} and (
        top.has_title_like_header or top.has_law_number_pattern
    ):
        return "issued_law_instrument"
    if doc_family == "enactment_notice":
        return "enactment_notice"
    if doc_family == "tracked_changes_amendment":
        return "tracked_change_markup"
    return doc_family or "generic_legal_document"


def catalog_page_semantics(
    *,
    doc_family: str,
    document_template_family: str,
    page_family: str,
    page_role: str,
    page_text: str,
    page_num: int,
    total_pages: int,
    top_structure: PageTopStructure,
) -> PageSemanticCatalog:
    """Catalog page template and authority priors.

    Args:
        doc_family: Existing document family.
        document_template_family: Document template family.
        page_family: Existing page family.
        page_role: Existing semantic page role.
        page_text: Raw page text.
        page_num: 1-based page number.
        total_pages: Total document pages.
        top_structure: Precomputed page top-structure signals.

    Returns:
        PageSemanticCatalog: Deterministic authority catalog.
    """

    del total_pages  # authority is local here; page_num/family/role carry enough context
    head = "\n".join(top_structure.top_lines)
    source_vs_reference_prior = 0.55
    officialness_score = 0.40

    if _REFERENCE_PAGE_RE.search(head) or page_family == "contents_like":
        return PageSemanticCatalog(
            document_template_family=document_template_family,
            page_template_family="duplicate_or_reference_like",
            officialness_score=0.10,
            source_vs_reference_prior=0.05,
        )

    if page_role == PageRole.TITLE_COVER:
        if page_num == 1 and (
            top_structure.has_issued_by_pattern
            or top_structure.has_date_of_issue_pattern
            or _NOTICE_PANEL_RE.search(head)
        ):
            return PageSemanticCatalog(
                document_template_family=document_template_family,
                page_template_family="issued_by_authority",
                officialness_score=0.95,
                source_vs_reference_prior=0.97,
            )
        return PageSemanticCatalog(
            document_template_family=document_template_family,
            page_template_family="title_cover",
            officialness_score=0.92 if top_structure.has_title_like_header else 0.80,
            source_vs_reference_prior=0.95,
        )
    if page_role == PageRole.CAPTION or top_structure.has_caption_block:
        return PageSemanticCatalog(
            document_template_family=document_template_family,
            page_template_family="caption_header",
            officialness_score=0.94,
            source_vs_reference_prior=0.96,
        )
    if page_role == PageRole.ISSUED_BY_BLOCK or (
        top_structure.has_issued_by_pattern and top_structure.has_date_of_issue_pattern
    ):
        return PageSemanticCatalog(
            document_template_family=document_template_family,
            page_template_family="issued_by_authority",
            officialness_score=0.95,
            source_vs_reference_prior=0.97,
        )
    if page_role == PageRole.ARTICLE_CLAUSE or any(
        line.lower().startswith(("article ", "section ", "schedule ")) for line in top_structure.heading_lines
    ):
        return PageSemanticCatalog(
            document_template_family=document_template_family,
            page_template_family="article_body",
            officialness_score=0.86,
            source_vs_reference_prior=0.88,
        )
    if page_role == PageRole.SCHEDULE_TABLE or page_family == "schedule_like":
        return PageSemanticCatalog(
            document_template_family=document_template_family,
            page_template_family="schedule_table",
            officialness_score=0.84,
            source_vs_reference_prior=0.86,
        )
    if page_role == PageRole.OPERATIVE_ORDER or page_family == "operative_order_like":
        return PageSemanticCatalog(
            document_template_family=document_template_family,
            page_template_family="operative_order",
            officialness_score=0.90 if doc_family in {"judgment", "order"} else 0.80,
            source_vs_reference_prior=0.92,
        )
    if "appendix" in head.lower() or "annex" in head.lower():
        return PageSemanticCatalog(
            document_template_family=document_template_family,
            page_template_family="appendix_reference",
            officialness_score=0.35,
            source_vs_reference_prior=0.25,
        )

    if top_structure.has_title_like_header or top_structure.field_labels_present:
        officialness_score += 0.20
        source_vs_reference_prior += 0.15
    if page_num == 1 and document_template_family in {"captioned_case_file", "issued_law_instrument"}:
        officialness_score += 0.10
        source_vs_reference_prior += 0.10

    return PageSemanticCatalog(
        document_template_family=document_template_family,
        page_template_family="official_primary" if officialness_score >= 0.65 else "other",
        officialness_score=min(1.0, round(officialness_score, 4)),
        source_vs_reference_prior=min(1.0, round(source_vs_reference_prior, 4)),
    )
