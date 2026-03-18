from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from rag_challenge.config import get_settings
from rag_challenge.config.logging import setup_logging
from rag_challenge.core.embedding import EmbeddingClient
from rag_challenge.core.qdrant import QdrantStore
from rag_challenge.core.sparse_bm25 import BM25SparseEncoder
from rag_challenge.ingestion.chunker import LegalChunker
from rag_challenge.ingestion.parser import DocumentParser
from rag_challenge.ingestion.sac import SACGenerator
from rag_challenge.llm import LLMProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from rag_challenge.models import Chunk, PageMetadata, ParsedDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preprocessing enrichment utilities (additive metadata, no architecture changes)
# ---------------------------------------------------------------------------

_CASE_ID_RE = re.compile(r"\b(?:CFI|CA|ARB|SCT|TCD|ENF)\s+\d{3}/\d{4}\b", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"(?:AED|USD|GBP|EUR)\s*[\d,]+(?:\.\d+)?", re.IGNORECASE)
_ARTICLE_REF_RE = re.compile(
    r"\b(?:Article|Section|Schedule|Part|Rule|Regulation|Clause)\s+[A-Za-z0-9]+(?:\([A-Za-z0-9]+\))*",
    re.IGNORECASE,
)
_LAW_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z&,'()/-]+?\s+Law(?:\s+\d{4}|,\s*DIFC\s+Law\s+No\.?\s*\d+\s+of\s+\d{4}|"
    r"\s+No\.?\s*\d+\s+of\s+\d{4})?)\b"
)
_COURT_RE = re.compile(
    r"\b(?:DIFC Courts?|Court of Appeal|Small Claims Tribunal|Tribunal|Registrar|Chief Justice)\b",
    re.IGNORECASE,
)
_PARTY_ROLE_RE = re.compile(
    r"\b(?:claimant|claimants|respondent|respondents|appellant|appellants|applicant|applicants|"
    r"defendant|defendants|plaintiff|plaintiffs|petitioner|petitioners)\b[:\s-]*([A-Z][A-Za-z0-9&.,'()/-]{2,80})",
    re.IGNORECASE,
)
_CROSS_REF_RE = re.compile(
    r"\b(?:Article|Section|Schedule|Part|Rule|Regulation|Clause)\s+[A-Za-z0-9]+(?:\([A-Za-z0-9]+\))*"
    r"|\bamended by\b|\bsee also\b|\bsubject to\b|\bpursuant to\b",
    re.IGNORECASE,
)
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\uFE58\uFE63\uFF0D]")
_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
_QUOTE_RE = re.compile(r"[\u2018\u2019\u201A\u201B]")
_DQUOTE_RE = re.compile(r"[\u201C\u201D\u201E\u201F]")
_EASTERN_ARABIC_MAP = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669", "0123456789")


def _classify_doc_family(title: str, summary: str, first_page_text: str) -> str:
    combined = f"{title} {first_page_text[:2000]}".lower()
    if "tracked change" in combined or "tracked_change" in combined:
        return "tracked_changes_amendment"
    if "enactment notice" in combined and len(first_page_text) < 2000:
        return "enactment_notice"
    if "consolidated version" in combined or "as amended by" in combined:
        return "consolidated_law"
    if "amendment law" in combined:
        return "amendment_law"
    if re.search(r"\bregulations?\b", combined) and "law no" not in combined:
        return "regulations"
    if "it is hereby ordered" in combined:
        return "order"
    if re.search(r"\bjudgment\b", combined):
        return "judgment"
    if _CASE_ID_RE.search(combined):
        return "order"
    if re.search(r"\blaw\s+no\b", combined):
        return "consolidated_law"
    return "other"


def _classify_page_family(page_text: str, page_num: int, total_pages: int) -> str:
    t = page_text[:3000].lower() if page_text else ""
    upper_t = page_text[:3000] if page_text else ""
    if page_num == 1 and _CASE_ID_RE.search(t):
        return "cover_like"
    if "table of contents" in t or (t.strip().startswith("contents") and page_num <= 3):
        return "contents_like"
    if "it is hereby ordered" in t:
        return "operative_order_like"
    if re.search(r"\bSCHEDULE\b", upper_t):
        return "schedule_like"
    if re.search(r"\bANNEX\b", upper_t) or re.search(r"\bAPPENDIX\b", upper_t):
        return "schedule_like"
    if "enactment notice" in t or "comes into force" in t:
        return "enactment_like"
    if re.search(r"\bcommencement\b", t) and page_num <= 5:
        return "commencement_like"
    if "administered by" in t:
        return "administration_like"
    if _CURRENCY_RE.search(t) and ("cost" in t or "pay" in t or "award" in t or "sum of" in t):
        return "costs_like"
    if page_num == 1 and re.search(r"\blaw\s+no\b", t):
        return "citation_title_like"
    return ""


def _normalize_identifiers(text: str) -> str:
    if not text:
        return ""
    result = unicodedata.normalize("NFKC", text)
    result = _DASH_RE.sub("-", result)
    result = _ZERO_WIDTH_RE.sub("", result)
    result = result.replace("\u00A0", " ")
    result = _QUOTE_RE.sub("'", result)
    result = _DQUOTE_RE.sub('"', result)
    result = result.translate(_EASTERN_ARABIC_MAP)
    return result


def _extract_amount_roles(page_text: str) -> list[str]:
    if not page_text or not _CURRENCY_RE.search(page_text):
        return []
    t = page_text.lower()
    roles: list[str] = []
    if any(kw in t for kw in ("claim value", "claimed amount", "claim for")):
        roles.append("claim_amount")
    if any(kw in t for kw in ("ordered to pay", "costs awarded", "assessed in the amount", "sum of")):
        roles.append("costs_awarded")
    if any(kw in t for kw in ("costs schedule", "claimed costs", "statement of costs")):
        roles.append("costs_claimed")
    if any(kw in t for kw in ("penalty", "fine", "prescribed penalty")):
        roles.append("penalty")
    if "damages" in t:
        roles.append("damages")
    return roles


def _unique_normalized(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        normalized = _normalize_identifiers(raw)
        compact = re.sub(r"\s+", " ", normalized).strip(" ,.;:")
        if not compact:
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(compact)
    return out


def _extract_party_names(text: str) -> list[str]:
    return _unique_normalized([match.group(1) for match in _PARTY_ROLE_RE.finditer(text or "")])


def _extract_court_names(text: str) -> list[str]:
    return _unique_normalized([match.group(0) for match in _COURT_RE.finditer(text or "")])


def _extract_law_titles(*texts: str) -> list[str]:
    values: list[str] = []
    for text in texts:
        values.extend(match.group(1) for match in _LAW_TITLE_RE.finditer(text or ""))
    return _unique_normalized(values)


def _extract_article_refs(text: str) -> list[str]:
    return _unique_normalized([match.group(0) for match in _ARTICLE_REF_RE.finditer(text or "")])


def _extract_case_numbers(*texts: str) -> list[str]:
    values: list[str] = []
    for text in texts:
        values.extend(match.group(0) for match in _CASE_ID_RE.finditer(text or ""))
    return _unique_normalized(values)


def _extract_cross_refs(text: str) -> list[str]:
    return _unique_normalized([match.group(0) for match in _CROSS_REF_RE.finditer(text or "")])


_ISSUED_BY_BLOCK_RE = re.compile(
    r"Issued by:\s*(?P<issuer>[^\n]+).*?"
    r"Date of Issue:\s*(?P<doi>[^\n]+)"
    r"(?:.*?Date of Re-issue:\s*(?P<reissue>[^\n]+))?",
    re.IGNORECASE | re.DOTALL,
)
_LAW_NO_RE = re.compile(r"\b(?:DIFC\s+)?Law\s+No\.?\s+(\d+)\s+of\s+(\d{4})\b", re.IGNORECASE)
_SCHEDULE_RE = re.compile(r"\bSchedule\s+(\d+[A-Z]?)\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"\bSection\s+(\d+[A-Z]?)\b", re.IGNORECASE)


def _page_role_for_text(page_text: str, page_num: int, total_pages: int) -> str:
    """Classify the semantic role of a page for grounding evidence selection.

    Args:
        page_text: Raw page text content.
        page_num: 1-based page number.
        total_pages: Total pages in the document.

    Returns:
        A PageRole string value.
    """
    from rag_challenge.models.schemas import PageRole

    t = page_text[:3000].lower() if page_text else ""
    if page_num == 1 and _CASE_ID_RE.search(t):
        return PageRole.TITLE_COVER
    if "issued by:" in t and "date of issue:" in t:
        return PageRole.ISSUED_BY_BLOCK
    if "it is hereby ordered" in t:
        return PageRole.OPERATIVE_ORDER
    if _CURRENCY_RE.search(t) and ("cost" in t or "pay" in t or "award" in t):
        return PageRole.COSTS_BLOCK
    if re.search(r"\barticle\s+\d+", t):
        return PageRole.ARTICLE_CLAUSE
    if re.search(r"\bschedule\b", t):
        return PageRole.SCHEDULE_TABLE
    if "comes into force" in t or "commencement" in t:
        return PageRole.COMMENCEMENT
    if "administered by" in t:
        return PageRole.ADMINISTRATION
    if page_num == 1 and re.search(r"\bv\.?\s+\b", page_text[:1500] if page_text else ""):
        return PageRole.CAPTION
    return PageRole.OTHER


def _extract_page_links(page_text: str) -> list[str]:
    """Extract intra-document reference anchors from page text.

    Args:
        page_text: Raw page text content.

    Returns:
        Unique normalized cross-reference strings (article/section/schedule).
    """
    values: list[str] = []
    values.extend(_extract_article_refs(page_text))
    values.extend(f"Section {m.group(1)}" for m in _SECTION_RE.finditer(page_text or ""))
    values.extend(f"Schedule {m.group(1)}" for m in _SCHEDULE_RE.finditer(page_text or ""))
    return _unique_normalized(values)


def _support_fact_search_text(
    *,
    doc_title: str,
    doc_family: str,
    page_family: str,
    page_role: str,
    fact_type: str,
    normalized_value: str,
    quote_text: str,
    scope_ref: str,
) -> str:
    """Build a concatenated search string for support-fact embedding.

    Args:
        doc_title: Document title.
        doc_family: Document family classification.
        page_family: Page family classification.
        page_role: Semantic page role.
        fact_type: Type of fact (e.g. date_of_issue, party).
        normalized_value: Normalized fact value.
        quote_text: Supporting quote text.
        scope_ref: Scope reference (e.g. article/case number).

    Returns:
        Pipe-separated search text string.
    """
    return " | ".join(
        part
        for part in [
            doc_title,
            doc_family,
            page_family,
            page_role,
            fact_type,
            normalized_value,
            scope_ref,
            quote_text[:600],
        ]
        if part
    )


def _extract_support_facts_for_page(
    *,
    doc_id: str,
    doc_title: str,
    doc_type: str,
    page_num: int,
    total_pages: int,
    page_text: str,
    doc_family: str,
) -> list[dict[str, object]]:
    """Extract grounding support facts from a single page.

    Args:
        doc_id: Document ID.
        doc_title: Document title.
        doc_type: Document type string.
        page_num: 1-based page number.
        total_pages: Total pages in document.
        page_text: Raw page text.
        doc_family: Document family classification.

    Returns:
        List of dicts with support fact fields (ready for SupportFact construction).
    """
    page_id = f"{doc_id}_{page_num}"
    page_family = _classify_page_family(page_text, page_num, total_pages)
    page_role = _page_role_for_text(page_text, page_num, total_pages)
    norm_text = _normalize_identifiers(page_text) if page_text else ""

    facts: list[dict[str, object]] = []

    def _append(
        *,
        fact_type: str,
        normalized_value: str,
        quote_text: str,
        field_explicitness: float,
        scope_ref: str = "",
    ) -> None:
        raw_key = f"{doc_id}:{page_num}:{fact_type}:{normalized_value}:{scope_ref}"
        fact_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        facts.append({
            "fact_id": fact_id,
            "doc_id": doc_id,
            "page_id": page_id,
            "page_num": page_num,
            "doc_title": doc_title,
            "doc_type": doc_type,
            "doc_family": doc_family,
            "page_family": page_family,
            "page_role": page_role,
            "fact_type": fact_type,
            "normalized_value": normalized_value,
            "quote_text": quote_text.strip()[:1200],
            "field_explicitness": field_explicitness,
            "scope_ref": scope_ref,
            "search_text": _support_fact_search_text(
                doc_title=doc_title,
                doc_family=doc_family,
                page_family=page_family,
                page_role=page_role,
                fact_type=fact_type,
                normalized_value=normalized_value,
                quote_text=quote_text,
                scope_ref=scope_ref,
            ),
        })

    # case numbers / law numbers
    for case_no in _extract_case_numbers(doc_title, page_text):
        _append(
            fact_type="case_number",
            normalized_value=case_no,
            quote_text=case_no,
            field_explicitness=1.0,
            scope_ref=case_no,
        )

    for m in _LAW_NO_RE.finditer(norm_text):
        law_no = f"Law No. {m.group(1)} of {m.group(2)}"
        _append(
            fact_type="law_number",
            normalized_value=law_no,
            quote_text=m.group(0),
            field_explicitness=1.0,
            scope_ref=law_no,
        )

    # title / caption parties
    from rag_challenge.models.schemas import PageRole as _PageRole

    if page_role in {_PageRole.TITLE_COVER, _PageRole.CAPTION}:
        for party in _extract_party_names(norm_text):
            _append(
                fact_type="party",
                normalized_value=party,
                quote_text=party,
                field_explicitness=0.8,
            )

    # issued-by block
    issued_match = _ISSUED_BY_BLOCK_RE.search(norm_text)
    if issued_match:
        issuer = _normalize_identifiers(issued_match.group("issuer") or "").strip()
        doi = _normalize_identifiers(issued_match.group("doi") or "").strip()
        reissue = _normalize_identifiers(issued_match.group("reissue") or "").strip() if issued_match.group("reissue") else ""

        if issuer:
            _append(
                fact_type="judge_or_registrar",
                normalized_value=issuer,
                quote_text=issuer,
                field_explicitness=1.0,
            )
        if doi:
            _append(
                fact_type="date_of_issue",
                normalized_value=doi,
                quote_text=f"Date of Issue: {doi}",
                field_explicitness=1.0,
            )
        if reissue:
            _append(
                fact_type="date_of_reissue",
                normalized_value=reissue,
                quote_text=f"Date of Re-issue: {reissue}",
                field_explicitness=1.0,
            )

    # article / section / schedule anchors
    for article_ref in _extract_article_refs(norm_text):
        _append(
            fact_type="article_id",
            normalized_value=article_ref,
            quote_text=article_ref,
            field_explicitness=1.0,
            scope_ref=article_ref,
        )

    for m in _SECTION_RE.finditer(norm_text):
        value = f"Section {m.group(1)}"
        _append(
            fact_type="section_id",
            normalized_value=value,
            quote_text=m.group(0),
            field_explicitness=1.0,
            scope_ref=value,
        )

    for m in _SCHEDULE_RE.finditer(norm_text):
        value = f"Schedule {m.group(1)}"
        _append(
            fact_type="schedule_id",
            normalized_value=value,
            quote_text=m.group(0),
            field_explicitness=1.0,
            scope_ref=value,
        )

    # amounts
    amount_roles = _extract_amount_roles(page_text)
    for role in amount_roles:
        _append(
            fact_type=role,
            normalized_value=role,
            quote_text=norm_text[:600],
            field_explicitness=0.5,
        )

    # operative order / outcome
    if "it is hereby ordered" in (norm_text.lower() if norm_text else ""):
        _append(
            fact_type="operative_order",
            normalized_value="operative_order",
            quote_text=norm_text[:1200],
            field_explicitness=0.7,
        )

    return facts


def _build_support_search_text_for_page(
    *,
    doc_title: str,
    doc_family: str,
    page_family: str,
    page_role: str,
    page_text: str,
    fact_search_texts: list[str],
) -> str:
    """Build enriched search text for page embedding with support-fact context.

    Args:
        doc_title: Document title.
        doc_family: Document family classification.
        page_family: Page family classification.
        page_role: Semantic page role.
        page_text: Raw page text (truncated).
        fact_search_texts: Search texts from extracted support facts.

    Returns:
        Concatenated search text for page embedding.
    """
    return " | ".join(
        [doc_title, doc_family, page_family, page_role, page_text[:1500], *fact_search_texts[:12]]
    )


def _build_shadow_search_text(
    *,
    doc_title: str,
    doc_family: str,
    page_family: str,
    section_path: str,
    normalized_refs: list[str],
    party_names: list[str],
    court_names: list[str],
    law_titles: list[str],
    article_refs: list[str],
    case_numbers: list[str],
    cross_refs: list[str],
    anchors: list[str],
    contextual_header: str,
    chunk_text: str,
) -> str:
    parts = [
        doc_title,
        doc_family,
        page_family,
        section_path,
        " | ".join(normalized_refs),
        " | ".join(party_names),
        " | ".join(court_names),
        " | ".join(law_titles),
        " | ".join(article_refs),
        " | ".join(case_numbers),
        " | ".join(cross_refs),
        " | ".join(anchors),
        contextual_header,
        chunk_text,
    ]
    return "\n".join(part.strip() for part in parts if str(part or "").strip())


_ANCHOR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("schedule_anchor", re.compile(r"\bSCHEDULE\b")),
    ("enactment_anchor", re.compile(r"enactment\s+notice|comes?\s+into\s+force", re.IGNORECASE)),
    ("commencement_anchor", re.compile(r"\bcommencement\b", re.IGNORECASE)),
    ("administration_anchor", re.compile(r"administered\s+by", re.IGNORECASE)),
    ("operative_order_anchor", re.compile(r"IT IS HEREBY ORDERED", re.IGNORECASE)),
    ("costs_anchor", re.compile(r"(?:costs?|sum\s+of|ordered\s+to\s+pay).*(?:AED|USD|GBP)", re.IGNORECASE)),
]


def _enrich_chunks(
    chunks: list[Chunk],
    doc: ParsedDocument,
    first_page_text: str,
    doc_summary: str,
) -> list[Chunk]:
    """Enrich chunks with doc_family, page_family, normalized fields, amount roles."""
    doc_fam = _classify_doc_family(doc.title, doc_summary, first_page_text)
    norm_title = _normalize_identifiers(doc.title)
    total_pages = sum(1 for s in doc.sections if s.section_path.startswith("page:"))

    page_texts: dict[int, str] = {}
    for section in doc.sections:
        if section.section_path.startswith("page:"):
            try:
                pn = int(section.section_path.split(":", 1)[1])
                page_texts[pn] = section.text
            except (ValueError, IndexError):
                pass

    enriched: list[Chunk] = []
    for chunk in chunks:
        page_num = 0
        if chunk.section_path.startswith("page:"):
            with suppress(ValueError, IndexError):
                page_num = int(chunk.section_path.split(":", 1)[1])

        pg_text = page_texts.get(page_num, "")
        pg_fam = _classify_page_family(pg_text, page_num, total_pages)
        norm_refs = [_normalize_identifiers(c) for c in chunk.citations] if chunk.citations else []
        amt_roles = _extract_amount_roles(pg_text) if pg_text else []
        party_names = _extract_party_names(chunk.chunk_text)
        court_names = _extract_court_names(f"{doc.title}\n{pg_text}\n{chunk.chunk_text}")
        law_titles = _extract_law_titles(doc.title, pg_text, chunk.chunk_text)
        article_refs = _extract_article_refs(f"{chunk.chunk_text}\n{pg_text}")
        case_numbers = _extract_case_numbers(doc.title, pg_text, chunk.chunk_text)
        cross_refs = _extract_cross_refs(f"{chunk.chunk_text}\n{pg_text}")
        contextual_header = pg_text[:240]
        shadow_search_text = _build_shadow_search_text(
            doc_title=doc.title,
            doc_family=doc_fam,
            page_family=pg_fam,
            section_path=chunk.section_path,
            normalized_refs=norm_refs,
            party_names=party_names,
            court_names=court_names,
            law_titles=law_titles,
            article_refs=article_refs,
            case_numbers=case_numbers,
            cross_refs=cross_refs,
            anchors=list(chunk.anchors),
            contextual_header=contextual_header,
            chunk_text=chunk.chunk_text,
        )

        enriched.append(chunk.model_copy(update={
            "doc_family": doc_fam,
            "page_family": pg_fam,
            "normalized_title": norm_title,
            "normalized_refs": norm_refs,
            "amount_roles": amt_roles,
            "shadow_search_text": shadow_search_text,
            "party_names": party_names,
            "court_names": court_names,
            "law_titles": law_titles,
            "article_refs": article_refs,
            "case_numbers": case_numbers,
            "cross_refs": cross_refs,
        }))
    return enriched


def _emit_anchor_chunks(
    doc: ParsedDocument,
    doc_family: str,
    doc_summary: str,
) -> list[Chunk]:
    """Emit auxiliary anchor chunks for key page sections."""
    from rag_challenge.models import Chunk as ChunkModel

    anchors: list[ChunkModel] = []
    total_pages = sum(1 for s in doc.sections if s.section_path.startswith("page:"))

    for section in doc.sections:
        if not section.section_path.startswith("page:"):
            continue
        try:
            page_num = int(section.section_path.split(":", 1)[1])
        except (ValueError, IndexError):
            continue
        text = section.text.strip()
        if not text:
            continue

        for anchor_type, pattern in _ANCHOR_PATTERNS:
            if pattern.search(text):
                snippet = text[:2000]
                chunk_id_hash = hashlib.sha256(f"{doc.doc_id}:{page_num}:{anchor_type}".encode()).hexdigest()[:8]
                chunk_id = f"{doc.doc_id}:{page_num}:anchor:{chunk_id_hash}"
                anchors.append(ChunkModel(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    doc_type=doc.doc_type,
                    jurisdiction=getattr(doc, "jurisdiction", ""),
                    section_path=section.section_path,
                    chunk_text=snippet,
                    chunk_text_for_embedding=f"{doc.title}\n{snippet}",
                    doc_summary=doc_summary,
                    citations=[],
                    anchors=[],
                    token_count=len(snippet.split()),
                    chunk_type=anchor_type,
                    doc_family=doc_family,
                    page_family=_classify_page_family(text, page_num, total_pages),
                    normalized_title=_normalize_identifiers(doc.title),
                    shadow_search_text=_build_shadow_search_text(
                        doc_title=doc.title,
                        doc_family=doc_family,
                        page_family=_classify_page_family(text, page_num, total_pages),
                        section_path=section.section_path,
                        normalized_refs=[],
                        party_names=_extract_party_names(text),
                        court_names=_extract_court_names(f"{doc.title}\n{text}"),
                        law_titles=_extract_law_titles(doc.title, text),
                        article_refs=_extract_article_refs(text),
                        case_numbers=_extract_case_numbers(doc.title, text),
                        cross_refs=_extract_cross_refs(text),
                        anchors=[anchor_type],
                        contextual_header=text[:240],
                        chunk_text=snippet,
                    ),
                    party_names=_extract_party_names(text),
                    court_names=_extract_court_names(f"{doc.title}\n{text}"),
                    law_titles=_extract_law_titles(doc.title, text),
                    article_refs=_extract_article_refs(text),
                    case_numbers=_extract_case_numbers(doc.title, text),
                    cross_refs=_extract_cross_refs(text),
                ))

        is_last_pages = page_num >= total_pages - 1 and total_pages > 2
        if is_last_pages and doc_family in ("judgment", "order"):
            snippet = text[:2000]
            chunk_id_hash = hashlib.sha256(f"{doc.doc_id}:{page_num}:conclusion_anchor".encode()).hexdigest()[:8]
            chunk_id = f"{doc.doc_id}:{page_num}:anchor:{chunk_id_hash}"
            if not any(a.chunk_id == chunk_id for a in anchors):
                anchors.append(ChunkModel(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    doc_type=doc.doc_type,
                    jurisdiction=getattr(doc, "jurisdiction", ""),
                    section_path=section.section_path,
                    chunk_text=snippet,
                    chunk_text_for_embedding=f"{doc.title}\n{snippet}",
                    doc_summary=doc_summary,
                    citations=[],
                    anchors=[],
                    token_count=len(snippet.split()),
                    chunk_type="conclusion_anchor",
                    doc_family=doc_family,
                    page_family=_classify_page_family(text, page_num, total_pages),
                    normalized_title=_normalize_identifiers(doc.title),
                    shadow_search_text=_build_shadow_search_text(
                        doc_title=doc.title,
                        doc_family=doc_family,
                        page_family=_classify_page_family(text, page_num, total_pages),
                        section_path=section.section_path,
                        normalized_refs=[],
                        party_names=_extract_party_names(text),
                        court_names=_extract_court_names(f"{doc.title}\n{text}"),
                        law_titles=_extract_law_titles(doc.title, text),
                        article_refs=_extract_article_refs(text),
                        case_numbers=_extract_case_numbers(doc.title, text),
                        cross_refs=_extract_cross_refs(text),
                        anchors=["conclusion_anchor"],
                        contextual_header=text[:240],
                        chunk_text=snippet,
                    ),
                    party_names=_extract_party_names(text),
                    court_names=_extract_court_names(f"{doc.title}\n{text}"),
                    law_titles=_extract_law_titles(doc.title, text),
                    article_refs=_extract_article_refs(text),
                    case_numbers=_extract_case_numbers(doc.title, text),
                    cross_refs=_extract_cross_refs(text),
                ))

    return anchors


def _error_list_factory() -> list[str]:
    return []


def _manifest_docs_factory() -> dict[str, _ManifestEntry]:
    return {}


def _path_list_factory() -> list[Path]:
    return []


def _deleted_entries_factory() -> dict[str, _ManifestEntry]:
    return {}


def _fingerprint_map_factory() -> dict[str, _FileFingerprint]:
    return {}


@dataclass(frozen=True)
class _FileFingerprint:
    sha256: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class _ManifestEntry:
    sha256: str
    size_bytes: int
    mtime_ns: int
    doc_id: str
    status: str


@dataclass
class _IngestionManifest:
    schema_version: int = 1
    ingest_version: str = ""
    documents: dict[str, _ManifestEntry] = field(default_factory=_manifest_docs_factory)


@dataclass
class _IngestionPlan:
    changed_files: list[Path] = field(default_factory=_path_list_factory)
    unchanged_count: int = 0
    deleted_entries: dict[str, _ManifestEntry] = field(default_factory=_deleted_entries_factory)
    fingerprints: dict[str, _FileFingerprint] = field(default_factory=_fingerprint_map_factory)


@dataclass
class IngestionStats:
    docs_parsed: int = 0
    docs_failed: int = 0
    docs_skipped_unchanged: int = 0
    docs_deleted: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_upserted: int = 0
    sac_summaries_generated: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=_error_list_factory)


class IngestionPipeline:
    """Orchestrates parse -> chunk -> SAC -> embed -> Qdrant upsert."""

    def __init__(
        self,
        *,
        parser: DocumentParser | None = None,
        chunker: LegalChunker | None = None,
        sac: SACGenerator | None = None,
        embedder: EmbeddingClient | None = None,
        store: QdrantStore | None = None,
    ) -> None:
        self._settings = get_settings()
        self._parser = parser or DocumentParser()
        self._chunker = chunker or LegalChunker()
        self._embedder = embedder or EmbeddingClient()
        self._store = store or QdrantStore()
        self._sparse_encoder: BM25SparseEncoder | None = None
        qdrant_settings = getattr(self._settings, "qdrant", None)
        if qdrant_settings is not None and bool(getattr(qdrant_settings, "enable_sparse_bm25", True)):
            cache_dir = str(getattr(qdrant_settings, "fastembed_cache_dir", "")).strip() or None
            threads = self._coerce_int(getattr(qdrant_settings, "sparse_threads", None))
            try:
                self._sparse_encoder = BM25SparseEncoder(
                    model_name=str(getattr(qdrant_settings, "sparse_model", "Qdrant/bm25")),
                    cache_dir=cache_dir,
                    threads=threads,
                )
            except Exception:
                logger.warning("Failed initializing BM25 sparse encoder; continuing dense-only ingestion", exc_info=True)
                self._sparse_encoder = None

        self._owned_llm: LLMProvider | None = None
        if sac is None:
            self._owned_llm = LLMProvider()
            self._sac = SACGenerator(self._owned_llm)
        else:
            self._sac = sac

    async def run(self, doc_dir: Path) -> IngestionStats:
        started = time.perf_counter()
        stats = IngestionStats()
        doc_dir = Path(doc_dir)
        manifest_path = self._manifest_path_for(doc_dir)

        logger.info(
            "ingestion_start",
            extra={
                "doc_dir": str(doc_dir),
                "manifest_path": str(manifest_path),
                "ingest_version": self._settings.ingestion.ingest_version,
            },
        )
        await self._store.ensure_collection()
        await self._store.ensure_payload_indexes()
        if bool(getattr(self._settings.ingestion, "build_shadow_collection", True)):
            await self._store.ensure_shadow_collection()
            await self._store.ensure_shadow_payload_indexes()
        await self._store.ensure_page_collection()
        await self._store.ensure_page_payload_indexes()

        manifest = self._load_manifest(manifest_path)
        ingest_version = self._settings.ingestion.ingest_version
        if manifest.ingest_version and manifest.ingest_version != ingest_version:
            logger.info(
                "Ingest version changed %r -> %r; invalidating incremental manifest",
                manifest.ingest_version,
                ingest_version,
            )
            manifest = _IngestionManifest(ingest_version=ingest_version)
        elif not manifest.ingest_version:
            manifest.ingest_version = ingest_version

        source_files = self._list_source_files(doc_dir)
        plan = self._build_ingestion_plan(doc_dir, source_files, manifest)
        stats.docs_skipped_unchanged = plan.unchanged_count

        docs = self._parse_files(plan.changed_files)
        stats.docs_parsed = len(docs)
        logger.info(
            "ingestion_plan",
            extra={
                "files_total": len(source_files),
                "files_changed": len(plan.changed_files),
                "files_unchanged": plan.unchanged_count,
                "files_deleted": len(plan.deleted_entries),
                "docs_parsed": stats.docs_parsed,
            },
        )

        if docs:
            total_docs = len(docs)
            progress_started = time.perf_counter()
            for processed_docs, doc in enumerate(docs, start=1):
                doc_started = time.perf_counter()
                rel_path = self._relative_manifest_key(doc_dir, Path(doc.source_path))
                fingerprint = plan.fingerprints.get(rel_path)
                if fingerprint is None:
                    logger.warning(
                        "ingestion_doc_missing_fingerprint",
                        extra={"rel_path": rel_path, "doc_id": doc.doc_id},
                    )
                else:
                    try:
                        chunks = self._chunker.chunk_document(doc)
                        if not chunks:
                            logger.warning(
                                "ingestion_doc_empty_chunks",
                                extra={"doc_id": doc.doc_id, "doc_title": doc.title},
                            )
                            self._set_manifest_entry(
                                manifest=manifest,
                                rel_path=rel_path,
                                doc_id=doc.doc_id,
                                fingerprint=fingerprint,
                                status="empty",
                            )
                            self._save_manifest(manifest_path, manifest)
                        else:
                            summary = await self._sac.generate_doc_summary(doc)
                            stats.sac_summaries_generated += 1
                            augmented = self._sac.augment_chunks(chunks, summary)

                            first_pg = ""
                            for sec in doc.sections:
                                if sec.section_path.startswith("page:"):
                                    first_pg = sec.text
                                    break
                            augmented = _enrich_chunks(augmented, doc, first_pg, summary)
                            anchor_chunks = _emit_anchor_chunks(doc, _classify_doc_family(doc.title, summary, first_pg), summary)
                            if anchor_chunks:
                                augmented = list(augmented) + anchor_chunks
                                logger.info("Emitted %d anchor chunks for doc_id=%s", len(anchor_chunks), doc.doc_id)

                            stats.chunks_created += len(augmented)
                            logger.info(
                                "ingestion_doc_processed",
                                extra={
                                    "doc_id": doc.doc_id,
                                    "doc_title": doc.title,
                                    "chunks": len(augmented),
                                    "summary_chars": len(summary),
                                },
                            )

                            embedding_texts = [chunk.chunk_text_for_embedding for chunk in augmented]
                            vectors = await self._embedder.embed_documents(embedding_texts)
                            stats.chunks_embedded += len(vectors)

                            sparse_vectors = None
                            if self._sparse_encoder is not None:
                                try:
                                    sparse_vectors = self._sparse_encoder.encode_documents(
                                        [chunk.chunk_text_for_embedding for chunk in augmented]
                                    )
                                except Exception:
                                    logger.warning(
                                        "BM25 sparse encoding failed; falling back to dense-only for doc_id=%s",
                                        doc.doc_id,
                                        exc_info=True,
                                    )
                                    sparse_vectors = None

                            upserted = await self._store.upsert_chunks(
                                augmented,
                                vectors,
                                sparse_vectors=sparse_vectors,
                            ) if sparse_vectors is not None else await self._store.upsert_chunks(augmented, vectors)
                            stats.chunks_upserted += upserted

                            if bool(getattr(self._settings.ingestion, "build_shadow_collection", True)):
                                shadow_texts = [
                                    chunk.shadow_search_text or chunk.chunk_text_for_embedding
                                    for chunk in augmented
                                ]
                                shadow_vectors = await self._embedder.embed_documents(shadow_texts)
                                shadow_sparse_vectors = None
                                if self._sparse_encoder is not None:
                                    try:
                                        shadow_sparse_vectors = self._sparse_encoder.encode_documents(shadow_texts)
                                    except Exception:
                                        logger.warning(
                                            "BM25 sparse encoding failed for shadow chunks; falling back to dense-only for doc_id=%s",
                                            doc.doc_id,
                                            exc_info=True,
                                        )
                                if shadow_sparse_vectors is not None:
                                    await self._store.upsert_shadow_chunks(
                                        augmented,
                                        shadow_vectors,
                                        sparse_vectors=shadow_sparse_vectors,
                                    )
                                else:
                                    await self._store.upsert_shadow_chunks(augmented, shadow_vectors)

                            await self._upsert_pages_for_doc(doc, summary)
                            await self._cleanup_stale_versions([doc.doc_id])

                            self._set_manifest_entry(
                                manifest=manifest,
                                rel_path=rel_path,
                                doc_id=doc.doc_id,
                                fingerprint=fingerprint,
                                status="indexed",
                            )
                            self._save_manifest(manifest_path, manifest)
                    except Exception as exc:
                        stats.docs_failed += 1
                        error_text = f"{doc.doc_id} ({doc.title}): {exc}"
                        stats.errors.append(error_text)
                        logger.error(
                            "ingestion_doc_failed",
                            extra={"doc_id": doc.doc_id, "doc_title": doc.title},
                            exc_info=True,
                        )
                elapsed_progress_s = max(0.0, time.perf_counter() - progress_started)
                avg_doc_s = elapsed_progress_s / max(1, processed_docs)
                remaining_docs = max(0, total_docs - processed_docs)
                logger.info(
                    "ingestion_progress",
                    extra={
                        "done_docs": processed_docs,
                        "total_docs": total_docs,
                        "remaining_docs": remaining_docs,
                        "progress_pct": round((processed_docs / max(1, total_docs)) * 100.0, 2),
                        "doc_elapsed_s": round(max(0.0, time.perf_counter() - doc_started), 3),
                        "avg_doc_s": round(avg_doc_s, 3),
                        "eta_s": int(avg_doc_s * remaining_docs),
                        "docs_failed": stats.docs_failed,
                        "chunks_upserted_total": stats.chunks_upserted,
                    },
                )

        deleted_rel_paths = await self._delete_removed_documents(plan.deleted_entries, stats)
        for rel_path in deleted_rel_paths:
            manifest.documents.pop(rel_path, None)
        manifest.ingest_version = self._settings.ingestion.ingest_version
        self._save_manifest(manifest_path, manifest)

        stats.elapsed_s = time.perf_counter() - started
        logger.info(
            "ingestion_complete",
            extra={
                "docs_parsed": stats.docs_parsed,
                "docs_skipped_unchanged": stats.docs_skipped_unchanged,
                "docs_deleted": stats.docs_deleted,
                "docs_failed": stats.docs_failed,
                "chunks_created": stats.chunks_created,
                "chunks_upserted": stats.chunks_upserted,
                "elapsed_s": round(stats.elapsed_s, 3),
            },
        )
        return stats

    def _manifest_path_for(self, doc_dir: Path) -> Path:
        filename = self._settings.ingestion.manifest_filename
        configured_dir = str(self._settings.ingestion.manifest_dir).strip()
        if configured_dir:
            base_dir = Path(configured_dir).expanduser()
            if not base_dir.is_absolute():
                base_dir = Path.cwd() / base_dir
            base_dir.mkdir(parents=True, exist_ok=True)
            return base_dir / filename

        candidate_dir = doc_dir.parent
        if self._is_dir_writable(candidate_dir):
            return candidate_dir / filename

        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
        fallback_dir = cache_root / "rag_challenge"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir / filename

    @staticmethod
    def _is_dir_writable(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("1", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def _list_source_files(self, doc_dir: Path) -> list[Path]:
        list_supported_files_obj: object = getattr(self._parser, "list_supported_files", None)
        if callable(list_supported_files_obj):
            list_supported_files = cast("Callable[[Path], object]", list_supported_files_obj)
            listed_obj: object = list_supported_files(doc_dir)
            listed_type_name = type(listed_obj).__name__
            if isinstance(listed_obj, list):
                listed_type_name = "list"
                listed_paths: list[Path] = []
                for item in cast("list[object]", listed_obj):
                    if not isinstance(item, Path):
                        break
                    listed_paths.append(item)
                else:
                    return listed_paths
            logger.warning(
                "Parser.list_supported_files returned unexpected value (%s); falling back to parse_directory path",
                listed_type_name,
            )

        docs = self._parser.parse_directory(doc_dir)
        return [Path(doc.source_path) for doc in docs if doc.source_path]

    def _parse_files(self, files: list[Path]) -> list[ParsedDocument]:
        if not files:
            return []

        parsed: list[ParsedDocument] = []
        for file_path in files:
            try:
                parsed.append(self._parser.parse_file(file_path))
            except Exception:
                logger.warning("Failed to parse %s; skipping", file_path, exc_info=True)
        return parsed

    async def _upsert_pages_for_doc(self, doc: ParsedDocument, doc_summary: str) -> int:
        """Build page-level points from parsed document sections and upsert into page collection.

        Also extracts support facts for the grounding sidecar and upserts them into the
        support-fact collection when enabled.
        """
        from rag_challenge.models import PageMetadata, SupportFact

        first_pg = ""
        for sec in doc.sections:
            if sec.section_path.startswith("page:") and sec.text.strip():
                first_pg = sec.text
                break
        doc_fam = _classify_doc_family(doc.title, doc_summary, first_pg)
        total_pages = sum(1 for s in doc.sections if s.section_path.startswith("page:"))

        pages: list[PageMetadata] = []
        all_support_fact_dicts: list[dict[str, object]] = []

        for section in doc.sections:
            if not section.section_path.startswith("page:"):
                continue
            page_text = section.text.strip()
            if not page_text:
                continue
            try:
                page_num = int(section.section_path.split(":", 1)[1])
            except (ValueError, IndexError):
                continue
            page_id = f"{doc.doc_id}_{page_num}"
            pg_fam = _classify_page_family(page_text, page_num, total_pages)
            page_role = _page_role_for_text(page_text, page_num, total_pages)
            law_titles = _extract_law_titles(doc.title, page_text)
            article_refs = _extract_article_refs(page_text)
            case_numbers = _extract_case_numbers(doc.title, page_text)
            normalized_refs = _unique_normalized([doc.title, *law_titles, *article_refs, *case_numbers])
            amount_roles = _extract_amount_roles(page_text)
            linked_refs = _extract_page_links(page_text)

            # Extract support facts for this page
            page_facts = _extract_support_facts_for_page(
                doc_id=doc.doc_id,
                doc_title=doc.title,
                doc_type=doc.doc_type.value if hasattr(doc.doc_type, "value") else str(doc.doc_type),
                page_num=page_num,
                total_pages=total_pages,
                page_text=page_text,
                doc_family=doc_fam,
            )
            all_support_fact_dicts.extend(page_facts)

            # Build enriched search text for page embedding
            fact_search_texts = [str(f.get("search_text", "")) for f in page_facts[:12]]
            support_search_text = _build_support_search_text_for_page(
                doc_title=doc.title,
                doc_family=doc_fam,
                page_family=pg_fam,
                page_role=page_role,
                page_text=page_text,
                fact_search_texts=fact_search_texts,
            )

            pages.append(PageMetadata(
                page_id=page_id,
                doc_id=doc.doc_id,
                page_num=page_num,
                doc_title=doc.title,
                doc_type=doc.doc_type,
                jurisdiction=getattr(doc, "jurisdiction", ""),
                section_path=section.section_path,
                ingest_version=self._settings.ingestion.ingest_version,
                page_text=page_text,
                doc_summary=doc_summary,
                page_family=pg_fam,
                doc_family=doc_fam,
                normalized_refs=normalized_refs,
                law_titles=law_titles,
                article_refs=article_refs,
                case_numbers=case_numbers,
                page_role=page_role,
                support_search_text=support_search_text,
                amount_roles=amount_roles,
                linked_refs=linked_refs,
            ))

        if not pages:
            return 0

        page_texts = [p.page_text for p in pages]
        dense_vectors = await self._embedder.embed_documents(page_texts)

        sparse_vectors = None
        if self._sparse_encoder is not None:
            try:
                sparse_vectors = self._sparse_encoder.encode_documents(page_texts)
            except Exception:
                logger.warning("BM25 sparse encoding failed for pages of doc_id=%s", doc.doc_id, exc_info=True)

        upserted = await self._store.upsert_pages(pages, dense_vectors, sparse_vectors=sparse_vectors)
        logger.info("Upserted %d page points for doc_id=%s", upserted, doc.doc_id)

        # Upsert support facts into the support-fact collection
        if all_support_fact_dicts:
            try:
                await self._store.ensure_support_fact_collection()
                await self._store.ensure_support_fact_payload_indexes()
                support_facts = [SupportFact(**fd) for fd in all_support_fact_dicts]  # type: ignore[arg-type]
                fact_texts = [f.search_text for f in support_facts]
                fact_dense = await self._embedder.embed_documents(fact_texts)
                fact_sparse = None
                if self._sparse_encoder is not None:
                    try:
                        fact_sparse = self._sparse_encoder.encode_documents(fact_texts)
                    except Exception:
                        logger.warning("BM25 sparse encoding failed for support facts of doc_id=%s", doc.doc_id, exc_info=True)
                await self._store.upsert_support_facts(support_facts, fact_dense, sparse_vectors=fact_sparse)
                logger.info("Upserted %d support facts for doc_id=%s", len(support_facts), doc.doc_id)
            except Exception:
                logger.warning("Failed to upsert support facts for doc_id=%s", doc.doc_id, exc_info=True)

        return upserted

    def _build_ingestion_plan(
        self,
        doc_dir: Path,
        source_files: list[Path],
        manifest: _IngestionManifest,
    ) -> _IngestionPlan:
        plan = _IngestionPlan()

        for file_path in source_files:
            rel_key = self._relative_manifest_key(doc_dir, file_path)
            stat_result = file_path.stat()
            fingerprint = _FileFingerprint(
                sha256=self._hash_file(file_path),
                size_bytes=int(stat_result.st_size),
                mtime_ns=int(stat_result.st_mtime_ns),
            )
            plan.fingerprints[rel_key] = fingerprint

            previous = manifest.documents.get(rel_key)
            if previous is not None and previous.sha256 == fingerprint.sha256:
                plan.unchanged_count += 1
                continue
            plan.changed_files.append(file_path)

        for rel_key, entry in manifest.documents.items():
            if rel_key not in plan.fingerprints:
                plan.deleted_entries[rel_key] = entry

        return plan

    async def _delete_removed_documents(
        self,
        deleted_entries: dict[str, _ManifestEntry],
        stats: IngestionStats,
    ) -> list[str]:
        deleted_rel_paths: list[str] = []
        for rel_path, entry in deleted_entries.items():
            try:
                await self._store.delete_by_doc_id(entry.doc_id)
                await self._store.delete_pages_by_doc_id(entry.doc_id)
            except Exception as exc:
                stats.errors.append(f"{rel_path} ({entry.doc_id}): delete failed: {exc}")
                logger.warning(
                    "Failed deleting removed document rel_path=%s doc_id=%s",
                    rel_path,
                    entry.doc_id,
                    exc_info=True,
                )
                continue

            stats.docs_deleted += 1
            deleted_rel_paths.append(rel_path)

        return deleted_rel_paths

    def _update_manifest_entries(
        self,
        *,
        manifest: _IngestionManifest,
        doc_dir: Path,
        docs_by_rel_path: dict[str, ParsedDocument],
        successful_doc_ids: list[str],
        chunked_doc_ids: set[str],
        fingerprints: dict[str, _FileFingerprint],
        deleted_rel_paths: list[str],
    ) -> None:
        successful_doc_id_set = set(successful_doc_ids)

        for rel_path in deleted_rel_paths:
            manifest.documents.pop(rel_path, None)

        for rel_path, doc in docs_by_rel_path.items():
            if doc.doc_id not in successful_doc_id_set:
                continue

            fingerprint = fingerprints.get(rel_path)
            if fingerprint is None:
                logger.warning(
                    "No fingerprint found for parsed doc rel_path=%s doc_id=%s; skipping manifest update",
                    rel_path,
                    doc.doc_id,
                )
                continue

            manifest.documents[rel_path] = _ManifestEntry(
                sha256=fingerprint.sha256,
                size_bytes=fingerprint.size_bytes,
                mtime_ns=fingerprint.mtime_ns,
                doc_id=doc.doc_id,
                status="indexed" if doc.doc_id in chunked_doc_ids else "empty",
            )

        manifest.ingest_version = self._settings.ingestion.ingest_version

    def _set_manifest_entry(
        self,
        *,
        manifest: _IngestionManifest,
        rel_path: str,
        doc_id: str,
        fingerprint: _FileFingerprint,
        status: str,
    ) -> None:
        manifest.documents[rel_path] = _ManifestEntry(
            sha256=fingerprint.sha256,
            size_bytes=fingerprint.size_bytes,
            mtime_ns=fingerprint.mtime_ns,
            doc_id=doc_id,
            status=status,
        )
        manifest.ingest_version = self._settings.ingestion.ingest_version

    @staticmethod
    def _relative_manifest_key(doc_dir: Path, file_path: Path) -> str:
        return file_path.relative_to(doc_dir).as_posix()

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        chunk_size = max(1, int(self._settings.ingestion.manifest_hash_chunk_size_bytes))
        with path.open("rb") as file_obj:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _load_manifest(self, manifest_path: Path) -> _IngestionManifest:
        expected_schema_version = int(self._settings.ingestion.manifest_schema_version)
        if not manifest_path.exists():
            return _IngestionManifest(schema_version=expected_schema_version)

        try:
            raw_obj: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load ingestion manifest %s; starting fresh", manifest_path, exc_info=True)
            return _IngestionManifest(schema_version=expected_schema_version)

        if not isinstance(raw_obj, dict):
            logger.warning("Ingestion manifest is not an object: %s", manifest_path)
            return _IngestionManifest(schema_version=expected_schema_version)

        data = cast("dict[str, object]", raw_obj)
        schema_version = data.get("schema_version")
        if schema_version != expected_schema_version:
            logger.info(
                "Ignoring ingestion manifest %s with schema_version=%r (expected=%d)",
                manifest_path,
                schema_version,
                expected_schema_version,
            )
            return _IngestionManifest(schema_version=expected_schema_version)

        ingest_version_obj = data.get("ingest_version")
        documents_obj = data.get("documents")
        manifest = _IngestionManifest(
            schema_version=expected_schema_version,
            ingest_version=str(ingest_version_obj or ""),
        )
        if not isinstance(documents_obj, dict):
            return manifest

        documents_map = cast("dict[object, object]", documents_obj)
        for key_obj, entry_obj in documents_map.items():
            if not isinstance(key_obj, str) or not isinstance(entry_obj, dict):
                continue
            entry_dict = cast("dict[str, object]", entry_obj)
            sha256_obj = entry_dict.get("sha256")
            doc_id_obj = entry_dict.get("doc_id")
            if not isinstance(sha256_obj, str) or not isinstance(doc_id_obj, str):
                continue

            size_bytes = self._coerce_int(entry_dict.get("size_bytes", 0))
            mtime_ns = self._coerce_int(entry_dict.get("mtime_ns", 0))
            if size_bytes is None or mtime_ns is None:
                continue

            status_obj = entry_dict.get("status", "indexed")
            status = status_obj if isinstance(status_obj, str) else "indexed"
            manifest.documents[key_obj] = _ManifestEntry(
                sha256=sha256_obj,
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
                doc_id=doc_id_obj,
                status=status,
            )

        return manifest

    def _save_manifest(self, manifest_path: Path, manifest: _IngestionManifest) -> None:
        payload = {
            "schema_version": manifest.schema_version,
            "ingest_version": manifest.ingest_version,
            "documents": {
                rel_path: {
                    "sha256": entry.sha256,
                    "size_bytes": entry.size_bytes,
                    "mtime_ns": entry.mtime_ns,
                    "doc_id": entry.doc_id,
                    "status": entry.status,
                }
                for rel_path, entry in sorted(manifest.documents.items())
            },
        }

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(manifest_path)

    @staticmethod
    def _coerce_int(value: object) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    async def close(self) -> None:
        close_errors: list[Exception] = []

        for closer in (self._embedder.close, self._store.close):
            try:
                await closer()
            except Exception as exc:
                close_errors.append(exc)
                logger.warning("Failed to close resource: %s", exc, exc_info=True)

        if self._owned_llm is not None:
            try:
                await self._owned_llm.close()
            except Exception as exc:
                close_errors.append(exc)
                logger.warning("Failed to close LLM provider: %s", exc, exc_info=True)

        if close_errors:
            raise RuntimeError(f"Failed closing {len(close_errors)} ingestion resources")

    async def _chunk_and_augment_docs(
        self,
        docs: list[ParsedDocument],
        stats: IngestionStats,
    ) -> tuple[dict[str, list[Chunk]], list[str]]:
        semaphore = asyncio.Semaphore(max(1, int(self._settings.ingestion.sac_concurrency)))
        successful_doc_ids: list[str] = []

        async def process_doc(doc: ParsedDocument) -> tuple[str, list[Chunk]]:
            try:
                chunks = self._chunker.chunk_document(doc)
                if not chunks:
                    logger.warning("Doc %s produced 0 chunks", doc.doc_id)
                    successful_doc_ids.append(doc.doc_id)
                    return doc.doc_id, []

                async with semaphore:
                    summary = await self._sac.generate_doc_summary(doc)
                stats.sac_summaries_generated += 1

                augmented = self._sac.augment_chunks(chunks, summary)
                logger.info(
                    "Processed doc %s title=%r chunks=%d summary_chars=%d",
                    doc.doc_id,
                    doc.title,
                    len(augmented),
                    len(summary),
                )
                successful_doc_ids.append(doc.doc_id)
                return doc.doc_id, augmented
            except Exception as exc:
                stats.docs_failed += 1
                error_text = f"{doc.doc_id} ({doc.title}): {exc}"
                stats.errors.append(error_text)
                logger.error("Failed processing doc %s: %s", doc.doc_id, exc, exc_info=True)
                return doc.doc_id, []

        results = await asyncio.gather(*[process_doc(doc) for doc in docs])
        return {doc_id: chunks for doc_id, chunks in results if chunks}, successful_doc_ids

    async def _cleanup_stale_versions(self, doc_ids: Iterable[str]) -> None:
        delete_stale = getattr(self._store, "delete_stale_doc_versions", None)
        if delete_stale is None:
            logger.info("QdrantStore has no stale-version cleanup method; skipping")
            return

        ingest_version = self._settings.ingestion.ingest_version
        for doc_id in doc_ids:
            try:
                await delete_stale(doc_id, keep_ingest_version=ingest_version)
            except Exception:
                logger.warning(
                    "Failed stale-version cleanup for doc_id=%s ingest_version=%s",
                    doc_id,
                    ingest_version,
                    exc_info=True,
                )


async def _async_main() -> int:
    parser = argparse.ArgumentParser(description="rag_challenge ingestion pipeline")
    parser.add_argument("--doc-dir", type=Path, required=True, help="Directory containing source documents")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_format)

    if not args.doc_dir.is_dir():
        logger.error("Not a directory: %s", args.doc_dir)
        return 1

    pipeline = IngestionPipeline()
    try:
        stats = await pipeline.run(args.doc_dir)
        logger.info("Stats: %s", stats)
    finally:
        await pipeline.close()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
