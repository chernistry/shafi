# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed support helpers for the pipeline hot path."""

from __future__ import annotations

import re
from contextlib import suppress
from typing import TYPE_CHECKING

from rag_challenge.core.classifier import QueryClassifier
from rag_challenge.core.local_page_reranker import score_pages_from_chunk_scores, select_top_pages_per_doc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models import RankedChunk

    from .builder import RAGPipelineBuilder
from .constants import _DIFC_CASE_ID_RE, _LAW_NO_REF_RE
from .query_rules import (
    _extract_question_title_refs,
    _is_broad_enumeration_query,
    _is_case_issue_date_name_compare_query,
    _is_case_outcome_query,
    _is_common_judge_compare_query,
    _is_named_amendment_query,
    _is_named_commencement_query,
    _is_named_multi_title_lookup_query,
)

ARTICLE_REF_RE = re.compile(r"(?:Article|Section)\s+(\d+(?:\(\d+\))*(?:\([a-z]\))*)", re.IGNORECASE)
OUTCOME_QUERY_RE = re.compile(r"(?:ruling|order|outcome|result|decision|dismiss|grant|cost|award)", re.IGNORECASE)
ENACTMENT_QUERY_RE = re.compile(r"(?:come[s ]? into force|enacted|enactment|commencement)", re.IGNORECASE)
ADMIN_QUERY_RE = re.compile(r"administered\s+by", re.IGNORECASE)
SCHEDULE_QUERY_RE = re.compile(r"\b(?:schedule|annex|appendix)\b", re.IGNORECASE)
LAW_REF_RE = re.compile(r"\blaw\s+no\b", re.IGNORECASE)
CLAIM_VALUE_QUERY_RE = re.compile(r"claim\s+value|monetary\s+amount|how\s+much", re.IGNORECASE)
COSTS_QUERY_RE = re.compile(r"costs?\s+(?:awarded|ordered|assessed)|ordered\s+to\s+pay", re.IGNORECASE)
PENALTY_QUERY_RE = re.compile(r"\b(?:penalty|fine|prescribed\s+penalty)\b", re.IGNORECASE)
FAMILY_BOOST_MAP: dict[str, frozenset[str]] = {
    "enactment": frozenset({"enactment_like", "commencement_like", "citation_title_like"}),
    "administration": frozenset({"administration_like"}),
    "outcome": frozenset({"operative_order_like", "conclusion_like", "costs_like"}),
}
CITATION_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "in",
        "of",
        "to",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "that",
        "this",
        "it",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "has",
        "have",
        "had",
        "be",
        "been",
        "being",
        "not",
        "no",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "its",
        "as",
        "if",
        "but",
        "so",
        "when",
        "which",
    }
)


def explicit_page_reference_support_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    context_chunks: Sequence[RankedChunk],
) -> list[str]:
    """Resolve support chunks for an explicit page anchor.

    Args:
        pipeline: Pipeline builder facade.
        query: Raw user question.
        context_chunks: Ranked context chunks available for support shaping.

    Returns:
        Chunk IDs that best represent the explicitly requested page.
    """
    explicit_ref = QueryClassifier.extract_explicit_page_reference(query)
    if explicit_ref is None or explicit_ref.requested_page is None or explicit_ref.requested_page <= 0:
        return []

    requested_page = explicit_ref.requested_page
    chunk_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    resolved_doc_ids: set[str] = set()

    for ref in pipeline.support_question_refs(query)[:4]:
        title_chunk_id = pipeline.best_title_support_chunk_id(title=ref, context_chunks=context_chunks)
        if not title_chunk_id:
            continue
        resolved_doc_ids.update(pipeline.doc_ids_for_chunk_ids(chunk_ids=[title_chunk_id], context_chunks=context_chunks))

    for resolved_doc_id in sorted(resolved_doc_ids):
        chunk_id = pipeline.best_support_chunk_id_for_doc_page(
            doc_id=resolved_doc_id,
            page_num=requested_page,
            context_chunks=context_chunks,
        )
        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            chunk_ids.append(chunk_id)

    if chunk_ids:
        return chunk_ids

    fallback_chunk_id = pipeline.best_support_chunk_id_for_doc_page(
        doc_id=None,
        page_num=requested_page,
        context_chunks=context_chunks,
    )
    return [fallback_chunk_id] if fallback_chunk_id else []


def explicit_anchor_page_ids(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    context_chunks: Sequence[RankedChunk],
    preferred_chunk_ids: Sequence[str] = (),
) -> list[str]:
    """Return the allowed page IDs for an explicit page-anchor query.

    Args:
        pipeline: Pipeline builder facade.
        query: Raw user question.
        context_chunks: Ranked context chunks available for support shaping.
        preferred_chunk_ids: Candidate chunk IDs whose pages should be kept
            when they already satisfy the explicit anchor.

    Returns:
        Ordered page IDs allowed by the explicit anchor, or an empty list when
        the query has no explicit page constraint.
    """
    explicit_ref = QueryClassifier.extract_explicit_page_reference(query)
    if explicit_ref is None or explicit_ref.requested_page is None or explicit_ref.requested_page <= 0:
        return []

    requested_page = explicit_ref.requested_page
    context_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
    page_ids: list[str] = []
    seen_page_ids: set[str] = set()

    def _push_page_id(chunk_id: str) -> None:
        chunk = context_by_id.get(chunk_id)
        if chunk is None:
            return
        doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
        page_num = pipeline.page_num(str(getattr(chunk, "section_path", "") or ""))
        if not doc_id or page_num != requested_page:
            return
        page_id = f"{doc_id}_{page_num}"
        if page_id in seen_page_ids:
            return
        seen_page_ids.add(page_id)
        page_ids.append(page_id)

    for chunk_id in preferred_chunk_ids:
        _push_page_id(str(chunk_id).strip())

    if page_ids:
        return page_ids

    for chunk_id in explicit_page_reference_support_chunk_ids(
        pipeline,
        query=query,
        context_chunks=context_chunks,
    ):
        _push_page_id(chunk_id)

    return page_ids


def trim_to_article_page(
    *,
    question: str,
    answer_type: str,
    context_chunks: Sequence[RankedChunk],
    current_page_ids: list[str],
) -> list[str] | None:
    """For article-specific questions, trim used_page_ids to only the page
    containing the referenced article. Returns trimmed list or None if
    this question isn't article-specific or no match found."""
    from rag_challenge.submission.common import chunk_id_to_page_id

    q = (question or "").strip()
    m = ARTICLE_REF_RE.search(q)
    if not m:
        return None

    article_num = m.group(1)
    article_pattern = re.compile(
        r"(?:Article|Section)\s+" + re.escape(article_num) + r"\b",
        re.IGNORECASE,
    )

    best_page: str | None = None
    best_score = -1
    for chunk in context_chunks:
        text = chunk.text or ""
        if not article_pattern.search(text):
            continue
        page_id = chunk_id_to_page_id(chunk.chunk_id)
        if not page_id:
            continue
        score = len(article_pattern.findall(text))
        sp = chunk.section_path or ""
        if sp.startswith("page:"):
            with suppress(ValueError, IndexError):
                pn = int(sp.split(":", 1)[1])
                score += 100 if pn > 1 else 0
        if score > best_score:
            best_score = score
            best_page = page_id

    if not best_page:
        return None

    if best_page in set(current_page_ids):
        return [best_page]
    return None
def extract_citation_pages(
    *,
    question: str,
    answer: str,
    answer_type: str,
    context_chunks: Sequence[RankedChunk],
) -> list[str]:
    """Post-answer citation: return only pages that contain/support the answer."""
    from rag_challenge.submission.common import chunk_id_to_page_id

    answer_norm = re.sub(r"\s+", " ", (answer or "").strip()).lower()

    if answer_norm in ("null", "none", "") or answer_norm.startswith("there is no information"):
        return []

    stopwords = CITATION_STOPWORDS

    def _terms(text: str) -> set[str]:
        return {w for w in re.sub(r"[^\w]", " ", text.lower()).split()
                if w and w not in stopwords and len(w) > 2}

    search_patterns: list[str] = []
    if answer_type in ("name", "names"):
        search_patterns = [answer_norm]
    elif answer_type == "number":
        raw_digits = re.sub(r"[^\d.]", "", answer_norm)
        search_patterns = [answer_norm, raw_digits]
        if raw_digits and "." not in raw_digits:
            with suppress(ValueError, OverflowError):
                search_patterns.append(f"{int(raw_digits):,}".lower())
    elif answer_type == "date":
        search_patterns = [answer_norm]
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", answer_norm)
        if m:
            y, mo, d = m.groups()
            months = ["", "january", "february", "march", "april", "may", "june",
                      "july", "august", "september", "october", "november", "december"]
            try:
                search_patterns.append(f"{int(d)} {months[int(mo)]} {y}")
                search_patterns.append(f"{months[int(mo)]} {int(d)}, {y}")
            except (IndexError, ValueError):
                pass

    seen_pages: set[str] = set()
    page_scores: list[tuple[float, str]] = []
    answer_terms = _terms(answer_norm)
    question_terms = _terms(question)

    for chunk in context_chunks:
        page_id = chunk_id_to_page_id(chunk.chunk_id)
        if not page_id or page_id in seen_pages:
            continue
        seen_pages.add(page_id)

        chunk_lower = chunk.text.lower()
        score = 0.0

        if answer_type in ("name", "names", "number", "date"):
            for pat in search_patterns:
                if pat and pat in chunk_lower:
                    score = max(score, 1.0)
                    break
            if answer_type == "number" and score < 0.5:
                raw_answer_digits = re.sub(r"[^\d]", "", answer_norm)
                raw_chunk_digits = re.sub(r"[^\d]", "", chunk_lower)
                if raw_answer_digits and len(raw_answer_digits) >= 4 and raw_answer_digits in raw_chunk_digits:
                    score = max(score, 0.8)

        elif answer_type == "boolean":
            if question_terms:
                chunk_terms = _terms(chunk.text)
                overlap = len(question_terms & chunk_terms)
                score = overlap / len(question_terms) if question_terms else 0

        else:
            if answer_terms:
                chunk_terms = _terms(chunk.text)
                overlap = len(answer_terms & chunk_terms)
                score = overlap / len(answer_terms) if answer_terms else 0

        page_scores.append((score, page_id))

    page_scores.sort(key=lambda x: x[0], reverse=True)

    thresholds = {
        "boolean": (0.08, 2),
        "number": (0.5, 1),
        "date": (0.5, 1),
        "name": (0.5, 1),
        "names": (0.3, 2),
        "free_text": (0.06, 3),
    }
    threshold, max_pages = thresholds.get(answer_type, (0.1, 2))

    cited = [pid for sc, pid in page_scores if sc >= threshold][:max_pages]

    if not cited and page_scores:
        cited = [page_scores[0][1]]

    return cited
def boost_family_context_chunks(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    answer_type: str,
    context_chunks: list[RankedChunk],
) -> list[RankedChunk]:
    """Reorder (not filter) context chunks so family-relevant ones come first.

    Uses page_family metadata on RankedChunk to identify which chunks belong
    to question-relevant page families, then promotes them to the front of
    the context window where the LLM/strict-answerer pays most attention.
    """
    if not context_chunks:
        return context_chunks

    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    target_families: set[str] = set()

    if ENACTMENT_QUERY_RE.search(q):
        target_families |= FAMILY_BOOST_MAP["enactment"]
    if ADMIN_QUERY_RE.search(q):
        target_families |= FAMILY_BOOST_MAP["administration"]
    if OUTCOME_QUERY_RE.search(q):
        target_families |= FAMILY_BOOST_MAP["outcome"]

    target_amount_roles: set[str] = set()
    if CLAIM_VALUE_QUERY_RE.search(q):
        target_amount_roles.add("claim_amount")
    if COSTS_QUERY_RE.search(q):
        target_amount_roles.add("costs_awarded")
    if PENALTY_QUERY_RE.search(q):
        target_amount_roles.add("penalty")

    if not target_families and not target_amount_roles:
        return context_chunks

    boosted: list[RankedChunk] = []
    rest: list[RankedChunk] = []
    for chunk in context_chunks:
        pf = getattr(chunk, "page_family", "")
        amt = set(getattr(chunk, "amount_roles", []) or [])
        if (pf and pf in target_families) or (target_amount_roles and amt & target_amount_roles):
            boosted.append(chunk)
        else:
            rest.append(chunk)

    if not boosted:
        return context_chunks
    return boosted + rest
def enhance_page_recall(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    answer_type: str,
    context_chunks: Sequence[RankedChunk],
    current_used_ids: list[str],
) -> list[str]:
    """Add family-relevant chunk IDs to used_ids to boost page recall.

    Only ADDS chunks, never removes. Since G uses beta=2.5 (recall 6x more
    important than precision), extra relevant pages are cheap while missing
    necessary ones is catastrophic.
    """
    doc_ids: set[str] = set()
    for chunk in context_chunks:
        if chunk.doc_id:
            doc_ids.add(chunk.doc_id)

    if not doc_ids:
        return current_used_ids

    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    existing = set(current_used_ids)
    additions: list[str] = []

    page1_chunks: dict[str, str] = {}
    family_chunks: dict[str, list[tuple[str, str]]] = {}
    last_page_chunks: dict[str, list[tuple[str, int]]] = {}

    for chunk in context_chunks:
        if chunk.doc_id not in doc_ids:
            continue
        cid = chunk.chunk_id
        if cid in existing:
            continue

        page_num = 0
        if chunk.section_path.startswith("page:"):
            with suppress(ValueError, IndexError):
                page_num = int(chunk.section_path.split(":", 1)[1])

        if page_num == 1 and chunk.doc_id not in page1_chunks:
            page1_chunks[chunk.doc_id] = cid

        pf = getattr(chunk, "page_family", "")
        if pf:
            family_chunks.setdefault(pf, []).append((cid, chunk.doc_id))

        if page_num > 0:
            last_page_chunks.setdefault(chunk.doc_id, []).append((cid, page_num))

    is_law_ref = bool(LAW_REF_RE.search(q))
    is_outcome = bool(OUTCOME_QUERY_RE.search(q))
    is_enactment = bool(ENACTMENT_QUERY_RE.search(q))
    is_admin = bool(ADMIN_QUERY_RE.search(q))
    is_schedule = bool(SCHEDULE_QUERY_RE.search(q))
    is_compare = len(doc_ids) >= 2

    if is_law_ref or is_compare:
        for _doc_id, cid in page1_chunks.items():
            if cid not in existing:
                additions.append(cid)
                existing.add(cid)

    if is_outcome:
        for pf_key in ("operative_order_like", "costs_like"):
            for cid, _ in family_chunks.get(pf_key, []):
                if cid not in existing:
                    additions.append(cid)
                    existing.add(cid)
        for did in doc_ids:
            candidates = last_page_chunks.get(did, [])
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                for cid, _ in candidates[:2]:
                    if cid not in existing:
                        additions.append(cid)
                        existing.add(cid)

    if is_enactment:
        for pf_key in ("enactment_like", "commencement_like"):
            for cid, _ in family_chunks.get(pf_key, []):
                if cid not in existing:
                    additions.append(cid)
                    existing.add(cid)

    if is_admin:
        for cid, _ in family_chunks.get("administration_like", []):
            if cid not in existing:
                additions.append(cid)
                existing.add(cid)

    if is_schedule:
        for cid, _ in family_chunks.get("schedule_like", []):
            if cid not in existing:
                additions.append(cid)
                existing.add(cid)

    if answer_type in ("name", "names") or is_compare:
        for pf_key in ("cover_like",):
            for cid, _ in family_chunks.get(pf_key, []):
                if cid not in existing:
                    additions.append(cid)
                    existing.add(cid)

    if not additions:
        return current_used_ids

    return current_used_ids + additions
def rerank_support_pages_within_selected_docs(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    answer_type: str,
    context_chunks: Sequence[RankedChunk],
    used_ids: Sequence[str],
) -> list[str]:
    """Collapse chunk support into a tighter page-level posterior.

    Args:
        pipeline: Pipeline builder facade.
        query: Raw user question.
        answer_type: Normalized answer type.
        context_chunks: Ranked context chunks available for reranking.
        used_ids: Candidate support chunk IDs.

    Returns:
        A narrowed list of representative support chunk IDs.
    """
    ordered_used_ids: list[str] = []
    seen_used_ids: set[str] = set()
    for raw_chunk_id in used_ids:
        chunk_id = str(raw_chunk_id).strip()
        if not chunk_id or chunk_id in seen_used_ids:
            continue
        seen_used_ids.add(chunk_id)
        ordered_used_ids.append(chunk_id)
    if not ordered_used_ids or not context_chunks:
        return ordered_used_ids
    if QueryClassifier.extract_explicit_page_reference(query) is not None:
        return ordered_used_ids
    if _is_broad_enumeration_query(query):
        return ordered_used_ids

    q_lower = re.sub(r"\s+", " ", query).strip().lower()
    normalized_answer_type = answer_type.strip().lower()
    compare_like = normalized_answer_type in {"boolean", "name", "names", "date", "number"} and (
        len(_DIFC_CASE_ID_RE.findall(query or "")) >= 2
        or len(pipeline.support_question_refs(query)) >= 2
        or _is_case_issue_date_name_compare_query(query, answer_type=answer_type)
        or _is_common_judge_compare_query(query)
        or "same year" in q_lower
        or "same party" in q_lower
        or "appeared in both" in q_lower
        or "administ" in q_lower
    )
    strict_single_doc_like = normalized_answer_type in {"boolean", "name", "names", "date", "number"} and not compare_like
    metadata_like = (
        pipeline.is_named_metadata_support_query(query)
        or _is_named_multi_title_lookup_query(query)
        or _is_named_commencement_query(query)
        or _is_named_amendment_query(query)
    )
    metadata_page_family_query = pipeline.is_metadata_page_family_query(query)
    if pipeline.named_metadata_requires_support_union(query) or metadata_page_family_query:
        context_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
        used_pages = {
            pipeline.page_num(str(getattr(context_by_id.get(chunk_id), "section_path", "") or ""))
            for chunk_id in ordered_used_ids
            if chunk_id in context_by_id
        }
        if pipeline.named_metadata_requires_support_union(query) and len(used_pages) >= 2:
            return ordered_used_ids
        if metadata_page_family_query and len(used_pages) == 2:
            return ordered_used_ids
    if not (compare_like or metadata_like or strict_single_doc_like):
        return ordered_used_ids

    selected_doc_ids = pipeline.doc_ids_for_chunk_ids(chunk_ids=ordered_used_ids, context_chunks=context_chunks)
    if not selected_doc_ids:
        return ordered_used_ids

    doc_order = [
        doc_id
        for doc_id in (
            str(getattr(chunk, "doc_id", "") or "").strip()
            for chunk_id in ordered_used_ids
            for chunk in context_chunks
            if chunk.chunk_id == chunk_id
        )
        if doc_id
    ]
    page_one_bias = 0.0
    early_page_bias = 0.0
    if metadata_like:
        page_one_bias = 0.18
        early_page_bias = 0.04
    elif compare_like:
        page_one_bias = 0.12
    if compare_like and any(
        term in q_lower for term in ("judge", "party", "claimant", "respondent", "title", "citation title")
    ):
        page_one_bias = max(page_one_bias, 0.20)
        early_page_bias = 0.0
    elif compare_like and any(
        term in q_lower for term in ("date of issue", "issue date", "issued", "commencement", "effective date")
    ):
        page_one_bias = min(page_one_bias, 0.08)
        early_page_bias = max(early_page_bias, 0.18)
    scored_pages = score_pages_from_chunk_scores(
        chunks=context_chunks,
        doc_ids=selected_doc_ids,
        page_one_bias=page_one_bias,
        early_page_bias=early_page_bias,
    )
    if not scored_pages:
        return ordered_used_ids

    if strict_single_doc_like:
        best_page = scored_pages[0]
        doc_id, _, page_raw = best_page.page_id.rpartition("_")
        if not doc_id or not page_raw.isdigit():
            return ordered_used_ids
        chunk_id = pipeline.best_support_chunk_id_for_doc_page(
            doc_id=doc_id,
            page_num=int(page_raw),
            context_chunks=context_chunks,
        )
        return [chunk_id] if chunk_id else ordered_used_ids

    selected_pages = select_top_pages_per_doc(
        scored_pages=scored_pages,
        doc_order=doc_order,
        per_doc_pages=2 if metadata_page_family_query else 1,
    )
    if not selected_pages:
        return ordered_used_ids

    reranked_ids: list[str] = []
    for row in selected_pages:
        doc_id, _, page_raw = row.page_id.rpartition("_")
        if not doc_id or not page_raw.isdigit():
            continue
        chunk_id = pipeline.best_support_chunk_id_for_doc_page(
            doc_id=doc_id,
            page_num=int(page_raw),
            context_chunks=context_chunks,
        )
        if chunk_id and chunk_id not in reranked_ids:
            reranked_ids.append(chunk_id)

    return reranked_ids or ordered_used_ids
def is_named_metadata_support_query(pipeline: RAGPipelineBuilder, query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).casefold()
    if not q or _is_broad_enumeration_query(query):
        return False
    ref_count = len(_extract_question_title_refs(query)) + len(_LAW_NO_REF_RE.findall(query or ""))
    if ref_count < 1:
        return False
    return any(
        term in q
        for term in (
            "title",
            "citation title",
            "updated",
            "consolidated version",
            "published",
            "enact",
            "effective date",
            "commencement",
            "administ",
            "made by",
            "who made",
        )
    )
def named_metadata_requires_support_union(pipeline: RAGPipelineBuilder, query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).casefold()
    if not pipeline.is_named_metadata_support_query(query):
        return False

    atoms = 0
    if any(term in q for term in ("citation title", "what is the title")):
        atoms += 1
    if any(term in q for term in ("official law number", "official difc law number")):
        atoms += 1
    if any(term in q for term in ("updated", "consolidated version", "published")):
        atoms += 1
    if any(term in q for term in ("enact", "effective date", "commencement")):
        atoms += 1
    if "administ" in q:
        atoms += 1
    if "made by" in q or "who made" in q:
        atoms += 1

    if "and any regulations made under it" in q:
        return False

    multiple_named_refs = (
        " and " in q
        and (
            len(_LAW_NO_REF_RE.findall(query or "")) >= 2
            or len(_extract_question_title_refs(query)) >= 2
            or len(_DIFC_CASE_ID_RE.findall(query or "")) >= 2
        )
    )
    return atoms >= 2 or (atoms >= 1 and multiple_named_refs)
def is_metadata_page_family_query(pipeline: RAGPipelineBuilder, query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).casefold()
    if not pipeline.is_named_metadata_support_query(query):
        return False
    if pipeline.named_metadata_requires_support_union(query):
        return False
    return any(
        term in q
        for term in (
            "citation title",
            "official law number",
            "official difc law number",
            "who made",
            "made by",
            "date of enactment",
            "when was",
            "on what date",
            "commencement",
            "come into force",
            "who administers",
            "administered by",
        )
    )
def apply_support_shape_policy(
    pipeline: RAGPipelineBuilder,
    *,
    answer_type: str,
    answer: str,
    query: str,
    context_chunks: Sequence[RankedChunk],
    cited_ids: Sequence[str],
    support_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Shape support chunk IDs before late page selection.

    Args:
        pipeline: Pipeline builder facade.
        answer_type: Normalized answer type.
        answer: Final answer text.
        query: Raw user question.
        context_chunks: Ranked context chunks.
        cited_ids: Chunk IDs cited by the answerer.
        support_ids: Chunk IDs localized as support.

    Returns:
        The shaped support chunk IDs and diagnostic flags.
    """
    ordered_ids = list(
        dict.fromkeys(
            str(chunk_id).strip()
            for chunk_id in [*cited_ids, *support_ids]
            if str(chunk_id).strip()
        )
    )
    if not ordered_ids or not context_chunks:
        return ordered_ids, []

    kind = answer_type.strip().lower()
    q_lower = re.sub(r"\s+", " ", (query or "").strip()).casefold()
    extras: list[str] = []
    seen_ids = set(ordered_ids)
    flags: list[str] = []
    explicit_page_forced = False

    def _push(chunk_id: str) -> None:
        normalized = str(chunk_id).strip()
        if not normalized or normalized in seen_ids:
            return
        seen_ids.add(normalized)
        extras.append(normalized)

    explicit_page_ref = QueryClassifier.extract_explicit_page_reference(query)
    if explicit_page_ref is not None and explicit_page_ref.requested_page is not None:
        explicit_page_chunk_ids = pipeline.explicit_page_reference_support_chunk_ids(
            query=query,
            context_chunks=context_chunks,
        )
        for chunk_id in explicit_page_chunk_ids:
            before_len = len(extras)
            _push(chunk_id)
            if len(extras) > before_len:
                explicit_page_forced = True

    compare_refs = pipeline.paired_support_question_refs(query)
    if len(compare_refs) < 2:
        case_refs: list[str] = []
        seen_case_refs: set[str] = set()
        for prefix, number, year in _DIFC_CASE_ID_RE.findall(query or ""):
            ref = f"{prefix.upper()} {int(number):03d}/{year}"
            if ref not in seen_case_refs:
                seen_case_refs.add(ref)
                case_refs.append(ref)
        if len(case_refs) >= 2:
            compare_refs = case_refs
    compare_shape = len(compare_refs) >= 2 and kind in {"boolean", "name", "number", "date"} and (
        kind == "boolean"
        or _is_case_issue_date_name_compare_query(query, answer_type=answer_type)
        or "same year" in q_lower
        or "administ" in q_lower
        or "same party" in q_lower
        or "appeared in both" in q_lower
        or ("judge" in q_lower and "both" in q_lower)
    )
    compare_doc_ids: set[str] = set()
    if compare_shape:
        if kind == "boolean":
            for chunk_id in pipeline.localize_boolean_compare_support_chunk_ids(
                query=query,
                context_chunks=context_chunks,
            ):
                _push(chunk_id)
        for ref in compare_refs[:2]:
            title_chunk_id = pipeline.best_title_support_chunk_id(title=ref, context_chunks=context_chunks)
            if not title_chunk_id:
                continue
            _push(title_chunk_id)
            compare_doc_ids.update(
                pipeline.doc_ids_for_chunk_ids(chunk_ids=[title_chunk_id], context_chunks=context_chunks)
            )
        for chunk_id in pipeline.context_family_chunk_ids(
            doc_ids=compare_doc_ids,
            context_chunks=context_chunks,
        ):
            _push(chunk_id)

    metadata_query = pipeline.named_metadata_requires_support_union(query)
    metadata_page_family_query = pipeline.is_metadata_page_family_query(query)
    metadata_doc_ids: set[str] = set()
    if metadata_query or metadata_page_family_query:
        for ref in pipeline.support_question_refs(query)[:4]:
            title_chunk_id = pipeline.best_title_support_chunk_id(title=ref, context_chunks=context_chunks)
            if not title_chunk_id:
                continue
            _push(title_chunk_id)
            metadata_doc_ids.update(
                pipeline.doc_ids_for_chunk_ids(chunk_ids=[title_chunk_id], context_chunks=context_chunks)
            )
        if metadata_query:
            for chunk_id in pipeline.context_family_chunk_ids(
                doc_ids=metadata_doc_ids,
                context_chunks=context_chunks,
            ):
                _push(chunk_id)

    costs_query = kind == "free_text" and _is_case_outcome_query(query) and (
        "cost" in q_lower or "final ruling" in q_lower
    )
    if costs_query:
        for fragment in ("no order as to costs", "costs", "cost"):
            cost_chunk_id = pipeline.best_support_chunk_id(
                answer_type="free_text",
                query=query,
                fragment=fragment,
                context_chunks=context_chunks,
                allow_first_chunk_fallback=False,
            )
            if cost_chunk_id:
                _push(cost_chunk_id)

    shaped_ids = pipeline.expand_page_spanning_support_chunk_ids(
        chunk_ids=[*ordered_ids, *extras],
        context_chunks=context_chunks,
    )
    context_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
    explicit_page_pruned = False
    if explicit_page_ref is not None and explicit_page_ref.requested_page is not None:
        requested_page_ids = [
            chunk_id
            for chunk_id in shaped_ids
            if pipeline.page_num(str(getattr(context_by_id.get(chunk_id), "section_path", "") or ""))
            == explicit_page_ref.requested_page
        ]
        if requested_page_ids and len(requested_page_ids) < len(shaped_ids):
            shaped_ids = requested_page_ids
            explicit_page_pruned = True

    shaped_doc_ids = pipeline.doc_ids_for_chunk_ids(chunk_ids=shaped_ids, context_chunks=context_chunks)
    if compare_doc_ids and len(shaped_doc_ids.intersection(compare_doc_ids)) < min(2, len(compare_doc_ids)):
        flags.append("comparison_support_missing_side")
    if metadata_doc_ids and not shaped_doc_ids.intersection(metadata_doc_ids):
        flags.append("named_metadata_title_missing")
    if costs_query and not any(
        re.search(r"\bcosts?\b|\bno order as to costs\b", str(context_by_id[chunk_id].text or ""), re.IGNORECASE)
        for chunk_id in shaped_ids
        if chunk_id in context_by_id
    ):
        flags.append("outcome_costs_support_missing")
    if explicit_page_ref is not None and explicit_page_ref.requested_page is not None:
        explicit_page_present = any(
            pipeline.page_num(str(getattr(chunk, "section_path", "") or "")) == explicit_page_ref.requested_page
            for chunk in context_chunks
            if chunk.chunk_id in shaped_ids
        )
        if explicit_page_present and explicit_page_forced:
            flags.append("explicit_page_reference_forced")
        elif not explicit_page_present:
            flags.append("explicit_page_reference_missing")
        if explicit_page_present and explicit_page_pruned:
            flags.append("explicit_page_reference_pruned")

    return shaped_ids, flags
