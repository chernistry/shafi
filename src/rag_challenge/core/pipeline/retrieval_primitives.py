# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Typed retrieval helpers for the pipeline hot path."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models import RetrievedChunk

    from .builder import RAGPipelineBuilder


logger = logging.getLogger(__name__)

_PROVISION_KIND_PATTERN = r"(?:Article|Section|Schedule|Part|Chapter)"
_PROVISION_REF_PATTERN = re.compile(
    rf"\b{_PROVISION_KIND_PATTERN}\s+(?:\d+[A-Za-z]?|[IVXLC]+|[A-Za-z])(?:\s*\(\s*[^)]+\s*\))*",
    re.IGNORECASE,
)


def _normalize_provision_ref(ref: str) -> str:
    normalized = re.sub(r"\s+", " ", str(ref or "")).strip()
    if not normalized:
        return ""
    normalized = re.sub(
        rf"\b({_PROVISION_KIND_PATTERN})\b",
        lambda match: match.group(1).title(),
        normalized,
        count=1,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s*\(\s*", "(", normalized)
    normalized = re.sub(r"\s*\)\s*", ")", normalized)
    return normalized


def _provision_sparse_variants(ref: str) -> list[str]:
    normalized = _normalize_provision_ref(ref)
    if not normalized:
        return []
    variants: list[str] = [normalized]
    if " " not in normalized:
        return variants
    _, body = normalized.split(" ", 1)
    body = body.strip()
    if not body:
        return variants

    variants.append(body)
    spaced_body = re.sub(r"^([0-9A-Za-zIVXLC]+)\(", r"\1 (", body)
    if spaced_body != body:
        variants.append(spaced_body)
    spaced_parens = re.sub(r"\)\(", ") (", spaced_body)
    if spaced_parens != spaced_body:
        variants.append(spaced_parens)
    if "(" in body:
        head = body.split("(", 1)[0].strip()
        if head:
            variants.append(head)
            if head[0].isdigit():
                variants.append(f"{head}.")
    elif body[0].isdigit():
        variants.append(f"{body}.")
    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = re.sub(r"\s+", " ", variant).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def augment_query_for_sparse_retrieval(query: str) -> str:
    """Add BM25-friendly variants for provision references.

    PDFs often render legal drafting references as bare section/article numbers
    or separated parentheses rather than the canonical drafting form.
    """
    raw = (query or "").strip()
    if not raw:
        return ""
    out = raw
    for match in _PROVISION_REF_PATTERN.finditer(raw):
        ref = _normalize_provision_ref(match.group(0))
        if not ref:
            continue
        for variant in _provision_sparse_variants(ref)[1:]:
            out += f" {variant}"
    return re.sub(r"\s+", " ", out).strip()


def extract_provision_refs(query: str) -> list[str]:
    raw = (query or "").strip()
    if not raw:
        return []
    refs: list[str] = []
    seen: set[str] = set()
    for match in _PROVISION_REF_PATTERN.finditer(raw):
        normalized = _normalize_provision_ref(match.group(0))
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        refs.append(normalized)
    return refs


def targeted_provision_ref_query(
    pipeline: RAGPipelineBuilder,
    *,
    query: str,
    ref: str,
    refs: Sequence[str],
) -> str:
    base_query = query or ""
    for other_ref in refs:
        other_clean = str(other_ref).strip()
        if not other_clean or other_clean.casefold() == ref.casefold():
            continue
        base_query = re.sub(re.escape(other_clean), " ", base_query, flags=re.IGNORECASE)
    base_query = re.sub(r"\s+", " ", base_query).strip()

    provision_terms: list[str] = []
    for provision_ref in pipeline.extract_provision_refs(query)[:3]:
        provision_terms.extend(_provision_sparse_variants(provision_ref))

    targeted = " ".join([ref, *provision_terms, base_query]).strip()
    return re.sub(r"\s+", " ", targeted).strip()


def seed_terms_for_query(query: str) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []
    q_lower = q.lower()
    terms: list[str] = []

    if "enact" in q_lower:
        terms += [
            "enactment notice",
            "hereby enact",
            "ruler of dubai",
            "enacted",
        ]
    if "come into force" in q_lower or "commencement" in q_lower:
        terms += [
            "come into force",
            "commencement",
            "commence",
        ]
    if "administ" in q_lower:
        terms += ["administer", "administered", "administration", "commissioner", "relevant authority"]
    if "claim value" in q_lower or "claim amount" in q_lower or "amount claimed" in q_lower:
        terms += ["claim value", "claim amount", "amount claimed", "value of the claim"]
    if "financial services" in q_lower:
        terms += ["financial services", "undertake", "shall not", "may not", "prohibit", "prohibited"]
    if "liable" in q_lower or "liability" in q_lower:
        terms += ["can be held liable", "cannot be held liable", "liable", "liability", "bad faith", "does not apply"]
    if "delegate" in q_lower or "delegat" in q_lower:
        terms += ["delegate", "delegat", "approval"]
    if "restriction" in q_lower and "transfer" in q_lower:
        terms += ["restriction", "ineffective", "actual knowledge", "uncertificated", "notified"]
    if any(token in q_lower for token in ("article ", "section ", "schedule ", "part ", "chapter ")):
        terms += [
            "provision",
            "subsection",
            "sub-paragraph",
            "subparagraph",
            "paragraph",
            "annex",
            "appendix",
        ]

    # Provision references: add drafting-friendly sparse variants.
    for match in _PROVISION_REF_PATTERN.finditer(q):
        key = _normalize_provision_ref(match.group(0)).lower()
        if not key:
            continue
        terms.extend(_provision_sparse_variants(key))

    # Dedupe, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        t = term.strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def dedupe_chunk_ids(chunk_ids: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in chunk_ids:
        chunk_id = str(raw).strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append(chunk_id)
    return out


def merge_retrieved_preserving_chunk_ids(
    pipeline: RAGPipelineBuilder,
    *,
    retrieved: Sequence[RetrievedChunk],
    extra: Sequence[RetrievedChunk],
    must_keep_chunk_ids: Sequence[str],
    limit: int,
) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}
    for chunk in [*retrieved, *extra]:
        existing = merged.get(chunk.chunk_id)
        if existing is None or float(chunk.score) > float(existing.score):
            merged[chunk.chunk_id] = chunk

    ranked = sorted(merged.values(), key=lambda chunk: (-chunk.score, chunk.doc_id, chunk.chunk_id))
    keep_ids = pipeline.dedupe_chunk_ids(must_keep_chunk_ids)
    keep_set = set(keep_ids)

    ordered: list[RetrievedChunk] = []
    for chunk_id in keep_ids:
        chunk = merged.get(chunk_id)
        if chunk is not None:
            ordered.append(chunk)
    for chunk in ranked:
        if chunk.chunk_id in keep_set:
            continue
        ordered.append(chunk)
    return ordered[: max(0, int(limit))]


def section_page_num(section_path: str) -> int:
    m = re.search(r"page:(\d+)", section_path or "", flags=re.IGNORECASE)
    if m is None:
        return 10_000
    try:
        return int(m.group(1))
    except ValueError:
        return 10_000
