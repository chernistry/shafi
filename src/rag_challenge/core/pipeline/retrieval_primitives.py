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


def augment_query_for_sparse_retrieval(query: str) -> str:
    """Add BM25-friendly variants for Article references (PDFs often render as '11. (1)' not 'Article 11(1)')."""
    raw = (query or "").strip()
    if not raw:
        return ""
    out = raw
    for match in re.finditer(r"\bArticle\s+(\d+)\s*\(\s*([^)]+?)\s*\)", raw, flags=re.IGNORECASE):
        num = match.group(1).strip()
        sub = match.group(2).strip()
        # Common PDF renderings.
        out += f" {num}({sub}) {num} ({sub}) {num}. ({sub})"
    return re.sub(r"\s+", " ", out).strip()
def extract_provision_refs(query: str) -> list[str]:
    raw = (query or "").strip()
    if not raw:
        return []
    refs: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"\b(?:Article|Section|Schedule|Part|Chapter)\s+\d+(?:\s*\(\s*[^)]+\s*\))?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(raw):
        normalized = re.sub(r"\s+", " ", match.group(0)).strip()
        normalized = re.sub(
            r"\b(article|section|schedule|part|chapter)\b",
            lambda m: m.group(1).title(),
            normalized,
            count=1,
        )
        normalized = re.sub(r"\s*\(\s*", "(", normalized)
        normalized = re.sub(r"\s*\)\s*", ")", normalized)
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
        provision_terms.append(provision_ref)
        if provision_ref.lower().startswith("article "):
            short = provision_ref[8:].strip()
            if short:
                provision_terms.append(short)
                provision_terms.append(re.sub(r"\(\s*", " (", short))

    targeted = " ".join([ref, *provision_terms, base_query]).strip()
    return re.sub(r"\s+", " ", targeted).strip()
def seed_terms_for_query(query: str) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []
    q_lower = q.lower()
    terms: list[str] = []

    if "enact" in q_lower:
        terms += ["enactment notice", "hereby enact", "ruler of dubai", "enacted"]
    if "come into force" in q_lower or "commencement" in q_lower:
        terms += ["come into force", "commencement", "commence"]
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

    # Article references: add both "article 11(1)" and "11 (1)".
    for match in re.finditer(r"\bArticle\s+\d+(?:\([^)]+\))?", q, flags=re.IGNORECASE):
        key = re.sub(r"\s+", " ", match.group(0)).strip().lower()
        if not key:
            continue
        terms.append(key)
        short = key.replace("article ", "").strip()
        if short:
            terms.append(short)
            terms.append(re.sub(r"\(\s*", " (", short))

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

    ranked = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)
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
