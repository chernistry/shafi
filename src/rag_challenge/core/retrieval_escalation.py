"""Retrieval escalation policy for low-confidence queries.

When initial retrieval returns low-confidence results, this module applies
progressively broader search strategies and merges the results additively.

Escalation levels:
  1. Double prefetch limits (dense + sparse).
  2. Dense-only search (bypass BM25).
  3. Title-based filter search (extract document titles from query).

All escalation is purely additive — it can only add chunks, never remove.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_challenge.models import RetrievedChunk

if TYPE_CHECKING:
    from rag_challenge.core.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_TITLE_PATTERN = re.compile(
    r"(?:law|regulation|rule|order|decree|directive|resolution|decision|"
    r"circular|notice|proclamation|code|act|statute|amendment|schedule|"
    r"article|part|chapter|section)\s+"
    r"(?:no\.?\s*)?[\w./-]+(?:\s+of\s+\d{4})?",
    re.IGNORECASE,
)


@dataclass
class EscalationResult:
    """Result of an escalation run."""

    chunks: list[RetrievedChunk]
    levels_applied: list[int] = field(default_factory=list)
    chunks_added_per_level: dict[int, int] = field(default_factory=dict)


def extract_title_terms(query: str) -> list[str]:
    """Extract potential document title fragments from a query string.

    Args:
        query: The search query text.

    Returns:
        List of extracted title-like fragments.
    """
    matches = _TITLE_PATTERN.findall(query)
    return [m.strip() for m in matches if m.strip()]


def _merge_chunks(
    base: list[RetrievedChunk],
    new: list[RetrievedChunk],
) -> tuple[list[RetrievedChunk], int]:
    """Merge new chunks into base, keeping max score per chunk_id.

    Returns:
        Tuple of (merged list sorted by score desc, count of newly added chunk_ids).
    """
    by_id: dict[str, RetrievedChunk] = {}
    for chunk in base:
        existing = by_id.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            by_id[chunk.chunk_id] = chunk

    existing_ids = set(by_id.keys())

    for chunk in new:
        existing = by_id.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            by_id[chunk.chunk_id] = chunk

    added = len(set(by_id.keys()) - existing_ids)
    merged = sorted(by_id.values(), key=lambda c: (-c.score, c.chunk_id))
    return merged, added


class RetrieverEscalationPolicy:
    """Wraps a retriever with multi-level escalation for low-confidence queries.

    Usage:
        policy = RetrieverEscalationPolicy(retriever, threshold=0.5)
        result = await policy.retrieve_with_escalation(
            query=query,
            query_vector=vector,
            initial_chunks=chunks_from_initial_retrieval,
            top_rerank_score=0.3,  # below threshold → escalate
        )
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        threshold: float = 0.5,
        prefetch_dense: int = 120,
        prefetch_sparse: int = 120,
    ) -> None:
        self._retriever = retriever
        self._threshold = threshold
        self._prefetch_dense = prefetch_dense
        self._prefetch_sparse = prefetch_sparse

    @property
    def threshold(self) -> float:
        return self._threshold

    async def retrieve_with_escalation(
        self,
        *,
        query: str,
        query_vector: list[float] | None = None,
        initial_chunks: list[RetrievedChunk],
        top_rerank_score: float,
        doc_refs: list[str] | None = None,
    ) -> EscalationResult:
        """Run escalation levels if top_rerank_score is below threshold.

        Args:
            query: Original search query.
            query_vector: Pre-computed embedding vector (reused across levels).
            initial_chunks: Chunks from initial retrieval (never removed).
            top_rerank_score: Highest rerank score from initial results.
            doc_refs: Optional document reference filters.

        Returns:
            EscalationResult with merged chunks and escalation metadata.
        """
        if top_rerank_score >= self._threshold:
            return EscalationResult(chunks=list(initial_chunks))

        logger.info(
            "Escalating retrieval: top_rerank_score=%.3f < threshold=%.3f",
            top_rerank_score,
            self._threshold,
        )

        merged = list(initial_chunks)
        levels_applied: list[int] = []
        chunks_added: dict[int, int] = {}

        # Level 1: Double prefetch limits
        level1_chunks = await self._level1_double_prefetch(
            query=query,
            query_vector=query_vector,
            doc_refs=doc_refs,
        )
        merged, added = _merge_chunks(merged, level1_chunks)
        levels_applied.append(1)
        chunks_added[1] = added
        logger.info("Escalation L1 (double prefetch): +%d new chunks", added)

        # Level 2: Dense-only search (bypass BM25)
        level2_chunks = await self._level2_dense_only(
            query=query,
            query_vector=query_vector,
            doc_refs=doc_refs,
        )
        merged, added = _merge_chunks(merged, level2_chunks)
        levels_applied.append(2)
        chunks_added[2] = added
        logger.info("Escalation L2 (dense-only): +%d new chunks", added)

        # Level 3: Title-based filter search
        title_terms = extract_title_terms(query)
        if title_terms:
            level3_chunks = await self._level3_title_filter(
                query=query,
                query_vector=query_vector,
                title_terms=title_terms,
            )
            merged, added = _merge_chunks(merged, level3_chunks)
            levels_applied.append(3)
            chunks_added[3] = added
            logger.info("Escalation L3 (title filter %r): +%d new chunks", title_terms, added)

        return EscalationResult(
            chunks=merged,
            levels_applied=levels_applied,
            chunks_added_per_level=chunks_added,
        )

    async def _level1_double_prefetch(
        self,
        *,
        query: str,
        query_vector: list[float] | None,
        doc_refs: list[str] | None,
    ) -> list[RetrievedChunk]:
        """Level 1: Retrieve with doubled prefetch limits."""
        return await self._retriever.retrieve(
            query,
            query_vector=query_vector,
            prefetch_dense=self._prefetch_dense,
            prefetch_sparse=self._prefetch_sparse,
            doc_refs=doc_refs,
        )

    async def _level2_dense_only(
        self,
        *,
        query: str,
        query_vector: list[float] | None,
        doc_refs: list[str] | None,
    ) -> list[RetrievedChunk]:
        """Level 2: Dense-only retrieval (bypass BM25 sparse index)."""
        return await self._retriever.retrieve(
            query,
            query_vector=query_vector,
            prefetch_dense=self._prefetch_dense,
            doc_refs=doc_refs,
            sparse_only=False,
        )

    async def _level3_title_filter(
        self,
        *,
        query: str,
        query_vector: list[float] | None,
        title_terms: list[str],
    ) -> list[RetrievedChunk]:
        """Level 3: Search using extracted title terms as the query."""
        all_chunks: list[RetrievedChunk] = []
        for title in title_terms:
            chunks = await self._retriever.retrieve(
                title,
                query_vector=query_vector,
            )
            all_chunks.extend(chunks)
        return all_chunks
