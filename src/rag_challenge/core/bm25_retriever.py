"""BM25 sparse retrieval for exact legal citation matching."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import bm25s
import numpy as np

from rag_challenge.core.legal_citation_parser import extract_citations, has_legal_citations

if TYPE_CHECKING:
    from rag_challenge.models import RetrievedChunk

logger = logging.getLogger(__name__)


def load_alias_map(path: str | Path) -> dict[str, list[str]]:
    """Load legal alias clusters and build a case-insensitive lookup.

    Each cluster maps any variant (lowercased) to all other variants in the
    cluster.  Used by BM25Retriever to expand queries with known law synonyms.

    Args:
        path: Path to ``data/legal_aliases.json``.

    Returns:
        dict mapping lowercased variant to list of other cluster variants.
        Returns empty dict if file is missing or malformed.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Legal alias map not found at %s; alias expansion disabled", path)
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed loading alias map from %s: %s", path, exc)
        return {}

    clusters: list[dict[str, object]] = data.get("clusters", [])
    lookup: dict[str, list[str]] = {}
    for cluster in clusters:
        variants: list[str] = [v for v in cluster.get("variants", []) if isinstance(v, str) and v.strip()]
        if len(variants) < 2:
            continue
        for variant in variants:
            key = variant.casefold()
            others = [v for v in variants if v.casefold() != key]
            lookup[key] = others
    logger.debug("Loaded alias map: %d variant keys from %d clusters", len(lookup), len(clusters))
    return lookup


def expand_query_with_aliases(query: str, alias_map: dict[str, list[str]]) -> str:
    """Append alias expansions to a BM25 query string.

    For each known alias variant found in the query (case-insensitive), appends
    the other variants in the same cluster to the query.  This lets BM25 match
    chunks that use a different alias form (e.g., query says "Law No. 2 of 2019"
    but chunks say "Employment Law").

    Args:
        query: Raw query string.
        alias_map: Lookup from lowercased variant to list of sibling variants.

    Returns:
        Query string with alias expansions appended (or original if no matches).
    """
    if not alias_map:
        return query
    query_lower = query.casefold()
    appended: set[str] = set()
    for key, others in alias_map.items():
        # Use word-boundary-aware substring check to avoid partial matches
        if re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", query_lower):
            for variant in others:
                v_lower = variant.casefold()
                if v_lower not in query_lower and v_lower not in appended:
                    appended.add(v_lower)
    if not appended:
        return query
    # Sort for determinism
    extras = sorted(appended)
    expanded = query + " " + " ".join(extras)
    logger.debug("BM25 alias expansion: +%d terms", len(extras))
    return expanded


def rrf_merge(
    dense_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
    bm25_weight: float = 0.3,
) -> list[str]:
    """Merge dense and BM25 results via weighted Reciprocal Rank Fusion.

    Args:
        dense_results: List of (chunk_id, score) from dense retrieval
        bm25_results: List of (chunk_id, score) from BM25 retrieval
        k: RRF constant (default 60 from literature)
        bm25_weight: Weight for BM25 scores (dense gets 1 - bm25_weight)

    Returns:
        List of chunk_ids sorted by RRF score (highest first)
    """
    scores: dict[str, float] = {}
    dense_weight = 1.0 - bm25_weight

    # Add dense scores
    for rank, (chunk_id, _) in enumerate(dense_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + dense_weight / (rank + k)

    # Add BM25 scores
    for rank, (chunk_id, _) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + bm25_weight / (rank + k)

    # Sort by RRF score descending
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


class BM25Retriever:
    """BM25 sparse retriever for legal text with citation-aware tokenization."""

    def __init__(
        self,
        index_dir: Path | None = None,
        alias_map: dict[str, list[str]] | None = None,
    ) -> None:
        self.retriever: bm25s.BM25 | None = None
        self.chunk_ids: list[str] = []
        self.chunk_id_to_idx: dict[str, int] = {}  # Fast lookup
        self.index_dir = index_dir or Path("data/bm25_index")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # Optional law-alias lookup for query expansion (keyed by lowercased variant).
        self._alias_map: dict[str, list[str]] = alias_map or {}

    def build_index(self, chunks: list[RetrievedChunk]) -> None:
        """Build BM25 index from chunks."""
        if not chunks:
            logger.warning("No chunks provided for BM25 indexing")
            return

        # Filter out empty chunks
        valid_chunks = [c for c in chunks if c.text and c.text.strip()]
        if not valid_chunks:
            logger.warning("No valid chunks with text")
            return

        # Extract texts and IDs
        corpus_texts = [c.text for c in valid_chunks]
        self.chunk_ids = [c.chunk_id for c in valid_chunks]
        self.chunk_id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

        # Tokenize entire corpus at once
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="english", show_progress=True)

        # Build index
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        logger.info("Built BM25 index with %d chunks", len(self.chunk_ids))

    def save(self) -> None:
        """Serialize BM25 index to disk."""
        if self.retriever is None:
            logger.warning("No BM25 index to save")
            return
        self.retriever.save(str(self.index_dir / "bm25_index"))
        np.save(str(self.index_dir / "chunk_ids.npy"), np.array(self.chunk_ids, dtype=object))
        logger.info("Saved BM25 index to %s", self.index_dir)

    def load(self) -> None:
        """Load BM25 index from disk."""
        index_path = self.index_dir / "bm25_index"
        chunk_ids_path = self.index_dir / "chunk_ids.npy"

        if not index_path.exists() or not chunk_ids_path.exists():
            logger.warning("BM25 index not found at %s", self.index_dir)
            return

        self.retriever = bm25s.BM25.load(str(index_path), mmap=True)
        self.chunk_ids = np.load(str(chunk_ids_path), allow_pickle=True).tolist()
        self.chunk_id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        logger.info("Loaded BM25 index with %d chunks from %s", len(self.chunk_ids), self.index_dir)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search BM25 index and return (chunk_id, score) pairs.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (chunk_id, score) tuples sorted by score descending
        """
        if self.retriever is None:
            logger.warning("BM25 index not built or loaded")
            return []

        # Log if query contains legal citations (BM25 will naturally boost exact matches)
        if has_legal_citations(query):
            citations = extract_citations(query)
            logger.debug("Query contains %d legal citations: %s", len(citations), citations[:3])

        # Optionally expand query with legal alias synonyms
        effective_query = expand_query_with_aliases(query, self._alias_map) if self._alias_map else query

        # Tokenize query
        query_tokens = bm25s.tokenize(effective_query, stopwords="english", show_progress=False)

        # Retrieve results
        results, scores = self.retriever.retrieve(
            query_tokens,
            k=min(top_k, len(self.chunk_ids)),
        )

        # Return (chunk_id, score) pairs
        return [(self.chunk_ids[i], float(scores[0][j])) for j, i in enumerate(results[0])]

    def rerank_chunks(
        self,
        chunks: list[RetrievedChunk],
        query: str,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """Rerank chunks using RRF fusion of dense scores and BM25 scores.

        Args:
            chunks: Chunks from dense retrieval (already scored)
            query: Query string
            bm25_weight: Weight for BM25 in RRF (dense gets 1 - bm25_weight)
            rrf_k: RRF constant

        Returns:
            Reranked chunks (same objects, new order)
        """
        if self.retriever is None or not chunks:
            return chunks

        # Get dense results as (chunk_id, score) pairs
        dense_results = [(c.chunk_id, c.score) for c in chunks]

        # Get BM25 results (top 2x to ensure coverage)
        bm25_results = self.search(query, top_k=len(chunks) * 2)

        # Merge via RRF
        merged_ids = rrf_merge(dense_results, bm25_results, k=rrf_k, bm25_weight=bm25_weight)

        # Reorder chunks by RRF ranking
        chunk_map: dict[str, RetrievedChunk] = {c.chunk_id: c for c in chunks}
        reranked: list[RetrievedChunk] = []
        for chunk_id in merged_ids:
            if chunk_id in chunk_map:
                reranked.append(chunk_map[chunk_id])

        # Add any chunks that weren't in BM25 results (shouldn't happen, but safety)
        seen = {c.chunk_id for c in reranked}
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                reranked.append(chunk)

        return reranked
