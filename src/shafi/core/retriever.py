from __future__ import annotations

import asyncio
import logging
import re
from contextlib import suppress
from copy import deepcopy
from typing import TYPE_CHECKING, cast

from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from shafi.config import get_settings
from shafi.core.bm25_retriever import BM25Retriever, load_alias_map
from shafi.core.circuit_breaker import CircuitBreaker
from shafi.core.citation_graph import CitationGraphExpander
from shafi.core.classifier import QueryClassifier
from shafi.core.entity_registry import EntityRegistry
from shafi.core.hyde import generate_hypothetical_document
from shafi.core.retriever_filters import (
    build_filter,
    build_sparse_query,
    coerce_float,
    coerce_int,
    coerce_str_list,
    doc_title_filter_variants,
    expand_doc_ref_variants,
)
from shafi.core.sparse_bm25 import BM25SparseEncoder
from shafi.models import (
    BridgeFactType,
    DocType,
    RetrievedBridgeFact,
    RetrievedChunk,
    RetrievedPage,
    RetrievedSegment,
    SegmentType,
)

if TYPE_CHECKING:
    from shafi.core.embedding import EmbeddingClient
    from shafi.core.qdrant import QdrantStore

logger = logging.getLogger(__name__)

_ANCHOR_CHUNK_TYPES = {
    "schedule_anchor",
    "enactment_anchor",
    "commencement_anchor",
    "administration_anchor",
    "operative_order_anchor",
    "costs_anchor",
    "conclusion_anchor",
}
_ANCHOR_QUERY_RE = re.compile(
    r"\b(article|section|schedule|page|title|cover|header|caption|commencement|enactment|administer(?:ed|ing)?|costs?)\b",
    re.IGNORECASE,
)
_SEGMENT_QUERY_RE = re.compile(
    r"\b(article|section|schedule|definition|definitions|caption|header|issued by|operative order|part|chapter)\b",
    re.IGNORECASE,
)
_BRIDGE_QUERY_RE = re.compile(
    r"\b(compare|common|same|judge|party|claimant|respondent|authority|issued by|date|title|law number|which case|which law)\b",
    re.IGNORECASE,
)


def _retrieved_chunk_sort_key(chunk: RetrievedChunk) -> tuple[float, str, str, str]:
    """Return a deterministic sort key for merged retrieval candidates.

    Args:
        chunk: Retrieved chunk candidate to order.

    Returns:
        tuple[float, str, str, str]: Sort key that keeps the highest retrieval
        score first and stabilizes equal-score ties on document and chunk
        metadata already present in the schema.
    """

    return (
        -float(chunk.score),
        str(chunk.doc_id),
        str(chunk.section_path),
        str(chunk.chunk_id),
    )


def _dedup_duplicate_docs(
    chunks: list[RetrievedChunk],
    *,
    min_title_length: int = 5,
) -> tuple[list[RetrievedChunk], int]:
    """Remove chunks from duplicate documents (same title, different doc_id).

    When multiple doc_ids in the retrieval results share the same normalized
    title, keeps only chunks from the doc_id with the highest total score.
    This prevents duplicate document copies from wasting retrieval slots and
    crowding out relevant context.

    Args:
        chunks: Retrieved chunks, potentially from duplicate documents.
        min_title_length: Skip dedup for titles shorter than this (avoids
            false-positive dedup on generic titles like "." or "1").

    Returns:
        Tuple of (deduplicated chunks, number of chunks removed).
    """
    if not chunks:
        return chunks, 0

    # Group doc_ids by normalized title.
    title_to_doc_scores: dict[str, dict[str, float]] = {}
    for chunk in chunks:
        norm_title = chunk.doc_title.strip().upper()
        if len(norm_title) < min_title_length:
            continue
        if norm_title not in title_to_doc_scores:
            title_to_doc_scores[norm_title] = {}
        scores = title_to_doc_scores[norm_title]
        if chunk.doc_id not in scores:
            scores[chunk.doc_id] = 0.0
        scores[chunk.doc_id] += chunk.score

    # For titles with >1 doc_id, pick primary (highest total score).
    blocked_doc_ids: set[str] = set()
    for _norm_title, doc_scores in title_to_doc_scores.items():
        if len(doc_scores) <= 1:
            continue
        primary = max(doc_scores, key=lambda d: doc_scores[d])
        for doc_id in doc_scores:
            if doc_id != primary:
                blocked_doc_ids.add(doc_id)

    if not blocked_doc_ids:
        return chunks, 0

    deduped = [c for c in chunks if c.doc_id not in blocked_doc_ids]
    removed = len(chunks) - len(deduped)
    return deduped, removed


def _build_fusion_query_variants(query: str, exact_legal_refs: list[str]) -> list[str]:
    """Build rule-based query variants for RAG-Fusion multi-prefetch.

    Generates 2 variant strings (not including the original query) using only
    string operations — zero LLM cost, deterministic, and safe to call in the
    hot retrieval path.

    Variant 1: entity-context expansion — prepends a legal domain prefix to
        shift the embedding toward provision-level content.
    Variant 2: article/ref-focused — leads with extracted legal refs when
        present (precise anchor retrieval) or strips question words to produce
        a keyword-style variant when no refs are found.

    Args:
        query: Original user query.
        exact_legal_refs: Exact legal references extracted from the query.

    Returns:
        list[str]: Up to 2 variant query strings.
    """
    base = str(query or "").strip()
    if not base:
        return []

    # Variant 1: legal domain context expansion
    entity_variant = f"DIFC legal provision regulation article: {base}"

    # Variant 2: article/ref-focused for precise anchor retrieval
    if exact_legal_refs:
        ref_fragment = " ".join(exact_legal_refs[:3])
        article_variant = f"{ref_fragment} {base}"
    else:
        # Content-focused: strip leading question words to produce keyword-style query
        stripped = re.sub(
            r"^\s*(?:what|when|where|which|who|how|is|are|was|were|does|did|can|should|would|will)\s+",
            "",
            base,
            flags=re.IGNORECASE,
        ).strip()
        article_variant = stripped if len(stripped) > 10 else base

    return [entity_variant, article_variant]


class RetrieverError(RuntimeError):
    pass


class HybridRetriever:
    """Qdrant hybrid search (dense + BM25) with server-side fusion."""

    def __init__(
        self,
        *,
        store: QdrantStore,
        embedder: EmbeddingClient,
    ) -> None:
        settings = get_settings()
        self._store = store
        self._embedder = embedder
        self._qdrant_settings = settings.qdrant
        self._bm25_enabled = bool(getattr(self._qdrant_settings, "enable_sparse_bm25", True))
        self._sparse_encoder: BM25SparseEncoder | None = None
        if self._bm25_enabled:
            cache_dir = str(getattr(self._qdrant_settings, "fastembed_cache_dir", "")).strip() or None
            threads = self._coerce_int(getattr(self._qdrant_settings, "sparse_threads", None))
            try:
                self._sparse_encoder = BM25SparseEncoder(
                    model_name=str(getattr(self._qdrant_settings, "sparse_model", "Qdrant/bm25")),
                    cache_dir=cache_dir,
                    threads=threads,
                )
            except Exception:
                logger.warning("Failed initializing BM25 sparse encoder; disabling BM25", exc_info=True)
                self._bm25_enabled = False
        self._pipeline_settings = settings.pipeline
        self._reranker_settings = settings.reranker
        self._last_retrieved_ids: list[str] = []
        self._last_retrieval_debug: dict[str, object] = {}
        self._collection_exists_cache: dict[str, bool] = {}
        self._entity_registry: EntityRegistry | None = None
        self._entity_registry_initialized = False
        self._qdrant_circuit = CircuitBreaker(
            name="qdrant",
            failure_threshold=int(self._qdrant_settings.circuit_failure_threshold),
            reset_timeout_s=float(self._qdrant_settings.circuit_reset_timeout_s),
        )

        # BM25 hybrid retrieval (separate from Qdrant BM25)
        self._bm25_hybrid_enabled = bool(getattr(self._pipeline_settings, "enable_bm25_hybrid", False))
        self._bm25_retriever: BM25Retriever | None = None
        if self._bm25_hybrid_enabled:
            from pathlib import Path

            alias_map: dict[str, list[str]] = {}
            if bool(getattr(self._pipeline_settings, "enable_bm25_alias_expand", False)):
                alias_map_path = str(getattr(self._pipeline_settings, "bm25_alias_map_path", "data/legal_aliases.json"))
                alias_map = load_alias_map(alias_map_path)
            self._bm25_retriever = BM25Retriever(
                index_dir=Path("data/bm25_index"),
                alias_map=alias_map or None,
            )
            try:
                self._bm25_retriever.load()
                logger.info(
                    "BM25 hybrid retrieval enabled (alias_expand=%s, %d aliases)",
                    bool(alias_map),
                    len(alias_map),
                )
            except Exception:
                logger.warning("Failed loading BM25 index; disabling BM25 hybrid", exc_info=True)
                self._bm25_hybrid_enabled = False
                self._bm25_retriever = None

    async def embed_query(self, query: str) -> list[float]:
        return await self._embedder.embed_query(query)

    def get_last_retrieved_ids(self) -> list[str]:
        return list(self._last_retrieved_ids)

    def get_last_retrieval_debug(self) -> dict[str, object]:
        return deepcopy(self._last_retrieval_debug)

    async def retrieve(
        self,
        query: str,
        *,
        query_vector: list[float] | None = None,
        prefetch_dense: int | None = None,
        prefetch_sparse: int | None = None,
        top_k: int | None = None,
        doc_refs: list[str] | tuple[str, ...] | None = None,
        doc_type_filter: DocType | None = None,
        jurisdiction_filter: str | None = None,
        sparse_only: bool = False,
    ) -> list[RetrievedChunk]:
        extracted_refs = [ref.strip() for ref in (list(doc_refs) if doc_refs is not None else []) if str(ref).strip()]
        expanded_refs = self._expand_doc_ref_variants(extracted_refs)
        sparse_query = self._build_sparse_query(query=query, extracted_refs=extracted_refs)
        exact_legal_refs = QueryClassifier.extract_exact_legal_refs(str(query or ""))
        retrieval_mode = (
            "sparse_only"
            if sparse_only and self._bm25_enabled
            else ("dense_only" if not self._bm25_enabled else "hybrid")
        )
        retrieval_debug: dict[str, object] = {
            "query": str(query or ""),
            "sparse_query": sparse_query,
            "shadow_query": "",
            "doc_refs": list(extracted_refs),
            "expanded_doc_refs": list(expanded_refs),
            "has_doc_refs": bool(extracted_refs),
            "has_exact_legal_refs_in_query": bool(exact_legal_refs),
            "exact_legal_refs": list(exact_legal_refs),
            "doc_type_filter_requested": doc_type_filter.value if doc_type_filter is not None else "",
            "doc_type_filter_inferred": False,
            "jurisdiction_filter": str(jurisdiction_filter or ""),
            "sparse_only_requested": bool(sparse_only),
            "bm25_enabled_at_start": bool(self._bm25_enabled),
            "retrieval_mode": retrieval_mode,
            "hybrid_degraded_to_dense_only": False,
            "dense_only_reason": "bm25_disabled" if not self._bm25_enabled else "",
            "fail_open_triggered": False,
            "fail_open_stage": "none",
            "fail_open_stages": [],
            "initial_doc_ref_filter_applied": bool(expanded_refs),
            "initial_doc_type_filter_applied": doc_type_filter.value if doc_type_filter is not None else "",
            "initial_chunk_count": 0,
            "drop_doc_type_chunk_count": 0,
            "drop_doc_refs_chunk_count": 0,
            "final_chunk_count": 0,
            "final_doc_ref_filter_applied": bool(expanded_refs),
            "final_doc_type_filter_applied": doc_type_filter.value if doc_type_filter is not None else "",
            "source_hits": {"baseline": 0, "shadow": 0, "anchor": 0, "segment": 0, "bridge": 0},
            "source_unique_additions": {"shadow": 0, "anchor": 0, "segment": 0, "bridge": 0},
            "source_survivors": {"baseline": 0, "shadow": 0, "anchor": 0, "segment": 0, "bridge": 0},
            "entity_boosted_chunk_count": 0,
            "cross_ref_boosted_chunk_count": 0,
            "boosted_chunk_ids": [],
            "shadow_collection_used": False,
            "anchor_retrieval_used": False,
            "segment_retrieval_used": False,
            "segment_hit_count": 0,
            "bridge_fact_retrieval_used": False,
            "bridge_fact_hit_count": 0,
            "rag_fusion_enabled": bool(getattr(self._pipeline_settings, "enable_rag_fusion", False)),
            "rag_fusion_variant_count": 0,
            "hyde_enabled": bool(getattr(self._pipeline_settings, "enable_hyde", False)),
            "hyde_hypothetical_doc": "",
            "step_back_enabled": bool(getattr(self._pipeline_settings, "enable_step_back", False)),
            "step_back_query": "",
        }

        if doc_type_filter is None and extracted_refs:
            case_ref_prefixes = {"CFI", "CA", "SCT", "ENF", "DEC", "TCD", "ARB"}
            has_case_ref = any(ref.split(" ", maxsplit=1)[0].upper() in case_ref_prefixes for ref in extracted_refs)
            if has_case_ref and bool(getattr(self._pipeline_settings, "doc_ref_case_law_filter", True)):
                doc_type_filter = DocType.CASE_LAW
                retrieval_debug["doc_type_filter_inferred"] = True
                retrieval_debug["initial_doc_type_filter_applied"] = doc_type_filter.value
                # Metadata-first: case-number queries skip embedding and use
                # sparse + metadata filter only (~1000ms TTFT savings).
                if (
                    not sparse_only
                    and self._bm25_enabled
                    and bool(getattr(self._pipeline_settings, "case_ref_metadata_first", True))
                ):
                    sparse_only = True
                    retrieval_mode = "sparse_only"
                    retrieval_debug["retrieval_mode"] = "sparse_only_case_ref_metadata_first"
                    retrieval_debug["case_ref_metadata_first"] = True

        dense_limit = int(prefetch_dense or self._qdrant_settings.prefetch_dense)
        sparse_limit = int(prefetch_sparse or self._qdrant_settings.prefetch_sparse)
        if extracted_refs:
            if prefetch_dense is None:
                dense_limit = int(getattr(self._pipeline_settings, "doc_ref_prefetch_dense", dense_limit))
            if prefetch_sparse is None:
                sparse_limit = int(getattr(self._pipeline_settings, "doc_ref_prefetch_sparse", sparse_limit))
        limit = int(top_k or self._reranker_settings.rerank_candidates)

        where = self._build_filter(
            doc_type_filter=doc_type_filter,
            jurisdiction_filter=jurisdiction_filter,
            doc_refs=expanded_refs,
        )
        if sparse_only and self._bm25_enabled:
            try:
                result = await self._query_sparse_only(query=sparse_query, limit=limit, where=where)
            except Exception as exc:
                logger.warning("Sparse-only retrieval failed; degrading to standard retrieval path: %s", exc)
                sparse_only = False
                retrieval_mode = "dense_only" if not self._bm25_enabled else "hybrid"
                retrieval_debug["retrieval_mode"] = retrieval_mode
                result = None
        else:
            result = None

        # Step-back query rewriting: abstract specific queries to general form before embedding.
        # Only runs when query_vector is None (we need to embed) and flag is enabled.
        # Modifies the embedding/BM25 query without touching display or EQA query.
        _step_back_enabled = bool(getattr(self._pipeline_settings, "enable_step_back", False))
        if _step_back_enabled and query_vector is None and query:
            try:
                from shafi.core.step_back_rewriter import rewrite_step_back

                _sb_settings = get_settings()
                _sb_api_key = _sb_settings.llm.resolved_api_key().get_secret_value()
                _sb_base_url: str | None = str(_sb_settings.llm.base_url).strip() or None
                embedding_query: str = await rewrite_step_back(
                    str(query),
                    api_key=_sb_api_key,
                    base_url=_sb_base_url,
                )
                retrieval_debug["step_back_query"] = embedding_query
            except Exception:
                logger.warning("step_back rewrite init failed; using original query", exc_info=True)
                embedding_query = str(query or "")
        else:
            embedding_query = str(query or "")

        if result is None:
            # HyDE: Generate hypothetical document and embed in parallel with original query
            hyde_enabled = bool(getattr(self._pipeline_settings, "enable_hyde", False))
            hypo_vector: list[float] | None = None

            if hyde_enabled and query_vector is None:
                # Parallel: embed original query + generate+embed hypothetical document
                hypo_doc_task = asyncio.create_task(generate_hypothetical_document(str(query or ""), get_settings()))
                original_embed_task = asyncio.create_task(self._embedder.embed_query(embedding_query))
                hypo_doc, query_vector = await asyncio.gather(hypo_doc_task, original_embed_task)
                retrieval_debug["hyde_hypothetical_doc"] = hypo_doc[:200] if hypo_doc else ""

                if hypo_doc:
                    try:
                        hypo_vector = await self._embedder.embed_query(hypo_doc)
                        logger.debug("HyDE: embedded hypothetical document (%d dims)", len(hypo_vector))
                    except Exception as exc:
                        logger.warning("HyDE embedding failed: %s", exc)
                        hypo_vector = None
            elif query_vector is None:
                query_vector = await self._embedder.embed_query(embedding_query)

            if not self._bm25_enabled:
                try:
                    result = await self._query_dense_only(
                        query_vector=query_vector,
                        limit=limit,
                        where=where,
                    )
                except Exception as exc:
                    raise RetrieverError(f"Qdrant dense retrieval failed: {exc}") from exc
            else:
                prefetch = self._build_prefetch(
                    query=sparse_query,
                    query_vector=query_vector,
                    prefetch_dense=dense_limit,
                    prefetch_sparse=sparse_limit,
                    where=where,
                )

                # HyDE: Add hypothetical document vector as an extra dense prefetch
                if hyde_enabled and hypo_vector is not None:
                    hyde_limit = int(getattr(self._pipeline_settings, "hyde_extra_prefetch", 40))
                    prefetch.append(
                        models.Prefetch(
                            query=hypo_vector,
                            using="dense",
                            limit=hyde_limit,
                            filter=where,
                        )
                    )
                    retrieval_debug["hyde_prefetch_added"] = True
                    logger.debug("HyDE: added hypothetical prefetch (limit=%d)", hyde_limit)
                else:
                    retrieval_debug["hyde_prefetch_added"] = False

                if getattr(self._pipeline_settings, "enable_rag_fusion", False):
                    prefetch = await self._extend_prefetch_with_fusion_variants(
                        prefetch=prefetch,
                        query=str(query or ""),
                        exact_legal_refs=exact_legal_refs,
                        where=where,
                        retrieval_debug=retrieval_debug,
                    )
                fusion = self._resolve_fusion_method()

                try:
                    result = await self._query_hybrid(prefetch=prefetch, fusion=fusion, limit=limit)
                except Exception as exc:
                    if self._is_qdrant_inference_unavailable(exc):
                        logger.warning("Qdrant BM25 inference unavailable, switching retriever to dense-only mode")
                        self._bm25_enabled = False
                    if self._is_fastembed_unavailable(exc):
                        logger.warning("Qdrant BM25 local model unavailable, switching retriever to dense-only mode")
                        self._bm25_enabled = False
                    logger.warning("Hybrid retrieval failed; degrading to dense-only search: %s", exc)
                    retrieval_mode = "dense_only"
                    retrieval_debug["retrieval_mode"] = retrieval_mode
                    retrieval_debug["hybrid_degraded_to_dense_only"] = True
                    retrieval_debug["dense_only_reason"] = "hybrid_failure"
                    try:
                        result = await self._query_dense_only(
                            query_vector=query_vector,
                            limit=limit,
                            where=where,
                        )
                    except Exception as dense_exc:
                        raise RetrieverError(f"Qdrant retrieval failed (hybrid+dense): {dense_exc}") from dense_exc

        chunks = self._map_results(result)
        retrieval_debug["initial_chunk_count"] = len(chunks)

        # Citation graph hop: expand candidates using external_citations enrichment data
        citation_hop_enabled = bool(getattr(self._pipeline_settings, "enable_citation_hop", False))
        if citation_hop_enabled and chunks:
            expander = CitationGraphExpander(enrichments_dir="data/enrichments")
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            expanded_ids = expander.expand_candidates(chunk_ids, max_hops=1)
            retrieval_debug["citation_hop_expanded_count"] = len(expanded_ids)

            if expanded_ids:
                logger.debug("Citation graph: expanding to %d additional chunks", len(expanded_ids))
                # Fetch expanded chunks from Qdrant
                try:
                    expanded_chunks = await self._fetch_chunks_by_ids(expanded_ids)
                    # Merge with original chunks (dedup by chunk_id)
                    existing_ids = {chunk.chunk_id for chunk in chunks}
                    for chunk in expanded_chunks:
                        if chunk.chunk_id not in existing_ids:
                            chunk.retrieval_sources = ["citation_hop"]
                            chunks.append(chunk)
                    # Re-sort by score
                    chunks = sorted(chunks, key=_retrieved_chunk_sort_key)[:limit]
                    retrieval_debug["citation_hop_added_count"] = len(expanded_chunks)
                except Exception as exc:
                    logger.warning("Citation graph expansion failed: %s", exc)
                    retrieval_debug["citation_hop_added_count"] = 0
            else:
                retrieval_debug["citation_hop_added_count"] = 0
        else:
            retrieval_debug["citation_hop_expanded_count"] = 0
            retrieval_debug["citation_hop_added_count"] = 0

        if extracted_refs and not chunks:
            # Step 1 fail-open: keep doc refs, relax doc_type constraint first.
            logger.info("Doc-ref filter produced 0 chunks; retrying without doc_type filter")
            retrieval_debug["fail_open_triggered"] = True
            fail_open_stages = cast("list[str]", retrieval_debug["fail_open_stages"])
            fail_open_stages.append("drop_doc_type")
            retrieval_debug["fail_open_stage"] = "drop_doc_type"
            fallback_where = self._build_filter(
                doc_type_filter=None,
                jurisdiction_filter=jurisdiction_filter,
                doc_refs=expanded_refs,
            )
            if query_vector is None:
                query_vector = await self._embedder.embed_query(query)
            if sparse_only and self._bm25_enabled:
                result = await self._query_sparse_only(query=sparse_query, limit=limit, where=fallback_where)
            elif not self._bm25_enabled:
                result = await self._query_dense_only(
                    query_vector=query_vector,
                    limit=limit,
                    where=fallback_where,
                )
            else:
                prefetch = self._build_prefetch(
                    query=sparse_query,
                    query_vector=query_vector,
                    prefetch_dense=dense_limit,
                    prefetch_sparse=sparse_limit,
                    where=fallback_where,
                )
                fusion = self._resolve_fusion_method()
                result = await self._query_hybrid(prefetch=prefetch, fusion=fusion, limit=limit)
            chunks = self._map_results(result)
            retrieval_debug["drop_doc_type_chunk_count"] = len(chunks)
            retrieval_debug["final_doc_type_filter_applied"] = ""

        if extracted_refs and not chunks:
            # Step 2 fail-open: final fallback without doc_refs.
            logger.info("Doc-ref filter still produced 0 chunks; retrying without doc_refs")
            retrieval_debug["fail_open_triggered"] = True
            fail_open_stages = cast("list[str]", retrieval_debug["fail_open_stages"])
            fail_open_stages.append("drop_doc_refs")
            retrieval_debug["fail_open_stage"] = "drop_doc_refs"
            fallback_where = self._build_filter(
                doc_type_filter=None,
                jurisdiction_filter=jurisdiction_filter,
                doc_refs=[],
            )
            if query_vector is None:
                query_vector = await self._embedder.embed_query(query)
            if not self._bm25_enabled:
                result = await self._query_dense_only(
                    query_vector=query_vector,
                    limit=limit,
                    where=fallback_where,
                )
            else:
                prefetch = self._build_prefetch(
                    query=sparse_query,
                    query_vector=query_vector,
                    prefetch_dense=dense_limit,
                    prefetch_sparse=sparse_limit,
                    where=fallback_where,
                )
                fusion = self._resolve_fusion_method()
                result = await self._query_hybrid(prefetch=prefetch, fusion=fusion, limit=limit)
            chunks = self._map_results(result)
            retrieval_debug["drop_doc_refs_chunk_count"] = len(chunks)
            retrieval_debug["final_doc_ref_filter_applied"] = False
            retrieval_debug["final_doc_type_filter_applied"] = ""

        baseline_chunks = list(chunks)
        retrieval_debug["source_hits"] = {
            "baseline": len(baseline_chunks),
            "shadow": 0,
            "anchor": 0,
            "segment": 0,
            "bridge": 0,
        }

        chunks = await self._apply_optional_retrieval_surfaces(
            chunks=chunks,
            query=str(query or ""),
            sparse_query=sparse_query,
            query_vector=query_vector,
            limit=limit,
            doc_type_filter=doc_type_filter,
            jurisdiction_filter=jurisdiction_filter,
            expanded_refs=expanded_refs,
            exact_legal_refs=exact_legal_refs,
            retrieval_debug=retrieval_debug,
        )

        chunks = self._apply_payload_boosts(
            chunks=chunks,
            query=str(query or ""),
            exact_legal_refs=exact_legal_refs,
            retrieval_debug=retrieval_debug,
        )

        # Doc-title match boost: when doc_refs are extracted, boost chunks whose
        # doc_title matches one of the refs.  This ensures the target document
        # ranks above documents that merely cross-reference it in their citations.
        if bool(getattr(self._pipeline_settings, "enable_doc_title_boost", False)) and extracted_refs and chunks:
            boost_val = float(getattr(self._pipeline_settings, "doc_title_boost_value", 0.15))
            title_boost_refs = {ref.strip().upper() for ref in extracted_refs if len(ref.strip()) >= 5}
            title_boosted = 0
            boosted_chunks: list[RetrievedChunk] = []
            for chunk in chunks:
                norm_title = chunk.doc_title.strip().upper()
                # Exact match OR ref is a prefix/substring of the title (handles
                # case refs where doc_title is "ARB 027/2024 Nalani v Netty" and
                # the ref is "ARB 027/2024").
                matched = norm_title in title_boost_refs or any(
                    norm_title.startswith(ref) or ref in norm_title for ref in title_boost_refs
                )
                if matched:
                    boosted_chunks.append(chunk.model_copy(update={"score": chunk.score + boost_val}))
                    title_boosted += 1
                else:
                    boosted_chunks.append(chunk)
            if title_boosted:
                chunks = sorted(boosted_chunks, key=_retrieved_chunk_sort_key)
                logger.info("Doc-title boost: boosted %d chunks by %.2f", title_boosted, boost_val)
            retrieval_debug["doc_title_boost_count"] = title_boosted
        else:
            retrieval_debug["doc_title_boost_count"] = 0

        # BM25 hybrid reranking via RRF (if enabled)
        if self._bm25_hybrid_enabled and self._bm25_retriever is not None and chunks:
            try:
                bm25_weight = float(getattr(self._pipeline_settings, "bm25_weight", 0.3))
                rrf_k = int(getattr(self._pipeline_settings, "bm25_rrf_k", 60))
                chunks = self._bm25_retriever.rerank_chunks(
                    chunks=chunks,
                    query=embedding_query,
                    bm25_weight=bm25_weight,
                    rrf_k=rrf_k,
                )
                logger.debug("BM25 RRF reranking applied to %d chunks", len(chunks))
            except Exception:
                logger.warning("BM25 RRF reranking failed; using dense-only results", exc_info=True)

        # Deduplicate chunks from duplicate documents (same title, different doc_id).
        # Prevents ingestion-level duplicates from crowding out relevant context.
        if bool(getattr(self._pipeline_settings, "dedup_duplicate_docs", False)) and chunks:
            min_title_len = int(getattr(self._pipeline_settings, "dedup_min_title_length", 5))
            chunks, dedup_removed = _dedup_duplicate_docs(chunks, min_title_length=min_title_len)
            retrieval_debug["dedup_duplicate_docs_removed"] = dedup_removed
            if dedup_removed:
                logger.info("Dedup: removed %d chunks from duplicate documents", dedup_removed)
        else:
            retrieval_debug["dedup_duplicate_docs_removed"] = 0

        self._last_retrieved_ids = [chunk.chunk_id for chunk in chunks]
        retrieval_debug["final_chunk_count"] = len(chunks)
        self._last_retrieval_debug = retrieval_debug
        logger.info(
            "Hybrid retrieval returned %d chunks (dense=%d sparse=%d top_k=%d doc_ref_filter=%s sparse_only=%s)",
            len(chunks),
            dense_limit,
            sparse_limit,
            limit,
            bool(extracted_refs),
            sparse_only,
        )
        return chunks

    async def _query_hybrid(
        self,
        *,
        collection_name: str | None = None,
        prefetch: list[models.Prefetch],
        fusion: models.Fusion,
        limit: int,
    ) -> object:
        if not self._qdrant_circuit.allow_request():
            raise RetrieverError("Qdrant circuit is open")
        try:
            result = await self._store.client.query_points(
                collection_name=collection_name or self._store.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=fusion),
                limit=limit,
                with_payload=self._payload_selector(),
            )
        except Exception:
            self._qdrant_circuit.record_failure()
            raise
        self._qdrant_circuit.record_success()
        return result

    async def _query_dense_only(
        self,
        *,
        collection_name: str | None = None,
        query_vector: list[float],
        limit: int,
        where: models.Filter | None,
    ) -> object:
        if not self._qdrant_circuit.allow_request():
            raise RetrieverError("Qdrant circuit is open")
        try:
            result = await self._store.client.query_points(
                collection_name=collection_name or self._store.collection_name,
                query=query_vector,
                using="dense",
                query_filter=where,
                limit=limit,
                with_payload=self._payload_selector(),
            )
        except Exception:
            self._qdrant_circuit.record_failure()
            raise
        self._qdrant_circuit.record_success()
        return result

    async def _query_sparse_only(
        self,
        *,
        collection_name: str | None = None,
        query: str,
        limit: int,
        where: models.Filter | None,
    ) -> object:
        if self._sparse_encoder is None:
            raise RetrieverError("BM25 sparse encoder unavailable")
        if not self._qdrant_circuit.allow_request():
            raise RetrieverError("Qdrant circuit is open")
        try:
            sparse_vector = self._sparse_encoder.encode_query(query)
            result = await self._store.client.query_points(
                collection_name=collection_name or self._store.collection_name,
                query=sparse_vector,
                using="bm25",
                query_filter=where,
                limit=limit,
                with_payload=self._payload_selector(),
            )
        except Exception:
            self._qdrant_circuit.record_failure()
            raise
        self._qdrant_circuit.record_success()
        return result

    async def retrieve_with_retry(
        self,
        query: str,
        *,
        expanded_query: str | None = None,
        query_vector: list[float] | None = None,
        doc_refs: list[str] | tuple[str, ...] | None = None,
        doc_type_filter: DocType | None = None,
        jurisdiction_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        search_query = expanded_query or query
        vector = query_vector if (query_vector is not None and expanded_query is None) else None
        return await self.retrieve(
            search_query,
            query_vector=vector,
            prefetch_dense=int(self._pipeline_settings.retry_dense_bias),
            prefetch_sparse=int(self._pipeline_settings.retry_sparse_bias),
            top_k=int(self._reranker_settings.rerank_candidates),
            doc_refs=doc_refs,
            doc_type_filter=doc_type_filter,
            jurisdiction_filter=jurisdiction_filter,
        )

    def _build_prefetch(
        self,
        *,
        query: str,
        query_vector: list[float],
        prefetch_dense: int,
        prefetch_sparse: int,
        where: models.Filter | None,
    ) -> list[models.Prefetch]:
        prefetch: list[models.Prefetch] = [
            models.Prefetch(
                query=query_vector,
                using="dense",
                limit=prefetch_dense,
                filter=where,
            )
        ]

        if self._bm25_enabled and self._sparse_encoder is not None:
            try:
                sparse_vector = self._sparse_encoder.encode_query(query)
            except Exception as exc:
                logger.warning("BM25 sparse query encoding failed; disabling BM25 for this retriever: %s", exc)
                self._bm25_enabled = False
            else:
                prefetch.append(
                    models.Prefetch(
                        query=sparse_vector,
                        using="bm25",
                        limit=prefetch_sparse,
                        filter=where,
                    )
                )
        return prefetch

    async def _extend_prefetch_with_fusion_variants(
        self,
        *,
        prefetch: list[models.Prefetch],
        query: str,
        exact_legal_refs: list[str],
        where: models.Filter | None,
        retrieval_debug: dict[str, object],
    ) -> list[models.Prefetch]:
        """Add RAG-Fusion variant dense prefetches to the existing prefetch list.

        Embeds 2 rule-based query variants in parallel and appends them as extra
        dense prefetch entries. Qdrant RRF then merges all prefetches server-side
        in a single round-trip — zero extra network latency beyond embed time.

        Args:
            prefetch: Existing prefetch list to extend.
            query: Original user query string.
            exact_legal_refs: Legal references extracted from the query.
            where: Qdrant filter applied to all prefetches.
            retrieval_debug: Mutable debug dict for telemetry.

        Returns:
            list[models.Prefetch]: Extended prefetch list with variant entries appended.
        """
        variant_queries = _build_fusion_query_variants(query, exact_legal_refs)
        if not variant_queries:
            return prefetch
        extra_limit = int(self._pipeline_settings.rag_fusion_extra_prefetch)
        try:
            variant_vectors: list[list[float]] = [  # type: ignore[assignment]
                vec for vec in await asyncio.gather(*[self._embedder.embed_query(v) for v in variant_queries])
            ]
        except Exception as exc:
            logger.warning("RAG-Fusion embed failed; skipping variants: %s", exc)
            return prefetch
        extended = list(prefetch)
        for vec in variant_vectors:
            extended.append(
                models.Prefetch(
                    query=vec,
                    using="dense",
                    limit=extra_limit,
                    filter=where,
                )
            )
        retrieval_debug["rag_fusion_variant_count"] = len(variant_vectors)
        return extended

    def _resolve_fusion_method(self) -> models.Fusion:
        fusion_name = str(getattr(self._qdrant_settings, "fusion_method", "RRF")).upper()
        return cast("models.Fusion", getattr(models.Fusion, fusion_name, models.Fusion.RRF))

    @classmethod
    def _build_sparse_query(cls, *, query: str, extracted_refs: list[str] | tuple[str, ...]) -> str:
        return build_sparse_query(query=query, extracted_refs=extracted_refs)

    @staticmethod
    def _is_qdrant_inference_unavailable(exc: Exception) -> bool:
        if not isinstance(exc, UnexpectedResponse):
            return False
        content = getattr(exc, "content", b"")
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = str(content)
        return getattr(exc, "status_code", None) == 500 and "InferenceService is not initialized" in text

    @staticmethod
    def _is_fastembed_unavailable(exc: Exception) -> bool:
        text = str(exc).lower()
        return "fastembed" in text or "onnxruntime" in text

    @staticmethod
    def _coerce_int(value: object) -> int | None:
        return coerce_int(value)

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        return coerce_float(value)

    @staticmethod
    def _coerce_str_list(value: object) -> list[str]:
        return coerce_str_list(value)

    @staticmethod
    def _build_filter(
        *,
        doc_type_filter: DocType | None,
        jurisdiction_filter: str | None,
        doc_refs: list[str] | tuple[str, ...] | None = None,
        chunk_types: list[str] | tuple[str, ...] | None = None,
        must_not_doc_ids: list[str] | tuple[str, ...] | None = None,
    ) -> models.Filter | None:
        return build_filter(
            doc_type_filter=doc_type_filter,
            jurisdiction_filter=jurisdiction_filter,
            doc_refs=doc_refs,
            chunk_types=chunk_types,
            must_not_doc_ids=must_not_doc_ids,
        )

    @staticmethod
    def _expand_doc_ref_variants(refs: list[str] | tuple[str, ...]) -> list[str]:
        return expand_doc_ref_variants(refs)

    @staticmethod
    def _doc_title_filter_variants(refs: list[str] | tuple[str, ...]) -> list[str]:
        return doc_title_filter_variants(refs)

    async def _apply_optional_retrieval_surfaces(
        self,
        *,
        chunks: list[RetrievedChunk],
        query: str,
        sparse_query: str,
        query_vector: list[float] | None,
        limit: int,
        doc_type_filter: DocType | None,
        jurisdiction_filter: str | None,
        expanded_refs: list[str],
        exact_legal_refs: list[str],
        retrieval_debug: dict[str, object],
    ) -> list[RetrievedChunk]:
        merged = {chunk.chunk_id: chunk.model_copy(update={"retrieval_sources": ["baseline"]}) for chunk in chunks}
        source_hits = cast("dict[str, int]", retrieval_debug["source_hits"])

        if bool(getattr(self._pipeline_settings, "enable_shadow_search_text", False)):
            shadow_collection = self._store.shadow_collection_name
            if await self._collection_exists(shadow_collection):
                shadow_query = self._build_shadow_query(query=query, exact_legal_refs=exact_legal_refs)
                retrieval_debug["shadow_query"] = shadow_query
                retrieval_debug["shadow_collection_used"] = True
                shadow_where = self._build_filter(
                    doc_type_filter=doc_type_filter,
                    jurisdiction_filter=jurisdiction_filter,
                    doc_refs=expanded_refs,
                )
                shadow_vector = (
                    query_vector
                    if query_vector is not None and shadow_query == query
                    else await self._embedder.embed_query(shadow_query)
                )
                shadow_limit = min(limit, int(getattr(self._pipeline_settings, "shadow_retrieval_top_k", 24)))
                try:
                    shadow_prefetch = self._build_prefetch(
                        query=shadow_query,
                        query_vector=shadow_vector,
                        prefetch_dense=max(shadow_limit, 8),
                        prefetch_sparse=max(shadow_limit, 8),
                        where=shadow_where,
                    )
                    shadow_result = await self._query_hybrid(
                        collection_name=shadow_collection,
                        prefetch=shadow_prefetch,
                        fusion=self._resolve_fusion_method(),
                        limit=shadow_limit,
                    )
                    shadow_chunks = [
                        chunk.model_copy(update={"retrieval_sources": ["shadow"]})
                        for chunk in self._map_results(shadow_result)
                    ]
                except Exception as exc:
                    logger.warning("Shadow retrieval failed; continuing without shadow surface: %s", exc)
                    shadow_chunks = []
                source_hits["shadow"] = len(shadow_chunks)
                merged = self._merge_chunk_source_maps(merged, shadow_chunks, source_name="shadow")
                cast("dict[str, int]", retrieval_debug["source_unique_additions"])["shadow"] = sum(
                    1
                    for chunk in merged.values()
                    if "shadow" in chunk.retrieval_sources and "baseline" not in chunk.retrieval_sources
                )

        if bool(
            getattr(self._pipeline_settings, "enable_parallel_anchor_retrieval", False)
        ) and self._is_anchor_sensitive_query(query):
            retrieval_debug["anchor_retrieval_used"] = True
            anchor_where = self._build_filter(
                doc_type_filter=doc_type_filter,
                jurisdiction_filter=jurisdiction_filter,
                doc_refs=expanded_refs,
                chunk_types=sorted(_ANCHOR_CHUNK_TYPES),
            )
            anchor_limit = min(limit, int(getattr(self._pipeline_settings, "anchor_retrieval_top_k", 16)))
            try:
                anchor_result = await self._query_sparse_only(
                    collection_name=self._store.collection_name,
                    query=sparse_query,
                    limit=anchor_limit,
                    where=anchor_where,
                )
                anchor_chunks = [
                    chunk.model_copy(update={"retrieval_sources": ["anchor"]})
                    for chunk in self._map_results(anchor_result)
                ]
            except Exception as exc:
                logger.warning("Anchor retrieval failed; continuing without anchor surface: %s", exc)
                anchor_chunks = []
            source_hits["anchor"] = len(anchor_chunks)
            merged = self._merge_chunk_source_maps(merged, anchor_chunks, source_name="anchor")
            cast("dict[str, int]", retrieval_debug["source_unique_additions"])["anchor"] = sum(
                1
                for chunk in merged.values()
                if "anchor" in chunk.retrieval_sources and "baseline" not in chunk.retrieval_sources
            )

        if (
            bool(getattr(self._pipeline_settings, "enable_segment_retrieval", False))
            and await self._collection_exists(self._qdrant_settings.segment_collection)
            and self._should_use_segment_retrieval(query=query, exact_legal_refs=exact_legal_refs)
        ):
            segment_limit = min(limit, int(getattr(self._pipeline_settings, "segment_retrieval_top_k", 4)))
            segment_page_budget = max(1, int(getattr(self._pipeline_settings, "segment_retrieval_page_budget", 4)))
            try:
                segment_hits = await self.retrieve_segments(
                    query=query,
                    query_vector=query_vector,
                    top_k=segment_limit,
                    doc_type_filter=doc_type_filter,
                )
            except Exception as exc:
                logger.warning("Segment retrieval failed; continuing without segment surface: %s", exc)
                segment_hits = []
            retrieval_debug["segment_retrieval_used"] = bool(segment_hits)
            retrieval_debug["segment_hit_count"] = len(segment_hits)
            source_hits["segment"] = len(segment_hits)
            if segment_hits:
                page_scores = self._segment_page_scores(segments=segment_hits, page_budget=segment_page_budget)
                segment_chunks = await self.retrieve_chunks_for_pages(list(page_scores))
                segment_ids = {segment.segment_id for segment in segment_hits}
                promoted_chunks = [
                    chunk.model_copy(
                        update={
                            "score": score,
                            "retrieval_sources": ["segment"],
                        }
                    )
                    for chunk in segment_chunks
                    if (
                        score := self._score_segment_chunk(
                            chunk=chunk,
                            segment_ids=segment_ids,
                            page_scores=page_scores,
                        )
                    )
                    > 0.0
                ]
                merged = self._merge_chunk_source_maps(merged, promoted_chunks, source_name="segment")
                cast("dict[str, int]", retrieval_debug["source_unique_additions"])["segment"] = sum(
                    1
                    for chunk in merged.values()
                    if "segment" in chunk.retrieval_sources and "baseline" not in chunk.retrieval_sources
                )

        if (
            bool(getattr(self._pipeline_settings, "enable_bridge_fact_retrieval", False))
            and await self._collection_exists(self._qdrant_settings.bridge_fact_collection)
            and self._should_use_bridge_fact_retrieval(query=query, exact_legal_refs=exact_legal_refs)
        ):
            bridge_limit = min(limit, int(getattr(self._pipeline_settings, "bridge_fact_retrieval_top_k", 6)))
            bridge_page_budget = max(1, int(getattr(self._pipeline_settings, "bridge_fact_page_budget", 4)))
            try:
                bridge_hits = await self.retrieve_bridge_facts(
                    query=query,
                    query_vector=query_vector,
                    top_k=bridge_limit,
                    doc_type_filter=doc_type_filter,
                )
            except Exception as exc:
                logger.warning("Bridge-fact retrieval failed; continuing without bridge surface: %s", exc)
                bridge_hits = []
            retrieval_debug["bridge_fact_retrieval_used"] = bool(bridge_hits)
            retrieval_debug["bridge_fact_hit_count"] = len(bridge_hits)
            source_hits["bridge"] = len(bridge_hits)
            if bridge_hits:
                page_scores = self._bridge_page_scores(bridge_facts=bridge_hits, page_budget=bridge_page_budget)
                bridge_chunks = await self.retrieve_chunks_for_pages(list(page_scores))
                promoted_bridge_chunks = [
                    chunk.model_copy(
                        update={
                            "score": score,
                            "retrieval_sources": ["bridge"],
                        }
                    )
                    for chunk in bridge_chunks
                    if (score := self._score_bridge_chunk(chunk=chunk, page_scores=page_scores)) > 0.0
                ]
                merged = self._merge_chunk_source_maps(merged, promoted_bridge_chunks, source_name="bridge")
                cast("dict[str, int]", retrieval_debug["source_unique_additions"])["bridge"] = sum(
                    1
                    for chunk in merged.values()
                    if "bridge" in chunk.retrieval_sources and "baseline" not in chunk.retrieval_sources
                )

        # Doc-diversity expansion: if unique doc count in top results is low,
        # do a second retrieval excluding already-found docs to discover new ones.
        enable_diversity = bool(getattr(self._pipeline_settings, "enable_doc_diversity_expansion", False))
        min_docs = int(getattr(self._pipeline_settings, "doc_diversity_min_unique_docs", 8))
        if enable_diversity and len(merged) >= min_docs and query_vector is not None:
            top_for_diversity = sorted(merged.values(), key=_retrieved_chunk_sort_key)[:limit]
            existing_doc_ids = {chunk.doc_id for chunk in top_for_diversity if chunk.doc_id}
            if len(existing_doc_ids) < min_docs:
                diversity_limit = int(getattr(self._pipeline_settings, "doc_diversity_expansion_limit", 40))
                must_not_doc_ids = sorted(existing_doc_ids)
                try:
                    diversity_where = self._build_filter(
                        doc_type_filter=doc_type_filter,
                        jurisdiction_filter=jurisdiction_filter,
                        must_not_doc_ids=must_not_doc_ids,
                    )
                    diversity_prefetch = self._build_prefetch(
                        query=sparse_query,
                        query_vector=query_vector,
                        prefetch_dense=diversity_limit,
                        prefetch_sparse=diversity_limit,
                        where=diversity_where,
                    )
                    diversity_result = await self._query_hybrid(
                        collection_name=self._store.collection_name,
                        prefetch=diversity_prefetch,
                        fusion=self._resolve_fusion_method(),
                        limit=diversity_limit,
                    )
                    diversity_chunks = [
                        chunk.model_copy(update={"retrieval_sources": ["diversity"]})
                        for chunk in self._map_results(diversity_result)
                    ]
                    merged = self._merge_chunk_source_maps(merged, diversity_chunks, source_name="diversity")
                    retrieval_debug["doc_diversity_expansion_used"] = True
                    retrieval_debug["doc_diversity_expansion_added"] = len(diversity_chunks)
                    retrieval_debug["doc_diversity_existing_docs"] = len(existing_doc_ids)
                except Exception as exc:
                    logger.warning("Doc-diversity expansion failed: %s", exc)
                    retrieval_debug["doc_diversity_expansion_used"] = False

        ordered = sorted(merged.values(), key=_retrieved_chunk_sort_key)[:limit]
        cast("dict[str, int]", retrieval_debug["source_survivors"])["baseline"] = sum(
            1 for chunk in ordered if "baseline" in chunk.retrieval_sources
        )
        cast("dict[str, int]", retrieval_debug["source_survivors"])["shadow"] = sum(
            1 for chunk in ordered if "shadow" in chunk.retrieval_sources
        )
        cast("dict[str, int]", retrieval_debug["source_survivors"])["anchor"] = sum(
            1 for chunk in ordered if "anchor" in chunk.retrieval_sources
        )
        cast("dict[str, int]", retrieval_debug["source_survivors"])["segment"] = sum(
            1 for chunk in ordered if "segment" in chunk.retrieval_sources
        )
        cast("dict[str, int]", retrieval_debug["source_survivors"])["bridge"] = sum(
            1 for chunk in ordered if "bridge" in chunk.retrieval_sources
        )
        return ordered

    @staticmethod
    def _merge_chunk_source_maps(
        base: dict[str, RetrievedChunk],
        extra: list[RetrievedChunk],
        *,
        source_name: str,
    ) -> dict[str, RetrievedChunk]:
        merged = dict(base)
        for chunk in extra:
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = chunk
                continue
            sources = list(existing.retrieval_sources)
            if source_name not in sources:
                sources.append(source_name)
            if chunk.score > existing.score:
                merged[chunk.chunk_id] = chunk.model_copy(update={"retrieval_sources": sources})
            else:
                merged[chunk.chunk_id] = existing.model_copy(update={"retrieval_sources": sources})
        return merged

    def _apply_payload_boosts(
        self,
        *,
        chunks: list[RetrievedChunk],
        query: str,
        exact_legal_refs: list[str],
        retrieval_debug: dict[str, object],
    ) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        enable_entity_boosts = bool(getattr(self._pipeline_settings, "enable_entity_boosts", False))
        enable_cross_ref_boosts = bool(getattr(self._pipeline_settings, "enable_cross_ref_boosts", False))
        if not enable_entity_boosts and not enable_cross_ref_boosts:
            return chunks

        query_normalized = re.sub(r"\s+", " ", query).strip().casefold()
        exact_refs_normalized = {ref.casefold() for ref in exact_legal_refs}
        canonical_query_entity_ids = self._get_entity_registry().enrich_query(query).canonical_entity_ids
        boosted_ids: list[str] = []
        entity_boosted = 0
        cross_ref_boosted = 0
        rescored: list[RetrievedChunk] = []
        for chunk in chunks:
            boost = 0.0
            if enable_entity_boosts and self._chunk_matches_entities(
                chunk=chunk,
                query_normalized=query_normalized,
                canonical_query_entity_ids=canonical_query_entity_ids,
            ):
                boost += 0.06
                entity_boosted += 1
                boosted_ids.append(chunk.chunk_id)
            if enable_cross_ref_boosts and self._chunk_matches_cross_refs(
                chunk=chunk, exact_refs_normalized=exact_refs_normalized
            ):
                boost += 0.08
                cross_ref_boosted += 1
                boosted_ids.append(chunk.chunk_id)
            rescored.append(chunk if boost <= 0 else chunk.model_copy(update={"score": chunk.score + boost}))

        retrieval_debug["entity_boosted_chunk_count"] = entity_boosted
        retrieval_debug["cross_ref_boosted_chunk_count"] = cross_ref_boosted
        retrieval_debug["boosted_chunk_ids"] = sorted(set(boosted_ids))
        retrieval_debug["query_canonical_entity_ids"] = list(canonical_query_entity_ids)
        # Use deterministic sort key to stabilize ties after boosting.
        # Name G variance (±25pp) was caused by non-deterministic score-only sort here.
        return sorted(rescored, key=_retrieved_chunk_sort_key)

    def _get_entity_registry(self) -> EntityRegistry:
        """Lazily load the compiler-produced entity registry.

        Returns:
            EntityRegistry: Runtime entity registry wrapper.
        """

        if self._entity_registry_initialized:
            return self._entity_registry or EntityRegistry()
        self._entity_registry_initialized = True
        registry_path = str(getattr(self._pipeline_settings, "canonical_entity_registry_path", "")).strip()
        self._entity_registry = EntityRegistry.load(registry_path) if registry_path else EntityRegistry()
        return self._entity_registry

    @staticmethod
    def _chunk_matches_entities(
        *,
        chunk: RetrievedChunk,
        query_normalized: str,
        canonical_query_entity_ids: tuple[str, ...],
    ) -> bool:
        """Determine whether a chunk should receive an entity-based retrieval boost.

        Args:
            chunk: Retrieved chunk under consideration.
            query_normalized: Normalized query text.
            canonical_query_entity_ids: Canonical IDs resolved from the query.

        Returns:
            bool: True when the chunk matches canonical or raw entity signals.
        """

        if canonical_query_entity_ids and set(chunk.canonical_entity_ids) & set(canonical_query_entity_ids):
            return True
        payload_values = [
            *chunk.party_names,
            *chunk.court_names,
            *chunk.law_titles,
            *chunk.case_numbers,
        ]
        return any(value and value.casefold() in query_normalized for value in payload_values)

    @staticmethod
    def _chunk_matches_cross_refs(*, chunk: RetrievedChunk, exact_refs_normalized: set[str]) -> bool:
        if not exact_refs_normalized:
            return False
        payload_values = {
            value.casefold() for value in [*chunk.article_refs, *chunk.cross_refs, *chunk.normalized_refs] if value
        }
        return bool(payload_values & exact_refs_normalized)

    @staticmethod
    def _build_shadow_query(*, query: str, exact_legal_refs: list[str]) -> str:
        compact = re.sub(r"\s+", " ", query).strip()
        if not exact_legal_refs:
            return compact
        refs = " ".join(ref for ref in exact_legal_refs[:4] for _ in range(2))
        return f"{compact}\n{refs}".strip()

    @staticmethod
    def _is_anchor_sensitive_query(query: str) -> bool:
        return bool(
            _ANCHOR_QUERY_RE.search(str(query or "")) or QueryClassifier.extract_exact_legal_refs(str(query or ""))
        )

    @staticmethod
    def _should_use_segment_retrieval(*, query: str, exact_legal_refs: list[str]) -> bool:
        """Decide whether to activate the additive legal-segment retrieval lane.

        Args:
            query: Raw user query text.
            exact_legal_refs: Exact legal references extracted from the query.

        Returns:
            bool: True when the segment collection should be queried.
        """

        return bool(exact_legal_refs or _SEGMENT_QUERY_RE.search(str(query or "")))

    @staticmethod
    def _should_use_bridge_fact_retrieval(*, query: str, exact_legal_refs: list[str]) -> bool:
        """Decide whether to activate the additive bridge-fact lane.

        Args:
            query: Raw user query text.
            exact_legal_refs: Exact legal references extracted from the query.

        Returns:
            bool: True when bridge facts should be queried.
        """

        return bool(exact_legal_refs or _BRIDGE_QUERY_RE.search(str(query or "")))

    @staticmethod
    def _segment_page_scores(
        *,
        segments: list[RetrievedSegment],
        page_budget: int,
    ) -> dict[str, float]:
        """Project retrieved segments into a bounded page-score map.

        Args:
            segments: Retrieved segments sorted by score.
            page_budget: Maximum number of unique page IDs to keep.

        Returns:
            dict[str, float]: Page-score mapping keyed by platform page ID.
        """

        scored_pages: dict[str, float] = {}
        for segment in segments:
            for page_id in segment.page_ids:
                scored_pages[page_id] = max(scored_pages.get(page_id, 0.0), float(segment.score))
        ordered = sorted(scored_pages.items(), key=lambda item: (-item[1], item[0]))[:page_budget]
        return dict(ordered)

    @staticmethod
    def _score_segment_chunk(
        *,
        chunk: RetrievedChunk,
        segment_ids: set[str],
        page_scores: dict[str, float],
    ) -> float:
        """Assign a bounded score to a chunk surfaced by the segment lane.

        Args:
            chunk: Retrieved chunk from a segment-backed page.
            segment_ids: Segment identifiers returned by the segment lane.
            page_scores: Score map for projected segment pages.

        Returns:
            float: Promoted retrieval score, or `0.0` to discard the chunk.
        """

        page_id = ""
        if chunk.section_path.startswith("page:"):
            try:
                page_num = int(chunk.section_path.split(":", 1)[1])
            except (TypeError, ValueError, IndexError):
                page_num = 0
            if page_num > 0:
                page_id = f"{chunk.doc_id}_{page_num}"
        page_score = page_scores.get(page_id, 0.0)
        if chunk.segment_id and chunk.segment_id in segment_ids:
            return page_score + 0.05
        return page_score

    @staticmethod
    def _bridge_page_scores(
        *,
        bridge_facts: list[RetrievedBridgeFact],
        page_budget: int,
    ) -> dict[str, float]:
        """Project retrieved bridge facts into a bounded evidence-page score map.

        Args:
            bridge_facts: Retrieved bridge facts.
            page_budget: Maximum number of unique evidence pages to keep.

        Returns:
            dict[str, float]: Page-score mapping keyed by page ID.
        """

        scored_pages: dict[str, float] = {}
        for fact in bridge_facts:
            for page_id in fact.evidence_page_ids:
                scored_pages[page_id] = max(scored_pages.get(page_id, 0.0), float(fact.score))
        ordered = sorted(scored_pages.items(), key=lambda item: (-item[1], item[0]))[:page_budget]
        return dict(ordered)

    @staticmethod
    def _score_bridge_chunk(
        *,
        chunk: RetrievedChunk,
        page_scores: dict[str, float],
    ) -> float:
        """Assign a bounded score to a chunk surfaced by the bridge-fact lane.

        Args:
            chunk: Retrieved chunk from a bridge-backed page.
            page_scores: Score map for bridge-fact evidence pages.

        Returns:
            float: Promoted retrieval score, or `0.0` to discard the chunk.
        """

        if not chunk.section_path.startswith("page:"):
            return 0.0
        try:
            page_num = int(chunk.section_path.split(":", 1)[1])
        except (TypeError, ValueError, IndexError):
            return 0.0
        return page_scores.get(f"{chunk.doc_id}_{page_num}", 0.0)

    async def _collection_exists(self, collection_name: str) -> bool:
        cached = self._collection_exists_cache.get(collection_name)
        if cached is not None:
            return cached
        try:
            exists = await self._store.client.collection_exists(collection_name)
        except Exception:
            exists = False
        self._collection_exists_cache[collection_name] = exists
        return exists

    @classmethod
    def _map_results(cls, result: object) -> list[RetrievedChunk]:
        points = cls._extract_points(result)
        mapped: list[RetrievedChunk] = []
        for point_obj in points:
            chunk = cls._map_point(point_obj)
            if chunk is not None:
                mapped.append(chunk)
        return mapped

    async def _fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        """Fetch specific chunks from Qdrant by their chunk IDs.

        Args:
            chunk_ids: List of chunk IDs to fetch.

        Returns:
            list[RetrievedChunk]: Retrieved chunks matching the IDs.
        """
        if not chunk_ids:
            return []

        # Build filter for specific chunk IDs
        where = models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchAny(any=chunk_ids),
                )
            ]
        )

        try:
            scroll_result = await self._store.client.scroll(
                collection_name=self._store.collection_name,
                scroll_filter=where,
                limit=len(chunk_ids),
                with_payload=self._payload_selector(),
            )
            points = scroll_result[0]
            return self._map_results(points)
        except Exception as exc:
            logger.warning("Failed to fetch chunks by IDs: %s", exc)
            return []

    @staticmethod
    def _extract_points(result: object) -> list[object]:
        if isinstance(result, list):
            return list(cast("list[object]", result))

        points_obj = getattr(result, "points", None)
        if isinstance(points_obj, list):
            return list(cast("list[object]", points_obj))

        result_obj = getattr(result, "result", None)
        if isinstance(result_obj, list):
            return list(cast("list[object]", result_obj))

        return []

    @staticmethod
    def _payload_selector() -> models.PayloadSelectorInclude:
        return models.PayloadSelectorInclude(
            include=[
                "chunk_id",
                "doc_id",
                "doc_title",
                "doc_type",
                "section_path",
                "chunk_text",
                "doc_summary",
                "citations",
                "anchors",
                "chunk_type",
                "doc_family",
                "page_family",
                "normalized_refs",
                "amount_roles",
                "shadow_search_text",
                "party_names",
                "court_names",
                "law_titles",
                "article_refs",
                "case_numbers",
                "cross_refs",
                "canonical_entity_ids",
                "segment_id",
            ]
        )

    @staticmethod
    def _segment_payload_selector() -> models.PayloadSelectorInclude:
        """Build the payload selector for additive legal-segment retrieval."""

        return models.PayloadSelectorInclude(
            include=[
                "segment_id",
                "segment_type",
                "doc_id",
                "doc_title",
                "doc_type",
                "canonical_doc_id",
                "legal_path",
                "text",
                "page_ids",
                "start_page",
                "end_page",
                "canonical_entity_ids",
                "search_text",
            ]
        )

    @staticmethod
    def _bridge_fact_payload_selector() -> models.PayloadSelectorInclude:
        """Build the payload selector for additive bridge-fact retrieval."""

        return models.PayloadSelectorInclude(
            include=[
                "fact_id",
                "fact_type",
                "canonical_text",
                "source_entity_ids",
                "source_doc_ids",
                "evidence_page_ids",
                "attributes",
            ]
        )

    async def retrieve_pages(
        self,
        query: str,
        *,
        query_vector: list[float] | None = None,
        top_k: int = 15,
        doc_refs: list[str] | tuple[str, ...] | None = None,
        doc_ids: list[str] | tuple[str, ...] | None = None,
        page_nums: list[int] | tuple[int, ...] | None = None,
        article_refs: list[str] | tuple[str, ...] | None = None,
        page_roles: list[str] | tuple[str, ...] | None = None,
        page_families: list[str] | tuple[str, ...] | None = None,
    ) -> list[RetrievedPage]:
        """Retrieve pages from the page-level collection using hybrid search.

        Args:
            query: Search query text.
            query_vector: Optional precomputed dense vector.
            top_k: Maximum pages to return.
            doc_refs: Document reference strings for sparse query boosting.
            doc_ids: Filter to specific document IDs.
            page_nums: Filter to specific page numbers.
            article_refs: Filter to pages containing these article references.
            page_roles: Filter to pages with these semantic roles.
            page_families: Filter to pages with these page families.

        Returns:
            List of retrieved pages sorted by relevance score.
        """
        page_coll = self._store.page_collection_name
        try:
            exists = await self._store.client.collection_exists(page_coll)
            if not exists:
                logger.warning("Page collection '%s' does not exist; falling back to empty", page_coll)
                return []
        except Exception:
            logger.warning("Failed checking page collection; falling back to empty", exc_info=True)
            return []

        if query_vector is None:
            query_vector = await self._embedder.embed_query(query)

        where = self._build_page_filter(
            doc_ids=doc_ids,
            page_nums=page_nums,
            article_refs=article_refs,
            page_roles=page_roles,
            page_families=page_families,
        )

        if self._bm25_enabled and self._sparse_encoder is not None:
            sparse_query = self._build_sparse_query(query=query, extracted_refs=list(doc_refs or []))
            try:
                sparse_vector = self._sparse_encoder.encode_query(sparse_query)
                prefetch = [
                    models.Prefetch(query=query_vector, using="dense", limit=top_k * 2, filter=where),
                    models.Prefetch(query=sparse_vector, using="bm25", limit=top_k * 2, filter=where),
                ]
                fusion = self._resolve_fusion_method()
                result = await self._store.client.query_points(
                    collection_name=page_coll,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=fusion),
                    limit=top_k,
                    with_payload=True,
                )
            except Exception:
                logger.warning("Hybrid page retrieval failed; falling back to dense-only", exc_info=True)
                result = await self._store.client.query_points(
                    collection_name=page_coll,
                    query=query_vector,
                    using="dense",
                    query_filter=where,
                    limit=top_k,
                    with_payload=True,
                )
        else:
            result = await self._store.client.query_points(
                collection_name=page_coll,
                query=query_vector,
                using="dense",
                query_filter=where,
                limit=top_k,
                with_payload=True,
            )

        pages = self._map_page_results(result)
        logger.info(
            "Page retrieval returned %d pages (top_k=%d doc_ids=%d page_nums=%d article_refs=%d)",
            len(pages),
            top_k,
            len([value for value in (doc_ids or []) if str(value).strip()]),
            len(list(page_nums or [])),
            len([value for value in (article_refs or []) if str(value).strip()]),
        )
        return pages

    async def retrieve_segments(
        self,
        query: str,
        *,
        query_vector: list[float] | None = None,
        top_k: int = 4,
        doc_type_filter: DocType | None = None,
    ) -> list[RetrievedSegment]:
        """Retrieve additive legal segments from the segment collection.

        Args:
            query: Search query text.
            query_vector: Optional precomputed dense embedding.
            top_k: Maximum segments to return.
            doc_type_filter: Optional document-type filter.

        Returns:
            list[RetrievedSegment]: Retrieved segments sorted by relevance.
        """

        segment_coll = self._store.segment_collection_name
        try:
            exists = await self._store.client.collection_exists(segment_coll)
            if not exists:
                return []
        except Exception:
            logger.warning("Failed checking segment collection; falling back to empty", exc_info=True)
            return []

        if query_vector is None:
            query_vector = await self._embedder.embed_query(query)

        segment_filter = (
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_type",
                        match=models.MatchValue(value=doc_type_filter.value),
                    )
                ]
            )
            if doc_type_filter is not None
            else None
        )

        if self._bm25_enabled and self._sparse_encoder is not None:
            sparse_query = self._build_sparse_query(query=query, extracted_refs=[])
            try:
                sparse_vector = self._sparse_encoder.encode_query(sparse_query)
                result = await self._store.client.query_points(
                    collection_name=segment_coll,
                    prefetch=[
                        models.Prefetch(query=query_vector, using="dense", limit=top_k * 2, filter=segment_filter),
                        models.Prefetch(query=sparse_vector, using="bm25", limit=top_k * 2, filter=segment_filter),
                    ],
                    query=models.FusionQuery(fusion=self._resolve_fusion_method()),
                    limit=top_k,
                    with_payload=self._segment_payload_selector(),
                )
            except Exception:
                logger.warning("Hybrid segment retrieval failed; falling back to dense-only", exc_info=True)
                result = await self._store.client.query_points(
                    collection_name=segment_coll,
                    query=query_vector,
                    using="dense",
                    query_filter=segment_filter,
                    limit=top_k,
                    with_payload=self._segment_payload_selector(),
                )
        else:
            result = await self._store.client.query_points(
                collection_name=segment_coll,
                query=query_vector,
                using="dense",
                query_filter=segment_filter,
                limit=top_k,
                with_payload=self._segment_payload_selector(),
            )
        return self._map_segment_results(result)

    async def retrieve_bridge_facts(
        self,
        query: str,
        *,
        query_vector: list[float] | None = None,
        top_k: int = 6,
        doc_type_filter: DocType | None = None,
    ) -> list[RetrievedBridgeFact]:
        """Retrieve additive bridge facts from the bridge-fact collection.

        Args:
            query: Search query text.
            query_vector: Optional precomputed dense embedding.
            top_k: Maximum bridge facts to return.
            doc_type_filter: Optional document-type filter, currently unused.

        Returns:
            list[RetrievedBridgeFact]: Retrieved bridge facts sorted by relevance.
        """

        del doc_type_filter
        bridge_coll = self._store.bridge_fact_collection_name
        try:
            exists = await self._store.client.collection_exists(bridge_coll)
            if not exists:
                return []
        except Exception:
            logger.warning("Failed checking bridge-fact collection; falling back to empty", exc_info=True)
            return []

        if query_vector is None:
            query_vector = await self._embedder.embed_query(query)

        if self._bm25_enabled and self._sparse_encoder is not None:
            sparse_query = self._build_sparse_query(query=query, extracted_refs=[])
            try:
                sparse_vector = self._sparse_encoder.encode_query(sparse_query)
                result = await self._store.client.query_points(
                    collection_name=bridge_coll,
                    prefetch=[
                        models.Prefetch(query=query_vector, using="dense", limit=top_k * 2),
                        models.Prefetch(query=sparse_vector, using="bm25", limit=top_k * 2),
                    ],
                    query=models.FusionQuery(fusion=self._resolve_fusion_method()),
                    limit=top_k,
                    with_payload=self._bridge_fact_payload_selector(),
                )
            except Exception:
                logger.warning("Hybrid bridge-fact retrieval failed; falling back to dense-only", exc_info=True)
                result = await self._store.client.query_points(
                    collection_name=bridge_coll,
                    query=query_vector,
                    using="dense",
                    limit=top_k,
                    with_payload=self._bridge_fact_payload_selector(),
                )
        else:
            result = await self._store.client.query_points(
                collection_name=bridge_coll,
                query=query_vector,
                using="dense",
                limit=top_k,
                with_payload=self._bridge_fact_payload_selector(),
            )
        return self._map_bridge_fact_results(result)

    async def retrieve_chunks_for_pages(
        self,
        page_ids: list[str],
        *,
        limit_per_page: int = 20,
    ) -> list[RetrievedChunk]:
        """Retrieve all chunks belonging to the given pages from the chunk collection."""
        if not page_ids:
            return []

        page_map: dict[str, tuple[str, int]] = {}
        for pid in page_ids:
            parts = pid.rsplit("_", 1)
            if len(parts) == 2:
                doc_id, page_str = parts
                with suppress(ValueError):
                    page_map[pid] = (doc_id, int(page_str))

        all_chunks: list[RetrievedChunk] = []
        for page_id, (doc_id, page_num) in page_map.items():
            section_path = f"page:{page_num}"
            try:
                scroll_result = await self._store.client.scroll(
                    collection_name=self._store.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                            models.FieldCondition(key="section_path", match=models.MatchValue(value=section_path)),
                        ]
                    ),
                    limit=limit_per_page,
                    with_payload=True,
                    with_vectors=False,
                )
                points = scroll_result[0]
                for pt in points:
                    chunk = self._map_point(pt)
                    if chunk is not None:
                        all_chunks.append(chunk)
            except Exception:
                logger.warning("Failed retrieving chunks for page_id=%s", page_id, exc_info=True)

        logger.info("Retrieved %d chunks for %d pages", len(all_chunks), len(page_ids))
        return all_chunks

    @classmethod
    def _map_page_results(cls, result: object) -> list[RetrievedPage]:
        points = cls._extract_points(result)
        mapped: list[RetrievedPage] = []
        for point_obj in points:
            payload_obj = getattr(point_obj, "payload", None)
            if not isinstance(payload_obj, dict):
                continue
            payload = cast("dict[str, object]", payload_obj)
            score_obj = getattr(point_obj, "score", 0.0)
            try:
                score = float(score_obj) if score_obj is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            try:
                page_num = cls._coerce_int(payload.get("page_num", 0)) or 0
            except TypeError:
                page_num = 0
            mapped.append(
                RetrievedPage(
                    page_id=str(payload.get("page_id") or ""),
                    doc_id=str(payload.get("doc_id") or ""),
                    page_num=page_num,
                    doc_title=str(payload.get("doc_title") or ""),
                    doc_type=str(payload.get("doc_type") or ""),
                    page_text=str(payload.get("page_text") or ""),
                    score=score,
                    page_family=str(payload.get("page_family") or ""),
                    doc_family=str(payload.get("doc_family") or ""),
                    normalized_refs=cls._coerce_str_list(payload.get("normalized_refs")),
                    law_titles=cls._coerce_str_list(payload.get("law_titles")),
                    article_refs=cls._coerce_str_list(payload.get("article_refs")),
                    case_numbers=cls._coerce_str_list(payload.get("case_numbers")),
                    page_role=str(payload.get("page_role") or ""),
                    amount_roles=cls._coerce_str_list(payload.get("amount_roles")),
                    linked_refs=cls._coerce_str_list(payload.get("linked_refs")),
                    top_lines=cls._coerce_str_list(payload.get("top_lines")),
                    heading_lines=cls._coerce_str_list(payload.get("heading_lines")),
                    field_labels_present=cls._coerce_str_list(payload.get("field_labels_present")),
                    has_caption_block=bool(payload.get("has_caption_block")),
                    has_title_like_header=bool(payload.get("has_title_like_header")),
                    has_issued_by_pattern=bool(payload.get("has_issued_by_pattern")),
                    has_date_of_issue_pattern=bool(payload.get("has_date_of_issue_pattern")),
                    has_claim_number_pattern=bool(payload.get("has_claim_number_pattern")),
                    has_law_number_pattern=bool(payload.get("has_law_number_pattern")),
                    document_template_family=str(payload.get("document_template_family") or ""),
                    page_template_family=str(payload.get("page_template_family") or ""),
                    officialness_score=cls._coerce_float(payload.get("officialness_score")) or 0.0,
                    source_vs_reference_prior=cls._coerce_float(payload.get("source_vs_reference_prior")) or 0.0,
                    canonical_law_family=str(payload.get("canonical_law_family") or ""),
                    law_title_aliases=cls._coerce_str_list(payload.get("law_title_aliases")),
                    related_law_families=cls._coerce_str_list(payload.get("related_law_families")),
                    canonical_entity_ids=cls._coerce_str_list(payload.get("canonical_entity_ids")),
                )
            )
        return mapped

    @classmethod
    def _map_segment_results(cls, result: object) -> list[RetrievedSegment]:
        """Map Qdrant results into typed additive segment records.

        Args:
            result: Raw Qdrant response object.

        Returns:
            list[RetrievedSegment]: Mapped additive legal segments.
        """

        points = cls._extract_points(result)
        mapped: list[RetrievedSegment] = []
        for point_obj in points:
            payload_obj = getattr(point_obj, "payload", None)
            if not isinstance(payload_obj, dict):
                continue
            payload = cast("dict[str, object]", payload_obj)
            try:
                segment_type = SegmentType(str(payload.get("segment_type") or SegmentType.SECTION.value))
            except ValueError:
                segment_type = SegmentType.SECTION
            try:
                doc_type = DocType(str(payload.get("doc_type") or DocType.OTHER.value))
            except ValueError:
                doc_type = DocType.OTHER
            mapped.append(
                RetrievedSegment(
                    segment_id=str(payload.get("segment_id") or ""),
                    doc_id=str(payload.get("doc_id") or ""),
                    doc_title=str(payload.get("doc_title") or ""),
                    doc_type=doc_type,
                    segment_type=segment_type,
                    legal_path=str(payload.get("legal_path") or ""),
                    text=str(payload.get("text") or ""),
                    score=cls._coerce_float(getattr(point_obj, "score", 0.0)) or 0.0,
                    page_ids=cls._coerce_str_list(payload.get("page_ids")),
                    start_page=cls._coerce_int(payload.get("start_page")) or 0,
                    end_page=cls._coerce_int(payload.get("end_page")) or 0,
                    canonical_doc_id=str(payload.get("canonical_doc_id") or ""),
                    canonical_entity_ids=cls._coerce_str_list(payload.get("canonical_entity_ids")),
                    search_text=str(payload.get("search_text") or ""),
                )
            )
        return mapped

    @classmethod
    def _map_bridge_fact_results(cls, result: object) -> list[RetrievedBridgeFact]:
        """Map Qdrant results into typed additive bridge-fact records.

        Args:
            result: Raw Qdrant response object.

        Returns:
            list[RetrievedBridgeFact]: Mapped bridge-fact records.
        """

        points = cls._extract_points(result)
        mapped: list[RetrievedBridgeFact] = []
        for point_obj in points:
            payload_obj = getattr(point_obj, "payload", None)
            if not isinstance(payload_obj, dict):
                continue
            payload = cast("dict[str, object]", payload_obj)
            try:
                fact_type = BridgeFactType(str(payload.get("fact_type") or BridgeFactType.ENTITY_DOCUMENT.value))
            except ValueError:
                fact_type = BridgeFactType.ENTITY_DOCUMENT
            attributes_obj = payload.get("attributes", {})
            if not isinstance(attributes_obj, dict):
                attributes: dict[str, str] = {}
            else:
                raw_attributes = cast("dict[object, object]", attributes_obj)
                attributes = {str(key): str(value) for key, value in raw_attributes.items() if isinstance(key, str)}
            mapped.append(
                RetrievedBridgeFact(
                    fact_id=str(payload.get("fact_id") or ""),
                    fact_type=fact_type,
                    canonical_text=str(payload.get("canonical_text") or ""),
                    score=cls._coerce_float(getattr(point_obj, "score", 0.0)) or 0.0,
                    source_entity_ids=cls._coerce_str_list(payload.get("source_entity_ids")),
                    source_doc_ids=cls._coerce_str_list(payload.get("source_doc_ids")),
                    evidence_page_ids=cls._coerce_str_list(payload.get("evidence_page_ids")),
                    attributes=attributes,
                )
            )
        return mapped

    @staticmethod
    def _build_page_filter(
        *,
        doc_ids: list[str] | tuple[str, ...] | None = None,
        page_nums: list[int] | tuple[int, ...] | None = None,
        article_refs: list[str] | tuple[str, ...] | None = None,
        page_roles: list[str] | tuple[str, ...] | None = None,
        page_families: list[str] | tuple[str, ...] | None = None,
    ) -> models.Filter | None:
        """Build a Qdrant filter for page-level queries.

        Args:
            doc_ids: Filter to specific document IDs.
            page_nums: Filter to specific page numbers.
            article_refs: Filter to pages with these article references.
            page_roles: Filter to pages with these semantic roles.
            page_families: Filter to pages with these page families.

        Returns:
            Qdrant filter or None if no constraints.
        """
        must: list[models.Condition] = []

        filtered_doc_ids = [str(value).strip() for value in (doc_ids or []) if str(value).strip()]
        if filtered_doc_ids:
            must.append(
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchAny(any=filtered_doc_ids),
                )
            )

        filtered_page_nums = [page_num for page_num in (page_nums or []) if int(page_num) > 0]
        if filtered_page_nums:
            must.append(
                models.FieldCondition(
                    key="page_num",
                    match=models.MatchAny(any=filtered_page_nums),
                )
            )

        filtered_article_refs = [str(value).strip() for value in (article_refs or []) if str(value).strip()]
        if filtered_article_refs:
            must.append(
                models.FieldCondition(
                    key="article_refs",
                    match=models.MatchAny(any=filtered_article_refs),
                )
            )

        filtered_page_roles = [str(value).strip() for value in (page_roles or []) if str(value).strip()]
        if filtered_page_roles:
            must.append(
                models.FieldCondition(
                    key="page_role",
                    match=models.MatchAny(any=filtered_page_roles),
                )
            )

        filtered_page_families = [str(value).strip() for value in (page_families or []) if str(value).strip()]
        if filtered_page_families:
            must.append(
                models.FieldCondition(
                    key="page_family",
                    match=models.MatchAny(any=filtered_page_families),
                )
            )

        return models.Filter(must=must) if must else None

    @staticmethod
    def _map_point(point_obj: object) -> RetrievedChunk | None:
        payload_obj = getattr(point_obj, "payload", None)
        if not isinstance(payload_obj, dict):
            logger.warning("Skipping Qdrant point with non-dict payload")
            return None

        payload = cast("dict[str, object]", payload_obj)
        point_id = getattr(point_obj, "id", "")
        score_obj = getattr(point_obj, "score", 0.0)

        try:
            doc_type_raw = str(payload.get("doc_type", DocType.OTHER.value))
            doc_type = DocType(doc_type_raw)
        except ValueError:
            doc_type = DocType.OTHER

        try:
            score = float(score_obj) if score_obj is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0

        try:
            return RetrievedChunk(
                chunk_id=str(payload.get("chunk_id") or point_id),
                doc_id=str(payload.get("doc_id") or ""),
                doc_title=str(payload.get("doc_title") or ""),
                doc_type=doc_type,
                section_path=str(payload.get("section_path") or ""),
                text=str(payload.get("chunk_text") or ""),
                score=score,
                doc_summary=str(payload.get("doc_summary") or ""),
                page_family=str(payload.get("page_family") or ""),
                doc_family=str(payload.get("doc_family") or ""),
                chunk_type=str(payload.get("chunk_type") or ""),
                amount_roles=HybridRetriever._coerce_str_list(payload.get("amount_roles")),
                normalized_refs=HybridRetriever._coerce_str_list(payload.get("normalized_refs")),
                shadow_search_text=str(payload.get("shadow_search_text") or ""),
                party_names=HybridRetriever._coerce_str_list(payload.get("party_names")),
                court_names=HybridRetriever._coerce_str_list(payload.get("court_names")),
                law_titles=HybridRetriever._coerce_str_list(payload.get("law_titles")),
                article_refs=HybridRetriever._coerce_str_list(payload.get("article_refs")),
                case_numbers=HybridRetriever._coerce_str_list(payload.get("case_numbers")),
                cross_refs=HybridRetriever._coerce_str_list(payload.get("cross_refs")),
                canonical_entity_ids=HybridRetriever._coerce_str_list(payload.get("canonical_entity_ids")),
                segment_id=str(payload.get("segment_id") or ""),
            )
        except Exception:
            logger.warning("Failed to map Qdrant point %s", point_id, exc_info=True)
            return None
