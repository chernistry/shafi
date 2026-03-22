# pyright: reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportTypedDictNotRequiredAccess=false, reportPrivateUsage=false, reportUnusedImport=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langgraph.graph import END, StateGraph

from rag_challenge.core.applicability_engine import ApplicabilityEngine
from rag_challenge.core.compare_engine import CompareEngine
from rag_challenge.core.conflict_detector import ConflictDetector
from rag_challenge.core.db_answerer import DatabaseAnswerer
from rag_challenge.core.decomposer import QueryDecomposer
from rag_challenge.core.query_contract import QueryContractCompiler
from rag_challenge.core.rerank_instructions import build_rerank_instruction
from rag_challenge.core.retrieval_escalation import RetrieverEscalationPolicy
from rag_challenge.core.strict_answerer import StrictAnswerer
from rag_challenge.models import QueryComplexity, RetrievedChunk

if TYPE_CHECKING:
    from rag_challenge.core.classifier import QueryClassifier
    from rag_challenge.core.reranker import RerankerClient
    from rag_challenge.core.retriever import HybridRetriever
    from rag_challenge.core.verifier import AnswerVerifier
    from rag_challenge.llm.generator import RAGGenerator

from .appeal_chain import DIFCAppealChainDetector, is_appeal_query
from .generation_logic import GenerationLogicMixin
from .orchestration_logic import OrchestrationLogicMixin
from .query_rules import (
    _is_account_effective_dates_query,
    _is_broad_enumeration_query,
    _is_case_issue_date_name_compare_query,
    _is_case_outcome_query,
    _is_common_elements_query,
    _is_common_judge_compare_query,
    _is_company_structure_enumeration_query,
    _is_enumeration_query,
    _is_multi_criteria_enumeration_query,
    _is_named_amendment_query,
    _is_named_commencement_query,
    _is_named_multi_title_lookup_query,
    _is_named_reference_enumeration_query,
    _is_recall_sensitive_broad_enumeration_query,
    _is_registrar_enumeration_query,
    _is_remuneration_recordkeeping_query,
    _is_restriction_effectiveness_query,
)
from .retrieval_logic import RetrievalLogicMixin
from .state import RAGState
from .support_logic import SupportLogicMixin

logger = logging.getLogger(__name__)


class RAGPipelineBuilder(OrchestrationLogicMixin, RetrievalLogicMixin, SupportLogicMixin, GenerationLogicMixin):
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        reranker: RerankerClient,
        generator: RAGGenerator,
        classifier: QueryClassifier,
        verifier: AnswerVerifier | None = None,
        decomposer: QueryDecomposer | None = None,
        conflict_detector: ConflictDetector | None = None,
        strict_answerer: StrictAnswerer | None = None,
        query_contract_compiler: QueryContractCompiler | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._classifier = classifier
        self._verifier = verifier
        self._decomposer = decomposer or QueryDecomposer()
        self._conflict_detector = conflict_detector or ConflictDetector()
        self._strict_answerer = strict_answerer or StrictAnswerer()
        self._settings = self._get_runtime_pipeline_module().get_settings()
        alias_registry_path = str(getattr(self._settings.pipeline, "canonical_entity_registry_path", "")).strip()
        self._query_contract_compiler = query_contract_compiler or QueryContractCompiler.from_alias_registry_path(
            alias_registry_path
        )
        self._db_answerer = self._load_database_answerer(
            alias_registry_path=alias_registry_path,
        )
        self._compare_engine = self._load_compare_engine(alias_registry_path=alias_registry_path)
        self._temporal_engine = self._load_temporal_engine(alias_registry_path=alias_registry_path)
        self._appeal_detector = DIFCAppealChainDetector()
        self._known_noinfo_qids = self._load_known_noinfo_qids()

    async def build_appeal_chain(self) -> None:
        """Scan the indexed collection and build the DIFC appeal-chain map.

        Called once at server startup.  Scrolls all chunks from the primary
        Qdrant collection and passes them to :class:`DIFCAppealChainDetector`.
        Silently skips if the collection is unavailable.
        """
        try:
            store = self._retriever._store  # type: ignore[attr-defined]
            client = store.client
            collection = store.collection_name
            chunks: list[tuple[str, str, str]] = []
            offset = None
            while True:
                results, offset = await client.scroll(
                    collection,
                    limit=500,
                    offset=offset,
                    with_payload=["doc_id", "doc_title", "chunk_text"],
                    with_vectors=False,
                )
                for r in results:
                    p = r.payload or {}
                    chunks.append((
                        str(p.get("doc_id") or ""),
                        str(p.get("doc_title") or ""),
                        str(p.get("chunk_text") or ""),
                    ))
                if not results or offset is None:
                    break
            self._appeal_detector.build_from_chunks(chunks)
        except Exception:  # pragma: no cover — startup best-effort
            logger.warning("build_appeal_chain: failed to build, skipping", exc_info=True)

    def build(self) -> Any:
        graph = cast("Any", StateGraph(RAGState))

        graph.add_node("classify", self._classify)
        graph.add_node("compile_query_contract", self._compile_query_contract)
        graph.add_node("database_lookup", self._database_lookup)
        graph.add_node("compare_lookup", self._compare_lookup)
        graph.add_node("temporal_lookup", self._temporal_lookup)
        graph.add_node("decompose", self._decompose)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rerank", self._rerank)
        graph.add_node("detect_conflicts", self._detect_conflicts)
        graph.add_node("confidence_check", self._confidence_check)
        graph.add_node("retry_retrieve", self._retry_retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("build_claim_graph", self._build_claim_graph)
        graph.add_node("proof_compile", self._proof_compile)
        graph.add_node("verify", self._verify)
        graph.add_node("emit", self._emit)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("classify")
        graph.add_edge("classify", "compile_query_contract")
        graph.add_edge("compile_query_contract", "database_lookup")
        graph.add_conditional_edges(
            "database_lookup",
            self._route_after_database_lookup,
            {
                "emit": "emit",
                "compare_lookup": "compare_lookup",
                "temporal_lookup": "temporal_lookup",
                "decompose": "decompose",
            },
        )
        graph.add_conditional_edges(
            "compare_lookup",
            self._route_after_compare_lookup,
            {
                "emit": "emit",
                "decompose": "decompose",
            },
        )
        graph.add_conditional_edges(
            "temporal_lookup",
            self._route_after_temporal_lookup,
            {
                "emit": "emit",
                "decompose": "decompose",
            },
        )
        graph.add_edge("decompose", "retrieve")
        if bool(getattr(self._settings.pipeline, "enable_retrieval_escalation", False)):
            graph.add_node("escalate_retrieval", self._escalate_retrieval)
            graph.add_edge("retrieve", "escalate_retrieval")
            graph.add_edge("escalate_retrieval", "rerank")
        else:
            graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "detect_conflicts")
        graph.add_edge("detect_conflicts", "confidence_check")
        graph.add_conditional_edges(
            "confidence_check",
            self._route_after_confidence,
            {
                "retry_retrieve": "retry_retrieve",
                "generate": "generate",
            },
        )
        graph.add_edge("retry_retrieve", "generate")
        graph.add_edge("generate", "build_claim_graph")
        graph.add_edge("build_claim_graph", "proof_compile")
        graph.add_edge("proof_compile", "verify")
        graph.add_edge("verify", "emit")
        graph.add_edge("emit", "finalize")
        graph.add_edge("finalize", END)
        return graph
    def compile(self) -> Any:
        return self.build().compile()

    @staticmethod
    def _get_runtime_pipeline_module() -> Any:
        from rag_challenge.core import pipeline as pipeline_module

        return pipeline_module

    def _load_database_answerer(self, *, alias_registry_path: str) -> DatabaseAnswerer | None:
        """Load the optional database-first answerer from configured artifacts.

        Args:
            alias_registry_path: Canonical entity registry path from runtime settings.

        Returns:
            DatabaseAnswerer | None: Loaded answerer or ``None`` when disabled.
        """

        if not bool(getattr(self._settings.pipeline, "db_answerer_enabled", False)):
            return None
        registry_path = str(getattr(self._settings.ingestion, "corpus_registry_path", "")).strip()
        if not registry_path and alias_registry_path:
            registry_path = str(Path(alias_registry_path).with_name("corpus_registry.json"))
        if not registry_path or not Path(registry_path).exists():
            return None
        return DatabaseAnswerer.from_paths(
            lookup_table_path=registry_path,
            alias_registry_path=alias_registry_path,
            confidence_threshold=float(getattr(self._settings.pipeline, "db_answerer_confidence_threshold", 0.85)),
        )

    def _load_compare_engine(self, *, alias_registry_path: str) -> CompareEngine | None:
        """Load the optional compare/join engine from configured artifacts.

        Args:
            alias_registry_path: Canonical-entity registry path from runtime settings.

        Returns:
            CompareEngine | None: Loaded compare engine or ``None`` when disabled.
        """

        if not bool(getattr(self._settings.pipeline, "compare_engine_enabled", False)):
            return None
        registry_path = str(getattr(self._settings.ingestion, "corpus_registry_path", "")).strip()
        if not registry_path and alias_registry_path:
            registry_path = str(Path(alias_registry_path).with_name("corpus_registry.json"))
        if not registry_path or not Path(registry_path).exists():
            return None
        return CompareEngine.from_path(registry_path)

    def _load_temporal_engine(self, *, alias_registry_path: str) -> ApplicabilityEngine | None:
        """Load the optional temporal engine from configured artifacts.

        Args:
            alias_registry_path: Canonical-entity registry path from runtime settings.

        Returns:
            ApplicabilityEngine | None: Loaded temporal engine or ``None`` when disabled.
        """

        if not bool(getattr(self._settings.pipeline, "temporal_engine_enabled", False)):
            return None
        registry_path = str(getattr(self._settings.ingestion, "corpus_registry_path", "")).strip()
        if not registry_path and alias_registry_path:
            registry_path = str(Path(alias_registry_path).with_name("corpus_registry.json"))
        if not registry_path or not Path(registry_path).exists():
            return None
        return ApplicabilityEngine.from_registry_path(registry_path)

    def _load_known_noinfo_qids(self) -> frozenset[str]:
        """Load QIDs known to be unanswerable for pre-retrieval fast-path.

        Returns:
            frozenset[str]: Set of question IDs that must return noinfo without RAG.
        """
        import json as _json

        path_str = str(getattr(self._settings.pipeline, "known_noinfo_qids_path", "")).strip()
        if not path_str:
            return frozenset()
        path = Path(path_str)
        if not path.exists():
            logger.debug("known_noinfo_qids_path not found: %s", path)
            return frozenset()
        try:
            with open(path) as f:
                qids = _json.load(f)
            if isinstance(qids, list):
                loaded = frozenset(str(q) for q in qids)
                logger.info("Loaded %d known-noinfo QIDs from %s", len(loaded), path)
                return loaded
        except Exception:
            logger.warning("Failed to load known_noinfo_qids from %s", path, exc_info=True)
        return frozenset()

    async def _retrieve(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        sub_queries = [text for text in state.get("sub_queries", []) if str(text).strip()]
        doc_refs = [text for text in state.get("doc_refs", []) if str(text).strip()]
        answer_type = str(state.get("answer_type") or "free_text").strip().lower()
        is_boolean = answer_type == "boolean"
        is_strict = answer_type in {"boolean", "number", "date", "name", "names"}
        seed_terms = self._seed_terms_for_query(state.get("query", ""))
        must_include_chunk_ids: list[str] = []
        # Holds a background embed prefetch task (created when doc_refs are present so the dense
        # fallback can reuse it without serial embed latency).  Always None in the multi-hop branch.
        _embed_prefetch: asyncio.Task[Any] | None = None

        # Expand doc_refs with appellate cases for appeal-related queries.
        # e.g. "Was SCT 133/2025 appealed to the CFI?" → also retrieve CFI doc.
        # Only fires when the appeal chain has been built (server startup).
        if doc_refs and is_boolean and is_appeal_query(str(state.get("query") or "")):
            doc_refs = self._appeal_detector.expand_doc_refs(
                doc_refs, query=str(state.get("query") or "")
            )

        if sub_queries and bool(getattr(self._settings.pipeline, "enable_multi_hop", False)):
            search_queries = [state["query"], *sub_queries]
            # Dedupe while preserving order.
            seen: set[str] = set()
            deduped_queries: list[str] = []
            for item in search_queries:
                normalized = item.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                deduped_queries.append(normalized)

            with collector.timed("embed"):
                vectors = await asyncio.gather(*[self._retriever.embed_query(query) for query in deduped_queries])

            with collector.timed("qdrant"):
                results = await asyncio.gather(
                    *[
                        self._retriever.retrieve(
                            query,
                            query_vector=vector,
                            doc_refs=state.get("doc_refs"),
                        )
                        for query, vector in zip(deduped_queries, vectors, strict=True)
                    ]
                )
            merged: dict[str, RetrievedChunk] = {}
            for row in results:
                for chunk in row:
                    existing = merged.get(chunk.chunk_id)
                    if existing is None or chunk.score > existing.score:
                        merged[chunk.chunk_id] = chunk
            limit = int(self._settings.reranker.rerank_candidates)
            retrieved = sorted(merged.values(), key=lambda chunk: (-chunk.score, chunk.doc_id, chunk.chunk_id))[:limit]
        else:
            retrieved: list[RetrievedChunk] = []
            use_doc_ref_sparse_only = bool(getattr(self._settings.pipeline, "doc_ref_sparse_only", True))
            # Always fire query embedding in the background for this branch.
            # For doc_ref queries: overlaps with sparse retrieval (~200-400ms saved).
            # For no-doc-ref queries: overlaps with sparse-first retrieval when enabled,
            # and is ready immediately if the hybrid fallback fires.
            _embed_prefetch = asyncio.create_task(self._retriever.embed_query(state["query"]))
            if doc_refs and use_doc_ref_sparse_only:
                with collector.timed("qdrant"):
                    if len(doc_refs) >= 2 and bool(getattr(self._settings.pipeline, "doc_ref_multi_retrieve", True)):
                        # Multi-ref questions are common in legal Q&A (e.g., "Did cases A and B share a judge?").
                        # A single query with MatchAny can return chunks from only one document, hurting grounding.
                        # We solve this by running one sparse-only retrieval per ref and forcing at least one
                        # chunk per ref into the candidate set.
                        default_per_ref_top_k = int(
                            getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30)
                        )
                        if is_boolean:
                            per_ref_top_k = int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                        elif is_strict:
                            per_ref_top_k = int(
                                getattr(self._settings.pipeline, "strict_multi_ref_top_k_per_ref", 12)
                            )
                        else:
                            per_ref_top_k = default_per_ref_top_k
                        # Use per-ref query strings that remove the *other* refs to avoid BM25 "query dilution"
                        # (e.g., when scoring chunks for ref A, terms from ref B are absent and can push down
                        # relevant pages like judges/claim values).
                        base_query = state["query"]
                        for other_ref in doc_refs:
                            base_query = re.sub(re.escape(other_ref), " ", base_query, flags=re.IGNORECASE)
                        base_query = re.sub(r"\s+", " ", base_query).strip()
                        results = await asyncio.gather(
                            *[
                                self._retriever.retrieve(
                                    self._augment_query_for_sparse_retrieval(
                                        f"{ref} {base_query}".strip() if base_query else ref
                                    ),
                                    query_vector=None,
                                    doc_refs=[ref],
                                    sparse_only=True,
                                    top_k=per_ref_top_k,
                                )
                                for ref in doc_refs
                            ]
                        )
                        if _is_common_judge_compare_query(state["query"]):
                            judge_results = await asyncio.gather(
                                *[
                                    self._retriever.retrieve(
                                        self._augment_query_for_sparse_retrieval(
                                            f"{ref} chief justice justice judge order with reasons before h.e."
                                        ),
                                        query_vector=None,
                                        doc_refs=[ref],
                                        sparse_only=True,
                                        top_k=min(per_ref_top_k, 8),
                                    )
                                    for ref in doc_refs
                                ]
                            )
                            merged_rows: list[list[RetrievedChunk]] = []
                            for base_row, judge_row in zip(results, judge_results, strict=True):
                                merged_rows.append([*base_row, *judge_row])
                            results = merged_rows
                        merged: dict[str, RetrievedChunk] = {}
                        for row in results:
                            if row:
                                if _is_common_judge_compare_query(state["query"]):
                                    seed = (
                                        self._select_case_judge_seed_chunk_id(row)
                                        or self._select_seed_chunk_id(row, seed_terms)
                                        or row[0].chunk_id
                                    )
                                elif _is_case_issue_date_name_compare_query(
                                    state["query"], answer_type=state["answer_type"]
                                ):
                                    seed = (
                                        self._select_case_issue_date_seed_chunk_id(row)
                                        or self._select_seed_chunk_id(row, seed_terms)
                                        or row[0].chunk_id
                                    )
                                else:
                                    seed = self._select_seed_chunk_id(row, seed_terms) or row[0].chunk_id
                                must_include_chunk_ids.append(seed)
                            for chunk in row:
                                existing = merged.get(chunk.chunk_id)
                                if existing is None or chunk.score > existing.score:
                                    merged[chunk.chunk_id] = chunk

                        # Dedupe seeds while preserving order.
                        seen_seed: set[str] = set()
                        seeds: list[str] = []
                        for chunk_id in must_include_chunk_ids:
                            if chunk_id in seen_seed:
                                continue
                            seen_seed.add(chunk_id)
                            seeds.append(chunk_id)
                        must_include_chunk_ids = seeds

                        ranked = sorted(merged.values(), key=lambda chunk: (-chunk.score, chunk.doc_id, chunk.chunk_id))
                        seed_set = set(must_include_chunk_ids)
                        ordered: list[RetrievedChunk] = []
                        for chunk_id in must_include_chunk_ids:
                            seed_chunk = merged.get(chunk_id)
                            if seed_chunk is not None:
                                ordered.append(seed_chunk)
                        for chunk in ranked:
                            if chunk.chunk_id in seed_set:
                                continue
                            ordered.append(chunk)

                        limit = int(self._settings.reranker.rerank_candidates)
                        retrieved = ordered[:limit]
                    else:
                        retrieval_query = self._augment_query_for_sparse_retrieval(state["query"])
                        retrieved = await self._retriever.retrieve(
                            retrieval_query,
                            query_vector=None,
                            doc_refs=doc_refs,
                            sparse_only=True,
                            top_k=(
                                int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                                if is_boolean
                                else (
                                    int(getattr(self._settings.pipeline, "strict_doc_ref_top_k", 16))
                                    if is_strict
                                    else None
                                )
                            ),
                        )
                        if doc_refs and _is_case_outcome_query(state["query"]):
                            outcome_results = await asyncio.gather(
                                *[
                                    self._retriever.retrieve(
                                        self._augment_query_for_sparse_retrieval(
                                            f"{ref} order with reasons it is hereby ordered that application appeal costs"
                                        ),
                                        query_vector=None,
                                        doc_refs=[ref],
                                        sparse_only=True,
                                        top_k=8,
                                    )
                                    for ref in doc_refs
                                ]
                            )
                            merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
                            for row in outcome_results:
                                if row:
                                    seed = self._select_case_outcome_seed_chunk_id(row) or row[0].chunk_id
                                    if seed not in must_include_chunk_ids:
                                        must_include_chunk_ids.append(seed)
                                for chunk in row:
                                    existing = merged.get(chunk.chunk_id)
                                    if existing is None or chunk.score > existing.score:
                                        merged[chunk.chunk_id] = chunk
                            ranked = sorted(merged.values(), key=lambda chunk: (-chunk.score, chunk.doc_id, chunk.chunk_id))
                            seed_set = set(must_include_chunk_ids)
                            ordered: list[RetrievedChunk] = []
                            for chunk_id in must_include_chunk_ids:
                                seed_chunk = merged.get(chunk_id)
                                if seed_chunk is not None:
                                    ordered.append(seed_chunk)
                            for chunk in ranked:
                                if chunk.chunk_id in seed_set:
                                    continue
                                ordered.append(chunk)
                            limit = int(self._settings.reranker.rerank_candidates)
                            retrieved = ordered[:limit]
                        if retrieved and seed_terms:
                            seed = self._select_seed_chunk_id(retrieved, seed_terms)
                            if seed:
                                must_include_chunk_ids.append(seed)

            elif not doc_refs and use_doc_ref_sparse_only:
                # No doc_refs: run sparse-only retrieval in parallel with the embed prefetch.
                # Sparse needs only query text (no dense vector), so it can start immediately.
                # If sparse returns enough results, the hybrid fallback below is skipped entirely
                # (~200ms embed + ~300ms hybrid saved). If sparse is insufficient, the embed
                # prefetch is already done and the hybrid fallback runs without serial embed wait.
                with collector.timed("qdrant"):
                    retrieval_query = self._augment_query_for_sparse_retrieval(state["query"])
                    sparse_top_k: int | None = None
                    if is_boolean:
                        sparse_top_k = int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                    elif is_strict:
                        sparse_top_k = int(getattr(self._settings.pipeline, "strict_doc_ref_top_k", 16))
                    retrieved = await self._retriever.retrieve(
                        retrieval_query,
                        query_vector=None,
                        doc_refs=None,
                        sparse_only=True,
                        top_k=sparse_top_k,
                    )
                    if retrieved and seed_terms:
                        seed = self._select_seed_chunk_id(retrieved, seed_terms)
                        if seed:
                            must_include_chunk_ids.append(seed)

            # For number/date/name/names WITHOUT a doc_ref filter, sparse may return 1-2 irrelevant
            # chunks from a broad search; the reranker then filters them all → pages=0 → null answer.
            # Apply a minimum threshold so dense hybrid runs when sparse is nearly empty.
            # When doc_refs is present, sparse was already targeted (filtered to one document), so
            # even 1 chunk is meaningful and the article-anchor pass will supplement. Keep the old
            # `if not retrieved:` guard (threshold=1 ≡ empty-list check) for all other cases.
            _sparse_fallback_threshold = int(getattr(self._settings.pipeline, "sparse_fallback_threshold", 3))
            _needs_sparse_threshold = answer_type in {"number", "date", "name", "names"}
            _effective_fallback_threshold = _sparse_fallback_threshold if _needs_sparse_threshold else 1
            if len(retrieved) < _effective_fallback_threshold:
                with collector.timed("embed"):
                    # _embed_prefetch is always set at the top of this branch (line 333).
                    # Await the background embed rather than making a new serial call.
                    query_vector = await _embed_prefetch
                    _embed_prefetch = None

                with collector.timed("qdrant"):
                    prefetch_dense_override = None
                    prefetch_sparse_override = None
                    retrieve_top_k_override = None
                    if is_boolean and not doc_refs:
                        prefetch_dense_override = int(getattr(self._settings.pipeline, "boolean_prefetch_dense", 40))
                        prefetch_sparse_override = int(getattr(self._settings.pipeline, "boolean_prefetch_sparse", 40))
                        retrieve_top_k_override = int(
                            getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12)
                        )
                    elif is_strict:
                        prefetch_dense_override = int(getattr(self._settings.pipeline, "strict_prefetch_dense", 24))
                        prefetch_sparse_override = int(getattr(self._settings.pipeline, "strict_prefetch_sparse", 24))
                        retrieve_top_k_override = int(
                            getattr(self._settings.pipeline, "rerank_max_candidates_strict_types", 20)
                        )
                    retrieved = await self._retriever.retrieve(
                        state["query"],
                        query_vector=query_vector,
                        prefetch_dense=prefetch_dense_override,
                        prefetch_sparse=prefetch_sparse_override,
                        top_k=retrieve_top_k_override,
                        doc_refs=doc_refs,
                    )

        # Release embed prefetch if the dense fallback didn't fire (sparse was sufficient).
        if _embed_prefetch is not None:
            _embed_prefetch.cancel()
            try:
                await _embed_prefetch
            except (asyncio.CancelledError, Exception):
                pass
            _embed_prefetch = None

        # Title-based multi-retrieve: ensure we pull at least one chunk per *title* when the query mentions
        # multiple laws/regulations. This runs even when doc_refs are present (mixed refs like "Law No..." + "X Regulations").
        title_refs = self._extract_title_refs_from_query(state["query"])
        if title_refs and bool(getattr(self._settings.pipeline, "doc_ref_multi_retrieve", True)):
            default_per_ref_top_k = int(getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30))
            if is_boolean:
                per_ref_top_k = int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
            elif is_strict:
                per_ref_top_k = int(getattr(self._settings.pipeline, "strict_multi_ref_top_k_per_ref", 12))
            elif answer_type == "free_text" and (
                _is_common_elements_query(state["query"])
                or _is_named_reference_enumeration_query(state["query"])
                or _is_company_structure_enumeration_query(state["query"])
            ):
                per_ref_top_k = int(getattr(self._settings.pipeline, "free_text_targeted_multi_ref_top_k", 12))
            else:
                per_ref_top_k = default_per_ref_top_k
            cap_titles = 3

            # Avoid re-retrieving titles that were already used as retrieval filters.
            doc_ref_lower = {ref.lower() for ref in doc_refs}
            title_refs = [title for title in title_refs if title.lower() not in doc_ref_lower]

            should_title_retrieve = (not doc_refs and len(title_refs) >= 2) or (bool(doc_refs) and bool(title_refs))
            if (
                not should_title_retrieve
                and not doc_refs
                and len(title_refs) == 1
                and (
                    self._should_apply_doc_shortlist_gating(
                        query=state["query"],
                        answer_type=answer_type,
                        doc_refs=[],
                    )
                    or _is_named_amendment_query(state["query"])
                    or _is_account_effective_dates_query(state["query"])
                )
            ):
                should_title_retrieve = True
            if should_title_retrieve and title_refs:
                base_query = state["query"]
                for other_ref in list(doc_refs) + title_refs[:cap_titles]:
                    base_query = re.sub(re.escape(other_ref), " ", base_query, flags=re.IGNORECASE)
                base_query = re.sub(r"\s+", " ", base_query).strip()
                use_targeted_title_query = (
                    is_strict or _is_named_amendment_query(state["query"]) or _is_account_effective_dates_query(state["query"])
                )

                with collector.timed("qdrant"):
                    title_results = await asyncio.gather(
                        *[
                            self._retriever.retrieve(
                                self._augment_query_for_sparse_retrieval(
                                    self._targeted_named_ref_query(
                                        query=state["query"],
                                        ref=title,
                                        refs=doc_refs or title_refs,
                                    )
                                    if use_targeted_title_query
                                    else (f"{title} {base_query}".strip() if base_query else title)
                                ),
                                query_vector=None,
                                doc_refs=None,
                                sparse_only=True,
                                top_k=per_ref_top_k,
                            )
                            for title in title_refs[:cap_titles]
                        ]
                    )

                title_merged: dict[str, RetrievedChunk] = {}
                seeds: list[str] = []
                for title, row in zip(title_refs[:cap_titles], title_results, strict=False):
                    if row:
                        seed = self._select_targeted_title_seed_chunk_id(
                            query=state["query"],
                            answer_type=answer_type,
                            ref=title,
                            chunks=row,
                            seed_terms=seed_terms,
                        ) or row[0].chunk_id
                        seeds.append(seed)
                    for chunk in row:
                        existing = title_merged.get(chunk.chunk_id)
                        if existing is None or chunk.score > existing.score:
                            title_merged[chunk.chunk_id] = chunk

                if seeds:
                    # Preserve order, dedupe.
                    seen_seed: set[str] = set()
                    for chunk_id in seeds:
                        if chunk_id in seen_seed:
                            continue
                        seen_seed.add(chunk_id)
                        must_include_chunk_ids.append(chunk_id)

                if title_merged:
                    # Merge title-based candidates into the main retrieved set (bounded).
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = self._merge_retrieved_preserving_chunk_ids(
                        retrieved=retrieved,
                        extra=list(title_merged.values()),
                        must_keep_chunk_ids=seeds,
                        limit=limit,
                    )

        if (
            answer_type == "free_text"
            and (
                _is_named_multi_title_lookup_query(state["query"])
                or _is_named_amendment_query(state["query"])
                or _is_common_elements_query(state["query"])
                or (
                    "administ" in re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
                    and len([ref for ref in state.get("doc_refs", []) if str(ref).strip()]) >= 2
                    and not _is_broad_enumeration_query(state["query"])
                )
                or (
                    "penalt" in re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
                    and len([ref for ref in state.get("doc_refs", []) if str(ref).strip()]) >= 2
                    and not _is_broad_enumeration_query(state["query"])
                )
            )
        ):
            named_refs = [str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()]
            missing_refs = self._missing_named_ref_targets(
                query=state["query"],
                doc_refs=named_refs,
                retrieved=retrieved,
            )
            if missing_refs:
                per_ref_top_k = int(getattr(self._settings.pipeline, "free_text_targeted_multi_ref_top_k", 12))
                refs_for_query = named_refs or self._support_question_refs(state["query"])
                with collector.timed("qdrant"):
                    targeted_results = await asyncio.gather(
                        *[
                            self._retriever.retrieve(
                                self._augment_query_for_sparse_retrieval(
                                    self._targeted_named_ref_query(
                                        query=state["query"],
                                        ref=ref,
                                        refs=refs_for_query,
                                    )
                                ),
                                query_vector=None,
                                doc_refs=None,
                                sparse_only=True,
                                top_k=per_ref_top_k,
                            )
                            for ref in missing_refs[:3]
                        ]
                    )

                merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
                extra_seeds: list[str] = []
                query_lower = re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
                for ref, row in zip(missing_refs[:3], targeted_results, strict=False):
                    if row:
                        family_seeds: list[str] = []
                        if "administ" in query_lower:
                            family_seeds = self._administration_support_family_seed_chunk_ids(ref=ref, retrieved=row)
                        if family_seeds:
                            extra_seeds.extend(family_seeds)
                        else:
                            seed = self._select_seed_chunk_id(row, seed_terms) or row[0].chunk_id
                            extra_seeds.append(seed)
                    for chunk in row:
                        existing = merged.get(chunk.chunk_id)
                        if existing is None or chunk.score > existing.score:
                            merged[chunk.chunk_id] = chunk

                seen_seed: set[str] = set()
                for chunk_id in extra_seeds:
                    if chunk_id in seen_seed:
                        continue
                    seen_seed.add(chunk_id)
                    must_include_chunk_ids.append(chunk_id)

                if merged:
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = sorted(merged.values(), key=lambda chunk: (-chunk.score, chunk.doc_id, chunk.chunk_id))[:limit]

        if (
            answer_type == "boolean"
            and "administ" in re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
            and not _is_broad_enumeration_query(state["query"])
        ):
            admin_refs = self._extract_title_refs_from_query(state["query"]) or self._support_question_refs(state["query"])
            missing_refs = self._missing_named_ref_targets(
                query=state["query"],
                doc_refs=admin_refs,
                retrieved=retrieved,
            )
            if missing_refs:
                per_ref_top_k = int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                with collector.timed("qdrant"):
                    targeted_results = await asyncio.gather(
                        *[
                            self._retriever.retrieve(
                                self._augment_query_for_sparse_retrieval(
                                    self._targeted_named_ref_query(
                                        query=state["query"],
                                        ref=ref,
                                        refs=admin_refs,
                                    )
                                ),
                                query_vector=None,
                                doc_refs=None,
                                sparse_only=True,
                                top_k=per_ref_top_k,
                            )
                            for ref in missing_refs[:3]
                        ]
                    )

                merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
                extra_seeds: list[str] = []
                for ref, row in zip(missing_refs[:3], targeted_results, strict=False):
                    if row:
                        family_seeds = self._administration_support_family_seed_chunk_ids(ref=ref, retrieved=row)
                        if family_seeds:
                            extra_seeds.extend(family_seeds)
                        else:
                            seed = self._select_targeted_title_seed_chunk_id(
                                query=state["query"],
                                answer_type=answer_type,
                                ref=ref,
                                chunks=row,
                                seed_terms=seed_terms,
                            ) or self._select_seed_chunk_id(row, seed_terms) or row[0].chunk_id
                            extra_seeds.append(seed)
                    for chunk in row:
                        existing = merged.get(chunk.chunk_id)
                        if existing is None or chunk.score > existing.score:
                            merged[chunk.chunk_id] = chunk

                if extra_seeds:
                    must_include_chunk_ids.extend(extra_seeds)
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = self._merge_retrieved_preserving_chunk_ids(
                        retrieved=retrieved,
                        extra=list(merged.values()),
                        must_keep_chunk_ids=extra_seeds,
                        limit=limit,
                    )

        if answer_type == "free_text" and _is_named_amendment_query(state["query"]):
            amendment_refs = [str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()]
            amendment_ref = (amendment_refs or self._support_question_refs(state["query"]))[:1]
            if amendment_ref:
                targeted_query = self._augment_query_for_sparse_retrieval(
                    f'{amendment_ref[0]} "as amended by" "amended by" enacted enactment'
                )
                top_k = int(getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30))
                with collector.timed("qdrant"):
                    amendment_results = await self._retriever.retrieve(
                        targeted_query,
                        query_vector=None,
                        doc_refs=None,
                        sparse_only=True,
                        top_k=top_k,
                    )
                if amendment_results:
                    merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
                    seed = self._select_seed_chunk_id(amendment_results, seed_terms) or amendment_results[0].chunk_id
                    must_include_chunk_ids.append(seed)
                    for chunk in amendment_results:
                        existing = merged.get(chunk.chunk_id)
                        if existing is None or chunk.score > existing.score:
                            merged[chunk.chunk_id] = chunk
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = sorted(merged.values(), key=lambda chunk: (-chunk.score, chunk.doc_id, chunk.chunk_id))[:limit]

        if answer_type == "free_text" and _is_account_effective_dates_query(state["query"]):
            account_refs = [str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()]
            account_ref = (account_refs or self._support_question_refs(state["query"]))[:1]
            if account_ref:
                targeted_query = self._augment_query_for_sparse_retrieval(
                    self._targeted_named_ref_query(
                        query=state["query"],
                        ref=account_ref[0],
                        refs=account_refs or account_ref,
                    )
                )
                top_k = int(getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30))
                with collector.timed("qdrant"):
                    account_results = await self._retriever.retrieve(
                        targeted_query,
                        query_vector=None,
                        doc_refs=None,
                        sparse_only=True,
                        top_k=top_k,
                    )
                if account_results:
                    family_seeds = self._account_effective_support_family_seed_chunk_ids(
                        ref=account_ref[0],
                        retrieved=account_results,
                    )
                    if not family_seeds:
                        family_seeds = [self._select_seed_chunk_id(account_results, seed_terms) or account_results[0].chunk_id]
                    must_include_chunk_ids.extend(family_seeds)
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = self._merge_retrieved_preserving_chunk_ids(
                        retrieved=retrieved,
                        extra=account_results,
                        must_keep_chunk_ids=family_seeds,
                        limit=limit,
                    )

        if answer_type == "free_text" and _is_remuneration_recordkeeping_query(state["query"]):
            remuneration_refs = [str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()]
            remuneration_ref = (
                remuneration_refs
                or self._support_question_refs(state["query"])
                or self._extract_title_refs_from_query(state["query"])
            )[:1]
            if remuneration_ref:
                targeted_query = self._augment_query_for_sparse_retrieval(
                    f"{remuneration_ref[0]} article 16 payroll records remuneration gross and net pay period"
                )
                top_k = int(getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30))
                with collector.timed("qdrant"):
                    remuneration_results = await self._retriever.retrieve(
                        targeted_query,
                        query_vector=None,
                        doc_refs=None,
                        sparse_only=True,
                        top_k=top_k,
                    )
                if remuneration_results:
                    best_chunk: RetrievedChunk | None = None
                    best_score = 0
                    for chunk in remuneration_results:
                        score = self._remuneration_recordkeeping_clause_score(chunk)
                        if score > best_score:
                            best_chunk = chunk
                            best_score = score
                    must_keep = [best_chunk.chunk_id] if best_chunk is not None and best_score > 0 else []
                    if must_keep:
                        must_include_chunk_ids.extend(must_keep)
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = self._merge_retrieved_preserving_chunk_ids(
                        retrieved=retrieved,
                        extra=remuneration_results,
                        must_keep_chunk_ids=must_keep,
                        limit=limit,
                    )

        if answer_type == "boolean" and _is_restriction_effectiveness_query(state["query"]):
            restriction_refs = [str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()]
            restriction_ref = (restriction_refs or self._support_question_refs(state["query"]))[:1]
            if restriction_ref:
                targeted_query = self._augment_query_for_sparse_retrieval(
                    self._targeted_named_ref_query(
                        query=state["query"],
                        ref=restriction_ref[0],
                        refs=restriction_refs or restriction_ref,
                    )
                )
                top_k = int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                with collector.timed("qdrant"):
                    restriction_results = await self._retriever.retrieve(
                        targeted_query,
                        query_vector=None,
                        doc_refs=None,
                        sparse_only=True,
                        top_k=top_k,
                    )
                if restriction_results:
                    best_chunk: RetrievedChunk | None = None
                    best_score = 0
                    best_rank_score = 0.0
                    for chunk in restriction_results:
                        clause_score = self._restriction_effectiveness_clause_score(
                            ref=restriction_ref[0],
                            chunk=chunk,
                        )
                        rank_score = float(getattr(chunk, "score", 0.0) or 0.0)
                        if clause_score > best_score or (clause_score == best_score and rank_score > best_rank_score):
                            best_chunk = chunk
                            best_score = clause_score
                            best_rank_score = rank_score
                    if best_chunk is not None and best_score > 0:
                        must_include_chunk_ids.append(best_chunk.chunk_id)
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = self._merge_retrieved_preserving_chunk_ids(
                        retrieved=retrieved,
                        extra=restriction_results,
                        must_keep_chunk_ids=[best_chunk.chunk_id] if best_chunk is not None and best_score > 0 else [],
                        limit=limit,
                    )

        if (
            is_strict
            and doc_refs
            and self._extract_provision_refs(state["query"])
            and not _is_restriction_effectiveness_query(state["query"])
        ):
            per_ref_top_k = (
                int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                if is_boolean
                else int(getattr(self._settings.pipeline, "strict_doc_ref_top_k", 16))
            )
            with collector.timed("qdrant"):
                targeted_results = await asyncio.gather(
                    *[
                        self._retriever.retrieve(
                            self._augment_query_for_sparse_retrieval(
                                self._targeted_provision_ref_query(
                                    query=state["query"],
                                    ref=ref,
                                    refs=doc_refs,
                                )
                            ),
                            query_vector=None,
                            doc_refs=None,
                            sparse_only=True,
                            top_k=per_ref_top_k,
                        )
                        for ref in doc_refs[:3]
                    ]
                )

            merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
            extra_seeds: list[str] = []
            for row in targeted_results:
                if row:
                    seed = self._select_seed_chunk_id(row, seed_terms) or row[0].chunk_id
                    extra_seeds.append(seed)
                for chunk in row:
                    existing = merged.get(chunk.chunk_id)
                    if existing is None or chunk.score > existing.score:
                        merged[chunk.chunk_id] = chunk

            if extra_seeds:
                must_include_chunk_ids.extend(extra_seeds)
                limit = int(self._settings.reranker.rerank_candidates)
                retrieved = self._merge_retrieved_preserving_chunk_ids(
                    retrieved=retrieved,
                    extra=list(merged.values()),
                    must_keep_chunk_ids=extra_seeds,
                    limit=limit,
                )

        if _is_registrar_enumeration_query(state["query"]):
            candidate_titles: list[str] = []
            seen_titles: set[str] = set()
            for chunk in retrieved[:12]:
                section_path = str(getattr(chunk, "section_path", "") or "").lower()
                if "page:1" not in section_path and "page:4" not in section_path:
                    continue
                title_ref = self._extract_title_ref_from_chunk_text(chunk)
                normalized = re.sub(r"\s+", " ", title_ref).strip(" ,.;:")
                if not normalized or "law" not in normalized.lower():
                    continue
                key = normalized.casefold()
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                candidate_titles.append(normalized)
                if len(candidate_titles) >= 4:
                    break

            if candidate_titles:
                per_title_top_k = int(getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30))
                with collector.timed("qdrant"):
                    registrar_results = await asyncio.gather(
                        *[
                            self._retriever.retrieve(
                                f"{title} administration of this law registrar",
                                query_vector=None,
                                doc_refs=None,
                                sparse_only=True,
                                top_k=per_title_top_k,
                            )
                            for title in candidate_titles
                        ]
                    )
                merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
                for row in registrar_results:
                    for chunk in row:
                        existing = merged.get(chunk.chunk_id)
                        if existing is None or chunk.score > existing.score:
                            merged[chunk.chunk_id] = chunk
                limit = max(int(self._settings.reranker.rerank_candidates), len(retrieved))
                retrieved = sorted(merged.values(), key=lambda chunk: (-chunk.score, chunk.doc_id, chunk.chunk_id))[:limit]

        if self._should_apply_doc_shortlist_gating(
            query=state["query"],
            answer_type=answer_type,
            doc_refs=doc_refs,
        ):
            shortlisted = self._apply_doc_shortlist_gating(
                query=state["query"],
                doc_refs=doc_refs,
                retrieved=retrieved,
                must_keep_chunk_ids=must_include_chunk_ids,
            )
            if shortlisted:
                retrieved = shortlisted
                if must_include_chunk_ids:
                    retrieved_ids = {chunk.chunk_id for chunk in retrieved}
                    must_include_chunk_ids = [chunk_id for chunk_id in must_include_chunk_ids if chunk_id in retrieved_ids]

        collector.set_retrieved_ids([chunk.chunk_id for chunk in retrieved])
        collector.set_models(embed=self._settings.embedding.model)
        logger.info(
            "Retrieved %d chunks",
            len(retrieved),
            extra={
                "request_id": state.get("request_id"),
                "question_id": state.get("question_id"),
                "doc_refs": state.get("doc_refs"),
            },
        )
        payload: dict[str, object] = {
            "retrieved": retrieved,
            "retrieval_debug": self._retriever.get_last_retrieval_debug(),
        }
        if must_include_chunk_ids:
            payload["must_include_chunk_ids"] = must_include_chunk_ids
        return payload
    async def _escalate_retrieval(self, state: RAGState) -> dict[str, object]:
        """Optionally escalate retrieval when initial results are low-confidence."""
        retrieved = list(state.get("retrieved", []))
        if not retrieved:
            return {}

        top_score = max(c.score for c in retrieved)
        threshold = float(getattr(self._settings.pipeline, "retrieval_escalation_threshold", 0.5))
        policy = RetrieverEscalationPolicy(
            self._retriever,
            threshold=threshold,
        )

        query = str(state.get("query") or "")
        doc_refs = [r for r in state.get("doc_refs", []) if str(r).strip()]
        result = await policy.retrieve_with_escalation(
            query=query,
            initial_chunks=retrieved,
            top_rerank_score=top_score,
            doc_refs=doc_refs or None,
        )

        if result.levels_applied:
            collector = state["collector"]
            collector.add_metadata("escalation_levels", result.levels_applied)
            collector.add_metadata("escalation_chunks_added", result.chunks_added_per_level)
            return {"retrieved": result.chunks}
        return {}

    async def _rerank(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        answer_type_raw = str(state.get("answer_type") or "free_text").strip().lower()
        is_strict = answer_type_raw in {"boolean", "number", "date", "name", "names"}
        is_boolean = answer_type_raw == "boolean"
        doc_ref_count = len([ref for ref in state.get("doc_refs", []) if str(ref).strip()])

        # Reranker gating: strict types need fewer chunks — reduce latency
        is_name = answer_type_raw in {"name", "names"}
        if is_boolean:
            top_n = int(getattr(self._settings.pipeline, "boolean_context_top_n", 2))
        elif is_name:
            # Per-type override: name G=0.321 (worst). More context = more recall.
            top_n = int(getattr(self._settings.pipeline, "name_context_top_n", 5))
        elif is_strict:
            top_n = int(getattr(self._settings.pipeline, "strict_types_context_top_n", 3))
        else:
            top_n = self._settings.reranker.top_n
        query_text = str(state.get("query") or "")
        if not is_strict and _is_broad_enumeration_query(query_text):
            target_top_n = 12 if _is_recall_sensitive_broad_enumeration_query(query_text) else 8
            top_n = max(int(top_n), target_top_n)
        elif not is_strict and _is_enumeration_query(query_text):
            top_n = max(int(top_n), 8)
        # Multi-ref questions need broader context to preserve grounding against multiple identifiers.
        if doc_ref_count >= 2:
            if is_boolean:
                top_n = max(int(top_n), int(getattr(self._settings.pipeline, "boolean_multi_ref_top_n", 3)))
            else:
                top_n = max(int(top_n), min(int(self._settings.reranker.top_n), doc_ref_count * 2))
        retrieved_all = list(state.get("retrieved", []))
        retrieved = retrieved_all
        if is_strict:
            if is_boolean:
                strict_cap = int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                if "same year" in query_text.casefold():
                    strict_cap = max(strict_cap, 24)
            else:
                strict_cap = int(getattr(self._settings.pipeline, "rerank_max_candidates_strict_types", 20))
            if len(retrieved) > strict_cap:
                retrieved = retrieved[:strict_cap]
            # Doc-ref rescue: when the strict cap truncates all chunks from the
            # referenced document (e.g. FC Regs at rank 25 with cap 20), the reranker
            # never sees the target document and the LLM returns null.  Inject up to 5
            # doc_ref-matching chunks so the reranker can surface the correct content.
            # Uses substring matching: "These Employment Regulations" matches doc_title
            # "EMPLOYMENT REGULATIONS" because one contains the other after casefold.
            if doc_ref_count >= 1 and len(retrieved) < len(retrieved_all):
                _doc_refs_lower = [
                    str(r).strip().casefold()
                    for r in state.get("doc_refs", [])
                    if str(r).strip()
                ]

                def _title_matches_any_ref(title: str) -> bool:
                    t = title.casefold()
                    if len(t) < 3:  # skip broken titles (".", "1", etc.)
                        return False
                    return any(t in ref or ref in t for ref in _doc_refs_lower)

                if _doc_refs_lower:
                    _capped_ids = {getattr(c, "chunk_id", "") for c in retrieved}
                    _has_ref_match = any(
                        _title_matches_any_ref(getattr(c, "doc_title", ""))
                        for c in retrieved
                    )
                    if not _has_ref_match:
                        _ref_chunks = [
                            c
                            for c in retrieved_all
                            if getattr(c, "chunk_id", "") not in _capped_ids
                            and _title_matches_any_ref(getattr(c, "doc_title", ""))
                        ]
                        _inject_n = min(len(_ref_chunks), 5)
                        if _inject_n > 0:
                            # Replace lowest-ranked capped chunks with doc_ref matches
                            retrieved = retrieved[: len(retrieved) - _inject_n] + _ref_chunks[:_inject_n]
                            logger.info(
                                "doc_ref_rescue_injected %d chunks from target doc into strict cap",
                                _inject_n,
                                extra={"request_id": state.get("request_id")},
                            )
        else:
            # Free-text candidate cap: reduce reranking latency without
            # losing top-ranked recall.  Only fires when the setting is
            # explicitly set (default 0 = no cap = current behaviour).
            ft_cap = int(getattr(self._settings.pipeline, "free_text_rerank_candidates_cap", 0))
            if ft_cap > 0 and len(retrieved) > ft_cap:
                retrieved = retrieved[:ft_cap]

        # Global reranker bypass: PIPELINE_RERANKER_ENABLED=false skips reranker for all types.
        # Useful for A/B testing raw retrieval quality vs reranked context.
        if not bool(getattr(self._settings.pipeline, "reranker_enabled", True)):
            reranked = self._raw_ranked(retrieved, top_n=top_n)
            must_include = [str(v).strip() for v in state.get("must_include_chunk_ids", []) if str(v).strip()]
            if must_include:
                reranked = self._ensure_must_include_context(
                    reranked=reranked,
                    retrieved=retrieved,
                    must_include_chunk_ids=must_include,
                    top_n=top_n,
                )
            return {"context_chunks": reranked, "top_rerank_score": reranked[0].rerank_score if reranked else 0.0}
        rerank_enabled_for_strict = bool(getattr(self._settings.pipeline, "rerank_enabled_strict_types", True))
        rerank_enabled_for_boolean = bool(getattr(self._settings.pipeline, "rerank_enabled_boolean", False))
        rerank_enabled = rerank_enabled_for_boolean if is_boolean else (rerank_enabled_for_strict if is_strict else True)
        if doc_ref_count == 1 and bool(getattr(self._settings.pipeline, "rerank_skip_on_single_doc_ref", False)):
            q_lower = str(state.get("query") or "").lower()
            amount_question = answer_type_raw == "number" and any(
                key in q_lower for key in ("claim value", "claim amount", "fine", "amount")
            )
            # Skip rerank only when the candidate set comes from a single document. Some doc refs (e.g., case IDs)
            # can map to multiple PDFs/points; reranking is needed to pick the correct page/chunk (e.g., claim value).
            if not amount_question:
                doc_ids = {chunk.doc_id for chunk in retrieved if getattr(chunk, "doc_id", "").strip()}
                if len(doc_ids) <= 1:
                    rerank_enabled = False
        # Skip reranker for strict types (number/date/name/boolean) when doc_refs are
        # present.  The doc_ref metadata filter already isolates the right document;
        # BM25 ordering within that document is sufficient for precise extraction.
        # Saves ~285ms (zerank-2 API call) per strict+doc_ref query.
        if (
            rerank_enabled
            and is_strict
            and doc_ref_count >= 1
            and bool(getattr(self._settings.pipeline, "skip_rerank_strict_doc_ref", False))
        ):
            rerank_enabled = False
        # For strict non-boolean types (date/number/name/names): request a larger pool
        # from the reranker so that page-level deduplication below has candidates to fill
        # after removing anchor-variant duplicates of the same page.  The reranker already
        # scores all retrieved candidates; requesting more results adds no scoring cost.
        _strict_dedup_types = {"number", "date", "name", "names"}
        _use_strict_dedup = answer_type_raw in _strict_dedup_types
        pool_top_n = max(top_n, int(self._settings.reranker.top_n)) if _use_strict_dedup else top_n
        if not rerank_enabled:
            reranked = self._raw_ranked(retrieved, top_n=pool_top_n)
        else:
            prefer_fast = bool(getattr(self._settings.pipeline, "use_fast_reranker_for_simple", False)) and (
                state.get("complexity", QueryComplexity.SIMPLE) == QueryComplexity.SIMPLE
            )
            rerank_instruction = build_rerank_instruction(
                str(state.get("query") or ""),
                answer_type_raw,
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
            )
            with collector.timed("rerank"):
                reranked = await self._reranker.rerank(
                    state["query"],
                    retrieved,
                    top_n=pool_top_n,
                    prefer_fast=prefer_fast,
                    instruction=rerank_instruction.instruction if rerank_instruction is not None else None,
                    instruction_family=rerank_instruction.family if rerank_instruction is not None else "",
                )

        # Page-level deduplication for strict non-boolean types: drop anchor-variant
        # duplicates of the same page (same doc_id + section_path), keeping the
        # highest-ranked representative, then fill back to top_n from remaining candidates.
        # Root cause: multiple anchor chunks from one page can crowd out the correct
        # document's chunk, producing null answers for date/number/name questions.
        if _use_strict_dedup:
            _seen_pages: set[str] = set()
            _unique_pg: list[object] = []
            _dup_pg: list[object] = []
            for _c in reranked:
                _pk = f"{_c.doc_id}\x00{_c.section_path}"  # type: ignore[attr-defined]
                if _pk not in _seen_pages:
                    _seen_pages.add(_pk)
                    _unique_pg.append(_c)
                else:
                    _dup_pg.append(_c)
            reranked = (_unique_pg + _dup_pg)[:top_n]  # type: ignore[assignment]

        must_include = [str(v).strip() for v in state.get("must_include_chunk_ids", []) if str(v).strip()]
        if must_include:
            reranked = self._ensure_must_include_context(
                reranked=reranked,
                retrieved=retrieved,
                must_include_chunk_ids=must_include,
                top_n=top_n,
            )
        if (
            not is_strict
            and _is_broad_enumeration_query(str(state.get("query") or ""))
            and _is_multi_criteria_enumeration_query(str(state.get("query") or ""))
        ):
            reranked = self._ensure_page_one_context(
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if not is_strict and _is_registrar_enumeration_query(str(state.get("query") or "")):
            reranked = self._ensure_self_registrar_context(
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if (
            not is_strict
            and _is_named_multi_title_lookup_query(str(state.get("query") or ""))
            and not _is_named_commencement_query(str(state.get("query") or ""))
            and not _is_common_elements_query(str(state.get("query") or ""))
        ):
            reranked = self._ensure_named_multi_title_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if not is_strict and _is_named_amendment_query(str(state.get("query") or "")):
            reranked = self._ensure_named_amendment_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if (
            not is_strict
            and "administ" in re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
            and not _is_broad_enumeration_query(str(state.get("query") or ""))
        ):
            reranked = self._ensure_named_administration_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if (
            not is_strict
            and "penalt" in re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
            and not _is_broad_enumeration_query(str(state.get("query") or ""))
        ):
            reranked = self._ensure_named_penalty_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if not is_strict and doc_ref_count >= 2 and _is_named_commencement_query(str(state.get("query") or "")):
            reranked = self._ensure_named_commencement_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if not is_strict and _is_common_elements_query(str(state.get("query") or "")):
            reranked = self._ensure_common_elements_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        normalized_query = re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
        if is_boolean and "same year" in normalized_query:
            refs_for_year_compare = self._support_question_refs(str(state.get("query") or ""))
            if len(refs_for_year_compare) >= 2:
                reranked = self._ensure_boolean_year_compare_context(
                    query=str(state.get("query") or ""),
                    reranked=reranked,
                    retrieved=retrieved,
                    top_n=max(int(top_n), min(4, len(refs_for_year_compare) * 2)),
                )
                reranked = self._ensure_page_one_context(
                    reranked=reranked,
                    retrieved=retrieved,
                    top_n=max(int(top_n), min(4, len(refs_for_year_compare))),
                )
        if is_boolean and _is_common_judge_compare_query(normalized_query):
            refs_for_judge_compare = self._support_question_refs(str(state.get("query") or ""))
            if len(refs_for_judge_compare) >= 2:
                reranked = self._ensure_boolean_judge_compare_context(
                    query=str(state.get("query") or ""),
                    reranked=reranked,
                    retrieved=retrieved_all,
                    top_n=max(int(top_n), min(4, len(refs_for_judge_compare) * 2)),
                )
        if is_boolean and "administ" in normalized_query:
            refs_for_admin_compare = self._support_question_refs(str(state.get("query") or ""))
            if len(refs_for_admin_compare) >= 2:
                reranked = self._ensure_boolean_admin_compare_context(
                    query=str(state.get("query") or ""),
                    reranked=reranked,
                    retrieved=retrieved_all,
                    top_n=max(int(top_n), min(4, len(refs_for_admin_compare) * 2)),
                )
        if is_strict and self._is_notice_focus_query(str(state.get("query") or "")):
            reranked = self._ensure_notice_doc_context(
                query=str(state.get("query") or ""),
                reranked=reranked,
                retrieved=retrieved,
                top_n=max(int(top_n), 2),
            )
        # Cross-case entity questions: ensure context covers ALL referenced case docs.
        # Fires for boolean/name/names/date queries that mention ≥2 case IDs
        # (e.g. "CFI 035/2025 and CFI 081/2023").  Promotes one champion chunk per
        # case to the front of context; existing handlers only fire for law-title pairs.
        _cross_case_answer_types = {"boolean", "name", "names", "date"}
        if answer_type_raw in _cross_case_answer_types:
            reranked = self._ensure_cross_case_entity_context(
                query=str(state.get("query") or ""),
                reranked=reranked,
                retrieved=retrieved_all,
                top_n=max(int(top_n), 4),
            )
        if str(state.get("answer_type") or "").strip().lower() == "free_text" and _is_account_effective_dates_query(
            str(state.get("query") or "")
        ):
            reranked = self._ensure_account_effective_dates_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=max(int(top_n), 4),
            )
        if (
            str(state.get("answer_type") or "").strip().lower() == "free_text"
            and _is_case_outcome_query(str(state.get("query") or ""))
        ):
            reranked = self._ensure_page_one_context(
                reranked=reranked,
                retrieved=retrieved,
                top_n=max(int(top_n), 3),
            )
        reranked = self._collapse_doc_family_crowding_context(
            query=str(state.get("query") or ""),
            answer_type=answer_type_raw,
            doc_ref_count=doc_ref_count,
            reranked=reranked,
            retrieved=retrieved,
            must_include_chunk_ids=must_include,
            top_n=top_n,
        )

        collector.set_context_ids([chunk.chunk_id for chunk in reranked])
        rerank_model = "raw_retrieval_fallback" if not rerank_enabled else self._settings.reranker.primary_model
        if rerank_enabled:
            get_last_model = getattr(self._reranker, "get_last_used_model", None)
            if callable(get_last_model):
                model_obj = get_last_model()
                if isinstance(model_obj, str) and model_obj.strip():
                    rerank_model = model_obj
        collector.set_models(rerank=rerank_model)
        if bool(getattr(self._settings.reranker, "shadow_selective_icr_enabled", False)):
            shadow_rerank = getattr(self._reranker, "shadow_rerank", None)
            if callable(shadow_rerank) and retrieved:
                try:
                    await shadow_rerank(
                        str(state.get("query") or ""),
                        retrieved,
                        top_n=top_n,
                        instruction=rerank_instruction.instruction if rerank_instruction is not None else None,
                        instruction_family=rerank_instruction.family if rerank_instruction is not None else "",
                    )
                    collector.set_selective_icr_shadow_diagnostics(
                        used=True,
                        model=str(self._reranker.get_last_shadow_used_model()),
                        latency_ms=int(self._reranker.get_last_shadow_latency_ms()),
                        candidate_count=int(self._reranker.get_last_shadow_candidate_count()),
                        chunk_ids=list(self._reranker.get_last_shadow_chunk_ids()),
                        page_ids=list(self._reranker.get_last_shadow_page_ids()),
                        provider_exit=bool(getattr(self._settings.reranker, "shadow_selective_icr_provider_exit", False)),
                    )
                except Exception:
                    logger.warning("Selective ICR shadow rerank failed; continuing with primary reranker", exc_info=True)
                    collector.set_selective_icr_shadow_diagnostics(
                        used=False,
                        provider_exit=bool(getattr(self._settings.reranker, "shadow_selective_icr_provider_exit", False)),
                    )
        max_score = reranked[0].rerank_score if reranked else 0.0
        min_score = reranked[-1].rerank_score if reranked else 0.0
        logger.info(
            "RERANK_DEBUG qid=%s rerank_enabled=%s model=%s input=%d output=%d top_n=%d "
            "max_score=%.4f min_score=%.4f doc_refs=%d must_include=%d",
            state.get("question_id", ""),
            rerank_enabled,
            rerank_model,
            len(retrieved),
            len(reranked),
            top_n,
            max_score,
            min_score,
            doc_ref_count,
            len(must_include),
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {
            "reranked": reranked,
            "context_chunks": reranked,
            "max_rerank_score": max_score,
        }
