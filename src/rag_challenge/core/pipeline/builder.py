# pyright: reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportTypedDictNotRequiredAccess=false, reportPrivateUsage=false, reportUnusedImport=false
from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any, cast

from langgraph.graph import END, StateGraph

from rag_challenge.core.conflict_detector import ConflictDetector
from rag_challenge.core.decomposer import QueryDecomposer
from rag_challenge.core.premise_guard import check_query_premise
from rag_challenge.core.strict_answerer import StrictAnswerer
from rag_challenge.models import Citation, QueryComplexity, RankedChunk, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Callable

    from rag_challenge.core.classifier import QueryClassifier
    from rag_challenge.core.reranker import RerankerClient
    from rag_challenge.core.retriever import HybridRetriever
    from rag_challenge.core.verifier import AnswerVerifier
    from rag_challenge.llm.generator import RAGGenerator

from .constants import _STRICT_REPAIR_HINT_TEMPLATE, _TITLE_REF_RE
from .query_rules import (
    _extract_question_title_refs,
    _is_account_effective_dates_query,
    _is_broad_enumeration_query,
    _is_case_issue_date_name_compare_query,
    _is_case_outcome_query,
    _is_citation_title_query,
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
    _is_ruler_enactment_query,
    _needs_long_free_text_answer,
)
from .retrieval_logic import RetrievalLogicMixin
from .state import RAGState
from .support_logic import SupportLogicMixin

logger = logging.getLogger(__name__)

class RAGPipelineBuilder(RetrievalLogicMixin, SupportLogicMixin):
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
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._classifier = classifier
        self._verifier = verifier
        self._decomposer = decomposer or QueryDecomposer()
        self._conflict_detector = conflict_detector or ConflictDetector()
        self._strict_answerer = strict_answerer or StrictAnswerer()
        from rag_challenge.core import pipeline as pipeline_module

        self._settings = pipeline_module.get_settings()

    def build(self) -> Any:
        graph = cast("Any", StateGraph(RAGState))

        graph.add_node("classify", self._classify)
        graph.add_node("decompose", self._decompose)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rerank", self._rerank)
        graph.add_node("detect_conflicts", self._detect_conflicts)
        graph.add_node("confidence_check", self._confidence_check)
        graph.add_node("retry_retrieve", self._retry_retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("verify", self._verify)
        graph.add_node("emit", self._emit)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("classify")
        graph.add_edge("classify", "decompose")
        graph.add_edge("decompose", "retrieve")
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
        graph.add_edge("generate", "verify")
        graph.add_edge("verify", "emit")
        graph.add_edge("emit", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def compile(self) -> Any:
        return self.build().compile()

    async def _classify(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        question_id = str(state.get("question_id") or state["request_id"])
        answer_type = str(state.get("answer_type") or "free_text").strip() or "free_text"
        with collector.timed("classify"):
            normalized_query = self._classifier.normalize_query(state["query"])
            complexity = self._classifier.classify(normalized_query)
            model = self._classifier.select_model(complexity)
            max_tokens = self._classifier.select_max_tokens(complexity)
            answer_type_lower = answer_type.strip().lower()
            if answer_type_lower in {"boolean", "number", "date", "name", "names"}:
                llm_settings = getattr(self._settings, "llm", None)
                strict_max = int(getattr(llm_settings, "strict_max_tokens", 150)) if llm_settings is not None else 150
                max_tokens = min(int(max_tokens), strict_max)
                if answer_type_lower == "boolean":
                    boolean_max = int(getattr(self._settings.pipeline, "boolean_max_tokens", 96))
                    max_tokens = min(max_tokens, max(1, boolean_max))
                if bool(getattr(self._settings.pipeline, "strict_types_force_simple_model", True)):
                    complexity = QueryComplexity.SIMPLE
                    if llm_settings is not None:
                        model = str(
                            getattr(llm_settings, "strict_model", getattr(llm_settings, "simple_model", model))
                        )
            elif answer_type_lower == "free_text":
                # Harder free_text questions benefit from the complex model when they
                # either reference multiple named titles or combine several filters.
                if complexity == QueryComplexity.SIMPLE and (
                    len(_TITLE_REF_RE.findall(normalized_query)) >= 2
                    or _is_multi_criteria_enumeration_query(normalized_query)
                    or _is_registrar_enumeration_query(normalized_query)
                    or _is_ruler_enactment_query(normalized_query)
                    or _is_broad_enumeration_query(normalized_query)
                ):
                    complexity = QueryComplexity.COMPLEX
                    model = self._classifier.select_model(complexity)
                    max_tokens = self._classifier.select_max_tokens(complexity)
                # Prevent truncation on list/enumeration questions that still classify as SIMPLE.
                if _needs_long_free_text_answer(normalized_query):
                    llm_settings = getattr(self._settings, "llm", None)
                    complex_cap = int(getattr(llm_settings, "complex_max_tokens", max_tokens)) if llm_settings is not None else int(max_tokens)
                    max_tokens = max(int(max_tokens), max(600, complex_cap))
                if _is_common_elements_query(normalized_query):
                    common_elements_cap = int(getattr(self._settings.pipeline, "common_elements_max_tokens", 360))
                    max_tokens = min(int(max_tokens), max(200, common_elements_cap))
            doc_refs = []
            # For retrieval filtering, free_text benefits from broader ref extraction (e.g., "Trust Law 2018"),
            # while strict types keep a narrower set (law numbers + case IDs).
            extractor_name = "extract_doc_refs"
            if answer_type_lower == "free_text":
                extractor_name = "extract_query_refs"
            extract_obj: object = getattr(self._classifier, extractor_name, None)
            if not callable(extract_obj) and extractor_name != "extract_doc_refs":
                extract_obj = getattr(self._classifier, "extract_doc_refs", None)
            if callable(extract_obj):
                try:
                    doc_refs_obj: object = extract_obj(normalized_query)
                    if isinstance(doc_refs_obj, list):
                        items = cast("list[object]", doc_refs_obj)
                        doc_refs = [text for item in items if (text := str(item).strip())]
                except Exception:
                    logger.warning("Failed extracting doc refs from query", exc_info=True)

            # Strict-types benefit from title refs too (e.g., "Personal Property Law 2005") to avoid
            # falling back to dense retrieval and to improve grounding for Article-based questions.
            if not doc_refs and answer_type_lower in {"boolean", "number", "date", "name", "names"}:
                extract_any: object = getattr(self._classifier, "extract_query_refs", None)
                if callable(extract_any):
                    try:
                        refs_obj: object = extract_any(normalized_query)
                        if isinstance(refs_obj, list):
                            items = cast("list[object]", refs_obj)
                            extra_refs = [text for item in items if (text := str(item).strip())]
                            extra_refs = [
                                ref
                                for ref in extra_refs
                                if (" Law" in ref or ref.endswith("Law") or "Regulations" in ref)
                            ]
                            # Bound to keep sparse filters stable and avoid over-filtering.
                            doc_refs = extra_refs[:3]
                    except Exception:
                        logger.warning("Failed extracting title refs for strict types", exc_info=True)

            collector.set_request_metadata(question_id=question_id, answer_type=answer_type, doc_refs=doc_refs)

            llm_settings = getattr(self._settings, "llm", None)
            upgrade_model = getattr(llm_settings, "upgrade_model", "")
            if (
                answer_type_lower == "free_text"
                and complexity == QueryComplexity.COMPLEX
                and upgrade_model
            ):
                triggers: list[str] = []
                if len(doc_refs) >= 3:
                    triggers.append("multi_entity")
                if _is_enumeration_query(normalized_query):
                    triggers.append("enumeration")

                if triggers:
                    model = upgrade_model
                    max_tokens = int(getattr(llm_settings, "upgrade_max_tokens", 1800))
                    collector.set_model_upgraded(True)
                    logger.info(
                        "Selective upgrade triggered on %s due to %s",
                        question_id,
                        triggers,
                        extra={"request_id": state.get("request_id"), "triggers": triggers}
                    )

        logger.info(
            "Classified query as %s -> model=%s",
            complexity.value,
            model,
            extra={
                "request_id": state.get("request_id"),
                "question_id": question_id,
                "answer_type": answer_type,
                "doc_refs": doc_refs,
            },
        )
        return {
            "query": normalized_query,
            "complexity": complexity,
            "model": model,
            "max_tokens": max_tokens,
            "question_id": question_id,
            "answer_type": answer_type,
            "doc_refs": doc_refs,
        }

    async def _decompose(self, state: RAGState) -> dict[str, object]:
        if not bool(getattr(self._settings.pipeline, "enable_multi_hop", False)):
            return {"sub_queries": []}
        complexity = state.get("complexity", QueryComplexity.SIMPLE)
        query = state["query"]
        if not self._decomposer.should_decompose(query, complexity):
            return {"sub_queries": []}

        max_subqueries = int(getattr(self._settings.pipeline, "multi_hop_max_subqueries", 3))
        sub_queries = self._decomposer.decompose(query, max_subqueries=max_subqueries)
        logger.info(
            "Decomposed query into %d sub-queries",
            len(sub_queries),
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {"sub_queries": sub_queries}

    async def _retrieve(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        sub_queries = [text for text in state.get("sub_queries", []) if str(text).strip()]
        doc_refs = [text for text in state.get("doc_refs", []) if str(text).strip()]
        answer_type = str(state.get("answer_type") or "free_text").strip().lower()
        is_boolean = answer_type == "boolean"
        is_strict = answer_type in {"boolean", "number", "date", "name", "names"}
        seed_terms = self._seed_terms_for_query(state.get("query", ""))
        must_include_chunk_ids: list[str] = []

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
            retrieved = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)[:limit]
        else:
            retrieved: list[RetrievedChunk] = []
            use_doc_ref_sparse_only = bool(getattr(self._settings.pipeline, "doc_ref_sparse_only", True))
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

                        ranked = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)
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
                            ranked = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)
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

            if not retrieved:
                with collector.timed("embed"):
                    query_vector = await self._retriever.embed_query(state["query"])

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
                    elif is_strict and not doc_refs:
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
                    retrieved = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)[:limit]

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
                    retrieved = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)[:limit]

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
                retrieved = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)[:limit]

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

    async def _rerank(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        answer_type_raw = str(state.get("answer_type") or "free_text").strip().lower()
        is_strict = answer_type_raw in {"boolean", "number", "date", "name", "names"}
        is_boolean = answer_type_raw == "boolean"
        doc_ref_count = len([ref for ref in state.get("doc_refs", []) if str(ref).strip()])

        # Reranker gating: strict types need fewer chunks — reduce latency
        if is_boolean:
            top_n = int(getattr(self._settings.pipeline, "boolean_context_top_n", 2))
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

        rerank_enabled_for_strict = bool(getattr(self._settings.pipeline, "rerank_enabled_strict_types", True))
        rerank_enabled_for_boolean = bool(getattr(self._settings.pipeline, "rerank_enabled_boolean", False))
        rerank_enabled = rerank_enabled_for_boolean if is_boolean else (rerank_enabled_for_strict if is_strict else True)
        if doc_ref_count == 1 and bool(getattr(self._settings.pipeline, "rerank_skip_on_single_doc_ref", True)):
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
        if not rerank_enabled:
            reranked = self._raw_ranked(retrieved, top_n=top_n)
        else:
            prefer_fast = bool(getattr(self._settings.pipeline, "use_fast_reranker_for_simple", False)) and (
                state.get("complexity", QueryComplexity.SIMPLE) == QueryComplexity.SIMPLE
            )
            with collector.timed("rerank"):
                reranked = await self._reranker.rerank(
                    state["query"],
                    retrieved,
                    top_n=top_n,
                    prefer_fast=prefer_fast,
                )

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
        max_score = reranked[0].rerank_score if reranked else 0.0
        logger.info(
            "Reranked %d chunks (max score %.3f)",
            len(reranked),
            max_score,
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {
            "reranked": reranked,
            "context_chunks": reranked,
            "max_rerank_score": max_score,
        }

    async def _detect_conflicts(self, state: RAGState) -> dict[str, object]:
        if not bool(getattr(self._settings.pipeline, "enable_conflict_detection", False)):
            return {}
        if state.get("complexity", QueryComplexity.SIMPLE) != QueryComplexity.COMPLEX:
            return {}

        max_chunks = int(getattr(self._settings.pipeline, "conflict_max_chunks", 8))
        report = self._conflict_detector.detect(state.get("context_chunks", [])[:max_chunks])
        prompt_context = report.to_prompt_context()
        if not prompt_context:
            return {}
        logger.info(
            "Conflict detector found %d potential conflicts",
            len(report.conflicts),
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {"conflict_prompt_context": prompt_context}

    async def _confidence_check(self, state: RAGState) -> dict[str, object]:
        del state
        return {}

    def _route_after_confidence(self, state: RAGState) -> str:
        answer_type = str(state.get("answer_type") or "free_text").strip().lower()
        if answer_type in {"boolean", "number", "date", "name", "names"}:
            return "generate"
        threshold = float(self._settings.pipeline.confidence_threshold)
        max_score = float(state.get("max_rerank_score", 0.0))
        already_retried = bool(state.get("retried", False))
        if max_score < threshold and not already_retried:
            logger.info(
                "Low confidence %.3f < %.3f; retrying retrieval",
                max_score,
                threshold,
                extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
            )
            return "retry_retrieve"
        return "generate"

    async def _retry_retrieve(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        collector.set_retried(True)

        expanded_query = self._expand_retry_query(state)
        with collector.timed("qdrant"):
            retrieved = await self._retriever.retrieve_with_retry(
                state["query"],
                expanded_query=expanded_query,
                doc_refs=state.get("doc_refs"),
            )
        collector.set_retrieved_ids([chunk.chunk_id for chunk in retrieved])
        reranked_state = await self._rerank({**state, "retrieved": retrieved, "collector": collector})
        reranked_obj = reranked_state.get("reranked", [])
        reranked = cast("list[RankedChunk]", reranked_obj if isinstance(reranked_obj, list) else [])
        max_score_raw = reranked_state.get("max_rerank_score", 0.0)
        max_score = float(max_score_raw) if isinstance(max_score_raw, (int, float, str)) else 0.0

        logger.info(
            "Retry retrieval produced %d chunks; reranked to %d",
            len(retrieved),
            len(reranked),
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )

        result_state: dict[str, object] = {
            "retrieved": retrieved,
            **reranked_state,
            "retried": True,
        }

        # If confidence is still low after retry, upgrade the model if configured
        threshold = float(getattr(self._settings.pipeline, "confidence_threshold", 0.5))
        llm_settings = getattr(self._settings, "llm", None)
        upgrade_model = getattr(llm_settings, "upgrade_model", "")
        if max_score < threshold and upgrade_model:
            collector.set_model_upgraded(True)
            result_state["model"] = upgrade_model
            result_state["max_tokens"] = int(getattr(llm_settings, "upgrade_max_tokens", 1800))
            logger.info(
                "Selective upgrade triggered on low confidence (%f < %f)",
                max_score,
                threshold,
                extra={"request_id": state.get("request_id")}
            )

        return result_state

    async def _generate(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        writer = self._get_stream_writer_or_noop()
        collector.set_models(llm=state["model"])

        answer_type = str(state.get("answer_type") or "free_text").strip().lower()
        strict_types = {"boolean", "number", "date", "name", "names"}
        strict_non_stream_types = {"boolean", "number", "date", "name", "names"}
        prompt_hint = str(state.get("conflict_prompt_context") or "").strip()
        strict_repair_enabled = bool(getattr(self._settings.pipeline, "strict_repair_enabled", True))

        # Pre-generation anti-hallucination guardrails for free_text.
        if answer_type == "free_text":
            context_for_guard = state.get("context_chunks", [])
            entity_scope_hint = self._build_entity_scope(context_for_guard)
            if entity_scope_hint:
                prompt_hint = f"{prompt_hint}\n\n{entity_scope_hint}".strip() if prompt_hint else entity_scope_hint
            query_text = str(state["query"] or "")
            query_lower = query_text.lower()
            common_elements_query = _is_common_elements_query(query_text)
            if common_elements_query:
                common_elements_hint = (
                    "For common-elements questions, answer concisely without IRAC or Issue/Rule/Application/"
                    "Conclusion headings. Every claimed common element must be supported by at least one citation "
                    "from every referenced document in the question; each numbered common element must cite that "
                    "support in the same item; if you cannot cite every referenced document for an element, omit "
                    "that element. Output ONLY a numbered list of common elements; do not add an explanatory "
                    "preamble, postscript, or cross-document caveat outside the numbered list. Each numbered "
                    "item must end with one parenthetical citation that includes at least one chunk ID from "
                    "every referenced document, for example (cite: id_a, id_b, id_c). Keep the list compact "
                    "and merge closely related interpretative rules into one item when they belong to the same "
                    "clause family. A structural "
                    "overlap counts as a valid common element "
                    "when each referenced document explicitly states the same structure, such as that Schedule 1 "
                    "contains interpretative provisions or a list of defined terms. List a more specific sub-item "
                    "only if that same sub-item is explicitly shown in every referenced document. If one "
                    "referenced document only shows the Schedule 1 structure but does not quote the substantive "
                    "interpretative rules, do not infer those sub-items as common from the other documents. "
                    "Do not end with "
                    "\"There is no information on this question.\" if you have already identified one or more "
                    "supported common elements. If no explicit common element remains after this check, say "
                    "exactly: \"There is no information on this question.\""
                )
                prompt_hint = f"{prompt_hint}\n\n{common_elements_hint}".strip() if prompt_hint else common_elements_hint
                named_docs = [ref for ref in (state.get("doc_refs") or []) if str(ref).strip()]
                if len(named_docs) >= 2:
                    docs_hint = (
                        "The referenced documents for this common-elements question are: "
                        f"{'; '.join(named_docs)}. If a claimed element is not explicitly supported in each of "
                        "those referenced documents, omit it."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{docs_hint}".strip() if prompt_hint else docs_hint
            if _is_broad_enumeration_query(query_text):
                source_block_count = len([chunk for chunk in context_for_guard if getattr(chunk, "chunk_id", "").strip()])
                enumeration_hint = (
                    "For this broad enumeration question, answer as a compact numbered list. Each item should "
                    "start with the exact matching law or document title supported by the sources. Use the "
                    "minimum number of citations needed to support each item. If separate source blocks are "
                    "needed to prove all criteria for the same law or to supply the exact citation title "
                    "requested, cite each needed block in that same item. Inspect every source block provided "
                    f"({source_block_count} total) before stopping. Output ONLY the numbered list items; do "
                    "not add caveats, comparative commentary, or summary text after the list."
                )
                prompt_hint = f"{prompt_hint}\n\n{enumeration_hint}".strip() if prompt_hint else enumeration_hint
                if (
                    query_lower.startswith("which laws")
                    and not _is_citation_title_query(query_text)
                    and not _is_registrar_enumeration_query(query_text)
                ):
                    titles_only_hint = (
                        "This question asks only which laws match. Each numbered item should mainly give the "
                        "law title itself. Do not add article-level, schedule-level, or explanatory detail "
                        "unless the question explicitly asks for that detail."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{titles_only_hint}".strip() if prompt_hint else titles_only_hint
                if _is_multi_criteria_enumeration_query(query_text):
                    multi_criteria_hint = (
                        "Review every source block before stopping. When multiple source blocks clearly refer to "
                        "the same law title, you may combine those blocks for that law only. Deduplicate by law "
                        "title, but do not stop after the first few matches if later blocks also satisfy all criteria. "
                        "List a law only if the source block or combined source blocks for that same law explicitly "
                        "support every criterion in the question; if one criterion is missing for that law, exclude it. "
                        "Each listed item's citations must collectively support every criterion for that same law."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{multi_criteria_hint}".strip() if prompt_hint else multi_criteria_hint
                if _is_multi_criteria_enumeration_query(query_text) and re.search(r"\b(19|20)\d{2}\b", query_text):
                    year_hint = (
                        "Use the exact year shown in each source block, such as in a law title like "
                        "\"DIFC Law No. 4 of 2018\" or \"Limited Partnership Law 2006\". Do not infer the "
                        "year from consolidated-version dates or amendment dates, and check every source block "
                        "before you finish the list. If the year/title evidence and the substantive matching "
                        "criterion appear in different source blocks for the same law, cite both blocks in that "
                        "same item."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{year_hint}".strip() if prompt_hint else year_hint
                if "administered by the registrar" in query_lower:
                    registrar_hint = (
                        "For this question, include a law only if the source block or combined source blocks for "
                        "that law explicitly state that the law is administered by the Registrar. A title, "
                        "enactment reference, or year alone is not enough. A definitional passage in another "
                        "document about 'legislation administered by the Registrar' or 'Prescribed Laws' does "
                        "NOT make that current document itself a match unless that same document explicitly "
                        "says this law is administered by the Registrar. Use one separate numbered item per "
                        "matching law, and do not merge two different laws into the same numbered item even if "
                        "their supporting clauses are similar."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{registrar_hint}".strip() if prompt_hint else registrar_hint
                if _is_citation_title_query(query_text):
                    citation_title_hint = (
                        "For each listed law, state the exact citation title as written in the source, usually "
                        "from a clause like \"This Law may be cited as ...\". Do not paraphrase, shorten, or "
                        "replace it with generic uppercase document headers. If the title clause and another "
                        "matching criterion appear in different source blocks for the same law, cite both blocks "
                        "in that same item."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{citation_title_hint}".strip() if prompt_hint else citation_title_hint
                question_title_refs = _extract_question_title_refs(query_text)
                if len(question_title_refs) >= 2 and any(term in query_lower for term in ("mention", "mentions", "reference", "references")):
                    named_refs_hint = (
                        "The named law references in this question are: "
                        f"{'; '.join(question_title_refs)}. List a document only if that same document explicitly "
                        "mentions every named law reference above. If a document mentions only one of them, "
                        "exclude it silently. If multiple documents in the sources satisfy all named references, "
                        "list all of them before stopping."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{named_refs_hint}".strip() if prompt_hint else named_refs_hint
                if _is_ruler_enactment_query(query_text):
                    enactment_hint = (
                        "For this question, a law matches if the Enactment Notice itself states both that the "
                        "Ruler of Dubai enacted the law and when it comes into force. A commencement rule written "
                        "as a relative period, such as 'on the 5th business day after enactment' or '90 days after "
                        "enactment', still counts as the commencement being specified in the Enactment Notice. "
                        "Do not exclude a law only because the notice gives a relative commencement period instead "
                        "of a calendar date."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{enactment_hint}".strip() if prompt_hint else enactment_hint

        streamed = False
        streamed_raw = ""
        answer = ""
        extracted = False
        strict_cited_ids: list[str] = []
        context_chunks = list(state.get("context_chunks", []))
        if answer_type in strict_types:
            context_chunks, strict_context_augmented = self._augment_strict_context_chunks(
                query=state["query"],
                answer_type=answer_type,
                context_chunks=context_chunks,
                retrieved=list(state.get("retrieved", [])),
            )
            if strict_context_augmented:
                collector.set_context_ids([chunk.chunk_id for chunk in context_chunks])
        context_chunk_ids = [c.chunk_id for c in context_chunks]
        retrieved_chunks = list(state.get("retrieved", []))
        if retrieved_chunks:
            collector.set_retrieved_ids([chunk.chunk_id for chunk in retrieved_chunks])
            collector.set_chunk_snippets(self._build_chunk_snippet_map(retrieved_chunks))
            collector.set_chunk_page_hints(self._build_chunk_page_hint_map(retrieved_chunks))
        if context_chunks:
            collector.set_chunk_snippets(self._build_chunk_snippet_map(context_chunks))
            collector.set_chunk_page_hints(self._build_chunk_page_hint_map(context_chunks))
        collector.set_context_ids(context_chunk_ids)
        get_context_debug_stats = getattr(self._generator, "get_context_debug_stats", None)
        if callable(get_context_debug_stats):
            context_stats_obj = get_context_debug_stats(
                question=state["query"],
                chunks=context_chunks,
                complexity=state.get("complexity", QueryComplexity.SIMPLE),
                answer_type=answer_type,
            )
            if isinstance(context_stats_obj, tuple):
                context_stats_items = cast("tuple[object, ...]", context_stats_obj)
                if len(context_stats_items) != 2:
                    context_stats_items = ()
            else:
                context_stats_items = ()
            if len(context_stats_items) == 2:
                chunk_count_obj, budget_obj = context_stats_items
                if isinstance(chunk_count_obj, int) and isinstance(budget_obj, int):
                    collector.set_context_stats(chunk_count=chunk_count_obj, budget_tokens=budget_obj)
        if answer_type == "free_text" and not context_chunks:
            answer = self._insufficient_sources_answer(())
            collector.set_generation_mode("single_shot")
            collector.set_models(llm="insufficient-sources")
            logger.info(
                "free_text_no_context_fallback",
                extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
            )
        if (
            answer_type == "free_text"
            and bool(getattr(self._settings.pipeline, "premise_guard_enabled", True))
        ):
            terms_obj = getattr(self._settings.pipeline, "premise_guard_terms", [])
            disallowed_terms: list[str] = []
            if isinstance(terms_obj, list):
                terms = cast("list[object]", terms_obj)
                disallowed_terms = [text for term in terms if (text := str(term).strip())]
            guard = check_query_premise(
                query=state["query"],
                context_chunks=context_chunks,
                disallowed_terms=disallowed_terms,
            )
            if guard.triggered:
                answer = self._insufficient_sources_answer(tuple(context_chunk_ids[:1]))
                collector.set_generation_mode("single_shot")
                collector.set_models(llm="premise-guard")
                logger.warning(
                    "premise_guard_triggered",
                    extra={
                        "request_id": state.get("request_id"),
                        "question_id": state.get("question_id"),
                        "term": guard.term,
                        "answer_type": answer_type,
                    },
                )
        context_chunks = self._prune_boolean_context_for_single_doc_article(
            query=state["query"],
            answer_type=answer_type,
            doc_refs=state.get("doc_refs"),
            context_chunks=context_chunks,
        )
        context_chunks = self._boost_family_context_chunks(
            query=state["query"],
            answer_type=answer_type,
            context_chunks=context_chunks,
        )
        if answer_type in strict_types and bool(getattr(self._settings.pipeline, "strict_types_extraction_enabled", True)):
            strict_result = self._strict_answerer.answer(
                answer_type=answer_type,
                query=state["query"],
                context_chunks=context_chunks,
                max_chunks=int(getattr(self._settings.pipeline, "strict_types_extraction_max_chunks", 4)),
            )
            if strict_result is not None and strict_result.confident:
                answer = strict_result.answer.strip()
                extracted = True
                strict_cited_ids = list(strict_result.cited_chunk_ids)
                collector.set_generation_mode("single_shot")
                collector.set_models(llm="strict-extractor")

        if answer_type in strict_non_stream_types:
            if not answer:
                collector.set_generation_mode("single_shot")
                # Strict-types path fallback: non-stream LLM answer.
                with collector.timed("llm"):
                    generated_text, _citations = await self._generator.generate(
                        state["query"],
                        context_chunks,
                        model=state["model"],
                        max_tokens=int(state["max_tokens"]),
                        collector=collector,
                        complexity=state.get("complexity", QueryComplexity.SIMPLE),
                        answer_type=answer_type,
                        prompt_hint=prompt_hint,
                    )
                answer = generated_text.strip()
        else:
            if not answer:
                build_structured_answer = getattr(self._generator, "build_structured_free_text_answer", None)
                if callable(build_structured_answer):
                    built_obj = build_structured_answer(
                        question=state["query"],
                        chunks=context_chunks,
                        doc_refs=state.get("doc_refs"),
                    )
                    if isinstance(built_obj, str) and built_obj.strip():
                        answer = built_obj.strip()
                        collector.set_generation_mode("single_shot")
                        collector.set_models(llm="structured-extractor")
                        collector.mark_first_token()
                        writer({"type": "token", "text": answer})
            if not answer:
                collector.set_generation_mode("stream")
                answer_parts: list[str] = []
                first_token = True
                # Adaptive max_tokens: boost for multi-entity queries to prevent truncation.
                effective_max_tokens = int(state["max_tokens"])
                doc_ref_count = len([r for r in (state.get("doc_refs") or []) if str(r).strip()])
                if doc_ref_count >= 2:
                    effective_max_tokens = min(int(effective_max_tokens * 1.5), 1800)
                with collector.timed("llm"):
                    async for token in self._generator.generate_stream(
                        state["query"],
                        context_chunks,
                        model=state["model"],
                        max_tokens=effective_max_tokens,
                        collector=collector,
                        complexity=state.get("complexity", QueryComplexity.SIMPLE),
                        answer_type=answer_type,
                        prompt_hint=prompt_hint,
                    ):
                        if first_token:
                            collector.mark_first_token()
                        if first_token:
                            first_token = False
                        writer({"type": "token", "text": token})
                        answer_parts.append(token)
                streamed = True
                answer = "".join(answer_parts).strip()
                if answer:
                    streamed_raw = answer
                    cleanup_obj = getattr(self._generator, "cleanup_truncated_answer", None)
                    if callable(cleanup_obj):
                        cleaned_obj = cleanup_obj(answer)
                        if isinstance(cleaned_obj, str):
                            malformed_tail_detected = cleaned_obj.strip() != answer.strip()
                            if malformed_tail_detected:
                                collector.set_llm_diagnostics(malformed_tail_detected=True)
                        if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                            answer = cleaned_obj.strip()
                    strip_neg = getattr(self._generator, "strip_negative_subclaims", None)
                    should_strip_neg = answer_type == "free_text" and (
                        _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
                    )
                    if callable(strip_neg) and should_strip_neg:
                        stripped_obj = strip_neg(answer)
                        if isinstance(stripped_obj, str) and stripped_obj.strip():
                            answer = stripped_obj.strip()
                    if streamed_raw and not answer.strip():
                        answer = streamed_raw
                else:
                    # Rare provider anomaly: stream completes without tokens. Fall back once to non-stream generation.
                    collector.set_generation_mode("single_shot")
                    with collector.timed("llm"):
                        generated_text, _citations = await self._generator.generate(
                            state["query"],
                            context_chunks,
                            model=state["model"],
                            max_tokens=effective_max_tokens,
                            collector=collector,
                            complexity=state.get("complexity", QueryComplexity.SIMPLE),
                            answer_type=answer_type,
                            prompt_hint=prompt_hint,
                        )
                    answer = generated_text.strip()

        if answer_type in strict_types:
            if not answer:
                answer = self._strict_type_fallback(answer_type, tuple(context_chunk_ids[:1]))

            # Coerce strict formats (parse-safe; citations handled via telemetry "used pages").
            cited_ids_raw = strict_cited_ids or list(context_chunk_ids)
            coerced, extracted_ok = self._coerce_strict_type_format(answer, answer_type, cited_ids_raw)
            answer = coerced.strip()

            # Rare second-pass "repair" if the first LLM output was not parseable.
            if (
                not extracted_ok
                and strict_repair_enabled
                and not extracted
                and not self._is_unanswerable_strict_answer(answer)
                and answer_type in {"boolean", "number", "date", "name", "names"}
            ):
                repair_hint = _STRICT_REPAIR_HINT_TEMPLATE.format(answer_type=answer_type)
                with collector.timed("llm"):
                    repaired_text, _ = await self._generator.generate(
                        state["query"],
                        context_chunks,
                        model=state["model"],
                        max_tokens=min(int(state["max_tokens"]), 64),
                        collector=collector,
                        complexity=QueryComplexity.SIMPLE,
                        answer_type=answer_type,
                        prompt_hint=repair_hint,
                    )
                repaired, extracted_ok_2 = self._coerce_strict_type_format(repaired_text, answer_type, cited_ids_raw)
                if extracted_ok_2:
                    answer = repaired.strip()
                    logger.info(
                        "strict_repair_succeeded",
                        extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
                    )
                else:
                    logger.warning(
                        "strict_repair_failed",
                        extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
                    )

            if not answer:
                answer = self._strict_type_fallback(answer_type, tuple(context_chunk_ids[:1]))

            if self._is_unanswerable_strict_answer(answer):
                used_ids = []
                cited_ids = []
            else:
                cited_ids = (
                    list(strict_cited_ids)
                    if (extracted and strict_cited_ids)
                    else self._localize_strict_support_chunk_ids(
                        answer_type=answer_type,
                        answer=answer,
                        query=state["query"],
                        context_chunks=context_chunks,
                    )
                )
                used_ids = self._expand_page_spanning_support_chunk_ids(
                    chunk_ids=cited_ids,
                    context_chunks=context_chunks,
                )
                if not cited_ids:
                    logger.warning(
                        "strict_support_localization_failed",
                        extra={
                            "request_id": state.get("request_id"),
                            "question_id": state.get("question_id"),
                            "answer_type": answer_type,
                        },
                    )
            shaped_used_ids, support_shape_flags = self._apply_support_shape_policy(
                answer_type=answer_type,
                answer=answer,
                query=state["query"],
                context_chunks=context_chunks,
                cited_ids=cited_ids,
                support_ids=[],
            )
            if support_shape_flags:
                logger.warning(
                    "support_shape_flags_detected",
                    extra={
                        "request_id": state.get("request_id"),
                        "question_id": state.get("question_id"),
                        "answer_type": answer_type,
                        "flags": support_shape_flags,
                    },
                )
            collector.set_cited_ids(cited_ids)
            _q_lower_fam = re.sub(r"\s+", " ", str(state["query"] or "").strip()).lower()
            _family_recall_priority = (
                self._ENACTMENT_QUERY_RE.search(_q_lower_fam)
                or self._ADMIN_QUERY_RE.search(_q_lower_fam)
                or self._OUTCOME_QUERY_RE.search(_q_lower_fam)
                or _is_broad_enumeration_query(state["query"])
                or _is_named_commencement_query(state["query"])
                or _is_named_multi_title_lookup_query(state["query"])
            )
            if _family_recall_priority:
                final_used_ids = list(shaped_used_ids if shaped_used_ids else used_ids)
            else:
                final_used_ids = self._rerank_support_pages_within_selected_docs(
                    query=state["query"],
                    answer_type=answer_type,
                    context_chunks=context_chunks,
                    used_ids=shaped_used_ids if shaped_used_ids else used_ids,
                )
            final_used_ids = self._enhance_page_recall(
                query=state["query"],
                answer_type=answer_type,
                context_chunks=context_chunks,
                current_used_ids=final_used_ids,
            )
            citation_pages = self._extract_citation_pages(
                question=state["query"],
                answer=answer,
                answer_type=answer_type,
                context_chunks=context_chunks,
            )
            if citation_pages:
                collector.set_used_page_ids_override(citation_pages)
            else:
                collector.set_used_ids(final_used_ids)
            trimmed = self._trim_to_article_page(
                question=state["query"],
                answer_type=answer_type,
                context_chunks=context_chunks,
                current_page_ids=citation_pages if citation_pages else final_used_ids,
            )
            if trimmed:
                collector.set_used_page_ids_override(trimmed)
            citations: list[Citation] = []
            cited_ids = list(cited_ids)
        else:
            # Sanitize citations: remove any chunk IDs not present in context
            sanitize_citations = getattr(self._generator, "sanitize_citations", None)
            if callable(sanitize_citations):
                sanitized_obj = sanitize_citations(answer, context_chunk_ids)
                if isinstance(sanitized_obj, str):
                    answer = sanitized_obj
            cleanup_list_preamble = getattr(self._generator, "cleanup_list_answer_preamble", None)
            if callable(cleanup_list_preamble) and (
                _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
            ):
                cleaned_obj = cleanup_list_preamble(answer)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_list_items = getattr(self._generator, "cleanup_numbered_list_items", None)
            if callable(cleanup_list_items) and (
                _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
            ):
                cleaned_obj = cleanup_list_items(
                    answer,
                    question=state["query"],
                    common_elements=_is_common_elements_query(state["query"]),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_titles_only = getattr(self._generator, "cleanup_broad_enumeration_titles_only", None)
            query_title_refs = _extract_question_title_refs(state["query"])
            named_ref_query = len(query_title_refs) >= 2 and any(
                term in str(state["query"]).lower() for term in ("mention", "mentions", "reference", "references")
            )
            if (
                callable(cleanup_titles_only)
                and _is_broad_enumeration_query(state["query"])
                and not _is_registrar_enumeration_query(state["query"])
                and not named_ref_query
            ):
                cleaned_obj = cleanup_titles_only(answer, question=state["query"], chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_interpretative_items = getattr(self._generator, "cleanup_interpretative_provisions_enumeration_items", None)
            if (
                callable(cleanup_interpretative_items)
                and _is_broad_enumeration_query(state["query"])
                and "interpretative provisions" in str(state["query"]).lower()
            ):
                cleaned_obj = cleanup_interpretative_items(answer, chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_ref_items = getattr(self._generator, "cleanup_named_ref_enumeration_items", None)
            if callable(cleanup_named_ref_items) and named_ref_query and _is_broad_enumeration_query(state["query"]):
                cleaned_obj = cleanup_named_ref_items(answer, question=state["query"], chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_ruler_enactment_items = getattr(self._generator, "cleanup_ruler_enactment_enumeration_items", None)
            if callable(cleanup_ruler_enactment_items) and _is_ruler_enactment_query(state["query"]):
                cleaned_obj = cleanup_ruler_enactment_items(answer, chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_registrar_items = getattr(self._generator, "cleanup_registrar_enumeration_items", None)
            if callable(cleanup_registrar_items) and (
                _is_registrar_enumeration_query(state["query"])
                or (_is_citation_title_query(state["query"]) and _is_enumeration_query(state["query"]))
            ):
                cleaned_obj = cleanup_registrar_items(answer, chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_list_postamble = getattr(self._generator, "cleanup_list_answer_postamble", None)
            if callable(cleanup_list_postamble) and (
                _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
            ):
                cleaned_obj = cleanup_list_postamble(answer)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            strip_neg = getattr(self._generator, "strip_negative_subclaims", None)
            if callable(strip_neg) and _is_common_elements_query(state["query"]):
                stripped_obj = strip_neg(answer)
                if isinstance(stripped_obj, str) and stripped_obj.strip():
                    answer = stripped_obj.strip()
            cleanup_common_elements_answer = getattr(self._generator, "cleanup_common_elements_canonical_answer", None)
            if callable(cleanup_common_elements_answer) and _is_common_elements_query(state["query"]):
                cleaned_obj = cleanup_common_elements_answer(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_commencement = getattr(self._generator, "cleanup_named_commencement_answer", None)
            if callable(cleanup_named_commencement):
                cleaned_obj = cleanup_named_commencement(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_administration = getattr(self._generator, "cleanup_named_administration_answer", None)
            if callable(cleanup_named_administration):
                cleaned_obj = cleanup_named_administration(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_penalty = getattr(self._generator, "cleanup_named_penalty_answer", None)
            if callable(cleanup_named_penalty):
                cleaned_obj = cleanup_named_penalty(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_multi_title_lookup = getattr(self._generator, "cleanup_named_multi_title_lookup_answer", None)
            if callable(cleanup_named_multi_title_lookup):
                cleaned_obj = cleanup_named_multi_title_lookup(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_amendment = getattr(self._generator, "cleanup_named_amendment_answer", None)
            if callable(cleanup_named_amendment):
                cleaned_obj = cleanup_named_amendment(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_enactment_date = getattr(self._generator, "cleanup_named_enactment_date_answer", None)
            if callable(cleanup_named_enactment_date):
                cleaned_obj = cleanup_named_enactment_date(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_made_by = getattr(self._generator, "cleanup_named_made_by_answer", None)
            if callable(cleanup_named_made_by):
                cleaned_obj = cleanup_named_made_by(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_registrar_authority = getattr(self._generator, "cleanup_named_registrar_authority_answer", None)
            if callable(cleanup_named_registrar_authority):
                cleaned_obj = cleanup_named_registrar_authority(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_translation_requirement = getattr(self._generator, "cleanup_named_translation_requirement_answer", None)
            if callable(cleanup_named_translation_requirement):
                cleaned_obj = cleanup_named_translation_requirement(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_account_effective_dates = getattr(self._generator, "cleanup_account_effective_dates_answer", None)
            if callable(cleanup_account_effective_dates):
                cleaned_obj = cleanup_account_effective_dates(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_final_answer = getattr(self._generator, "cleanup_final_answer", None)
            if callable(cleanup_final_answer):
                cleaned_obj = cleanup_final_answer(answer)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    if cleaned_obj.strip() != answer.strip():
                        collector.set_llm_diagnostics(malformed_tail_detected=True)
                    answer = cleaned_obj.strip()

            citations = self._generator.extract_citations(answer, context_chunks)
            cited_ids = self._generator.extract_cited_chunk_ids(answer)
            is_unanswerable_free_text = self._is_unanswerable_free_text_answer(answer)
            support_ids: list[str] = []
            prefer_citation_trace = _is_named_multi_title_lookup_query(state["query"]) or _is_named_amendment_query(
                state["query"]
            )
            if not cited_ids and not is_unanswerable_free_text:
                support_ids = self._localize_free_text_support_chunk_ids(
                    answer=answer,
                    query=state["query"],
                    context_chunks=context_chunks,
                )
                cited_ids = list(support_ids)
                if support_ids and not citations:
                    citations = self._citations_from_chunk_ids(
                        chunk_ids=support_ids,
                        context_chunks=context_chunks,
                    )
                elif not support_ids:
                    logger.warning(
                        "free_text_support_localization_failed",
                        extra={
                            "request_id": state.get("request_id"),
                            "question_id": state.get("question_id"),
                            "answer_type": answer_type,
                        },
                    )
            elif not is_unanswerable_free_text and not prefer_citation_trace:
                support_ids = self._localize_free_text_support_chunk_ids(
                    answer=answer,
                    query=state["query"],
                    context_chunks=context_chunks,
                )
            if is_unanswerable_free_text:
                citations = []
                cited_ids = []
                support_ids = []
                collector.set_retrieved_ids([])
                collector.set_context_ids([])
            elif support_ids:
                support_ids = self._suppress_named_administration_family_orphan_support_ids(
                    query=state["query"],
                    cited_ids=cited_ids,
                    support_ids=support_ids,
                    context_chunks=context_chunks,
                )
            collector.set_cited_ids(cited_ids)
            shaped_used_ids, support_shape_flags = self._apply_support_shape_policy(
                answer_type=answer_type,
                answer=answer,
                query=state["query"],
                context_chunks=context_chunks,
                cited_ids=cited_ids,
                support_ids=support_ids,
            )
            if support_shape_flags:
                logger.warning(
                    "support_shape_flags_detected",
                    extra={
                        "request_id": state.get("request_id"),
                        "question_id": state.get("question_id"),
                        "answer_type": answer_type,
                        "flags": support_shape_flags,
                    },
                )
            _q_lower_fam2 = re.sub(r"\s+", " ", str(state["query"] or "").strip()).lower()
            _family_recall_priority2 = (
                self._ENACTMENT_QUERY_RE.search(_q_lower_fam2)
                or self._ADMIN_QUERY_RE.search(_q_lower_fam2)
                or self._OUTCOME_QUERY_RE.search(_q_lower_fam2)
                or _is_broad_enumeration_query(state["query"])
                or _is_named_commencement_query(state["query"])
                or _is_named_multi_title_lookup_query(state["query"])
            )
            if _family_recall_priority2:
                final_used_ids = list(shaped_used_ids)
            else:
                final_used_ids = self._rerank_support_pages_within_selected_docs(
                    query=state["query"],
                    answer_type=answer_type,
                    context_chunks=context_chunks,
                    used_ids=shaped_used_ids,
                )
            final_used_ids = self._enhance_page_recall(
                query=state["query"],
                answer_type=answer_type,
                context_chunks=context_chunks,
                current_used_ids=final_used_ids,
            )
            citation_pages = self._extract_citation_pages(
                question=state["query"],
                answer=answer,
                answer_type=answer_type,
                context_chunks=context_chunks,
            )
            if citation_pages:
                collector.set_used_page_ids_override(citation_pages)
            else:
                collector.set_used_ids(final_used_ids)
            trimmed = self._trim_to_article_page(
                question=state["query"],
                answer_type=answer_type,
                context_chunks=context_chunks,
                current_page_ids=citation_pages if citation_pages else final_used_ids,
            )
            if trimmed:
                collector.set_used_page_ids_override(trimmed)
            if answer_type == "free_text" and streamed and answer.strip():
                writer({"type": "answer_final", "text": answer})

        logger.info(
            "Generated answer %d chars with %d citations (strict_extracted=%s)",
            len(answer),
            len(citations),
            extracted,
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {
            "answer": answer,
            "citations": citations,
            "cited_chunk_ids": cited_ids,
            "streamed": streamed,
        }

    async def _verify(self, state: RAGState) -> dict[str, object]:
        verifier_settings = getattr(self._settings, "verifier", None)
        if verifier_settings is not None and not bool(getattr(verifier_settings, "enabled", True)):
            return {}
        if self._verifier is None:
            return {}

        answer = str(state.get("answer") or "").strip()
        cited_ids = list(state.get("cited_chunk_ids", []))

        # Skip LLM verifier entirely for strict types — deterministic coerce already ran in _generate
        answer_type = str(state.get("answer_type") or "free_text").strip().lower()
        if answer_type in {"boolean", "number", "date", "name", "names"}:
            return {}
        should_verify = self._verifier.should_verify(answer, cited_ids, force=False)
        if not should_verify:
            return {}

        collector = state["collector"]
        context_chunks = state.get("context_chunks", [])
        with collector.timed("verify"):
            verification = await self._verifier.verify(
                state["query"],
                answer,
                context_chunks,
            )

        # For free_text, verifier is audit-only (post-hoc revision cannot affect already-streamed output).
        if answer_type == "free_text":
            logger.info(
                "Verifier audit grounded=%s unsupported_claims=%d",
                verification.is_grounded,
                len(verification.unsupported_claims),
            )
            return {}

        next_answer = answer
        if not verification.is_grounded:
            if verification.revised_answer:
                next_answer = verification.revised_answer.strip()
                logger.info(
                    "Verifier revised answer grounded=%s unsupported_claims=%d",
                    verification.is_grounded,
                    len(verification.unsupported_claims),
                )
            else:
                # Fail-safe: keep strict-type output format deterministic even when sources are insufficient.
                next_answer = self._strict_type_fallback(answer_type, cited_ids)

        effective_cited_ids = self._generator.extract_cited_chunk_ids(next_answer) or cited_ids
        next_answer, _ = self._coerce_strict_type_format(next_answer, answer_type, effective_cited_ids)
        next_answer = next_answer.strip()

        # Sanitize citations after coercion
        context_chunk_ids = [c.chunk_id for c in context_chunks]
        sanitize_citations = getattr(self._generator, "sanitize_citations", None)
        if callable(sanitize_citations):
            sanitized_obj = sanitize_citations(next_answer, context_chunk_ids)
            if isinstance(sanitized_obj, str):
                next_answer = sanitized_obj

        if next_answer == answer:
            return {}

        citations = self._generator.extract_citations(next_answer, context_chunks)
        next_cited_ids = self._generator.extract_cited_chunk_ids(next_answer)
        if self._is_unanswerable_strict_answer(next_answer):
            citations = []
            next_cited_ids = []
            collector.set_retrieved_ids([])
            collector.set_context_ids([])
        collector.set_cited_ids(next_cited_ids)
        return {
            "answer": next_answer,
            "citations": citations,
            "cited_chunk_ids": next_cited_ids,
        }

    async def _emit(self, state: RAGState) -> dict[str, object]:
        """Emit answer tokens for strict-types flows that generated non-streaming output."""
        if bool(state.get("streamed", False)):
            return {}

        answer = str(state.get("answer") or "")
        if not answer.strip():
            return {}

        collector = state["collector"]
        writer = self._get_stream_writer_or_noop()
        collector.mark_first_token()
        writer({"type": "token", "text": answer})
        writer({"type": "answer_final", "text": answer})
        return {"streamed": True}

    async def _finalize(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        telemetry = collector.finalize()
        writer = self._get_stream_writer_or_noop()
        writer({"type": "telemetry", "payload": telemetry.model_dump()})
        return {"telemetry": telemetry}

    @staticmethod
    def _get_stream_writer_or_noop() -> Callable[[dict[str, object]], None]:
        from rag_challenge.core import pipeline as pipeline_module

        try:
            writer = pipeline_module.get_stream_writer()
        except RuntimeError:
            return lambda _: None
        return cast("Callable[[dict[str, object]], None]", writer)

    def _expand_retry_query(self, state: RAGState) -> str:
        base_query = state["query"]
        anchors: list[str] = []
        for chunk in state.get("reranked", []):
            if chunk.section_path:
                anchors.append(chunk.section_path)
        if not anchors:
            return base_query
        unique_anchors = list(dict.fromkeys(anchors))
        max_anchors = max(0, int(self._settings.pipeline.retry_query_max_anchors))
        return f"{base_query} {' '.join(unique_anchors[:max_anchors])}".strip()
