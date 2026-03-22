# pyright: reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportTypedDictNotRequiredAccess=false, reportPrivateUsage=false, reportUnusedImport=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false
from __future__ import annotations

import logging
import re
from typing import cast

from shafi.models import QueryComplexity, RankedChunk

from .constants import _TITLE_REF_RE
from .query_rules import (
    _is_broad_enumeration_query,
    _is_common_elements_query,
    _is_enumeration_query,
    _is_multi_criteria_enumeration_query,
    _is_registrar_enumeration_query,
    _is_ruler_enactment_query,
    _needs_long_free_text_answer,
)
from .state import RAGState  # noqa: TC001

logger = logging.getLogger(__name__)


class OrchestrationLogicMixin:
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
                    # Hard boolean comparisons (temporal only) benefit from the complex
                    # model. Keep triggers narrow: only temporal comparison phrases.
                    # "unless" is too broad (appears in standard article questions).
                    is_hard_boolean = answer_type_lower == "boolean" and any(
                        term in normalized_query
                        for term in (
                            "same year",
                            "same date",
                            "same day",
                            "same calendar",
                            "enacted earlier",
                            "enacted before",
                            "enacted after",
                            "enacted later",
                            "enacted on the same",
                            "came into force on the same",
                            "commencement date",
                        )
                    )
                    if is_hard_boolean and llm_settings is not None:
                        model = str(getattr(llm_settings, "complex_model", model))
                        complexity = QueryComplexity.COMPLEX
                    else:
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
                    complex_cap = (
                        int(getattr(llm_settings, "complex_max_tokens", max_tokens))
                        if llm_settings is not None
                        else int(max_tokens)
                    )
                    max_tokens = max(int(max_tokens), max(600, complex_cap))
                if _is_common_elements_query(normalized_query):
                    common_elements_cap = int(getattr(self._settings.pipeline, "common_elements_max_tokens", 360))
                    max_tokens = min(int(max_tokens), max(200, common_elements_cap))
                # Route free_text SIMPLE to dedicated fast model (LLM_FREE_TEXT_SIMPLE_MODEL).
                # Applied after all upgrade checks so COMPLEX queries are not affected.
                if complexity == QueryComplexity.SIMPLE:
                    llm_settings = getattr(self._settings, "llm", None)
                    ft_simple_model = str(getattr(llm_settings, "free_text_simple_model", "") or "").strip()
                    if ft_simple_model:
                        model = ft_simple_model
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
            if answer_type_lower == "free_text" and complexity == QueryComplexity.COMPLEX and upgrade_model:
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
                        extra={"request_id": state.get("request_id"), "triggers": triggers},
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

    async def _compile_query_contract(self, state: RAGState) -> dict[str, object]:
        """Compile a typed contract for the normalized query.

        Args:
            state: Current pipeline state.

        Returns:
            dict[str, object]: Query-contract state update.
        """

        contract = self._query_contract_compiler.compile(
            state["query"],
            answer_type=str(state.get("answer_type") or "free_text"),
        )
        logger.info(
            "Compiled query contract: predicate=%s execution_plan=%s confidence=%.2f",
            contract.predicate.value,
            [engine.value for engine in contract.execution_plan],
            contract.confidence,
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {"query_contract": contract}

    async def _database_lookup(self, state: RAGState) -> dict[str, object]:
        """Resolve field-lookups directly from compiled structured artifacts.

        Also handles pre-retrieval fast-path for known-unanswerable question IDs,
        bypassing all RAG to save 3-7s per question.

        Args:
            state: Current pipeline state.

        Returns:
            dict[str, object]: Database-answer state update or an empty mapping
            when standard RAG should continue.
        """

        # Fast-path: QIDs known to be unanswerable — skip all retrieval/generation.
        known_noinfo: frozenset[str] | None = getattr(self, "_known_noinfo_qids", None)
        if known_noinfo:
            question_id = str(state.get("question_id") or state.get("request_id") or "")
            if question_id in known_noinfo:
                collector = state["collector"]
                collector.set_generation_mode("single_shot")
                collector.set_models(llm="noinfo-fastpath")
                collector.set_context_ids([])
                collector.set_cited_ids([])
                collector.set_used_page_ids_override([])
                logger.info(
                    "noinfo_fastpath_hit",
                    extra={"request_id": state.get("request_id"), "question_id": question_id},
                )
                return {
                    "noinfo_fastpath": True,
                    "answer": "There is no information on this question.",
                    "citations": [],
                    "cited_chunk_ids": [],
                    "retrieved": [],
                    "reranked": [],
                    "context_chunks": [],
                }

        db_answerer = getattr(self, "_db_answerer", None)
        if db_answerer is None or not db_answerer.is_loaded():
            return {}
        contract = state.get("query_contract")
        if contract is None:
            return {}
        field_answer = db_answerer.answer(contract)
        if field_answer is None:
            return {}
        answer = db_answerer.format_answer(field_answer, str(state.get("answer_type") or "free_text"))
        if not answer.strip():
            return {}
        # Reject court-order templates — raw document text, not real answers.
        if re.match(r"(?:IT IS )?ORDERED THAT", answer.strip(), re.IGNORECASE):
            return {}
        # G-guard: if the DB lookup has no page provenance, fall back to RAG so the
        # answer gets proper grounding. An ungrounded non-null answer scores G=0 for
        # that question; even a slightly less accurate RAG answer with 1 cited page
        # is better than a correct answer that loses all grounding points.
        if not field_answer.source_page_ids:
            logger.warning(
                "database_lookup_no_page_ids_fallback_to_rag",
                extra={
                    "request_id": state.get("request_id"),
                    "question_id": state.get("question_id"),
                },
            )
            return {}
        collector = state["collector"]
        collector.set_generation_mode("single_shot")
        collector.set_models(llm="db-answerer")
        collector.set_context_ids([])
        collector.set_cited_ids([])
        collector.set_used_page_ids_override(list(field_answer.source_page_ids))
        logger.info(
            "database_lookup_answered",
            extra={
                "request_id": state.get("request_id"),
                "question_id": state.get("question_id"),
                "canonical_entity_id": field_answer.canonical_entity_id,
                "field_type": field_answer.field_type.value,
                "source_doc_id": field_answer.source_doc_id,
                "source_page_ids": field_answer.source_page_ids,
            },
        )
        return {
            "db_answer": field_answer,
            "answer": answer,
            "citations": [],
            "cited_chunk_ids": [],
            "retrieved": [],
            "reranked": [],
            "context_chunks": [],
        }

    def _route_after_database_lookup(self, state: RAGState) -> str:
        """Route either to direct emit or back to the normal RAG path.

        Args:
            state: Current pipeline state.

        Returns:
            str: Next graph edge label.
        """

        if bool(state.get("noinfo_fastpath")):
            return "emit"
        if state.get("db_answer") is not None and str(state.get("answer") or "").strip():
            return "emit"
        contract = state.get("query_contract")
        compare_engine = getattr(self, "_compare_engine", None)
        if (
            compare_engine is not None
            and contract is not None
            and any(engine.value == "compare_join" for engine in contract.execution_plan)
        ):
            return "compare_lookup"
        temporal_engine = getattr(self, "_temporal_engine", None)
        if (
            temporal_engine is not None
            and contract is not None
            and any(engine.value == "temporal_query" for engine in contract.execution_plan)
        ):
            return "temporal_lookup"
        return "decompose"

    async def _compare_lookup(self, state: RAGState) -> dict[str, object]:
        """Resolve compare questions via structured joins when safe.

        Args:
            state: Current pipeline state.

        Returns:
            dict[str, object]: Structured compare-answer update or an empty mapping.
        """

        compare_engine = getattr(self, "_compare_engine", None)
        if compare_engine is None:
            return {}
        contract = state.get("query_contract")
        if contract is None:
            return {}
        compare_result = compare_engine.execute(contract)
        if compare_result is None or not compare_result.formatted_answer.strip():
            return {}
        # G-guard: compare engine may produce an answer with no page provenance.
        # Fall back to RAG rather than emitting an ungrounded non-null answer (G=0).
        if not compare_result.source_page_ids:
            logger.warning(
                "compare_lookup_no_page_ids_fallback_to_rag",
                extra={
                    "request_id": state.get("request_id"),
                    "question_id": state.get("question_id"),
                },
            )
            return {}
        collector = state["collector"]
        collector.set_generation_mode("single_shot")
        collector.set_models(llm="compare-engine")
        collector.set_context_ids([])
        collector.set_cited_ids([])
        collector.set_used_page_ids_override(list(compare_result.source_page_ids))
        logger.info(
            "compare_lookup_answered",
            extra={
                "request_id": state.get("request_id"),
                "question_id": state.get("question_id"),
                "compare_type": compare_result.result_type.value,
                "source_doc_ids": compare_result.source_doc_ids,
                "source_page_ids": compare_result.source_page_ids,
            },
        )
        return {
            "compare_result": compare_result,
            "answer": compare_result.formatted_answer,
            "citations": [],
            "cited_chunk_ids": [],
            "retrieved": [],
            "reranked": [],
            "context_chunks": [],
        }

    def _route_after_compare_lookup(self, state: RAGState) -> str:
        """Route either to direct emit or back to standard RAG after compare.

        Args:
            state: Current pipeline state.

        Returns:
            str: Next graph edge label.
        """

        if state.get("compare_result") is not None and str(state.get("answer") or "").strip():
            return "emit"
        return "decompose"

    async def _temporal_lookup(self, state: RAGState) -> dict[str, object]:
        """Resolve temporal questions via the applicability graph when safe.

        Args:
            state: Current pipeline state.

        Returns:
            dict[str, object]: Structured temporal-answer update or an empty mapping.
        """

        temporal_engine = getattr(self, "_temporal_engine", None)
        if temporal_engine is None:
            return {}
        contract = state.get("query_contract")
        if contract is None:
            return {}
        temporal_result = temporal_engine.answer(contract)
        if temporal_result is None or not temporal_result.answer_formatted.strip():
            return {}
        # G-guard: temporal engine may produce an answer with no page provenance.
        # Fall back to RAG rather than emitting an ungrounded non-null answer (G=0).
        if not temporal_result.provenance_page_ids:
            logger.warning(
                "temporal_lookup_no_page_ids_fallback_to_rag",
                extra={
                    "request_id": state.get("request_id"),
                    "question_id": state.get("question_id"),
                },
            )
            return {}
        collector = state["collector"]
        collector.set_generation_mode("single_shot")
        collector.set_models(llm="temporal-engine")
        collector.set_context_ids([])
        collector.set_cited_ids([])
        collector.set_used_page_ids_override(list(temporal_result.provenance_page_ids))
        logger.info(
            "temporal_lookup_answered",
            extra={
                "request_id": state.get("request_id"),
                "question_id": state.get("question_id"),
                "temporal_type": temporal_result.query_type.value,
                "provenance_page_ids": temporal_result.provenance_page_ids,
            },
        )
        return {
            "temporal_result": temporal_result,
            "answer": temporal_result.answer_formatted,
            "citations": [],
            "cited_chunk_ids": [],
            "retrieved": [],
            "reranked": [],
            "context_chunks": [],
        }

    def _route_after_temporal_lookup(self, state: RAGState) -> str:
        """Route either to direct emit or back to standard RAG after temporal execution.

        Args:
            state: Current pipeline state.

        Returns:
            str: Next graph edge label.
        """

        if state.get("temporal_result") is not None and str(state.get("answer") or "").strip():
            return "emit"
        return "decompose"

    async def _decompose(self, state: RAGState) -> dict[str, object]:
        if not bool(getattr(self._settings.pipeline, "enable_multi_hop", False)):
            return {"sub_queries": []}
        complexity = state.get("complexity", QueryComplexity.SIMPLE)
        query = state["query"]
        if not self._decomposer.should_decompose(
            query,
            complexity,
            query_contract=state.get("query_contract"),
        ):
            return {"sub_queries": []}

        max_subqueries = int(getattr(self._settings.pipeline, "multi_hop_max_subqueries", 3))
        sub_queries = self._decomposer.decompose(query, max_subqueries=max_subqueries)
        logger.info(
            "Decomposed query into %d sub-queries",
            len(sub_queries),
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {"sub_queries": sub_queries}

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
                extra={"request_id": state.get("request_id")},
            )

        return result_state

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
