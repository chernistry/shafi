# pyright: reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportTypedDictNotRequiredAccess=false, reportPrivateUsage=false, reportUnusedImport=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false
from __future__ import annotations

import logging
from typing import cast

from rag_challenge.models import QueryComplexity, RankedChunk

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
