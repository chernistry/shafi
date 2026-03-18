# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportPrivateUsage=false
"""Grounding evidence selector sidecar.

Selects minimal evidentiary page IDs independently from the answer path.
Runs after answer generation, using the query, answer, scope prediction,
retrieved context chunks, support-fact index, and page collection to
predict the best grounding pages.

This module must NEVER modify the answer text, citations, or context chunks.
It only produces ``used_page_ids``.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING, cast

from qdrant_client import models

from rag_challenge.core.grounding.query_scope_classifier import classify_query_scope
from rag_challenge.models.schemas import (
    QueryScopePrediction,
    RetrievedPage,
    RetrievedSupportFact,
    ScopeMode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.config.settings import PipelineSettings
    from rag_challenge.core.embedding import EmbeddingClient
    from rag_challenge.core.qdrant import QdrantStore
    from rag_challenge.core.retriever import HybridRetriever
    from rag_challenge.core.sparse_bm25 import BM25SparseEncoder
    from rag_challenge.models import RankedChunk

logger = logging.getLogger(__name__)

_CASE_REF_RE = re.compile(r"\b[A-Z]{2,4}\s+\d{3}/\d{4}\b", re.IGNORECASE)


def _normalize_answer_value(answer: str, answer_type: str) -> str:
    """Normalize an answer value for matching against support facts.

    Args:
        answer: Raw answer text.
        answer_type: Normalized answer type.

    Returns:
        Cleaned, normalized answer value.
    """
    text = (answer or "").strip()
    if not text:
        return ""
    if answer_type == "boolean":
        low = text.lower()
        if low in {"yes", "true"}:
            return "true"
        if low in {"no", "false"}:
            return "false"
        if low in {"null", "none", "no information"}:
            return ""
    return re.sub(r"\s+", " ", text).strip(" .;:")


class GroundingEvidenceSelector:
    """Selects minimal evidentiary pages using support facts and page retrieval.

    Args:
        retriever: The hybrid retriever for page and support-fact queries.
        store: The Qdrant store for collection access.
        embedder: The embedding client for query vectorization.
        sparse_encoder: Optional BM25 sparse encoder.
        pipeline_settings: Pipeline settings for grounding configuration.
    """

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        store: QdrantStore,
        embedder: EmbeddingClient,
        sparse_encoder: BM25SparseEncoder | None,
        pipeline_settings: PipelineSettings,
    ) -> None:
        self._retriever = retriever
        self._store = store
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder
        self._settings = pipeline_settings

    async def select_page_ids(
        self,
        *,
        query: str,
        answer: str,
        answer_type: str,
        context_chunks: Sequence[RankedChunk],
    ) -> list[str] | None:
        """Select minimal evidentiary page IDs for grounding.

        Args:
            query: Raw user question.
            answer: Final answer text (already generated).
            answer_type: Normalized answer type.
            context_chunks: Ranked context chunks used for answering.

        Returns:
            List of page IDs, empty list for null/negative, or None to fall back to legacy.
        """
        scope = classify_query_scope(query, answer_type)

        # Force empty grounding on null answers for negative/unanswerable queries
        if scope.should_force_empty_grounding_on_null:
            answer_low = (answer or "").strip().lower()
            if not answer_low or answer_low in {"null", "none", "no information", "n/a"}:
                return []

        # Only override legacy for scope modes where the sidecar demonstrably helps:
        # - compare_pair: needs 1 page per doc across multiple docs
        # - full_case_files: needs all-doc coverage
        # For single-field, explicit-page, etc., the legacy path is already good.
        _SIDECAR_ACTIVE_MODES = {ScopeMode.COMPARE_PAIR, ScopeMode.FULL_CASE_FILES}
        if scope.scope_mode not in _SIDECAR_ACTIVE_MODES:
            return None  # fall back to legacy

        doc_ids = self._select_doc_scope(
            query=query,
            scope=scope,
            context_chunks=context_chunks,
        )
        if not doc_ids:
            return None  # fall back to legacy

        # Collect pages already cited by the answer path — strong prior.
        # chunk_id format: "doc_id:page_num:chunk_idx:hash"
        # page_id format: "doc_id_{page_num+1}" (1-indexed)
        context_page_ids: set[str] = set()
        for chunk in context_chunks:
            if not chunk.doc_id:
                continue
            parts = chunk.chunk_id.split(":")
            if len(parts) >= 2:
                try:
                    page_num = int(parts[1])
                    context_page_ids.add(f"{chunk.doc_id}_{page_num + 1}")
                except (ValueError, IndexError):
                    pass

        answer_value = _normalize_answer_value(answer, answer_type)

        support_candidates = await self._retrieve_support_fact_candidates(
            query=query,
            scope=scope,
            doc_ids=doc_ids,
            answer_value=answer_value,
        )

        page_candidates = await self._retrieve_page_candidates(
            query=query,
            scope=scope,
            doc_ids=doc_ids,
        )

        scored = self._score_candidates(
            query=query,
            answer_type=answer_type,
            scope=scope,
            answer_value=answer_value,
            support_candidates=support_candidates,
            page_candidates=page_candidates,
            context_page_ids=context_page_ids,
        )

        selected = self._select_minimal_pages(scope=scope, scored=scored)
        if selected:
            return selected

        return None  # fall back to legacy

    def _select_doc_scope(
        self,
        *,
        query: str,
        scope: QueryScopePrediction,
        context_chunks: Sequence[RankedChunk],
    ) -> list[str]:
        """Determine document scope from context chunks and query.

        Args:
            query: Raw user question.
            scope: Query scope prediction.
            context_chunks: Ranked context chunks from the answer path.

        Returns:
            Sorted list of document IDs in scope.
        """
        if scope.scope_mode is ScopeMode.FULL_CASE_FILES:
            # For full-case, use all doc_ids from context
            return sorted({chunk.doc_id for chunk in context_chunks if chunk.doc_id})

        # Default: use doc footprint from answer path
        return sorted({chunk.doc_id for chunk in context_chunks if chunk.doc_id})

    async def _retrieve_support_fact_candidates(
        self,
        *,
        query: str,
        scope: QueryScopePrediction,
        doc_ids: Sequence[str],
        answer_value: str,
    ) -> list[RetrievedSupportFact]:
        """Retrieve support-fact candidates from the support-fact collection.

        Args:
            query: Raw user question.
            scope: Query scope prediction.
            doc_ids: Document IDs in scope.
            answer_value: Normalized answer value for matching.

        Returns:
            List of retrieved support facts ranked by relevance.
        """
        if not doc_ids:
            return []

        coll = self._store.support_fact_collection_name
        try:
            exists = await self._store.client.collection_exists(coll)
            if not exists:
                return []
        except Exception:
            return []

        must: list[models.Condition] = [
            models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))
        ]
        if scope.target_page_roles:
            must.append(
                models.FieldCondition(key="page_role", match=models.MatchAny(any=scope.target_page_roles))
            )
        if scope.hard_anchor_strings:
            must.append(
                models.FieldCondition(key="scope_ref", match=models.MatchAny(any=list(scope.hard_anchor_strings)))
            )

        top_k = self._settings.grounding_support_fact_top_k
        embed_text = f"{query} | answer {answer_value}" if answer_value else query
        qvec = await self._embedder.embed_query(embed_text)
        where = models.Filter(must=must)

        prefetch = [
            models.Prefetch(query=qvec, using="dense", limit=top_k, filter=where),
        ]

        bm25_enabled = getattr(self._retriever, "_bm25_enabled", False)
        if bm25_enabled and self._sparse_encoder is not None:
            sparse_text = f"{query} {answer_value}".strip()
            anchor_refs = list(scope.hard_anchor_strings) if scope.hard_anchor_strings else []
            sparse_query = self._retriever._build_sparse_query(query=sparse_text, extracted_refs=anchor_refs)
            svec = self._sparse_encoder.encode_query(sparse_query)
            prefetch.append(
                models.Prefetch(query=svec, using="bm25", limit=top_k, filter=where),
            )

        try:
            result = await self._store.client.query_points(
                collection_name=coll,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=self._retriever._resolve_fusion_method()),
                limit=top_k,
                with_payload=True,
            )
        except Exception:
            logger.warning("Support-fact retrieval failed", exc_info=True)
            return []

        return self._map_support_fact_results(result)

    async def _retrieve_page_candidates(
        self,
        *,
        query: str,
        scope: QueryScopePrediction,
        doc_ids: Sequence[str],
    ) -> list[RetrievedPage]:
        """Retrieve page candidates using existing retriever with grounding filters.

        Args:
            query: Raw user question.
            scope: Query scope prediction.
            doc_ids: Document IDs in scope.

        Returns:
            List of retrieved pages ranked by relevance.
        """
        return await self._retriever.retrieve_pages(
            query,
            top_k=self._settings.grounding_page_top_k,
            doc_ids=list(doc_ids),
            page_roles=list(scope.target_page_roles) if scope.target_page_roles else None,
            article_refs=list(scope.hard_anchor_strings) if scope.hard_anchor_strings else None,
        )

    def _score_candidates(
        self,
        *,
        query: str,
        answer_type: str,
        scope: QueryScopePrediction,
        answer_value: str,
        support_candidates: Sequence[RetrievedSupportFact],
        page_candidates: Sequence[RetrievedPage],
        context_page_ids: set[str] | None = None,
    ) -> dict[str, float]:
        """Score page candidates using support facts and page retrieval signals.

        Args:
            query: Raw user question.
            answer_type: Normalized answer type.
            scope: Query scope prediction.
            answer_value: Normalized answer value.
            support_candidates: Retrieved support facts.
            page_candidates: Retrieved pages.
            context_page_ids: Page IDs cited by the answer path (strong prior).

        Returns:
            Dict mapping page_id to aggregate score, sorted descending.
        """
        scores: dict[str, float] = defaultdict(float)

        # Page retrieval score (primary signal)
        for page in page_candidates:
            scores[page.page_id] += float(page.score)

        # Strong anchor: pages the answer path already cited deserve a large bonus.
        # The legacy path chose these pages based on full context; the sidecar should
        # only override when it has very strong evidence pointing elsewhere.
        cited = context_page_ids or set()
        for pid in cited:
            scores[pid] += 3.0

        # Support-fact score (secondary signal — reduced from 2.0 to 0.8 to avoid
        # schedule/interpretation pages dominating over the actual article page)
        for fact in support_candidates:
            scores[fact.page_id] += 0.8 * float(fact.score)
            if answer_value and answer_value.casefold() in (fact.normalized_value or "").casefold():
                scores[fact.page_id] += 1.0
            if fact.page_role and fact.page_role in scope.target_page_roles:
                scores[fact.page_id] += 0.5
            if fact.fact_type == "date_of_issue" and "date of issue" in query.lower():
                scores[fact.page_id] += 1.0
            if answer_type == "number" and fact.fact_type in {"claim_amount", "costs_awarded", "penalty"}:
                scores[fact.page_id] += 0.5

        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    def _select_minimal_pages(
        self,
        *,
        scope: QueryScopePrediction,
        scored: dict[str, float],
    ) -> list[str]:
        """Select minimal page set from scored candidates respecting scope budget.

        Args:
            scope: Query scope prediction with page budget.
            scored: Page IDs scored by relevance (sorted descending).

        Returns:
            List of selected page IDs.
        """
        if not scored:
            return []

        ordered = list(scored.keys())

        if scope.scope_mode is ScopeMode.EXPLICIT_PAGE:
            return ordered[:1]

        if scope.scope_mode is ScopeMode.FULL_CASE_FILES:
            per_doc: dict[str, str] = {}
            for page_id in ordered:
                doc_id, _, _ = page_id.rpartition("_")
                if doc_id and doc_id not in per_doc:
                    per_doc[doc_id] = page_id
            return list(per_doc.values())

        if scope.scope_mode is ScopeMode.COMPARE_PAIR:
            per_doc_cmp: dict[str, str] = {}
            for page_id in ordered:
                doc_id, _, _ = page_id.rpartition("_")
                if doc_id and doc_id not in per_doc_cmp:
                    per_doc_cmp[doc_id] = page_id
            return list(per_doc_cmp.values())[: max(2, scope.page_budget)]

        return ordered[: scope.page_budget]

    @staticmethod
    def _map_support_fact_results(result: object) -> list[RetrievedSupportFact]:
        """Map Qdrant query result to RetrievedSupportFact list.

        Args:
            result: Qdrant query_points result object.

        Returns:
            List of RetrievedSupportFact instances.
        """
        points = getattr(result, "points", None) or getattr(result, "result", None) or []
        mapped: list[RetrievedSupportFact] = []
        for pt in points:
            payload = getattr(pt, "payload", None)
            if not isinstance(payload, dict):
                continue
            payload_dict = cast("dict[str, object]", payload)
            try:
                page_num = int(str(payload_dict.get("page_num") or 0))
            except (TypeError, ValueError):
                page_num = 0
            try:
                score = float(str(getattr(pt, "score", 0.0) or 0.0))
            except (TypeError, ValueError):
                score = 0.0
            mapped.append(
                RetrievedSupportFact(
                    fact_id=str(payload_dict.get("fact_id") or ""),
                    doc_id=str(payload_dict.get("doc_id") or ""),
                    page_id=str(payload_dict.get("page_id") or ""),
                    page_num=page_num,
                    doc_title=str(payload_dict.get("doc_title") or ""),
                    fact_type=str(payload_dict.get("fact_type") or ""),
                    normalized_value=str(payload_dict.get("normalized_value") or ""),
                    quote_text=str(payload_dict.get("quote_text") or ""),
                    page_role=str(payload_dict.get("page_role") or ""),
                    page_family=str(payload_dict.get("page_family") or ""),
                    score=score,
                )
            )
        return mapped
