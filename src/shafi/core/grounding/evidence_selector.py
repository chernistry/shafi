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

from shafi.core.grounding.authority_priors import select_authoritative_single_page
from shafi.core.grounding.condition_audit import ConditionAuditResult, audit_candidate_pages
from shafi.core.grounding.evidence_portfolio import (
    EvidencePortfolio,
    PortfolioCandidate,
    build_law_bundle_candidate,
    select_best_page_set,
)
from shafi.core.grounding.necessity_pruner import prune_redundant_pages
from shafi.core.grounding.page_rank_merge import (
    merge_ranked_candidate_subset,
    ordered_rankable_candidate_page_ids,
)
from shafi.core.grounding.page_semantic_lane import (
    authority_signal_score,
    rerank_pages_with_shadow_signal,
    select_semantic_page_set,
)
from shafi.core.grounding.query_scope_classifier import (
    classify_query_scope,
    extract_explicit_page_numbers,
)
from shafi.core.grounding.relevance_verifier import BoundedPageRelevanceVerifier
from shafi.core.grounding.scope_policy import select_sidecar_doc_scope
from shafi.core.grounding.search_escalation import decide_search_escalation
from shafi.core.grounding.typed_panel_extractor import build_typed_comparison_panel
from shafi.core.retrieval.query_rewrite_shadow import build_shadow_rewrite_query
from shafi.models.schemas import (
    QueryScopePrediction,
    RetrievedPage,
    RetrievedSupportFact,
    ScopeMode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shafi.config.settings import PipelineSettings, VerifierSettings
    from shafi.core.embedding import EmbeddingClient
    from shafi.core.qdrant import QdrantStore
    from shafi.core.retriever import HybridRetriever
    from shafi.core.sparse_bm25 import BM25SparseEncoder
    from shafi.llm.provider import LLMProvider
    from shafi.ml.page_scorer_runtime import RuntimePageScorer
    from shafi.models import RankedChunk
    from shafi.telemetry.collector import TelemetryCollector

logger = logging.getLogger(__name__)
_EMPTY_GROUNDING_ANSWERS = frozenset(
    {
        "",
        "n/a",
        "no information",
        "none",
        "null",
        "there is no information on this question",
    }
)


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


def answer_requires_empty_grounding(answer: str) -> bool:
    """Return whether an answer should emit empty grounding.

    Args:
        answer: Final answer text.

    Returns:
        True when the answer is an explicit null/unsupported response.
    """
    normalized = re.sub(r"\s+", " ", str(answer or "")).strip(" .;:").casefold()
    if normalized in _EMPTY_GROUNDING_ANSWERS:
        return True
    # Catch LLM-generated variants like "There is no information on bail
    # conditions in case ARB 035/2025." — any answer that STARTS with the
    # canonical noinfo prefix and contains no additional factual content
    # should also force empty grounding.
    return normalized.startswith("there is no information on this") or normalized.startswith(
        "there is no information on "
    )


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
        verifier_settings: VerifierSettings | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._retriever = retriever
        self._store = store
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder
        self._settings = pipeline_settings
        self._trained_page_scorer: RuntimePageScorer | None = None
        self._relevance_verifier: BoundedPageRelevanceVerifier | None = None
        model_path = str(getattr(self._settings, "trained_page_scorer_model_path", "") or "").strip()
        if bool(getattr(self._settings, "enable_trained_page_scorer", False)) and model_path:
            from shafi.ml.page_scorer_runtime import RuntimePageScorer

            self._trained_page_scorer = RuntimePageScorer(model_path=model_path)
        if (
            llm_provider is not None
            and verifier_settings is not None
            and bool(getattr(self._settings, "grounding_relevance_verifier_enabled", True))
        ):
            from shafi.config import get_settings

            llm_settings = get_settings().llm
            self._relevance_verifier = BoundedPageRelevanceVerifier(
                llm=llm_provider,
                model=llm_settings.simple_model,
                max_tokens=int(verifier_settings.max_tokens),
                temperature=float(verifier_settings.temperature),
                min_confidence=float(getattr(self._settings, "grounding_relevance_verifier_min_confidence", 0.7)),
            )

    async def select_page_ids(
        self,
        *,
        query: str,
        answer: str,
        answer_type: str,
        context_chunks: Sequence[RankedChunk],
        current_used_ids: Sequence[str] = (),
        collector: TelemetryCollector | None = None,
    ) -> list[str] | None:
        """Select minimal evidentiary page IDs for grounding.

        Args:
            query: Raw user question.
            answer: Final answer text (already generated).
            answer_type: Normalized answer type.
            context_chunks: Ranked context chunks used for answering.
            current_used_ids: Answer-path used chunk IDs before sidecar override.
            collector: Optional telemetry collector for scorer diagnostics.

        Returns:
            List of page IDs, empty list for null/negative, or None to fall back to legacy.
        """
        scope = classify_query_scope(query, answer_type, settings=self._settings)

        if not self._should_activate_sidecar(
            query=query,
            scope=scope,
            answer_type=answer_type,
            context_chunks=context_chunks,
        ):
            return None

        # Force empty grounding on null answers for negative/unanswerable queries
        if scope.should_force_empty_grounding_on_null and answer_requires_empty_grounding(answer):
            return []

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
        if doc_ids:
            allowed_doc_ids = set(doc_ids)
            context_page_ids = self._page_ids_from_context_chunks(
                context_chunks=context_chunks,
                allowed_doc_ids=allowed_doc_ids,
            )
            legacy_used_page_ids = self._page_ids_from_chunk_ids(
                chunk_ids=current_used_ids,
                allowed_doc_ids=allowed_doc_ids,
            )
        else:
            context_page_ids = []
            legacy_used_page_ids = []

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
            context_page_ids=set(context_page_ids),
        )
        scored = self._maybe_apply_trained_page_scorer(
            query=query,
            answer_type=answer_type,
            scope=scope,
            answer_value=answer_value,
            doc_ids=doc_ids,
            page_candidates=page_candidates,
            context_page_ids=context_page_ids,
            legacy_used_page_ids=legacy_used_page_ids,
            heuristic_scores=scored,
            collector=collector,
        )

        selected = self._select_minimal_pages(
            query=query,
            answer_type=answer_type,
            scope=scope,
            scored=scored,
            page_candidates=page_candidates,
            collector=collector,
        )
        if not selected:
            return None

        # Recall floor: guarantee at least 1 page per doc_id in scope.
        # F-beta 2.5 math: missing 1 gold page costs G=0.537 (from 1.0),
        # adding 1 wrong page only costs G=0.935.  Always include over exclude.
        selected_doc_ids = {pid.rpartition("_")[0] for pid in selected if "_" in pid}
        if len(doc_ids) >= 2:
            missing_docs = set(doc_ids) - selected_doc_ids
            for page in page_candidates:
                page_doc = page.page_id.rpartition("_")[0] if "_" in page.page_id else ""
                if page_doc in missing_docs and page.page_id not in selected:
                    selected.append(page.page_id)
                    missing_docs.discard(page_doc)
                if not missing_docs:
                    break

        if not bool(getattr(self._settings, "grounding_escalation_enabled", True)):
            return selected

        selected_pages, audit = self._audit_selected_page_ids(
            query=query,
            answer_type=answer_type,
            scope=scope,
            selected_page_ids=selected,
            page_candidates=page_candidates,
        )
        escalation = decide_search_escalation(
            query=query,
            answer_type=answer_type,
            scope=scope,
            context_chunks=context_chunks,
            ordered_pages=page_candidates,
            selected_pages=selected_pages,
            audit=audit,
            rerank_margin_threshold=float(getattr(self._settings, "grounding_low_rerank_margin_threshold", 0.06)),
            page_margin_threshold=float(getattr(self._settings, "grounding_close_page_margin_threshold", 0.2)),
            authority_strength_threshold=float(getattr(self._settings, "grounding_authority_strength_threshold", 0.95)),
        )
        final_selected = list(selected)
        final_selected_pages = list(selected_pages)
        final_audit = audit
        active_page_candidates = list(page_candidates)
        shadow_rewrite_used = False
        shadow_rewrite_family = ""
        relevance_verifier_used = False
        relevance_verifier_confidence = 0.0
        relevance_verifier_fallback_reason = ""

        if escalation.should_escalate and escalation.allow_shadow_rewrite:
            rewrite = build_shadow_rewrite_query(
                query=query,
                family=escalation.allowed_family,
                hard_anchor_strings=scope.hard_anchor_strings,
                page_candidates=page_candidates,
                scope_mode=scope.scope_mode,
            )
            if rewrite is not None:
                rewritten_pages = await self._retrieve_page_candidates(
                    query=rewrite.rewritten_query,
                    scope=scope,
                    doc_ids=doc_ids,
                )
                merged_pages = self._merge_page_candidates(
                    query=query,
                    primary=page_candidates,
                    secondary=rewritten_pages,
                )
                rescored = self._score_candidates(
                    query=query,
                    answer_type=answer_type,
                    scope=scope,
                    answer_value=answer_value,
                    support_candidates=support_candidates,
                    page_candidates=merged_pages,
                    context_page_ids=set(context_page_ids),
                )
                rescored = self._maybe_apply_trained_page_scorer(
                    query=query,
                    answer_type=answer_type,
                    scope=scope,
                    answer_value=answer_value,
                    doc_ids=doc_ids,
                    page_candidates=merged_pages,
                    context_page_ids=context_page_ids,
                    legacy_used_page_ids=legacy_used_page_ids,
                    heuristic_scores=rescored,
                    collector=None,
                )
                rewritten_selected = self._select_minimal_pages(
                    query=query,
                    answer_type=answer_type,
                    scope=scope,
                    scored=rescored,
                    page_candidates=merged_pages,
                    collector=collector,
                )
                if rewritten_selected:
                    shadow_rewrite_used = True
                    shadow_rewrite_family = rewrite.family
                    final_selected = list(rewritten_selected)
                    active_page_candidates = merged_pages
                    final_selected_pages, final_audit = self._audit_selected_page_ids(
                        query=query,
                        answer_type=answer_type,
                        scope=scope,
                        selected_page_ids=final_selected,
                        page_candidates=active_page_candidates,
                    )
                    escalation = decide_search_escalation(
                        query=query,
                        answer_type=answer_type,
                        scope=scope,
                        context_chunks=context_chunks,
                        ordered_pages=active_page_candidates,
                        selected_pages=final_selected_pages,
                        audit=final_audit,
                        rerank_margin_threshold=float(
                            getattr(self._settings, "grounding_low_rerank_margin_threshold", 0.06)
                        ),
                        page_margin_threshold=float(
                            getattr(self._settings, "grounding_close_page_margin_threshold", 0.2)
                        ),
                        authority_strength_threshold=float(
                            getattr(self._settings, "grounding_authority_strength_threshold", 0.95)
                        ),
                    )

        if escalation.should_escalate and escalation.allow_relevance_verifier and self._relevance_verifier is not None:
            verifier_pages = self._build_relevance_verifier_candidates(
                ordered_pages=active_page_candidates,
                selected_pages=final_selected_pages,
                max_candidates=int(getattr(self._settings, "grounding_relevance_verifier_max_candidates", 3)),
            )
            if verifier_pages:
                if collector is not None:
                    with collector.timed("verify"):
                        verification = await self._relevance_verifier.verify(
                            query=query,
                            answer_type=answer_type,
                            required_slots=final_audit.required_slots,
                            candidate_pages=verifier_pages,
                            max_selected_pages=self._max_selected_pages_for_family(
                                family=escalation.allowed_family,
                                scope_mode=scope.scope_mode,
                            ),
                        )
                else:
                    verification = await self._relevance_verifier.verify(
                        query=query,
                        answer_type=answer_type,
                        required_slots=final_audit.required_slots,
                        candidate_pages=verifier_pages,
                        max_selected_pages=self._max_selected_pages_for_family(
                            family=escalation.allowed_family,
                            scope_mode=scope.scope_mode,
                        ),
                    )
                if verification.used and verification.selected_page_ids:
                    relevance_verifier_used = True
                    relevance_verifier_confidence = verification.confidence
                    final_selected = list(verification.selected_page_ids)
                    final_selected_pages, final_audit = self._audit_selected_page_ids(
                        query=query,
                        answer_type=answer_type,
                        scope=scope,
                        selected_page_ids=final_selected,
                        page_candidates=active_page_candidates,
                    )
                else:
                    relevance_verifier_fallback_reason = verification.fallback_reason

        if collector is not None:
            collector.set_grounding_escalation_diagnostics(
                triggered=escalation.should_escalate or shadow_rewrite_used or relevance_verifier_used,
                reasons=list(escalation.reasons),
                family=escalation.allowed_family,
                shadow_rewrite_used=shadow_rewrite_used,
                shadow_rewrite_family=shadow_rewrite_family,
                relevance_verifier_used=relevance_verifier_used,
                relevance_verifier_confidence=relevance_verifier_confidence,
                relevance_verifier_fallback_reason=relevance_verifier_fallback_reason,
            )

        if final_selected:
            return final_selected

        return None  # fall back to legacy

    @staticmethod
    def _page_ids_from_context_chunks(
        *,
        context_chunks: Sequence[RankedChunk],
        allowed_doc_ids: set[str],
    ) -> list[str]:
        """Build ordered unique page IDs from answer-path context chunks.

        Args:
            context_chunks: Ranked answer-path chunks.
            allowed_doc_ids: Document IDs allowed by sidecar scope.

        Returns:
            Ordered page IDs limited to the allowed doc scope.
        """
        ordered: list[str] = []
        seen: set[str] = set()
        for chunk in context_chunks:
            page_id = GroundingEvidenceSelector._chunk_id_to_page_id(chunk.chunk_id)
            if not page_id or page_id in seen:
                continue
            if page_id.rpartition("_")[0] not in allowed_doc_ids:
                continue
            seen.add(page_id)
            ordered.append(page_id)
        return ordered

    @staticmethod
    def _page_ids_from_chunk_ids(
        *,
        chunk_ids: Sequence[str],
        allowed_doc_ids: set[str],
    ) -> list[str]:
        """Build ordered unique page IDs from chunk IDs.

        Args:
            chunk_ids: Chunk IDs to convert.
            allowed_doc_ids: Document IDs allowed by sidecar scope.

        Returns:
            Ordered page IDs limited to the allowed doc scope.
        """
        ordered: list[str] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            page_id = GroundingEvidenceSelector._chunk_id_to_page_id(chunk_id)
            if not page_id or page_id in seen:
                continue
            if page_id.rpartition("_")[0] not in allowed_doc_ids:
                continue
            seen.add(page_id)
            ordered.append(page_id)
        return ordered

    @staticmethod
    def _chunk_id_to_page_id(chunk_id: str) -> str:
        """Convert a starter-kit chunk ID into a page ID.

        Args:
            chunk_id: Starter-kit chunk ID.

        Returns:
            Competition page ID, or an empty string when conversion fails.
        """
        parts = str(chunk_id).split(":")
        if len(parts) < 2:
            return ""
        doc_id = parts[0].strip()
        page_idx = parts[1].strip()
        if not doc_id or not page_idx.isdigit():
            return ""
        return f"{doc_id}_{int(page_idx) + 1}"

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
        return select_sidecar_doc_scope(
            query=query,
            scope=scope,
            context_chunks=context_chunks,
        )

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

        must: list[models.Condition] = [models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))]
        if scope.target_page_roles:
            must.append(models.FieldCondition(key="page_role", match=models.MatchAny(any=scope.target_page_roles)))
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
        page_nums = extract_explicit_page_numbers(query) if scope.scope_mode is ScopeMode.EXPLICIT_PAGE else None
        pages = await self._retriever.retrieve_pages(
            query,
            top_k=self._settings.grounding_page_top_k,
            doc_ids=list(doc_ids),
            page_nums=page_nums,
            page_roles=list(scope.target_page_roles) if scope.target_page_roles else None,
            article_refs=list(scope.hard_anchor_strings) if scope.hard_anchor_strings else None,
        )
        return rerank_pages_with_shadow_signal(query, pages)

    @staticmethod
    def _should_activate_sidecar(
        *,
        query: str,
        scope: QueryScopePrediction,
        answer_type: str,
        context_chunks: Sequence[RankedChunk],
    ) -> bool:
        """Decide whether the grounding sidecar is safe to apply.

        Args:
            query: Raw user question.
            scope: Predicted grounding scope for the query.
            answer_type: Normalized answer type.
            context_chunks: Ranked context chunks used by the answer path.

        Returns:
            True when the sidecar should attempt page selection, otherwise False.
        """
        doc_ids = {chunk.doc_id for chunk in context_chunks if chunk.doc_id}

        if scope.scope_mode in {ScopeMode.COMPARE_PAIR, ScopeMode.FULL_CASE_FILES}:
            return True

        if scope.scope_mode is ScopeMode.NEGATIVE_UNANSWERABLE:
            return True

        if scope.scope_mode is ScopeMode.EXPLICIT_PAGE:
            return len(doc_ids) == 1

        # For SINGLE_FIELD: always activate sidecar — page_budget=2 gives recall floor.
        # F-beta 2.5 math: missing 1 gold page costs G=0.537; adding 1 wrong page costs G=0.065.
        # 17/20 correct-G=0 cases had only 1 page (wrong page chosen by legacy path);
        # sidecar with page_budget=2 gives top-2 candidates, typically including gold page.
        del answer_type
        if scope.scope_mode is ScopeMode.SINGLE_FIELD_SINGLE_DOC:
            return True  # Always activate for page_budget=2 recall floor
        if scope.scope_mode is ScopeMode.BROAD_FREE_TEXT:
            return len(doc_ids) <= 5
        return False

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

        candidate_page_ids = {page.page_id for page in page_candidates if page.page_id}

        # Page retrieval score (primary signal)
        for page in page_candidates:
            scores[page.page_id] += float(page.score)
            scores[page.page_id] += authority_signal_score(
                query,
                page,
                peer_pages=list(page_candidates),
            )

        # Strong anchor: pages the answer path already cited deserve a large bonus.
        # The legacy path chose these pages based on full context; the sidecar should
        # only override when it has very strong evidence pointing elsewhere.
        cited = context_page_ids or set()
        for pid in cited:
            if scope.scope_mode is ScopeMode.EXPLICIT_PAGE and candidate_page_ids and pid not in candidate_page_ids:
                continue
            scores[pid] += 3.0

        # Support-fact score (secondary signal — reduced from 2.0 to 0.8 to avoid
        # schedule/interpretation pages dominating over the actual article page)
        for fact in support_candidates:
            scores[fact.page_id] += 0.8 * float(fact.score)
            if answer_value and answer_value.casefold() == (fact.normalized_value or "").casefold():
                scores[fact.page_id] += 1.0
            if fact.page_role and fact.page_role in scope.target_page_roles:
                scores[fact.page_id] += 0.5
            if fact.fact_type == "date_of_issue" and "date of issue" in query.lower():
                scores[fact.page_id] += 1.0
            if answer_type == "number" and fact.fact_type in {"claim_amount", "costs_awarded", "penalty"}:
                scores[fact.page_id] += 0.5

        return dict(sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])))

    def _maybe_apply_trained_page_scorer(
        self,
        *,
        query: str,
        answer_type: str,
        scope: QueryScopePrediction,
        answer_value: str,
        doc_ids: Sequence[str],
        page_candidates: Sequence[RetrievedPage],
        context_page_ids: Sequence[str],
        legacy_used_page_ids: Sequence[str],
        heuristic_scores: dict[str, float],
        collector: TelemetryCollector | None,
    ) -> dict[str, float]:
        """Reorder heuristic scores with the trained runtime scorer when safe.

        Args:
            query: Raw user question.
            answer_type: Normalized answer type.
            scope: Query scope prediction.
            answer_value: Normalized answer value.
            doc_ids: Scoped document IDs.
            page_candidates: Retrieved page candidates available to the sidecar.
            context_page_ids: Ordered context page IDs from the answer path.
            legacy_used_page_ids: Ordered legacy used page IDs.
            heuristic_scores: Current heuristic page scores.
            collector: Optional telemetry collector.

        Returns:
            Possibly reordered score mapping with the same page IDs.
        """
        scorer = self._trained_page_scorer
        if scorer is None:
            return heuristic_scores

        fallback_reason = ""
        rankable_candidate_ids: list[str] = []
        if scope.scope_mode is ScopeMode.EXPLICIT_PAGE:
            fallback_reason = "explicit_page_terminal"
        elif len(page_candidates) < 1:
            fallback_reason = "insufficient_page_candidates"
        else:
            candidate_page_ids = {page.page_id for page in page_candidates if page.page_id}
            rankable_candidate_ids = ordered_rankable_candidate_page_ids(
                heuristic_order=list(heuristic_scores),
                candidate_page_ids=candidate_page_ids,
            )
            if len(rankable_candidate_ids) < 1:
                fallback_reason = "insufficient_page_candidates"
        if fallback_reason:
            if collector is not None:
                collector.set_trained_page_scorer_diagnostics(
                    used=False,
                    model_path=scorer.model_path,
                    fallback_reason=fallback_reason,
                )
            return heuristic_scores

        from shafi.ml.page_scorer_runtime import RuntimePageScoringRequest

        rankable_candidate_id_set = set(rankable_candidate_ids)
        result = scorer.rank_pages(
            RuntimePageScoringRequest(
                query=query,
                normalized_answer=answer_value,
                answer_type=answer_type,
                scope=scope,
                doc_ids=doc_ids,
                page_candidates=[page for page in page_candidates if page.page_id in rankable_candidate_id_set],
                context_page_ids=context_page_ids,
                legacy_used_page_ids=legacy_used_page_ids,
                heuristic_scores=heuristic_scores,
            )
        )
        if not result.used:
            if collector is not None:
                collector.set_trained_page_scorer_diagnostics(
                    used=False,
                    model_path=result.model_path,
                    page_ids=result.ranked_page_ids,
                    fallback_reason=result.fallback_reason,
                )
            return heuristic_scores

        merged_page_ids = merge_ranked_candidate_subset(
            heuristic_order=list(heuristic_scores),
            ranked_candidate_page_ids=result.ranked_page_ids,
            candidate_page_ids=rankable_candidate_ids,
        )
        if merged_page_ids is None:
            if collector is not None:
                collector.set_trained_page_scorer_diagnostics(
                    used=False,
                    model_path=result.model_path,
                    page_ids=result.ranked_page_ids,
                    fallback_reason="ranked_subset_mismatch",
                )
            return heuristic_scores
        if collector is not None:
            collector.set_trained_page_scorer_diagnostics(
                used=True,
                model_path=result.model_path,
                page_ids=merged_page_ids,
                fallback_reason="",
            )
        return {page_id: heuristic_scores[page_id] for page_id in merged_page_ids}

    def _select_minimal_pages(
        self,
        *,
        query: str,
        answer_type: str = "free_text",
        scope: QueryScopePrediction,
        scored: dict[str, float],
        page_candidates: Sequence[RetrievedPage],
        collector: TelemetryCollector | None = None,
    ) -> list[str]:
        """Select minimal page set from scored candidates respecting scope budget.

        Args:
            query: Raw grounding query.
            answer_type: Normalized answer type.
            scope: Query scope prediction with page budget.
            scored: Page IDs scored by relevance (sorted descending).
            page_candidates: Retrieved page candidates available to the sidecar.
            collector: Optional telemetry collector for selector diagnostics.

        Returns:
            List of selected page IDs.
        """
        if not scored:
            return []

        ordered = list(scored.keys())
        pages_by_id = {page.page_id: page for page in page_candidates if page.page_id}
        ordered_pages = [pages_by_id[page_id] for page_id in ordered if page_id in pages_by_id]
        safe_page_ids = self._select_safe_sidecar_pages(
            query=query,
            scope=scope,
            answer_type=answer_type,
            ordered=ordered,
            ordered_pages=ordered_pages,
            page_candidates=page_candidates,
        )
        if not ordered_pages:
            return safe_page_ids
        if scope.scope_mode not in {
            ScopeMode.SINGLE_FIELD_SINGLE_DOC,
            ScopeMode.COMPARE_PAIR,
            ScopeMode.BROAD_FREE_TEXT,
        }:
            return safe_page_ids

        candidates = self._build_portfolio_candidates(
            query=query,
            scope=scope,
            safe_page_ids=safe_page_ids,
            ordered=ordered,
            ordered_pages=ordered_pages,
            page_candidates=page_candidates,
        )
        portfolio = EvidencePortfolio(
            query=query,
            scope_mode=scope.scope_mode,
            candidates=tuple(candidates),
        )
        selection = select_best_page_set(
            query=query,
            answer_type=answer_type,
            scope=scope,
            page_lookup=pages_by_id,
            portfolio=portfolio,
        )
        selected_candidate = selection.selected_candidate
        selected_pages = [pages_by_id[page_id] for page_id in selected_candidate.page_ids if page_id in pages_by_id]
        pruned_page_ids = prune_redundant_pages(
            query=query,
            answer_type=answer_type,
            scope_mode=scope.scope_mode,
            ordered_pages=selected_pages,
            page_budget=scope.page_budget,
        )
        pruned_pages = [pages_by_id[page_id] for page_id in pruned_page_ids if page_id in pages_by_id]
        audit = audit_candidate_pages(
            query=query,
            answer_type=answer_type,
            scope_mode=scope.scope_mode,
            pages=pruned_pages,
        )
        final_page_ids = list(pruned_page_ids)
        fallback_used = False
        if selected_candidate.name != "safe_sidecar" and not audit.success:
            final_page_ids = list(safe_page_ids)
            fallback_used = True
        if collector is not None:
            collector.set_grounding_portfolio_diagnostics(
                candidate_names=[candidate.name for candidate in selection.ranked_candidates],
                selected_candidate="safe_sidecar" if fallback_used else selected_candidate.name,
                decision_reasons=list(selection.decision_reasons),
                failed_slots=list(audit.failed_slots),
                fallback_used=fallback_used,
            )
        return final_page_ids

    @staticmethod
    def _audit_selected_page_ids(
        *,
        query: str,
        answer_type: str,
        scope: QueryScopePrediction,
        selected_page_ids: Sequence[str],
        page_candidates: Sequence[RetrievedPage],
    ) -> tuple[list[RetrievedPage], ConditionAuditResult]:
        """Build selected pages plus a typed audit for them.

        Args:
            query: Raw user query.
            answer_type: Normalized answer type.
            scope: Current query scope.
            selected_page_ids: Selected page IDs.
            page_candidates: Available page candidates.

        Returns:
            tuple[list[RetrievedPage], ConditionAuditResult]: Selected pages and typed audit.
        """

        pages_by_id = {page.page_id: page for page in page_candidates if page.page_id}
        selected_pages = [pages_by_id[page_id] for page_id in selected_page_ids if page_id in pages_by_id]
        audit = audit_candidate_pages(
            query=query,
            answer_type=answer_type,
            scope_mode=scope.scope_mode,
            pages=selected_pages,
        )
        return selected_pages, audit

    @staticmethod
    def _merge_page_candidates(
        *,
        query: str,
        primary: Sequence[RetrievedPage],
        secondary: Sequence[RetrievedPage],
    ) -> list[RetrievedPage]:
        """Merge primary and shadow-rewrite page candidates without widening blindly.

        Args:
            query: Raw user query.
            primary: Current primary page candidates.
            secondary: Shadow-rewrite page candidates.

        Returns:
            list[RetrievedPage]: Deduplicated merged candidates reranked for the
            original query.
        """

        by_page_id: dict[str, RetrievedPage] = {}
        for page in [*primary, *secondary]:
            if not page.page_id:
                continue
            current = by_page_id.get(page.page_id)
            if current is None or float(page.score) > float(current.score):
                by_page_id[page.page_id] = page
        merged = list(by_page_id.values())
        return rerank_pages_with_shadow_signal(query, merged)

    @staticmethod
    def _build_relevance_verifier_candidates(
        *,
        ordered_pages: Sequence[RetrievedPage],
        selected_pages: Sequence[RetrievedPage],
        max_candidates: int,
    ) -> list[RetrievedPage]:
        """Build the bounded candidate set for the runtime relevance verifier.

        Args:
            ordered_pages: Candidate pages in current score order.
            selected_pages: Currently selected pages.
            max_candidates: Maximum verifier candidate count.

        Returns:
            list[RetrievedPage]: Ordered bounded verifier candidates.
        """

        ordered: list[RetrievedPage] = []
        seen: set[str] = set()
        for page in [*selected_pages, *ordered_pages]:
            if not page.page_id or page.page_id in seen:
                continue
            seen.add(page.page_id)
            ordered.append(page)
            if len(ordered) >= max(1, max_candidates):
                break
        return ordered

    @staticmethod
    def _max_selected_pages_for_family(*, family: str, scope_mode: ScopeMode) -> int:
        """Return the bounded maximum selection size for one supported family.

        Args:
            family: Escalation family label.
            scope_mode: Current query scope mode.

        Returns:
            int: Maximum number of allowed selected pages.
        """

        # F-beta 2.5 recall optimization: recall weighted 6.25x over precision.
        # Missing 1 gold page costs ~46% G; adding 1 wrong page costs ~6.5%.
        # Organizers confirmed: "indicate the most appropriate and COMPLETE sources."
        # Raise caps to match page_budget=8 for maximum recall.
        if scope_mode is ScopeMode.BROAD_FREE_TEXT:
            return 8
        if scope_mode in {ScopeMode.COMPARE_PAIR, ScopeMode.SINGLE_FIELD_SINGLE_DOC}:
            return 6
        if family in {"exact_provision", "compare_authoritative_pair"}:
            return 4
        return 3

    def _build_portfolio_candidates(
        self,
        *,
        query: str,
        scope: QueryScopePrediction,
        safe_page_ids: list[str],
        ordered: list[str],
        ordered_pages: Sequence[RetrievedPage],
        page_candidates: Sequence[RetrievedPage],
    ) -> list[PortfolioCandidate]:
        """Assemble deterministic grounding portfolio candidates.

        Args:
            query: Raw grounding query.
            scope: Query scope prediction.
            safe_page_ids: Current safe sidecar selection.
            ordered: Ordered page IDs by heuristic score.
            ordered_pages: Ordered page objects by heuristic score.
            page_candidates: Retrieved page candidates.

        Returns:
            list[PortfolioCandidate]: Unique portfolio candidates in build order.
        """
        del page_candidates
        candidates: list[PortfolioCandidate] = []
        seen: set[tuple[str, ...]] = set()

        def add_candidate(candidate: PortfolioCandidate | None) -> None:
            if candidate is None:
                return
            if not candidate.page_ids:
                return
            if candidate.page_ids in seen:
                return
            seen.add(candidate.page_ids)
            candidates.append(candidate)

        add_candidate(
            PortfolioCandidate(
                name="safe_sidecar",
                page_ids=tuple(safe_page_ids),
                activation_family="safe_sidecar",
                reasons=("current_sidecar_baseline",),
            )
        )

        budget = max(1, scope.page_budget)
        add_candidate(
            PortfolioCandidate(
                name="ordered_budget",
                page_ids=tuple(ordered[:budget]),
                activation_family="ordered_budget",
                reasons=("heuristic_budget_slice",),
            )
        )

        semantic_decision = select_semantic_page_set(
            query=query,
            scope_mode=scope.scope_mode,
            ordered_page_ids=ordered,
            page_candidates=list(ordered_pages),
            page_budget=scope.page_budget,
        )
        if semantic_decision is not None:
            add_candidate(
                PortfolioCandidate(
                    name="semantic_lane",
                    page_ids=semantic_decision.page_ids,
                    activation_family=semantic_decision.activation_family,
                    reasons=("semantic_page_set",),
                )
            )

        if scope.scope_mode is ScopeMode.SINGLE_FIELD_SINGLE_DOC:
            authoritative_page = select_authoritative_single_page(query, ordered_pages)
            if authoritative_page is not None:
                add_candidate(
                    PortfolioCandidate(
                        name="authority_single",
                        page_ids=(authoritative_page.page_id,),
                        activation_family="authority_single",
                        reasons=("single_doc_authority",),
                    )
                )
            add_candidate(build_law_bundle_candidate(query=query, ordered_pages=ordered_pages))

        if scope.scope_mode is ScopeMode.COMPARE_PAIR:
            compare_panel = build_typed_comparison_panel(query=query, ordered_pages=list(ordered_pages))
            if compare_panel:
                add_candidate(
                    PortfolioCandidate(
                        name="typed_compare_panel",
                        page_ids=tuple(panel.page_id for panel in compare_panel if panel.page_id),
                        activation_family="typed_compare_panel",
                        reasons=tuple(f"{panel.doc_id}:{panel.case_ref or 'context_doc'}" for panel in compare_panel),
                    )
                )

        return candidates

    @staticmethod
    def _select_safe_sidecar_pages(
        *,
        query: str,
        scope: QueryScopePrediction,
        answer_type: str = "free_text",
        ordered: list[str],
        ordered_pages: Sequence[RetrievedPage],
        page_candidates: Sequence[RetrievedPage],
    ) -> list[str]:
        """Return the current safe sidecar page selection without portfolio logic.

        Args:
            query: Raw grounding query.
            scope: Query scope prediction.
            answer_type: Normalized answer type.
            ordered: Ordered page IDs by current heuristic score.
            ordered_pages: Ordered page objects by current heuristic score.
            page_candidates: Retrieved page candidates available to the sidecar.

        Returns:
            list[str]: Safe sidecar page IDs.
        """
        semantic_decision = select_semantic_page_set(
            query=query,
            scope_mode=scope.scope_mode,
            ordered_page_ids=ordered,
            page_candidates=list(page_candidates),
            page_budget=scope.page_budget,
        )
        if semantic_decision is not None:
            return list(semantic_decision.page_ids)

        if scope.scope_mode is ScopeMode.EXPLICIT_PAGE:
            return ordered[:1]

        if scope.scope_mode is ScopeMode.SINGLE_FIELD_SINGLE_DOC:
            if scope.page_budget <= 0:
                return []
            budget = max(1, scope.page_budget)
            authoritative_page = select_authoritative_single_page(query, ordered_pages)
            if authoritative_page is not None:
                # Start with authoritative page, fill remaining budget from ordered
                pages = [authoritative_page.page_id]
                for pid in ordered:
                    if pid != authoritative_page.page_id and len(pages) < budget:
                        pages.append(pid)
                return pages
            return ordered[:budget]

        if scope.scope_mode is ScopeMode.FULL_CASE_FILES:
            per_doc: dict[str, str] = {}
            for page_id in ordered:
                doc_id, _, _ = page_id.rpartition("_")
                if doc_id and doc_id not in per_doc:
                    per_doc[doc_id] = page_id
            return list(per_doc.values())

        if scope.scope_mode is ScopeMode.COMPARE_PAIR:
            per_doc_cmp: dict[str, str] = {}
            ordered_set = set(ordered)
            for page_id in ordered:
                doc_id, _, _ = page_id.rpartition("_")
                if doc_id and doc_id not in per_doc_cmp:
                    per_doc_cmp[doc_id] = page_id
            # For name/boolean type: gold citation is typically case header/cover or law
            # cover page (page suffix _1). NOGA-24a: 5 case-comparison name Qs cite
            # doc_18 instead of doc_1. For boolean enactment-date comparison (bb67fc19,
            # d5bc7441): law enactment year/date lives on page 1 (law cover), not content
            # pages. Extending page_1 preference to boolean + COMPARE_PAIR.
            if answer_type in {"name", "boolean"}:
                for doc_id in list(per_doc_cmp.keys()):
                    page1_id = f"{doc_id}_1"
                    if page1_id in ordered_set:
                        per_doc_cmp[doc_id] = page1_id
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
