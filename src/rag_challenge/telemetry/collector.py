from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Literal

from rag_challenge.models import TelemetryPayload

logger = logging.getLogger(__name__)

StageName = Literal["embed", "qdrant", "rerank", "llm", "classify", "verify", "grounding"]


class TelemetryCollector:
    """Per-request in-process telemetry accumulator."""

    def __init__(
        self,
        request_id: str,
        *,
        question_id: str = "",
        answer_type: str = "free_text",
    ) -> None:
        self._request_id = request_id
        self._question_id = question_id
        self._answer_type = answer_type
        self._start_time = time.perf_counter()
        self._ttft_time: float | None = None
        self._timings_ms: dict[str, float] = {}

        self._retrieved_ids: list[str] = []
        self._context_ids: list[str] = []
        self._cited_ids: list[str] = []
        self._used_ids: list[str] = []
        self._chunk_snippets: dict[str, str] = {}
        self._chunk_page_hints: dict[str, str] = {}
        self._doc_refs: list[str] = []

        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

        self._model_embed = ""
        self._model_rerank = ""
        self._model_llm = ""
        self._llm_provider = ""
        self._llm_finish_reason = ""
        self._generation_mode = ""
        self._context_chunk_count = 0
        self._context_budget_tokens = 0
        self._malformed_tail_detected = False
        self._retried = False
        self._model_upgraded: bool = False
        self._used_page_ids_override: list[str] | None = None
        self._trained_page_scorer_used = False
        self._trained_page_scorer_model_path = ""
        self._trained_page_scorer_page_ids: list[str] = []
        self._trained_page_scorer_fallback_reason = ""
        self._grounding_portfolio_candidates: list[str] = []
        self._grounding_portfolio_selected = ""
        self._grounding_portfolio_decision_reasons: list[str] = []
        self._grounding_audit_failed_slots: list[str] = []
        self._grounding_audit_fallback_used = False
        self._grounding_escalation_triggered = False
        self._grounding_escalation_reasons: list[str] = []
        self._grounding_escalation_family = ""
        self._grounding_shadow_rewrite_used = False
        self._grounding_shadow_rewrite_family = ""
        self._grounding_relevance_verifier_used = False
        self._grounding_relevance_verifier_confidence = 0.0
        self._grounding_relevance_verifier_fallback_reason = ""
        self._claim_graph_enabled = False
        self._claim_graph_count = 0
        self._claim_graph_unsupported_count = 0
        self._claim_graph_support_coverage = 0.0
        self._proof_compiler_enabled = False
        self._proof_compiler_used = False
        self._proof_compiler_support_coverage = 0.0
        self._proof_compiler_verified_claim_count = 0
        self._proof_compiler_dropped_claim_count = 0
        self._proof_compiler_fallback_reason = ""
        self._proof_compiler_fully_supported = False
        self._selective_icr_shadow_used = False
        self._selective_icr_shadow_model = ""
        self._selective_icr_shadow_latency_ms = 0
        self._selective_icr_shadow_candidate_count = 0
        self._selective_icr_shadow_chunk_ids: list[str] = []
        self._selective_icr_shadow_page_ids: list[str] = []
        self._selective_icr_shadow_provider_exit = False

    @contextmanager
    def timed(self, stage: StageName):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._timings_ms[stage] = self._timings_ms.get(stage, 0.0) + elapsed_ms

    def mark_first_token(self) -> None:
        if self._ttft_time is None:
            self._ttft_time = time.perf_counter()

    def set_retrieved_ids(self, ids: list[str] | tuple[str, ...]) -> None:
        self._retrieved_ids = list(ids)

    def set_context_ids(self, ids: list[str] | tuple[str, ...]) -> None:
        self._context_ids = list(ids)

    def set_cited_ids(self, ids: list[str] | tuple[str, ...]) -> None:
        self._cited_ids = list(ids)

    def set_used_ids(self, ids: list[str] | tuple[str, ...]) -> None:
        self._used_ids = list(ids)

    def set_used_page_ids_override(self, page_ids: list[str]) -> None:
        """Directly set used_page_ids, bypassing chunk-to-page inference."""
        self._used_page_ids_override = list(page_ids)

    def set_trained_page_scorer_diagnostics(
        self,
        *,
        used: bool,
        model_path: str = "",
        page_ids: list[str] | None = None,
        fallback_reason: str = "",
    ) -> None:
        """Record runtime trained-page-scorer diagnostics.

        Args:
            used: Whether the trained scorer actively reordered pages.
            model_path: Model artifact path used for scoring, if any.
            page_ids: Ranked page IDs returned by the scorer.
            fallback_reason: Fail-closed reason when the scorer was skipped.
        """
        self._trained_page_scorer_used = bool(used)
        self._trained_page_scorer_model_path = str(model_path).strip()
        self._trained_page_scorer_page_ids = list(page_ids or [])
        self._trained_page_scorer_fallback_reason = str(fallback_reason).strip()

    def get_trained_page_scorer_page_ids(self) -> list[str]:
        """Return ranked page IDs from trained page scorer (best first).

        Returns:
            Page IDs ordered by scorer confidence, empty if scorer not used.
        """
        return list(self._trained_page_scorer_page_ids)

    def set_grounding_portfolio_diagnostics(
        self,
        *,
        candidate_names: list[str],
        selected_candidate: str,
        decision_reasons: list[str],
        failed_slots: list[str] | None = None,
        fallback_used: bool = False,
    ) -> None:
        """Record grounding portfolio selection diagnostics.

        Args:
            candidate_names: Candidate strategy names considered.
            selected_candidate: Final chosen candidate strategy.
            decision_reasons: Short ordered reasons for the choice.
            failed_slots: Optional typed-audit slots that failed.
            fallback_used: Whether a challenger failed and baseline fallback won.
        """
        self._grounding_portfolio_candidates = list(candidate_names)
        self._grounding_portfolio_selected = str(selected_candidate).strip()
        self._grounding_portfolio_decision_reasons = [str(reason).strip() for reason in decision_reasons if str(reason).strip()]
        self._grounding_audit_failed_slots = [str(slot).strip() for slot in (failed_slots or []) if str(slot).strip()]
        self._grounding_audit_fallback_used = bool(fallback_used)

    def set_grounding_escalation_diagnostics(
        self,
        *,
        triggered: bool,
        reasons: list[str] | None = None,
        family: str = "",
        shadow_rewrite_used: bool = False,
        shadow_rewrite_family: str = "",
        relevance_verifier_used: bool = False,
        relevance_verifier_confidence: float = 0.0,
        relevance_verifier_fallback_reason: str = "",
    ) -> None:
        """Record bounded escalation diagnostics for the grounding sidecar.

        Args:
            triggered: Whether bounded escalation was triggered.
            reasons: Ordered escalation reasons.
            family: Supported family label for the escalation.
            shadow_rewrite_used: Whether shadow rewrite retrieval was used.
            shadow_rewrite_family: Family label for the shadow rewrite.
            relevance_verifier_used: Whether the bounded verifier was used.
            relevance_verifier_confidence: Verifier confidence in ``[0.0, 1.0]``.
            relevance_verifier_fallback_reason: Fail-closed verifier reason.
        """

        self._grounding_escalation_triggered = bool(triggered)
        self._grounding_escalation_reasons = [
            str(reason).strip() for reason in (reasons or []) if str(reason).strip()
        ]
        self._grounding_escalation_family = str(family).strip()
        self._grounding_shadow_rewrite_used = bool(shadow_rewrite_used)
        self._grounding_shadow_rewrite_family = str(shadow_rewrite_family).strip()
        self._grounding_relevance_verifier_used = bool(relevance_verifier_used)
        self._grounding_relevance_verifier_confidence = max(
            0.0,
            float(relevance_verifier_confidence or 0.0),
        )
        self._grounding_relevance_verifier_fallback_reason = str(
            relevance_verifier_fallback_reason
        ).strip()

    def set_claim_graph_diagnostics(
        self,
        *,
        enabled: bool,
        claim_count: int,
        unsupported_claim_count: int,
        support_coverage: float,
    ) -> None:
        """Record claim-graph construction diagnostics.

        Args:
            enabled: Whether claim-graph construction ran.
            claim_count: Number of extracted claims.
            unsupported_claim_count: Number of unsupported claims.
            support_coverage: Fraction of claims with evidence.
        """

        self._claim_graph_enabled = bool(enabled)
        self._claim_graph_count = max(0, int(claim_count))
        self._claim_graph_unsupported_count = max(0, int(unsupported_claim_count))
        self._claim_graph_support_coverage = max(0.0, min(1.0, float(support_coverage or 0.0)))

    def set_proof_compiler_diagnostics(
        self,
        *,
        enabled: bool,
        used: bool,
        support_coverage: float,
        verified_claim_count: int,
        dropped_claim_count: int,
        fallback_reason: str = "",
        is_fully_supported: bool = False,
    ) -> None:
        """Record proof-compiler diagnostics.

        Args:
            enabled: Whether the proof compiler is enabled.
            used: Whether proof compilation replaced the answer.
            support_coverage: Claim-graph support coverage.
            verified_claim_count: Number of claims kept.
            dropped_claim_count: Number of claims dropped.
            fallback_reason: Fail-closed reason when proof compilation was skipped.
            is_fully_supported: Whether all claims were supported.
        """

        self._proof_compiler_enabled = bool(enabled)
        self._proof_compiler_used = bool(used)
        self._proof_compiler_support_coverage = max(0.0, min(1.0, float(support_coverage or 0.0)))
        self._proof_compiler_verified_claim_count = max(0, int(verified_claim_count))
        self._proof_compiler_dropped_claim_count = max(0, int(dropped_claim_count))
        self._proof_compiler_fallback_reason = str(fallback_reason).strip()
        self._proof_compiler_fully_supported = bool(is_fully_supported)

    def set_selective_icr_shadow_diagnostics(
        self,
        *,
        used: bool,
        model: str = "",
        latency_ms: int = 0,
        candidate_count: int = 0,
        chunk_ids: list[str] | None = None,
        page_ids: list[str] | None = None,
        provider_exit: bool = False,
    ) -> None:
        """Record diagnostics for the shadow selective ICR rerank lane.

        Args:
            used: Whether the selective ICR shadow lane ran.
            model: Local model or heuristic label used.
            latency_ms: Shadow rerank latency in milliseconds.
            candidate_count: Number of candidates scored.
            chunk_ids: Ranked chunk identifiers returned by the shadow lane.
            page_ids: Ranked page identifiers returned by the shadow lane.
            provider_exit: Whether the shadow lane is configured for provider exit.
        """

        self._selective_icr_shadow_used = bool(used)
        self._selective_icr_shadow_model = str(model).strip()
        self._selective_icr_shadow_latency_ms = max(0, int(latency_ms))
        self._selective_icr_shadow_candidate_count = max(0, int(candidate_count))
        self._selective_icr_shadow_chunk_ids = [str(chunk_id).strip() for chunk_id in (chunk_ids or []) if str(chunk_id).strip()]
        self._selective_icr_shadow_page_ids = [str(page_id).strip() for page_id in (page_ids or []) if str(page_id).strip()]
        self._selective_icr_shadow_provider_exit = bool(provider_exit)

    def set_chunk_snippets(self, snippets: dict[str, str]) -> None:
        for raw_chunk_id, raw_snippet in snippets.items():
            chunk_id = str(raw_chunk_id).strip()
            snippet = str(raw_snippet).strip()
            if not chunk_id or not snippet:
                continue
            self._chunk_snippets[chunk_id] = snippet

    def set_chunk_page_hints(self, page_hints: dict[str, str]) -> None:
        for raw_chunk_id, raw_page_id in page_hints.items():
            chunk_id = str(raw_chunk_id).strip()
            page_id = str(raw_page_id).strip()
            if not chunk_id or not page_id:
                continue
            self._chunk_page_hints[chunk_id] = page_id

    def set_token_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self._prompt_tokens = max(0, int(prompt_tokens))
        self._completion_tokens = max(0, int(completion_tokens))
        self._total_tokens = max(0, int(total_tokens))

    def set_models(self, *, embed: str = "", rerank: str = "", llm: str = "") -> None:
        if embed:
            self._model_embed = embed
        if rerank:
            self._model_rerank = rerank
        if llm:
            self._model_llm = llm

    def set_retried(self, retried: bool = True) -> None:
        self._retried = retried

    def set_model_upgraded(self, upgraded: bool = True) -> None:
        self._model_upgraded = upgraded

    def set_generation_mode(self, mode: str) -> None:
        normalized = mode.strip().lower()
        if normalized in {"stream", "single_shot"}:
            self._generation_mode = normalized

    def set_llm_diagnostics(
        self,
        *,
        provider: str = "",
        finish_reason: str = "",
        malformed_tail_detected: bool | None = None,
    ) -> None:
        if provider.strip():
            self._llm_provider = provider.strip()
        if finish_reason.strip():
            self._llm_finish_reason = finish_reason.strip()
        if malformed_tail_detected is not None:
            self._malformed_tail_detected = bool(malformed_tail_detected)

    def set_context_stats(self, *, chunk_count: int | None = None, budget_tokens: int | None = None) -> None:
        if chunk_count is not None:
            self._context_chunk_count = max(0, int(chunk_count))
        if budget_tokens is not None:
            self._context_budget_tokens = max(0, int(budget_tokens))

    def set_request_metadata(self, *, question_id: str | None = None, answer_type: str | None = None, doc_refs: list[str] | None = None) -> None:
        if question_id is not None and question_id.strip():
            self._question_id = question_id.strip()
        if answer_type is not None and answer_type.strip():
            self._answer_type = answer_type.strip()
        if doc_refs is not None:
            self._doc_refs = list(doc_refs)

    def finalize(self) -> TelemetryPayload:
        now = time.perf_counter()
        total_ms = int((now - self._start_time) * 1000.0)

        ttft_ms = (
            total_ms
            if self._ttft_time is None
            else int((self._ttft_time - self._start_time) * 1000.0)
        )

        if ttft_ms < 0:
            ttft_ms = 0
        if total_ms < ttft_ms:
            total_ms = ttft_ms

        self._validate_chunk_subset_chain()

        time_per_output_token_ms = 0
        if self._completion_tokens > 0:
            # End-to-end per-token time after the first token is emitted.
            time_per_output_token_ms = int(max(0.0, (total_ms - ttft_ms) / self._completion_tokens))

        retrieved_pages = self._chunk_ids_to_page_ids(self._retrieved_ids)
        context_pages = self._chunk_ids_to_page_ids(self._context_ids)
        cited_pages = self._chunk_ids_to_page_ids(self._cited_ids)
        if self._used_page_ids_override is not None:
            used_pages = list(self._used_page_ids_override)
        elif self._used_ids:
            used_pages = self._chunk_ids_to_page_ids(self._used_ids)
        else:
            used_pages = list(cited_pages)

        return TelemetryPayload(
            request_id=self._request_id,
            question_id=self._question_id,
            answer_type=self._answer_type,
            ttft_ms=ttft_ms,
            time_per_output_token_ms=time_per_output_token_ms,
            total_ms=total_ms,
            embed_ms=int(self._timings_ms.get("embed", 0.0)),
            qdrant_ms=int(self._timings_ms.get("qdrant", 0.0)),
            rerank_ms=int(self._timings_ms.get("rerank", 0.0)),
            llm_ms=int(self._timings_ms.get("llm", 0.0)),
            verify_ms=int(self._timings_ms.get("verify", 0.0)),
            classify_ms=self._timings_ms.get("classify", 0.0),
            grounding_ms=int(self._timings_ms.get("grounding", 0.0)),
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._total_tokens,
            retrieved_chunk_ids=list(self._retrieved_ids),
            context_chunk_ids=list(self._context_ids),
            cited_chunk_ids=list(self._cited_ids),
            chunk_snippets=self._finalize_chunk_snippets(),
            retrieved_page_ids=retrieved_pages,
            context_page_ids=context_pages,
            cited_page_ids=cited_pages,
            used_page_ids=used_pages,
            doc_refs=list(self._doc_refs),
            model_embed=self._model_embed,
            model_rerank=self._model_rerank,
            selective_icr_shadow_used=self._selective_icr_shadow_used,
            selective_icr_shadow_model=self._selective_icr_shadow_model,
            selective_icr_shadow_latency_ms=self._selective_icr_shadow_latency_ms,
            selective_icr_shadow_candidate_count=self._selective_icr_shadow_candidate_count,
            selective_icr_shadow_chunk_ids=list(self._selective_icr_shadow_chunk_ids),
            selective_icr_shadow_page_ids=list(self._selective_icr_shadow_page_ids),
            selective_icr_shadow_provider_exit=self._selective_icr_shadow_provider_exit,
            model_llm=self._model_llm,
            llm_provider=self._llm_provider,
            llm_finish_reason=self._llm_finish_reason,
            generation_mode=self._generation_mode,
            context_chunk_count=self._context_chunk_count,
            context_budget_tokens=self._context_budget_tokens,
            malformed_tail_detected=self._malformed_tail_detected,
            retried=self._retried,
            model_upgraded=self._model_upgraded,
            trained_page_scorer_used=self._trained_page_scorer_used,
            trained_page_scorer_model_path=self._trained_page_scorer_model_path,
            trained_page_scorer_page_ids=list(self._trained_page_scorer_page_ids),
            trained_page_scorer_fallback_reason=self._trained_page_scorer_fallback_reason,
            grounding_portfolio_candidates=list(self._grounding_portfolio_candidates),
            grounding_portfolio_selected=self._grounding_portfolio_selected,
            grounding_portfolio_decision_reasons=list(self._grounding_portfolio_decision_reasons),
            grounding_audit_failed_slots=list(self._grounding_audit_failed_slots),
            grounding_audit_fallback_used=self._grounding_audit_fallback_used,
            grounding_escalation_triggered=self._grounding_escalation_triggered,
            grounding_escalation_reasons=list(self._grounding_escalation_reasons),
            grounding_escalation_family=self._grounding_escalation_family,
            grounding_shadow_rewrite_used=self._grounding_shadow_rewrite_used,
            grounding_shadow_rewrite_family=self._grounding_shadow_rewrite_family,
            grounding_relevance_verifier_used=self._grounding_relevance_verifier_used,
            grounding_relevance_verifier_confidence=self._grounding_relevance_verifier_confidence,
            grounding_relevance_verifier_fallback_reason=self._grounding_relevance_verifier_fallback_reason,
            claim_graph_enabled=self._claim_graph_enabled,
            claim_graph_count=self._claim_graph_count,
            claim_graph_unsupported_count=self._claim_graph_unsupported_count,
            claim_graph_support_coverage=self._claim_graph_support_coverage,
            proof_compiler_enabled=self._proof_compiler_enabled,
            proof_compiler_used=self._proof_compiler_used,
            proof_compiler_support_coverage=self._proof_compiler_support_coverage,
            proof_compiler_verified_claim_count=self._proof_compiler_verified_claim_count,
            proof_compiler_dropped_claim_count=self._proof_compiler_dropped_claim_count,
            proof_compiler_fallback_reason=self._proof_compiler_fallback_reason,
            proof_compiler_fully_supported=self._proof_compiler_fully_supported,
        )

    def _chunk_ids_to_page_ids(self, ids: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in ids:
            chunk_id = str(raw).strip()
            if not chunk_id:
                continue
            page_id = self._chunk_page_hints.get(chunk_id) or TelemetryCollector._chunk_id_to_page_id(chunk_id)
            if not page_id or page_id in seen:
                continue
            seen.add(page_id)
            out.append(page_id)
        return out

    @staticmethod
    def _chunk_id_to_page_id(chunk_id: str) -> str:
        # If a starter-kit chunk ID already comes as `pdf_id_page`, keep it.
        if ":" not in chunk_id and "_" in chunk_id:
            return chunk_id
        parts = chunk_id.split(":")
        if len(parts) < 2:
            return ""
        doc_id = parts[0].strip()
        page_idx_raw = parts[1].strip()
        if not doc_id or not page_idx_raw.isdigit():
            return ""
        # Internally we use 0-based section/page index; competition expects 1-based page numbers.
        return f"{doc_id}_{int(page_idx_raw) + 1}"

    def _finalize_chunk_snippets(self) -> dict[str, str]:
        ordered_ids = self._ordered_unique(
            [
                *self._retrieved_ids,
                *self._context_ids,
                *self._cited_ids,
                *self._used_ids,
            ]
        )
        return {
            chunk_id: self._chunk_snippets[chunk_id]
            for chunk_id in ordered_ids
            if chunk_id in self._chunk_snippets
        }

    @staticmethod
    def _ordered_unique(ids: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for raw in ids:
            chunk_id = str(raw).strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            ordered.append(chunk_id)
        return ordered

    def _validate_chunk_subset_chain(self) -> None:
        retrieved_set = set(self._retrieved_ids)
        context_set = set(self._context_ids)
        cited_set = set(self._cited_ids)
        used_set = set(self._used_ids)

        if context_set and not context_set.issubset(retrieved_set):
            logger.warning(
                "context_chunk_ids not subset of retrieved_chunk_ids",
                extra={"request_id": self._request_id},
            )
        if cited_set and not cited_set.issubset(context_set):
            logger.warning(
                "cited_chunk_ids not subset of context_chunk_ids",
                extra={"request_id": self._request_id},
            )
        if used_set and not used_set.issubset(context_set):
            logger.warning(
                "used_chunk_ids not subset of context_chunk_ids",
                extra={"request_id": self._request_id},
            )
