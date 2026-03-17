from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Literal

from rag_challenge.models import TelemetryPayload

logger = logging.getLogger(__name__)

StageName = Literal["embed", "qdrant", "rerank", "llm", "classify", "verify"]


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
            model_llm=self._model_llm,
            llm_provider=self._llm_provider,
            llm_finish_reason=self._llm_finish_reason,
            generation_mode=self._generation_mode,
            context_chunk_count=self._context_chunk_count,
            context_budget_tokens=self._context_budget_tokens,
            malformed_tail_detected=self._malformed_tail_detected,
            retried=self._retried,
            model_upgraded=self._model_upgraded,
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
