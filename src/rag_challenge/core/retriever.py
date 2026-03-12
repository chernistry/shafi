from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, cast

from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_challenge.config import get_settings
from rag_challenge.core.circuit_breaker import CircuitBreaker
from rag_challenge.core.sparse_bm25 import BM25SparseEncoder
from rag_challenge.models import DocType, RetrievedChunk

if TYPE_CHECKING:
    from rag_challenge.core.embedding import EmbeddingClient
    from rag_challenge.core.qdrant import QdrantStore

logger = logging.getLogger(__name__)

_DIFC_CASE_RE = re.compile(r"^(CFI|CA|SCT|ENF|DEC|TCD|ARB)\s+0*(\d{1,4})/(\d{4})$", re.IGNORECASE)
_LAW_NO_RE = re.compile(r"^Law\s+No\.?\s*(\d+)\s+of\s+(\d{4})$", re.IGNORECASE)
_TITLE_WITH_YEAR_RE = re.compile(r"^(?P<title>.+?)\s+(?P<year>19\d{2}|20\d{2})$", re.IGNORECASE)
_MONEY_VALUE_RE = re.compile(r"\b(?:aed|usd|gbp|eur)\s*[\d,]+(?:\.\d+)?\b", re.IGNORECASE)


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
        self._qdrant_circuit = CircuitBreaker(
            name="qdrant",
            failure_threshold=int(self._qdrant_settings.circuit_failure_threshold),
            reset_timeout_s=float(self._qdrant_settings.circuit_reset_timeout_s),
        )

    async def embed_query(self, query: str) -> list[float]:
        return await self._embedder.embed_query(query)

    def get_last_retrieved_ids(self) -> list[str]:
        return list(self._last_retrieved_ids)

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

        if doc_type_filter is None and extracted_refs:
            case_ref_prefixes = {"CFI", "CA", "SCT", "ENF", "DEC", "TCD", "ARB"}
            has_case_ref = any(ref.split(" ", maxsplit=1)[0].upper() in case_ref_prefixes for ref in extracted_refs)
            if has_case_ref and bool(getattr(self._pipeline_settings, "doc_ref_case_law_filter", True)):
                doc_type_filter = DocType.CASE_LAW

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
                result = await self._query_sparse_only(query=query, limit=limit, where=where)
            except Exception as exc:
                logger.warning("Sparse-only retrieval failed; degrading to standard retrieval path: %s", exc)
                sparse_only = False
                result = None
        else:
            result = None

        if result is None:
            if query_vector is None:
                query_vector = await self._embedder.embed_query(query)

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
                    query=query,
                    query_vector=query_vector,
                    prefetch_dense=dense_limit,
                    prefetch_sparse=sparse_limit,
                    where=where,
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
                    try:
                        result = await self._query_dense_only(
                            query_vector=query_vector,
                            limit=limit,
                            where=where,
                        )
                    except Exception as dense_exc:
                        raise RetrieverError(f"Qdrant retrieval failed (hybrid+dense): {dense_exc}") from dense_exc

        chunks = self._map_results(result)

        if extracted_refs and not chunks:
            # Step 1 fail-open: keep doc refs, relax doc_type constraint first.
            logger.info("Doc-ref filter produced 0 chunks; retrying without doc_type filter")
            fallback_where = self._build_filter(
                doc_type_filter=None,
                jurisdiction_filter=jurisdiction_filter,
                doc_refs=expanded_refs,
            )
            if query_vector is None:
                query_vector = await self._embedder.embed_query(query)
            if sparse_only and self._bm25_enabled:
                result = await self._query_sparse_only(query=query, limit=limit, where=fallback_where)
            elif not self._bm25_enabled:
                result = await self._query_dense_only(
                    query_vector=query_vector,
                    limit=limit,
                    where=fallback_where,
                )
            else:
                prefetch = self._build_prefetch(
                    query=query,
                    query_vector=query_vector,
                    prefetch_dense=dense_limit,
                    prefetch_sparse=sparse_limit,
                    where=fallback_where,
                )
                fusion = self._resolve_fusion_method()
                result = await self._query_hybrid(prefetch=prefetch, fusion=fusion, limit=limit)
            chunks = self._map_results(result)

        if extracted_refs and not chunks:
            # Step 2 fail-open: final fallback without doc_refs.
            logger.info("Doc-ref filter still produced 0 chunks; retrying without doc_refs")
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
                    query=query,
                    query_vector=query_vector,
                    prefetch_dense=dense_limit,
                    prefetch_sparse=sparse_limit,
                    where=fallback_where,
                )
                fusion = self._resolve_fusion_method()
                result = await self._query_hybrid(prefetch=prefetch, fusion=fusion, limit=limit)
            chunks = self._map_results(result)

        chunks = self._apply_query_rescoring(
            query=query,
            chunks=chunks,
            extracted_refs=extracted_refs,
        )
        chunks = await self._inject_anchor_rescue_chunks(
            query=query,
            chunks=chunks,
            extracted_refs=extracted_refs,
        )
        self._last_retrieved_ids = [chunk.chunk_id for chunk in chunks]
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

    async def _inject_anchor_rescue_chunks(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        extracted_refs: list[str],
    ) -> list[RetrievedChunk]:
        target_page = self._anchor_rescue_target_page(query=query, extracted_refs=extracted_refs)
        if target_page is None or not chunks:
            return chunks

        multi_doc = len(extracted_refs) >= 2 or self._is_multi_doc_party_compare_query(query=query)
        max_docs = 3 if multi_doc else 1
        doc_ids = self._anchor_rescue_doc_ids(chunks=chunks, max_docs=max_docs)
        if not doc_ids:
            return chunks

        existing_chunk_ids = {chunk.chunk_id for chunk in chunks}
        rescued: list[RetrievedChunk] = []
        for doc_id in doc_ids:
            if self._doc_has_target_anchor(chunks=chunks, doc_id=doc_id, target_page=target_page):
                continue
            chunk = await self._fetch_best_anchor_chunk_for_doc(
                doc_id=doc_id,
                target_page=target_page,
                query=query,
                existing_chunks=chunks,
            )
            if chunk is None or chunk.chunk_id in existing_chunk_ids:
                continue
            existing_chunk_ids.add(chunk.chunk_id)
            rescued.append(chunk)

        if not rescued:
            return chunks
        return self._apply_query_rescoring(
            query=query,
            chunks=[*chunks, *rescued],
            extracted_refs=extracted_refs,
        )

    @classmethod
    def _anchor_rescue_target_page(cls, *, query: str, extracted_refs: list[str]) -> int | None:
        q = re.sub(r"\s+", " ", query).strip().casefold()
        if not q:
            return None
        if "page 2" in q or "second page" in q:
            return 2

        explicit_anchor = any(term in q for term in ("title page", "cover page", "first page", "header", "caption"))
        if explicit_anchor:
            return 1
        return None

    @staticmethod
    def _is_multi_doc_party_compare_query(*, query: str) -> bool:
        q = re.sub(r"\s+", " ", query).strip().casefold()
        if not q:
            return False
        return (
            "both cases" in q
            or "across all documents" in q
            or "common to both" in q
            or "appears as a main party" in q
            or "named as a main party" in q
            or "main party to both" in q
        )

    @staticmethod
    def _anchor_rescue_doc_ids(*, chunks: list[RetrievedChunk], max_docs: int) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            doc_id = str(chunk.doc_id or "").strip()
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            ordered.append(doc_id)
            if len(ordered) >= max(1, max_docs):
                break
        return ordered

    @staticmethod
    def _doc_has_target_anchor(
        *,
        chunks: list[RetrievedChunk],
        doc_id: str,
        target_page: int,
    ) -> bool:
        for chunk in chunks:
            if str(chunk.doc_id or "").strip() != doc_id:
                continue
            if chunk.page_number == target_page:
                return True
            if target_page == 1 and str(chunk.page_type or "").strip().lower() in {"title_anchor", "caption_anchor"}:
                return True
            if target_page == 2 and str(chunk.page_type or "").strip().lower() == "page2_anchor":
                return True
        return False

    async def _fetch_best_anchor_chunk_for_doc(
        self,
        *,
        doc_id: str,
        target_page: int,
        query: str,
        existing_chunks: list[RetrievedChunk],
    ) -> RetrievedChunk | None:
        try:
            result = await self._store.client.scroll(
                collection_name=self._store.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                        models.FieldCondition(key="page_number", match=models.MatchValue(value=target_page)),
                    ]
                ),
                limit=16,
                with_payload=self._payload_selector(),
                with_vectors=False,
            )
        except Exception:
            logger.debug("Anchor rescue scroll failed for doc_id=%s page=%s", doc_id, target_page, exc_info=True)
            return None

        points_obj = result[0]
        candidates = self._map_results(points_obj)
        if not candidates:
            return None

        best_score = max(float(chunk.score) for chunk in existing_chunks if str(chunk.doc_id or "").strip() == doc_id) if any(
            str(chunk.doc_id or "").strip() == doc_id for chunk in existing_chunks
        ) else 0.0

        best_candidate: tuple[int, float, RetrievedChunk] | None = None
        for candidate in candidates:
            score = self._anchor_rescue_candidate_score(query=query, chunk=candidate, target_page=target_page)
            if score <= 0:
                continue
            retrieval_score = float(candidate.score)
            current = (score, retrieval_score, candidate)
            if best_candidate is None or current[:2] > best_candidate[:2]:
                best_candidate = current
        if best_candidate is None:
            return None

        adjusted_score = best_score + 0.25 + (best_candidate[0] / 10_000.0)
        return best_candidate[2].model_copy(update={"score": adjusted_score})

    @staticmethod
    def _anchor_rescue_candidate_score(*, query: str, chunk: RetrievedChunk, target_page: int) -> int:
        q = re.sub(r"\s+", " ", query).strip().casefold()
        page_type = str(chunk.page_type or "").strip().casefold()
        page_num = chunk.page_number
        if page_num != target_page:
            return 0

        score = 100
        if target_page == 1:
            if page_type == "caption_anchor":
                score += 120
            elif page_type == "title_anchor":
                score += 100
            if chunk.has_caption_terms:
                score += 40
            if any(term in q for term in ("header", "caption", "title page", "cover page", "first page")):
                score += 40
        elif target_page == 2:
            if page_type == "page2_anchor":
                score += 120
            if chunk.has_order_terms:
                score += 30
        return score

    @classmethod
    def _apply_query_rescoring(
        cls,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        extracted_refs: list[str],
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        rescored: list[tuple[float, int, RetrievedChunk]] = []
        for index, chunk in enumerate(chunks):
            adjusted = float(chunk.score) + cls._metadata_bonus(
                query=query,
                chunk=chunk,
                extracted_refs=extracted_refs,
            )
            rescored.append((adjusted, index, chunk.model_copy(update={"score": adjusted})))
        rescored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in rescored]

    @classmethod
    def _metadata_bonus(cls, *, query: str, chunk: RetrievedChunk, extracted_refs: list[str]) -> float:
        q = re.sub(r"\s+", " ", query).strip().casefold()
        if not q:
            return 0.0

        page_number = chunk.page_number if chunk.page_number is not None else cls._page_number_from_section(chunk.section_path)
        page_type = (chunk.page_type or "").strip().casefold()
        heading_text = (chunk.heading_text or "").strip().casefold()
        doc_title = (chunk.doc_title or "").strip().casefold()
        doc_refs = {ref.casefold() for ref in chunk.doc_refs if ref.strip()}
        article_refs = {ref.casefold() for ref in chunk.article_refs if ref.strip()}
        bonus = 0.0

        if extracted_refs:
            matched_refs = 0
            for ref in extracted_refs:
                norm_ref = ref.casefold()
                if norm_ref in doc_title or norm_ref in doc_refs:
                    matched_refs += 1
            bonus += min(0.75, matched_refs * 0.25)

        wants_page_two = "page 2" in q or "second page" in q
        wants_title_anchor = any(term in q for term in ("title page", "cover page", "first page", "header", "caption"))
        wants_monetary_anchor = any(
            term in q
            for term in (
                "monetary claim",
                "higher monetary claim",
                "lower monetary claim",
                "higher claim",
                "lower claim",
                "claim amount",
                "financial limit",
            )
        )
        wants_outcome_anchor = any(term in q for term in ("outcome", "result", "costs", "it is hereby ordered", "order"))
        wants_heading_anchor = "article " in q or "schedule " in q or "definitions" in q
        chunk_text = chunk.text.casefold()
        has_monetary_signal = bool(_MONEY_VALUE_RE.search(chunk.text)) or any(
            term in chunk_text for term in ("financial limit", "claim form", "claim amount", "aed ")
        )

        if wants_page_two and page_number == 2:
            bonus += 0.8
            if page_type == "page2_anchor":
                bonus += 0.35

        if wants_title_anchor and page_number == 1:
            bonus += 0.35
        if wants_title_anchor and page_type in {"title_anchor", "caption_anchor"}:
            bonus += 0.65

        if wants_heading_anchor:
            heading_hits = 0
            if "article " in q and ("article " in heading_text or any(ref.startswith("article ") for ref in article_refs)):
                heading_hits += 1
            if "schedule " in q and ("schedule " in heading_text or any(ref.startswith("schedule ") for ref in article_refs)):
                heading_hits += 1
            if "definitions" in q and "definition" in heading_text:
                heading_hits += 1
            if heading_hits:
                bonus += 0.25 * heading_hits
                if page_type == "heading_window":
                    bonus += 0.25

        if wants_outcome_anchor:
            if chunk.has_order_terms:
                bonus += 0.55
            if page_type == "heading_window":
                bonus += 0.2

        if wants_monetary_anchor:
            if has_monetary_signal:
                bonus += 0.6
                if page_number is not None and page_number > 1:
                    bonus += 0.1
            elif page_type in {"title_anchor", "caption_anchor"} or page_number == 1:
                bonus -= 0.2

        if len(extracted_refs) >= 2 and wants_title_anchor and page_type in {"title_anchor", "caption_anchor", "page2_anchor"}:
            bonus += 0.15

        return bonus

    @staticmethod
    def _page_number_from_section(section_path: str | None) -> int | None:
        match = re.search(r"page:(\d+)", section_path or "", flags=re.IGNORECASE)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _coerce_optional_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_optional_int(value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return int(text)
            except ValueError:
                return None
        try:
            return int(str(value).strip())
        except ValueError:
            return None

    @staticmethod
    def _coerce_str_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [text for item in cast("list[object]", value) if (text := str(item).strip())]

    async def _query_hybrid(
        self,
        *,
        prefetch: list[models.Prefetch],
        fusion: models.Fusion,
        limit: int,
    ) -> object:
        if not self._qdrant_circuit.allow_request():
            raise RetrieverError("Qdrant circuit is open")
        try:
            result = await self._store.client.query_points(
                collection_name=self._store.collection_name,
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
        query_vector: list[float],
        limit: int,
        where: models.Filter | None,
    ) -> object:
        if not self._qdrant_circuit.allow_request():
            raise RetrieverError("Qdrant circuit is open")
        try:
            result = await self._store.client.query_points(
                collection_name=self._store.collection_name,
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
                collection_name=self._store.collection_name,
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

    def _resolve_fusion_method(self) -> models.Fusion:
        fusion_name = str(getattr(self._qdrant_settings, "fusion_method", "RRF")).upper()
        return cast("models.Fusion", getattr(models.Fusion, fusion_name, models.Fusion.RRF))

    @staticmethod
    def _is_qdrant_inference_unavailable(exc: Exception) -> bool:
        if not isinstance(exc, UnexpectedResponse):
            return False
        content = getattr(exc, "content", b"")
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = str(content)
        return (
            getattr(exc, "status_code", None) == 500
            and "InferenceService is not initialized" in text
        )

    @staticmethod
    def _is_fastembed_unavailable(exc: Exception) -> bool:
        text = str(exc).lower()
        return "fastembed" in text or "onnxruntime" in text

    @staticmethod
    def _coerce_int(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return int(stripped)
            except ValueError:
                return None
        return None

    @staticmethod
    def _build_filter(
        *,
        doc_type_filter: DocType | None,
        jurisdiction_filter: str | None,
        doc_refs: list[str] | tuple[str, ...] | None = None,
    ) -> models.Filter | None:
        conditions: list[object] = []
        if doc_type_filter is not None:
            conditions.append(
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(value=doc_type_filter.value),
                )
            )
        if jurisdiction_filter:
            conditions.append(
                models.FieldCondition(
                    key="jurisdiction",
                    match=models.MatchValue(value=jurisdiction_filter),
                )
            )
        refs = [ref.strip() for ref in (list(doc_refs) if doc_refs is not None else []) if str(ref).strip()]
        title_refs = HybridRetriever._doc_title_filter_variants(refs)
        if refs:
            ref_conditions: list[models.Condition] = [
                models.FieldCondition(
                    key="citations",
                    match=models.MatchAny(any=refs),
                )
            ]
            if title_refs:
                ref_conditions.append(
                    models.FieldCondition(
                        key="doc_title",
                        match=models.MatchAny(any=title_refs),
                    )
                )
            return models.Filter(
                must=cast("list[models.Condition]", conditions),
                should=ref_conditions,
            )
        return models.Filter(must=cast("list[models.Condition]", conditions)) if conditions else None

    @staticmethod
    def _expand_doc_ref_variants(refs: list[str] | tuple[str, ...]) -> list[str]:
        variants: list[str] = []
        for raw in refs:
            ref = raw.strip()
            if not ref:
                continue
            variants.append(ref)

            case_match = _DIFC_CASE_RE.match(ref)
            if case_match is not None:
                prefix = case_match.group(1).upper()
                num_raw = int(case_match.group(2))
                year = case_match.group(3)
                variants.append(f"{prefix} {num_raw}/{year}")
                variants.append(f"{prefix} {num_raw:03d}/{year}")
                variants.append(f"{prefix}{num_raw:03d}/{year}")
                variants.append(f"{prefix}{num_raw}/{year}")

            law_match = _LAW_NO_RE.match(ref)
            if law_match is not None:
                num = int(law_match.group(1))
                year = law_match.group(2)
                variants.append(f"Law No. {num} of {year}")
                variants.append(f"Law No {num} of {year}")
                variants.append(f"DIFC Law No. {num} of {year}")
                continue

            title_with_year_match = _TITLE_WITH_YEAR_RE.match(ref)
            if title_with_year_match is not None:
                title_only = re.sub(r"\s+", " ", title_with_year_match.group("title")).strip(" ,.;:")
                if title_only:
                    variants.append(title_only)
                    variants.append(title_only.upper())

        seen: set[str] = set()
        out: list[str] = []
        for candidate in variants:
            key = candidate.strip()
            if not key or key.lower() in seen:
                continue
            seen.add(key.lower())
            out.append(key)
        return out

    @staticmethod
    def _doc_title_filter_variants(refs: list[str] | tuple[str, ...]) -> list[str]:
        variants: list[str] = []
        for raw in refs:
            ref = re.sub(r"\s+", " ", str(raw).strip())
            if not ref or _LAW_NO_RE.match(ref) is not None or _DIFC_CASE_RE.match(ref) is not None:
                continue
            variants.append(ref)
            title_with_year_match = _TITLE_WITH_YEAR_RE.match(ref)
            if title_with_year_match is not None:
                title_only = re.sub(r"\s+", " ", title_with_year_match.group("title")).strip(" ,.;:")
                if title_only:
                    variants.append(title_only)
                    variants.append(title_only.upper())

        seen: set[str] = set()
        out: list[str] = []
        for candidate in variants:
            normalized = candidate.strip()
            if not normalized or normalized.casefold() in seen:
                continue
            seen.add(normalized.casefold())
            out.append(normalized)
        return out

    @classmethod
    def _map_results(cls, result: object) -> list[RetrievedChunk]:
        points = cls._extract_points(result)
        mapped: list[RetrievedChunk] = []
        for point_obj in points:
            chunk = cls._map_point(point_obj)
            if chunk is not None:
                mapped.append(chunk)
        return mapped

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
                "page_number",
                "page_type",
                "heading_text",
                "doc_refs",
                "law_no",
                "law_year",
                "article_refs",
                "has_caption_terms",
                "has_order_terms",
            ]
        )

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
                page_number=HybridRetriever._coerce_optional_int(payload.get("page_number")),
                page_type=HybridRetriever._coerce_optional_str(payload.get("page_type")),
                heading_text=HybridRetriever._coerce_optional_str(payload.get("heading_text")),
                doc_refs=HybridRetriever._coerce_str_list(payload.get("doc_refs")),
                law_no=HybridRetriever._coerce_optional_str(payload.get("law_no")),
                law_year=HybridRetriever._coerce_optional_str(payload.get("law_year")),
                article_refs=HybridRetriever._coerce_str_list(payload.get("article_refs")),
                has_caption_terms=bool(payload.get("has_caption_terms") or False),
                has_order_terms=bool(payload.get("has_order_terms") or False),
                doc_summary=str(payload.get("doc_summary") or ""),
            )
        except Exception:
            logger.warning("Failed to map Qdrant point %s", point_id, exc_info=True)
            return None
