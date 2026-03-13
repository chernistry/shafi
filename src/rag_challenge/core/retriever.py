from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, cast

from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_challenge.config import get_settings
from rag_challenge.core.circuit_breaker import CircuitBreaker
from rag_challenge.core.classifier import QueryClassifier
from rag_challenge.core.sparse_bm25 import BM25SparseEncoder
from rag_challenge.models import DocType, RetrievedChunk

if TYPE_CHECKING:
    from rag_challenge.core.embedding import EmbeddingClient
    from rag_challenge.core.qdrant import QdrantStore

logger = logging.getLogger(__name__)

_DIFC_CASE_RE = re.compile(r"^(CFI|CA|SCT|ENF|DEC|TCD|ARB)\s+0*(\d{1,4})/(\d{4})$", re.IGNORECASE)
_LAW_NO_RE = re.compile(r"^Law\s+No\.?\s*(\d+)\s+of\s+(\d{4})$", re.IGNORECASE)
_TITLE_WITH_YEAR_RE = re.compile(r"^(?P<title>.+?)\s+(?P<year>19\d{2}|20\d{2})$", re.IGNORECASE)


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
        sparse_query = self._build_sparse_query(query=query, extracted_refs=extracted_refs)

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
                result = await self._query_sparse_only(query=sparse_query, limit=limit, where=where)
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
                    query=sparse_query,
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
                    query=sparse_query,
                    query_vector=query_vector,
                    prefetch_dense=dense_limit,
                    prefetch_sparse=sparse_limit,
                    where=fallback_where,
                )
                fusion = self._resolve_fusion_method()
                result = await self._query_hybrid(prefetch=prefetch, fusion=fusion, limit=limit)
            chunks = self._map_results(result)

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

    @classmethod
    def _build_sparse_query(cls, *, query: str, extracted_refs: list[str] | tuple[str, ...]) -> str:
        base_query = str(query or "").strip()
        if not base_query:
            return ""
        if any(_DIFC_CASE_RE.match(str(ref).strip()) is not None for ref in extracted_refs):
            return base_query

        exact_refs = QueryClassifier.extract_exact_legal_refs(base_query)
        if not exact_refs:
            return base_query

        capped_refs = exact_refs[:4]
        boosted_tail = " ".join([*capped_refs, *capped_refs]).strip()
        if not boosted_tail:
            return base_query
        return f"{base_query}\n{boosted_tail}".strip()

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
                doc_summary=str(payload.get("doc_summary") or ""),
            )
        except Exception:
            logger.warning("Failed to map Qdrant point %s", point_id, exc_info=True)
            return None
