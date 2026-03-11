from __future__ import annotations

import logging
import uuid as uuid_lib
from typing import cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Document,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorStruct,
)

from rag_challenge.config import get_settings
from rag_challenge.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


class QdrantStore:
    """Async wrapper around Qdrant for legal chunk storage and retrieval."""

    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        settings = get_settings()
        self._settings = settings.qdrant
        self._embedding_settings = settings.embedding
        self._ingestion_settings = settings.ingestion
        self._external_client = client is not None
        self._bm25_enabled = bool(getattr(self._settings, "enable_sparse_bm25", True))
        self._sparse_model = str(getattr(self._settings, "sparse_model", "Qdrant/bm25"))
        self._client = client or AsyncQdrantClient(
            url=self._settings.url,
            api_key=self._settings.api_key or None,
            timeout=int(self._settings.timeout_s),
            cloud_inference=bool(getattr(self._settings, "use_cloud_inference", False)),
            check_compatibility=bool(getattr(self._settings, "check_compatibility", True)),
        )
        # NOTE: We do NOT rely on Qdrant's server-side BM25 inference. Sparse vectors are computed
        # client-side (fastembed) and sent as SparseVector objects during upsert/query.

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client

    @property
    def collection_name(self) -> str:
        return self._settings.collection

    async def close(self) -> None:
        if not self._external_client:
            await self._client.close()

    async def ensure_collection(self) -> None:
        """Create collection with dense + sparse vector configs if missing."""
        exists = await self._client.collection_exists(self._settings.collection)
        if exists:
            logger.info("Collection '%s' already exists", self._settings.collection)
            return

        dense_vectors = {
            "dense": VectorParams(
                size=self._embedding_settings.dimensions,
                distance=Distance.COSINE,
            ),
        }

        try:
            if self._bm25_enabled:
                await self._client.create_collection(
                    collection_name=self._settings.collection,
                    vectors_config=dense_vectors,
                    sparse_vectors_config={
                        "bm25": SparseVectorParams(modifier=Modifier.IDF),
                    },
                )
            else:
                await self._client.create_collection(
                    collection_name=self._settings.collection,
                    vectors_config=dense_vectors,
                )
        except UnexpectedResponse as exc:
            if self._is_collection_already_exists(exc):
                logger.info("Collection '%s' was created concurrently", self._settings.collection)
                return
            raise
        logger.info("Created collection '%s'", self._settings.collection)

    async def ensure_payload_indexes(self) -> None:
        """Create keyword indexes on filterable payload fields."""
        index_fields = [
            "doc_id",
            "doc_title",
            "doc_type",
            "jurisdiction",
            "chunk_id",
            "ingest_version",
            # Identifier-aware retrieval (arrays of keywords).
            "citations",
            "anchors",
        ]
        for field in index_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=self._settings.collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug("Payload index on '%s' may already exist", field, exc_info=True)
            else:
                logger.info("Created payload index on '%s'", field)

    async def upsert_chunks(
        self,
        chunks: list[Chunk] | tuple[Chunk, ...],
        dense_vectors: list[list[float]] | tuple[list[float], ...],
        *,
        sparse_vectors: list[SparseVector] | tuple[SparseVector, ...] | None = None,
    ) -> int:
        """Upsert chunks with dense vectors and optional/precomputed sparse vectors."""
        if len(chunks) != len(dense_vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and dense_vectors ({len(dense_vectors)}) must have equal length"
            )
        if sparse_vectors is not None and len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and sparse_vectors ({len(sparse_vectors)}) must have equal length"
            )
        total = 0
        batch_size = self._ingestion_settings.upsert_batch_size

        for offset in range(0, len(chunks), batch_size):
            batch_chunks = list(chunks[offset : offset + batch_size])
            batch_dense = list(dense_vectors[offset : offset + batch_size])
            batch_sparse = None if sparse_vectors is None else list(sparse_vectors[offset : offset + batch_size])
            points: list[PointStruct] = []

            for index, (chunk, dense_vec) in enumerate(zip(batch_chunks, batch_dense, strict=True)):
                vector_data: dict[str, list[float] | Document | SparseVector] = {"dense": dense_vec}
                if batch_sparse is not None:
                    vector_data["bm25"] = batch_sparse[index]
                elif self._bm25_enabled:
                    # Fail-open: dense-only upsert if sparse vectors are not provided (keeps ingestion running
                    # without depending on Qdrant InferenceService).
                    logger.debug("Skipping BM25 sparse vector upsert for chunk_id=%s (sparse_vectors not provided)", chunk.chunk_id)

                payload = ChunkMetadata(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    doc_type=chunk.doc_type,
                    jurisdiction=chunk.jurisdiction,
                    section_path=chunk.section_path,
                    citations=chunk.citations,
                    anchors=chunk.anchors,
                    ingest_version=self._ingestion_settings.ingest_version,
                    chunk_text=chunk.chunk_text,
                    doc_summary=chunk.doc_summary,
                ).model_dump(mode="json")

                point_uuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_URL, chunk.chunk_id))
                points.append(
                    PointStruct(
                        id=point_uuid,
                        vector=cast("VectorStruct", vector_data),
                        payload=payload,
                    )
                )

            try:
                await self._client.upsert(
                    collection_name=self._settings.collection,
                    points=points,
                )
            except Exception as exc:
                if isinstance(exc, UnexpectedResponse) and self._should_fallback_to_dense_only(exc):
                    logger.warning(
                        "Qdrant BM25 inference unavailable, retrying upsert as dense-only for batch at %d",
                        offset,
                    )
                    self._bm25_enabled = False
                    dense_only_points = [
                        PointStruct(
                            id=point.id,
                            vector={"dense": dense_vec},
                            payload=point.payload,
                        )
                        for point, dense_vec in zip(points, batch_dense, strict=True)
                    ]
                    await self._client.upsert(
                        collection_name=self._settings.collection,
                        points=dense_only_points,
                    )
                elif self._should_fallback_on_fastembed_error(exc):
                    logger.warning(
                        "Qdrant BM25 local model unavailable, retrying upsert as dense-only for batch at %d",
                        offset,
                    )
                    self._bm25_enabled = False
                    dense_only_points = [
                        PointStruct(
                            id=point.id,
                            vector={"dense": dense_vec},
                            payload=point.payload,
                        )
                        for point, dense_vec in zip(points, batch_dense, strict=True)
                    ]
                    await self._client.upsert(
                        collection_name=self._settings.collection,
                        points=dense_only_points,
                    )
                else:
                    raise
            total += len(points)
            logger.debug("Upserted batch starting at %d (%d points)", offset, len(points))

        logger.info("Upserted %d total points", total)
        return total

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all points for a document id."""
        await self._client.delete(
            collection_name=self._settings.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        logger.info("Deleted points for doc_id='%s'", doc_id)

    async def delete_stale_doc_versions(self, doc_id: str, *, keep_ingest_version: str) -> None:
        """Delete points for a doc where ingest_version != keep_ingest_version."""
        selector = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))],
            must_not=[
                FieldCondition(
                    key="ingest_version",
                    match=MatchValue(value=keep_ingest_version),
                )
            ],
        )
        try:
            await self._client.delete(
                collection_name=self._settings.collection,
                points_selector=selector,
            )
        except UnexpectedResponse as exc:
            if self._is_missing_ingest_version_index(exc):
                logger.warning(
                    "Missing payload index for ingest_version, creating it and retrying stale cleanup for doc_id=%s",
                    doc_id,
                )
                await self._client.create_payload_index(
                    collection_name=self._settings.collection,
                    field_name="ingest_version",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                await self._client.delete(
                    collection_name=self._settings.collection,
                    points_selector=selector,
                )
            else:
                raise
        logger.info(
            "Deleted stale points for doc_id='%s' (keeping ingest_version='%s')",
            doc_id,
            keep_ingest_version,
        )

    async def count_points(self) -> int:
        """Return total number of points in the collection."""
        info = await self._client.get_collection(self._settings.collection)
        points_count = getattr(info, "points_count", 0)
        return int(points_count or 0)

    async def health_check(self) -> bool:
        """Check Qdrant connectivity."""
        try:
            await self._client.get_collections()
        except Exception:
            logger.warning("Qdrant health check failed", exc_info=True)
            return False
        return True

    @staticmethod
    def _should_fallback_to_dense_only(exc: UnexpectedResponse) -> bool:
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
    def _should_fallback_on_fastembed_error(exc: Exception) -> bool:
        text = str(exc)
        return "fastembed" in text.lower() or "onnxruntime" in text.lower()

    @staticmethod
    def _is_collection_already_exists(exc: UnexpectedResponse) -> bool:
        status_code = getattr(exc, "status_code", None)
        content = getattr(exc, "content", b"")
        try:
            text = content.decode("utf-8", errors="ignore").lower()
        except Exception:
            text = str(content).lower()
        return status_code == 409 or "already exists" in text

    @staticmethod
    def _is_missing_ingest_version_index(exc: UnexpectedResponse) -> bool:
        status_code = getattr(exc, "status_code", None)
        content = getattr(exc, "content", b"")
        try:
            text = content.decode("utf-8", errors="ignore").lower()
        except Exception:
            text = str(content).lower()
        return status_code == 400 and "index required" in text and "ingest_version" in text
