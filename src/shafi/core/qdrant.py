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

from shafi.config import get_settings
from shafi.models import (
    BridgeFact,
    Chunk,
    ChunkMetadata,
    LegalSegment,
    PageMetadata,
    SupportFact,
    SupportFactMetadata,
)

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

    @property
    def shadow_collection_name(self) -> str:
        return self._settings.shadow_collection

    async def close(self) -> None:
        if not self._external_client:
            await self._client.close()

    async def ensure_collection(self) -> None:
        """Create collection with dense + sparse vector configs if missing."""
        await self._ensure_chunk_collection(self._settings.collection)

    async def ensure_shadow_collection(self) -> None:
        """Create the retrieval-only shadow collection if missing."""
        await self._ensure_chunk_collection(self._settings.shadow_collection)

    async def ensure_payload_indexes(self) -> None:
        """Create keyword indexes on filterable payload fields."""
        await self._ensure_chunk_payload_indexes(self._settings.collection)

    async def ensure_shadow_payload_indexes(self) -> None:
        """Create keyword indexes on the retrieval-only shadow collection."""
        await self._ensure_chunk_payload_indexes(self._settings.shadow_collection)

    @property
    def page_collection_name(self) -> str:
        return self._settings.page_collection

    @property
    def segment_collection_name(self) -> str:
        """Return the configured additive legal-segment collection name."""

        return self._settings.segment_collection

    async def ensure_page_collection(self) -> None:
        """Create page-level collection with dense + sparse vector configs if missing."""
        coll = self._settings.page_collection
        exists = await self._client.collection_exists(coll)
        if exists:
            logger.info("Page collection '%s' already exists", coll)
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
                    collection_name=coll,
                    vectors_config=dense_vectors,
                    sparse_vectors_config={
                        "bm25": SparseVectorParams(modifier=Modifier.IDF),
                    },
                )
            else:
                await self._client.create_collection(
                    collection_name=coll,
                    vectors_config=dense_vectors,
                )
        except UnexpectedResponse as exc:
            if self._is_collection_already_exists(exc):
                logger.info("Page collection '%s' was created concurrently", coll)
                return
            raise
        logger.info("Created page collection '%s'", coll)

    async def ensure_segment_collection(self) -> None:
        """Create segment-level collection with dense + sparse vector configs if missing."""

        await self._ensure_chunk_collection(self._settings.segment_collection)

    async def ensure_page_payload_indexes(self) -> None:
        """Create keyword indexes on page collection payload fields."""
        coll = self._settings.page_collection
        keyword_fields = (
            "doc_id",
            "page_id",
            "doc_title",
            "doc_type",
            "ingest_version",
            "article_refs",
            "normalized_refs",
            "law_titles",
            "case_numbers",
            "page_family",
            "doc_family",
            "page_role",
            "amount_roles",
            "linked_refs",
            "field_labels_present",
            "document_template_family",
            "page_template_family",
            "canonical_law_family",
            "law_title_aliases",
            "related_law_families",
            "canonical_entity_ids",
        )
        for field_name in keyword_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=coll,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug("Page payload index on '%s' may already exist", field_name, exc_info=True)
        try:
            await self._client.create_payload_index(
                collection_name=coll,
                field_name="page_num",
                field_schema=PayloadSchemaType.INTEGER,
            )
        except Exception:
            logger.debug("Page payload index on 'page_num' may already exist", exc_info=True)

    async def ensure_segment_payload_indexes(self) -> None:
        """Create keyword indexes on segment collection payload fields."""

        coll = self._settings.segment_collection
        keyword_fields = (
            "segment_id",
            "segment_type",
            "doc_id",
            "doc_title",
            "doc_type",
            "canonical_doc_id",
            "page_ids",
            "parent_segment_id",
            "child_segment_ids",
            "canonical_entity_ids",
        )
        for field_name in keyword_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=coll,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug("Segment payload index on '%s' may already exist", field_name, exc_info=True)
        for field_name in ("start_page", "end_page"):
            try:
                await self._client.create_payload_index(
                    collection_name=coll,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.INTEGER,
                )
            except Exception:
                logger.debug("Segment payload index on '%s' may already exist", field_name, exc_info=True)

    async def upsert_pages(
        self,
        pages: list[PageMetadata],
        dense_vectors: list[list[float]],
        *,
        sparse_vectors: list[SparseVector] | None = None,
    ) -> int:
        """Upsert page-level points into the page collection."""
        if len(pages) != len(dense_vectors):
            raise ValueError(f"pages ({len(pages)}) and dense_vectors ({len(dense_vectors)}) length mismatch")
        if sparse_vectors is not None and len(pages) != len(sparse_vectors):
            raise ValueError(f"pages ({len(pages)}) and sparse_vectors ({len(sparse_vectors)}) length mismatch")

        coll = self._settings.page_collection
        total = 0
        batch_size = self._ingestion_settings.upsert_batch_size

        for offset in range(0, len(pages), batch_size):
            batch_pages = pages[offset : offset + batch_size]
            batch_dense = dense_vectors[offset : offset + batch_size]
            batch_sparse = None if sparse_vectors is None else sparse_vectors[offset : offset + batch_size]
            points: list[PointStruct] = []

            for idx, (page, dense_vec) in enumerate(zip(batch_pages, batch_dense, strict=True)):
                vector_data: dict[str, list[float] | Document | SparseVector] = {"dense": dense_vec}
                if batch_sparse is not None:
                    vector_data["bm25"] = batch_sparse[idx]

                point_uuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_URL, page.page_id))
                points.append(
                    PointStruct(
                        id=point_uuid,
                        vector=cast("VectorStruct", vector_data),
                        payload=page.model_dump(mode="json"),
                    )
                )

            await self._client.upsert(collection_name=coll, points=points)
            total += len(points)
            logger.debug("Upserted page batch at %d (%d points)", offset, len(points))

        logger.info("Upserted %d page points into '%s'", total, coll)
        return total

    async def upsert_segments(
        self,
        segments: list[LegalSegment],
        dense_vectors: list[list[float]],
        *,
        sparse_vectors: list[SparseVector] | None = None,
    ) -> int:
        """Upsert legal segments into the additive segment collection.

        Args:
            segments: Compiled legal segments to upsert.
            dense_vectors: Dense embedding vectors, one per segment.
            sparse_vectors: Optional sparse BM25 vectors, one per segment.

        Returns:
            int: Number of points upserted.
        """

        if len(segments) != len(dense_vectors):
            raise ValueError(f"segments ({len(segments)}) and dense_vectors ({len(dense_vectors)}) length mismatch")
        if sparse_vectors is not None and len(segments) != len(sparse_vectors):
            raise ValueError(f"segments ({len(segments)}) and sparse_vectors ({len(sparse_vectors)}) length mismatch")

        coll = self._settings.segment_collection
        total = 0
        batch_size = self._ingestion_settings.upsert_batch_size

        for offset in range(0, len(segments), batch_size):
            batch_segments = segments[offset : offset + batch_size]
            batch_dense = dense_vectors[offset : offset + batch_size]
            batch_sparse = None if sparse_vectors is None else sparse_vectors[offset : offset + batch_size]
            points: list[PointStruct] = []

            for idx, (segment, dense_vec) in enumerate(zip(batch_segments, batch_dense, strict=True)):
                vector_data: dict[str, list[float] | Document | SparseVector] = {"dense": dense_vec}
                if batch_sparse is not None:
                    vector_data["bm25"] = batch_sparse[idx]

                point_uuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_URL, segment.segment_id))
                points.append(
                    PointStruct(
                        id=point_uuid,
                        vector=cast("VectorStruct", vector_data),
                        payload=segment.model_dump(mode="json"),
                    )
                )

            await self._client.upsert(collection_name=coll, points=points)
            total += len(points)

        logger.info("Upserted %d legal segments into '%s'", total, coll)
        return total

    async def delete_pages_by_doc_id(self, doc_id: str) -> None:
        """Delete all page points for a document id from the page collection."""
        coll = self._settings.page_collection
        try:
            exists = await self._client.collection_exists(coll)
            if not exists:
                return
        except Exception:
            return
        await self._client.delete(
            collection_name=coll,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        )

    async def delete_segments_by_doc_id(self, doc_id: str) -> None:
        """Delete all segment points for a document id from the segment collection."""

        coll = self._settings.segment_collection
        try:
            exists = await self._client.collection_exists(coll)
            if not exists:
                return
        except Exception:
            return
        await self._client.delete(
            collection_name=coll,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        )

    # -- Support-fact collection --

    @property
    def support_fact_collection_name(self) -> str:
        return self._settings.support_fact_collection

    @property
    def bridge_fact_collection_name(self) -> str:
        """Return the configured bridge-fact collection name."""

        return self._settings.bridge_fact_collection

    async def ensure_support_fact_collection(self) -> None:
        """Create support-fact collection with dense + sparse vector configs if missing."""
        coll = self._settings.support_fact_collection
        exists = await self._client.collection_exists(coll)
        if exists:
            logger.info("Support-fact collection '%s' already exists", coll)
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
                    collection_name=coll,
                    vectors_config=dense_vectors,
                    sparse_vectors_config={
                        "bm25": SparseVectorParams(modifier=Modifier.IDF),
                    },
                )
            else:
                await self._client.create_collection(
                    collection_name=coll,
                    vectors_config=dense_vectors,
                )
        except UnexpectedResponse as exc:
            if self._is_collection_already_exists(exc):
                logger.info("Support-fact collection '%s' was created concurrently", coll)
                return
            raise
        logger.info("Created support-fact collection '%s'", coll)

    async def ensure_bridge_fact_collection(self) -> None:
        """Create bridge-fact collection with dense + sparse vector configs if missing."""

        await self._ensure_chunk_collection(self._settings.bridge_fact_collection)

    async def ensure_support_fact_payload_indexes(self) -> None:
        """Create keyword indexes on support-fact collection payload fields."""
        coll = self._settings.support_fact_collection
        keyword_fields = (
            "fact_id",
            "doc_id",
            "page_id",
            "doc_title",
            "doc_type",
            "doc_family",
            "page_family",
            "page_role",
            "fact_type",
            "normalized_value",
            "scope_ref",
        )
        for field_name in keyword_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=coll,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug("Support-fact index on '%s' may already exist", field_name, exc_info=True)
        try:
            await self._client.create_payload_index(
                collection_name=coll,
                field_name="page_num",
                field_schema=PayloadSchemaType.INTEGER,
            )
        except Exception:
            logger.debug("Support-fact index on 'page_num' may already exist", exc_info=True)

    async def ensure_bridge_fact_payload_indexes(self) -> None:
        """Create payload indexes on the bridge-fact collection."""

        coll = self._settings.bridge_fact_collection
        keyword_fields = (
            "fact_id",
            "fact_type",
            "source_entity_ids",
            "source_doc_ids",
            "evidence_page_ids",
        )
        for field_name in keyword_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=coll,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug("Bridge-fact index on '%s' may already exist", field_name, exc_info=True)

    async def upsert_support_facts(
        self,
        facts: list[SupportFact],
        dense_vectors: list[list[float]],
        *,
        sparse_vectors: list[SparseVector] | None = None,
    ) -> int:
        """Upsert support-fact points into the support-fact collection.

        Args:
            facts: Support facts to upsert.
            dense_vectors: Dense embedding vectors, one per fact.
            sparse_vectors: Optional sparse BM25 vectors, one per fact.

        Returns:
            Number of points upserted.
        """
        if len(facts) != len(dense_vectors):
            raise ValueError(f"facts ({len(facts)}) and dense_vectors ({len(dense_vectors)}) length mismatch")
        if sparse_vectors is not None and len(facts) != len(sparse_vectors):
            raise ValueError(f"facts ({len(facts)}) and sparse_vectors ({len(sparse_vectors)}) length mismatch")

        coll = self._settings.support_fact_collection
        total = 0
        batch_size = self._ingestion_settings.upsert_batch_size

        for offset in range(0, len(facts), batch_size):
            batch_facts = facts[offset : offset + batch_size]
            batch_dense = dense_vectors[offset : offset + batch_size]
            batch_sparse = None if sparse_vectors is None else sparse_vectors[offset : offset + batch_size]
            points: list[PointStruct] = []

            for idx, (fact, dense_vec) in enumerate(zip(batch_facts, batch_dense, strict=True)):
                vector_data: dict[str, list[float] | SparseVector] = {"dense": dense_vec}
                if batch_sparse is not None:
                    vector_data["bm25"] = batch_sparse[idx]

                point_uuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_URL, fact.fact_id))
                payload = SupportFactMetadata(**fact.model_dump(mode="json")).model_dump(mode="json")
                points.append(
                    PointStruct(
                        id=point_uuid,
                        vector=cast("VectorStruct", vector_data),
                        payload=payload,
                    )
                )

            await self._client.upsert(collection_name=coll, points=points)
            total += len(points)

        logger.info("Upserted %d support facts into '%s'", total, coll)
        return total

    async def upsert_bridge_facts(
        self,
        facts: list[BridgeFact],
        dense_vectors: list[list[float]],
        *,
        sparse_vectors: list[SparseVector] | None = None,
    ) -> int:
        """Upsert bridge facts into the additive bridge-fact collection.

        Args:
            facts: Bridge facts to upsert.
            dense_vectors: Dense embedding vectors, one per fact.
            sparse_vectors: Optional sparse vectors, one per fact.

        Returns:
            int: Number of bridge-fact points upserted.
        """

        if len(facts) != len(dense_vectors):
            raise ValueError(f"facts ({len(facts)}) and dense_vectors ({len(dense_vectors)}) length mismatch")
        if sparse_vectors is not None and len(facts) != len(sparse_vectors):
            raise ValueError(f"facts ({len(facts)}) and sparse_vectors ({len(sparse_vectors)}) length mismatch")

        coll = self._settings.bridge_fact_collection
        total = 0
        batch_size = self._ingestion_settings.upsert_batch_size
        for offset in range(0, len(facts), batch_size):
            batch_facts = facts[offset : offset + batch_size]
            batch_dense = dense_vectors[offset : offset + batch_size]
            batch_sparse = None if sparse_vectors is None else sparse_vectors[offset : offset + batch_size]
            points: list[PointStruct] = []
            for idx, (fact, dense_vec) in enumerate(zip(batch_facts, batch_dense, strict=True)):
                vector_data: dict[str, list[float] | SparseVector] = {"dense": dense_vec}
                if batch_sparse is not None:
                    vector_data["bm25"] = batch_sparse[idx]
                point_uuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_URL, fact.fact_id))
                points.append(
                    PointStruct(
                        id=point_uuid,
                        vector=cast("VectorStruct", vector_data),
                        payload=fact.model_dump(mode="json"),
                    )
                )
            await self._client.upsert(collection_name=coll, points=points)
            total += len(points)
        logger.info("Upserted %d bridge facts into '%s'", total, coll)
        return total

    async def delete_support_facts_by_doc_id(self, doc_id: str) -> None:
        """Delete all support-fact points for a document id."""
        coll = self._settings.support_fact_collection
        try:
            exists = await self._client.collection_exists(coll)
            if not exists:
                return
        except Exception:
            return
        await self._client.delete(
            collection_name=coll,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        )

    async def delete_bridge_facts_by_doc_id(self, doc_id: str) -> None:
        """Delete bridge facts that reference the given source document ID."""

        coll = self._settings.bridge_fact_collection
        try:
            exists = await self._client.collection_exists(coll)
            if not exists:
                return
        except Exception:
            return
        await self._client.delete(
            collection_name=coll,
            points_selector=Filter(must=[FieldCondition(key="source_doc_ids", match=MatchValue(value=doc_id))]),
        )

    async def upsert_chunks(
        self,
        chunks: list[Chunk] | tuple[Chunk, ...],
        dense_vectors: list[list[float]] | tuple[list[float], ...],
        *,
        sparse_vectors: list[SparseVector] | tuple[SparseVector, ...] | None = None,
    ) -> int:
        """Upsert chunks with dense vectors and optional/precomputed sparse vectors."""
        return await self._upsert_chunks_to_collection(
            collection_name=self._settings.collection,
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
        )

    async def upsert_shadow_chunks(
        self,
        chunks: list[Chunk] | tuple[Chunk, ...],
        dense_vectors: list[list[float]] | tuple[list[float], ...],
        *,
        sparse_vectors: list[SparseVector] | tuple[SparseVector, ...] | None = None,
    ) -> int:
        """Upsert retrieval-only shadow chunks into the shadow collection."""
        return await self._upsert_chunks_to_collection(
            collection_name=self._settings.shadow_collection,
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
        )

    async def _upsert_chunks_to_collection(
        self,
        *,
        collection_name: str,
        chunks: list[Chunk] | tuple[Chunk, ...],
        dense_vectors: list[list[float]] | tuple[list[float], ...],
        sparse_vectors: list[SparseVector] | tuple[SparseVector, ...] | None = None,
    ) -> int:
        """Upsert chunks into a specific collection."""
        if len(chunks) != len(dense_vectors):
            raise ValueError(f"chunks ({len(chunks)}) and dense_vectors ({len(dense_vectors)}) must have equal length")
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
                    logger.debug(
                        "Skipping BM25 sparse vector upsert for chunk_id=%s (sparse_vectors not provided)",
                        chunk.chunk_id,
                    )

                payload = self._chunk_payload(chunk)

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
                    collection_name=collection_name,
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
                        collection_name=collection_name,
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

        logger.info("Upserted %d total points into '%s'", total, collection_name)
        return total

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all points for a document id."""
        await self._delete_doc_from_collection(collection_name=self._settings.collection, doc_id=doc_id)
        await self._delete_doc_from_collection(collection_name=self._settings.shadow_collection, doc_id=doc_id)
        await self._delete_doc_from_collection(collection_name=self._settings.segment_collection, doc_id=doc_id)
        await self.delete_bridge_facts_by_doc_id(doc_id)
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
        await self._delete_stale_doc_versions_from_shadow(doc_id=doc_id, keep_ingest_version=keep_ingest_version)

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
        return getattr(exc, "status_code", None) == 500 and "InferenceService is not initialized" in text

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

    async def _ensure_chunk_collection(self, collection_name: str) -> None:
        exists = await self._client.collection_exists(collection_name)
        if exists:
            logger.info("Collection '%s' already exists", collection_name)
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
                    collection_name=collection_name,
                    vectors_config=dense_vectors,
                    sparse_vectors_config={
                        "bm25": SparseVectorParams(modifier=Modifier.IDF),
                    },
                )
            else:
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=dense_vectors,
                )
        except UnexpectedResponse as exc:
            if self._is_collection_already_exists(exc):
                logger.info("Collection '%s' was created concurrently", collection_name)
                return
            raise
        logger.info("Created collection '%s'", collection_name)

    async def _ensure_chunk_payload_indexes(self, collection_name: str) -> None:
        index_fields = [
            "doc_id",
            "doc_title",
            "doc_type",
            "jurisdiction",
            "chunk_id",
            "ingest_version",
            "citations",
            "anchors",
            "chunk_type",
            "doc_family",
            "page_family",
            "normalized_title",
            "normalized_refs",
            "amount_roles",
            "party_names",
            "court_names",
            "law_titles",
            "article_refs",
            "case_numbers",
            "cross_refs",
            "canonical_entity_ids",
            "segment_id",
        ]
        for field in index_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug("Payload index on '%s' may already exist", field, exc_info=True)
            else:
                logger.info("Created payload index on '%s' for '%s'", field, collection_name)

    def _chunk_payload(self, chunk: Chunk) -> dict[str, object]:
        return ChunkMetadata(
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
            chunk_type=chunk.chunk_type,
            doc_family=chunk.doc_family,
            page_family=chunk.page_family,
            normalized_title=chunk.normalized_title,
            normalized_refs=chunk.normalized_refs,
            amount_roles=chunk.amount_roles,
            shadow_search_text=chunk.shadow_search_text,
            party_names=chunk.party_names,
            court_names=chunk.court_names,
            law_titles=chunk.law_titles,
            article_refs=chunk.article_refs,
            case_numbers=chunk.case_numbers,
            cross_refs=chunk.cross_refs,
            canonical_entity_ids=chunk.canonical_entity_ids,
            segment_id=chunk.segment_id,
        ).model_dump(mode="json")

    async def _delete_doc_from_collection(self, *, collection_name: str, doc_id: str) -> None:
        try:
            exists = await self._client.collection_exists(collection_name)
            if not exists:
                return
        except Exception:
            return
        await self._client.delete(
            collection_name=collection_name,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        )

    async def _delete_stale_doc_versions_from_shadow(self, *, doc_id: str, keep_ingest_version: str) -> None:
        try:
            exists = await self._client.collection_exists(self._settings.shadow_collection)
            if not exists:
                return
        except Exception:
            return
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
                collection_name=self._settings.shadow_collection,
                points_selector=selector,
            )
        except UnexpectedResponse as exc:
            if self._is_missing_ingest_version_index(exc):
                await self._client.create_payload_index(
                    collection_name=self._settings.shadow_collection,
                    field_name="ingest_version",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                await self._client.delete(
                    collection_name=self._settings.shadow_collection,
                    points_selector=selector,
                )
            else:
                raise
