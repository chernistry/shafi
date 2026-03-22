"""Build BM25 index from Qdrant chunks."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from qdrant_client import models

from rag_challenge.config import get_settings
from rag_challenge.core.bm25_retriever import BM25Retriever
from rag_challenge.core.qdrant import QdrantStore
from rag_challenge.models import DocType, RetrievedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fetch_all_chunks(store: QdrantStore) -> list[RetrievedChunk]:
    """Fetch all chunks from Qdrant collection."""
    chunks: list[RetrievedChunk] = []
    offset = None

    while True:
        scroll_result = await store.client.scroll(
            collection_name=store.collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points, next_offset = scroll_result

        for point in points:
            payload = point.payload or {}
            chunk = RetrievedChunk(
                chunk_id=str(payload.get("chunk_id", str(point.id))),
                doc_id=str(payload.get("doc_id", "")),
                doc_title=str(payload.get("doc_title", "")),
                doc_type=DocType(payload.get("doc_type", "regulation")),
                section_path=str(payload.get("section_path", "")),
                text=str(payload.get("chunk_text", "")),  # Use chunk_text field
                score=0.0,
                doc_summary=str(payload.get("doc_summary", "")),
                page_family=str(payload.get("page_family", "")),
                doc_family=str(payload.get("doc_family", "")),
                chunk_type=str(payload.get("chunk_type", "")),
            )
            chunks.append(chunk)

        if next_offset is None:
            break
        offset = next_offset

    logger.info("Fetched %d chunks from Qdrant", len(chunks))
    return chunks


async def main() -> None:
    """Build and save BM25 index."""
    settings = get_settings()
    store = QdrantStore()

    logger.info("Fetching chunks from collection: %s", store.collection_name)
    chunks = await fetch_all_chunks(store)

    if not chunks:
        logger.error("No chunks found in collection")
        return

    logger.info("Building BM25 index...")
    retriever = BM25Retriever(index_dir=Path("data/bm25_index"))
    retriever.build_index(chunks)
    retriever.save()

    logger.info("Testing BM25 search...")
    test_query = "Article 5 of DIFC Employment Law"
    results = retriever.search(test_query, top_k=5)
    logger.info("Test query: %s", test_query)
    logger.info("Top 5 results: %s", [(cid[:16], score) for cid, score in results[:5]])

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
