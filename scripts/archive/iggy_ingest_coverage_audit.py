import asyncio
import os
import json
from collections import Counter
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

async def audit_ingest_coverage():
    collection_name = "legal_chunks_private_1792"
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")
    
    client = AsyncQdrantClient(url=url, api_key=api_key)
    
    try:
        print(f"Counting chunks for all documents in '{collection_name}'...")
        
        doc_counts = Counter()
        next_offset = None
        
        while True:
            records, next_offset = await client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=["doc_id"],
                with_vectors=False,
                offset=next_offset
            )
            
            for point in records:
                doc_id = point.payload.get("doc_id")
                if doc_id:
                    doc_counts[doc_id] += 1
            
            if not next_offset:
                break
        
        print(f"Total unique doc_ids: {len(doc_counts)}")
        
        under_threshold = {doc_id: count for doc_id, count in doc_counts.items() if count < 3}
        under_threshold_titles = {}
        
        if under_threshold:
            print(f"Fetching titles for {len(under_threshold)} documents...")
            for doc_id in under_threshold:
                records, _ = await client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
                    limit=1,
                    with_payload=["doc_title"]
                )
                if records:
                    under_threshold_titles[doc_id] = records[0].payload.get("doc_title", "Unknown")
        
        print("\n" + "="*50)
        print("INGEST COVERAGE AUDIT SUMMARY")
        print(f"Total documents: {len(doc_counts)}")
        print(f"Documents with <3 chunks: {len(under_threshold)}")
        print("="*50)
        
        if under_threshold:
            print("\nDocs with low chunk counts:")
            for doc_id, count in sorted(under_threshold.items(), key=lambda x: x[1]):
                title = under_threshold_titles.get(doc_id, "Unknown")
                print(f"- {doc_id} ({title}): {count} chunks")
        
        with open("data/noam_ingest_coverage_audit.json", "w") as f:
            json.dump({
                "total_docs": len(doc_counts),
                "under_threshold_count": len(under_threshold),
                "under_threshold_docs": {
                    doc_id: {"count": count, "title": under_threshold_titles.get(doc_id)}
                    for doc_id, count in under_threshold.items()
                }
            }, f, indent=2)

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(audit_ingest_coverage())
