import asyncio
import os
import random
import json
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

async def audit_shadow_search():
    collection_name = "legal_chunks_private_1792"
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")
    
    client = AsyncQdrantClient(url=url, api_key=api_key)
    
    try:
        query_filter = Filter(
            should=[
                FieldCondition(key="doc_type", match=MatchValue(value="case_law")),
                FieldCondition(key="doc_family", match=MatchValue(value="order"))
            ]
        )
        
        print(f"Sampling 10 case-law chunks from '{collection_name}'...")
        
        records, next_offset = await client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=50,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            print("ERROR: No case-law records found.")
            return
            
        random_samples = random.sample(records, min(10, len(records)))
        
        results = []
        for i, point in enumerate(random_samples, 1):
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id", "")
            doc_title = payload.get("doc_title", "")
            shadow_text = payload.get("shadow_search_text", "")
            
            # Check content
            is_copy = shadow_text.strip() == doc_title.strip()
            length_diff = len(shadow_text) - len(doc_title)
            
            print(f"\n--- Chunk {i} ({chunk_id}) ---")
            print(f"Title: {doc_title}")
            print(f"Shadow: {shadow_text[:150]}...")
            if is_copy: print("!! FLAG: Shadow text is identical to doc_title")
            else: print(f"Info: Shadow text is unique (+{length_diff} chars)")
            
            results.append({
                "chunk_id": chunk_id,
                "doc_title": doc_title,
                "shadow_search_text": shadow_text,
                "is_copy": is_copy,
                "length_diff": length_diff
            })

        # Summary
        total = len(results)
        copies = sum(1 for r in results if r["is_copy"])
        
        print("\n" + "="*50)
        print("SHADOW SEARCH QUALITY AUDIT SUMMARY")
        print(f"Total sampled: {total}")
        print(f"Shadow text is identical to doc_title: {copies}/{total}")
        print("="*50)
        
        with open("data/noam_shadow_search_audit.json", "w") as f:
            json.dump(results, f, indent=2)

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(audit_shadow_search())
