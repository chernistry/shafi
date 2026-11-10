import asyncio
import os
import random
import logging
import re
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import ScrollRequest

# Simple heuristic for garbage text
def is_garbage(text: str) -> bool:
    if not text.strip():
        return True
    # High ratio of non-alphanumeric characters
    non_alnum = len(re.sub(r'[a-zA-Z0-9\s\.\,\;\:\!\?\(\)\[\]\{\}\-\_\"\'\/]', '', text))
    ratio = non_alnum / len(text) if len(text) > 0 else 0
    if ratio > 0.4 and len(text) > 50:
        return True
    # Too many repeated characters
    if any(text.count(c) > len(text) * 0.5 for c in set(text) if not c.isspace()):
        return True
    return False

async def audit_data_quality():
    collection_name = "legal_chunks_private_1792"
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")
    
    client = AsyncQdrantClient(url=url, api_key=api_key)
    
    try:
        # 1. Get total count
        collection_info = await client.get_collection(collection_name)
        total_points = collection_info.points_count
        print(f"Total points in '{collection_name}': {total_points}")
        
        if total_points == 0:
            print("ERROR: Collection is empty.")
            return

        # 2. Get 20 random points
        # Scroll doesn't support random sampling directly, so we'll use random offsets
        # Since points are distributed, we can scroll with different offsets
        num_samples = 20
        samples = []
        
        # We'll do 20 separate scroll calls with limit 1 and a random offset
        # Actually, Qdrant scroll offset is a point ID (string), not an integer.
        # But wait, ScrollRequest allows 'offset' which can be int/uuid.
        # No, 'offset' in scroll is the ID of the first point of the next page.
        
        # Alternative: use 'scroll' with a large limit and pick 20 from there,
        # or use multiple 'scroll' calls if we want more spread.
        # Let's get 100 points and pick 20 random ones.
        
        print(f"Sampling {num_samples} random chunks...")
        
        records, next_offset = await client.scroll(
            collection_name=collection_name,
            limit=500, # Get a decent batch to sample from
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            print("ERROR: No records found.")
            return
            
        random_samples = random.sample(records, min(num_samples, len(records)))
        
        results = []
        for i, point in enumerate(random_samples, 1):
            payload = point.payload or {}
            chunk_text = payload.get("chunk_text", "")
            doc_title = payload.get("doc_title", "")
            doc_id = payload.get("doc_id", "")
            chunk_id = payload.get("chunk_id", "")
            
            garbage = is_garbage(chunk_text)
            empty = not chunk_text.strip()
            
            # Simple relevance check: title in text? (not always true for chunks, but a hint)
            # Better check: is it legal text?
            is_legal = any(word in chunk_text.lower() for word in ["article", "law", "court", "judgment", "regulation", "difc", "uae"])
            
            results.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "chunk_text_preview": chunk_text[:200].replace("\n", " "),
                "is_garbage": garbage,
                "is_empty": empty,
                "is_legal_looking": is_legal,
                "length": len(chunk_text)
            })
            
            print(f"\n--- Chunk {i} ({chunk_id}) ---")
            print(f"Doc Title: {doc_title}")
            print(f"Doc ID: {doc_id}")
            print(f"Length: {len(chunk_text)}")
            print(f"Preview: {chunk_text[:150]}...")
            if garbage: print("!! FLAG: Garbage detected")
            if empty: print("!! FLAG: Empty text detected")
            if not is_legal: print("? NOTE: Might not look like legal text (check preview)")

        # Summary report
        garbage_count = sum(1 for r in results if r["is_garbage"])
        empty_count = sum(1 for r in results if r["is_empty"])
        non_legal_count = sum(1 for r in results if not r["is_legal_looking"])
        
        print("\n" + "="*50)
        print("AUDIT SUMMARY")
        print(f"Total sampled: {len(results)}")
        print(f"Garbage chunks: {garbage_count}")
        print(f"Empty chunks: {empty_count}")
        print(f"Non-legal looking: {non_legal_count}")
        print("="*50)
        
        # Save results to a file for later reference
        with open("data/noam_data_quality_audit.json", "w") as f:
            import json
            json.dump(results, f, indent=2)

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(audit_data_quality())
