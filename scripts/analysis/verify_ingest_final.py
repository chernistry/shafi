import httpx
import json
import os

async def verify_ingest():
    url = "http://localhost:6333/collections/legal_chunks_private_1792/points/scroll"
    
    # Get 300 hashes
    private_dir = "data/private"
    expected_hashes = set()
    for f in os.listdir(private_dir):
        if f.endswith(".pdf"):
            expected_hashes.add(f[:16])
    
    print(f"Expected private hashes: {len(expected_hashes)}")
    
    found_hashes = set()
    next_offset = None
    
    async with httpx.AsyncClient() as client:
        while True:
            payload = {
                "limit": 1000,
                "with_payload": ["doc_id"]
            }
            if next_offset:
                payload["offset"] = next_offset
                
            resp = await client.post(url, json=payload)
            data = resp.json()
            
            points = data.get("result", {}).get("points", [])
            for p in points:
                doc_id = p.get("payload", {}).get("doc_id")
                if doc_id:
                    found_hashes.add(doc_id[:16])
            
            next_offset = data.get("result", {}).get("next_page_offset")
            if not next_offset:
                break
                
    print(f"Total unique doc_ids in Qdrant: {len(found_hashes)}")
    
    missing = expected_hashes - found_hashes
    print(f"Missing private hashes: {len(missing)}")
    if missing:
        print(f"Sample missing: {list(missing)[:10]}")
    else:
        print("SUCCESS: All 300 private documents found in Qdrant.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(verify_ingest())
