import asyncio
import os
from qdrant_client import AsyncQdrantClient

async def verify_fixes():
    client = AsyncQdrantClient(url="http://localhost:6333")
    
    # 1. Check SCT party names for "fbcf7b546efdab6e21f2023d43ab8edd654c91d75e9e5c7d8bb134a662711699"
    # Title was "Numair v Naufil [2024] DIFC SCT 391"
    records, _ = await client.scroll(
        collection_name="legal_chunks_private_1792",
        scroll_filter= {
            "must": [{"key": "doc_id", "match": {"value": "fbcf7b546efdab6e21f2023d43ab8edd654c91d75e9e5c7d8bb134a662711699"}}]
        },
        limit=1,
        with_payload=True
    )
    if records:
        p = records[0].payload
        print(f"Doc: {p.get('doc_title')}")
        print(f"Parties: {p.get('party_names')}")
    
    # 2. Check Enactment Notice title for "96853cbb2873718b7613b42de18dcf417aac607e7fd3e143cab9db7f40622263"
    records, _ = await client.scroll(
        collection_name="legal_chunks_private_1792",
        scroll_filter= {
            "must": [{"key": "doc_id", "match": {"value": "96853cbb2873718b7613b42de18dcf417aac607e7fd3e143cab9db7f40622263"}}]
        },
        limit=1,
        with_payload=True
    )
    if records:
        p = records[0].payload
        print(f"Doc: {p.get('doc_title')}")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(verify_fixes())
