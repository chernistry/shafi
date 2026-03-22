import asyncio
import os
import random
import logging
import json
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

async def audit_enrichment_quality():
    collection_name = "legal_chunks_private_1792"
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")
    
    client = AsyncQdrantClient(url=url, api_key=api_key)
    
    try:
        # Filter for case-law or order
        # doc_type=case_law OR doc_family=order
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
            limit=50, # Get a batch to sample from
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
            doc_type = payload.get("doc_type", "")
            doc_family = payload.get("doc_family", "")
            
            party_names = payload.get("party_names", [])
            case_numbers = payload.get("case_numbers", [])
            court_names = payload.get("court_names", [])
            
            print(f"\n--- Chunk {i} ({chunk_id}) ---")
            print(f"Title: {doc_title}")
            print(f"Type: {doc_type}, Family: {doc_family}")
            print(f"Party Names: {party_names}")
            print(f"Case Numbers: {case_numbers}")
            print(f"Court Names: {court_names}")
            
            results.append({
                "chunk_id": chunk_id,
                "doc_title": doc_title,
                "doc_type": doc_type,
                "doc_family": doc_family,
                "party_names": party_names,
                "case_numbers": case_numbers,
                "court_names": court_names,
                "has_parties": len(party_names) > 0,
                "has_cases": len(case_numbers) > 0,
                "has_courts": len(court_names) > 0
            })

        # Summary
        total = len(results)
        with_parties = sum(1 for r in results if r["has_parties"])
        with_cases = sum(1 for r in results if r["has_cases"])
        with_courts = sum(1 for r in results if r["has_courts"])
        
        print("\n" + "="*50)
        print("ENRICHMENT QUALITY AUDIT SUMMARY (CASE-LAW)")
        print(f"Total sampled: {total}")
        print(f"With party_names: {with_parties}/{total}")
        print(f"With case_numbers: {with_cases}/{total}")
        print(f"With court_names: {with_courts}/{total}")
        print("="*50)
        
        with open("data/noam_enrichment_quality_audit.json", "w") as f:
            json.dump(results, f, indent=2)

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(audit_enrichment_quality())
