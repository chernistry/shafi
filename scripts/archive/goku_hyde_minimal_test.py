#!/usr/bin/env python3
"""Minimal HyDE A/B test — 10 complex questions, baseline vs HyDE."""
import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def query_api(question: str, endpoint: str = "http://localhost:8000/query") -> dict:
    """Query the API and return response (handles SSE streaming)."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json={"question": question}, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            # API returns SSE stream, collect all events
            final_data = {}
            async for line in resp.content:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    if data_str and data_str != '[DONE]':
                        try:
                            final_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            pass
            return final_data

async def main():
    golden_path = Path("dataset/public_dataset.json")
    if not golden_path.exists():
        print(f"ERROR: {golden_path} not found")
        return 1

    with golden_path.open() as f:
        golden = json.load(f)

    # Select 10 complex free_text questions
    complex_qs = [item for item in golden if item.get("answer_type") == "free_text"][:10]
    
    print(f"Testing {len(complex_qs)} complex questions")
    print(f"Endpoint: http://localhost:8000/query")
    
    results = []
    for i, item in enumerate(complex_qs, 1):
        q = item["question"]
        print(f"\n[{i}/{len(complex_qs)}] {q[:80]}...")
        try:
            resp = await query_api(q)
            results.append({
                "question": q,
                "answer": resp.get("answer"),
                "used_pages": len(resp.get("used_page_ids", [])),
                "ttft_ms": resp.get("ttft_ms", 0),
            })
            print(f"  → Answer: {resp.get('answer', '')[:60]}...")
            print(f"  → Pages: {len(resp.get('used_page_ids', []))}, TTFT: {resp.get('ttft_ms', 0):.0f}ms")
        except Exception as exc:
            print(f"  → ERROR: {exc}")
            results.append({"question": q, "error": str(exc)})
    
    # Save results
    output_dir = Path(".sdd/evals/amir_hyde_minimal")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    successful = [r for r in results if "error" not in r]
    if successful:
        avg_ttft = sum(r.get("ttft_ms", 0) for r in successful) / len(successful)
    else:
        avg_ttft = 0
    print(f"\n=== SUMMARY ===")
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Avg TTFT: {avg_ttft:.0f}ms")
    print(f"Results: {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
