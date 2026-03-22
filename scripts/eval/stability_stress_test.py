import asyncio
import json
import time
import httpx
import statistics

async def send_query(client, question_id, question, answer_type):
    url = "http://localhost:8000/query"
    payload = {
        "request_id": f"stress-{question_id}",
        "question_id": question_id,
        "question": question,
        "answer_type": answer_type
    }
    
    t0 = time.perf_counter()
    ttft = None
    try:
        async with client.stream("POST", url, json=payload, timeout=60.0) as response:
            if response.status_code != 200:
                return {"id": question_id, "error": f"HTTP {response.status_code}", "ttft": None}
            
            async for line in response.aiter_lines():
                if line.startswith("data: ") and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                # Consume the whole stream
                pass
            
            return {"id": question_id, "error": None, "ttft": ttft}
    except Exception as e:
        return {"id": question_id, "error": str(e), "ttft": None}

async def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "concurrent"
    
    with open("eval_golden_warmup.json") as f:
        questions = json.load(f)[:50]
    
    async with httpx.AsyncClient() as client:
        if mode == "sequential":
            print(f"Sending {len(questions)} sequential requests...")
            results = []
            for q in questions:
                res = await send_query(client, q["id"], q["question"], q["answer_type"])
                results.append(res)
                if res["error"]:
                    print(f"  - {q['id']}: {res['error']}")
        else:
            tasks = [
                send_query(client, q["id"], q["question"], q["answer_type"])
                for q in questions
            ]
            print(f"Sending {len(tasks)} concurrent requests...")
            results = await asyncio.gather(*tasks)
    
    errors = [r for r in results if r["error"]]
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    
    print(f"\nResults:")
    print(f"Total: {len(results)}")
    print(f"Errors: {len(errors)}")
    for e in errors[:5]:
        print(f"  - {e['id']}: {e['error']}")
    
    if ttfts:
        print(f"Avg TTFT: {statistics.mean(ttfts):.2f} ms")
        print(f"Min TTFT: {min(ttfts):.2f} ms")
        print(f"Max TTFT: {max(ttfts):.2f} ms")
        print(f"Stdev TTFT: {statistics.stdev(ttfts):.2f} ms" if len(ttfts) > 1 else "N/A")
    else:
        print("No TTFT data (all requests failed or no tokens received)")

if __name__ == "__main__":
    asyncio.run(main())
