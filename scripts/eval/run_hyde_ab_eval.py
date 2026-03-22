#!/usr/bin/env python3
"""Mini-eval for HyDE + citation graph A/B testing on complex warmup questions.

Selects 20-30 complex questions (free_text + multi-hop patterns) from warmup set,
runs them with HyDE enabled, and compares G scores vs baseline.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import httpx

REPO = Path(__file__).resolve().parents[1]
WARMUP_PATH = REPO / "eval_golden_warmup_verified.json"
SERVER_URL = "http://localhost:8002/query"
TIMEOUT = 120.0

# Complex question patterns (free_text + multi-hop)
COMPLEX_TYPES = {"free_text"}
MULTI_HOP_MIN_GOLD = 2  # Questions with 2+ gold pages are multi-hop


def load_complex_questions() -> list[dict]:
    """Load complex questions from warmup set."""
    data = json.loads(WARMUP_PATH.read_text())
    
    complex_questions = []
    for q in data:
        answer_type = q.get("answer_type", "")
        gold_ids = q.get("gold_chunk_ids", [])
        
        # Select free_text questions
        if answer_type in COMPLEX_TYPES:
            complex_questions.append(q)
        # Also include multi-hop questions (2+ gold pages)
        elif len(gold_ids) >= MULTI_HOP_MIN_GOLD:
            complex_questions.append(q)
    
    # Take first 25 complex questions for mini-eval
    return complex_questions[:25]


def run_question(client: httpx.Client, question: dict) -> dict:
    """Run single question through pipeline."""
    qid = question.get("id", "")[:8]
    text = question.get("question", "")
    answer_type = question.get("answer_type", "")
    gold_ids = set(question.get("gold_chunk_ids", []))
    
    t0 = time.monotonic()
    try:
        resp = client.post(
            SERVER_URL,
            json={"question": text, "answer_type": answer_type},
            timeout=TIMEOUT
        )
        resp.raise_for_status()
        
        # Parse SSE response
        used_pages = set()
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                if payload.get("type") == "telemetry":
                    used_pages = set(payload.get("payload", {}).get("used_page_ids", []))
        
        elapsed = time.monotonic() - t0
        
        # Calculate G score (hit@k)
        overlap = len(used_pages & gold_ids)
        g_score = 1.0 if overlap > 0 else 0.0
        
        return {
            "qid": qid,
            "question": text[:100],
            "answer_type": answer_type,
            "g_score": g_score,
            "gold_count": len(gold_ids),
            "used_count": len(used_pages),
            "overlap": overlap,
            "ttft_ms": elapsed * 1000,
        }
    except Exception as e:
        return {
            "qid": qid,
            "error": str(e),
            "g_score": 0.0,
            "ttft_ms": 0,
        }


def run_mini_eval(profile_name: str, output_path: Path) -> dict:
    """Run mini-eval with specified profile."""
    print(f"\n{'='*60}")
    print(f"Mini-eval: {profile_name}")
    print(f"{'='*60}")
    
    questions = load_complex_questions()
    print(f"Selected {len(questions)} complex questions")
    
    results = []
    total_g = 0.0
    total_ttft = 0.0
    
    with httpx.Client(timeout=TIMEOUT) as client:
        for i, q in enumerate(questions, 1):
            result = run_question(client, q)
            results.append(result)
            
            total_g += result.get("g_score", 0)
            total_ttft += result.get("ttft_ms", 0)
            
            status = "✓" if result.get("g_score", 0) > 0 else "✗"
            print(f"  [{i}/{len(questions)}] {status} {result['qid']}: G={result.get('g_score', 0):.1f} TTFT={result.get('ttft_ms', 0):.0f}ms")
    
    # Calculate averages
    avg_g = total_g / len(results) if results else 0
    avg_ttft = total_ttft / len(results) if results else 0
    
    summary = {
        "profile": profile_name,
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(results),
        "avg_g": avg_g,
        "avg_ttft_ms": avg_ttft,
        "results": results,
    }
    
    # Save results
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output_path}")
    print(f"Average G: {avg_g:.3f} ({avg_g*100:.1f}%)")
    print(f"Average TTFT: {avg_ttft:.0f}ms")
    
    return summary


def main():
    """Run A/B test: baseline vs HyDE vs HyDE+graph."""
    print("AMIR Mini-Eval: HyDE + Citation Graph A/B Test")
    print("=" * 60)
    
    # Note: User needs to set profile in .env.local and restart server
    print("\nInstructions:")
    print("1. Copy profile to .env.local: cp profiles/private_v9_hyde.env .env.local")
    print("2. Restart server: make restart (or equivalent)")
    print("3. Run this script: uv run python scripts/run_hyde_ab_eval.py")
    print("\nProfiles to test:")
    print("  - private_v9_eqa.env (baseline)")
    print("  - private_v9_hyde.env (HyDE only)")
    print("  - private_v9_hyde_graph.env (HyDE + citation graph)")
    
    # Auto-run if server is available
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get("http://localhost:8002/health")
            if resp.status_code == 200:
                print("\n✓ Server detected at localhost:8002")
                print("Starting mini-eval with current profile...")
                
                output_dir = REPO / ".sdd" / "evals" / "amir_hyde_ab"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"mini_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                run_mini_eval("current", output_path)
            else:
                print("\n✗ Server not responding")
    except Exception as e:
        print(f"\n✗ Cannot connect to server: {e}")
        print("Please start the server and try again.")


if __name__ == "__main__":
    main()
