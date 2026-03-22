#!/usr/bin/env python3
"""Direct retrieval test: HyDE vs baseline on 10 complex questions."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_retrieval():
    from shafi.config.settings import get_settings
    from shafi.core.retriever import HybridRetriever
    from shafi.core.qdrant import QdrantStore
    
    # Load test questions
    golden_path = Path("dataset/public_dataset.json")
    with golden_path.open() as f:
        golden = json.load(f)
    
    complex_qs = [item for item in golden if item.get("answer_type") == "free_text"][:10]
    
    # Test baseline (HyDE disabled)
    print("=== BASELINE (HyDE disabled) ===")
    settings_baseline = get_settings()
    settings_baseline.pipeline.enable_hyde = False
    
    store = QdrantStore(settings_baseline)
    retriever = HybridRetriever(store=store, settings=settings_baseline)
    
    baseline_results = []
    for i, item in enumerate(complex_qs, 1):
        q = item["question"]
        print(f"[{i}/10] {q[:60]}...")
        try:
            results = await retriever.retrieve(query=q, limit=20)
            baseline_results.append({
                "question": q,
                "num_results": len(results),
                "top_scores": [r.score for r in results[:5]] if results else [],
            })
            print(f"  → Retrieved: {len(results)} chunks")
        except Exception as exc:
            print(f"  → ERROR: {exc}")
            baseline_results.append({"question": q, "error": str(exc)})
    
    # Test HyDE (enabled)
    print("\n=== HYDE (enabled) ===")
    settings_hyde = get_settings()
    settings_hyde.pipeline.enable_hyde = True
    
    store_hyde = QdrantStore(settings_hyde)
    retriever_hyde = HybridRetriever(store=store_hyde, settings=settings_hyde)
    
    hyde_results = []
    for i, item in enumerate(complex_qs, 1):
        q = item["question"]
        print(f"[{i}/10] {q[:60]}...")
        try:
            results = await retriever_hyde.retrieve(query=q, limit=20)
            hyde_results.append({
                "question": q,
                "num_results": len(results),
                "top_scores": [r.score for r in results[:5]] if results else [],
            })
            print(f"  → Retrieved: {len(results)} chunks")
        except Exception as exc:
            print(f"  → ERROR: {exc}")
            hyde_results.append({"question": q, "error": str(exc)})
    
    # Save results
    output_dir = Path(".sdd/evals/amir_hyde_direct")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with (output_dir / "baseline.json").open("w") as f:
        json.dump(baseline_results, f, indent=2)
    with (output_dir / "hyde.json").open("w") as f:
        json.dump(hyde_results, f, indent=2)
    
    print(f"\n=== SUMMARY ===")
    print(f"Baseline successful: {len([r for r in baseline_results if 'error' not in r])}/10")
    print(f"HyDE successful: {len([r for r in hyde_results if 'error' not in r])}/10")
    print(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(test_retrieval()))
