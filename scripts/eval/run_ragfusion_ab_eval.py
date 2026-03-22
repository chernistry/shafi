#!/usr/bin/env python3
"""Mini-eval for RAG fusion isolated A/B test.

Tests PIPELINE_ENABLE_RAG_FUSION=true vs baseline (v9_eqa).
Runs on 25 complex questions (free_text + multi-hop patterns).
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_challenge.eval.harness import run_eval


async def main():
    """Run RAG fusion mini-eval."""
    golden_path = Path("dataset/public_dataset.json")
    if not golden_path.exists():
        print(f"ERROR: {golden_path} not found")
        return 1

    # Load golden dataset
    with golden_path.open() as f:
        golden = json.load(f)

    # Filter to complex questions (free_text + multi-hop patterns)
    # Multi-hop indicators: "and", "or", "both", "either", "also", "additionally"
    complex_questions = []
    for item in golden:
        if item.get("answer_type") == "free_text":
            complex_questions.append(item)
        elif any(
            keyword in item["question"].lower()
            for keyword in ["and", "or", "both", "either", "also", "additionally", "as well as"]
        ):
            complex_questions.append(item)

    # Limit to 25 questions
    test_set = complex_questions[:25]
    print(f"Selected {len(test_set)} complex questions for RAG fusion A/B test")

    # Create temp golden file
    temp_golden = Path(".sdd/evals/amir_ragfusion_ab/mini_eval_golden.json")
    temp_golden.parent.mkdir(parents=True, exist_ok=True)
    with temp_golden.open("w") as f:
        json.dump(test_set, f, indent=2)

    # Run eval
    output_path = Path(".sdd/evals/amir_ragfusion_ab/mini_eval_ragfusion.json")
    print(f"\nRunning RAG fusion eval (ENV_FILE=profiles/private_v9_ragfusion.env)...")
    print(f"Output: {output_path}")

    result = await run_eval(
        golden_path=str(temp_golden),
        endpoint="http://localhost:8000/query",
        concurrency=4,
        output_path=str(output_path),
        emit_cases=True,
    )

    print(f"\n=== RAG FUSION RESULTS ===")
    print(f"G (local): {result.get('grounding_score', 0):.3f}")
    print(f"Det: {result.get('deterministic_correct', 0)}/{result.get('deterministic_total', 0)}")
    print(f"Avg TTFT: {result.get('avg_ttft_ms', 0):.0f}ms")
    print(f"\nResults saved to: {output_path}")
    print(f"\nNEXT: Restart server with ENV_FILE=profiles/private_v9_eqa.env and run baseline comparison")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
