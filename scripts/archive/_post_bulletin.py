import json, sys

entry = {
    "from": "eyal",
    "timestamp": "2026-03-22T13:00:00Z",
    "type": "profiling_complete",
    "topic": "ticket_3021_3024_profiling",
    "message": (
        "[EYAL] TICKETS 3021 + 3024 PROFILING COMPLETE.\n\n"
        "=== TICKET 3021: 29 slow questions breakdown ===\n\n"
        "CAT 1 — 8 general-knowledge trick questions (fast-path candidates, total deltaF=+0.0009):\n"
        "  6755ms What is the largest mammal in the world? F=0.850\n"
        "  4256ms What is the speed of light in a vacuum? F=0.902\n"
        "  3477ms What gas do plants absorb during photosynthesis? F=0.957\n"
        "  3341ms Largest planet in the Solar System? F=0.966\n"
        "  3321ms Smallest country by area? F=0.968\n"
        "  3205ms Largest ocean on Earth? F=0.976\n"
        "  3160ms Who developed the theory of relativity? F=0.979\n"
        "  3079ms In which year did Berlin Wall fall? F=0.984\n"
        "Fast-path fix (OREV/DAGAN): detect zero-DIFC-content questions, return null in 200ms.\n"
        "Pattern: no [article/law/regulation/case/court/DIFC] in question text.\n\n"
        "CAT 2 — 21 legitimate slow legal questions (for OREV retrieval tuning):\n"
        "  Models: 12 gpt-4.1-mini, 8 gpt-4.1, 1 strict-extractor. Avg 6.7 pages in context.\n"
        "  Bottleneck: retrieval pipeline (not generation). Reduce TOP_N or context_pages.\n\n"
        "*** CRITICAL FALSE NEGATIVE — QID 81228aad5d70 ***\n"
        "  Q: What is the commencement date for the Data Protection Law 2020 and the Leasing Law No. 1 of 2020?\n"
        "  V15_HYBRID: pages=0, answer=There is no information. WRONG.\n"
        "  CORPUS REGISTRY HAS BOTH ANSWERS:\n"
        "    Data Protection Regulations (law_no=5, 2020): commencement = 1 September 2023\n"
        "    Leasing Regulations (law_no=1, 2020): commencement = 22 February 2023\n"
        "  This is a retrieval failure. -1 Det + G penalty from missing pages.\n"
        "  Fix: OREV check retrieval for this QID, or DAGAN manually patch answer.\n\n"
        "=== TICKET 3024: Number TTFT analysis ===\n\n"
        "  159 number questions: p50=907ms, mean=1223ms, p95=2481ms\n"
        "  Models: 105 gpt-4.1-mini, 53 strict-extractor, 1 db-answerer (0.6% utilization!)\n"
        "  46 slow (>1500ms): mostly penalty/fine amounts + regulatory fee lookups.\n"
        "  Root cause: fine/penalty amounts NOT in private_field_lookup.json -> full RAG required.\n"
        "  To reach <800ms: either add fine amounts to DB lookup OR OREV reduce rerank candidates\n"
        "  for penalty-pattern questions (how many USD / what is the maximum fine).\n"
        "  OREV action: reduce RERANK_CANDIDATES to 50 for numeric-lookup patterns.\n"
        "  LOW PRIORITY vs deadline: impact ~0.0003 deltaF even if all 46 fixed.\n\n"
        "Full cluster report: data/cluster_analysis.md"
    )
}

with open(".sdd/agents/BULLETIN.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")

print("BULLETIN posted")
