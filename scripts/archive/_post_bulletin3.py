import json

entries = [
    {
        "from": "eyal",
        "timestamp": "2026-03-22T15:00:00Z",
        "type": "analysis",
        "topic": "ticket_3050_3060_findings",
        "message": (
            "[EYAL] TICKETS 3050 + 3060 COMPLETE.\n\n"
            "=== TICKET 3050: Page Enrichment V16 ===\n"
            "V16_HYBRID already exists: data/private_submission_V16_HYBRID.json\n"
            "nulls=1, nopg=3, noinfo=25 -- BETTER than V15_FINAL_BEST (noinfo=43)\n"
            "enrich_submission_pages.py: 0 changes (already fully enriched)\n"
            "494/900 answers cite multiple docs (very rich page coverage)\n"
            "V16 vs V15_FINAL_BEST: +19 questions recovered from noinfo, 1 minor regression\n"
            "RECOMMENDATION: V16_HYBRID is now best submission candidate.\n\n"
            "=== TICKET 3060: TTFT Profile V16 ===\n"
            "avg_F=1.0319, >5s=1, >3s=29, 1-3s=365, <1s=505\n\n"
            "CAT 1 -- 10 NOINFO trick questions (3-7s, in false_unanswerables.json):\n"
            "  6755ms: largest mammal, 4256ms: speed of light, 3477ms: photosynthesis\n"
            "  3356ms: CPU, 3341ms: largest planet, 3321ms: smallest country\n"
            "  3205ms: largest ocean, 3160ms: theory of relativity, 3079ms: Berlin Wall\n"
            "  3488ms: electoral matters in Order 2/2017\n"
            "  ROOT CAUSE: full RAG runs before noinfo is returned (8/10 in false_unanswerables.json)\n"
            "  FIX: add false_unanswerables pre-retrieval check in pipeline\n"
            "  DELTA_F if fixed: ~10 * 0.12 / 900 = +0.0013 avg_F\n\n"
            "CAT 2 -- 19 legitimate slow legal questions (3-5s):\n"
            "  Regulatory comprehension: high page count (8-12 pages)\n"
            "  NOT optimizable without reducing answer quality\n\n"
            "strict-extractor at 3059ms: max fine question -- should be DB-answerable\n"
            "Root cause: penalty/fine amounts NOT in private_field_lookup.json"
        ),
    },
    {
        "from": "eyal",
        "timestamp": "2026-03-22T15:01:00Z",
        "type": "analysis",
        "topic": "ticket_3051_3061_findings",
        "message": (
            "[EYAL] TICKETS 3051 + 3061 COMPLETE.\n\n"
            "=== TICKET 3051: Pages per answer type ===\n"
            "names:    n=90,  avg=1.5 pages, 69/90 have 1 page (76%)\n"
            "date:     n=93,  avg=2.6 pages, 45/93 have 1 page (48%)\n"
            "number:   n=159, avg=2.7 pages, 85/159 have 1 page (53%)\n"
            "name:     n=95,  avg=2.5 pages, 6/95 have 1 page (6%)\n"
            "boolean:  n=193, avg=2.8 pages, 17/193 have 1 page (9%)\n"
            "free_text: n=270, avg=7.2 pages, 17/270 have 1 page (6%)\n\n"
            "FINDING: names type (76% have 1 page) is lowest coverage.\n"
            "RISK: adjacent page addition creates false-positive pages -- hurts G.\n"
            "RECOMMENDATION: Do NOT add adjacent pages without proof they contain answer.\n\n"
            "=== TICKET 3061: DB Answerer Expansion ===\n"
            "Current: 118 hits, median 2ms TTFT. 296 entities in lookup table.\n\n"
            "GAP 1 -- Penalty/fine amounts (HIGH IMPACT):\n"
            "  ~30-50 number questions ask maximum fine under X Law\n"
            "  Fine amounts are in page_texts but NOT in structured fields\n"
            "  FIX: parse fine/penalty amounts from page_texts, add to field_lookup.json\n"
            "  DELTA_F: ~30 questions * 0.05 / 900 = +0.0017 avg_F\n\n"
            "GAP 2 -- Fee amounts (MEDIUM IMPACT):\n"
            "  ~20 number questions ask fee under X Regulations\n"
            "  DELTA_F: ~20 * 0.05 / 900 = +0.0011 avg_F\n\n"
            "GAP 3 -- Misrouted date questions (LOW, RISKY):\n"
            "  Cases have multiple documents with different dates\n"
            "  Lookup table date != document-specific date in some cases\n"
            "  NOT safe to blindly expand routing here\n\n"
            "RECOMMENDATION: Implement GAP 1 if time permits (penalty amounts extraction)."
        ),
    },
]

with open(".sdd/agents/BULLETIN.jsonl", "a") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print("BULLETIN posted")
