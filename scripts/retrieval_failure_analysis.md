# Retrieval Failure Analysis — Warmup 100

**Date**: 2026-03-20
**Eval**: `data/eval/warmup_raw_20260320_133338.json`
**Golden**: `.sdd/golden/synthetic-ai-generated/golden_labels_v2.json`

## Executive Summary

| Metric | Value | Description |
|--------|-------|-------------|
| Doc-level recall | 91/100 (91%) | System finds the right document |
| Page in retrieved set | 63/100 (63%) | Gold page appears in retrieval candidates |
| Page in LLM context | 49/100 (49%) | Gold page reaches the LLM |
| Page used (G>0) | 40/100 (40%) | Gold page in final grounding |

**The 37 page-miss cases break into:**
- 9 wrong document entirely (4 are unsupported trap questions)
- 28 right document, wrong page selected

## Gold Page Position Analysis

For the 28 right-doc-wrong-page cases, gold pages are:

| Position | Count | Description |
|----------|-------|-------------|
| Before retrieved range | 52 | Gold page number < min retrieved page from same doc |
| Within range but skipped | 11 | Gold page between min/max retrieved but not selected |
| Beyond range | 1 | Gold page > max retrieved page |

**Key insight**: 52 of 64 missed gold pages are at LOWER page numbers than what we retrieve.
The embedding search skips early pages (articles, enactment notices) in favor of later content.

## Failure Categories

### Category 1: Article-Level Provision Miss (17 questions)
Single-doc queries asking about a specific article (e.g., "Article 14(2)(b)").
Gold page is typically 1-3 pages below the retrieval window.

Examples:
- e59a0dc4: Employment Law Art 14(1), gold=page6, retrieved=[7-32]
- be09fbfe: Employment Law Art 14(2)(l), gold=page6, retrieved=[8-41]
- 0149374f: Employment Law Art 11(2)(b), gold=page5, retrieved=[7-32]
- 146567e3: GP Law Art 14(2)(b), gold=page10, retrieved=[6-23] (within range but skipped)

**Root cause**: Embedding similarity ranks general content pages higher than specific article pages.
BM25 component could help (article numbers are keywords) but current sparse prefetch is only 60.

**Fix**: Increase prefetch (60→120) + enable cross-ref boosts.

### Category 2: Multi-Document Recall Miss (7 questions)
Questions requiring 6-20+ documents (e.g., "Which laws mention the Ruler of Dubai?").
System retrieves 43-63 pages but only 1-17 of 18-20 gold docs.

Examples:
- fcabd6aa: 22 gold docs, 1 hit — "Which DIFC Laws were amended by Law No. 2 of 2022?"
- 89fd4fbc: 18 gold docs, 17 hit — "Which laws mention interpretative provisions?"
- 4aa0f4e2: 20 gold docs, 9 hit — "Which laws mention Ruler of Dubai?"

**Root cause**: Prefetch limit caps retrieval at ~60 chunks → ~15-20 unique docs max.
These questions inherently need broader sweep.

**Fix**: Increase prefetch to 120/120.

### Category 3: Page-1 / Title Page Miss (6 questions)
Compare/temporal queries where gold is page 1 (title/parties/judges).
System retrieves content pages (2+) but not the title page.

Examples:
- bd8d0bef: gold=[1], retrieved=[2-40] — Employment Law compare
- bb67fc19: gold=[1], retrieved=[2-40] — IP Law compare
- f378457d: gold=[1], retrieved=[2-49] — Data Protection Law

**Root cause**: Page 1 (enactment notice, case caption) has low embedding similarity
to the query. The BM25 component should match keywords but isn't surfacing page 1.

**Fix**: Enable cross-ref boosts (would prioritize docs matching law references).

### Category 4: Wrong Document Entirely (5 real misses)
Questions where no gold document appears in retrieval at all.
(Excluding 4 unsupported trap questions that correctly return empty.)

- 6351cfe2: "Which laws administered by Registrar enacted 2004-2010?" (needs 4 docs)
- d5bc7441: "Leasing Law vs Real Property Law Amendment Law" (needs 2 docs)
- 82664b58: "Full title of enacted law?" (ambiguous query)
- 4ced374a: "Enactment notice date?" (specific doc not found)
- be535a44: "Latest DIFC Law number amending Civil/Commercial Laws?" (needs amendment doc)

**Root cause**: Query terms don't match any document in embedding space, OR
the document reference is ambiguous/indirect.

## Recommended Fixes (Priority Order)

### Fix 1: Increase Prefetch (HIGHEST ROI)
- Change: `QDRANT_PREFETCH_DENSE: 60 → 120`, `QDRANT_PREFETCH_SPARSE: 60 → 120`
- Expected: Recover 5-10 article-level page misses + broader multi-doc coverage
- Risk: More noise in candidate pool, slightly slower retrieval
- Flag: Settings change, instant rollback

### Fix 2: Enable Cross-Ref Boosts
- Change: `PIPELINE_ENABLE_CROSS_REF_BOOSTS: false → true`
- Expected: Boost chunks containing referenced article numbers, recover 3-5 provisions
- Risk: Low — only adds +0.08 to matching chunks within existing pool
- Flag: `PIPELINE_ENABLE_CROSS_REF_BOOSTS`

### Fix 3: Enable Shadow Search
- Change: `PIPELINE_ENABLE_SHADOW_SEARCH_TEXT: false → true`
- Expected: Query shadow collection with richer text, find alternative matches
- Risk: Medium — needs shadow collection to exist and be populated
- Flag: `PIPELINE_ENABLE_SHADOW_SEARCH_TEXT`
