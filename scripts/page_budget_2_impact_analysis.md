# Page Budget 1→2 Impact Analysis

**Author**: TAMAR
**Date**: 2026-03-20
**Purpose**: Identify which specific warmup questions benefit from page_budget=2, to help OREV validate the #1 priority change.

## Background

From KNOWLEDGE_BASE (EYAL finding):
- page_budget=1: hit@1 = 65.6%
- page_budget=2: hit@2 = 82.8% (**+17.2pp** — largest single improvement)
- page_budget=3: hit@3 = 89.1% (+6.3pp marginal)

47/93 questions have multi-page gold answers.

## TAMAR's Page-Budget-2 Question Analysis

From gold_chunk_ids in eval_golden_warmup_verified.json + full_answer_verification.json:

### Questions with EXACTLY 2 Gold Pages (most critical for page_budget=2)

These 47 questions need BOTH their gold pages in context. page_budget=1 drops one.

| Category | Count | Pattern |
|----------|-------|---------|
| Dual-case booleans (cross-case comparison) | ~12 | 2 pages from 2 different case docs |
| Same-law 2-page questions (two versions of same law) | ~15 | 2 pages from doc v1 + doc v2 |
| Cross-law 2-page comparisons | ~10 | 1 page from each of 2 different laws |
| Same-doc multi-page (need page N and page M) | ~10 | 2 pages from same doc, different page numbers |

### High-Impact Questions for page_budget=2

These specific questions have G < 0.9 and EXACTLY 2 gold pages that should both be in context:

**Cross-case booleans (answered correctly but low G):**
1. `9f9fb4b9` G=0.537 → expected G~0.9 (CA 004/2025 + SCT 295/2025)
2. `bfa089d5` G=0.5 → expected G~0.9 (CFI 010/2024 + DEC 001/2025)
3. `52a35cfa` G=0.0 → expected G~0.9 (CA 004/2025 + ARB 034/2025) [grounding miss]
4. `54d56331` G=0.349 → expected G~0.9 (TCD 001/2024 + CFI 016/2025)
5. `2d436eb3` G=0.349 → expected G~0.9 (CA 005/2025 + CFI 067/2025)

**Same-law 2-version questions (two law versions, both cited):**
Many Operating Law, Employment Law, RP Law questions have 2 gold pages from 2 document versions of the same law. page_budget=2 ensures both versions are cited for full G.

**Date comparison questions (need date from 2 case docs):**
6. `b9dc2dae` G=0.5 → expected G~0.9 (CFI 016/2025 vs ENF 269/2023 date)
7. `d9d27c9c` G=0.537 → expected G~0.9 (ARB 034/2025 vs SCT 295/2025 date)
8. `3dc92e33` G=0.537 → expected G~0.7 (CFI 010/2024 vs SCT 169/2025 date — WRONG answer)
9. `fbe661b9` G=0.537 → expected G~0.9 (CA 004/2025 vs SCT 295/2025 date)

### Summary of Expected page_budget=2 Gains

| Question bucket | Count | Avg G current | Expected avg G with budget=2 |
|---|---|---|---|
| Cross-case booleans (G>0 already) | 5 | ~0.42 | ~0.85 |
| Cross-case booleans (G=0, correct) | 7 | 0.0 | ~0.7 |
| Date comparisons (correct answer) | 5 | ~0.4 | ~0.8 |
| Single-doc multi-page (same law versions) | ~15 | ~0.5 | ~0.8 |
| Hard multi-doc enumeration | ~10 | ~0.1 | ~0.2 (marginal) |

**Conservative total G gain from page_budget=2 alone**: +8-12 raw G points on 100 questions → avg G from 46.8% to 55-60%.

## Validation Test Slice for OREV

After implementing page_budget=2, validate by running ONLY these 15 questions:
```
9f9fb4b9, bfa089d5, 52a35cfa, 54d56331, 2d436eb3, 737940cf, fba6e86a,
1e1121d0, 3c19ecbe, b9dc2dae, d9d27c9c, 3dc92e33, fbe661b9, d4157e6a, 0f6e75bd
```

**Expected**: All 15 should show G > 0 after fix (currently 8 are G=0).
**Success threshold**: ≥10 of 15 now have G > 0.5.
**Regression check**: Questions with existing G=1.0 should NOT drop below 0.8.

## Interaction with Dual-Case Fix

These two fixes are COMPLEMENTARY and ADDITIVE:
- page_budget=2: gets both gold pages into context pool
- Dual-case context guarantee: ensures both case pages are SELECTED into context (not just retrieved)

OREV should implement BOTH for maximum G uplift. Expected combined gain: +12-15 pp G.

## One Critical Warning

`8e3b4683` (CA 005/2025 + TCD 001/2024 same judge?):
- page_budget=2 will help get both case judge pages in context
- BUT the pipeline answer is WRONG (says "No", gold=True)
- Getting correct pages into context is necessary but not sufficient for this question
- The answer logic must also detect the judge overlap
- OREV: please verify this question specifically after implementing both fixes
