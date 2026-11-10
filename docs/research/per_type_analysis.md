# R007: Per-Question-Type Performance Analysis

**Date**: 2026-03-22
**Source**: `data/submit/tzur_labs_primary.json` (900 private questions)

---

## Question Type Distribution

| Type | Count | % of Total | Nulls | No-info | No-pages | >5s | 3-5s |
|------|-------|-----------|-------|---------|----------|-----|------|
| free_text | 270 | 30.0% | 0 | 26 | 26 | 1 | 22 |
| boolean | 193 | 21.4% | 0 | 0 | 0 | 0 | 3 |
| number | 159 | 17.7% | 3 | 0 | 0 | 0 | 1 |
| name | 95 | 10.6% | 0 | 0 | 0 | 0 | 1 |
| date | 93 | 10.3% | 0 | 0 | 1 | 0 | 1 |
| names | 90 | 10.0% | 0 | 0 | 0 | 0 | 0 |
| **Total** | **900** | **100%** | **3** | **26** | **27** | **1** | **28** |

All 3 nulls are in `number` type. All 26 no-info and 26 no-pages are in `free_text`. The single >5s outlier is also `free_text`. These facts establish `free_text` as the hardest type by every quality and latency metric.

---

## TTFT Performance by Type

| Type | Avg (ms) | Median (ms) | P5 | P25 | P75 | P95 | Max | Avg F |
|------|----------|-------------|-----|------|------|------|------|-------|
| names | 403 | 340 | 2 | 3 | 690 | 1237 | 1609 | 1.0477 |
| date | 634 | 331 | 2 | 3 | 915 | 2123 | 4058 | 1.0402 |
| name | 807 | 415 | 2 | 348 | 1183 | 2328 | 3751 | 1.0381 |
| boolean | 829 | 813 | 2 | 373 | 994 | 2141 | 4704 | 1.0401 |
| number | 1223 | 907 | 320 | 832 | 1860 | 2481 | 3059 | 1.0330 |
| free_text | 1668 | 1400 | 371 | 1062 | 2313 | 3337 | 6755 | 1.0151 |

### TTFT Bucket Distribution per Type

| Type | <500ms | 500ms-1s | 1-2s | 2-3s | 3-5s | >5s |
|------|--------|----------|------|------|------|-----|
| names | 66 (73%) | 17 (19%) | 7 (8%) | 0 | 0 | 0 |
| date | 61 (66%) | 9 (10%) | 16 (17%) | 6 (6%) | 1 (1%) | 0 |
| name | 50 (53%) | 15 (16%) | 22 (23%) | 6 (6%) | 2 (2%) | 0 |
| boolean | 83 (43%) | 63 (33%) | 33 (17%) | 11 (6%) | 3 (2%) | 0 |
| number | 20 (13%) | 72 (45%) | 33 (21%) | 33 (21%) | 1 (1%) | 0 |
| free_text | 16 (6%) | 33 (12%) | 135 (50%) | 63 (23%) | 22 (8%) | 1 (0.4%) |

Key observation: `names` and `date` have the bimodal profile -- a large cluster at <500ms (DB answerer hits) and a long tail when falling through to LLM. `free_text` is overwhelmingly in the 1-2s band (50%), reflecting LLM generation as the dominant cost.

### F Coefficient Breakdown per Type

| Type | Avg F | @1.05 | @1.02 | @1.00 | <1.00 |
|------|-------|-------|-------|-------|-------|
| names | 1.0477 | 83 (92%) | 7 (8%) | 0 | 0 |
| date | 1.0402 | 70 (75%) | 16 (17%) | 6 (6%) | 1 (1%) |
| boolean | 1.0401 | 146 (76%) | 33 (17%) | 11 (6%) | 3 (2%) |
| name | 1.0381 | 65 (68%) | 22 (23%) | 6 (6%) | 2 (2%) |
| number | 1.0330 | 92 (58%) | 33 (21%) | 33 (21%) | 1 (1%) |
| free_text | 1.0151 | 49 (18%) | 135 (50%) | 63 (23%) | 23 (9%) |

`names` achieves near-maximum F (1.0477 vs theoretical max 1.05) because 92% of answers are served in <1s. `free_text` drags the overall F down -- 9% of free_text answers are in the penalty zone (<1.00), and only 18% earn the full 1.05 bonus.

---

## Model Routing by Type

| Model | Total | boolean | number | date | name | names | free_text |
|-------|-------|---------|--------|------|------|-------|-----------|
| db-answerer | 118 | 28 | 1 | 32 | 16 | 33 | 8 |
| strict-extractor | 281 | 89 | 53 | 55 | 44 | 40 | 0 |
| gpt-4.1-mini | 377 | 76 | 105 | 6 | 35 | 17 | 138 |
| gpt-4.1 | 99 | 0 | 0 | 0 | 0 | 0 | 99 |
| structured-extractor | 22 | 0 | 0 | 0 | 0 | 0 | 22 |
| premise-guard | 3 | 0 | 0 | 0 | 0 | 0 | 3 |

### Per-Model TTFT

| Model | Count | Avg TTFT | Median TTFT | P95 TTFT |
|-------|-------|----------|-------------|----------|
| db-answerer | 118 | 10ms | 2ms | 7ms |
| strict-extractor | 281 | 807ms | 415ms | 2269ms |
| structured-extractor | 22 | 782ms | 612ms | 2106ms |
| gpt-4.1-mini | 377 | 1478ms | 1116ms | 3079ms |
| gpt-4.1 | 99 | 1707ms | 1391ms | 3340ms |
| premise-guard | 3 | 1754ms | 1411ms | 2792ms |

Three speed tiers are visible:
1. **Instant** (<50ms): DB answerer -- 118 questions short-circuited via corpus registry metadata lookup. No retrieval, no LLM.
2. **Fast** (400-800ms): Strict/structured extractors -- LLM call with span extraction, smaller prompt.
3. **Standard** (1100-1700ms): Full LLM generation via gpt-4.1-mini or gpt-4.1.

---

## Per-Type Deep Dive

### Boolean (193 questions)

- **Answer distribution**: Yes=50 (26%), No=143 (74%). Heavy skew toward "No".
- **Model routing**: strict-extractor (46%), gpt-4.1-mini (39%), db-answerer (15%).
- **Zero nulls, zero no-pages, zero no-info**. Boolean is our cleanest type.
- **Known issues**: 22 party-overlap hallucinations (True->False corrected in FINAL_SUBMISSION), 10 additional TAMAR boolean fixes for ENF/ARB co-retrieval hallucinations.
- **TTFT**: bimodal -- 43% under 500ms (db-answerer + fast strict-extractor hits), 17% in 1-2s band (LLM fallthrough).
- **Warmup exact match**: 0.80 (V6 public), improved to est. ~0.97 after 32 boolean corrections.
- **Strategy**: DB answerer handles party-existence checks, strict-extractor handles yes/no extraction from retrieved pages, gpt-4.1-mini handles complex boolean reasoning.

### Number (159 questions)

- **3 nulls** -- the only type with null answers. All 3 are genuinely unanswerable (confirmed: e.g., ARB 035/2025 award amount not in corpus).
- **Model routing**: gpt-4.1-mini (66%), strict-extractor (33%), db-answerer (1%).
- **Avg answer**: numeric string. 16 number corrections applied in final submission (per-claimant vs total confusion, penalty regex errors).
- **TTFT**: no sub-500ms cluster (only 13% under 500ms) because DB answerer handles just 1 number question. Most go through strict-extractor (500-1000ms) or gpt-4.1-mini (1-2s).
- **Known issues**: count questions were misrouted to LOOKUP_FIELD (DAGAN commit a7d195d fix), penalty answers matched year numbers (FIXED).

### Date (93 questions)

- **All answers are 10-character ISO strings** (YYYY-MM-DD). Zero nulls.
- **1 no-pages question**: minor grounding gap.
- **Model routing**: strict-extractor (59%), db-answerer (34%), gpt-4.1-mini (6%).
- **TTFT**: 66% under 500ms -- second-fastest type. DB answerer covers 32/93 dates (34%) via DOI lookup.
- **Known issues**: 56 DOI dates were wrong in early versions (body dates vs cover page dates). All PDF-verified and corrected.

### Name (95 questions)

- **Avg answer length**: 14 chars (min=3, max=50). Compact entity extraction.
- **Model routing**: strict-extractor (46%), gpt-4.1-mini (37%), db-answerer (17%).
- **53% under 500ms** -- DB answerer and fast strict-extractor hits.
- **Known issues**: case prefix confusion (CFI vs SCT vs CA), entity resolution ambiguity. Warmup exact match was 0.6429 (lowest among deterministic types).
- **Corrections applied**: case number trimming (full sentence -> "CA 006/2024"), name consistency across related questions.

### Names (90 questions)

- **All answers are lists** (avg size=1.2, max=3). No strings, no nulls.
- **Fastest type overall**: avg 403ms, median 340ms, 92% achieve F=1.05.
- **Model routing**: strict-extractor (44%), db-answerer (37%), gpt-4.1-mini (19%).
- **Zero questions in penalty zone** (3-5s or >5s). The cleanest TTFT profile.
- **DB answerer covers 33/90** (37%) -- entity list extraction from corpus registry.
- **Warmup exact match**: 1.000 on golden labels (3/3 matched).

### Free Text (270 questions)

- **The hardest type by every metric**:
  - All 26 no-info answers are here. All 26 no-pages are here.
  - The single >5s outlier (6755ms) is here. 22 answers in the 3-5s penalty zone.
  - Avg F=1.0151 (lowest), 9% of answers penalized.
- **Avg answer length**: 175 chars (min=26, max=280). 280-char platform limit is binding.
- **Model routing**: gpt-4.1-mini (51%), gpt-4.1 (37%), structured-extractor (8%), db-answerer (3%), premise-guard (1%).
- **TTFT distribution**: 50% in 1-2s band, only 6% under 500ms.
- **Known issues**:
  - 42% of V9.1 answers were truncated mid-sentence at 280 chars (SHAI fix in V10.1 reduced to 1).
  - 27 false "no information" answers where pages were actually present.
  - 60% of early answers started with "The..." instead of evidence-first ("Article X...").
  - gpt-4.1-mini for simple free_text trades quality for speed.
- **Asst bottleneck**: free_text is the only type scored by LLM judge (Asst component). Estimated Asst ~0.75 vs leader 0.83 -- the single largest scoring gap.

---

## Retrieval Pages by Type

| Type | Avg Pages | Implication |
|------|-----------|-------------|
| free_text | 6.3 | Most pages retrieved -- complex multi-page evidence synthesis |
| boolean | 2.6 | Moderate -- cross-document comparison |
| date | 2.6 | Moderate -- sometimes multiple date candidates |
| name | 2.3 | Low -- usually a single entity on one page |
| number | 2.1 | Low -- single numeric fact |
| names | 1.5 | Lowest -- entity lists from single sections |

The correlation between page count and TTFT is clear: more retrieved pages means more context for the LLM, longer prompt, slower generation.

---

## Scoring Impact by Type

The total score formula is `Total = (0.7 * Det + 0.3 * Asst) * G * T * F`.

- **Det** (70% weight): boolean, number, date, name, names -- 630/900 questions (70%).
- **Asst** (30% weight): free_text only -- 270/900 questions (30%).

Deterministic types dominate the score. A single wrong boolean costs `0.7 * (1/630) = 0.11%` of Total. A single bad free_text costs `0.3 * (1/270) = 0.11%` of Total -- same weight per question, but free_text is judged by LLM (softer, partial credit) while Det is exact match (binary).

### Score Contribution Estimates

| Type | Det/Asst | Est. Accuracy | Score Contribution |
|------|----------|---------------|-------------------|
| boolean | Det | ~0.97 (after 32 fixes) | Solid -- corrected party-overlap hallucinations |
| number | Det | ~0.96 (3 nulls, 16 fixes) | Good -- count misrouting fixed |
| date | Det | ~0.98 (56 DOI fixes) | Strong -- DOI dates PDF-verified |
| name | Det | ~0.95 (lowest Det type) | Weakest Det -- entity resolution ambiguity |
| names | Det | ~0.99 | Near-perfect -- DB answerer + clean lists |
| free_text | Asst | ~0.75 (LLM-judged) | Largest gap vs leader (0.83) |

---

## Summary

1. **Names is our best type**: fastest (403ms avg), highest F (1.0477), 100% warmup exact match, zero quality issues.
2. **Free_text is our worst type**: slowest (1668ms avg), lowest F (1.0151), all no-info/no-pages, Asst bottleneck.
3. **DB answerer is the TTFT lever**: 118/900 questions (13.1%) answered in ~10ms. Covers 37% of names, 34% of dates, 15% of booleans.
4. **Model routing works**: gpt-4.1 only for complex free_text (99 questions), gpt-4.1-mini for everything else that needs LLM, extractors for structured types.
5. **Det types are strong post-corrections**: 103+ corrections pushed boolean/number/date/name toward ~0.96-0.99. The gap to leader (1.000) is ~2-4 questions.
6. **Free_text Asst is the ceiling**: estimated 0.75 vs leader 0.83. Improving Asst by 0.08 would add ~2.4% to Total -- more than any other lever.
