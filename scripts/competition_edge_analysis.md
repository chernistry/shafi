# Competition Edge Analysis — Warmup 100

**Date**: 2026-03-20
**Eval**: `data/eval/warmup_raw_20260320_133338.json`
**Based on**: `full_answer_verification.json` (100 questions) + raw telemetry

---

## 1. Where We Lose Points

### Pipeline Funnel (92 supported + 8 unanswerable questions)

| Stage | Questions with gold page present | % |
|-------|----------------------------------|---|
| In `retrieved_page_ids` | 86 / 92 | **93.5%** |
| In `context_page_ids` | 75 / 92 | **81.5%** |
| In `used_page_ids` (G > 0) | 69 / 92 | **75.0%** |

Current average G-score: **0.4676** (46.8%)

The funnel loses 17 supported questions between retrieval and grounding.
The gap between G>0 rate (75%) and actual average G (46.8%) is explained by multi-doc questions
where we get 1–3 of 10+ gold pages — partial grounding that yields low G scores.

### Failure Stage Breakdown (92 supported questions)

| Stage | Count | Loss Type |
|-------|-------|-----------|
| Miss at retrieval | **6** | Gold page not in `retrieved_page_ids` at all |
| Miss at context (budget drop) | **11** | Gold in retrieved, not passed to LLM |
| Miss at grounding | **9** | Gold in LLM context, not used |
| Success (G > 0) | **69** | Gold in used_page_ids |
| Unanswerable (correct) | **4** | Empty gold, pipeline declined correctly |
| Unanswerable (wrong) | **4** | Empty gold, pipeline wrongly answered |

---

## 2. Score Ceiling Per Component

| Scenario | Oracle G | Gain vs Current |
|----------|----------|-----------------|
| Current system | **0.4676** | — |
| Perfect grounding from current context pool | **0.8300** | +36.2 pp |
| Perfect grounding from full retrieval pool | **0.9400** | +47.2 pp |
| True ceiling (all answers correct) | **1.0000** | +53.2 pp |

**Key insight**: The page scorer / grounding component is our biggest bottleneck.
Even with perfect retrieval, we'd need the grounding to select the right pages from
a 60-page pool to reach the 94% ceiling. Today's grounding reaches 83% of the
context ceiling — meaning 17% of questions fail to use gold pages that are in context.

---

## 3. Loss by Answer Type

| Answer Type | Count | Avg G | Correct | Partial | Wrong | Notes |
|-------------|-------|-------|---------|---------|-------|-------|
| boolean | 35 | 0.433 | 23 | — | 8 | Multi-doc comparison bool questions score poorly |
| number | 17 | 0.514 | 8 | — | 7 | Many number answers correct but gold page missed |
| free_text | 30 | 0.483 | 9 | 12 | 7 | Multi-law enumeration questions lose most |
| name/names | 17 | 0.396 | 13 | 1 | 3 | Case-ID and article-name questions mostly correct |
| date | 1 | 0.879 | 1 | — | 0 | — |

**Biggest loss categories:**
1. **Boolean cross-case comparison** (e.g., "Did these two cases share a judge?"): need 2 pages from different docs; context budget drops one
2. **Multi-law enumeration free_text** (e.g., "Which laws mention X?"): need 5–12 pages; retrieval/context caps out
3. **Number questions with wrong page selected**: right doc retrieved, wrong page used

---

## 4. Miss-at-Retrieval (6 questions — hardest to fix)

| QID | Question | Root Cause |
|-----|----------|------------|
| `89fd4fbc` | Which laws mention 'interpretative provisions' in schedules? | gold=12 pages across 12 docs; prefetch cap |
| `54d56331` | Any party common to both CFI 010/2024 and DEC 001/2025? | gold=3 pages; doc not retrieved |
| `fcabd6aa` | Which DIFC Laws amended by Law No. 2 of 2022? | gold=1 page of amendment law; wrong doc retrieved |
| `6351cfe2` | Laws administered by Registrar, enacted 2004–2010? | gold=3 pages; needs 3 specific 2004 laws |
| `82664b58` | Full title of the enacted law? | Ambiguous query; wrong doc surface |
| `f378457d` | Law number of the Data Protection Law? | gold=page 1; enactment notice has low embedding score |

**Fix**: Increase prefetch limits (60→120) and enable cross-ref boost for law number mentions.

---

## 5. Miss-at-Context (11 questions — medium fix)

All 11 have gold page in `retrieved_page_ids` but it's pruned before reaching the LLM.

**Pattern**: 9 of 11 are **dual-case boolean questions** (gold=2 pages from 2 different cases).
The context budget selector picks one case page but not the other.

| QID | Question | Gold pages | Issue |
|-----|----------|-----------|-------|
| `9f9fb4b9` | CA 004/2025 vs SCT 295/2025 common party? | 2 (different cases) | One case page dropped |
| `737940cf` | CFI 016/2025 vs CFI 010/2024 common party? | 2 | " |
| `bfa089d5` | Same judge: CFI 010/2024 + SCT 169/2025? | 2 | " |
| `fba6e86a` | Same judge: DEC 001/2025 + CFI 057/2025? | 2 | " |
| `1e1121d0` | DEC 001/2025 + SCT 514/2025 same party? | 2 | " |
| `3c19ecbe` | Same judge: DEC 001/2025 + ARB 034/2025? | 2 | " |
| `8e3b4683` | Same judge: CA 005/2025 + TCD 001/2025? | 2 | " |
| `46927f37` | Strata Title Regulations "Law" reference? | 2 | Regulation + Law both needed |
| `4aa0f4e2` | Laws mentioning 'Ruler of Dubai'? | 11 | Multi-doc cap |
| `5b78eff4` | Body Corporate Exclusive Use Right vesting? | 1 | Context page selection miss |
| `f0329296` | Law number for 'Law on Application of Civil Code'? | 1 | Specific enactment page dropped |

**Fix**: For dual-doc comparison queries, expand context budget to include at least 1 page per
referenced case. Query classification for "cross-case" pattern → guaranteed dual-doc inclusion.

---

## 6. Miss-at-Grounding (9 questions — easiest to fix)

Gold page is in LLM context but not cited in the answer. Page scorer or proof compiler drops it.

| QID | Question | Notes |
|-----|----------|-------|
| `52a35cfa` | CA 004/2025 + ARB 034/2025 shared judge? | Both case pages in context, one not cited |
| `75bf397c` | RP Law Art 10: can land be subdivided in 2+ phases? | Context has the page; grounding misses it |
| `b1d0245b` | Operating Law: Registrar personal liability? | 2 gold pages in context, not both used |
| `860c44c7` | LP Law Art 11(1): person become partner without consent? | Pipeline returned null (malformed) |
| `0f6e75bd` | ENF 269/2023 vs SCT 169/2025 earlier? | Both case pages in context but only one cited |
| `d4157e6a` | ENF 269/2023 vs SCT 514/2025 earlier? | Same pattern |
| `06034335` | Strata Title Law Art 15(1) Common Property holder? | In context, not cited |
| `1e1238c6` | Strata Title Law Art 17(1) resolution type? | In context, not cited |
| `be535a44` | Latest DIFC Law number amending Civil/Commercial laws? | Amendment page in context; not cited |

**Fix**: Proof compiler / page scorer should be retrained on multi-source grounding patterns.
For `860c44c7`: investigate malformed null output — likely a generation mode edge case.

---

## 7. Easy Wins vs Hard Cases

### Easy Wins (high ROI, low engineering cost)

**EW-1: Fix dual-case context budget (9 context misses, potential +9 G-score points)**
- Pattern: cross-case comparison queries with exactly 2 referenced case IDs
- Detect by: check if query contains 2+ case IDs (regex: `[A-Z]{2,3} \d{3}/\d{4}`)
- Fix: guarantee both referenced case title pages in context regardless of score rank
- Expected gain: ~9 questions × ~0.8 G = +7.2 raw G points → +7.2% avg G

**EW-2: Fix grounding for in-context pages (9 grounding misses)**
- Pattern: gold page is in context but proof compiler/page scorer drops it
- Fix: lower page scorer confidence threshold or widen citation window
- Expected gain: ~7 questions × ~0.8 G = +5.6 raw G points → +5.6% avg G

**EW-3: Unanswerable trap fix (4 questions with empty gold)**
- 4 questions have empty `gold_chunk_ids` in verified labels but pipeline answered
- All 4 pipeline answers are wrong (G=0); correct answer is "no information"
- Fix: audit these 4 QIDs; if classifier can detect no-doc cases, gain back 4 × 1.0 = +4%

**EW-4: Null/malformed response fix (860c44c7)**
- Pipeline returned `null` for LP Law Art 11(1) question
- Fix: ensure generation fallback on empty responses
- Gain: 1 question × ~0.8 G = +0.8%

### Hard Cases (low ROI, requires core changes)

**HC-1: Multi-doc enumeration (89fd4fbc, 4aa0f4e2, 6351cfe2)**
- Requires 10+ pages across 10+ docs; current prefetch cap is ~60 chunks (~15 unique docs)
- Even with prefetch=120, need perfect page selection across 12 docs
- Gain ceiling: +3 questions, high engineering cost

**HC-2: Ambiguous query retrieval (82664b58: "full title of enacted law")**
- Query too vague to match specific document; needs query expansion or clarification
- Gain: 1 question

**HC-3: Wrong golden labels (d6eb4a64)**
- 1 confirmed golden label error (Art 11(1) Employment Law — gold says False, correct is True)
- Pipeline answer is correct; scoring it as "wrong" penalizes us unfairly
- Fix: flag to organizers; potential G recovery if label corrected

---

## 8. Top-10 Fixable Questions

Ordered by expected G gain with feasible engineering fix:

| # | QID | Current G | Expected G | Fix Required | Category |
|---|-----|-----------|------------|--------------|----------|
| 1 | `737940cf` | 0.0 | ~0.9 | Dual-case context budget | Context miss |
| 2 | `bfa089d5` | 0.0 | ~0.9 | Dual-case context budget | Context miss |
| 3 | `fba6e86a` | 0.0 | ~0.9 | Dual-case context budget | Context miss |
| 4 | `1e1121d0` | 0.0 | ~0.9 | Dual-case context budget | Context miss |
| 5 | `3c19ecbe` | 0.0 | ~0.9 | Dual-case context budget | Context miss |
| 6 | `8e3b4683` | 0.0 | ~0.9 | Dual-case context budget | Context miss |
| 7 | `9f9fb4b9` | 0.0 | ~0.9 | Dual-case context budget | Context miss |
| 8 | `52a35cfa` | 0.0 | ~0.9 | Grounding: 2 in-context pages | Grounding miss |
| 9 | `75bf397c` | 0.0 | ~0.9 | Grounding: page in context not cited | Grounding miss |
| 10 | `b1d0245b` | 0.0 | ~0.9 | Grounding: 2 gold pages in context | Grounding miss |

**Projected gain from top-10**: ~10 × 0.9 = +9.0 raw G-points → avg G rises from 46.8% → ~55.8%

---

## 9. Summary Scorecard

| Component | Current | With Easy Wins | With Hard Fixes |
|-----------|---------|---------------|-----------------|
| Avg G-score | 46.8% | ~60% | ~75% |
| G=0 supported questions | 23 | ~10 | ~4 |
| Partial (0<G<1) | 20 | ~20 | ~15 |
| G=1 | 8 | ~15 | ~25 |

**The single highest-ROI change**: Implement dual-case query detection + guaranteed
dual-doc context inclusion for cross-case comparison questions. This fixes 7–9 of
the 11 context misses with a single routing rule, at zero cost to retrieval or training.

---

## 10. Answer Quality Summary (from full_answer_verification.json)

| Verdict | Count |
|---------|-------|
| correct | 66 |
| partially_correct | 12 |
| wrong | 18 |
| unanswerable_confirmed | 4 |
| **Total** | **100** |

Of the 18 wrong answers:
- 8 are retrieval/context misses (can't be answered without the right page)
- 4 are wrong-entity answers (pipeline picks wrong doc/law)
- 3 are format errors (wrong answer format or incomplete)
- 2 are multi-doc partial answers scored as wrong
- 1 is a golden label error (d6eb4a64 — pipeline is actually correct)
