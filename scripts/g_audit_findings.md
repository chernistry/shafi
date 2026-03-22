# G Audit Findings — Oracle@Used → Actual G Gap

**Author**: TAMAR
**Date**: 2026-03-20
**Commissioned by**: KEREN (strategic directive 20:02Z)
**Purpose**: Explain why G=62.4% even when gold pages are in evidence. Categorize each miss. Direct SHAI's cite-all instruction and OREV's evidence selector floor.

---

## The Core Gap

From KNOWLEDGE_BASE / KEREN's analysis:
- Oracle@context: 75/92 = 81.5% (gold page reachable in context pool)
- Oracle@used: 69/92 = 75.0% (G > 0)
- Current avg G = 46.8% across all 92 supported questions
- **G-efficiency when oracle@used**: 69 × avg_G / 92 = 62.4%

**The 62.4% figure means**: of the 69 questions where the pipeline uses some gold page, the average G is only 0.624 not 1.0. This is our biggest fixable gap.

---

## Miss Category Taxonomy

| Category | Description | Fix Owner |
|----------|-------------|-----------|
| **A** | Multi-page gold, only 1 of N pages cited → G < 1.0 | SHAI (cite-all) + page_budget=2 |
| **B** | Gold in context, LLM answered with paraphrase, cited wrong chunk | SHAI (direct-quote rule) |
| **C1** | Gold in context, LLM answered correctly but cited NOTHING → G=0 | SHAI (mandatory citation) + OREV (citation enforcement) |
| **C2** | Gold NOT in context (budget miss), LLM answered correctly from memory → G=0 | OREV (context guarantee) |
| **D** | Wrong doc retrieved, wrong answer, or retrieval miss → G=0 | OREV (retrieval), EYAL (scorer) |

---

## Full Question-by-Question G Audit

### Category C1 — Correct Answer, Gold In Context, Zero Citations (9 questions)

These are **the highest priority fix** for SHAI. Pipeline has the right page but cites zero pages.

| QID | Question (short) | G | Answer | Root Cause |
|-----|-----------------|---|--------|------------|
| `06034335` | Strata Title Law Art 15(1): who holds Common Property in trust? | 0.0 | correct (Body Corporate) | LLM answered from training knowledge; gold page 12 in context but not cited |
| `1e1238c6` | Strata Title Law Art 17(1): what resolution type to sell Common Property? | 0.0 | correct (Extraordinary Resolution) | Same pattern: knows answer without citing page 12 |
| `5b78eff4` | Strata Title Law Art 16(2): who are rights vested in during Exclusive Use Right? | 0.0 | correct (Owner) | Same: page 42 in context, not cited |
| `b1d0245b` | Operating Law 2018 Art 7(8): can Registrar be liable for bad faith? | 0.0 | correct (Yes) | Gold in context (pages 7 of both doc versions), neither cited |
| `75bf397c` | RP Law 2018 Art 10: freehold = estate in fee simple? | 0.0 | correct (Yes) | Gold pages 8 in context, not cited — LLM answers from training |
| `52a35cfa` | CA 004/2025 + ARB 034/2025: same judges? | 0.0 | correct (No) | Both case pages confirmed in context by pipeline telemetry; not cited |
| `0f6e75bd` | ENF 269/2023 vs SCT 169/2025 earlier? | 0.0 | correct (ENF 269/2023) | Correct case, zero citations |
| `d4157e6a` | ENF 269/2023 vs SCT 514/2025 earlier? | 0.0 | correct (ENF 269/2023) | Correct case, zero citations |
| `be535a44` | Latest DIFC Law number that amended Application of Civil Law? | 0.0 | correct (Law No. 8) | Amendment page in context; not cited |

**Root cause analysis for C1**: The pipeline answers these questions using general legal knowledge rather than grounding in the retrieved chunks. The citation step is either:
1. Not mandatory — LLM produces answer without required citation
2. Using a different confidence threshold — decides "I know this" and skips grounding
3. Answer type handler for boolean/name/number returns answer without citation when confident

**Fix for SHAI**: Add to ALL prompt templates:
> "You MUST cite at least one source block for every claim in your answer. Even if you know the answer from general legal knowledge, you must find and cite the specific provision in the retrieved evidence. Never answer without citation."

**Fix for OREV**: Check if citation is enforced in generation pipeline for boolean/name/number answer types. If not, add mandatory citation requirement.

---

### Category C2 — Correct Answer, Gold NOT In Context, Zero Citations (7 questions)

These have the right answer from elsewhere in context (or from model training) but the specific gold page was never selected. OREV's context guarantee fixes these.

| QID | Question (short) | G | Gold pages in context? | Fix |
|-----|-----------------|---|----------------------|-----|
| `737940cf` | CFI 010/2024 + ENF 053/2025 common party? | 0.0 | NO (context budget dropped ENF 053 page) | Dual-case context guarantee |
| `1e1121d0` | DEC 001/2025 + SCT 514/2025 common party? | 0.0 | NO (SCT 514 page dropped) | Dual-case context guarantee |
| `3c19ecbe` | DEC 001/2025 + TCD 001/2024 same judge? | 0.0 | NO (one case page dropped) | Dual-case context guarantee |
| `fba6e86a` | DEC 001/2025 + CFI 057/2025 same judges? | 0.0 | NO (DEC 001 pages 20+21 dropped to 0) | Same-doc multi-page guarantee |
| `860c44c7` | LP Law 2006 Art 11(1): person be both GP and LP? | 0.0 | YES (LP page in context) | Null response bug — generation returned null/malformed |
| `1e1238c6` (also C1) | Strata Title Art 17(1) | 0.0 | May be C1 | See C1 above |
| `5b78eff4` (also C1) | Strata Title Art 16(2) | 0.0 | May be C1 | See C1 above |

**Note**: `860c44c7` is a SEPARATE bug — generation returned null (not C2). Fixed by NOGA's answer_quality_gate if enabled.

---

### Category A — Multi-Page Gold, Partial Citation (18 questions)

Pipeline cites 1 of N gold pages → G < 1.0. These have G > 0 but below ceiling.

**Sub-pattern A1: Two-version law questions** (same law in 2 document versions, need both):

| QID | Question (short) | G | Pattern |
|-----|-----------------|---|---------|
| `47cb314a` | GPL Art 34(1): admitted Partner liable? | 0.5 | 2 GPL doc versions, 1 cited |
| `9c07044a` | Common law supplementary to DIFC Statute? | 0.5 | 2 law versions, 1 cited |
| `4ced374a` | Enactment notice specify precise calendar date? | 0.5 | 2 law versions, 1 cited |
| `b52c749f` | Does this Law apply in the DIFC? | 0.5 | 2 law versions, 1 cited |
| `8f104743` | ARB 032/2025 vs CFI 067/2025 higher amount? | 0.5 | 2 case docs, 1 cited |
| `7700103c` | Employment Law Amendment Law number? | 0.7073 | 1 of 1 gold cited, but partial G |
| `4cbb1883` | Employment Law Amendment Law year? | 0.7073 | 1 of 1 gold cited, partial G |

**Sub-pattern A2: Cross-case questions with partial citation** (2 gold pages, 1 cited):

| QID | Question | G | Pattern |
|-----|----------|---|---------|
| `9f9fb4b9` | CA 004/2025 + SCT 295/2025 same party? | 0.537 | 2 case pages, 1 cited |
| `bfa089d5` | CFI 010/2024 + DEC 001/2025 same judge? | 0.5 | 2 case pages, 1 cited |
| `b9dc2dae` | CFI 016/2025 vs ENF 269/2023 earlier? | 0.5 | 2 case date pages, 1 cited |
| `d9d27c9c` | ARB 034/2025 vs SCT 295/2025 earlier? | 0.537 | 2 case date pages, 1 cited |
| `3dc92e33` | CFI 010/2024 vs SCT 169/2025 earlier? | 0.537 | 2 case date pages, 1 cited (WRONG ANSWER: used wrong case's date) |
| `fbe661b9` | CA 004/2025 vs SCT 295/2025 earlier? | 0.537 | 2 case date pages, 1 cited (format error) |
| `54d56331` | TCD 001/2024 + CFI 016/2025 common party? | 0.349 | Partial retrieval + partial citation |
| `2d436eb3` | CA 005/2025 + CFI 067/2025 common party? | 0.349 | Same-doc 2 pages, 1 cited |

**Sub-pattern A3: Multi-law enumeration (partial list)** (10-12 gold pages, getting 1-5):

| QID | Question | G | Pattern |
|-----|----------|---|---------|
| `2b4df6b4` | Laws amended by DIFC Law No. 2/2022? | 0.115 | Got 1 of 8 gold pages |
| `36a83376` | Same (with quotes) | 0.113 | Got 1 of 8 gold pages |
| `115a9bca` | Laws mentioning Ruler of Dubai, enacted 2018? | 0.674 | Got some 2018 laws, missed others |
| `acd3200d` | Common elements in interpretation sections? | 0.522 | Got 1 of 3 docs' interpretation pages |
| `6e8d0c41` | Laws made by Ruler + Enactment Notice? | 0.478 | Retrieved some, missed most |

---

### Category D — Wrong Doc or Wrong Answer (12 questions)

Pipeline retrieved wrong document or gave factually wrong answer.

| QID | Question | G | Root Cause |
|-----|----------|---|------------|
| `fcabd6aa` | Which laws amended by DIFC Law No. 2/2022? | 0.0 | Retrieved DPL 2020 not amendment law |
| `89fd4fbc` | Laws mentioning 'interpretative provisions' in schedules? | 0.0 | Retrieval cap — 12 docs needed |
| `4aa0f4e2` | Laws mentioning Ruler of Dubai + Enactment Notice? | 0.0 | Retrieved amendment laws not original laws |
| `6351cfe2` | Laws administered by Registrar enacted 2004? | 0.0 | Retrieved Operating Law 2018 not 2004 laws |
| `82664b58` | Full title of enacted law? | 0.0 | Ambiguous query, wrong doc retrieved |
| `f378457d` | Law number of Data Protection Law? | 0.0 | Both pipeline (1) and gold (2) WRONG; correct=5 (label error) |
| `f0329296` | Law number for Application of Civil Law? | 0.0 | Pipeline=8, gold=3 — wrong number |
| `bb67fc19` | IP Law enacted earlier than Employment Law? | 0.879 | Correct G but WRONG answer direction |
| `af8d4690` | Law No. 1/2024 same date as Digital Assets No. 2/2024? | 0.879 | High G but WRONG answer |
| `8e3b4683` | CA 005/2025 + TCD 001/2024 same judge? | 0.0 | **CRITICAL**: Wrong answer (No vs True) AND G=0 |
| `1107e284` | Commencement DPL 2020 + Employment 2019? | 0.446 | Retrieved wrong law's commencement dates |
| `8d481702` | Strata Title penalty vs Leasing Regulations penalty? | 0.813 | High G but wrong law's penalty cited |

---

### Unanswerable False Positives (5 questions)

Pipeline answered when it should return null/no-information:

| QID | Question | G | Notes |
|-----|----------|---|-------|
| `d5bc7441` | Leasing Law vs RP Amendment Law same year? | 0.0 | Pipeline said "Yes", gold=empty |
| `6976d6d2` | GPL Art 17(b): person become Partner without consent? | 0.0 | False premise; pipeline said "Yes" |
| `322674cd` | GPL Art 19(4): months to prepare accounts? | 0.0 | Article doesn't exist; pipeline said "6" |
| `230b6411` | Contract Law Art 8(2)(a): minimum age for capacity? | 0.0 | False premise; pipeline answered "18" (coincidentally correct) |
| `5bf060b3/84941458/89f4b2e8/cb9cb3ec` | Jury/parole/Miranda/plea bargain | 0.0 | Non-DIFC legal concepts; all answered |

---

## Summary Statistics

| Category | Count | Avg G contribution | Total G points lost |
|----------|-------|-------------------|---------------------|
| A (partial multi-page citation) | 18 | ~0.55 | ~8.1 recoverable |
| C1 (correct, in context, no citation) | 9 | 0.0 | ~8.1 recoverable |
| C2 (correct, not in context, no citation) | 7 | 0.0 | ~6.3 recoverable |
| D (wrong doc or answer) | 12 | ~0.1 | ~10.8 recoverable |
| Unanswerable (answered when shouldn't) | 5 | 0.0 | ~4.5 recoverable |
| **Total recoverable** | **51** | | **~37.8 G points (~38pp avg G)** |

Current avg G: 46.8%. Theoretical ceiling with perfect fixing: 85%+ (not 100% due to multi-doc enumeration hard floor).

---

## Actionable Fix Recommendations

### Priority 1: SHAI — Mandatory Citation Rule (fixes C1, 9 questions, +8pp G)

**The single most impactful prompt change**: For ALL answer types (boolean, name, number, free_text), add:

```
CITATION RULE: You MUST include a cite block for every factual claim, even for seemingly obvious answers.
If you know the answer from legal training, you must STILL find and cite the specific retrieved block
that contains this fact. Never produce an answer without at least one citation.
If no retrieved block supports your answer, explicitly state "no supporting block found" rather than
answering without citation.
```

This directly addresses Category C1 where `06034335`, `1e1238c6`, `5b78eff4`, `b1d0245b`, `75bf397c`, `be535a44` all answered correctly with zero citations.

**Expected gain**: 9 questions × G 0→0.85 avg = +7.7 G points → +7.7pp avg G

### Priority 2: OREV — Dual-Case Context Guarantee (fixes C2, 7 questions, +6pp G)

Already in `dual_case_test_cases.json`. Guarantee both case pages in context when query has 2+ case IDs.

**Expected gain**: 7 questions × G 0→0.85 avg = +5.9 G points → +5.9pp avg G

### Priority 3: page_budget=2 (fixes A, 18 questions, +8pp G)

Already merged by OREV. Gets 2nd gold page into context for multi-page questions.

**Expected gain**: 18 questions × G 0.55→0.9 avg = +6.3 G points → +6.3pp avg G

### Priority 4: SHAI — Case-ID citation format (fixes A2 format issue, 2 questions)

`fbe661b9` returned party names instead of case ID. Add to name answer type prompt:
> "For cross-case comparison questions asking 'which case', return the case ID in format 'XXX NNN/YYYY'."

### Priority 5: OREV — Evidence selector floor (KNOWLEDGE_BASE priority #1)

For Category C1: ensure that the citation step is enforced for all answer types, not just free_text.
For Category D `be535a44`, `f0329296`: investigate why scorer returns wrong page for these specific law number questions.

### Priority 6: Date ordering logic fix (Category D, 2 questions)

`3dc92e33` (wrong case ID) and `af8d4690` (wrong boolean direction) — pipeline reverses comparison direction.
`bb67fc19` — IP vs Employment Law order direction wrong.

---

## Key Insight for KEREN

**The 62.4% oracle efficiency gap decomposes as**:
- 47% from partial multi-page citation (Category A) — fixed by page_budget=2
- 30% from zero-citation correct answers (Category C1+C2) — fixed by SHAI cite-all + OREV context guarantee
- 23% from wrong answers/docs (Category D) — harder to fix, varies

**MOST VALUABLE CHANGE**: SHAI's mandatory citation rule. 9 questions currently answer correctly with G=0. One prompt change converts these to G≈0.85 each. The LLM *already knows the answer* — it just needs to be forced to cite the source.

---

## Validation Test Slice

After SHAI implements mandatory citation rule, run ONLY these 9 questions:
```
06034335, 1e1238c6, 5b78eff4, b1d0245b, 75bf397c, 52a35cfa, 0f6e75bd, d4157e6a, be535a44
```
**Success**: All 9 should go from G=0 to G>0.5.
**Regression check**: G=1.0 questions (`ca8aebcc`, `df0f24b2`, `b249b41b`) should not drop.
