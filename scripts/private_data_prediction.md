# Private Data Document Prediction

**Author**: TAMAR
**Date**: 2026-03-20 (updated 2026-03-20T22:40Z)
**Purpose**: Predict what documents, question types, and patterns will appear in private set.
**Source**: Analysis of 100 warmup questions + 39-document corpus index + pipeline telemetry + Phase 5 diagnostic findings.

---

## 0. Private Day Strategy (updated with Phase 5 findings)

**Expected score boost from current fixes (pre-private):**
| Fix | Expected ΔG | Status |
|-----|------------|--------|
| page_budget=2 | +25.2pp F-beta (EYAL oracle model) | COMMITTED |
| SHAI 5a cite-all | +16–23pp local F-beta (34 C1 → <10) | COMMITTED (no new eval yet) |
| OREV dual-case | +6pp | COMMITTED |
| SHAI 4a+4b prompt | +1-2pp Det | COMMITTED |
| Boolean extraction fixes | +3-5 Det questions | COMMITTED |

**Private day actions by priority:**
1. Run `bash scripts/private_fast_retrain.sh` (EYAL retraining on private data)
2. Start ingestion immediately — expect 300–400 docs, 60-90 min to ingest
3. Submit with v7_enhanced profile first
4. Use `uv run python scripts/test_grounding_recall.py` after each eval run to measure G funnel

---

## 1. Expected Question Type Distribution

Based on warmup (100 questions):

| Answer Type | Warmup Count | Warmup % | Predicted Private % |
|------------|-------------|---------|-------------------|
| boolean | 35 | 35% | 33–38% |
| free_text | 30 | 30% | 28–33% |
| number | 17 | 17% | 15–20% |
| name | 14 | 14% | 12–16% |
| names | 3 | 3% | 2–5% |
| date | 1 | 1% | 1–2% |
| unanswerable | 8 | 8% | 7–10% |

For ~1,000 private questions: expect ~350 boolean, ~300 free_text, ~170 number, ~140 name/names, ~80 unanswerable.

---

## 2. Document Type Distribution

### Warmup corpus (39 docs):
- **16 laws** (41% of corpus) — most heavily questioned
- **16 court cases** (41%) — cross-case comparison dominant
- **5 enactment notices** (13%) — mostly 1-page, used for date/law-number questions
- **2 amendment laws** (5%) — multi-document chain questions

### Law document breakdown (16 laws in warmup):
| Law | Pages | Warmup Q count | Priority |
|-----|-------|---------------|---------|
| Operating Law 2018 | 43pp | 9 | HIGH |
| Employment Law 2019 | 44pp | 8 | HIGH |
| Data Protection Law 2020 | 54pp | 4 | MEDIUM |
| General Partnership Law 2004 | 23pp | 6 | HIGH |
| Real Property Law 2018 | 70pp | 4 | MEDIUM |
| Strata Title Law 2007 | 73pp | 5 | MEDIUM |
| Limited Partnership Law 2006 | 27pp | 2 | MEDIUM |
| LLP Law 2004 | 25pp | 2 | LOW |
| Contract Law 2004 | 41pp | 1 | LOW |
| Foundations Law 2018 | 47pp | 2 | MEDIUM |
| CRS Law 2018 | ~15pp | 3 | MEDIUM |
| Law on Application of Civil Law 2004 | 7pp | 4 | MEDIUM |

### Court case breakdown (16 cases in warmup):
| Case Type | Count in Warmup | Note |
|-----------|----------------|------|
| CFI | 4 cases, 11 question refs | Highest frequency |
| SCT | 4 cases, 7 refs | Second highest |
| ARB | 2 cases, 6 refs | Arbitration |
| CA | 2 cases, 6 refs | Court of Appeal |
| ENF | 2 cases, 6 refs | Enforcement |
| TCD | 1 case, 4 refs | Technology & Construction |
| DEC | 1 case, 4 refs | Declaratory |

---

## 3. Predicted New Documents in Private Set

**For ~1,000 private questions, expect ~300–400 documents.** The warmup corpus is a representative sample. Private corpus is 8–10× larger.

### High-probability new laws (referenced in warmup questions but not as standalone docs):
1. **Companies Law 2018** — referenced in CRS Law context, likely 50-100pp
2. **Trust Law 2018** — appears in several question comparisons with LP/LLP laws
3. **Intellectual Property Law 2019** — referenced in enactment notice questions
4. **Leasing Law 2020** — mentioned in penalty comparison questions
5. **Non-Profit Incorporated Organisations Law 2012** — in registrar comparisons
6. **Insolvency Law** — referenced alongside Companies Law
7. **Arbitration Law** — given ARB case frequency, substantive arbitration law likely
8. **Personal Property Law 2005** — mentioned in article reference questions
9. **Common Law Act / Contract Law amendments** — multi-version pattern

### Expected new court case types:
- More CFI cases (most frequent case type in warmup, likely more in private)
- More SCT small claims cases
- Potentially new types: APP (appellate), FID (fiduciary/family?), LST (land/strata?)
- More ARB arbitration cases covering different financial instruments
- International enforcement cases (ENF with foreign parties)

### Documents unlikely to appear:
- Regulations (only Strata Title Regulations appeared in warmup — 1 case)
- Forms, practice directions — probably too procedural for Q&A
- Non-English documents — DIFC uses English exclusively

---

## 4. Question Pattern Predictions

### Cross-case comparison questions (~20–25% of private set):
The warmup had 44 questions referencing case IDs in the question text. Private will have proportionally:
- `[Case A] vs [Case B]: which was decided earlier?` (date comparison)
- `[Case A] + [Case B]: common party?` (shared party boolean)
- `[Case A] + [Case B]: same judges?` (shared judge boolean)
- These REQUIRE dual-case context guarantee. OREV fix is critical.

### Multi-document enumeration (~5–8%):
The hardest category. Private likely has:
- `Which laws were amended by DIFC Law No. X of YYYY?` (requires reading many docs)
- `Which laws mention [concept]?` (12+ docs needed)
- `Laws enacted in [year]?` (spanning corpus)
- **Prediction**: ~50–80 questions in private set with 10+ gold pages needed

### Article-specific legal questions (~40–50%):
Single-document questions asking about specific Article N(M) provisions:
- `Under Article X of Law Y, can Z do W?` (boolean)
- `According to Article X of Law Y, what is the [specific term/number]?` (name/number)
- These are the Category C1 pattern — SHAI mandatory citation rule critical

### Unanswerable traps (~7–10%):
- Questions about legal concepts not in DIFC law (jury/parole/Miranda pattern)
- False-premise questions (`GPL Art 19(4)` which doesn't exist)
- Cross-law comparisons where both laws have the same provision (ambiguous)

---

## 5. Document Version Duplication Pattern

**Critical structural feature**: 9 of 16 laws in warmup have duplicate PDF versions (different hashes, same content). This occurs because:
- Original version + Consolidated version (March 2022, March 2024 etc.)
- Two versions have slightly different hashing due to different PDF metadata

**Impact on private data**:
- Questions about dual-version laws always need BOTH pages cited for full G
- Expect ~50–60% of law questions to have multi-version gold pages
- The `two-version law A1 pattern` from g_audit_findings will apply to all law questions

---

## 6. Page-Level Citation Statistics

From warmup gold analysis:
- **Page 1**: cited 78× — covers case captions, law titles, preambles
- **Page 4**: cited 30× — first substantive articles
- Pages 1–9 cover ~90% of citations — most content near document start
- Very deep pages (39+) only appear for long laws like Real Property Law

**Private data implication**: Pipeline's retrieval by embedding should work well since DIFC docs are densely structured with provisions on early pages.

---

## 6b. Det Failure Patterns (from warmup audit — watch for these on private data)

From NOGA-7b exhaustive audit (15 real Det failures in warmup):

| Category | Count | QIDs | Fix Status |
|---------- |-------|------|-----------|
| null_return (retrieval miss) | 2 | bb67fc19, d5bc7441 | OREV: enactment notice boost needed |
| null_return (extraction failure) | 3 | 30ab0e56, af8d4690, 47cb314a | SHAI 5d prompt committed |
| polarity_inversion | 5 | df0f24b2, bd8d0bef, b249b41b, 4ced374a, 75bf397c | SHAI 5c boolean negation committed |
| wrong_name | 2 | 3dc92e33, 82664b58 | SHAI 4b committed; 82664b58 ambiguous query |
| wrong_number | 3 | d204a130, f378457d, f2ea23e9 | Some label errors, SHAI fix partial |

**Key private data warnings:**
- Polarity inversion (5/15 failures): GPT-4.1-mini comprehension error for negated booleans — expected rate on private ~3-5%
- Null extraction when evidence present (3/15): affects prohibition-type booleans ("is X NOT permitted?")
- Enactment date retrieval: 15/100 warmup questions need enactment notice page 1 — private set will have proportionally (150/1000 questions)

## 6c. Enactment Date Retrieval Pattern (new finding)

**Critical for private data**: 8 of 15 date/enacted questions require page 1 of **enactment notice documents** (separate 1-page PDF). These are NOT retrieved reliably because 1-page docs have low embedding density.

Two types of date questions:
1. **Enactment notice type** (needs page 1 of enactment notice doc): "Was [Law A] enacted earlier than [Law B]?"
2. **Law commencement type** (needs page 4 of substantive law): "What is the commencement date of [Law]?"

**OREV fix needed**: For queries containing "enacted", "came into force", "enactment notice", "commencement" — boost retrieval of 1-page enactment notice docs.

## 7. OCR / Extraction Risk

### Low-risk documents (clean PDFs):
- Most DIFC laws are professionally typeset PDFs — clean text extraction
- Court case PDFs are court-issued, well-structured

### Medium-risk (may require OCR):
- **Enactment Notices**: 1-page documents, sometimes scanned signatures
- **Older laws (2004–2007)**: may have lower-quality PDFs
- **Very long laws (Real Property 70pp, DPL 54pp)**: large docs may have extraction issues

### High-risk (potential OCR failures):
- **Regulations attachments**: Strata Title Regulations had pipeline miss issues
- **Amendment schedules**: amendment laws sometimes have tabular data
- Any document stored as image PDF (rare in DIFC but possible)

---

## 8. Edge Cases to Pre-Test

**Before private data arrives**:
1. Test pipeline on a long law (Real Property Law 70pp) — ensure all pages indexed
2. Test a 4-case cross-comparison question (private may have 3+ case queries)
3. Test enumeration query ("all laws amended by X") — budget and recall
4. Test enactment notice date extraction (1-page docs, ensure not dropped by page scorer)
5. Test amendment law questions — Law No. 2 of 2022 amended 19 laws; private set may have similar

---

## 9. High-ROI Pre-Arrival Checklist

| Priority | Action | Impact |
|---------|--------|--------|
| 1 | Verify all dual-version docs properly indexed in Qdrant | Fixes A1 pattern for 50% of law questions |
| 2 | Confirm page_budget=2 active in private profiles | Fixes cross-case C2 misses |
| 3 | Test enactment notice retrieval (1-page docs) | Ensures date/law-number questions answered |
| 4 | Run test_grounding_recall.py after each pipeline change | Empirical baseline for comparison |
| 5 | Pre-index if corpus >500 docs — start before questions arrive | Avoids ingestion time crunch |

---

## Summary

**The private corpus is structurally similar to warmup** with the same DIFC document families (laws, court cases, enactment notices, amendment laws). Key differences:
- ~8–10× more documents and questions
- New laws we haven't seen yet — but same DIFC structure/style
- More cross-case comparisons (CFI/SCT/ARB growth proportional)
- Multi-doc enumeration questions will remain the hardest category

**Biggest private set risk**: A major new document type we haven't ingested. Monitor for: company records, financial instruments, regulatory frameworks (e.g., DFSA regulations), or case types not in warmup (FID, APP, COB).
