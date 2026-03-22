# Private Data Playbook — Step-by-Step Checklist

**Author**: TAMAR (QA analyst)
**Date**: 2026-03-20
**Time budget**: ~10 hours from private data arrival
**Goal**: Maximize expected score on private set (1st place)

---

## Pre-Arrival Checklist (do NOW, before data arrives)

- [ ] Confirm TZUF's fixes are deployed: joblib in pyproject.toml, v6 profiles have explicit segment/diversity=false
- [ ] Confirm v4 scorer model exists at `models/page_scorer/v4_tamar_augmented/page_scorer.joblib`
- [ ] Confirm OREV's dual-case fix is implemented (or mark as pending, track ETA)
- [ ] Confirm SHAI's prompt templates are merged (shai/prompt-engineering @ a389bc6)
- [ ] Confirm NOGA's answer quality gate is integrated (generation_logic.py:448)
- [ ] Confirm EYAL's fast-iteration runbook is available (`scripts/private_fast_iteration_runbook.md`)
- [ ] Run `pytest --co -q` to confirm 1100+ tests pass on current main
- [ ] Smoke-test v6_regime on 5 warmup questions to confirm baseline works
- [ ] Smoke-test v7_enhanced on same 5 questions to confirm no crash

---

## Hour 0-1: Arrival and Ingestion Setup

### 0:00 — Private data arrives

**CRITICAL**: Do NOT skip to submission. Run this checklist in order.

1. **Inspect private docs immediately**
   - Count documents: how many PDFs?
   - Check for OCR-heavy docs (screenshots, scanned text) → flag for OCR pipeline
   - Run `scripts/scan_private_doc_anomalies.py` to detect anomalies
   - Note any doc format departures from warmup (new case types, law formats, regulation formats)

2. **Start v6 ingestion immediately** (while you review questions)
   ```bash
   ENV_FILE=profiles/private_v6_regime.env docker compose up ingest
   ```
   This is the longest step; start it first.

3. **Review private questions (while ingest runs)**
   - Count question types: boolean, number, names, free_text, date
   - Count cross-case comparison questions (pattern: `[A-Z]{2,3} \d{3}/\d{4}`)
   - Count multi-law enumeration questions (pattern: "which laws mention/include/have")
   - Flag any new answer type patterns not seen in warmup
   - Compare distribution to warmup: if radically different → alert team

---

## Hour 1-3: First Run (v6_regime — safest baseline)

### Profile: `private_v6_regime.env`

**Purpose**: Establish safe baseline. If everything else fails, this is our answer.

```bash
cp profiles/private_v6_regime.env .env.local
# Run full evaluation
docker compose run pipeline python -m pipeline.eval --output runs/private_v6_run1.json
```

**Monitor for**:
- Telemetry completeness: every question has `retrieved_page_ids`, `context_page_ids`, `used_page_ids`
- G=0 rate: expect ~50% (baseline)
- TTFT: log first token times
- Any crash/timeout → investigate before proceeding

**After run**:
- [ ] Save artifact: `runs/private_v6_run1.json`
- [ ] Run `scripts/score_against_golden.py` if golden available, else inspect answers manually
- [ ] Count obvious wrong answers (boolean returned as name, null answers for non-unanswerable questions)
- [ ] DO NOT SUBMIT YET — establish comparison first

---

## Hour 3-5: Second Run (v7_enhanced — primary candidate)

### Profile: `private_v7_enhanced.env`

**Purpose**: Primary submission candidate with all validated improvements.

```bash
cp profiles/private_v7_enhanced.env .env.local
# Ensure v6 ingestion is complete before running v7 (same collection)
docker compose run pipeline python -m pipeline.eval --output runs/private_v7_run1.json
```

**Monitor for**:
- Same telemetry completeness checks
- Scorer output: `page_scorer_used=true` for article-provision questions
- Sidecar output: `sidecar_activated=true` for appropriate queries
- Cross-ref boost: `cross_ref_boost_applied=true` appearing in logs

**A/B Quick Comparison (after v7 run)**:

```bash
python scripts/compare_artifact_outputs.py \
  runs/private_v6_run1.json \
  runs/private_v7_run1.json \
  --output runs/v6_vs_v7_delta.json
```

**Decision gate**:
| Condition | Action |
|-----------|--------|
| v7 G-avg > v6 G-avg by ≥2 pp | Submit v7 as primary |
| v7 G-avg ≈ v6 G-avg (±1 pp) | Inspect manually; prefer v7 if telemetry clean |
| v7 G-avg < v6 G-avg by >2 pp | REGRESSION — use v6, investigate v7 |
| v7 has telemetry gaps (missing fields) | STOP — fix telemetry before submitting |

---

## Hour 5-6: First Submission

**Submit whichever profile won the A/B gate (v7 or v6 fallback).**

```bash
# Submit best run
python scripts/competition_supervisor.py --run runs/[best_run].json --submit
```

**Immediate post-submit**:
- [ ] Record exact config hash, commit SHA, profile name in `platform_runs/private/` manifest
- [ ] Note exact submission time
- [ ] Do NOT change code between submission and score check

---

## Hour 6-7: Fast Iteration (if v7 won)

**Only proceed if**:
- v7 submitted and score looks strong
- Dual-case fix from OREV is available

### If dual-case fix is ready:
1. Create challenger: v7_enhanced + dual_case_fix
2. Run on 30-question sample only (known dual-case questions)
3. Compare G scores for only those 9 questions
4. If all 9 show improvement → run full eval → submit as submission 2

### EYAL fast-iteration protocol:
- Use `scripts/private_fast_iteration_runbook.md`
- Intelligence toolkit: ~1.2s per cycle
- Retrain if >5 new grounding labels available: ~0.3s

### If v6 was forced (v7 regressed):
- Investigate sidecar issue: grep logs for `sidecar_error`
- Check scorer: is `page_scorer.joblib` loading? Try `PIPELINE_ENABLE_TRAINED_PAGE_SCORER=false` to isolate
- If scorer is causing regression → create v7_no_scorer profile and test

---

## Hour 7-8: Optional Third Run (v8_full or 1792)

**Conditions required to run v8_full**:
- v7 is already submitted and performing
- >= 2.5 hours remaining
- No telemetry fires in v7 run
- OREV confirms segment_retrieval/diversity features are stable

**Conditions required to run 1792 profiles**:
- v6 or v7 ingest complete on 1024-dim collection
- Separate 1792 ingest complete (parallel or sequential)
- `QDRANT_COLLECTION=legal_chunks_private_1792` points to non-empty collection

---

## Hour 8-9: Answer Repair Pass (if time permits)

Focus on highest-impact repairs:
1. **Null/empty answers**: Any G=0 with non-empty gold → inspect context, try manual answer
2. **Unanswerable false positives**: Did we return answers for questions that should be null?
3. **Format errors**: Wrong answer type returned (boolean as name, number as text)

Run `scripts/build_exactness_review_queue.py` to prioritize.

---

## Hour 9-10: Final Submission

- [ ] Confirm best run artifact is intact and telemetry-complete
- [ ] Double-check submission count — do not waste submissions on speculative changes
- [ ] Final submission: best run with highest expected private score
- [ ] Record manifest entry with config, commit, profile, timestamp

---

## Rollback Plan

### Scenario: v7 submitted, then v8 regresses

1. Do NOT submit v8 if A/B shows regression
2. Fall back to v7 artifact already submitted
3. If v8 submission was made and score regressed → immediately queue another submission with v7 artifact
4. Log regression details for post-competition analysis

### Scenario: All runs produce worse score than expected

1. Revert to v6_regime baseline — it worked at 0.742 on warmup
2. Verify private doc distribution isn't radically different from warmup
3. Check for ingestion errors: any PDFs that failed to parse?
4. If OCR-heavy docs: run OCR pipeline and re-ingest affected documents

### Hard rollback triggers (use v6 immediately):
- Any sidecar producing `page_ids=[]` for answered questions
- Scorer outputting negative probabilities or NaN
- TTFT >5s median (latency penalty threshold)
- Any crash in v7_enhanced that doesn't appear in v6

---

## TAMAR Oracle Reference Data (for sanity checks)

### Warmup distribution (100 questions):
| Type | Count | Avg G current | Post-improvement target |
|------|-------|--------------|------------------------|
| boolean | 35 | 0.433 | ~0.60 |
| free_text | 30 | 0.483 | ~0.58 |
| number | 17 | 0.514 | ~0.62 |
| names | 17 | 0.396 | ~0.52 |
| date | 1 | 0.879 | ~0.88 |

### Critical success thresholds (TAMAR estimates):
- G-avg > 0.55 with v7_enhanced → likely 1st place range
- G-avg between 0.50-0.55 → strong submission, watch for competitors
- G-avg < 0.50 → investigate before proceeding, possible regression

### Cross-case comparison questions (expected ~15-25% of private set):
- Use dual_case_test_cases.json as test harness
- All 9 warmup dual-case questions had G=0.0 → high gain opportunity
- If private set has 20+ dual-case questions → dual-case fix is worth 15-18 pp alone

### Multi-law enumeration questions (expected ~5-8% of private set):
- Current ceiling: ~0.1-0.15 G for these
- v8 doc-diversity helps but won't reach >0.5 without prefetch=200+
- Accept low G on these and optimize everywhere else

---

## Emergency Contacts / Escalation

- Telemetry broken → OREV (owns pipeline telemetry)
- Page scorer crash → EYAL (owns scorer)
- Prompt format error → SHAI (owns generation prompts)
- Integration failure → TZUF (owns integration testing)
- Answer quality issues → NOGA (owns validation/consensus)
