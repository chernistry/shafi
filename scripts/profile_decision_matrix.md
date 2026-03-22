# Profile Decision Matrix — Private Dataset

**Author**: TAMAR (QA analyst)
**Date**: 2026-03-20
**Purpose**: Choose which profile to run first (safest) and which to run second (highest expected) on private data

---

## The 5 Profiles

| Profile | File | Embedding | Retrieval | Grounding | Status |
|---------|------|-----------|-----------|-----------|--------|
| **v6_regime** | `private_v6_regime.env` | 1024-dim | prefetch=60, rerank=80, no cross-ref | No sidecar, no scorer | **PROVEN** — scored 0.742 on warmup |
| **1792_regime** | `private_1792_regime.env` | 1792-dim | prefetch=60, rerank=80, no cross-ref | No sidecar, no scorer | **UNTESTED** — 1792-dim hypothesis |
| **v7_enhanced** | `private_v7_enhanced.env` | 1024-dim | prefetch=120, rerank=120, cross-ref=true | sidecar=true, scorer=v7_regex_fixed | **CHALLENGER** — all v7 improvements |
| **v7_1792_enhanced** | `private_v7_1792_enhanced.env` | 1792-dim | prefetch=120, rerank=120, cross-ref=true | sidecar=true, scorer=v4 | **DOUBLE CHALLENGER** — 1792-dim + v7 |
| **v8_full** | `private_v8_full.env` | 1024-dim | prefetch=120, rerank=120, cross-ref=true + segment + diversity | sidecar=true, scorer=v4 | **KITCHEN SINK** — all features enabled |

---

## Feature Coverage Matrix

| Feature | v6 | 1792 | v7 | v7_1792 | v8 |
|---------|----|----|-----|---------|-----|
| Proven warmup score | ✅ 0.742 | ❌ | ❌ | ❌ | ❌ |
| 1792-dim embeddings | ❌ | ✅ | ❌ | ✅ | ❌ |
| Prefetch 60→120 | ❌ | ❌ | ✅ | ✅ | ✅ |
| Rerank 80→120 | ❌ | ❌ | ✅ | ✅ | ✅ |
| Cross-ref boosts | ❌ | ❌ | ✅ | ✅ | ✅ |
| Segment retrieval | ❌ | ❌ | ❌ | ❌ | ✅ |
| Doc-diversity expansion | ❌ | ❌ | ❌ | ❌ | ✅ |
| v7_regex_fixed page scorer | ❌ | ❌ | ✅ | ✅ | ✅ |
| Grounding sidecar | ❌ | ❌ | ✅ | ✅ | ✅ |
| Warmup validated | ✅ | ❌ | ❌ | ❌ | ❌ |
| Overfit risk | low | low | medium | medium | high |

---

## Risk Assessment

### v6_regime — SAFEST
- **Score**: Known 0.742 on warmup
- **Risk**: Near-zero regression risk; exact warmup reproduction
- **Downside**: Leaves all improvement potential on table
- **Private generalization**: High (proven, no new features to fail)
- **Verdict**: Use as baseline / fallback anchor

### v7_enhanced — BEST RISK-ADJUSTED CHALLENGER
- **Expected score**: +7-10 pp over v6 on warmup-like questions
- **Improvements proven**: EYAL cv_hit@1 +15.6%, OREV sidecar gate fix, TZUF integration verified
- **Risk factors**:
  - LightGBM v4 scorer: trained on TAMAR-corrected 100-question labels — may overfit to warmup patterns
  - Sidecar gate expansion: new behavior for single_field_single_doc queries — regression risk for simple single-doc questions
  - Cross-ref boosts: increases recall but may bring in false positives for multi-law enumeration questions
- **Private generalization**: Medium-high (scorer uses structural features, not content-specific)
- **TZUF integration**: ✅ Tested, 1113/1114 tests pass
- **Verdict**: **RECOMMENDED PRIMARY PRIVATE SUBMISSION**

### 1792_regime — SPECULATIVE UPSIDE
- **Expected**: Unknown delta vs v6; embedding improvement hypothesis only
- **Risk**: Requires full re-ingestion into separate Qdrant collections; no warmup score to compare
- **Blocker**: No collection data yet for private docs — requires full ingest pipeline run
- **Verdict**: Run ONLY if time allows AND v6 ingest completed first

### v7_1792_enhanced — DOUBLE CHALLENGER
- **Expected**: Best theoretical upside (better embeddings + better retrieval + scorer)
- **Risk**: Two independent untested improvements compound; debugging failures harder
- **Verdict**: Run only if v7_enhanced produces strong signal and time permits

### v8_full — KITCHEN SINK
- **Expected**: Highest theoretical ceiling but highest regression risk
- **Risk**:
  - segment_retrieval: unvalidated on private doc structure
  - doc_diversity: may inflate context with low-quality pages on private doc distributions
  - Combined with scorer: may amplify errors in both directions
- **Verdict**: Last to run if at all; only if v7_enhanced score is already strong

---

## Recommended Run Order

```
Priority 1: v7_enhanced        ← primary submission candidate
Priority 2: v6_regime          ← fallback / comparison anchor
Priority 3: v8_full            ← only if time + v7 looks good
Priority 4: 1792_regime        ← only if ingestion pipeline runs clean
Priority 5: v7_1792_enhanced   ← combination of 2+3, only if both individual runs succeeded
```

---

## Decision Criteria for Profile Selection

### Use v6_regime if:
- v7_enhanced shows regression vs warmup score in quick spot-check
- Sidecar or scorer produces obviously wrong answers (e.g., returns empty pages, wrong page IDs)
- Any telemetry errors appear in v7 run that are absent in v6

### Use v7_enhanced if:
- Initial spot-check shows no regressions on known-correct warmup questions
- Telemetry looks clean: page_ids populated, G values non-zero for single-doc questions
- TTFT is within acceptable bounds (< +20% vs v6 baseline)

### Use v8_full only if:
- v7_enhanced shows strong improvement (>3 pp vs v6 spot-check)
- No segment_retrieval errors in logs
- Still have 3+ hours remaining and at least 1 safe submission banked

---

## Quick A/B Comparison Protocol

1. Run v6_regime on first 30 questions of private set (sample)
2. Run v7_enhanced on same 30 questions
3. Compare:
   - G-score average (delta target: >0 pp)
   - Number of G=0 responses (target: fewer in v7)
   - Telemetry completeness (target: same or better)
   - TTFT mean (target: <+30% vs v6)
4. If v7 better on 3/4 criteria → proceed with v7 for full run
5. If v7 worse on any critical criterion → fall back to v6

---

## TAMAR Oracle Reference Data

From warmup 100-question analysis (quizzical-cerf branch):

| Failure type | Count | Profile that fixes it |
|---|---|---|
| Context miss (dual-case) | 7 | v7+ (OREV dual-case fix if implemented) |
| Context miss (same-doc multi-page) | 1 | v7 (multi-page inclusion) |
| Context miss (multi-doc) | 3 | v8 (doc-diversity expansion) |
| Retrieval miss | 6 | v7+ (prefetch 120 + cross-ref) |
| Grounding miss | 9 | v7 (scorer v4 + sidecar) |
| Unanswerable wrong | 4 | Not profile-dependent |
| Wrong answer (pipeline error) | 4 | SHAI prompt improvements |

Expected warmup G improvement by profile:
- v6 → v7: +7-12 pp (grounding + retrieval improvements)
- v6 → v8: +10-15 pp (all improvements) [but higher regression risk]

---

## Notes on TZUF Integration

TZUF (integration-testing branch 6f0b63c) fixed 2 critical bugs:
1. joblib missing from main deps → now added to pyproject.toml
2. v6 profiles silently enabled segment_retrieval + doc_diversity (defaults=True) → now explicitly =false in v6 profiles

**This means v6 profiles on the private machine need TZUF's fix deployed first.**
Verify: `private_v6_regime.env` has `PIPELINE_ENABLE_SEGMENT_RETRIEVAL` and `PIPELINE_ENABLE_DOC_DIVERSITY_EXPANSION` = false (added by TZUF).
