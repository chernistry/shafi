# Budget=2 Oracle Validation — Page Scorer Impact

**Date**: 2026-03-20 | **Model**: v6_version_full | **Data**: TAMAR-corrected labels (93 questions)

## Key Finding

Budget=2 delivers **+25.2pp mean F-beta 2.5** over budget=1 from the scorer alone.
This is the single most impactful change in the pipeline.

## Results

| Metric | Budget=1 | Budget=2 | Delta |
|--------|----------|----------|-------|
| Mean recall (all 93 q) | 0.439 | **0.752** | **+0.312** |
| Mean F-beta 2.5 (all) | 0.454 | **0.706** | **+0.252** |
| Mean recall (47 multi-gold q) | 0.359 | **0.700** | **+0.341** |
| Mean F-beta 2.5 (multi-gold) | 0.387 | **0.705** | **+0.318** |

## Recovery Statistics

- Questions where budget=2 recovers additional gold: **50/93**
- Questions where budget=1 already had all gold: 24/93
- Questions where both budgets find 0 gold: 14/93 (retrieval misses)
- Questions with >1 gold page: 47/93 (51%)

## Expected G-Score Impact

The scorer's F-beta 2.5 = 0.706 with budget=2 is the **upper bound from scoring alone**.
Actual G depends on evidence_selector recall floor and retrieval quality.

If we assume current G=0.801 and the scorer was the main bottleneck:
- Expected new G ≈ 0.801 × (0.706 / 0.454) ≈ ~0.93 (theoretical max)
- Realistic: +15-20pp G after accounting for retrieval misses and selector overhead

## Note

14 questions find 0 gold in either budget — these are retrieval failures, not scorer failures.
The evidence_selector recall floor (OREV task 5c) would help these.
