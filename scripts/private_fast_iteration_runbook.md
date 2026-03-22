# Private Dataset Fast Iteration Runbook

**Owner**: EYAL (page scorer / ML)
**Time budget**: <5 minutes total
**Validated**: 2026-03-20 on warmup data

## Prerequisites

- `uv` environment with lightgbm installed
- Base training data at `data/derived/grounding_ml/v2_reviewed/train.jsonl`
- Current best model at `models/page_scorer/v4_tamar_augmented/page_scorer.joblib`

## Step 1: Analyze Private Dataset (~30s)

```bash
python scripts/private_dataset_intelligence.py \
  --dataset-dir dataset_private/ \
  --output-dir analysis/private/
```

**Outputs** (check all 4):
- `document_profile.json` — per-document entity counts, types, warmup similarity
- `query_distribution_report.json` — per-question scope, complexity, answer type
- `distribution_shift_report.md` — READ THIS FIRST for distribution shift vs warmup
- `synthetic_training_triples.jsonl` — weak-supervision triples for retraining

**Key signals in distribution_shift_report.md**:
- New document families not seen in warmup → may need retrieval tuning
- Answer type shift (more boolean = more G-score pressure)
- Scope mode shift (more compare_pair = sidecar helps, more single_field = doesn't)

## Step 2: Run First Eval

Run the pipeline on private data. Collect raw results JSON.

## Step 3: Build Target Map from Eval Failures

If TAMAR produces a `retrieval_target_map.json` from eval failures:
```json
[{"question_id": "...", "verified_gold_pages": ["doc_page1"], "used_pages": ["doc_page2"], "failure_category": "..."}]
```

## Step 4: Retrain Page Scorer (~0.5s)

```bash
python scripts/retrain_page_scorer_on_private.py \
  --base-train-jsonl data/derived/grounding_ml/v2_reviewed/train.jsonl \
  --eval-target-map scripts/private_target_map.json \
  --base-model models/page_scorer/v4_tamar_augmented/page_scorer.joblib \
  --output-dir models/page_scorer/v5_private/
```

**Metric gates**:
- `train_hit@1` >= 0.60 (else label corrections may be wrong)
- `dev_fbeta_2_5` >= 0.70 (else model is regressing)

## Step 5: Deploy Retrained Model

```bash
export PIPELINE_TRAINED_PAGE_SCORER_MODEL_PATH=models/page_scorer/v5_private/page_scorer.joblib
export PIPELINE_ENABLE_TRAINED_PAGE_SCORER=true
export PIPELINE_ENABLE_GROUNDING_SIDECAR=true
```

## Timing Reference (warmup validation)

| Step | Duration |
|------|----------|
| Intelligence analysis | 1.2s |
| Retrain page scorer | 0.3s |
| Total ML pipeline | **<2s** |

## Known Limitations

1. **Sidecar gating**: Trained scorer only activates for compare/full-case/negative scopes. single_field_single_doc (65% of warmup) bypasses the scorer. Requires evidence_selector.py changes.
2. **Training data size**: Only 100 warmup questions. Watch CV stability (std < 0.10).

## Rollback

```bash
# Revert to v4
export PIPELINE_TRAINED_PAGE_SCORER_MODEL_PATH=models/page_scorer/v4_tamar_augmented/page_scorer.joblib
# Or disable entirely
export PIPELINE_ENABLE_TRAINED_PAGE_SCORER=false
```
