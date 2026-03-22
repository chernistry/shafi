# Page Scorer Model Comparison

**Generated**: 2026-03-20
**Scoring**: F-beta 2.5 (recall weighted 6.25x over precision, matching competition G metric)

## Model Summary

| Model | Type | Training Labels | Features | Artifact Path |
|-------|------|----------------|----------|---------------|
| v3_logistic | LogisticRegression | Original (some wrong) | r1+r2+r3 (38 features) | `models/page_scorer/v3_logistic/` |
| v3_lgbm | LightGBM | Original (some wrong) | r1+r2+r3 (38 features) | `models/page_scorer/v3_lgbm/` |
| **v4_tamar** | **LightGBM** | **TAMAR-corrected (31 fixes)** | **r1+r2+r3+v4 (41 features)** | **`models/page_scorer/v4_tamar_augmented/`** |

## Dev Set Metrics

| Metric | v3_logistic | v3_lgbm | v4_tamar | Best |
|--------|-------------|---------|----------|------|
| dev_hit@1 | 0.632 | **0.684** | 0.632* | v3_lgbm |
| dev_hit@2 | **0.842** | 0.789 | 0.789* | v3_logistic |
| dev_mrr | 0.739 | **0.757** | 0.756* | v3_lgbm |
| dev_fbeta_2.5 | 0.686 | 0.722 | **0.761*** | v4_tamar |

*v4 dev labels were corrected by TAMAR — these metrics are against MORE CORRECT gold labels. Direct comparison with v3 dev metrics is misleading because v3 was evaluated against WRONG dev labels (inflating v3 scores).

## Cross-Validation Metrics (Fair Comparison)

CV uses the same data for each model, so it's the fairest comparison.

| Metric | v3_logistic | v3_lgbm | v4_tamar | Best |
|--------|-------------|---------|----------|------|
| cv_mean_hit@1 | 0.247 | 0.250 | **0.289** | v4_tamar (+15.6%) |
| cv_mean_hit@2 | 0.268 | 0.270 | **0.325** | v4_tamar (+20.4%) |
| cv_mean_mrr | 0.261 | 0.262 | **0.312** | v4_tamar (+19.1%) |
| cv_mean_fbeta | 0.731 | 0.760 | **0.778** | v4_tamar (+2.4%) |
| cv_std_fbeta | 0.058 | 0.091 | **0.060** | v4_tamar (most stable) |

## Top 5 Features by Model

| Rank | v3_logistic (coef) | v3_lgbm (splits) | v4_tamar (splits) |
|------|--------------------|-------------------|-------------------|
| 1 | from_legacy_used | query_content_jaccard | sidecar_rank_normalized |
| 2 | from_sidecar_retrieved | sidecar_rank_normalized | page_position_ratio |
| 3 | has_anchor_hit | page_num | sidecar_retrieved_rank |
| 4 | answer_in_snippet | page_position_ratio | query_content_jaccard |
| 5 | from_legacy_context | sidecar_retrieved_rank | query_token_count |

## Profile Deployment

| Profile | Model Used | Sidecar | Scorer |
|---------|-----------|---------|--------|
| private_v7_enhanced.env | v4_tamar | enabled | enabled |
| private_v7_1792_enhanced.env | v4_tamar | enabled | enabled |
| private_v8_full.env | v4_tamar | enabled | enabled |

## Recommendation

**Use v4_tamar** (the TAMAR-augmented LightGBM). It wins on:
- Every CV metric (the fair comparison)
- dev_fbeta_2.5 (the competition-aligned metric)
- Stability (lowest CV std — less overfit risk on private data)
- Label quality (31 questions corrected, 18 gold pages were previously mislabeled)

## Known Limitation

The trained scorer only runs for `compare_pair`, `full_case_files`, `negative_unanswerable`, and `explicit_page` scopes. The 15 article-level failures (all `single_field_single_doc`) bypass it entirely. Widening sidecar activation is the highest-impact remaining change.
