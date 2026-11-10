# Page Scorer Feature Impact Analysis

Generated: 2026-03-20

Models compared:
- **v3_logistic**: Logistic regression, 72 train questions, 12 features positive / 12 negative
- **v3_lgbm**: LightGBM, 72 train questions, 12 features (split-based importance)
- **v4_lgbm** (v4_tamar_augmented): LightGBM, 74 train questions (+2 from label corrections), 12 features

---

## 1. Top 10 Features by Importance (v4 LightGBM)

| Rank | Feature | Importance | Description | Version |
|------|---------|------------|-------------|---------|
| 1 | `sidecar_rank_normalized` | 49 | Sidecar retrieval rank / total retrieved pages (0-1) | r3 |
| 2 | `page_position_ratio` | 47 | Page number / total candidate pages for the query | r3 |
| 3 | `sidecar_retrieved_rank` | 45 | Raw rank from sidecar embedding retrieval | r1 |
| 4 | `query_content_jaccard` | 43 | Jaccard token similarity between query and page snippet | r3 |
| 5 | `query_token_count` | 32 | Number of tokens in the query | r2 |
| 6 | `doc_rank` | 31 | 1-based document rank within the candidate set | r1 |
| 7 | `page_num` | 31 | Raw page number within the document | r1 |
| 8 | `answer_char_count` | 20 | Character length of the golden answer | r2 |
| 9 | `from_legacy_cited` | 20 | Whether page came from legacy citation extraction | r2 |
| 10 | `legacy_cited_rank` | 20 | Rank among legacy-cited pages | r2 |

**Notable**: `legal_keyword_density` (importance=17) ranks 11th, and `doc_candidate_count` (14) ranks 12th. The three v4-specific features (`article_ref_match`, `page_is_early_in_doc`, `legal_keyword_density`) appear in the top 12 only via `legal_keyword_density`. The tree-based model may be absorbing `article_ref_match` and `page_is_early_in_doc` through interactions with existing features rather than giving them standalone split importance.

---

## 2. Feature Evolution Across Models

### Ranking shifts (top features by model)

| Feature | v3_logistic | v3_lgbm | v4_lgbm | Trend |
|---------|-------------|---------|---------|-------|
| `query_content_jaccard` | #1 (5.17) | #1 (54) | #4 (43) | Declining as model gets richer signals |
| `sidecar_rank_normalized` | Negative (-2.11) | #2 (43) | #1 (49) | Logistic penalized it; LightGBM exploits it |
| `page_position_ratio` | #6 (0.86) | #4 (38) | #2 (47) | Steady rise; strong interaction feature |
| `from_sidecar_retrieved` | #2 (2.62) | Not in top 12 | Not in top 12 | Binary flag subsumed by continuous rank |
| `is_first_page` | #3 (2.56) | #11 (17) | Not in top 12 | Replaced by `page_is_early_in_doc` (v4) |
| `page_num` | Not in top 12 | #3 (41) | #7 (31) | Tree model exploits raw position |
| `sidecar_retrieved_rank` | Not in top 12 | #5 (32) | #3 (45) | Rose sharply in v4 |
| `legacy_context_rank` | Not in top 12 | #8 (20) | Not in top 12 | Dropped, likely collinear with cited rank |
| `legal_keyword_density` | N/A | N/A | #11 (17) | New in v4, immediate value |

### Key transitions

**v3_logistic -> v3_lgbm**: The logistic model relied heavily on binary provenance flags (`from_sidecar_retrieved`, `is_first_page`, `from_legacy_used`). The LightGBM shift unlocked continuous features: `query_content_jaccard` stayed #1 but continuous rank/position features surged. Notably, `sidecar_rank_normalized` flipped from the strongest negative weight (-2.11) to #2 positive importance -- the logistic model could not model the nonlinear "low rank = good" relationship.

**v3_lgbm -> v4_lgbm**: Two changes: (a) 3 new features added, (b) 31 label corrections from TAMAR review + 2 new questions. The corrected labels shifted `sidecar_rank_normalized` to #1 (from #2), `sidecar_retrieved_rank` rose from #5 to #3, and `is_first_page` dropped out entirely (replaced by `page_is_early_in_doc`). The label corrections likely amplified the learning signal for article-level pages that are not page 1 but still early in documents.

### Metric progression

| Metric | v3_logistic | v3_lgbm | v4_lgbm | Delta (v3_lgbm -> v4) |
|--------|-------------|---------|---------|----------------------|
| dev_hit@1 | 0.632 | 0.684 | 0.632 | -0.053 |
| dev_hit@2 | 0.842 | 0.789 | 0.789 | 0.000 |
| dev_MRR | 0.739 | 0.757 | 0.756 | -0.001 |
| dev_F2.5 | 0.686 | 0.722 | 0.761 | **+0.039** |
| cv_mean_hit@1 | 0.247 | 0.250 | 0.289 | **+0.039** |
| cv_mean_MRR | 0.261 | 0.262 | 0.312 | **+0.050** |
| cv_mean_F2.5 | 0.731 | 0.760 | 0.778 | **+0.018** |
| cv_std_F2.5 | 0.058 | 0.091 | 0.060 | **-0.031** |

**v4 is the clear winner on cross-validation stability**: CV MRR improved +0.050, CV hit@1 improved +0.039, and CV F2.5 std dropped from 0.091 to 0.060 (much less fold variance). The dev_F2.5 improvement (+0.039) confirms the recall-oriented objective improved. The dev_hit@1 drop is not concerning because the model is optimizing for recall (F2.5), not precision.

---

## 3. Features Contributing to Article-Level Recovery

### The failure mode

TAMAR's analysis identified 16 `article_level_provision_miss` cases. Of these, 15 have `retrieval_gap_type: gold_page_within_window_but_skipped` (1 has null -- no verified gold pages). In all 15, the gold page was retrieved by the sidecar but the scorer ranked it below the selection threshold.

Typical pattern: query asks about "Article 34(1) of the General Partnership Law" -- the gold page (e.g., page 11) is in the retrieval window alongside pages 10, 12, 16, but the scorer preferred other pages. The gold page contains the specific article provision but may have lower embedding similarity than surrounding pages.

### v4 features targeting this failure mode

| Feature | Mechanism | Coverage of 15 cases |
|---------|-----------|---------------------|
| `article_ref_match` | Matches full article references like "Article 14(2)(b)" from query in page content via regex `Article \d+(\(\d+\))?(\([a-z]\))?` | High -- all 15 cases have explicit article references in the query |
| `page_is_early_in_doc` | Boolean: page_num <= 30% of max_page. Many article provisions in DIFC laws cluster in the first third | Moderate -- depends on law structure; long laws with articles past 30% will not benefit |
| `legal_keyword_density` | Count of legislative terms (shall, must, entitled, prohibited, pursuant, notwithstanding, herein, thereof) in snippet | High -- article provisions are dense with these terms vs. schedules/title pages |

### Label correction amplification

The v4 training set includes 31 questions with corrected page labels from TAMAR's review. These corrections:
- Fixed false positives: pages scored as gold that were actually neighbors of the true gold page
- Fixed false negatives: true gold pages marked as non-gold because they were not in the LLM's used_pages
- Added 2 new questions (74 vs 72 train questions)

Impact on the article-level failure mode: corrected labels gave the model clean positive signal for the exact article page rather than noisy signal from adjacent pages. This explains why `sidecar_retrieved_rank` importance surged (+13, from 32 to 45) -- with correct labels, rank discrimination within a retrieval window becomes more learnable.

### Remaining gap

Even with v4 features, the article-level scorer is limited by the candidate set. In 1 of the 16 cases, the gold document was not even retrieved (`gold_doc_in_retrieved_set: false`). This is a prefetch problem, not a scorer problem. The 15 scorer-addressable cases represent the ceiling for page scorer improvements on this failure category.

---

## 4. Potential Features for Private Data

| Feature | Expected Value | Complexity | Overfit Risk | Priority |
|---------|---------------|------------|-------------|----------|
| `embedding_score` | **High**. Direct cosine similarity from the embedding model. Currently only used for retrieval cutoff, not as a scorer feature. Would give the LightGBM model a continuous relevance signal independent of rank. | Low -- score is already computed at retrieval time, just needs to be passed through `PageCandidateRecord`. | Low -- embedding similarity is law-agnostic and generalizes well. | **#1** |
| `doc_type_match` | **Medium**. Whether doc type (regulation, case law, contract, amendment) matches query's expected document type. Could help the "wrong_document_miss" category (4 cases). | Medium -- needs a document type classifier or metadata mapping. | Medium -- doc type distribution may shift on private data. | #3 |
| `cross_reference_count` | **Medium**. Number of cross-references to other documents/articles/sections on the page. Pages with many cross-references tend to be definitional or structural, not substantive. | Medium -- needs regex extraction for "see Article X", "pursuant to Law Y" patterns. | Low -- cross-reference patterns are structural, not content-dependent. | #4 |
| `page_role_match` | **Medium-High**. Whether the page's structural role (title_cover, article_clause, schedule_table, etc.) matches the query's expected target role from `target_page_roles`. Already partially captured by `targets_*` features but not as a per-page match. | Low-Medium -- roles are already extracted; need per-page role detection. | Medium -- role distribution may differ. | #2 |
| `temporal_relevance` | **Low-Medium**. Whether page mentions dates/years relevant to query (e.g., "2022 amendment"). Useful for amendment-law questions. | Medium -- needs date extraction from both query and page content. | High -- specific dates in private data will be entirely different. Needs to match structural patterns (amendment year, enactment date) not specific values. | #5 |

---

## 5. Private Data Distribution Shift Risk

### High-risk features (may behave differently on private data)

| Feature | Risk | Reason |
|---------|------|--------|
| `scope_mode=*` (categorical) | **Medium-High** | Private questions may introduce scope patterns not seen in warmup. The logistic model had 4 scope_mode features with negative weights; tree models handle unseen categories as missing but learned splits may not generalize. |
| `page_num` (raw) | **Medium** | Private docs may have very different page counts. A law with 200 pages vs warmup's typical 20-50 will shift the raw page_num distribution. Mitigated by `page_position_ratio` which normalizes. |
| `article_ref_match` | **Medium** | Private docs may use different article numbering conventions (Schedule X, Rule Y, Regulation Z). The regex `Article \d+` will miss non-Article references. |
| `is_first_page` / `page_is_early_in_doc` | **Medium** | Different document structures -- some private docs may front-load substantive content, others may have long preambles. |
| `answer_char_count` | **Medium-High** | This uses golden answer length which is NOT available at runtime for private data (only at training time). If used at inference with predicted/empty answer length, behavior will be very different. |

### Stable features (law-agnostic, should generalize)

| Feature | Why Stable |
|---------|-----------|
| `query_content_jaccard` | Token overlap is content-agnostic. Works for any legal domain. |
| `sidecar_rank_normalized` | Normalized to [0,1]. Embedding model rank is relative, not absolute. |
| `sidecar_retrieved_rank` | Ordinal rank; distribution is bounded by retrieval depth. |
| `doc_rank` | Relative ordering within candidate set. |
| `legal_keyword_density` | Legislative keywords (shall, must, pursuant, etc.) are universal in legal text. |
| `from_legacy_cited` / `legacy_cited_rank` | Provenance flags depend on pipeline behavior, not document content. |
| `page_position_ratio` | Normalized to [0,1]; structure-invariant. |

### Critical note on `answer_char_count`

This feature (rank #8, importance=20) uses golden answer text length. At inference time for private data, the golden answer is unknown. If the inference pipeline passes 0 or a proxy value, the model's learned splits on answer length will fire incorrectly. This needs investigation -- either:
1. Remove it from the feature set (safest, loses some signal)
2. Replace with a predicted answer length proxy from the query classifier
3. Verify the inference code path handles this correctly

---

## 6. Recommendations

### Ranked by expected G-score impact x confidence x safety

**1. Add `embedding_score` as a scorer feature** (Priority: IMMEDIATE)
- Expected impact: +0.5-1.5% G-score. The embedding model's cosine similarity is the single strongest relevance signal we have but the scorer currently only sees rank (ordinal) not score (continuous). Within-rank score differences would help discriminate between close candidates -- exactly the article-level failure mode.
- Confidence: High. This is a proven signal already computed at retrieval time.
- Safety: Very high. Embedding similarity is fully law-agnostic.
- Implementation: Pass `embedding_score` through `PageCandidateRecord`, add to `build_page_feature_dict`, retrain.
- RAEI: 0.85

**2. Audit `answer_char_count` runtime behavior on private data** (Priority: IMMEDIATE)
- Expected impact: Prevents potential -1-3% G-score regression if this feature fires incorrectly at inference.
- Confidence: High that this is a risk; needs code path verification.
- Safety: Defensive fix, no downside.
- Implementation: Trace inference code path for `answer_char_count` value. Either gate it behind a training-only flag or replace with query-derived proxy.
- RAEI: 0.80

**3. Expand `article_ref_match` regex to cover non-Article references** (Priority: HIGH)
- Expected impact: +0.3-0.8% G-score. Private data likely includes Schedules, Rules, Regulations, Sections not just Articles. Current regex only matches `Article \d+`.
- Confidence: Medium-high. The 15 article-level cases show this feature targets a real failure mode; broadening coverage increases generalization.
- Safety: High. Regex expansion is narrow and testable.
- Implementation: Extend `_ARTICLE_FULL_RE` to match `(Article|Section|Rule|Regulation|Schedule|Part|Clause)\s+\d+...`. Retrain scorer.
- RAEI: 0.65
