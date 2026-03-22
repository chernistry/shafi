# V6 Page Scorer: Version-Confusion Features Audit for Private Data

Audit date: 2025-03-20
Scorer artifact: `models/page_scorer/v6_version_full/`
Feature policy: `runtime_safe_r3`
Source files: `training_scaffold.py` (lines 370-377), `page_scorer_runtime.py` (lines 349-356)

---

## 1. Current Version-Confusion Features

Four features were introduced under the "v5 features: version confusion disambiguation" label:

| Feature | Type | In top-12 importance? | Weight |
|---|---|---|---|
| `doc_pages_in_candidates` | int | Yes | 27 |
| `same_page_num_competitors` | int | No | -- |
| `query_law_name_in_content` | bool->int | No | -- |
| `is_amendment_content` | bool->int | No | -- |

Only `doc_pages_in_candidates` has learned weight in v6. The other three version-confusion features are present in the feature vector but have zero importance in the top-12. They may have small non-zero importance below the reporting cutoff, or truly zero splits.

### 1a. `query_law_name_in_content`

**Regex** (identical in both files):
```python
_LAW_NAME_RE = re.compile(
    r"\b(?:the\s+)?([A-Z][A-Za-z\s]+(?:Law|Regulations?|Rules?|Order))\s+\d{4}\b",
)
```

**Logic**: Extract law names from the query via `_LAW_NAME_RE.findall(query)`, then check if `name.strip().lower() in content_lower` for any extracted name.

**BUG FOUND -- Regex captures leading context words.** The `\b` anchor matches any word boundary, so when the query is "Under the Trust Law 2018", the regex matches starting at "Under" (a word boundary), the optional `(?:the\s+)?` fails (because "Under" != "the"), and the capture group `[A-Z][A-Za-z\s]+` greedily consumes "Under the Trust Law". Result:

| Query | Captured name | `name.lower()` | Works? |
|---|---|---|---|
| `"Companies Law 2018"` | `Companies Law` | `companies law` | YES |
| `"the Trust Law 2018"` | `Trust Law` | `trust law` | YES |
| `"Under the Trust Law 2018, Article 28(4)..."` | `Under the Trust Law` | `under the trust law` | **NO** -- will not match page content |
| `"What does the Trust Law 2018 say about..."` | `What does the Trust Law` | `what does the trust law` | **NO** |
| `"According to the Companies Law 2018"` | `According to the Companies Law` | `according to the companies law` | **NO** |
| `"Is the IP Law 2019 applicable?"` | `Is the IP Law` | `is the ip law` | **NO** |
| `"the Employment Law 2019 provides..."` | `Employment Law` | `employment law` | YES |
| `"per the Insolvency Law 2019..."` | `Insolvency Law` | `insolvency law` | YES |

The feature only fires correctly when (a) the law name starts the query, or (b) "the" immediately precedes the law name at a word boundary with no prior capitalized word in the same "run". In typical question phrasing ("Under the...", "What does the...", "According to the..."), the regex captures garbage prefixes. This likely explains why the feature has zero importance -- it rarely fires correctly.

### 1b. `is_amendment_content`

**Markers**: `("amendment", "amending", "amended")`
**Logic**: `any(marker in content_lower for marker in _AMENDMENT_MARKERS)`

This is a simple substring check on lowercased page content. No regex, no query dependency.

- "The Companies Law Amendment Law 2020" -> lowered content contains "amendment" -> **fires correctly**
- Any page discussing amendments, amending provisions, or amended sections -> fires correctly
- False positives: pages that mention amendments in passing (e.g., "this law was amended by...") will also fire, but that is acceptable signal

**Verdict**: Works correctly. Simple and robust. Zero importance in v6 likely means warmup data did not have enough amendment-vs-original version confusion cases for the tree to learn useful splits.

### 1c. `same_page_num_competitors`

Counts how many other documents have a candidate page with the same page number. Useful when e.g., page 5 of "Employment Law 2005" and page 5 of "Employment Law 2019 Amendment" are both candidates.

**Verdict**: Correct implementation. Zero importance likely means the signal is weak or correlated with `doc_pages_in_candidates`.

### 1d. `doc_pages_in_candidates`

Counts how many pages from the same document appear in the candidate set. High values indicate a heavily-retrieved document.

**Verdict**: Working and important (weight 27, rank 6 of 12). Generalizes well to private data.

---

## 2. Predicted Private Data Laws

| Law | In warmup? | Amendment risk | `_LAW_NAME_RE` match | `is_amendment_content` fires? | Risk |
|---|---|---|---|---|---|
| Companies Law 2018 (DIFC Law No. 5 of 2018) | No | High -- likely has amendment laws | Yes (`Companies Law`) | Yes on amendment docs | **Regex bug**: queries like "Under the Companies Law 2018" will capture "Under the Companies Law" and fail the content check |
| Trust Law 2018 (DIFC Law No. 4 of 2018) | Yes | Moderate | Yes (`Trust Law`) | Depends on content | Same regex bug risk |
| IP Law 2019 | No | Low | Yes (`IP Law`) -- `[A-Z]` matches `I`, `[A-Za-z\s]+` matches `P Law` | Unlikely | Low risk; short name "IP" is unusual but regex handles it |
| Insolvency Law 2019 | No | Moderate | Yes (`Insolvency Law`) | If amendments exist | Regex bug applies to typical question phrasings |
| Employment Law 2019 | Yes (2 versions: original + amendment) | High -- known two-version case | Yes (`Employment Law`) | Yes on amendment version | Best-tested case from warmup; version confusion is a real retrieval failure mode here |

### Per-law notes

**Companies Law 2018**: DIFC's company formation law. Likely paired with amendment laws (common for DIFC). The regex bug means `query_law_name_in_content` will fail on most natural-language questions. However, since this feature has zero importance in v6, the practical impact is nil -- the model does not use it.

**Trust Law 2018**: Already in warmup. The model has seen examples. Low incremental risk.

**IP Law 2019**: Two-character abbreviation "IP" is uncommon but the regex handles it (matches `[A-Z]` then `[A-Za-z\s]+`). No known amendment structure. Low risk.

**Insolvency Law 2019**: Unknown structure. If DIFC published amendments, version confusion is possible. Same regex bug applies but same zero-importance mitigation.

**Employment Law 2019**: Best-known case. Two versions exist in warmup data. The model has already been trained on this version-confusion scenario. The `doc_pages_in_candidates` feature (weight 27) provides the main signal here, not the broken `query_law_name_in_content`.

---

## 3. Regex Coverage Analysis

Pattern: `\b(?:the\s+)?([A-Z][A-Za-z\s]+(?:Law|Regulations?|Rules?|Order))\s+\d{4}\b`

### What it matches (standalone names)

| Input | Match | Captured |
|---|---|---|
| `Companies Law 2018` | Yes | `Companies Law` |
| `IP Law 2019` | Yes | `IP Law` |
| `DIFC Insolvency Regulations 2019` | Yes | `DIFC Insolvency Regulations` |
| `The Strata Title Law 2018` | Yes | `The Strata Title Law` (includes "The") |
| `Data Protection Law 2020` | Yes | `Data Protection Law` |
| `Trust Law 2018` | Yes | `Trust Law` |
| `Employment Law 2019` | Yes | `Employment Law` |
| `Insolvency Law 2019` | Yes | `Insolvency Law` |
| `Operating Law 2004` | Yes | `Operating Law` |

### Systematic failure: leading context capture

The root cause: `[A-Z][A-Za-z\s]+` is greedy and `\s` includes spaces. When preceded by any capitalized word ("Under", "What", "According", "Is"), the capture group consumes everything from that word through the law name.

**Fix** (not yet applied): Use a more restrictive pattern that anchors on the law-type suffix:

```python
_LAW_NAME_RE = re.compile(
    r"\b(?:the\s+)?((?:[A-Z][A-Za-z]*\s+)*(?:Law|Regulations?|Rules?|Order))\s+\d{4}\b",
)
```

Or extract law names by finding the suffix keyword and walking backward. However, since the feature currently has zero importance, this fix has no score impact on v6.

### Edge cases

- Acronyms with periods ("I.P. Law 2019") -- not matched (periods break `[A-Za-z\s]+`)
- Hyphenated names ("Anti-Money Laundering Rules 2019") -- not matched (hyphen breaks the pattern)
- All-caps headers ("COMPANIES LAW 2018") -- not matched (`[A-Z][A-Za-z\s]+` requires mixed case after first char, but "OMPANIES" is all uppercase which does match `[A-Za-z]`)

Verified: "COMPANIES LAW 2018" does match because `[A-Za-z]` accepts uppercase letters. The regex is case-sensitive only on the first character (`[A-Z]`).

---

## 4. Risks and Recommendations

### Risks

1. **Regex bug is latent, not active.** `query_law_name_in_content` has zero learned importance in v6. The model effectively ignores it. The bug causes no current score harm, but it means a potentially useful feature is wasted.

2. **Private data may introduce new version-confusion scenarios.** If Companies Law 2018 appears alongside a Companies Law Amendment Law 2020, the model relies on `doc_pages_in_candidates` (weight 27) and general retrieval signals to disambiguate, not on the dedicated version-confusion features.

3. **Amendment detection is robust but unused.** `is_amendment_content` works correctly but has zero importance. This may change with private data if amendment documents are more prevalent.

4. **No "law year mismatch" feature exists.** There is no feature that compares the year in the query ("Trust Law 2018") against the year in the document metadata. This could be a high-value feature for version confusion.

### Recommendations

1. **Do NOT fix the regex before private eval.** The feature has zero importance, so fixing it changes nothing in v6 scores. Risk of introducing regressions outweighs benefit. Classification: F (reject).

2. **After private eval, consider adding a year-match feature.** Extract the year from the query law name and compare against document metadata year. This would directly address version confusion. Classification: B (narrow feature change), for post-private-eval retrain only.

3. **The law-agnostic design generalizes well.** All features use regex/substring matching, not law-name lookup tables. New laws (IP Law, Insolvency Law) are handled automatically. No hardcoded law names.

4. **Retrain with private labels using fast-retrain script.** After private data evaluation, the version-confusion features may gain importance with more training examples. Use `scripts/train_page_scorer.py` for fast retrain with updated labels.

5. **Monitor `doc_pages_in_candidates` on private data.** This is the only active version-confusion signal (weight 27). If private data has more multi-version documents, this feature's importance should increase.

---

## Summary

The v6 page scorer has four version-confusion features. Only `doc_pages_in_candidates` has learned importance. The three finer-grained features (`query_law_name_in_content`, `is_amendment_content`, `same_page_num_competitors`) have zero importance, partly because `query_law_name_in_content` has a regex bug that causes it to rarely fire correctly, and partly because warmup data has limited version-confusion examples. The features are law-agnostic and will generalize to predicted private laws. No action needed before private eval; retrain afterward with corrected labels.
