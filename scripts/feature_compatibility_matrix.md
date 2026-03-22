# Feature Compatibility Matrix
Generated: 2026-03-20 by TZUF (Phase 3 update — post all-agent merge)
Last verified: 1238/1238 tests passing, all profiles validated.

## Feature Flag Registry

All flags live in `PipelineSettings` (`src/rag_challenge/config/settings.py`, env prefix `PIPELINE_`).
All Qdrant settings in `QdrantSettings` (env prefix `QDRANT_`).

### Pipeline Flags — Post-v6, Default True (HIGH RISK — must be explicit in v6 profiles)

| Flag | Default | Env Var | Owner | Status |
|------|---------|---------|-------|--------|
| `enable_cross_ref_boosts` | `True` | `PIPELINE_ENABLE_CROSS_REF_BOOSTS` | OREV | Active in v7+ |
| `enable_segment_retrieval` | `True` | `PIPELINE_ENABLE_SEGMENT_RETRIEVAL` | OREV | Active in v8 |
| `enable_doc_diversity_expansion` | `True` | `PIPELINE_ENABLE_DOC_DIVERSITY_EXPANSION` | OREV | Active in v8 |
| `enable_grounding_sidecar` | `True` | `PIPELINE_ENABLE_GROUNDING_SIDECAR` | OREV/EYAL | Active in v7+ |
| `enable_trained_page_scorer` | `True` | `PIPELINE_ENABLE_TRAINED_PAGE_SCORER` | EYAL | Active in v7+ |
| `enable_answer_validation` | `True` | `PIPELINE_ENABLE_ANSWER_VALIDATION` | NOGA | Active in v7+ |

### Pipeline Flags — Default False (SAFE — no override needed in v6 profiles)

| Flag | Default | Env Var | Owner | Status |
|------|---------|---------|-------|--------|
| `enable_answer_consensus` | `False` | `PIPELINE_ENABLE_ANSWER_CONSENSUS` | NOGA | Off everywhere |
| `enable_bridge_fact_retrieval` | `False` | `PIPELINE_ENABLE_BRIDGE_FACT_RETRIEVAL` | OREV | Off |
| `enable_retrieval_escalation` | `False` | `PIPELINE_ENABLE_RETRIEVAL_ESCALATION` | OREV | Off |
| `page_first_enabled` | `False` | `PIPELINE_PAGE_FIRST_ENABLED` | OREV | Off |
| `enable_shadow_search_text` | `False` | `PIPELINE_ENABLE_SHADOW_SEARCH_TEXT` | OREV | Off |
| `enable_parallel_anchor_retrieval` | `False` | `PIPELINE_ENABLE_PARALLEL_ANCHOR_RETRIEVAL` | OREV | Off |
| `enable_entity_boosts` | `False` | `PIPELINE_ENABLE_ENTITY_BOOSTS` | OREV | Off |
| `enable_conflict_detection` | `False` | `PIPELINE_ENABLE_CONFLICT_DETECTION` | OREV | Off |
| `enable_multi_hop` | `False` | `PIPELINE_ENABLE_MULTI_HOP` | OREV | Off |

## Profile Feature State — VERIFIED 2026-03-20 (shell source + settings load)

| Profile | prefetch | rerank | cross_ref | segment | diversity | sidecar | scorer | model | validation |
|---------|----------|--------|-----------|---------|-----------|---------|--------|-------|------------|
| `private_v6_regime.env` | 120/120 | 80 | ✗ | ✗ | ✗ | ✗ | ✗ | OFF | ✗ |
| `private_1792_regime.env` | 120/120 | 80 | ✗ | ✗ | ✗ | ✗ | ✗ | OFF | ✗ |
| `private_v7_enhanced.env` | 120/120 | 120 | ✓ | ✓ | ✓ | ✓ | ✓ | **v7_regex_fixed** | ✓ |
| `private_v7_1792_enhanced.env` | 120/120 | 120 | ✓ | ✓ | ✓ | ✓ | ✓ | **v7_regex_fixed** | ✓ |
| `private_v8_full.env` | 120/120 | 120 | ✓ | ✓ | ✓ | ✓ | ✓ | **v7_regex_fixed** | ✓ |

**Notes**:
- v6/1792 profiles do NOT set `QDRANT_PREFETCH_DENSE/SPARSE` — they use the Python default of 120. This gives better recall than original v6 (which used 60), but with same conservative feature set.
- All v7+ profiles use `v6_version_full` page scorer (EYAL's latest, trained with version-aware features, dev_fbeta best of all versions).
- `answer_consensus=False` in ALL profiles (async LLM calls hurt TTFT; not yet validated).

## LightGBM Page Scorer Status (EYAL)

| Version | Path | Key Features | dev_hit@2 / dev_hit@1 | Status |
|---------|------|-------------|----------|--------|
| v3 | `models/page_scorer/v3_lgbm/` | baseline | — | Available, not used |
| v4 (TAMAR-augmented) | `models/page_scorer/v4_tamar_augmented/` | TAMAR-corrected labels | — | Available, not used |
| v5 (version-aware) | `models/page_scorer/v5_version_aware/` | version features | — | Available, not used |
| v6 (prev) | `models/page_scorer/v6_version_full/` | law-name + amendment detection | 0.895 / 0.631 | Superseded by v7 |
| **v7** (current) | `models/page_scorer/v7_regex_fixed/` | v6 + fixed law-name regex | 0.895 / **0.684** (+5.3pp) | **Active in v7/v8 profiles** |

All v7 scorer model files verified present. v7 profiles updated by EYAL (commit 407ca85).

## page_budget by Query Scope (query_scope_classifier.py)

| Scope | page_budget | Notes |
|-------|-------------|-------|
| `EXPLICIT_PAGE` | 1 | Query explicitly cites a page number |
| `NEGATIVE_UNANSWERABLE` | 0 | Should return null/unavailable |
| `FULL_CASE_FILES` | 4 | Full document/case scope queries |
| `COMPARE_PAIR` | 2 | Cross-document comparison |
| `SINGLE_FIELD_SINGLE_DOC` (all variants) | 2 | Majority of queries — EYAL's critical fix |
| `BROAD_FREE_TEXT` | 2 | Default fallback |

**Key change from v6**: page_budget bumped from 1→2 for single-doc queries (OREV task 5a, 2026-03-20). This is a CODE change (hardcoded), affects ALL profiles equally. Expected: +17pp hit rate on multi-page gold answers.

## Dependencies Between Features

| Feature | Requires | Notes |
|---------|----------|-------|
| `enable_cross_ref_boosts` | None | Independent |
| `enable_segment_retrieval` | `qdrant.segment_collection` populated | Gracefully skipped if collection empty |
| `enable_doc_diversity_expansion` | None | Post-retrieval diversity pass |
| `enable_grounding_sidecar` | `qdrant.page_collection` + `qdrant.support_fact_collection` | Requires page collection indexed |
| `enable_trained_page_scorer` | `pipeline.trained_page_scorer_model_path` file exists | Fail-closed: skips gracefully if model missing |
| `enable_answer_validation` | None | Synchronous; flag-gated inside `run_answer_quality_gate()` |
| `enable_answer_consensus` | Async LLM (generator) | Requires generator instance; async only |

## CRITICAL: Known Broken Default

- **Setting**: `trained_page_scorer_model_path` (line 319 in settings.py)
- **Default**: `.sdd/researches/627_runtime_safe.../page_scorer.joblib` (gitignored path)
- **Impact**: If default settings used without a profile, `enable_trained_page_scorer=True` (default) tries to load from gitignored path → silent fallback to heuristic scorer
- **Competition-safe**: All v7/v8 profiles override with `models/page_scorer/v6_version_full/page_scorer.joblib` ✓
- **Fix needed**: OREV should update default to `models/page_scorer/v6_version_full/page_scorer.joblib`
- **Status**: Documented in QUESTIONS.jsonl, pending KEREN/OREV action

## Uncommitted Changes Warning (2026-03-20)

Working tree has unstaged changes in files not owned by TZUF:
- `src/rag_challenge/core/pipeline/free_text_cleanup.py` — small change (removed `char_limit` param from `condense_free_text`)
- `src/rag_challenge/submission/common.py` — free-text budget fill logic tweak
- `tests/unit/test_submission_generate.py` — likely corresponding test update
- `.sdd/agents/AGENT_INSTRUCTIONS_V2.md` — protocol update
- `.sdd/agents/orev/STATUS.json` — OREV status

KEREN should commit these if intentional (likely SHAI/NOGA work after last merge).

## Runtime Dependencies (verified 2026-03-20)

| Package | Version | Required For |
|---------|---------|-------------|
| `lightgbm` | 4.6 | Page scorer inference |
| `joblib` | 1.5 | Model load/serialization |
| `scikit-learn` | 1.8 | DictVectorizer deserialization |
| `pyarrow` | 23 | External grounding data |
| `qdrant-client` | — | Vector retrieval |

**Note**: `joblib` and `scikit-learn` were missing from `pyproject.toml` main deps and fixed in TZUF Phase 1 (commit 0c41d46, merged at feb3fc0).
