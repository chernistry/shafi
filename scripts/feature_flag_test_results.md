# Feature Flag Interaction Test Results
Generated: 2026-03-20 by TZUF Phase 2 (tzuf-2b)

## Test Methodology

All flags tested via import-level verification and full test suite runs.
Import test: verifies `Settings`, `GroundingEvidenceSelector`, `HybridRetriever`,
`GenerationLogicMixin`, and `RAGPipelineBuilder` all load cleanly with each flag combination.

Full test suite: `uv run pytest tests/ -q --ignore=tests/integration`

---

## Flag Registry (settings.py PipelineSettings)

| Flag | Default | Env Var | Status in codebase |
|------|---------|---------|-------------------|
| `enable_cross_ref_boosts` | `True` | `PIPELINE_ENABLE_CROSS_REF_BOOSTS` | Implemented |
| `enable_segment_retrieval` | `True` | `PIPELINE_ENABLE_SEGMENT_RETRIEVAL` | Implemented |
| `enable_doc_diversity_expansion` | `True` | `PIPELINE_ENABLE_DOC_DIVERSITY_EXPANSION` | Implemented |
| `enable_grounding_sidecar` | `True` | `PIPELINE_ENABLE_GROUNDING_SIDECAR` | Implemented |
| `enable_trained_page_scorer` | `True` | `PIPELINE_ENABLE_TRAINED_PAGE_SCORER` | Implemented |
| `enable_answer_validation` | **NOT IN SETTINGS** | — | NOGA's code not yet merged to main |
| `enable_answer_consensus` | **NOT IN SETTINGS** | — | NOGA's code not yet merged to main |

**Note**: The KNOWLEDGE_BASE.md lists `enable_answer_validation` and `enable_answer_consensus` as flags
but they are NOT yet in `settings.py` or the pipeline. NOGA's answer_quality_gate.py and
answer_validator.py are NOT in the codebase on this branch. No import errors caused by their absence.

---

## Isolation Tests: Each Flag OFF, Others ON

| Flag (set to false) | Import test | Notes |
|--------------------|-------------|-------|
| `enable_cross_ref_boosts=false` | ✅ OK | No conflict with others ON |
| `enable_segment_retrieval=false` | ✅ OK | No conflict with others ON |
| `enable_doc_diversity_expansion=false` | ✅ OK | No conflict with others ON |
| `enable_grounding_sidecar=false` | ✅ OK | No conflict with others ON |
| `enable_trained_page_scorer=false` | ✅ OK | No conflict with others ON |

## Combination Tests

| Combination | Import test | Full test suite |
|-------------|-------------|-----------------|
| All ON | ✅ OK | 1121/1122 pass |
| All OFF (v6 regime) | ✅ OK | 1121/1122 pass |
| v6 profile (explicit disables) | ✅ OK | 1121/1122 pass |
| v7 profile (all ON + v4 model path) | ✅ OK | 1121/1122 pass |

**Result: Zero flag-induced test failures. All combinations are compatible.**

The 1 persistent failure (`test_trainer_cross_validation_beats_heuristic_on_public_slice`)
is an environment-only issue: requires `.sdd/` research data that is gitignored and absent
in worktrees. Not flag-related; not a code regression.

---

## Critical Finding: Default Model Path Broken in Competition Environment

**Flag**: `enable_trained_page_scorer` (default: `True`)
**Setting**: `trained_page_scorer_model_path`
**Default value**: `.sdd/researches/627_runtime_safe_page_scorer.../page_scorer.joblib`

This path points to a `.sdd/` directory which is **gitignored** and will NOT exist in the
competition Docker container or any fresh checkout.

**Impact**: With default settings (no profile), `enable_trained_page_scorer=True` but the model
file doesn't exist → the runtime falls back gracefully (fail-closed), but the trained scorer
is silently skipped. This means the default settings don't actually use the LightGBM scorer.

**Fix needed**: Change the default in `settings.py` to point to
`models/page_scorer/v4_tamar_augmented/page_scorer.joblib` (which IS committed).

**Owner**: OREV (settings.py owner). Posted to BULLETIN.

**Workaround**: All 4 private profiles (`private_v7_enhanced.env`,
`private_v7_1792_enhanced.env`, `private_v8_full.env`) correctly override this path to
`models/page_scorer/v4_tamar_augmented/page_scorer.joblib`. The v6 profiles have the
scorer disabled (`enable_trained_page_scorer=false`), so the broken default doesn't affect them.

---

## Profile Verification Results

| Profile | Loads | cross_ref | seg | div | sidecar | scorer | model_path_valid |
|---------|-------|-----------|-----|-----|---------|--------|-----------------|
| `private_v6_regime.env` | ✅ | Off | Off | Off | Off | Off | N/A (scorer off) |
| `private_1792_regime.env` | ✅ | Off | Off | Off | Off | Off | N/A (scorer off) |
| `private_v7_enhanced.env` | ✅ | On | On | On | On | On | ✅ v4 path |
| `private_v7_1792_enhanced.env` | ✅ | On | On | On | On | On | ✅ v4 path |
| `private_v8_full.env` | ✅ | On | On | On | On | On | ✅ v4 path |
