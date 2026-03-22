# Integration Issues Log
Generated: 2026-03-20 by TZUF

## FIXED Issues

### ISSUE-001: `joblib` not in main dependencies (CRITICAL)
- **Severity**: Critical (breaks import of page scorer runtime in all environments without manual install)
- **File**: `pyproject.toml`
- **Root cause**: `src/rag_challenge/ml/page_scorer_runtime.py` imports `joblib` at module level, but `joblib` was not listed as a main dependency. It is only pulled in transitively in some environments.
- **Fix applied**: Added `"joblib>=1.4"` and `"scikit-learn>=1.5"` to `[project.dependencies]` in `pyproject.toml`. Removed duplicate `scikit-learn` from `[dev]` extras.
- **Verification**: `uv run pytest tests/unit/test_grounding_sidecar.py` passes after fix.

### ISSUE-002: v6 profiles silently enabling post-v6 features (CRITICAL)
- **Severity**: Critical (incorrect baselines for private evaluation comparison)
- **Files**: `profiles/private_v6_regime.env`, `profiles/private_1792_regime.env`
- **Root cause**: Commit `0f284d2` (2026-03-20) set `enable_segment_retrieval` and `enable_doc_diversity_expansion` to `True` as defaults in `settings.py`. The v6 profiles (which say "explicitly disable ALL post-v6 features") did not disable these new flags. This silently enabled two Phase-2 features in the "pure v6" baseline profiles.
- **Fix applied**: Added `PIPELINE_ENABLE_SEGMENT_RETRIEVAL=false` and `PIPELINE_ENABLE_DOC_DIVERSITY_EXPANSION=false` to both v6 profiles.
- **Verification**: Profile validation confirms `segment_retrieval=False, doc_diversity=False` for both v6 profiles.

### ISSUE-003: v7 profiles missing explicit enables for new Phase-2 features
- **Severity**: Medium (undocumented behavior — features enabled by default silently)
- **Files**: `profiles/private_v7_enhanced.env`, `profiles/private_v7_1792_enhanced.env`
- **Root cause**: After commit `0f284d2`, the v7 profiles silently gained `enable_segment_retrieval=True` and `enable_doc_diversity_expansion=True` (both default True). These were not documented in the profile headers as "Changes from v6".
- **Fix applied**: Added explicit `PIPELINE_ENABLE_SEGMENT_RETRIEVAL=true` and `PIPELINE_ENABLE_DOC_DIVERSITY_EXPANSION=true` to both v7 profiles. Updated header comments to document all 5 changes from v6 (not just 3).
- **Verification**: Profile validation confirms `segment_retrieval=True, doc_diversity=True` for both v7 profiles.

## Documented Issues (Not Fixed)

### ISSUE-004: `run_private_dataset_pipeline.py` referenced but doesn't exist
- **Severity**: Low (no functional impact; task description may be outdated)
- **Details**: The integration task description referenced this script, but it doesn't exist. The equivalent functionality is in `scripts/rehearse_private_run.py`.
- **Action needed**: None. The rehearsal script covers this use case.

### ISSUE-005: `scripts/pre_submit_sanity_check.py` referenced but doesn't exist
- **Severity**: Low (equivalent exists)
- **Details**: Equivalent is `scripts/check_submission_projection.py`.
- **Action needed**: None.

### ISSUE-006: NOGA modules not implemented
- **Severity**: Info only
- **Details**: `answer_validator.py`, `answer_consensus.py`, `enable_answer_consensus`, `enable_retrieval_escalation` were listed in the task description as potential NOGA additions but do not exist in the codebase. No broken imports; NOGA did not add these features.
- **Action needed**: None. These are simply not implemented.

### ISSUE-007: Test requiring `.sdd/` research data fails in worktree
- **Severity**: Low (environment-only; test passes in main repo)
- **Test**: `tests/unit/test_retrieval_utility.py::test_trainer_cross_validation_beats_heuristic_on_public_slice`
- **Root cause**: Test references `.sdd/researches/639_grounding_resume_after_devops_baseline_r1_2026-03-19/` which is gitignored and not present in git worktrees.
- **Action needed**: None for production. For CI, consider skipping this test when `.sdd/` data is unavailable (add `pytest.mark.skipif(not Path(...).exists(), reason="...")`).

### ISSUE-008: `private_doctor_preflight.py` requires `PYTHONPATH=.`
- **Severity**: Low (pre-existing; not new)
- **Details**: `src/rag_challenge/submission/platform.py` imports `from scripts.build_platform_truth_audit import ...`. This requires the repo root in PYTHONPATH. Works in Docker (CWD in sys.path via `-m` execution) but fails with `uv run python scripts/...` without `PYTHONPATH=.`.
- **Action needed**: Run all scripts as `PYTHONPATH=. uv run python scripts/<script>.py` or `python -m scripts.<script>` from repo root.
