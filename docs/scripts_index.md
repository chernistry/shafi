# Scripts Index

This index separates the canonical operator scripts from the broader research toolkit in `scripts/`.

## Canonical operator surface

Use these first.

- `scripts/build_private_submission.py`
  - builds the final private submission JSON from an eval checkpoint (primary build tool, private phase)
- `scripts/build_v9_recovery.py` / `scripts/build_v10_recovery.py` / `scripts/build_v14_v91_hybrid.py`
  - version-specific recovery/hybrid builders from checkpoints
- `scripts/build_v16_hybrid.py` / `scripts/build_v16_super_hybrid.py`
  - V16 hybrid builders (V16 answers + V15 pages)
- `scripts/build_v17_hybrid.py` / `scripts/build_v17_super_hybrid.py`
  - V17 hybrid builders (V17 answers + V15 pages, with DOI/boolean/number corrections)
- `scripts/tzuf_v9_full900.py` / `scripts/tzuf_v11_full900.py` / `scripts/tzuf_v12_full900.py` / `scripts/tzuf_v15_full900.py`
  - full 900-question eval runners (use with appropriate profile)
- `scripts/start_dashboard.sh`
  - start the live dashboard (frontend + parsers)
- `scripts/patch_best_formatting.py`
  - fixes formatting issues (double periods, truncation artifacts) in BEST.json
- `scripts/enrich_submission_pages.py`
  - enriches submission page references
- `scripts/container_contract_smoke.py`
  - deterministic packaging/container smoke
- `scripts/private_doctor_preflight.py`
  - bounded private-day doctor / archive safety checks
- `scripts/rehearse_private_run.py`
  - deterministic private-run rehearsal flow
- `scripts/build_run_manifest.py`
  - deterministic run-manifest fingerprinting
- `scripts/pre_submit_sanity_check.py`
  - pre-submission sanity validation (format, pages, coverage, regressions)
- `scripts/build_submit_v2.sh`
  - ~~one-command Submit #2 builder~~ — **DEPRECATED: V18 failed, Submit #2 cancelled**

## Submission gating tools

- `scripts/v13_vs_v91_gate.py`
  - gate comparison: V13 vs V9.1 on key metrics
- `scripts/v16_vs_v15_gate.py`
  - gate comparison: V16 vs V15 on key metrics
- `scripts/question_cluster_analysis.py`
  - cluster private questions by type/family for targeted analysis

## Grounding / evaluation tools

Use when working on reviewed grounding changes.

- `scripts/capture_query_artifacts.py`
- `scripts/score_against_golden.py`
- `scripts/run_grounding_sidecar_ablation.py`
- `scripts/export_grounding_ml_dataset.py`
- `scripts/train_grounding_router.py`
- `scripts/train_page_scorer.py`
- `scripts/eval_grounding_models.py`
- `scripts/mine_within_doc_rerank_opportunities.py`

## Research and advisory scripts

The rest of `scripts/` is primarily research-facing:

- audits
- scanners
- candidate builders
- ranking/search helpers
- one-off diagnostics

Treat those as advisory unless a ticket explicitly names them.

## Not a cleanup target right now

- No broad script-tree reorganization is planned in the current competition window.
- If a script is not listed above, do not assume it is production-critical.
- Add new scripts to this index only when they become part of the canonical operator or evaluation path.
