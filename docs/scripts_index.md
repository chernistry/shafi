# Scripts Index

This index separates the canonical operator scripts from the broader research toolkit in `scripts/`.

## Canonical operator surface

Use these first.

- `scripts/container_contract_smoke.py`
  - deterministic packaging/container smoke
- `python -m rag_challenge.submission.platform`
  - platform archive and submission flow
- `scripts/private_doctor_preflight.py`
  - bounded private-day doctor / archive safety checks
- `scripts/rehearse_private_run.py`
  - deterministic private-run rehearsal flow
- `scripts/build_run_manifest.py`
  - deterministic run-manifest fingerprinting

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
