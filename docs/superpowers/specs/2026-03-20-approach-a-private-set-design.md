# Approach A: 1792-dim Embeddings + Bug Fixes + Answer-Stable Replay

**Date**: 2026-03-20
**Status**: Approved
**Goal**: Maximize total score on the private dataset. Current best: 0.742 (rank 46). Target: top 5 (>0.9).

## Strategy Summary

Three-layer approach to close the gap with rank 1 (0.982):

1. **Bug fixes** (committed): 7 platform-transfer bugs that inflated local scores or degraded answers
2. **1792-dim embeddings**: Proven +0.32 grounding lift from kanon-2 at 1792 dimensions
3. **Answer-stable replay**: Merge champion answers (1024-dim) with better page grounding (1792-dim), zero answer drift

## Score Model

```
Total = (0.7 * Det + 0.3 * Asst) * G * T * F
```

- **Det** (deterministic answers): boolean, number, date, name, names types
- **Asst** (assistant-judged): free-text quality
- **G** (grounding): F-beta 2.5 of used_pages vs gold pages
- **T** (TTFT): time-to-first-token multiplier
- **F** (format): submission format compliance

**Key insight**: G is a multiplier. Moving G from 0.6 to 0.8 lifts total by ~33%. The 1792-dim embedding improvement targets G specifically.

## Architecture

### Phase 1: Dual-Pipeline Execution

Run the pipeline twice against the private dataset:

**Run A (answer source)** — `profiles/private_v6_regime.env`
- 1024-dim embeddings (existing `legal_chunks` collection)
- v6-era config (proven answer quality)
- Produces: `submission_v6.json`, `raw_results_v6.json`, `preflight_summary_v6.json`

**Run B (page source)** — `profiles/private_1792_regime.env`
- 1792-dim embeddings (fresh `legal_chunks_private_1792` collection)
- Same v6-era config except embedding dimensions
- Produces: `submission_1792.json`, `raw_results_1792.json`, `preflight_summary_1792.json`

### Phase 2: Ingest at 1792-dim

Before running pipeline, ingest the private dataset documents at 1792-dim:

```bash
ENV_FILE=profiles/private_1792_regime.env docker compose --profile tools run --rm ingest
```

Collection names in the 1792 profile are isolated:
- `legal_chunks_private_1792`
- `legal_chunks_shadow_private_1792`
- `legal_pages_private_1792`
- `legal_support_facts_private_1792`

No risk of polluting existing 1024-dim collections.

### Phase 3: Answer-Stable Replay

Use existing replay tooling to merge:

```bash
python scripts/run_answer_stable_grounding_replay.py \
  --answer-source platform_runs/private/v6/ \
  --page-source platform_runs/private/1792/ \
  --output platform_runs/private/replay/ \
  --page-source-pages-default all
```

This produces a merged submission with:
- **Answers**: Frozen from Run A (champion quality)
- **Used pages**: Swapped from Run B (better grounding)
- **Zero answer drift**: Replay validates question set alignment

### Phase 4: Validation and Submission

1. Run `replay_summary.json` drift check (should show 0 answer changes)
2. Compare grounding scores between v6-only and replay
3. Submit the merged artifact

## Projected Impact

| Component | Current (est.) | After Fixes | After 1792 Replay |
|-----------|---------------|-------------|-------------------|
| Det+Asst  | ~0.85         | ~0.87       | ~0.87             |
| G (grounding) | ~0.60    | ~0.63       | ~0.80+            |
| T (TTFT)  | ~0.98         | ~0.98       | ~0.98             |
| F (format)| ~1.0          | ~1.0        | ~1.0              |
| **Total** | **~0.74**     | **~0.77**   | **~0.88+**        |

Conservative estimate. The +0.32 grounding lift was observed on warmup; private set may vary.

## Risk Mitigation

- **Answer regression**: Replay freezes answers. No answer drift by construction.
- **Page regression**: If 1792-dim pages are worse, submit the v6-only run.
- **Time pressure**: Dual pipeline adds ~30 min. Replay is <5 min. Total budget: ~1 hour.
- **Ingest failure**: 1792-dim ingest proven in 4+ prior warmup runs.

## Files Changed (Bug Fixes — Already Committed)

- `src/rag_challenge/llm/generator_cleanup.py` — Truncation false positive fix
- `src/rag_challenge/core/pipeline/support_formatting.py` — Name word limit 12→20
- `src/rag_challenge/core/pipeline/generation_logic.py` — Skip coercion on confident extraction
- `src/rag_challenge/core/retriever_filters.py` — Doc-ref filter should→must
- `src/rag_challenge/eval/sources.py` — Align used_pages with submission logic
- `src/rag_challenge/eval/harness.py` — Add submission_answer field
- `profiles/private_v6_regime.env` — v6 config profile
- `profiles/private_1792_regime.env` — 1792-dim config profile

## Dependencies

- Qdrant running locally
- Isaacus API key active (`ISAACUS_API_KEY`)
- Private dataset PDFs in docs directory
- Private dataset questions JSON
