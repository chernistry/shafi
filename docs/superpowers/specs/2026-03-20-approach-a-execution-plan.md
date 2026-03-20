# Approach A: Private Set Execution Plan

**Prerequisite**: Bug fix branch merged to main. All 7 fixes + profiles committed.

## Pre-Arrival Setup (Do Now)

### Step 1: Merge bug fixes to main

```bash
cd /Users/sasha/IdeaProjects/personal_projects/rag_challenge
git merge claude/quizzical-cerf --no-ff -m "Merge Approach A bug fixes and profiles"
```

### Step 2: Verify infrastructure

```bash
docker compose up -d qdrant
# Verify Qdrant is healthy
curl -s http://localhost:6333/healthz
```

### Step 3: Prepare directory structure

```bash
mkdir -p platform_runs/final/documents
mkdir -p platform_runs/final/manifests/v6
mkdir -p platform_runs/final/manifests/private_1792
```

---

## When Private Dataset Arrives

### Step 4: Stage private dataset

```bash
# Copy PDFs to docs directory
cp /path/to/private/documents/*.pdf platform_runs/final/documents/
# Copy questions file
cp /path/to/private/questions.json platform_runs/final/questions.json
```

### Step 5: Ingest at 1024-dim (v6 baseline)

```bash
# Uses existing legal_chunks collection (or create fresh)
ENV_FILE=profiles/private_v6_regime.env \
docker compose --profile tools run --rm ingest \
  --doc-dir platform_runs/final/documents
```

**Estimated time**: ~10 min for ~200 documents.

### Step 6: Run pipeline at 1024-dim (answer source)

```bash
ENV_FILE=profiles/private_v6_regime.env \
EVAL_PHASE=final \
docker compose --profile tools run --rm eval \
  python -m rag_challenge.eval.harness \
  --golden platform_runs/final/questions.json
```

**Artifacts produced** in `platform_runs/final/`:
- `submission.json` → rename to `submission_v6.json`
- `raw_results.json` → rename to `raw_results_v6.json`
- `preflight_summary.json` → rename to `preflight_summary_v6.json`

```bash
cd platform_runs/final
for f in submission raw_results preflight_summary; do
  cp "${f}.json" "${f}_v6.json"
done
```

### Step 7: Ingest at 1792-dim

```bash
ENV_FILE=profiles/private_1792_regime.env \
docker compose --profile tools run --rm ingest \
  --doc-dir platform_runs/final/documents
```

Separate collections (`legal_chunks_private_1792` etc.) — no collision with 1024-dim.

### Step 8: Run pipeline at 1792-dim (page source)

```bash
ENV_FILE=profiles/private_1792_regime.env \
EVAL_PHASE=final \
docker compose --profile tools run --rm eval \
  python -m rag_challenge.eval.harness \
  --golden platform_runs/final/questions.json
```

**Artifacts produced** → rename:
```bash
cd platform_runs/final
for f in submission raw_results preflight_summary; do
  cp "${f}.json" "${f}_1792.json"
done
```

### Step 9: Answer-stable replay

```bash
python scripts/run_answer_stable_grounding_replay.py \
  --answer-submission platform_runs/final/submission_v6.json \
  --answer-raw-results platform_runs/final/raw_results_v6.json \
  --page-submission platform_runs/final/submission_1792.json \
  --page-raw-results platform_runs/final/raw_results_1792.json \
  --out-dir platform_runs/final/replay/ \
  --page-source-pages-default all
```

### Step 10: Validate replay

Check `platform_runs/final/replay/replay_summary.json`:
- `answer_changed_qids` should be empty (zero drift)
- `page_changed_qids` shows questions where pages improved
- Grounding score delta should be positive

### Step 11: Build code archive

```bash
EVAL_PHASE=final \
docker compose --profile tools run --rm eval \
  python -m rag_challenge.submission.platform --archive-only
```

### Step 12: Submit

**Option A** (replay submission — preferred):
```bash
EVAL_PHASE=final \
docker compose --profile tools run --rm eval \
  python -m rag_challenge.submission.platform \
  --submit-existing \
  --submission-path platform_runs/final/replay/submission_answer_stable_replay.json \
  --code-archive-path platform_runs/final/code_archive.zip \
  --poll
```

**Option B** (v6-only fallback if replay shows no improvement):
```bash
EVAL_PHASE=final \
docker compose --profile tools run --rm eval \
  python -m rag_challenge.submission.platform \
  --submit-existing \
  --submission-path platform_runs/final/submission_v6.json \
  --code-archive-path platform_runs/final/code_archive.zip \
  --poll
```

---

## Decision Points

| Checkpoint | If Good | If Bad |
|-----------|---------|--------|
| v6 pipeline run | Proceed to 1792 ingest | Debug; check Qdrant, API keys |
| 1792 ingest | Proceed to 1792 pipeline | Fall back to v6-only submission |
| Replay drift | 0 answer changes → proceed | Non-zero → inspect, fix allowlist |
| Replay grounding | Positive delta → submit replay | Negative → submit v6-only |

## Time Budget

| Step | Est. Duration |
|------|--------------|
| Ingest 1024-dim | 10 min |
| Pipeline 1024-dim | 15 min |
| Ingest 1792-dim | 12 min |
| Pipeline 1792-dim | 15 min |
| Replay | 2 min |
| Validation + Submit | 5 min |
| **Total** | **~60 min** |
