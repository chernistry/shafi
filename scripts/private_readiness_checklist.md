# Private Dataset Readiness Checklist

**Date**: 2026-03-20
**Based on**: warmup 100 eval analysis, pipeline config, competition structure

---

## Pre-Flight Overview

The private phase provides a fresh document corpus (different from warmup). The system must
ingest, index, and answer questions against documents it has never seen. All collection
names, ingestion manifests, and trained models must be either re-pointed or re-run.

---

## Section 1: Configuration Files to Update

### 1.1 `.env` / `.env.local` — Required Changes

| Variable | Current (warmup) | Required (private) | Why |
|----------|------------------|--------------------|-----|
| `EVAL_PHASE` | `warmup` | `private` | Tells platform which phase to submit to |
| `QDRANT_COLLECTION` | `legal_chunks` | `legal_chunks_private` (or new name) | Avoid collision with warmup index |
| `EVAL_COLLECTION_PREFIX` | `legal_chunks_platform` | New prefix | Isolate private corpus index |

**Action**: Before private run, set these in `.env.local`:
```
EVAL_PHASE=private
QDRANT_COLLECTION=legal_chunks_private
EVAL_COLLECTION_PREFIX=legal_chunks_private_platform
```

### 1.2 `dataset_document_index.json` — Rebuild

The `dataset_document_index.json` currently covers **39 warmup documents**.
For private phase it must be rebuilt from scratch against private corpus documents.

**Action**: After downloading private documents, run the document index builder
(or the existing `build_*` scripts) to regenerate this file.

### 1.3 Golden Labels — Not Applicable

`eval_golden_warmup_verified.json` is warmup-specific. There are no golden labels
for private questions during the competition. The pipeline runs blind.

---

## Section 2: Ingestion Sequence

### 2.1 Download Private Documents

Documents will be provided by the platform in the same directory structure as warmup:
```
platform_runs/<run_id>/documents/
```

**Action**: Once private phase opens, wait for document download to complete before
proceeding. Verify document count matches expected corpus size.

### 2.2 Clear / Create Private Collection in Qdrant

```bash
# Verify Qdrant is running
curl http://localhost:6333/healthz

# Check existing collections
curl http://localhost:6333/collections

# The ingestion pipeline will create the new collection automatically
# but confirm no stale private collection exists
```

### 2.3 Run Ingestion

```bash
# Standard ingestion command (verify exact invocation from Makefile or scripts)
python -m rag_challenge.ingest --documents-dir platform_runs/<run_id>/documents/ \
    --collection $QDRANT_COLLECTION
```

**Expected time**: ~10–20 min for corpus of similar size to warmup (39 docs).
**Verify**: Check chunk count matches expected range (warmup had ~2,400 chunks for 39 docs).

### 2.4 Verify Ingestion Manifest

The `private_doctor_preflight.py` script checks for a `run_manifest` with valid fingerprint.

```bash
python scripts/private_doctor_preflight.py --manifest path/to/manifest.json
```

A missing or empty fingerprint will fail preflight. If it fails, re-ingest.

---

## Section 3: Trained Model Compatibility

### 3.1 Page Scorer (`trained_page_scorer`)

The current page scorer was trained on DIFC legal documents (law + court case structure).
Private corpus likely has the same document types (laws and court cases).

**Risk**: LOW — DIFC document structure is consistent across phases.
**Action**: No retraining needed unless private corpus has new document types.
**Verify**: Check `PIPELINE_TRAINED_PAGE_SCORER_ENABLED` and model path in `.env`.

### 3.2 Grounding Sidecar

If enabled, verify that the sidecar model path is still accessible.

---

## Section 4: Execution Sequence (Day-of)

```
Hour 0:  Private phase opens
  ├── Download private documents
  ├── Update .env.local: EVAL_PHASE=private, QDRANT_COLLECTION=legal_chunks_private
  ├── Clear any stale private Qdrant collection
  ├── Run ingestion
  └── Run preflight check: python scripts/private_doctor_preflight.py

Hour 1:  Ingest complete
  ├── Run sanity eval on 5 random questions (manual)
  ├── Start full batch eval: python scripts/batch_eval_warmup.py (adapted for private)
  └── Monitor for LLM/embed errors in logs

Hour 2+: Eval complete
  ├── Review answer quality spot-check (10 questions)
  ├── Generate submission file
  └── Submit via platform API
```

---

## Section 5: Known Failure Modes

### FM-1: Stale Collection in Qdrant
**Symptom**: Pipeline returns stale warmup answers for private questions.
**Cause**: `QDRANT_COLLECTION` not updated; pipeline hits warmup index.
**Fix**: Verify `.env.local` overrides; restart pipeline service after change.

### FM-2: Context Budget Too Low for Multi-Doc Questions
**Symptom**: G=0 on cross-doc boolean questions ("Did these two cases share a judge?").
**Cause**: `PIPELINE_BOOLEAN_CONTEXT_TOP_N=2` only allows 2 pages; dual-case needs 2 from different docs.
**Fix**: Set `PIPELINE_BOOLEAN_MULTI_REF_TOP_N=4` for queries with 2+ case references.
**Current setting**: `PIPELINE_BOOLEAN_MULTI_REF_TOP_N=3` (partially helps).

### FM-3: LLM Null Response (malformed output)
**Symptom**: Pipeline answer is `null`; G=0 despite correct retrieval.
**Cause**: Generation mode edge case in LP/GP-type boolean questions.
**Seen**: `860c44c7` in warmup (LP Law Art 11(1) question).
**Fix**: Ensure generation fallback is triggered on empty/null responses. Check
`PIPELINE_MAX_RETRIES=1` is active.

### FM-4: Unanswerable Traps Pipeline Gets Wrong
**Symptom**: G=0 on questions with empty gold_chunk_ids; pipeline gives wrong answer.
**Cause**: Relevant documents may not exist in private corpus; pipeline should say "no info."
**Seen**: 4 cases in warmup where pipeline answered when it shouldn't.
**Fix**: No config change needed; classifier should detect no-doc scenario.
**Watch**: If private corpus has more unusual laws, trap question rate may be higher.

### FM-5: API Rate Limits / Timeout
**Symptom**: Long eval run stalls; many retries.
**Cause**: LLM or embed API hitting rate limits.
**Fix**: Reduce `EVAL_QUERY_CONCURRENCY` from 2 to 1; check API key quota.

### FM-6: Ingestion Manifest Missing Fingerprint
**Symptom**: `private_doctor_preflight.py` reports `missing_fingerprint`.
**Fix**: Re-run ingestion with correct manifest schema version (`INGEST_MANIFEST_SCHEMA_VERSION=2`).

---

## Section 6: Panic Guide (Something Is Broken)

### P-1: Qdrant returns empty results for all questions
```bash
# Check Qdrant health
curl http://localhost:6333/healthz
# Check collection exists and has vectors
curl http://localhost:6333/collections/legal_chunks_private
# If missing: re-ingest
```

### P-2: All pipeline answers are "I don't have enough information"
- Either Qdrant collection is empty, or collection name mismatch
- Check `QDRANT_COLLECTION` in active env
- Check that `retrieved_page_ids` in telemetry is non-empty

### P-3: LLM API key invalid / expired
- Check `LLM_API_KEY` in `.env`
- OpenAI key format: `sk-proj-...`
- Fallback: set `LLM_COMPLEX_MODEL=gpt-4.1-mini` to reduce cost while fixing key

### P-4: Eval submission rejected by platform
- Check `EVAL_PHASE=private` is set
- Check `EVAL_API_KEY` is valid for private phase
- Verify submission format matches `EVAL_SUBMISSION_FILENAME=submission.json`

### P-5: Page scorer model file not found
```bash
# Check trained_page_scorer_model_path in last telemetry output
grep "trained_page_scorer_model_path" platform_runs/<run>/telemetry.json | head -5
# If path missing: set PIPELINE_TRAINED_PAGE_SCORER_ENABLED=false as fallback
```

---

## Section 7: Quick Wins to Apply Before Private Run

Based on warmup analysis, these settings changes require no code changes:

| Change | Expected Impact | Risk | How to Apply |
|--------|----------------|------|--------------|
| `QDRANT_PREFETCH_DENSE=120` | +2–4% G on multi-doc questions | Low | `.env.local` |
| `QDRANT_PREFETCH_SPARSE=120` | +1–2% G on article-reference questions | Low | `.env.local` |
| `PIPELINE_BOOLEAN_MULTI_REF_TOP_N=4` | Fix ~7 dual-case boolean misses | Medium | `.env.local` |
| Review `PIPELINE_BOOLEAN_CONTEXT_TOP_N` | May need 3 for cross-case | Medium | Test first |

**DO NOT change** ingestion parameters or model versions during the run window —
a re-ingest mid-eval would invalidate all in-progress answers.

---

## Section 8: Readiness Checklist (pre-submission)

- [ ] `EVAL_PHASE=private` confirmed in active env
- [ ] `QDRANT_COLLECTION` set to private-specific name
- [ ] Qdrant collection created and chunk count verified
- [ ] Preflight script passes: `python scripts/private_doctor_preflight.py`
- [ ] 5-question manual spot-check completed
- [ ] No null/empty answers in spot-check telemetry
- [ ] LLM API key confirmed valid
- [ ] Embed API key confirmed valid
- [ ] Submission file generated: `submission.json`
- [ ] Submission sent to platform and acknowledged
