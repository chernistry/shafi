# Ops Runbook

This is the canonical local operator surface for the competition repo.

Use this document instead of memorizing one-off commands from research notes.

## Environment contract

- Host-local commands use `QDRANT_URL=http://localhost:6333`.
- Docker Compose containers always use `QDRANT_URL=http://qdrant:6333`.
- Config precedence is:
  - process environment
  - `.env.local`
  - `.env`
  - code defaults
- Keep shared defaults in `.env`.
- Keep workstation-specific overrides and secrets in `.env.local`.
- If you need an extra overlay for Compose, run:

```bash
ENV_FILE=/abs/path/to/override.env docker compose ...
```

## Canonical commands

From `/Users/sasha/IdeaProjects/personal_projects/rag_challenge`:

```bash
make install
make lint
make typecheck
make test
make docker-up
make docker-ingest
make ops-smoke
make platform-archive
```

## Local host workflow

1. Bootstrap env files:

```bash
cp .env.example .env
# Optional machine-specific overrides:
$EDITOR .env.local
```

2. Sync the environment:

```bash
make install
```

3. Run local code-quality gates:

```bash
make lint
make typecheck
make test
```

## Docker workflow

Start the local stack:

```bash
make docker-up
```

That brings up:

- `qdrant`
- `api`

Run the deterministic packaging smoke:

```bash
make ops-smoke
```

This proves:

- the image builds
- packaged prompts are present
- `api` boots inside the image
- non-root cache dirs are writable
- `ingest` entrypoints run from the packaged image

## Ingest workflow

Canonical ingest:

```bash
make docker-ingest
```

Manual live-provider caveat:

- `make ops-smoke` is mandatory and deterministic
- full provider-backed ingest remains a manual pre-submit check when secrets and remote services are available

## Eval workflow

Minimal eval entrypoint:

```bash
make docker-eval
```

To run the public harness:

```bash
make docker-eval EVAL_ARGS="python -m rag_challenge.eval.harness \
  --golden dataset/public_dataset.json \
  --endpoint http://api:8000/query \
  --concurrency 4 \
  --emit-cases \
  --judge \
  --judge-scope free_text \
  --judge-docs-dir dataset/dataset_documents \
  --judge-out data/judge_run.jsonl \
  --out data/eval_run.json"
```

## Platform submission workflow

Build the curated code archive without uploading:

```bash
make platform-archive
```

Submit an already-inspected artifact:

```bash
make platform-submit-existing \
  SUBMISSION_PATH=platform_runs/warmup/submission.json \
  CODE_ARCHIVE_PATH=platform_runs/warmup/code_archive.zip
```

Use `platform-submit-existing` only after inspecting:

- `submission.json`
- `preflight_summary.json`
- `code_archive.zip`
- `code_archive_audit.json`

## Troubleshooting

### Qdrant works locally but not inside Docker

- Host tools use `http://localhost:6333`
- Compose containers use `http://qdrant:6333`
- If a container resolves `localhost`, the env contract is broken

### Fastembed cache warnings

- The repo normalizes the old `/home/appuser/.cache/fastembed` container preset to the workspace cache for local runs
- The canonical local cache path is under:
  - `/Users/sasha/IdeaProjects/personal_projects/rag_challenge/.cache/fastembed`

### Ingest works locally but fails in the image

Run:

```bash
make ops-smoke
```

If that fails, fix packaging or mount issues before touching scoring code.

### Platform archive uncertainty

Use:

```bash
make platform-archive
```

before any submit-prep work. This is the safe preflight surface for archive correctness.
