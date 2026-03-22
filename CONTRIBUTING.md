# Contributing to RAG Challenge

Thanks for your interest! Here's how to get started.

## Quick Start

```bash
git clone https://github.com/chernistry/shafi && cd shafi
uv sync --extra dev
uv run pytest tests/unit -q
```

## Ways to Contribute

- **Bug reports** -- open an issue with steps to reproduce
- **Feature ideas** -- open a discussion or issue
- **Code** -- fork, branch, PR (see below)
- **Docs** -- typo fixes, examples, architectural notes
- **Pipeline improvements** -- retrieval strategies, reranking approaches, prompt tuning

## Development Environment

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker and Docker Compose (for the full stack)
- API keys: `OPENAI_API_KEY`, `COHERE_API_KEY` (in `.env.local`)

### Setup

```bash
# 1. Install dependencies
uv sync --extra dev

# 2. Copy environment template
cp .env.example .env
# Put machine-specific secrets in .env.local (gitignored)

# 3. Start local infrastructure
docker compose up --build -d

# 4. Ingest the corpus
docker compose --profile tools run --rm ingest

# 5. Verify the API is running
curl http://localhost:8000/health
```

### Environment Contract

- Host-local commands (`uv run ...`) use `QDRANT_URL=http://localhost:6333`
- Docker Compose services use `QDRANT_URL=http://qdrant:6333` internally
- Config precedence: process env > `.env.local` > `.env` > code defaults

## Development Workflow

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes
3. Run checks:
   ```bash
   make all          # or individually:
   make lint         # ruff check src tests scripts
   make typecheck    # pyright src/shafi tests/unit
   make test         # pytest tests/unit -q
   ```
4. Commit with a clear message
5. Open a PR against `master`

## Running Tests

```bash
# Unit tests (fast, no infrastructure needed)
uv run pytest tests/unit -q

# Integration tests (requires running Qdrant + API)
docker compose up -d
uv run pytest tests/integration -q

# Full check suite
make all
```

## Code Style

- Python 3.12+, type hints everywhere
- `ruff` for linting and formatting, `pyright` strict mode for types
- Max line length: 120
- Tests go in `tests/unit/` or `tests/integration/`
- Source code lives in `src/shafi/`

### Formatting

```bash
make format    # auto-fix with ruff
```

## Architecture Principles

- **Page-first retrieval** -- hybrid search at page level, then anchor-based filtering
- **DB short-circuit** -- metadata-answerable questions resolved without LLM calls
- **Model routing** -- cheap models for strict answer types, heavy models for reasoning
- **Faithfulness guardrails** -- premise guard, conflict detection, citation verification
- **Telemetry everywhere** -- per-stage latency, page IDs, and model metadata in every response

## PR Guidelines

- Keep PRs focused on a single concern
- Include test coverage for new pipeline stages or answer-type handlers
- Run `make all` before submitting
- Reference the relevant issue number if applicable
- Describe what changed and why in the PR description

## License

By contributing, you agree that your contributions will be licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
