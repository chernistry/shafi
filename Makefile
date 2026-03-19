.PHONY: install lint typecheck test format docker-up docker-ingest docker-eval ops-smoke platform-archive platform-submit-existing all

UV ?= uv
UV_DEV := $(UV) run --extra dev
COMPOSE ?= docker compose
EVAL_ARGS ?= python -m rag_challenge.eval.harness --help

install:
	$(UV) sync --extra dev

lint:
	$(UV_DEV) ruff check src tests scripts

format:
	$(UV_DEV) ruff format src tests scripts
	$(UV_DEV) ruff check --fix src tests scripts

typecheck:
	$(UV_DEV) pyright src/rag_challenge tests/unit

test:
	$(UV_DEV) pytest tests/unit -q

docker-up:
	$(COMPOSE) up --build -d qdrant api

docker-ingest:
	$(COMPOSE) --profile tools run --rm ingest

docker-eval:
	$(COMPOSE) --profile tools run --rm eval $(EVAL_ARGS)

ops-smoke:
	$(UV) run python scripts/container_contract_smoke.py

platform-archive:
	$(COMPOSE) --profile tools run --rm eval python -m rag_challenge.submission.platform --archive-only

platform-submit-existing:
	@test -n "$(SUBMISSION_PATH)" || (echo "SUBMISSION_PATH is required"; exit 1)
	@test -n "$(CODE_ARCHIVE_PATH)" || (echo "CODE_ARCHIVE_PATH is required"; exit 1)
	$(COMPOSE) --profile tools run --rm eval \
		python -m rag_challenge.submission.platform \
		--submit-existing \
		--submission-path $(SUBMISSION_PATH) \
		--code-archive-path $(CODE_ARCHIVE_PATH) \
		--poll

all: lint typecheck test
