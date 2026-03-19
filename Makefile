.PHONY: install lint typecheck test format ops-smoke platform-archive all

UV ?= uv
UV_DEV := $(UV) run --extra dev
COMPOSE ?= docker compose

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

ops-smoke:
	$(UV) run python scripts/container_contract_smoke.py

platform-archive:
	$(COMPOSE) --profile tools run --rm eval python -m rag_challenge.submission.platform --archive-only

all: lint typecheck test
